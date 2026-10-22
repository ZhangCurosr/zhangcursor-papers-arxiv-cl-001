# Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?

Ziquan Wang<sup>1</sup> Yuxuan Nie<sup>2</sup> Jian Ma<sup>1</sup> Yingjie Li<sup>1†</sup>

<sup>1</sup>Hangzhou International Innovation Institute, Beihang University, Hangzhou, China <sup>2</sup>Peking University, Beijing, China {ziquanwang,yingjieli}@buaa.edu.cn

## Abstract

Retrieval-based in-context learning (ICL) enables large language models (LLMs) to predict tabular labels from similar examples, but it remains unclear whether LLM inference improves on direct voting over those same examples. We address this question by comparing four general-purpose LLMs with distanceweighted nearest-neighbor voting (Distance-KNN) using identical retrieved neighbors across 19 tabular classification datasets. With the number of demonstrations fixed, enlarging the retrieval candidate pool improves accuracy for all predictors, yet Distance-KNN achieves the highest mean accuracy at every pool size. More detailed analyses reveal that LLM gains are concentrated where retrieved neighbors exhibit low label consensus and vary across models. These gains further depend on the average feature-space distance to the test query: under low label consensus, some LLMs outperform Distance-KNN in close and intermediate neighborhoods, but all fall behind on average in the most distant group. Complementary experiments on the number of demonstrations, prompt formulation, retrieval representation, and label-balanced demonstration selection largely preserve the overall findings. These results characterize when LLM inference adds predictive value beyond distance-weighted voting in retrieval-based tabular ICL.

## 1 Introduction

Tabular classification is a widely studied machine learning task with applications in healthcare (Yang et al., 2024), commerce (Sun and Li, 2024), and energy (Alawi et al., 2024). The goal is to predict class labels from instances described by numerical and categorical features. Traditional methods include nearest-neighbor classifiers, which predict labels from similar training instances (Dudani, 1976), and tree-based ensembles, which have demonstrated strong performance on heterogeneous tabular data (Chen and Guestrin, 2016; Grinsztajn et al., 2022).

![](images/b7f07d447cc42cc7000d9b7496ad8f6081de52eb651ada2bf707fb2e22b62272.jpg)  
Figure 1: Illustration of shared-neighbor prediction in tabular classification. Distance-KNN and the LLM use the same eight retrieved neighbors. Label consensus measures the majority-label fraction, while feature proximity is measured by the mean query-to-neighbor distance. In this example, Distance-KNN correctly predicts class A, whereas the LLM predicts class B.

Recent research has explored alternative approaches that leverage pretrained models for tabular prediction. Specialized tabular foundation models, including TabPFN, learn transferrable predictive patterns through pretraining (Hollmann et al., 2023, 2025; Ma et al., 2025). Another line of work adapts language models to tabular data through supervised or instruction tuning (Hegselmann et al., 2023; Li et al., 2024; Wen et al., 2024; Zhang et al., 2024). More recently, researchers have explored applying frozen general-purpose large language models (LLMs) through few-shot in-context learning (ICL) (Brown et al., 2020; Wu and Hou, 2025; Yuan et al., 2025). In this setting, labeled tabular instances are serialized as demonstrations, often selected according to their similarity to a test query, enabling prediction without task-specific parameter updates.

Despite these advances, it remains unclear how much predictive value LLM inference adds beyond direct voting on the same retrieved examples. In retrieval-based tabular ICL, performance differences may arise from either the selection of demonstrations or the predictions made from them, making these contributions difficult to distinguish when different retrieved examples are used. Moreover, expanding the candidate pool changes the available retrieval evidence, but whether LLM inference benefits more from this change than direct voting remains unclear. Their relative performance may also depend on two characteristics of the retrieved neighborhood: label consensus, the fraction of neighbors sharing the most frequent label, and feature proximity, measured by the mean featurespace distance from the test query to those neighbors, as illustrated in Figure 1. A controlled comparison on identical retrieved neighbors is therefore needed to assess when LLMs can provide additional predictive value.

In this work, we systematically compare four general-purpose LLMs with distance-weighted nearest-neighbor voting (Distance-KNN) across 19 tabular classification datasets using a sharedneighbor evaluation protocol. At each candidatepool size, both predictors use the same eight retrieved examples: LLMs predict from their serialized features and labels, while Distance-KNN predicts from their labels and retrieval distances. By expanding the candidate pool while keeping the number of retrieved examples fixed, we examine their performance and how LLM gains vary with label consensus and feature proximity. Our results show that Distance-KNN achieves the highest mean accuracy across all evaluated candidate pools, while LLMs exhibit selective gains under low label consensus. Feature proximity further distinguishes these gains, with improvements occurring in selected close and intermediate neighborhoods but disappearing in the most distant ones.

In summary, our contributions are as follows: 1) We present a systematic shared-neighbor comparison of four frozen LLMs and Distance-KNN across 19 datasets, enabling a direct assessment of the contribution of LLM inference over distanceweighted voting. 2) We find that Distance-KNN achieves the highest mean accuracy across all evaluated candidate pools, while LLMs exhibit selective gains in low-consensus neighborhoods that vary with feature proximity. 3) We further examine how the number of demonstrations, prompt formulation, retrieval representation, and label-balanced demonstration affect the comparison, finding that the advantage of Distance-KNN largely persists across these experiments.

## 2 Related Work

Tabular Prediction and Tabular Foundation Models Tabular prediction has traditionally relied on distance-based methods such as k-nearest neighbors (KNN) and tree-based ensembles, including Random Forest, XGBoost, and CatBoost (Cover and Hart, 1967; Breiman, 2001; Chen and Guestrin, 2016; Prokhorenkova et al., 2018). To circumvent task-specific training while achieving strong generalization across unseen datasets, recent work has introduced tabular foundation models. For instance, TabPFN is pretrained on synthetic tabular tasks to make predictions directly from labeled examples (Hollmann et al., 2023, 2025). Extending this paradigm to real-world data, Tab-DPT introduces a dedicated tabular architecture for in-context prediction, rather than adapting generalpurpose LLMs (Ma et al., 2025).

Adapting Language Models for Tabular Prediction A common paradigm relies on additional training to adapt language models for tabular prediction. Existing methods fine-tune LLMs on serialized rows for few-shot classification (Hegselmann et al., 2023), train on synthetic table tasks (Li et al., 2024), or leverage diverse tabular datasets for cross-dataset transfer (Wen et al., 2024; Gardner et al., 2024), while others explore retrievalguided tuning (Wen et al., 2025) and structureaware parameter adaptation (He et al., 2025). In contrast, our study investigates tabular prediction using frozen, general-purpose LLMs without additional task-specific training, leveraging Distance-KNN to provide a controlled comparison on identical retrieved demonstrations.

Retrieval-Based In-Context Learning with LLMs In-context learning (ICL) enables LLMs to adapt to new tasks using a few labeled demonstrations without fine-tuning (Brown et al., 2020). Subsequent studies have examined how models exploit these demonstrations, analyzing the effects of input distributions (Min et al., 2022), label mappings (Zhao et al., 2021; Sun et al., 2025), and prompt formatting (Peng et al., 2025). Demonstration selection has also been widely explored through semantic similarity (Liu et al., 2022), learned retrieval (Rubin et al., 2022; Liu et al., 2025), and influence estimation (Zhang et al., 2025b; Du et al., 2025). In tabular classification, Wu and Hou (2025) propose a training-free retrieval-based approach that prompts LLMs with relevant labeled examples, while InsightTab distills task-specific insights and combines them with selected demonstrations for classification (Yuan et al., 2025). However, these studies primarily investigate how retrieval strategies or additional information can improve LLM predictions. In contrast, our work compares frozen LLMs with Distance-KNN using identical retrieved neighbors, revealing when LLM inference provides additional predictive value and how these gains depend on neighborhood structure of demonstrations.

![](images/5943c4d8d06247e0c2ce1590d118b9858a54dce089c9abc009f24c2a84cb7955.jpg)  
Figure 2: Shared-neighbor comparison of Distance-KNN and LLMs for tabular classification. For each test query, a common retriever selects k = 8 labeled neighbors from the candidate pool. Distance-KNN uses their labels and distances for weighted voting, while the LLM uses the same neighbors as serialized demonstrations, followed by the query features. Retrieval distances are excluded from the LLM prompt. Varying the candidate-pool size while fixing k enables a controlled comparison on identical retrieved examples.

## 3 Experimental Setup

For each test query, a shared retriever extracts the k nearest (most similar) labeled training instances based on feature-space distance. Distance-KNN predicts from their labels and metric distances, whereas the LLM receives their serialized features and labels followed by the query features. By scaling the candidate-pool size under a fixed demonstration budget (k = 8), we compare their predictive behaviors on identical retrieved evidence (Figure 2).

## 3.1 Models

We evaluate four open-weighted LLMs: Qwen3- 8B and Qwen3-32B chat models (Yang et al.,

2025), alongside instruction-tuned Gemma3-12B and Gemma3-27B (Gemma Team et al., 2025). All generative LLMs are deployed locally via vLLM without quantization. Qwen3-Embedding-4B (Zhang et al., 2025a) is employed for text embedding retrieval in §5. Throughout our experiments, Distance-KNN serves as the reference predictor, using the exact same retrieved neighbors as the LLMs. Full checkpoint identifiers and implementation details are provided in Appendix A.1.

## 3.2 Datasets

We experiment on 19 public tabular classification datasets from UCI Machine Learning Repository (Kelly et al., 2023) and OpenML (Vanschoren et al., 2014). These datasets span binary and multiclass tasks across diverse domains, including healthcare, education, finance, industry, and software engineering. Each dataset is partitioned into training, validation, and test splits with the ratio of 8:1:1 via stratified sampling. All predictors share identical splits within each random seed. The training split serves as the retrieval candidate pool. We report macro-averaged accuracy across all datasets to prevent large test sets from dominating aggregate evaluation. Dataset statistics are detailed in Appendix A.2.

## 3.3 Protocol

Preprocessing. Feature types are determined from the training split. Integer-valued numerical features with at most 20 unique values are treated as categorical. Numerical features are standardized via Z-score normalization, with imputation and standardization statistics fitted strictly on the current candidate pool.

Candidate pools. For each dataset and random seed, we construct a fixed candidate sequence shared by all test queries and predictors. To ensure initial class coverage, we first randomly draw one training example from each class, and then sample remaining examples in proportion to class frequencies. Candidate pools of size $n \in$ {16, 32, 64, 128, 256, 512} are formed as nested prefixes of this sequence, with the full pool comprising the entire training split. For any query x, let $N _ { k } ( x ; P _ { n } )$ denote the k nearest neighbors retrieved from pool $P _ { n }$

Shared feature-space retrieval and voting. Both predictors use the same k nearest neighbors, selected using a mixed-feature distance:

$$
d ( x , z ) = \sqrt { \sum _ { j \in { \cal N } } ( \tilde { x } _ { j } - \tilde { z } _ { j } ) ^ { 2 } + \sum _ { j \in \mathcal { C } } { \bf 1 } [ x _ { j } \ne z _ { j } ] } ,\tag{1}
$$

where N and C denote numerical and categorical features, respectively, and tildes denote standardized values. Distance-KNN couples nearestneighbor retrieval with distance-weighted voting (Cover and Hart, 1967; Dudani, 1976). Predictions are aggregated via inverse distance weighting. Tie-breaking and near-zero distance handling are detailed in Appendix A.3.

LLM inference. The prompt concatenates the task description, feature names, label verbalizers, and the retrieved demonstrations. Demonstrations are serialized as feature key–value pairs with their ground-truth labels and ordered from nearest to farthest, followed by the query features. Predictions are obtained greedily (temperature 0) from label-token log-probabilities. Crucially, the LLM receives the exact same retrieved neighbors as Distance-KNN, with retrieval distances omitted from the prompt. The prompt template is provided in Appendix A.4.

Consensus and proximity. Label consensus summarizes neighborhood label homogeneity. Following Klingspohn et al. (2017), we define consensus as the majority class fraction among the retrieved neighbors:

$$
C ( x ; P _ { n } ) = \frac { 1 } { k } \operatorname* { m a x } _ { c } \sum _ { i \in N _ { k } ( x ; P _ { n } ) } { \bf 1 } [ y _ { i } = c ] .\tag{2}
$$

For example, among eight retrieved neighbors, a 4–4 label split yields a consensus of 4/8, whereas a 7–1 split yields 7/8. Unlike Distance-KNN voting, this measure does not weight labels by their retrieval distances.

We quantify neighborhood proximity (Klingspohn et al., 2017) using the mean distance between a test query and its retrieved neighbors (k = 8):

$$
\bar { d } _ { k } ( x ; P _ { n } ) = \frac { 1 } { k } \sum _ { i \in N _ { k } ( x ; P _ { n } ) } d ( x , x _ { i } ) .\tag{3}
$$

For each test query, we compute the mean distance to its retrieved neighbors, with smaller distances indicating closer neighborhoods in feature space. Within each dataset and seed, these distances are ranked across test queries and converted into percentiles to enable comparisons of relative proximity across datasets. Lower percentiles indicate closer neighborhoods, with the 0–20 percentile range representing the closest group. We examine LLM performance by label consensus in §4.2 and by its interaction with proximity in §4.3. Details of percentile computation are provided in Appendix B.3.

Evaluation. Accuracy is first averaged over several random seeds per dataset, and then macroaveraged across datasets (Gardner et al., 2024; Wu and Hou, 2025). Mean rank and wins are computed using seed-averaged accuracy. For paired comparisons, we report 95% bootstrap confidence intervals alongside exact sign-flip tests, with Holm-Bonferroni correction across four LLMs at each pool size (Dror et al., 2018). Shaded bands show ±1 standard deviation across seeds.

## 4 Results and Discussion

## 4.1 Do LLMs Outperform Distance-KNN as the Candidate Pool Expands?

Larger candidate pools improve all predictors, but Distance-KNN maintains the highest mean accuracy. Figure 3 summarizes the comparison across seven candidate-pool sizes with the retrieval budget fixed at $k = 8$ . As shown in Figure 3(a), mean accuracy increases consistently for all methods as the candidate pool expands, with Distance-KNN maintaining the highest accuracy at every pool size. Distance-KNN improves from 72.81% at 16 candidates to 85.50% at the full pool, while the strongest LLM, Gemma3-27B, follows a similar trajectory, rising from 72.73% to 85.26%. These results indicate that larger candidate pools benefit both approaches, but do not yield an aggregate accuracy advantage for LLM inference over distanceweighted voting.

![](images/7fe2e8c8935d96ea1668e558595b3fade6eccdb5d29467650e510af5e22e1ace.jpg)  
Figure 3: Performance of Distance-KNN and four LLMs using identical retrieved examples (k = 8) across 19 tabular prediction datasets with five seeds. (a) Mean accuracy as candidate-pool size increases, with shaded bands indicating ±1 standard deviation across seeds. (b) Mean rank across datasets at full pools (lower is better), with methods ranked by seed-averaged accuracy and ties assigned average ranks. (c) Number of wins at full pools (higher is better), with each dataset contributing one win divided equally among methods tied for the highest accuracy.

<table><tr><td>Model</td><td> $\Delta \mathrm { A c c }$  95% CI</td><td>PHolm</td></tr><tr><td>Gemma3-12B</td><td>-1.93 [-3.31,-0.79]</td><td>0.001</td></tr><tr><td>Gemma3-27B</td><td>-0.24 [-1.12, 0.53]</td><td>0.625</td></tr><tr><td>Qwen3-8B</td><td>-3.53 [-5.42, -1.84]</td><td>&lt; 0.001</td></tr><tr><td>Qwen3-32B</td><td>-1.95 [-3.30, -0.77]</td><td>0.012</td></tr></table>

Table 1: Statistical comparison of full-pool accuracy between Distance-KNN and four LLMs across 19 datasets. Paired differences $\Delta \mathrm { A c c } = \mathrm { A c c } _ { \mathrm { L L M } } - \mathrm { A c c } _ { \mathrm { K N N } }$ are computed after averaging five seeds. We report 95% paired percentile-bootstrap confidence intervals over datasets and exact sign-flip p-values adjusted for four comparisons using the Holm correction.

Distance-KNN also leads in mean rank and wins, although its advantage over the strongest LLM is marginal. Figure 3(b–c) further shows that Distance-KNN attains the lowest mean rank (2.03) and the highest wins (6.5) at the full pool, compared with 2.24 and 6.0 for Gemma3-27B. Paired comparisons in Table 1 show that Distance-KNN significantly outperforms three of the four LLMs after Holm correction, further substantiating its effectiveness. These results indicate that LLM inference does not improve mean accuracy over Distance-KNN, motivating a closer examination of how their relative performance varies with label consensus among the retrieved examples. Fullpool dataset-level accuracies are reported in $\mathsf { A p - }$ pendix B.1.

![](images/cdd2867d67dcb0e863a174e73a33561285b1f0dbbb55b128ce846c57fd4e9989.jpg)  
Figure 4: Accuracy differences between LLMs and Distance-KNN by retrieved-label consensus at full training pools $( k \ : = \ : 8 )$ . Consensus is the majority-label fraction among retrieved neighbors. Curves show mean $\Delta \mathrm { A c c } = \mathrm { A c c } _ { \mathrm { L L M } } - \mathrm { A c c } _ { \mathrm { K N N } }$ across 19 datasets and five seeds. Shaded bands denote ±1 standard deviation.

Takeaway. With the same retrieved examples, both LLMs and Distance-KNN benefit from larger candidate pools, yet LLM inference does not surpass distance-weighted voting in mean accuracy at any evaluated pool size.

## 4.2 When Does LLM Inference Add Value Beyond Distance-KNN?

To further examine when LLM inference adds value beyond Distance-KNN, we group test queries by label consensus among the shared retrieved examples. Figure 4 reports the mean accuracy differences of LLMs relative to Distance-KNN under the full-pool setting.

Low label consensus creates selective opportunities for LLM gains. At the lowest reported consensus level $( 4 / 8 )$ , three of the four LLMs surpass Distance-KNN with positive mean accuracy gaps, led by Gemma3-27B showing the largest improvement of 3.60 percentage points. Qwen3-8B, however, falls behind Distance-KNN, indicating that low consensus does not benefit all models equally. The override analysis in Appendix B.2 shows that beneficial corrections outweigh harmful overrides for the three improving models. These results suggest that some LLMs can leverage information in mixed-label demonstrations beyond what is captured by distance-weighted voting.

High label consensus favors Distance-KNN. Across consensus levels of $5 / 8 – 7 / 8$ Distance-KNN consistently outperforms all evaluated LLMs. The most pronounced deficit occurs for Qwen3- 8B, which trails Distance-KNN by 9.91 percentage points at $5 / 8$ consensus, while Gemma3-27B remains the closest competitor throughout this range. As detailed in Appendix B.2, harmful overrides exceed beneficial corrections across all models in these groups. This suggests that LLM inference can disrupt useful local evidence when the retrieved examples exhibit a high label consensus. At full label consensus $( 8 / 8 )$ , both types of overrides become rare, and the accuracy gaps approach zero.

Takeaway. The benefits of LLM inference are concentrated in low label consensus neighborhoods and depend on the model, whereas higher label consensus generally favors distance-weighted voting.

## 4.3 How Do Label Consensus and Feature Proximity Jointly Shape LLM Gains?

To examine whether feature proximity further distinguishes LLM gains under different levels of label consensus, Figure 5 shows how the accuracy gaps between LLMs and Distance-KNN vary across combinations of label consensus and feature proximity. Distance percentiles are computed from the mean distance to the eight shared neighbors within each dataset and seed, with lower values indicating closer neighborhoods (Appendix B.3).

Under low label consensus, feature proximity further distinguishes LLM gains and losses. $\mathrm { { A t } \leq 4 / 8 }$ consensus, three LLMs outperform Distance-KNN in the closest stratum (0–20 percentile), with Gemma3-27B reaching a gain of 8.1%. Notably, all four LLMs achieve positive gaps at the intermediate 40–60 percentile, showing that the advantages are not restricted to the closest neighborhoods. Conversely, all four models fall behind Distance-KNN at the most distant 80–

![](images/1f1f931f8e0c2c6de875b599b1aa55512ec6f6e71554d89d5f6be4c015d69eef.jpg)  
Figure 5: Accuracy differences between LLMs and Distance-KNN across retrieved-label consensus $( k = 8 )$ and feature proximity at full training pools. Cells show mean $\Delta \mathrm { A c c } = \mathrm { A c c } _ { \mathrm { L L M } } - \mathrm { A c c } _ { \mathrm { K N N } }$ across datasets. Columns indicate label consensus, while rows group test queries by distance percentile based on mean neighbor distance per dataset and seed (lower values indicate closer neighborhoods).

100 percentile, with accuracy gaps ranging from $- 9 . 8 \% \mathrm { t o } - 1 . 6 \%$ . These results indicate that feature proximity helps characterize LLM gains under label disagreement, although closer neighbors do not consistently yield larger advantages.

LLM gains largely disappear at higher label consensus, regardless of proximity. At $5 / 8 – 7 / 8$ consensus, Distance-KNN outperforms the LLMs in 58 of the 60 model–proximity combinations, including nearly all cases involving the closest neighbors. The only two positive gaps occur for Gemma3-27B, indicating that proximity alone is insufficient to recover a consistent LLM advantage when label consensus increases. At full consensus $( 8 / 8 )$ , the absolute accuracy gaps shrink to at most 0.7% across all models and proximity strata. Together, these findings suggest that feature proximity is most informative for distinguishing LLM gains under low label consensus, while higher consensus consistently limits such gains across proximity levels.

Takeaway. Feature proximity further differentiates LLM gains within low-consensus neighborhoods. LLMs benefit from selected close and intermediate neighborhoods but consistently fall behind Distance-KNN in the most distant ones. These gains remain model-dependent and largely disappear as label consensus increases.

![](images/f8ced96bd39daefc5235723942559bfc2c5b41ea9237362cb0e2857a3499934f.jpg)  
Figure 6: Effect of the number of demonstrations on classification accuracy across candidate-pool sizes. We compare (a) k = 4, (b) k = 8, and (c) k = 16, with Distance-KNN and LLMs using the same retrieved neighbors at each k. Curves show mean accuracy across 19 datasets and three seeds. Shaded bands denote ±1 standard deviation.

## 5 Complementary Analyses

We conduct four complementary analyses to examine how the number of demonstrations, prompt formulation, retrieval representation, and labelbalanced selection affect the comparison between LLM inference and distance-weighted voting. All experiments cover 19 datasets and three seeds using Gemma3-12B and Qwen3-8B. The first two analyses retain feature-based nearest-neighbor retrieval, while the latter two examine alternative retrieval representations and label-balanced selection, respectively.

Number of demonstrations. We vary the number of demonstrations $k ~ \in ~ \{ 4 , 8 , 1 6 \}$ to test whether the findings in §4.1 hold beyond the default setting of $k = 8$ . Figure 6 compares accuracy across all candidate-pool sizes, using the same retrieved examples for each LLM and Distance-KNN. Results show that mean accuracy increases with candidate-pool size for all predictors across three demonstration budgets, consistent with the results in Figure 3. Distance-KNN also maintains the highest mean accuracy in 19 of the 21 pool–k settings, with Gemma3-12B slightly ahead at the two smallest pools when $k = 1 6$ . Increasing k benefits Gemma3-12B more consistently than Qwen3-8B, while Distance-KNN shows no further improvement beyond eight neighbors. Overall, the main findings in Section 4.1 remain largely consistent across demonstration budgets, with limited exceptions at small candidate pools. Additional results are provided in Appendix C.1.

Prompt formulation. To examine whether the default prompt formulation limits LLM performance, we compare three formulations: key–value serialization (KV), natural-language serialization (NL), and rule-augmented natural-language serialization (Rule-Augmented NL). Our main experiments use KV, which presents features as key– value pairs, whereas NL expresses them as naturallanguage statements. Rule-augmented NL further supplements NL with training-derived classification rules using a procedure adapted from Insight-Tab (Yuan et al., 2025). All three formulations use the same eight retrieved demonstrations and their order, with additional rules supplied only in Rule-Augmented NL.

Figure 7 shows that prompt formulation affects the two LLMs differently, while Distance-KNN maintains the highest mean accuracy across all seven candidate-pool sizes. Gemma3-12B generally performs best with KV, with only minor differences among formulations. In contrast, Qwen3-8B consistently benefits from NL and Rule-Augmented NL over KV, with the latter achieving a notable accuracy gain of 3.53% at 64 candidates. These observations indicate that prompt formulation can improve LLM predictions, but the performance gap identified in §4.1 is not solely attributable to the default prompt formulation. Prompt templates and rule construction are detailed in Appendix C.2.

Retrieval representation. We compare the default feature-space retrieval (as defined in §3.3) with text-embedding retrieval, which represents serialized tabular instances using Qwen3-Embedding-4B. Table 2 reports the full-pool comparison, with results across all candidate-pool sizes in Appendix

![](images/099a8813a885b6cba7b6f21304969415a5c4c3ff9311cbdf20d610a3225b4e92.jpg)

![](images/bf6680e09f5e0efe741085cbd9ffd9e4b90b6d8c5b1ca1d608708bd2d6a1bc30.jpg)  
Figure 7: Effect of prompt formulation on classification accuracy for (a) Gemma3-12B and (b) Qwen3-8B across KV, NL and Rule-Augmented NL using identical retrieved examples (k = 8). Curves show mean accuracy across 19 datasets and three seeds as candidate-pool size increases. Shaded bands denote ±1 standard deviation.

<table><tr><td>Model</td><td>Feat-Space Text-Embed ∆Acc</td><td></td><td></td></tr><tr><td>Distance-KNN</td><td>85.75</td><td>69.89</td><td>+15.86</td></tr><tr><td>Gemma3-12B</td><td>84.19</td><td>68.51</td><td>+15.68</td></tr><tr><td>Qwen3-8B</td><td>82.59</td><td>58.39</td><td>+24.20</td></tr></table>

Table 2: Effect of retrieval representation on classification accuracy for Distance-KNN and two LLMs at full training pools. Feature-space retrieval is compared against text-embedding retrieval (k = 8) across 19 datasets and three seeds.

Table 7. Feature-space retrieval achieves higher mean accuracy for all three predictors at every evaluated pool size. At the full pool, switching to textembedding retrieval reduces accuracy by approximately 16 percentage points for both Distance-KNN and Gemma3-12B, and by 24.20 points for Qwen3-8B. Despite these changes, Distance-KNN maintains the highest mean accuracy under both retrieval representations. These results extend §4.1 by showing that retrieval representation strongly influences prediction quality, but changing the retrieval space does not lead to a consistent advantage for LLM inference.

Label-balanced demonstration selection. To complement our analyses of label consensus and feature proximity in §4.2 and §4.3, we compare the nearest eight demonstrations with eight labelbalanced examples selected from the nearest 32 candidates at full training pools. Table 3 shows that this strategy reduces mean accuracy by 7.49 and 10.58 percentage points for Gemma3-12B and Qwen3-8B, respectively, with declines on all 19 datasets. Unlike naturally occurring low-consensus neighborhoods examined earlier, this procedure increases label diversity by replacing some nearby examples with more distant ones, changing both label consensus and feature proximity. These results indicate that enforcing label balance does not improve LLM accuracy compared with nearestneighbor selection.

<table><tr><td>Model</td><td>Nearest Label-Balanced ∆Acc</td><td></td><td></td></tr><tr><td>Gemma3-12B</td><td>84.19</td><td>76.70</td><td>-7.49</td></tr><tr><td>Qwen3-8B</td><td>82.59</td><td>72.01</td><td>-10.58</td></tr></table>

Table 3: Effect of demonstration selection on classification accuracy for two LLMs at full training pools. The eight nearest neighbors are compared with eight label-balanced examples selected from the 32 nearest candidates (k = 8) across 19 datasets and three seeds.

## 6 Conclusion

We investigate whether and when LLMs add predictive value beyond distance-weighted voting on the same retrieved tabular examples. Across 19 datasets and expanding candidate pools, Distance-KNN achieves the highest mean accuracy at every pool size, although its margin over the strongest LLM is small. Some LLMs outperform voting under low label consensus, with gains varying by model and feature proximity. As label consensus increases, Distance-KNN generally retains the advantage, and low-consensus gains do not extend to the most distant neighborhoods. Complementary analyses indicate that changing the number of demonstrations, prompt formulation, retrieval representation, or label-balanced selection does not substantially alter these findings. Our work highlights the importance of comparing LLMs against traditional classifiers on the same retrieved examples and motivates further work on identifying the neighborhoods in which LLMs are most useful.

## Limitations

Our study focuses on the predictive value of general-purpose LLMs given the same retrieved neighbors as Distance-KNN. Although this design controls demonstration selection, it does not evaluate independently optimized end-to-end systems or task-specific model adaptation. Moreover, while our evaluation spans 19 diverse datasets, validating whether these behavioral patterns generalize to other emerging tabular benchmarks or highly domain-specific datasets remains a promising direction for future work.

## Ethical Considerations

Our study uses publicly available benchmark datasets and existing pretrained language models, involving no new data collection or human annotation. Our experiments aim to provide empirical insights into retrieval-based tabular prediction rather than establish the suitability of these methods for real-world deployment. Downstream applications in sensitive domains would require additional evaluation of reliability and fairness.

## References

Omer A. Alawi, Haslinda Mohamed Kamar, and Zaher Mundher Yaseen. 2024. Optimizing building energy performance predictions: A comparative study of artificial intelligence models. Journal ofBuilding Engineering, 88:109247.

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. Association for Computing Machinery.

Thomas M. Cover and Peter E. Hart. 1967. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1):21–27.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1383–1392, Melbourne, Australia. Association for Computational Linguistics.

Qiunan Du, Zhiliang Tian, Zhen Huang, Kailun Bian, Tianlun Liu, Zhaoning Zhang, Xinwang Liu, Feng Liu, and Dongsheng Li. 2025. Correlation-aware example selection for in-context learning with nonsymmetric determinantal point processes. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27066–27082, Suzhou, China. Association for Computational Linguistics.

Sahibsingh A. Dudani. 1976. The distance-weighted knearest-neighbor rule. IEEE Transactions on Systems, Man, and Cybernetics, SMC-6(4):325–327.

Josh Gardner, Juan C. Perdomo, and Ludwig Schmidt. 2024. Large scale transfer learning for tabular data via language modeling. In Advances in Neural Information Processing Systems, volume 37, pages 45155– 45205. Curran Associates, Inc.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. 2022. Why do tree-based models still outperform deep learning on typical tabular data? In Advances in Neural Information Processing Systems, volume 35, pages 507–520. Curran Associates, Inc.

Xinyi He, Yihao Liu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Zejian Yuan, and Dongmei Zhang. 2025. TableLoRA: Low-rank adaptation on table structure understanding for large language models. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 22376–22391, Vienna, Austria. Association for Computational Linguistics.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. 2023. TabLLM: Few-shot classification of tabular data with large language models. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5549–5581. PMLR.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. 2023. TabPFN: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025.

Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326.

Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. 2023. The UCI machine learning repository. University of California, Irvine.

Waldemar Klingspohn, Miriam Mathea, Antonius ter Laak, Nikolaus Heinrich, and Knut Baumann. 2017. Efficiency of different measures for defining the applicability domain of classification models. Journal ofCheminformatics, 9(1):44.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23, pages 611–626. Association for Computing Machinery.

Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2024. Table-GPT: Table fine-tuned GPT for diverse table tasks. Proceedings of the ACM on Management of Data, 2(3):176:1–176:28.

Hui Liu, Wenya Wang, Hao Sun, Chris Xing Tian, Chenqi Kong, Xin Dong, and Haoliang Li. 2025. Unraveling the mechanics of learning-based demonstration selection for in-context learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2623–2641, Vienna, Austria. Association for Computational Linguistics.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integrationfor Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.

Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari, Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, and Maksims Volkovs. 2025. TabDPT: Scaling tabular foundation models on real data. In Advances in Neural Information Processing Systems, volume 38, pages 172692–172722. Curran Associates, Inc.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Keqin Peng, Liang Ding, Yuanxin Ouyang, Meng Fang, Yancheng Yuan, and Dacheng Tao. 2025. Enhancing

input-label mapping in in-context learning with contrastive decoding. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 997–1004, Vienna, Austria. Association for Computational Linguistics.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings ofthe 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States. Association for Computational Linguistics.

Chenghao Sun, Zhen Huang, Yonggang Zhang, Le Lu, Houqiang Li, Xinmei Tian, Xu Shen, and Jieping Ye. 2025. Interpret and improve in-context learning via the lens of input-label mappings. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3873–3895, Vienna, Austria. Association for Computational Linguistics.

Yupeng Sun and Tian Li. 2024. A transformer-based framework for enterprise sales forecasting. PeerJ Computer Science, 10:e2503.

Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. 2014. OpenML: Networked science in machine learning. ACM SIGKDD Explorations Newsletter, 15(2):49–60.

Xumeng Wen, Han Zhang, Shun Zheng, Wei Xu, and Jiang Bian. 2024. From supervised to generative: A novel paradigm for tabular deep learning with large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3323–3333. Association for Computing Machinery.

Xumeng Wen, Shun Zheng, Zhen Xu, Yiming Sun, and Jiang Bian. 2025. Scalable in-context learning on tabular data via retrieval-augmented large language models. arXiv preprint arXiv:2502.03147.

Jie Wu and Mengshu Hou. 2025. An efficient retrievalbased method for tabular prediction with LLM. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 9917–9925, Abu Dhabi, UAE. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Jenny Yang, Hagen Triendl, Andrew A. S. Soltan, Mangal Prakash, and David A. Clifton. 2024. Addressing label noise for electronic health records: Insights from computer vision for tabular data. BMC Medical Informatics and Decision Making, 24(1):183.

Yifei Yuan, Jiatong Li, Weijia Zhang, Mohammad Aliannejadi, Evangelos Kanoulas, and Renjun Hu. 2025. Summarize-exemplify-reflect: Data-driven insight distillation empowers LLMs for few-shot tabular classification. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 12324– 12348, Suzhou, China. Association for Computational Linguistics.

Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2024. TableLlama: Towards open large generalist models for tables. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6024–6044, Mexico City, Mexico. Association for Computational Linguistics.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025a. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Ziniu Zhang, Zhenshuo Zhang, Dongyue Li, Lu Wang, Jennifer Dy, and Hongyang R. Zhang. 2025b. Lineartime demonstration selection for in-context learning via gradient estimation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 16459–16477, Suzhou, China. Association for Computational Linguistics.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12697–12706. PMLR.

## A Evaluation Protocol and Reproducibility

## A.1 Models and inference

Checkpoints. We use Qwen3-8B and Qwen3- 32B post-trained chat models (Yang et al., 2025), and gemma-3-12b-it and gemma-3-27bit instruction-tuned models (Gemma Team et al., 2025). All LLM parameters remain frozen. Models run locally through vLLM (Kwon et al., 2023) without quantization: Qwen3-8B and Gemma3- 12B each use one NVIDIA GeForce RTX 4090, while Qwen3-32B and Gemma3-27B each use one NVIDIA RTX PRO 6000 Blackwell.

Label scoring. Classes are mapped to numeric identifiers $1 , \ldots , C ,$ where $\begin{array} { r l r } { C } & { { } \le } & { 1 0 } \end{array}$ is the number of classes. Qwen3 uses enable\_thinking=False. Each request returns at most one output token and the top 20 first-token log-probabilities. After stripping token whitespace, we select the highest-scoring valid numeric label; labels absent from the returned list receive a score of −∞.

Embedding retrieval. Qwen/Qwen3- Embedding-4B runs through Hugging Face Transformers in FP16 without quantization on a single NVIDIA GeForce RTX 4090. Rows are serialized as compact feature–value JSON, excluding target labels. Query inputs use the Instruct: ... and Query: ... format with the instruction “Given a tabular record, retrieve training records with similar feature values for classification.” Candidate rows receive no instruction prefix. Last-token pooling and $L _ { 2 }$ normalization produce 2,560-dimensional embeddings. Retrieval uses cosine similarity, and embedding-KNN uses inverse-distance voting with $d _ { \mathrm { e m b } } = 1 - \cos ( e _ { x } , e _ { i } )$

## A.2 Datasets and analysis cohorts

Table 4 lists dataset characteristics and exact split sizes under the partitioning protocol described in Section 3. Feature counts refer to the processed inputs; dataset identifiers and target columns are specified in the released configuration. The main comparison covers 19 datasets, five seeds, seven candidate pools, and five predictors. The four complementary analyses use the same 19 datasets and three seeds; label-balanced retrieval uses full pools, and the other three analyses use all seven pool sizes.

## A.3 Implementation and statistics

Preprocessing details. Identifier-like and constant columns are removed by the dataset loader. Missing numerical values are imputed using candidate-pool medians; remaining categorical missing values use a pool-derived mode. Categorical values are converted to strings before overlap comparison.

Voting details. If any retrieved distance is at most $1 0 ^ { - 1 2 }$ (indicating “zero-distance” neighbors that almost exactly coincide with the test sample), only these identical neighbors are trusted to vote with equal weights, while all other retrieved neighbors are excluded from voting. Class-score ties are resolved by the first retrieved neighbor whose label belongs to a tied class.

<table><tr><td>Dataset</td><td>Rows</td><td>Features</td><td>Classes</td><td>Train / validation / test</td></tr><tr><td>AI4I Predictive Maintenance</td><td>10,000</td><td>6</td><td>2</td><td>8,000 / 1,000 / 1,000</td></tr><tr><td>Car</td><td>1,728</td><td>6</td><td>4</td><td>1,382 / 173 / 173</td></tr><tr><td>Cardiotocography</td><td>2,126</td><td>21</td><td>3</td><td>1,700 / 213 / 213</td></tr><tr><td>Credit G</td><td>1,000</td><td>20</td><td>2</td><td>800 / 100 / 100</td></tr><tr><td>Diabetes</td><td>768</td><td>8</td><td>2</td><td>614 / 77 / 77</td></tr><tr><td>Dry Bean</td><td>13,611</td><td>16</td><td>7</td><td>10,888 / 1,361 / 1,362</td></tr><tr><td>Glioma Grading</td><td>839</td><td>23</td><td>2</td><td>671 / 84 / 84</td></tr><tr><td>Iranian Churn</td><td>3,150</td><td>13</td><td>2</td><td>2,520 / 315 / 315</td></tr><tr><td>KC1</td><td>2,109</td><td>21</td><td>2</td><td>1,687 / 211 / 211</td></tr><tr><td>MAGIC Gamma</td><td>19,020</td><td>10</td><td>2</td><td>15,216 / 1,902 / 1,902</td></tr><tr><td>Online Shoppers</td><td>12,330</td><td>17</td><td>2</td><td>9,864 / 1,233 / 1,233</td></tr><tr><td>PC3</td><td>1,563</td><td>37</td><td>2</td><td>1,250 / 156 / 157</td></tr><tr><td>PC4</td><td>1,458</td><td>37</td><td>2</td><td>1,166 / 146 / 146</td></tr><tr><td>Phishing Websites</td><td>11,055</td><td>30</td><td>2</td><td>8,844 / 1,105 / 1,106</td></tr><tr><td>QSAR Biodegradation</td><td>1,055</td><td>41</td><td>2</td><td>844 / 105 / 106</td></tr><tr><td>Spambase</td><td>4,601</td><td>57</td><td>2</td><td>3,680 / 460 / 461</td></tr><tr><td>Steel Plates Fault</td><td>1,941</td><td>27</td><td>7</td><td>1,552 / 194 / 195</td></tr><tr><td>Student Dropout</td><td>4,424</td><td>36</td><td>3</td><td>3,539 / 442 / 443</td></tr><tr><td>Vehicle</td><td>846</td><td>18</td><td>4</td><td>676 / 85 / 85</td></tr></table>

Table 4: Characteristics and train/validation/test split sizes of the 19 tabular classification datasets. Splits are stratified by class in an 8:1:1 ratio, and feature counts refer to the processed inputs.

Statistical details. Percentile-bootstrap intervals use 10,000 paired dataset resamples and the 2.5th and 97.5th percentiles. Reported sign-flip tests enumerate all $2 ^ { 1 9 }$ sign assignments. Holm correction applies across the four LLM comparisons within each metric and candidate-pool size. Bootstrap intervals are pointwise, not simultaneous confidence bands. Seed standard deviations use the sample estimator with denominator S − 1.

## A.4 Classification prompt

The main classification prompt is shown below.   
Label scoring follows Appendix A.1.

Retrieved-Example-Based Classification   
You are a helpful data analyst.   
Task description:   
{task\_description}   
Features:   
{feature\_names}   
Label mapping:   
{label\_mapping}   
The following are the top-k most   
similar training instances retrieved   
for the target instance. They are   
ordered from the most similar to the   
least similar.   
Use these retrieved examples to infer   
the label of the target instance.   
CRITICAL: Your response must   
be EXACTLY one number from   
{candidate\_labels}.

Output ONLY the number, nothing else.   
No explanations, no reasoning, no   
tags, no extra text.   
Labeled instances:   
{retrieved\_examples}   
Target instance:   
{target\_instance}   
Your answer (one number only):

## B Supplementary Results for RQ1–RQ3

## B.1 RQ1: Dataset-level accuracy

Table 5 reports full-pool dataset-level accuracy.

## B.2 RQ2: Complementary predictions and overrides

Figure 8 illustrates the beneficial and harmful cases defined in Section 4.2.

Aggregation and empty groups. At each consensus level, we compute beneficial and harmful override rates separately for each dataset and seed. The denominator is the total number of queries at that consensus level. Within each seed, we average these rates equally across datasets with non-empty groups, and then average the resulting values across the five seeds. Empty groups are excluded rather than assigned a rate of zero. Figures 9 and 4 use the same query pairing and aggregation procedure.

The displayed consensus levels are $4 / 8 – 8 / 8 .$ At $4 / 8$ , Glioma Grading and PC3 each have one empty seed-level group, yielding 18 or 19 datasets per seed. All groups contain 19 datasets at 5/8–8/8.

<table><tr><td>Dataset</td><td>Distance-KNN</td><td>Gemma3-12B</td><td>Gemma3-27B</td><td>Qwen3-8B</td><td>Qwen3-32B</td></tr><tr><td>AI4I Maintenance</td><td>97.58</td><td>97.56</td><td>97.32</td><td>96.78</td><td>97.04</td></tr><tr><td>Car</td><td>90.17</td><td>89.25</td><td>93.06</td><td>89.83</td><td>90.87</td></tr><tr><td>Cardiotocography</td><td>91.92</td><td>91.27</td><td>92.02</td><td>90.42</td><td>93.52</td></tr><tr><td>Credit G</td><td>73.60</td><td>73.60</td><td>73.00</td><td>73.20</td><td>71.40</td></tr><tr><td>Diabetes</td><td>74.29</td><td>63.90</td><td>68.31</td><td>63.12</td><td>69.35</td></tr><tr><td>Dry Bean</td><td>92.29</td><td>91.78</td><td>92.13</td><td>92.20</td><td>92.35</td></tr><tr><td>Glioma Grading</td><td>84.52</td><td>85.24</td><td>86.19</td><td>84.29</td><td>83.33</td></tr><tr><td>Iranian Churn</td><td>94.35</td><td>93.71</td><td>94.79</td><td>91.75</td><td>94.10</td></tr><tr><td>KC1</td><td>86.82</td><td>80.57</td><td>85.59</td><td>75.26</td><td>80.38</td></tr><tr><td>MAGIC Gamma</td><td>84.54</td><td>84.22</td><td>84.71</td><td>84.64</td><td>82.64</td></tr><tr><td>Online Shoppers</td><td>87.98</td><td>82.32</td><td>86.08</td><td>78.02</td><td>78.98</td></tr><tr><td>PC3</td><td>89.30</td><td>85.35</td><td>88.54</td><td>85.35</td><td>86.24</td></tr><tr><td>PC4</td><td>88.49</td><td>84.11</td><td>87.40</td><td>82.88</td><td>84.79</td></tr><tr><td>Phishing Websites</td><td>96.93</td><td>97.00</td><td>96.73</td><td>86.71</td><td>96.93</td></tr><tr><td>QSAR Biodegradation</td><td>85.28</td><td>84.34</td><td>85.28</td><td>85.66</td><td>84.72</td></tr><tr><td>Spambase</td><td>91.28</td><td>90.54</td><td>90.11</td><td>87.20</td><td>91.63</td></tr><tr><td>Steel Plates Fault</td><td>75.18</td><td>73.85</td><td>75.38</td><td>74.46</td><td>75.59</td></tr><tr><td>Student Dropout</td><td>70.74</td><td>70.56</td><td>73.18</td><td>68.89</td><td>64.15</td></tr><tr><td>Vehicle</td><td>69.18</td><td>68.71</td><td>70.12</td><td>66.82</td><td>69.41</td></tr></table>

Table 5: Full-pool classification accuracy (%) of Distance-KNN and four LLMs on 19 datasets using shared retrieved neighbors $( k = 8 )$ . Each entry averages five seeds. Bold marks the highest displayed accuracy within each dataset, including ties.

![](images/5851d849549059b58874d825539c8c3e4e4c386e6ec1efe5be47214ff391c03f.jpg)  
Figure 8: Schematic illustration of beneficial and harmful overrides on shared queries and retrieved neighbors. Beneficial overrides occur when an LLM predicts correctly and Distance-KNN predicts incorrectly, while harmful overrides reverse these outcomes.

Empty groups are omitted consistently across models, rather than assigned zero.

Dataset-level rates. Table 6 reports full-pool rates without consensus conditioning. Each entry averages five seeds; before rounding, beneficial minus harmful equals the corresponding LLM–KNN accuracy difference in Table 5.

## B.3 RQ3: Proximity and label composition

We measure proximity using the mean distance from each test query x to its eight retrieved neighbors under the full-pool setting, denoted by $\bar { d } _ { 8 } ( x ; P _ { \mathrm { f u l l } } )$ . Because raw distances may have different scales across datasets, we convert them into relative distance percentiles within each dataset and seed.

![](images/342452d2a91fe5387a5f7bcf7a37cabba0c80594e02572bded0b19f8a1c09978.jpg)  
Figure 9: Beneficial (B) and harmful (H) override rates (%) by retrieved-label consensus at full training pools $( k = 8 )$ . Each rate uses all queries within its dataset–seed–consensus group as the denominator. Rates are averaged equally over non-empty dataset groups within each seed and then over five seeds. As shown in Figure 4, the mean accuracy difference satisfies $\mathrm { A c c } _ { \mathrm { L L M } } - \mathrm { A c c } _ { \mathrm { K N N } } = B - H$

Specifically, we rank the test queries by their mean neighbor distance in ascending order. The distance percentile is

$$
p _ { d , s } ( x ) = 1 0 0 \times \frac { \mathrm { r a n k } _ { \mathrm { a v g } } \big ( \bar { d } _ { 8 } ( x ; P _ { \mathrm { f u l l } } ) \big ) } { | \mathcal { Q } _ { d , s } | } ,\tag{4}
$$

<table><tr><td></td><td colspan="2">Gemma3-12B</td><td colspan="2">Gemma3-27B</td><td colspan="2">Qwen3-8B</td><td colspan="2">Qwen3-32B</td></tr><tr><td>Dataset</td><td>B</td><td>H</td><td>B</td><td>H</td><td>B</td><td>H</td><td>B</td><td>H</td></tr><tr><td>AI4I Maintenance</td><td>0.58</td><td>0.60</td><td>0.44</td><td>0.70</td><td>0.58</td><td>1.38</td><td>0.66</td><td>1.20</td></tr><tr><td>Car</td><td>5.66</td><td>6.59</td><td>5.55</td><td>2.66</td><td>4.28</td><td>4.62</td><td>5.55</td><td>4.86</td></tr><tr><td>Cardiotocography</td><td>1.69</td><td>2.35</td><td>1.60</td><td>1.50</td><td>1.41</td><td>2.91</td><td>2.91</td><td>1.31</td></tr><tr><td>Credit G</td><td>4.80</td><td>4.80</td><td>5.00</td><td>5.60</td><td>4.80</td><td>5.20</td><td>5.40</td><td>7.60</td></tr><tr><td>Diabetes</td><td>11.17</td><td>21.56</td><td>8.57</td><td>14.55</td><td>12.21</td><td>23.38</td><td>7.79</td><td>12.73</td></tr><tr><td>Dry Bean</td><td>0.84</td><td>1.35</td><td>0.57</td><td>0.73</td><td>0.75</td><td>0.84</td><td>0.69</td><td>0.63</td></tr><tr><td>Glioma Grading</td><td>4.29</td><td>3.57</td><td>4.05</td><td>2.38</td><td>5.24</td><td>5.48</td><td>4.05</td><td>5.24</td></tr><tr><td>Iranian Churn</td><td>1.52</td><td>2.16</td><td>1.59</td><td>1.14</td><td>1.27</td><td>3.87</td><td>1.84</td><td>2.10</td></tr><tr><td>KC1</td><td>4.08</td><td>10.33</td><td>1.80</td><td>3.03</td><td>6.16</td><td>17.73</td><td>3.98</td><td>10.43</td></tr><tr><td>MAGIC Gamma</td><td>3.55</td><td>3.88</td><td>3.50</td><td>3.33</td><td>2.80</td><td>2.70</td><td>3.05</td><td>4.95</td></tr><tr><td>Online Shoppers</td><td>3.88</td><td>9.54</td><td>3.26</td><td>5.16</td><td>5.27</td><td>15.23</td><td>5.03</td><td>14.03</td></tr><tr><td>PC3</td><td>4.97</td><td>8.92</td><td>4.08</td><td>4.84</td><td>4.33</td><td>8.28</td><td>3.95</td><td>7.01</td></tr><tr><td>PC4</td><td>4.66</td><td>9.04</td><td>3.70</td><td>4.79</td><td>5.48</td><td>11.10</td><td>4.38</td><td>8.08</td></tr><tr><td>Phishing Websites</td><td>1.01</td><td>0.94</td><td>0.67</td><td>0.87</td><td>0.99</td><td>11.21</td><td>0.90</td><td>0.90</td></tr><tr><td>QSAR Biodegradation</td><td>2.26</td><td>3.21</td><td>2.45</td><td>2.45</td><td>2.08</td><td>1.70</td><td>2.26</td><td>2.83</td></tr><tr><td>Spambase</td><td>2.52</td><td>3.25</td><td>2.52</td><td>3.69</td><td>2.56</td><td>6.64</td><td>2.34</td><td>2.00</td></tr><tr><td>Steel Plates Fault</td><td>3.79</td><td>5.13</td><td>2.46</td><td>2.26</td><td>2.15</td><td>2.87</td><td>3.28</td><td>2.87</td></tr><tr><td>Student Dropout</td><td>5.46</td><td>5.64</td><td>5.15</td><td>2.71</td><td>3.88</td><td>5.73</td><td>8.89</td><td>15.49</td></tr><tr><td>Vehicle</td><td>3.76</td><td>4.24</td><td>3.06</td><td>2.12</td><td>2.82</td><td>5.18</td><td>2.82</td><td>2.59</td></tr></table>

Table 6: Full-pool beneficial (B) and harmful (H) override rates (%) for four LLMs relative to Distance-KNN across 19 datasets (k = 8). Within each dataset, rates use all test queries as the denominator for each seed and are averaged over five seeds. Note that $\mathrm { A c c } _ { \mathrm { L L M } } - \mathrm { A c c } _ { \mathrm { K N N } } = B - H$ (before rounding).

where $\mathcal { Q } _ { d , s }$ is the set of test queries for dataset d and seed s. Ranks start at 1, and tied distances receive their average rank. A lower percentile indicates a relatively closer neighborhood within the same dataset and seed; it does not measure absolute proximity across datasets.

These percentiles are computed from shared neighbor distances, so they remain identical across predictors. They require no ground-truth query labels and serve only as a post-hoc diagnostic, without altering retrieval or prediction. For the heatmap, we group percentiles into five equal-width bands: (0, 20], (20, 40], (40, 60], (60, 80], and (80, 100]. These bands need not contain equal numbers of queries because of tied distances.

Figure 10 illustrates how neighborhoods with the same label consensus can differ in proximity.

![](images/5678c8173dc3d59eb8a5a12b3155f0d5869b9511a4ff53d6dc816ccaa271e599.jpg)  
Figure 10: Schematic comparison of label consensus and feature proximity. The two neighborhoods have the same label composition, with five of eight neighbors in the majority class, but differ in their distances to the query.

## C Robustness and Complementary Analyses

## C.1 Demonstration count

Figure 6 compares $k = 4 , 8 ,$ 16 on 19 datasets and three seeds. Full-pool Distance-KNN accuracy rises by 0.80 percentage points from k = 4 to 8, then falls by 0.21 at k = 16. Gemma3-12B leads at the two smallest pools with k = 16, while Distance-KNN leads at full pools for every tested budget.

## C.2 Prompt formulation

Conditions. KV uses feature key–value pairs; NL uses statements of the form “the {feature} is {value}”; Rule-Augmented NL augments NL with rules from labeled candidate-pool data. All conditions share the same eight retrieved examples, labels, and order, and use three seeds.

Rule construction. For Rule-Augmented NL, we adapt the Summarize–Exemplify–Reflect ruledistillation procedure of InsightTab (Yuan et al., 2025). For each dataset, seed, and candidate pool, a class-balanced subset of up to 128 samples trains an auxiliary XGBoost classifier (100 trees, maximum depth 6, learning rate 0.3, exact tree method). The first tree’s leaves define sample groups. Predictive entropy guides the selection of eight lowentropy examples and a high-entropy subset containing approximately half of the selected samples, with selection aiming to balance classes.

The classification checkpoint (google/gemma-3- 12b-it or Qwen/Qwen3-8B) also summarizes and merges group-wise rules. Using these rules and the low-entropy examples, it predicts the high-entropy examples; errors and their correct labels support one additional round of rule generation. XGBoost assists rule construction, while the LLM makes the final test predictions.

Validation and reuse. Rules are JSON objects with one to three feature conditions joined by AND and a predicted class. Generated, merged, and additional rules are evaluated on the entire current candidate pool. Acceptance requires support ≥ 3, coverage ≥ 0.01, precision ≥ 0.70, and a 95% Wilson lower bound $\geq 0 . 4 0$ ; this empirical lower bound replaces the rule confidence. Unparseable outputs produce no rules, invalid rules are discarded, and an empty retained set gives empty rule arrays in the prediction template.

Construction, validation, and refinement use training features and labels only. Rules are cached per dataset, seed, candidate pool, and model configuration, then reused across test queries. The construction examples do not replace the eight retrieved demonstrations. Rule-generation requests use temperature 0 and at most 512 output tokens; input sections for summarization, merging, and reflection are limited to 24,000 characters, with excess content omitted or truncated. Final classification uses temperature 0 and first-token label log-probabilities. LLM parameters remain frozen; only the auxiliary XGBoost model is trained.

Prediction template. The merged and additional rules below are validated JSON arrays. The labeled examples are the same eight rows used by Distance-KNN.

Rule-Augmented NL: Classification Prompt   
You are performing tabular   
classification using distilled   
training-data insights.   
Task description: {task\_description}   
Label mapping:   
{label\_mapping}   
Useful patterns for the task:   
{merged\_rules}   
Additional patterns learned from   
difficult misclassified examples:   
{additional\_rules}   
[FEW-SHOT EXAMPLES START]   
{eight\_retrieved\_labeled\_examples}   
[FEW-SHOT EXAMPLES END]

```objectivec
[CURRENT QUESTION START]
{query_features_in_natural_language}
[CURRENT QUESTION END]
Your response must be EXACTLY one
number from {valid_label_ids}.
Output ONLY the number, nothing else:
```

## C.3 Feature-based versus embedding retrieval

Table 7 extends Table 2 to all seven candidate pools. Values average the same 19 datasets and three seeds, using the k = 8 feature-based results and the embedding results with matched coverage. Feature-based retrieval has higher accuracy in all 21 predictor-by-pool comparisons.

At full pools, feature-based retrieval improves Distance-KNN and Qwen3-8B on all 19 datasets and Gemma3-12B on 18; embedding retrieval improves Gemma3-12B on KC1 by 2.21 percentage points. Figure 11 reports the dataset-level embedding results.

![](images/38e0dc10b0c34bbac7108fab31270174132b84e56428d4b83eea1b00ecb116f6.jpg)  
Figure 11: Dataset-level classification accuracy (%) under text-embedding retrieval at full training pools (k = 8). Results compare Distance-KNN, Gemma3- 12B, and Qwen3-8B on 19 datasets, with each value averaged over three seeds.

## C.4 Label-balanced selection

At full pools, the nearest eight rows are compared with eight rows selected from the nearest 32 for label balance. Each step selects an available label with the fewest chosen examples; ties favor the closest remaining row. The final eight rows are ordered by distance, making label counts as even as candidate availability permits. The comparison uses seeds 42, 3407, and 5201, with matched splits, prompts, and inference settings between selection conditions within each seed.

<table><tr><td></td><td colspan="3">Feature-space retrieval</td><td colspan="3">Text-embedding retrieval</td></tr><tr><td>Pool</td><td>Distance- KNN</td><td>Gemma3-12B</td><td>Qwen3-8B</td><td>Distance- KNN</td><td>Gemma3-12B</td><td>Qwen3-8B</td></tr><tr><td>16</td><td>73.03</td><td>70.44</td><td>67.45</td><td>67.79</td><td>67.04</td><td>60.12</td></tr><tr><td>32</td><td>75.95</td><td>73.80</td><td>70.10</td><td>67.46</td><td>66.43</td><td>60.80</td></tr><tr><td>64</td><td>77.96</td><td>75.38</td><td>71.69</td><td>68.07</td><td>66.86</td><td>60.14</td></tr><tr><td>128</td><td>80.67</td><td>78.30</td><td>75.31</td><td>67.99</td><td>67.05</td><td>59.26</td></tr><tr><td>256</td><td>82.38</td><td>79.49</td><td>77.13</td><td>68.33</td><td>67.52</td><td>57.02</td></tr><tr><td>512</td><td>83.43</td><td>81.39</td><td>79.12</td><td>69.53</td><td>68.35</td><td>58.37</td></tr><tr><td>full</td><td>85.75</td><td>84.19</td><td>82.59</td><td>69.89</td><td>68.51</td><td>58.39</td></tr></table>

Table 7: Effect of retrieval representation on classification accuracy (%) across candidate-pool sizes. Feature-space and text-embedding retrieval are compared for Distance-KNN, Gemma3-12B, and Qwen3-8B with k = 8. Entries report mean accuracy across 19 datasets and three seeds.

<table><tr><td></td><td colspan="3">Gemma3-12B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>Dataset</td><td>Nearest</td><td>Balanced</td><td>∆Acc</td><td>Nearest</td><td>Balanced</td><td>∆Acc</td></tr><tr><td>AI4I Predictive Maintenance</td><td>97.57</td><td>95.60</td><td>-1.97</td><td>96.63</td><td>90.17</td><td>-6.47</td></tr><tr><td>Car</td><td>89.40</td><td>86.51</td><td>-2.89</td><td>90.75</td><td>84.39</td><td>-6.36</td></tr><tr><td>Cardiotocography</td><td>90.92</td><td>89.51</td><td>-1.41</td><td>90.45</td><td>84.98</td><td>-5.48</td></tr><tr><td>Credit G</td><td>75.00</td><td>69.33</td><td>-5.67</td><td>74.67</td><td>72.67</td><td>-2.00</td></tr><tr><td>Diabetes</td><td>64.50</td><td>64.07</td><td>-0.43</td><td>61.90</td><td>50.65</td><td>-11.26</td></tr><tr><td>Dry Bean</td><td>91.61</td><td>87.00</td><td>-4.60</td><td>92.32</td><td>91.34</td><td>-0.98</td></tr><tr><td>Glioma Grading</td><td>87.30</td><td>83.73</td><td>-3.57</td><td>86.11</td><td>80.95</td><td>-5.16</td></tr><tr><td>Iranian Churn</td><td>94.29</td><td>90.90</td><td>-3.39</td><td>92.80</td><td>84.02</td><td>-8.78</td></tr><tr><td>KC1</td><td>82.62</td><td>72.20</td><td>-10.43</td><td>77.09</td><td>69.35</td><td>-7.74</td></tr><tr><td>MAGIC Gamma</td><td>84.45</td><td>81.42</td><td>-3.03</td><td>84.98</td><td>80.20</td><td>-4.78</td></tr><tr><td>Online Shoppers</td><td>82.54</td><td>65.29</td><td>-17.25</td><td>78.08</td><td>59.23</td><td>-18.84</td></tr><tr><td>PC3</td><td>86.41</td><td>65.61</td><td>-20.81</td><td>85.99</td><td>66.88</td><td>-19.11</td></tr><tr><td>PC4</td><td>84.70</td><td>68.04</td><td>-16.67</td><td>84.25</td><td>66.44</td><td>-17.81</td></tr><tr><td>Phishing Websites</td><td>96.93</td><td>95.00</td><td>-1.93</td><td>87.25</td><td>73.96</td><td>-13.29</td></tr><tr><td>QSAR Biodegradation</td><td>87.74</td><td>85.53</td><td>-2.20</td><td>88.68</td><td>82.70</td><td>-5.97</td></tr><tr><td>Spambase</td><td>90.17</td><td>76.79</td><td>-13.38</td><td>87.06</td><td>71.73</td><td>-15.33</td></tr><tr><td>Steel Plates Fault</td><td>72.65</td><td>65.47</td><td>-7.18</td><td>72.82</td><td>64.10</td><td>-8.72</td></tr><tr><td>Student Dropout</td><td>70.58</td><td>53.27</td><td>-17.31</td><td>68.77</td><td>49.36</td><td>-19.41</td></tr><tr><td>Vehicle</td><td>70.20</td><td>61.96</td><td>-8.24</td><td>68.63</td><td>45.10</td><td>-23.53</td></tr><tr><td>Dataset mean</td><td>84.19</td><td>76.70</td><td>-7.49</td><td>82.59</td><td>72.01</td><td>-10.58</td></tr></table>

Table 8: Effect of demonstration selection on full-pool classification accuracy (%) for Gemma3-12B and Qwen3-8B across 19 datasets. Nearest uses the eight closest examples, while Balanced selects eight label-balanced examples from the nearest 32 candidates. Dataset entries average three seeds. $\Delta \mathrm { A c c } = \mathrm { A c c } _ { \mathrm { B a l a n c e d } } - \mathrm { A c c } _ { \mathrm { N e a r e s t } }$ negative values favor nearest-neighbor selection.

Table 8 gives the dataset-level results behind Table 3. Balancing reduces mean accuracy by 7.49 percentage points for Gemma3-12B and 10.58 for Qwen3-8B, with decreases on every dataset. The selection changes both label composition and the identity and proximity of the retrieved rows.

## D Complete Accuracy Results

Tables 9–11 report five-seed mean accuracy (%) for every dataset and candidate-pool size. Within each dataset and pool, the highest displayed accuracy is bold and the second-highest distinct accuracy is underlined; ties share the same emphasis. Distance-KNN denotes distance-weighted voting.

Table 9: Classification accuracy (%) of Distance-KNN and four LLMs by dataset and candidate-pool size (part 1 of 3, k = 8). Values average five seeds. Within each dataset and pool, bold and underlined values mark the highest and second-highest distinct displayed accuracies, respectively, including ties.
<table><tr><td>Pool</td><td>Distance-KNN</td><td>Gemma3-12B</td><td>Gemma3-27B</td><td>Qwen3-8B</td><td>Qwen3-32B</td></tr><tr><td colspan="6">AI4I Predictive Maintenance</td></tr><tr><td>16</td><td>96.60</td><td>94.06</td><td>93.72</td><td>94.06</td><td>90.30</td></tr><tr><td>32</td><td>96.60</td><td>94.90</td><td>95.46</td><td>95.52</td><td>93.42</td></tr><tr><td>64</td><td>96.60</td><td>95.66</td><td>95.52</td><td>95.60</td><td>94.24</td></tr><tr><td>128</td><td>96.58</td><td>95.62</td><td>95.72</td><td>95.46</td><td>94.72</td></tr><tr><td>256</td><td>96.54</td><td>96.34</td><td>96.08</td><td>95.76</td><td>95.02</td></tr><tr><td>512</td><td>96.72</td><td>96.44</td><td>96.38</td><td>95.74</td><td>95.86</td></tr><tr><td>full</td><td>97.58</td><td>97.56</td><td>97.32</td><td>96.78</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>97.04</td></tr><tr><td colspan="6">Car</td></tr><tr><td>16</td><td>70.17</td><td>68.32</td><td>70.75</td><td>70.75</td><td>69.36</td></tr><tr><td>32</td><td>71.56</td><td>71.79</td><td>73.64</td><td>72.37</td><td>72.02</td></tr><tr><td>64</td><td>74.68</td><td>74.91</td><td>76.65</td><td>74.57</td><td>75.14</td></tr><tr><td>128</td><td>78.84</td><td>78.96</td><td>80.12</td><td>76.88</td><td></td></tr><tr><td>256</td><td>81.39</td><td>81.27</td><td>82.77</td><td>80.12</td><td>78.50</td></tr><tr><td>512</td><td>84.62</td><td>85.32</td><td>86.94</td><td>84.16</td><td>80.46</td></tr><tr><td>full</td><td>90.17</td><td>89.25</td><td>93.06</td><td>89.83</td><td>85.66 90.87</td></tr><tr><td colspan="2">Cardiotocography</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>78.03</td><td>81.78</td><td>79.81</td><td>80.94</td><td>81.60</td></tr><tr><td>32</td><td>80.94</td><td>84.32</td><td>81.97</td><td>82.82</td><td>83.00</td></tr><tr><td>64</td><td>83.29</td><td>84.69</td><td>83.94</td><td>82.72</td><td>84.32</td></tr><tr><td>128</td><td>86.85</td><td>86.95</td><td>86.76</td><td>85.63</td><td>87.04</td></tr><tr><td>256</td><td>88.45</td><td>87.70</td><td>87.89</td><td>87.61</td><td>88.08</td></tr><tr><td>512</td><td>89.39</td><td>89.39</td><td>90.42</td><td>88.54</td><td>89.48</td></tr><tr><td>full</td><td>91.92</td><td>91.27</td><td>92.02</td><td>90.42</td><td>93.52</td></tr><tr><td colspan="2">Credit G</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>66.20</td><td>68.60</td><td>65.80</td><td>69.80</td><td>70.00</td></tr><tr><td>32</td><td>70.80</td><td>68.40</td><td>67.20</td><td>72.00</td><td>69.40</td></tr><tr><td>64</td><td>70.20</td><td>69.00</td><td>67.40</td><td>69.40</td><td>67.80</td></tr><tr><td>128</td><td>70.80</td><td>71.40</td><td>69.80</td><td>70.80</td><td>68.80</td></tr><tr><td>256</td><td>74.00</td><td>71.60</td><td>72.00</td><td>73.00</td><td>68.20</td></tr><tr><td>512</td><td>72.20</td><td>70.40</td><td>69.80</td><td>71.20</td><td>70.00</td></tr><tr><td>full</td><td>73.60</td><td>73.60</td><td>73.00</td><td>73.20</td><td>71.40</td></tr><tr><td colspan="2">Diabetes</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>70.65</td><td>65.97</td><td>67.01</td><td>48.57</td><td>73.25</td></tr><tr><td>32</td><td>71.43</td><td>67.01</td><td>68.31</td><td>57.66</td><td>71.43</td></tr><tr><td>64</td><td>70.91</td><td>65.71</td><td>64.16</td><td>57.14</td><td>70.91</td></tr><tr><td>128</td><td>75.58</td><td>67.53</td><td>70.65</td><td>63.12</td><td>72.21</td></tr><tr><td>256</td><td>74.81</td><td>65.71</td><td>69.87</td><td>59.22</td><td>70.39</td></tr><tr><td>512</td><td>74.29</td><td>66.23</td><td>70.65</td><td>60.78</td><td>71.17</td></tr><tr><td>full</td><td>74.29</td><td>63.90</td><td>68.31</td><td>63.12</td><td>69.35</td></tr><tr><td colspan="2">Dry Bean</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>64.49</td><td>71.89</td><td>74.39</td><td>70.69</td><td>76.04</td></tr><tr><td>32</td><td>77.33</td><td>80.09</td><td>81.60</td><td>79.62</td><td>83.08</td></tr><tr><td>64</td><td>84.43</td><td>84.93</td><td>86.20</td><td>85.65</td><td>86.27</td></tr><tr><td>128</td><td>87.65</td><td>88.09</td><td>88.21</td><td>87.91</td><td>88.22</td></tr><tr><td>256</td><td>89.24</td><td>89.06</td><td>89.71</td><td>89.22</td><td>89.09</td></tr><tr><td>512</td><td>90.12</td><td>89.52</td><td>89.99</td><td>90.31</td><td>90.18</td></tr><tr><td>full</td><td>92.29</td><td>91.78</td><td>92.13</td><td>92.20</td><td>92.35</td></tr><tr><td colspan="2">Glioma Grading</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>76.43</td><td>75.48</td><td>76.90</td><td>76.19</td><td>80.00</td></tr><tr><td>32</td><td>78.10</td><td>78.10</td><td>81.43</td><td>75.71</td><td>80.71</td></tr><tr><td>64</td><td>80.71</td><td>79.29</td><td>80.95</td><td>78.81</td><td>79.52</td></tr><tr><td>128</td><td>83.57</td><td>82.14</td><td>83.81</td><td>84.52</td><td>83.81</td></tr><tr><td>256</td><td>83.10</td><td>81.43</td><td>80.95</td><td>80.71</td><td>83.57</td></tr><tr><td>512</td><td>84.29</td><td>83.81</td><td>84.52</td><td>83.57</td><td>82.62</td></tr><tr><td>full</td><td>84.52</td><td>85.24</td><td>86.19</td><td>84.29</td><td>83.33</td></tr></table>

Table 10: Classification accuracy (%) of Distance-KNN and four LLMs by dataset and candidate-pool size (part 2 of 3, k = 8). Values average five seeds. Within each dataset and pool, bold and underlined values mark the highest and second-highest distinct displayed accuracies, respectively, including ties
<table><tr><td>Pool</td><td>Distance-KNN</td><td>Gemma3-12B</td><td>Gemma3-27B</td><td>Qwen3-8B</td><td>Qwen3-32B</td></tr><tr><td>Iranian Churn</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>84.13</td><td>82.29</td><td>84.32</td><td>73.65</td><td>81.84</td></tr><tr><td>32</td><td>84.70</td><td>86.35</td><td>87.94</td><td>78.98</td><td>83.75</td></tr><tr><td>64</td><td>85.14</td><td>86.29</td><td>87.81</td><td>81.78</td><td>85.65</td></tr><tr><td>128</td><td>86.98</td><td>87.56</td><td>88.00</td><td>83.81</td><td>88.57</td></tr><tr><td>256</td><td>89.14</td><td>87.62</td><td>89.27</td><td>82.92</td><td>90.60</td></tr><tr><td>512</td><td>92.00</td><td>90.92</td><td>92.51</td><td>87.56</td><td>91.81</td></tr><tr><td>full</td><td>94.35</td><td>93.71</td><td>94.79</td><td>91.75</td><td>94.10</td></tr><tr><td>KC1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>84.64</td><td>73.93</td><td>81.90</td><td>71.66</td><td>77.35</td></tr><tr><td>32</td><td>83.41</td><td>73.74</td><td>81.52</td><td>64.27</td><td>75.92</td></tr><tr><td>64</td><td>82.37</td><td>73.27</td><td>81.99</td><td>66.45</td><td>74.60</td></tr><tr><td>128</td><td>83.79</td><td>73.46</td><td>81.42</td><td>68.44</td><td>74.79</td></tr><tr><td>256</td><td>83.89</td><td>75.26</td><td>80.38</td><td>67.49</td><td>76.68</td></tr><tr><td>512</td><td>83.60</td><td>77.73</td><td>82.94</td><td>70.62</td><td>77.25</td></tr><tr><td>full</td><td>86.82</td><td>80.57</td><td>85.59</td><td>75.26</td><td>80.38</td></tr><tr><td>MAGIC Gamma</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>68.69</td><td>65.95</td><td>67.14</td><td></td><td></td></tr><tr><td>32</td><td>69.91</td><td>73.14</td><td>73.21</td><td>69.01 71.48</td><td>69.87 73.15</td></tr><tr><td>64</td><td>74.41</td><td>76.66</td><td>76.60</td><td>75.15</td><td></td></tr><tr><td>128</td><td>76.56</td><td>77.88</td><td>77.15</td><td>77.06</td><td>75.81 76.69</td></tr><tr><td>256</td><td>78.35</td><td>78.23</td><td>78.39</td><td>78.44</td><td></td></tr><tr><td>512</td><td>79.32</td><td>79.12</td><td>79.40</td><td>79.54</td><td>77.55</td></tr><tr><td>full</td><td>84.54</td><td>84.22</td><td>84.71</td><td>84.64</td><td>78.58 82.64</td></tr><tr><td>Online Shoppers</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>84.51</td><td>75.26</td><td>81.75</td><td>73.53</td><td></td></tr><tr><td>32</td><td>85.81</td><td>76.03</td><td>82.79</td><td>74.06</td><td>81.46</td></tr><tr><td>64</td><td>86.12</td><td>75.13</td><td>82.68</td><td>72.02</td><td>80.54 78.64</td></tr><tr><td>128</td><td>87.01</td><td>77.92</td><td>83.83</td><td>74.53</td><td>79.16</td></tr><tr><td>256</td><td>87.09</td><td>78.90</td><td>84.66</td><td>74.97</td><td>78.80</td></tr><tr><td>512</td><td>87.32</td><td>79.11</td><td>85.04</td><td>76.35</td><td>79.53</td></tr><tr><td>full</td><td>87.98</td><td>82.32</td><td>86.08</td><td></td><td></td></tr><tr><td>PC3</td><td></td><td></td><td></td><td>78.02</td><td>78.98</td></tr><tr><td>16</td><td>89.94</td><td>72.10</td><td>79.75</td><td></td><td></td></tr><tr><td>32</td><td>89.94</td><td>77.96</td><td>81.27</td><td>72.48 79.36</td><td>72.61 77.71</td></tr><tr><td>64</td><td>89.55</td><td>76.82</td><td>81.15</td><td>78.34</td><td>77.83</td></tr><tr><td>128</td><td>89.43</td><td>77.71</td><td>82.04</td><td>78.98</td><td>79.24</td></tr><tr><td>256</td><td>89.68</td><td>81.78</td><td>85.73</td><td>81.78</td><td>82.42</td></tr><tr><td>512</td><td>88.79</td><td>82.42</td><td>86.50</td><td>82.17</td><td>84.08</td></tr><tr><td>full</td><td>89.30</td><td>85.35</td><td>88.54</td><td>85.35</td><td>86.24</td></tr><tr><td>PC4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>87.67</td><td>69.32</td><td>76.30</td><td>67.12</td><td>72.60</td></tr><tr><td>32</td><td>87.53</td><td>70.82</td><td>78.36</td><td>72.60</td><td>74.38</td></tr><tr><td>64</td><td>87.81</td><td>73.97</td><td>77.53</td><td>71.37</td><td>75.62</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>128</td><td>87.53</td><td>75.62</td><td>81.23</td><td>74.11</td><td>75.75</td></tr><tr><td>256</td><td>88.22</td><td>78.08 78.77</td><td>81.78 83.84</td><td>75.48</td><td>79.18</td></tr><tr><td>512 full</td><td>88.08 88.49</td><td>84.11</td><td>87.40</td><td>77.67 82.88</td><td>82.60 84.79</td></tr></table>

Table 11: Classification accuracy (%) of Distance-KNN and four LLMs by dataset and candidate-pool size (part 3 of 3, k = 8). Values average five seeds. Within each dataset and pool, bold and underlined values mark the highest and second-highest distinct displayed accuracies, respectively, including ties
<table><tr><td>Pool</td><td>Distance-KNN</td><td>Gemma3-12B</td><td>Gemma3-27B</td><td>Qwen3-8B</td><td>Qwen3-32B</td></tr><tr><td colspan="6">Phishing Websites</td></tr><tr><td>16</td><td>74.32</td><td>79.64</td><td>77.79</td><td>56.98</td><td>79.62</td></tr><tr><td>32</td><td>82.95</td><td>84.09</td><td>83.74</td><td>59.11</td><td>83.78</td></tr><tr><td>64</td><td>85.75</td><td>85.66</td><td>86.06</td><td>62.15</td><td>85.55</td></tr><tr><td>128</td><td>89.19</td><td>88.12</td><td>88.73</td><td>65.17</td><td>88.79</td></tr><tr><td>256</td><td>90.25</td><td>89.24</td><td>90.16</td><td>69.46</td><td>89.58</td></tr><tr><td>512</td><td>91.63</td><td>91.03</td><td>91.92</td><td>73.04</td><td>91.32</td></tr><tr><td>full</td><td>96.93</td><td>97.00</td><td>96.73</td><td>86.71</td><td>96.93</td></tr><tr><td colspan="6">QSAR Biodegradation</td></tr><tr><td>16</td><td>70.57</td><td>66.60</td><td>65.09</td><td>69.06</td><td>67.17</td></tr><tr><td>32</td><td>73.21</td><td>69.81</td><td>70.38</td><td>72.83</td><td>70.75</td></tr><tr><td>64</td><td>77.17</td><td>76.60</td><td>78.87</td><td>76.23</td><td>78.49</td></tr><tr><td>128</td><td>80.19</td><td>79.06</td><td>80.00</td><td>80.19</td><td>80.38</td></tr><tr><td>256</td><td>83.02</td><td>81.32</td><td>81.89</td><td>83.40</td><td>82.26</td></tr><tr><td>512</td><td>83.58</td><td>83.21</td><td>83.58</td><td>83.21</td><td>82.64</td></tr><tr><td>full</td><td>85.28</td><td>84.34</td><td>85.28</td><td>85.66</td><td>84.72</td></tr><tr><td colspan="2">Spambase</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>73.19</td><td>68.89</td><td>74.19</td><td>58.05</td><td>78.79</td></tr><tr><td>32</td><td>75.92</td><td>71.80</td><td>73.28</td><td>66.98</td><td>78.87</td></tr><tr><td>64</td><td>80.61</td><td>74.92</td><td>76.70</td><td>68.03</td><td>81.39</td></tr><tr><td>128</td><td>81.52</td><td>81.30</td><td>82.73</td><td>78.09</td><td>84.60</td></tr><tr><td>256</td><td>85.47</td><td>83.99</td><td>84.12</td><td>81.56</td><td>86.25</td></tr><tr><td>512</td><td>86.42</td><td>84.21</td><td>85.03</td><td>82.30</td><td>86.98</td></tr><tr><td>full</td><td>91.28</td><td>90.54</td><td>90.11</td><td>87.20</td><td>91.63</td></tr><tr><td colspan="2">Steel Plates Fault</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>44.62</td><td>49.54</td><td>46.97</td><td>48.21</td><td>49.44</td></tr><tr><td>32</td><td>49.85</td><td>51.28</td><td>52.72</td><td>49.33</td><td>51.18</td></tr><tr><td>64</td><td>52.82</td><td>52.72</td><td>54.05</td><td>52.92</td><td>54.46</td></tr><tr><td>128</td><td>57.74</td><td>58.77</td><td>59.28</td><td>56.10</td><td>59.90</td></tr><tr><td>256</td><td>61.74</td><td>62.87</td><td>63.49</td><td>61.64</td><td>64.62</td></tr><tr><td>512</td><td>67.49</td><td>69.23</td><td>67.08</td><td>66.05</td><td>68.72</td></tr><tr><td>full</td><td>75.18</td><td>73.85</td><td>75.38</td><td>74.46</td><td>75.59</td></tr><tr><td colspan="2">Student Dropout</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>54.99</td><td>62.75</td><td>69.39</td><td>53.14</td><td>46.64</td></tr><tr><td>32</td><td>60.41</td><td>62.26</td><td>68.85</td><td>54.27</td><td>48.31</td></tr><tr><td>64</td><td>63.25</td><td>63.84</td><td>69.39</td><td>58.24</td><td>50.38</td></tr><tr><td>128</td><td>65.19</td><td>64.51</td><td>70.20</td><td>61.40</td><td>54.45</td></tr><tr><td>256</td><td>65.33</td><td>64.97</td><td>69.80</td><td>62.03</td><td>56.12</td></tr><tr><td>512</td><td>66.73</td><td>66.14</td><td>70.47</td><td>64.70</td><td>57.79</td></tr><tr><td>full</td><td>70.74</td><td>70.56</td><td>73.18</td><td>68.89</td><td>64.15</td></tr><tr><td colspan="2">Vehicle</td><td></td><td></td><td></td><td></td></tr><tr><td>16</td><td>43.53</td><td>46.82</td><td>48.94</td><td>38.12</td><td>43.53</td></tr><tr><td>32</td><td>48.94</td><td>54.12</td><td>52.00</td><td>39.76</td><td>52.24</td></tr><tr><td>64</td><td>55.29</td><td>59.29</td><td>58.59</td><td>49.65</td><td>58.35</td></tr><tr><td>128</td><td>63.06</td><td>64.00</td><td>62.82</td><td>58.35</td><td>62.82</td></tr><tr><td>256</td><td>66.59</td><td>66.12</td><td>67.53</td><td>63.29</td><td>68.24</td></tr><tr><td>512</td><td>70.35</td><td>69.41</td><td>69.41</td><td>68.47</td><td>68.94</td></tr><tr><td>full</td><td>69.18</td><td>68.71</td><td>70.12</td><td>66.82</td><td>69.41</td></tr></table>