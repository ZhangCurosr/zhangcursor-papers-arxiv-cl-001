# ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification

Jianzong Wang<sup>1</sup>†, Chuhang Liu<sup>1,2</sup>†, Botao Zhao<sup>1</sup>, Zuheng Kang<sup>1</sup>, Xulong Zhang<sup>1</sup>, Xiaoyang Qu<sup>1</sup>, Junqing Peng<sup>1</sup>, Zhiewei Ye<sup>3</sup>, and Yayun He<sup>1(B)</sup>

1 Ping An Technology (Shenzhen) Co., Ltd., Shenzhen, China, 2 Tsinghua Shenzhen International Graduate School, Shenzhen, China, 3 Hubei University of Technology, Hubei, China heyayun0618@163.com

Abstract. Large language models (LLMs) have achieved strong performance across a broad range of classification settings, yet the reliability of their predictions remains a major obstacle to deployment in high-stakes scenarios. Although confidence estimation for LLMs has been widely studied, confidence calibration for LLM-based classification remains underexplored. We introduce ProbPlug, a lightweight confidence estimation framework for LLM-based binary classification, which predicts whether an output is correct using internal token features extracted from a frozen LLM. ProbPlug employs a self-attention module to aggregate hidden representations and can be integrated into the original inference pipeline without modifying the base model. Experiments across multiple tasks involving both text-based and multimodal large models show that Prob-Plug provides more reliable confidence estimates, improves classification performance with negligible additional overhead, and exhibits strong generalization across tasks. These results indicate that ProbPlug serves as a practical solution for confidence estimation in LLM-based classification. Our code is publicly available at this repository.

Keywords: Model Confidence · Multimodal Large Models · Text Classification · Speech Emotion Classification.

## 1 Introduction

Large language models (LLMs) have achieved promising results across both textual and multimodal classification tasks. [1], they can still produce misleading or incorrect predictions. Unlike conventional classifiers, which provide probability scores that can be used to adjust precision-recall trade-ofs, LLM-based methods often lack reliable confidence estimates. This limitation hinders their use in reliability-sensitive applications. Therefore, accurately estimating model confidence is crucial for practical deployment.

In recent years, researchers have explored various approaches to solve the confidence estimation problem for LLMs [2], including: (1) Verbalization: enabling LLMs to directly output confidence scores [3]; (2) Logit-based methods: leveraging the probability distributions of generated tokens [4], with recent work such as LogTokU [5] further analyzing token-level logit uncertainty for confidence estimation; (3) Self-consistency: taking the consistency of results from multiple repeated queries as the confidence metric [6], as exemplified by SelfCheckGPT [7]; (4) Trained probes: employing probabilities output by pre-trained classifiers as confidence estimates [8]; (5) Hybrid methods: integrating two or more of the aforementioned approaches (e.g., CISC [9]). Directly applying these existing methods to LLM-based classification tasks is undoubtedly a simple and feasible approach. However, since these methods are not specifically designed for this scenario, their performance fails to reach an optimal level. Accordingly, this work focuses on confidence estimation for LLM-based binary classification.

Inspired by findings in neuroscience [10], we note that during the human brain’s decision-making process, signals flow between diferent brain regions. For instance, when the human ear perceives a question, sensory organs first receive the information; the signal then propagates from lower-level brain regions to higher-order cortices, where comprehension occurs in Wernicke’s Area, analysis in the prefrontal cortex, answer generation in Broca’s Area, and finally output via the motor cortex. Notably, the resulting answer and its confidence are associated with each step of this process.

Drawing on this insight, we hypothesize that LLM decision-making is closely related to representations across Transformer layers. Accordingly, we propose ProbPlug, a confidence network that takes layer-wise outputs as input and predicts confidence scores. To improve generalization, we focus on binary classification with Yes”/No” outputs and use only newly generated tokens as input. Experiments show that ProbPlug provides reliable confidence estimation while maintaining strong task performance and cross-task generalization, combining the generalization strength of LLM-based methods with the calibration ability of non-LLM-based approaches.The main contributions of this work are summarized below:

– We investigate the problem of confidence estimation in LLM-based classification and highlight that unreliable confidence remains a major obstacle to deploying LLMs in risk-sensitive scenarios.

– We propose ProbPlug, a lightweight plug-and-play framework that derives confidence scores from hidden representations of a frozen LLM, enabling estimation of prediction correctness without altering the base model.

– Extensive experiments on multiple benchmarks show that ProbPlug provides better-calibrated confidence estimates, improves downstream classification performance with negligible additional inference overhead, and demonstrates promising cross-task generalization ability.

## 2 Problem Statement

As shown in Fig. 1, LLMs can perform classification tasks through prompting, where the model generates a discrete textual response such as “Yes” or “No”. However, such outputs do not naturally provide calibrated confidence scores, limiting their use in reliability-sensitive scenarios.

Given an input prompt $x _ { 1 : t - 1 }$ , the LLM generates the next token $y _ { t }$

$$
P ( y _ { t } | x _ { 1 : t - 1 } ) = \mathrm { L L M } ( x _ { 1 : t - 1 } ) ,\tag{1}
$$

For binary classification tasks, the generated token is constrained to the set {Yes, No}. While it indicates the model’s decision, it does not explicitly reflect decision confidence. In this work, we estimate the probability that the generated answer is correct by leveraging hidden representations from diferent transformer layers, which capture complementary information at diferent stages of model computation. Let $Z$ denote the hidden representations corresponding to the final generated token across all transformer layers:

$$
Z = \{ z ^ { ( l ) } \} _ { l = 0 } ^ { L }\tag{2}
$$

where $L$ denotes the total number of transformer layers. Our goal is to learn a lightweight confidence estimator $f _ { \mathrm { p r o b } } ( \cdot )$ such that

$$
P ( Y = 1 | Z ) = f _ { \mathrm { p r o b } } ( Z )\tag{3}
$$

where $Y = 1$ denotes that the generated answer is correct. Based on this formulation, we propose ProbPlug, a plug-in uncertainty network that estimates decision confidence from layer-wise hidden representations of a frozen LLM. Only a lightweight external network needs to be trained, making ProbPlug plug-andplay and computationally eficient.

## 3 Method

## 3.1 Model Architecture

The hidden representations across Transformer layers encode semantic, reasoning, and decision-related information, which may also reflect model uncertainty. Based on this insight, ProbPlug aggregates these layer-wise signals for confidence estimation. As shown in Fig. 2, the framework is composed of three modules: Token Compression (TC), Layer Information Aggregation (LIA), and a classification head. The backbone LLM remains frozen during training.

Hidden State Extraction Given an input prompt, the frozen LLM produces a binary output token. We extract the hidden representations associated with the final generated token from all transformer layers.

$$
Z = [ z ^ { ( 1 ) } , z ^ { ( 2 ) } , . . . , z ^ { ( L ) } ]\tag{4}
$$

![](images/7ec65855efb8b4a57db33f03aab5ac6acdcdd66c0a5a8de22e42914152b7c7a2.jpg)  
Fig. 1. Motivation for the proposed method. (a) Traditional LLM inference relies on next-token probabilities, which are often poorly calibrated and dificult to threshold in high-stakes settings. (b) ProbPlug estimates confidence from multi-layer hidden representations of a frozen LLM, enabling reliable and flexible threshold-based filtering.

Let Z denote the set of hidden states across layers, which has a dimension of $( N , S , L , D )$ , where N denotes batch size, S the number of tokens, L the number of distinct layers, and D the feature dimension. In our default configuration, we select only the final generated token $( S = 1 )$ , which directly reflects the models decision and empirically provides stronger cross-task generalization ability.

Token Compression Module The extracted hidden representations may contain redundant token-level information. To refine layer-wise features, we introduce a token-wise feature transformation module. For $S = 1$ , it enhances singletoken features via nonlinear mapping. Subsequently, the token representations are aggregated using two fully connected layers: it first maps the dimension S to 2S, and then compresses it to 1.

$$
\tilde { z } = W _ { 2 } \sigma ( W _ { 1 } z ^ { ( l ) } + b _ { 1 } ) + b _ { 2 }\tag{5}
$$

where $W _ { 1 } , W _ { 2 } , b _ { 1 }$ , and $b _ { 2 }$ denote the trainable parameters. σ denotes a nonlinear activation function. This module compresses token-level features into a single representation per layer, producing $\tilde { Z } \mathrm { : }$

$$
\tilde { Z } = [ z ^ { ( \tilde { 1 } ) } , z ^ { ( \tilde { 2 } ) } , . . . , z ^ { ( \tilde { L } ) } ] ,\tag{6}
$$

which summarizes the decision-related information from each transformer layer, has a dimension of $( N , 1 , L , D )$ .

Layer Information Aggregation Module Diferent transformer layers contribute unequally to the models decision confidence. To adaptively combine these

![](images/64edbf642472ae2776a9e0e367b7054b48b67f3cdeb6335a23dd31c27862c09e.jpg)  
Fig. 2. The overview of the proposed method. The left sub-figure illustrates the frozen large language model, while the right subfigure presents the architecture of our proposed ProbPlug.

representations, we leverage multi-head attention to adaptively fuse layer-wise representations. Let

$$
Q = W _ { Q } q , K = W _ { K } \tilde { Z } , V = W _ { V } \tilde { Z } ,\tag{7}
$$

where $q$ is a learnable query vector. The attention weights are computed as

$$
\alpha = \operatorname { s o f t m a x } ( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } )\tag{8}
$$

The aggregated representation is then obtained as

$$
z _ { \mathrm { a g g } } = \alpha V\tag{9}
$$

This formulation allows the model to adaptively weight information from diferent transformer layers during confidence estimation.

Confidence Prediction The aggregated representation $z _ { \mathrm { a g g } }$ is passed through a lightweight classifier to produce the confidence score:

$$
P = \sigma ( W _ { c } z _ { \mathrm { a g g } } + b _ { c } )\tag{10}
$$

Table 1. Performance comparison (Part 1) of our proposed ProbPlug across in-task and cross-task scenarios. Trained on the SMS-SPAM dataset [11], the model was evaluated on SST2 [12] and Toxic Comment Classification [13]. ∗ indicates the in-task dataset. Bold values denote the best performance.
<table><tr><td rowspan="2">Methods</td><td colspan="2">SMS-SPAM*</td><td colspan="2">SST2</td><td colspan="2">Toxic Comment Class</td></tr><tr><td>F1-score</td><td>AUPRC</td><td>F1-score</td><td>AUPRC</td><td>F1-score</td><td>AUPRC</td></tr><tr><td>Qwen3-8B [15]</td><td>88.49</td><td></td><td>88.75</td><td></td><td>81.07</td><td></td></tr><tr><td>Verbalization [3]</td><td>87.66</td><td>82.48</td><td>90.10</td><td>86.27</td><td>81.25</td><td>63.35</td></tr><tr><td>[Logit [4]</td><td>88.64</td><td>92.02</td><td>89.44</td><td>95.28</td><td>78.91</td><td>72.42</td></tr><tr><td>[Self-Consist [6]</td><td>88.44</td><td>70.13</td><td>89.55</td><td>87.18</td><td>81.66</td><td>48.63</td></tr><tr><td>CISC [9]</td><td>86.86</td><td>68.80</td><td>91.12</td><td>93.49</td><td>81.50</td><td>50.64</td></tr><tr><td>SAPLMA [8]</td><td>92.05</td><td>95.85</td><td>87.89</td><td>96.85</td><td>81.09</td><td>70.80</td></tr><tr><td>ProbPlug (Ours)</td><td>94.78</td><td>97.56</td><td>88.47</td><td>97.42</td><td>81.68</td><td>73.20</td></tr></table>

where $W _ { c }$ and $b _ { c }$ are learnable parameters. P represents the estimated probability that the LLMs generated answer is correct.

## 3.2 Cross-task Generalization and Multi-class Classification

A key advantage of this formulation is that it naturally supports cross-task generalization. Since many classification tasks can be reformulated as Yes/No decision problems, the trained confidence estimator can be directly applied to unseen tasks by modifying the prompt template.

For multi-class classification tasks, we decompose the task into multiple onvvs-rest binary decisions. Let K be the total number of classes, and $y _ { c }$ denote the confidence score associated with class c. The final class probability is computed via a softmax normalization:

$$
P ( \hat { y } _ { c } = c ) = \frac { e ^ { y _ { c } } } { \sum _ { j = 1 } ^ { K } e ^ { y _ { j } } }\tag{11}
$$

This formulation allows ProbPlug to be extended to multi-class and multimodal classification scenarios.

## 4 Experimental Results and Analysis

## 4.1 Experimental Configuration

Dataset To evaluate generalization, we conduct experiments on five public text datasets: SMS Spam Collection [11], SST-2 [12], Toxic Comment Classification, Civil Comments [13], and Amazon Polarity [14], covering spam detection, sentiment analysis, toxicity detection, and product review classification. We further use IEMOCAP [16], which contains 5,531 utterances from four emotion categories, to evaluate performance on multimodal large models.

Table 2. Performance comparison (Part 2) of our proposed ProbPlug across in-task and cross-task scenarios. The model was evaluated on Civil Comments [13] and Amazonpolarity [14].
<table><tr><td rowspan="2">Methods</td><td colspan="2">Civil comments</td><td colspan="2">Amazon-polarity</td></tr><tr><td>F1-score</td><td>AUPRC</td><td>F1-score</td><td>AUPRC</td></tr><tr><td>Qwen3-8B [15]</td><td>66.91</td><td></td><td>94.42</td><td></td></tr><tr><td>Verbalization [3]</td><td>66.65</td><td>51.82</td><td>88.96</td><td>85.54</td></tr><tr><td>Logit [4]</td><td>65.86</td><td>55.15</td><td>94.52</td><td>97.06</td></tr><tr><td>Self-Consist [6]</td><td>66.41</td><td>45.41</td><td>94.55</td><td>94.16</td></tr><tr><td>CISC [9]</td><td>67.09</td><td>45.51</td><td>94.10</td><td>93.70</td></tr><tr><td>[SAPLMA [8]</td><td>57.19</td><td>56.50</td><td>94.46</td><td>97.65</td></tr><tr><td>ProbPlug (Ours)</td><td>67.20</td><td>58.02</td><td>94.72</td><td>98.72</td></tr></table>

Baselines We adopt the Qwen3-8B model as the baseline and directly apply it to binary text classification across all datasets. We further explore variants where Qwen outputs either Verbalization [3] or Logit [4]. In addition, we compare advanced inference strategies, including Self-Consistency [6] and CISC [9]. Meanwhile, we evaluate SAPLMA [8], which attaches and trains a three-layer MLP classification head on top of the models internal representations. Model performance is quantified using F1-score and the area under the precision-recall curve (AUPRC), as is common in related literature [19].

## 4.2 Main Performance

As shown in Table 1 and 2, the ProbPlug outperforms several existing competitive confidence estimation methods on the same task in terms of both F1-score and AUPRC. Additionally, we evaluated our method on cross tasks without additional training by simply modifying prompts. The results demonstrate that our method outperforms current state-of-the-art approaches across most metrics. On the SST2 dataset, our method achieves a slightly lower F1-score than the CISC method. However, it exhibits significant advantages in other aspects: first, our method needs only one inference pass, with inherent eficiency advantages over CISC, which requires multiple repeated inferences. Furthermore, our method performs better in AUPRC, which implies that it could achieve higher performance by adjusting the classification threshold. Overall, the experiments demonstrate that our method achieves consistently improved performance on both in-task and cross-task datasets, with results averaged over multiple runs.

## 4.3 Multi-class Classification Performance

Furthermore, to evaluate the performance of our method on multi-class classification tasks and multimodal large models, we trained our ProbPlug model using Qwen2-audio as the base model. As shown in Table 3, our method exhibits significant advantages across all metrics compared to both existing non-LLMbased methods (e.g., Emotion2Vec) and other confidence estimation methods mentioned earlier. This indicates that our method can not only be generalized to multi-class classification tasks but also be applied to multimodal large models.

Table 3. Comparison of the performance on the speech emotion recognition (Iemocap from the EmoBox benchmark [18]) task with multimodal large model. UA means unweighted accuracy, and WA denotes the weighted accuracy. The performance of Non-LLM-based methods was cited from the corresponding papers.
<table><tr><td>Type</td><td>Methods</td><td>UA</td><td>WA</td><td>F1-score</td></tr><tr><td>Non-LLM-based</td><td>data2vec 2.0 large [20] [Whisper large v3 [21] Emotion2vec large [22]</td><td>57.30 73.54 70.70</td><td>56.23 72.86 63.30</td><td>56.70 73.11</td></tr><tr><td>LLM-based</td><td>Qwen2-audio [17] Verbalization [3] Logit [4] [Self-Consist [6]</td><td>64.33 66.45 66.42 67.24</td><td>60.37 63.24 64.35 66.21</td><td>/ 61.61 62.56 61.26 63.13</td></tr></table>

Table 4. Calibration performance of SAPLMA and ProbPlug.
<table><tr><td rowspan="2">Calibration</td><td colspan="2">ECE</td><td colspan="2">Brier Score</td></tr><tr><td>SAPLMA</td><td>ProbPlug</td><td>SAPLMA</td><td>ProbPlug</td></tr><tr><td>SMS-SPAM</td><td>0.0194</td><td>0.0126</td><td>0.0240</td><td>0.0210</td></tr><tr><td rowspan="3">SST2 Toxic Comment Class Civil comments</td><td>0.1123</td><td>0.1251</td><td>0.0987</td><td>0.0706</td></tr><tr><td>0.0695</td><td>0.0654</td><td>0.0740</td><td>0.0739</td></tr><tr><td>0.1884</td><td>0.1580</td><td>0.2183</td><td>0.2092</td></tr><tr><td>Amazon-polarity</td><td>0.1015</td><td>0.0412</td><td>0.0563</td><td>0.0374</td></tr><tr><td>Average</td><td>0.0982</td><td>0.0805</td><td>0.0943</td><td>0.0824</td></tr></table>

## 4.4 Calibration Experiments

To evaluate confidence reliability, we adopt Expected Calibration Error (ECE) and Brier Score as calibration metrics. ECE evaluates the consistency between predicted confidence and observed accuracy, whereas Brier Score quantifies the discrepancy between predicted probabilities and the corresponding ground-truth labels. As reported in Table 4, ProbPlug achieves lower average ECE (0.0805) and Brier Score (0.0824) than SAPLMA, indicating better calibration, with notable gains on SMS-SPAM and Amazon-polarity. Figure 3 (a) further shows that ProbPlug stays closer to the perfect calibration line across most bins, while SAPLMA exhibits clear overconfidence in high-confidence regions. These results demonstrate that ProbPlug provides more reliable confidence estimates.

## 4.5 Ablation Studies

We further analyze the contribution of each model component and the efect of the token selection hyperparameter (number of tokens, denoted as S in Section 3.1). Models with diferent configurations are trained on SMS-SPAM and evaluated on both SMS-SPAM and SST2. As shown in Table 5, both the token compression module and the layer-wise information aggregation module improve performance. Using 10 tokens slightly improves in-task performance compared with (S=1), but noticeably reduces cross-task generalization. This is likely because preceding tokens contain task-specific semantic information, causing the confidence estimator to overfit to the source domain. In contrast, the final token mainly captures decision confidence with less task-specific noise. Therefore, using only the final token (S=1) provides better robustness across tasks.

![](images/dfd9ba7ea8a50369a8efd5f5d7ae0f25a7067205c8d936c03bea62aa3302de7c.jpg)  
(a)

![](images/9e721c1f8193ed8643356eac566f718a8397e8f67706d998a45951cd7667572f.jpg)  
(b)  
Fig. 3. (a)Reliability diagrams of confidence calibration for SAPLMA and ProbPlug on Amazon-polarity. (b)Visualization of the learned attention weights α across the 36- layer base model. The distribution demonstrates the concentration of decision certainty within the terminal high-order reasoning layers (L30L35).

Table 5. Evaluation of in-task and cross-task performance under diferent ablation settings. TC means token compression module, while LIA means layer information aggregation module.
<table><tr><td></td><td rowspan=2 colspan=2>Configurations</td><td rowspan=1 colspan=2>In-task (SMS-SPAM)</td><td rowspan=1 colspan=2>Cross-task (SST2)</td></tr><tr><td></td><td rowspan=1 colspan=1>F1-score</td><td rowspan=1 colspan=1>AUPRC</td><td rowspan=1 colspan=1>F1-score</td><td rowspan=1 colspan=1>AUPRC</td></tr><tr><td></td><td rowspan=1 colspan=2>w/o TC</td><td rowspan=1 colspan=1>94.05</td><td rowspan=1 colspan=1>96.99</td><td rowspan=1 colspan=1>74.98</td><td rowspan=1 colspan=1>95.44</td></tr><tr><td rowspan=1 colspan=2>w/o LIA</td><td rowspan=3 colspan=2>10 tokenOurs</td><td rowspan=1 colspan=1>92.26</td><td rowspan=1 colspan=1>95.16</td><td rowspan=1 colspan=1>86.53</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>95.94</td><td rowspan=1 colspan=1>97.84</td><td rowspan=1 colspan=1>86.84</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>94.78</td><td rowspan=1 colspan=1>97.56</td><td rowspan=1 colspan=1>88.47</td><td></td></tr></table>

## 4.6 Layer-wise Attention Visualization and Interpretability

To investigate where decision uncertainty is encoded across the 36 Transformer layers, we visualize the learned attention weights α under diferent inference conditions. As shown in Figure 3 (b), ProbPlug exhibits a clear bimodal pattern in high-confidence cases, with attention concentrated on the early lexical layers (L0L5) and the final reasoning layers (L30L35), while intermediate layers receive little weight.This suggests that confident predictions mainly rely on basic lexical cues and high-level reasoning, whereas intermediate semantic representations contribute limited information and may introduce task-specific noise. By suppressing these less informative layers, ProbPlug maintains a focused attention distribution and improves confidence robustness. The results indicate that the learned layer-wise attention efectively reflects the evolution of decision uncertainty across the Transformer hierarchy.

## 5 Conclusion

In this work, we introduce ProbPlug, a plug-in framework that estimates prediction confidence for LLM-based binary classification from hidden token representations. By operating on internal representations of the frozen base model without modifying its original architecture, ProbPlug addresses the problem of uncalibrated confidence in LLM outputs. Extensive experiments on multiple classification tasks, using both standard large language models (e.g., Qwen3) and multimodal large models (e.g., Qwen2-Audio), show that ProbPlug achieves three main benefits. First, it provides more reliable confidence estimates, improving the trustworthiness of LLM predictions. Second, it brings consistent gains in downstream classification performance with only limited additional inference overhead. Third, it demonstrates promising cross-task generalization, reducing the need for extensive retraining when adapting to new tasks. Overall, the experimental results demonstrate that ProbPlug ofers an efective approach to confidence estimation for LLM-based classification. This study further highlights its potential to support more reliable deployment of LLMs in high-stakes scenarios that require trustworthy confidence assessment.

## References

1. Zhang, Y., Wang, M., Li, Q., Tiwari, P., Qin, J.: Pushing the limit of LLM capacity for text classification. In: Companion Proceedings of the ACM on Web Conference 2025, pp. 1524–1528 (2025)

2. Mahaut, M., Aina, L., Czarnowska, P., Hardalov, M., Müller, T., Màrquez, L.: Factual confidence of LLMs: on reliability and robustness of current estimators. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4554–4570 (2024)

3. Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., Manning, C.D.: Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5433–5442 (2023)

4. Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., Huang, X.-J.: Do large language models know what they dont know? In: Findings of the Association for Computational Linguistics: ACL 2023, pp. 8653–8665 (2023)

5. Ma, H., Chen, J., Zhou, J.T., Wang, G., Zhang, C.: Estimating LLM uncertainty with evidence. arXiv preprint arXiv:2502.00290 (2025)

6. Wang, X., Wei, J., Schuurmans, D., Le, Q.V., Chi, E.H., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. In: International Conference on Learning Representations (ICLR) (2023)

7. Manakul, O., Liusie, A., Gales, M.: SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9004–9017 (2023)

8. Azaria, A., Mitchell, T.: The internal state of an LLM knows when its lying. In: Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 967– 976 (2023)

9. Taubenfeld, A., Shefer, T., Ofek, E., Feder, A., Goldstein, A., Gekhman, Z., Yona, G.: Confidence improves self-consistency in LLMs. In: Findings of the Association for Computational Linguistics: ACL 2025, pp. 20090–20111 (2025)

10. Warren, J.: How does the brain process music? Clinical Medicine 8(1), 32–36 (2008)

11. Almeida, T.A., Hidalgo, J.M.G., Yamakami, A.: Contributions to the study of SMS spam filtering: new collection and results. In: Proceedings of the 11th ACM Symposium on Document Engineering, pp. 259–262 (2011)

12. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C.D., Ng, A.Y., Potts, C.: Recursive deep models for semantic compositionality over a sentiment treebank. In: Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642 (2013)

13. Borkan, D., Dixon, L., Sorensen, J., Thain, N., Vasserman, L.: Nuanced metrics for measuring unintended bias with real data for text classification. In: Companion Proceedings of the 2019 World Wide Web Conference, pp. 491–500 (2019)

14. Zhang, X., Zhao, J., LeCun, Y.: Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems 28 (2015)

15. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505 (2025)

16. Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J.N., Lee, S., Narayanan, S.S.: IEMOCAP: Interactive emotional dyadic motion capture database. Language Resources and Evaluation 42(4), 335–359 (2008)

17. Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., et al.: Qwen2-Audio technical report. arXiv preprint arXiv:2407.10759 (2024)

18. Ma, Z., Chen, M., Zhang, H., Zheng, Z., Chen, W., Li, X., Ye, J., Chen, X., Hain, T.: EmoBox: Multilingual multi-corpus speech emotion recognition toolkit and benchmark. In: Proc. Interspeech 2024, pp. 1580–1584 (2024)

19. Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al.: Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221 (2022)

20. Baevski, A., Babu, A., Hsu, W.-N., Auli, M.: Eficient self-supervised learning with contextualized target representations for vision, speech and language. In: International Conference on Machine Learning, pp. 1416–1429 (2023)

21. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision. In: International Conference on Machine Learning, pp. 28492–28518 (2023)

22. Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., Chen, X.: Emotion2Vec: Self-supervised pre-training for speech emotion representation. In: Findings of the Association for Computational Linguistics: ACL 2024, pp. 15747–15760 (2024)