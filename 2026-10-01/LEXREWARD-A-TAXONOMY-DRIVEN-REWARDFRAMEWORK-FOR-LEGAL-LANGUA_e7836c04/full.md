# LEXREWARD: A TAXONOMY-DRIVEN REWARDFRAMEWORK FOR LEGAL LANGUAGE MODELS

Yida Cai<sup>1,3∗</sup>, Xin Dai<sup>2∗</sup>, Bingxiang He<sup>3</sup>, Huiyuan Xie<sup>3†</sup>, Yuxiao Ye<sup>3</sup>, Zhenghao Liu<sup>2</sup>, Yang Bai<sup>1</sup>, Zhiyuan Liu<sup>3</sup>

<sup>1</sup>Peking University <sup>2</sup>Northeastern University <sup>3</sup>Tsinghua University caiyida26@stu.pku.edu.cn, 20216401@stu.neu.edu.cn, xieh@tsinghua.edu.cn

## ABSTRACT

Legal language models require reward signals that capture not only answer correctness but also the multidimensional quality of legal responses. Existing reward methods, however, often rely on coarse-grained holistic judgments, providing limited domain specificity and interpretability. We introduce LexReward, a taxonomy-driven framework for legal reward modeling. LexReward characterizes legal response quality along three complementary dimensions: Style, covering lexical and syntactic quality; Element, assessing legal subjects, facts, statutes, and decisions; and Chain, evaluating the order, completeness, correctness, and non-redundancy of legal reasoning. For each dimension, we develop rubrics that specify evaluation criteria and quality levels. The resulting rewards are used to construct pairwise preference data for Direct Preference Optimization (DPO) and reward-model training. Experiments show that the rubric-based rewards reliably distinguish legal responses of different quality and that DPO training on the preference data improves performance across all three dimensions. The learned reward models, LexRM, also support effective downstream optimization: each dimension-specific reward model improves policy performance in its corresponding dimension through reinforcement learning, without requiring reference answers at reward time. Dimension-wise analyses further support the effectiveness of the proposed taxonomy and reward construction.<sup>1</sup>

## 1 INTRODUCTION

Legal large language models have demonstrated growing potential across a wide range of legal generation tasks, such as legal question answering and legal reasoning (Fei et al., 2024; Xie et al., 2026; Yao et al., 2025; Li et al., 2024). Nevertheless, producing a high-quality legal response requires considerably more than arriving at the correct final answer. A legally sound response should identify the relevant subjects and facts, accurately invoke applicable statutes, construct a complete and co herent reasoning chain, and express its conclusions in precise and objective legal language. Models may reach a correct conclusion through incomplete reasoning, omit legally decisive facts, cite an inappropriate statutory basis, or produce fluent but legally unreliable explanations. These failures cannot be adequately captured by final-answer accuracy alone.

As reinforcement learning (RL) becomes increasingly important for improving model reasoning, reward design plays a central role in specifying which aspects of legal response quality models are encouraged to improve. Existing reward approaches in legal AI often rely on general-purpose reward models or reward functions that assess only selected aspects of legal response quality, such as judgment-outcome accuracy (Cai et al., 2025; Zhang et al., 2025). Although useful in some settings, these approaches may overlook other aspects of a response, making it difficult to determine whether improvements reflect better legal response quality or superficial features such as response length and fluency. More fundamentally, without a structured definition of legal response quality, it remains unclear what a legal reward should assess.

![](images/655f35114961c4c2e69181ec2e409e32e716a456cee98ff2e5a755a156e7425c.jpg)  
Figure 1: Overview of the LexReward framework. LexReward first constructs a legal taxonomy through expert knowledge, error induction, and conceptual refinement. It then operationalizes the taxonomy into dimension-specific rubric-based rewards, which construct preference data for training a reward model and guide reinforcement learning.

To clarify which aspects of legal response quality should be rewarded, as shown in Fig. 1, we introduce LexReward, a taxonomy-driven reward framework for legal language models. Drawing on legal experts’ domain knowledge, we first develop a taxonomy of legal response quality with three complementary dimensions: Style, which captures lexical and syntactic properties of legal writing; Element, which assesses the identification and treatment of legal subjects, facts, statutes, and decisions; and Chain, which evaluates the order, completeness, correctness, and non-redundancy of legal reasoning. Legal experts further refine the taxonomy through an empirical analysis comparing existing models’ responses with reference answers, yielding fine-grained criteria for each dimension. We design a scoring rubric for each fine-grained criterion and use either rule-based or LLM-as-ajudge evaluators, depending on the context. Experiments show that the resulting rewards reliably distinguish responses of different quality along their corresponding dimensions.

To provide dense reward signals and evaluate the practical utility of LexReward, we score diverse responses from a pool of models using rubric-based rewards and construct pairwise preference data both to evaluate their utility for Direct Preference Optimization (DPO), and to train LexRM, a fam ily of Chinese legal reward models. To the best of our knowledge, LexRM is the first collection of reward models developed for the Chinese legal context. We evaluate LexRM through Test-Time Scaling (TTS), using it to select the highest-scoring response from candidates generated by multiple models and comparing against random selection from the same candidate pool. We further assess its effectiveness for Group Relative Policy Optimization (GRPO). Our experiments show that DPO improves generation performance across all three dimensions, while LexRM-guided selection outperforms random selection. When used for GRPO, LexRM also yields gains across all dimensions and outperforms rule-based outcome rewards (Dai et al., 2026; Cai et al., 2025; Zhang et al., 2025).

Together, these results establish LexReward as a framework for defining and rewarding legal response quality, enabling legal domain knowledge to guide both reward construction and model optimization. Our main contributions are as follows:

• We introduce a structured taxonomy of legal response quality that decomposes the requirements of a high-quality legal response into three complementary dimensions (Style, Element, and Chain) and fine-grained criteria, providing a principled foundation for legal reward modeling.

![](images/d8d2c9184335f2626af6ed56a64426c5ba7618ac4643cf4536039a92d0758581.jpg)  
Figure 2: The LexReward taxonomy and its rubric-oriented operationalization. Legal response quality is decomposed into three complementary dimensions: Style (red), Element (orange), and Chain (green). Colored boxes show the dimensions and their constituent criteria, while gray boxes summarize the corresponding evaluation objectives. This structured decomposition supports the development of fine-grained rubrics for legal reward modeling.

• We develop interpretable, rubric-based rewards for each fine-grained criterion in the taxonomy using rule-based and LLM-based evaluation, and demonstrate that they reliably distinguish response quality along their respective dimensions.

• We construct preference data using the rubric-based rewards and demonstrate that DPO training on these data improves legal response generation.

• We train LexRM, to the best of our knowledge the first family of Chinese legal reward models, on rubric-derived preference data and demonstrate its effectiveness in Test-Time Scaling (TTS).

• We further integrate the learned reward models into a Group Relative Policy Optimization (GRPO) pipeline and demonstrate that their supervision improves the generation quality of the policy.

## 2 RELATED WORK

## 2.1 LEGAL LANGUAGE MODELS AND DOMAIN-SPECIFIC EVALUATION

Large language models (LLMs) have been increasingly adapted to the legal domain, with prior work improving their legal knowledge and reasoning abilities through domain-specific training and alignment methods (Chalkidis et al., 2020; Xu et al., 2025; Dai et al., 2026). Meanwhile, benchmarks such as LegalBench (Guha et al., 2023) and LawBench (Fei et al., 2024) provide standardized testbeds for evaluating legal knowledge, rule understanding, and reasoning ability across different legal tasks (Zhong et al., 2020). However, existing legal evaluations often emphasize task-level per formance, which does not fully capture the multidimensional quality of a legal response. A reliable legal response should also exhibit appropriate legal expression, sufficient coverage of relevant legal elements, faithful grounding in applicable statutes, and coherent reasoning. This motivates a structured reward framework that decomposes legal response quality into interpretable dimensions and translates them into training and optimization signals.

## 2.2 RUBRIC-BASED REWARD MODELING

Reward modeling learns to assess response quality for candidate selection and policy optimization. Early approaches learn scalar rewards from holistic human preferences (Ouyang et al., 2022), while subsequent work introduces finer-grained supervision over error types, text segments, and multiple quality objectives (Wu et al., 2023; Wang et al., 2024). Complementing this decomposition, rubricbased evaluation (Kim et al., 2024) makes judgment criteria and quality levels explicit, enabling evaluators to assess responses against specified standards. In the legal domain, existing work (Chen et al., 2026) learns process rewards for criminal-law knowledge-graph reasoning, but focuses on a specific task. Building on the broader shift from holistic preferences toward structured quality supervision, LexReward systematically defines reward dimensions and evaluation criteria through a legal quality taxonomy, then operationalizes it into rubrics and cross-task preference data, connecting domain-specific quality definitions with reward model training and reinforcement learning.

## 3 THE LEXREWARD TAXONOMY

Taxonomy construction. Legal response quality encompasses multiple requirements that a single holistic criterion leaves implicit. We therefore construct a taxonomy that decomposes legal response quality into explicit, assessable dimensions. With the assistance of legal experts, we combine top-down specification with bottom-up error analysis. In the top-down component, legal experts draw on their domain knowledge to identify the core requirements of a high-quality legal response (Goodrich, 1990; Osbeck, 2011; Maley, 2014). In the bottom-up component, they examine responses generated by multiple language models (Qwen Team, 2025; Llama Team, 2024) across legal AI benchmarks (Fei et al., 2024; Ma et al., 2026; Li et al., 2024), comparing them with reference answers to identify errors and unmet quality requirements. The requirements identified through both components are then integrated and refined into a unified taxonomy.

As illustrated in Fig. 2, the resulting taxonomy comprises three complementary dimensions: Style, Element, and Chain. These dimensions characterize how legal content is expressed, what legally relevant information is included, and how that information is connected to support a conclusion. Each dimension is further decomposed into finer-grained criteria for rubric-based assessment.

Style assesses the linguistic quality of legal responses through lexical features and syntactic features. Lexical features capture the precise use of legal terminology (word specificity) and the use of objective language without unwarranted subjective judgments (subjective word control). Syntactic features capture cohesion across sentences (sentence cohesion), clarity of sentence structure (sentence structure), and appropriate combinations of words in legal expressions (collocation correctness). Together, these criteria assess the precision, objectivity, and clarity of legal expressions.

Element assesses whether a response includes the legally relevant information needed to address a legal problem. It comprises four categories: subjects, facts, statutes, and decisions, covering the relevant entities, case circumstances, statutory provisions, and legal conclusions, respectively.

Chain assesses the reasoning that connects case information to a legal conclusion. It comprises four categories: order, completeness, correctness, and non-redundancy. These criteria assess whether the reasoning steps are logically arranged, sufficiently developed, legally valid, and free from un necessary repetition. Whereas Element assesses coverage of relevant information, Chain assesses the inferential connections among that information.

Together, the three dimensions organize legal response quality in terms of stylistic expression, substantive content, and reasoning. The taxonomy provides a structured basis for translating expertdefined quality requirements into explicit rubric criteria, which support fine-grained assessment and rubric-based reward modeling.

## 4 TAXONOMY-DRIVEN REWARDS

## 4.1 RUBRIC-BASED REWARD OPERATIONALIZATION

We operationalize the fine-grained criteria in our taxonomy using two complementary scoring strategies, selected according to whether evaluating a criterion requires the input context. For contextindependent criteria, the score can be computed from intrinsic properties of the generated response. We therefore use rule-based reward functions calibrated against a corpus of high-quality legal texts, providing deterministic and interpretable scores. For context-dependent criteria, evaluation requires determining whether the response appropriately addresses the legal problem specified by the input. We use an LLM-as-a-judge evaluator for these criteria, leveraging its ability to assess the relationship between the input and the generated response under a criterion-specific rubric.

For a context-independent criterion k, we define the reward as:

$$
r _ { k } ( y ) = { \mathrm { E v a l } } _ { k } ^ { \mathrm { r u l e } } \left( y ; \theta _ { k } , \mathcal { C } \right) ,\tag{1}
$$

where y denotes the generated response, $\theta _ { k }$ denotes the criterion-specific scoring parameters, and C is an optional corpus of high-quality legal texts used to calibrate the evaluator.

For a context-dependent criterion $k ,$ we define the reward as:

$$
r _ { k } ( x , y ) = \mathrm { E v a l } _ { k } ^ { \mathrm { L L M } } \left( x , y ; \rho _ { k } \right) ,\tag{2}
$$

where x denotes the input context, y denotes the generated response, and $\rho _ { k }$ specifies the scoring rubric supplied to the LLM judge.

We assign an evaluation strategy to each taxonomy criterion based on its dependence on the input context. Style primarily concerns intrinsic linguistic properties of the response and is therefore evaluated using a rule-based evaluator. Element requires assessing whether the response identifies and treats the legally relevant information in the input and is therefore evaluated using an LLM judge. The evaluation of Chain is task-dependent. When a task provides an explicit reasoning structure, we use rule-based functions to assess conformity to the prescribed steps and order. When the appropriate reasoning path must be inferred from the legal context, we instead use an LLM judge. We next describe the rubric and scoring procedure for each fine-grained criterion:

Style. The Style reward measures how closely a response conforms to the lexical and syntactic patterns of authentic Chinese judicial documents. We evaluate five attributes: $\kappa _ { \mathrm { { S t y l e } } } ~ =$ {ws, sw, coh, str, col}, corresponding to word specificity, subjective word control, sentence cohesion, sentence structure, and collocation correctness, respectively.

For each attribute $k ,$ we extract a feature representation $f _ { k } ( y )$ from response y and measure its discrepancy from the corresponding reference representation $f _ { k } ^ { \mathrm { r e f } }$ , estimated from a corpus of authentic judicial documents:

$$
d _ { k } ( y ) = D _ { k } \big ( f _ { k } ( y ) , f _ { k } ^ { \mathrm { r e f } } \big ) , \qquad k \in { \mathcal { K } } _ { \mathrm { S t y l e } } ,\tag{3}
$$

where $D _ { k }$ is a non-negative discrepancy measure appropriate to the feature type. Word specificity and subjective word control use KL divergence to compare legal-term and sentiment-word distributions, respectively. Sentence cohesion uses the absolute difference in conjunction frequency. Sentence structure and collocation correctness use Euclidean distance to compare sentence-length statistics and 3- to 6-gram coverage vectors, respectively.

To account for scale differences, we divide each discrepancy by its mean over calibration corpus C:

$$
s _ { k } = \frac { 1 } { \vert \mathcal { C } \vert } \sum _ { y ^ { \prime } \in \mathcal { C } } d _ { k } ( y ^ { \prime } ) , \qquad \widetilde { d } _ { k } ( y ) = \frac { d _ { k } ( y ) } { s _ { k } + \epsilon } ,\tag{4}
$$

where $\epsilon > 0$ ensures stability. The Style reward is the negated mean normalized discrepancy:

$$
R _ { \mathrm { S t y l e } } ( y ) = - \frac { 1 } { | \mathcal { K } _ { \mathrm { S t y l e } } | } \sum _ { k \in \mathcal { K } _ { \mathrm { S t y l e } } } \widetilde { d } _ { k } ( y ) .\tag{5}
$$

Higher rewards indicate greater similarity with the reference writing style across the five attributes.   
Feature extraction, reference estimation, and calibration procedures are detailed in Appendix A.1.

Element. The Element rubric evaluates whether a legal response matches the task-required legal components. We instantiate this rubric with an LLM-as-a-Judge evaluator, which scores the candidate response along four sub-dimensions: subjects, facts, statutes, and decisions. Subjects capture legal actors and their roles;facts capture case-relevant factual information; statutes capture statutory grounds; and decisions capture the final legal conclusion. Given a query and a candidate response, the judge assigns each sub-dimension a score from 0 to 1 without access to the reference answer, and the Element reward is computed as the average of the four scores. The judge is instructed to assess whether the candidate answer matches the query requirements in each Element sub-dimension, considering element accuracy, coverage, specificity, and unsupported or fabricated content. The full prompt is provided in Appendix A.2.

Chain. The Chain rubric evaluates the structure of legal reasoning along four sub-dimensions: order, completeness, correctness, and non-redundancy. Order assesses the logical sequence of reasoning steps; completeness measures coverage of required steps; correctness checks whether each step serves an appropriate reasoning function; and non-redundancy assesses unnecessary repetition or conflicting content across repeated steps. When a task specifies predefined reasoning steps, a rule-based evaluator scores these dimensions against the prescribed step types, required groups, and ordering constraints. Otherwise, an LLM-as-a-Judge evaluates the response along the same dimensions based on the query requirements. The dimension-level scores are aggregated into the final Chain reward. Detailed scoring rules and the judge prompt are provided in Appendix A.3.

## 4.2 REWARD MODEL TRAINING FROM RUBRIC-GUIDED PREFERENCES

Rubric-based rewards can be sparse, often assigning identical scores to responses of differing quality. Such ties limit their ability to distinguish between candidate responses and can reduce their effective ness in downstream applications. We therefore construct preference data from rubric-derived scores and train LexRM, a family of Chinese legal reward models, to learn legal preference functions that generalize beyond the discrete distinctions captured by the rubrics.

Dimension-specific reward models. To diversify response sources and reduce reliance on stylistic cues specific to a single generator, we maintain a pool of n models that generate candidate responses for each query x. For each quality dimension d, we score these candidates using the corresponding rubric and construct preference pairs within the same query. To capture quality differences across the score range, we organize the pairs into three categories: (i) the highest-scoring response versus the nearest lower-scoring response in the high-score range; (ii) a high-scoring response versus a medium-scoring response; and (iii) a medium-scoring response versus a substantially lower-scoring response. These categories provide supervision for both fine-grained distinctions among strong responses and broader differences across quality levels.

The resulting preference dataset $\mathcal { P } _ { d }$ contains tuples $( x , y ^ { + } , y ^ { - } )$ , where $y ^ { + }$ receives a higher rubric score than $y ^ { - }$ . We train a separate reward model $R _ { \theta _ { d } }$ per dimension using the Bradley–Terry objective (Bradley & Terry, 1952):

$$
\mathcal { L } _ { \mathrm { B T } } ^ { ( d ) } = - \mathbb { E } _ { ( x , y ^ { + } , y ^ { - } ) \sim \mathcal { P } _ { d } } \left[ \log \sigma \big ( R _ { \theta _ { d } } ( x , y ^ { + } ) - R _ { \theta _ { d } } ( x , y ^ { - } ) \big ) \right] ,\tag{6}
$$

where σ denotes the sigmoid function. This objective encourages the model to assign higher scalar scores to preferred responses. The resulting dimension-specific reward models support separate assessment of each taxonomy dimension, enabling analysis of where legal response quality improves or remains deficient.

Multi-dimensional reward models. The dimension-specific reward models, one per dimension $d \in \mathcal { D } =$ {element, style, chain}, share one pretrained backbone $\theta _ { 0 }$ and differ only in the preference datasets used for fine-tuning. For each dimension, we represent the parameter update as a task vector $\tau _ { d } = \theta _ { d } - \theta _ { 0 }$ . Across the backbone’s weight matrices, which carry virtually all of its parameters, the pairwise cosine similarities between these task vectors are small $( | \cos | \le 0 . 0 3$ on average, never above 0.14), with the little overlap that exists confined to the value and output projections of the topmost layers. These low pairwise cosine similarities motivate exploring additive composition of the dimension-specific updates. We form a single multi-dimensional reward model using task arithmetic (Ilharco et al., 2023):

$$
\begin{array} { r } { \theta _ { \mathrm { m e r g e d } } = \theta _ { 0 } + \sum _ { d \in \mathcal { D } } \lambda _ { d } \tau _ { d } , \qquad \lambda _ { d } = 1 , } \end{array}\tag{7}
$$

where $\lambda _ { d }$ scales the update contributed by dimension d. The merged model is a single reward model that scores all dimensions, removing the need to run three backbones at inference time.

Table 1: Performance of rubric-based rewards and reward models on datasets evaluating three dimensions. Bold denotes the best result in each column.
<table><tr><td colspan="2">Style: CLASE</td><td colspan="8">Element: Legal∆</td></tr><tr><td>Method</td><td>Accuracy (%)</td><td colspan="2">Method</td><td>SPP-F</td><td>CCP</td><td>SLP</td><td>CAS</td><td>CAC</td><td>Avg.</td></tr><tr><td>Random</td><td>50.00</td><td colspan="2">Five-model average</td><td>53.37</td><td>46.34</td><td>28.46</td><td>35.88</td><td>84.12</td><td>49.63</td></tr><tr><td>Rubric</td><td>79.00</td><td colspan="2">Rubric</td><td>79.04</td><td>57.03</td><td>33.10</td><td>39.40</td><td>92.80</td><td>60.27</td></tr><tr><td>Skywork-Qwen</td><td>78.50</td><td colspan="2">Skywork-Qwen</td><td>75.97</td><td>57.84</td><td>49.80</td><td>64.20</td><td>92.40</td><td>68.04</td></tr><tr><td>Skywork-Llama</td><td>52.50</td><td colspan="2">Skywork-Llama</td><td>61.15</td><td>54.35</td><td>29.90</td><td>51.80</td><td>91.00</td><td>57.64</td></tr><tr><td>Lawformer</td><td>68.50</td><td colspan="2">Lawformer</td><td>50.95</td><td>44.92</td><td>39.00</td><td>30.40</td><td>88.60</td><td>50.77</td></tr><tr><td>Legal-BERT</td><td>41.00</td><td colspan="2">Legal-BERT</td><td>31.19</td><td>39.27</td><td>20.10</td><td>29.00</td><td>74.80</td><td>38.87</td></tr><tr><td>LexRM-Style</td><td>81.75</td><td colspan="2">LexRM-Element</td><td>77.72</td><td>57.80</td><td>44.70</td><td>59.20</td><td>92.20</td><td>66.32</td></tr><tr><td>LexRM-Merge</td><td>69.75</td><td colspan="2">LexRM-Merge</td><td>75.57</td><td>58.27</td><td>43.80</td><td>58.80</td><td>92.40</td><td>65.77</td></tr><tr><td colspan="10">Chain: LexChain</td></tr><tr><td>Method</td><td>Plaintiff</td><td>Defendant</td><td>Dispute</td><td>Statute</td><td>Liability</td><td>Damages</td><td></td><td>Judgment</td><td>Overall</td></tr><tr><td>Five-model average</td><td>96.30</td><td>87.47</td><td>18.01</td><td>21.68</td><td>22.82</td><td>23.07</td><td></td><td>24.24</td><td>45.41</td></tr><tr><td>Rubric</td><td>96.00</td><td>87.45</td><td>21.20</td><td>27.25</td><td>25.75</td><td>25.85</td><td></td><td>27.82</td><td>47.80</td></tr><tr><td>Skywork-Qwen</td><td>96.75</td><td>88.60</td><td>27.00</td><td>29.95</td><td>27.95</td><td>28.35</td><td></td><td>32.06</td><td>50.19</td></tr><tr><td>Skywork-Llama</td><td>96.55</td><td>87.75</td><td>26.10</td><td>29.80</td><td>28.20</td><td>28.25</td><td></td><td>30.73</td><td>49.83</td></tr><tr><td>Lawformer</td><td>96.20</td><td>88.15</td><td>18.20</td><td>23.65</td><td>23.05</td><td>23.15</td><td></td><td>24.53</td><td>45.93</td></tr><tr><td>Legal-BERT</td><td>95.85</td><td>85.80</td><td>13.50</td><td>16.95</td><td>19.85</td><td>21.10</td><td></td><td>19.84</td><td>42.70</td></tr><tr><td>LexRM-Chain</td><td>96.80</td><td>89.15</td><td>27.40</td><td>30.65</td><td>28.75</td><td>28.00</td><td></td><td>31.46</td><td>50.46</td></tr><tr><td>LexRM-Merge</td><td>96.25</td><td>88.00</td><td>19.30</td><td>23.45</td><td>24.15</td><td>26.15</td><td></td><td>26.81</td><td>46.84</td></tr></table>

## 5 EXPERIMENTS AND RESULTS

We evaluate taxonomy-driven rewards at three levels. First, we assess the taxonomy-derived rubricbased rewards on pairwise selection tasks and MoE-based test-time scaling (TTS), where the reward selects the highest-scoring response from five candidates generated by different models, with random selection from the same pool as the baseline. Second, we use rubric-derived preference data to train reward models and evaluate them under the same MoE-based TTS setting. Third, we assess whether this supervision improves generation quality through DPO on the preference data and reinforcement learning from an SFT checkpoint using the learned reward models.

## 5.1 SETTINGS

## 5.1.1 DATASETS

For Style, we use CLASE (Ma et al., 2026), with 4,000 training instances and an official test set of 1,000 instances. Reward scorers are evaluated by pairwise accuracy in selecting the gold response over a model-generated negative; policies are evaluated by the CLASE-Mix score. For Element, we draw data from the criminal questions of JEC-QA (Zhong et al., 2020) and the civil judgments of LexChain (Xie et al., 2026), using 2,244 preference pairs built from 2,736 instances for training. Evaluation follows the in-domain protocol of Legal∆ (Dai et al., 2026) on its 3,000-instance test set, which reports F1 for statutory-article and charge prediction and accuracy for sentence-length prediction, case analysis, and financial calculation; the overall score is the unweighted mean of the five values. For Chain, we use LexChain (Xie et al., 2026), with 9,550 training instances and an official test set of 1,000 instances, and adopt its native LLM-based evaluation, which scores seven aspects of a judgment and reports an overall score. Dataset statistics and the construction of preference data are detailed in Appendix B.1.

Table 2: Downstream policy performance across the three LexReward dimensions. SPP-F and CCP report entity-level F1; SLP, CAS and CAC report exact-match accuracy; Avg. is the mean of these five Element metrics. Bold denotes the best result in each column, including ties.
<table><tr><td rowspan="2">Method</td><td>Style: CLASE</td><td colspan="7">Element: Legal∆</td></tr><tr><td>CLASE-Mix (/10)</td><td>SPP-F</td><td>CCP</td><td></td><td>SLP</td><td>CAS</td><td>CAC</td><td>Avg.</td></tr><tr><td>Qwen3-8B</td><td></td><td>3.08</td><td>76.80</td><td>51.89</td><td>45.00</td><td>35.00</td><td>80.20</td><td>57.78</td></tr><tr><td>SFT</td><td></td><td>7.32</td><td>75.45</td><td>49.85</td><td>46.80</td><td>49.20</td><td>81.80</td><td>60.62</td></tr><tr><td>DPO</td><td></td><td>5.21</td><td>75.67</td><td>52.99</td><td>44.20</td><td>36.20</td><td>81.60</td><td>58.13</td></tr><tr><td>GRPO (rule)</td><td></td><td>8.43</td><td>80.45</td><td>48.67</td><td>47.60</td><td>51.80</td><td>75.80</td><td>60.86</td></tr><tr><td>GRPO (RM)</td><td></td><td>8.54</td><td>78.64</td><td>54.77</td><td>56.70</td><td>53.40</td><td>89.80</td><td>66.66</td></tr><tr><td colspan="9">Chain: LexChain</td></tr><tr><td>Method</td><td>Plaintiff</td><td>Defendant</td><td>Dispute</td><td>Statute</td><td>Liability</td><td>Damages</td><td>Judgment</td><td>Overall</td></tr><tr><td>Qwen3-8B</td><td>97.25</td><td>88.90</td><td>34.20</td><td>42.10</td><td>34.30</td><td>28.45</td><td>26.46</td><td>53.55</td></tr><tr><td>SFT</td><td>97.95</td><td>90.85</td><td>39.00</td><td>39.40</td><td>35.05</td><td>35.15</td><td>36.06</td><td>55.99</td></tr><tr><td>DPO</td><td>97.50</td><td>89.05</td><td>36.20</td><td>42.75</td><td>35.40</td><td>30.10</td><td>27.79</td><td>54.47</td></tr><tr><td>GRPO (rule)</td><td>97.40</td><td>89.80</td><td>36.50</td><td>37.90</td><td>35.55</td><td>34.00</td><td>37.64</td><td>55.29</td></tr><tr><td>GRPO (RM)</td><td>98.25</td><td>90.80</td><td>37.30</td><td>39.55</td><td>35.20</td><td>36.70</td><td>38.49</td><td>56.40</td></tr></table>

## 5.1.2 MODELS

We use DeepSeek-V4-Flash (DeepSeek-AI, 2026) for all LLM-based components of the rubric evaluators and Qwen3-8B (Qwen Team, 2025) as the backbone for all trained models. To generate candidate responses for preference data construction, we use a pool of five models: Qwen3-4B (Qwen Team, 2025), Qwen3.5-4B, Qwen3.5-9B (Qwen Team, 2026), Llama-3.1-8B-Instruct (Llama Team, 2024), and gemma-4-12B-it (Gemma Team, 2026). To evaluate reward model performance, we compare LexRM against Skywork-Reward-V2-Llama-3.1-8B, Skywork-Reward-V2-Qwen3-8B (Liu et al., 2025), as well as Lawformer (Xiao et al., 2021) and LegalBERT (Chalkidis et al., 2020). Training and inference configurations are provided in Appendix B.2.

## 5.2 RESULTS

## 5.2.1 RUBRIC AND RM EVALUATION

In this section, we evaluate the rubric-based rewards, together with LexRM trained on the preference data we construct, on selecting among candidate responses from multiple models.

As shown in Table 1, on all three datasets the rubric-based rewards (Rubric) select better responses than the five-model average or random selection, and the reward models trained on the preference pairs achieve further improvements: despite using substantially less training data than the Skywork reward models, LexRM-Style and LexRM-Chain achieve the best overall results on their respective datasets, while LexRM-Element ranks second. These results suggest that the reward models successfully internalize the rubric criteria as continuous scoring functions, enabling them to distinguish candidates that the rubrics may score equally. LexRM-Merge combines the three dimensionspecific reward models into one. On Legal∆, it matches the element expert and obtains the highest charge-prediction score, while on CLASE and LexChain it falls below the respective expert models.

## 5.2.2 DPO AND RL EVALUATION

In this section, we evaluate how rewards derived from the LexReward taxonomy transfer to downstream policies by examining performance after DPO and comparing RL with LexRM against RL with an outcome reward.

As shown in Table 2, DPO improves over the vanilla model across all three dimensions, indicating that the rubric-derived preference data provide useful reward information for policy optimization.

Table 3: Ablation study of the Style, Element, and Chain rubric dimensions. Bold denotes the best result in each column.
<table><tr><td colspan="2">Style: CLASE</td><td colspan="7">Element: Legal∆</td></tr><tr><td>Rubric</td><td>Accuracy (%)</td><td>Rubric</td><td>SPP-F</td><td>CCP</td><td>SLP</td><td>CAS</td><td>CAC</td><td>Avg.</td></tr><tr><td rowspan="4">Lexical Syntactic Overall</td><td>76.50</td><td>Subjects</td><td>65.70</td><td>48.76</td><td>7.60</td><td>32.80</td><td>93.00</td><td>49.57</td></tr><tr><td>77.50</td><td>Facts</td><td>68.13</td><td>50.73</td><td>11.20</td><td>33.60</td><td>93.00</td><td>51.33</td></tr><tr><td>79.00</td><td>Statutes</td><td>79.52</td><td>57.64</td><td>30.60</td><td>39.20</td><td>92.80</td><td>59.95</td></tr><tr><td></td><td>Disposition Overall</td><td>79.54 79.04</td><td>56.76 57.03</td><td>17.50 33.10</td><td>31.80 39.40</td><td>93.00 92.80</td><td>55.72 60.27</td></tr><tr><td rowspan="2"></td><td colspan="8"></td></tr><tr><td colspan="8">Chain: LexChain</td></tr><tr><td>Rubric</td><td>Plaintiff</td><td>Defendant</td><td>Dispute</td><td>Statute</td><td>Liability</td><td>Damages</td><td>Judgment</td><td>Overall</td></tr><tr><td>Order</td><td>95.70</td><td>87.30</td><td>19.80</td><td>24.65</td><td>23.65</td><td>23.35</td><td>25.94</td><td>46.25</td></tr><tr><td>Completeness</td><td>96.00</td><td>87.20</td><td>21.80</td><td>27.95</td><td>24.60</td><td>24.25</td><td>27.33</td><td>47.43</td></tr><tr><td>Correctness</td><td>95.95</td><td>87.15</td><td>20.00</td><td>25.85</td><td>23.95</td><td>24.40</td><td>26.60</td><td>46.77</td></tr><tr><td>Non-redundancy</td><td>96.05</td><td>88.15</td><td>18.10</td><td>22.10</td><td>24.55</td><td>25.30</td><td>26.79</td><td>46.43</td></tr><tr><td>Overall</td><td>96.00</td><td>87.45</td><td>21.20</td><td>27.25</td><td>25.75</td><td>25.85</td><td>27.82</td><td>47.80</td></tr></table>

Comparing the vanilla model, SFT, and GRPO initialized from SFT further demonstrates the benefits of reinforcement learning: GRPO (RM) achieves the highest score on every dimension, with the largest gain on Element, where it improves all five metrics and raises their average by 6.04 points over SFT. These results show that LexRM provides effective supervision for improving legal generation beyond supervised fine-tuning.

Legal RL is currently driven almost entirely by outcome rewards defined on a verifiable final answer, such as a predicted charge, statute, or monetary amount (Dai et al., 2026; Cai et al., 2025; Zhang et al., 2025). Following this practice, we use outcome-based rewards for Element and Chain, and ROUGE against the reference text for Style. As shown in Table 2, for Style, GRPO (rule) improves performance, but the gain is smaller than that achieved by GRPO (RM), suggesting that lexical overlap provides less effective supervision than the learned reward. GRPO (rule) leaves the Element and Chain averages largely unchanged: its gains are confined to the final answer, whereas the dimensions that ground that answer, such as Legal Basis, stagnate or decline. An outcome reward is indifferent to whether the conclusion it credits was reached through a complete and correctly grounded chain. However, rubric-derived rewards are subject to neither restriction. They are equally applicable to open-ended legal scenarios and do not depend on reference labels. This shows that the benefit comes from making the reward dimensional: what the taxonomy supplies is supervision on the reasoning that leads to an answer, not a better estimate of the answer itself.

## 5.2.3 DIMENSION ABLATION ANALYSIS

In this section, we examine whether the sub-dimensions within each taxonomy dimension carry complementary signal by deriving a reward from each sub-dimension in isolation and comparing it against their aggregation.

As shown in Table 3, aggregation gives the best overall score in all three dimensions of the taxonomy. On individual tasks it is occasionally beaten by a single criterion, but always by a small margin, whereas no single criterion consistently performs best across tasks. Within Element, for instance, a rubric using subjects alone never checks which provision is cited, and statute prediction drops well below the aggregate, which is restored by adding statutes; within Chain, the aggregate improves over every single criterion, with the gain concentrated in the later stages of the chain, liability, loss and judgment, whose conclusions must be reasoned to rather than read off the case description, while the earlier fact-identification stages are already saturated. This shows that the value of the taxonomy is not that one criterion suffices, but that scoring along several at once keeps a task-irrelevant criterion from deciding the outcome.

## 6 CONCLUSION

We introduce LexReward, a taxonomy-driven framework that characterizes legal response quality along three complementary dimensions: Style, Element, and Chain. LexReward operationalizes these dimensions as fine-grained scoring rubrics that effectively distinguish responses of different quality. The rubric-derived preference data improve legal response generation through DPO and support the training of LexRM, a family of Chinese legal reward models. LexRM improves multimodel response selection and provides effective reinforcement-learning rewards for separately optimizing policies in Style, Element, and Chain. These results establish taxonomy-driven rewards as effective supervision for improving legal language models.

## AI USE STATEMENT

In this work, we used generative AI tools for language refinement, partial code implementation, the generation of preference data, and assistance with interpreting results. We did not use these tools to develop theoretical models or conceptual frameworks, formulate mathematical claims, or provide essential components of mathematical proofs; the remaining disclosure categories are not applicable to this work. The authors reviewed and validated all AI-assisted outputs. Specifically, we checked AI-refined text for accuracy, tested AI-generated code and data for correctness, and independently verified AI-assisted interpretations of the results. The authors take full responsibility for the fina content of this work, including all text, claims, code, and other artifacts produced with the assistance of generative AI.

## REPRODUCIBILITY STATEMENT

The construction of the legal quality taxonomy is described in Section 3, while Section 4 specifies the rubric operationalization, preference data construction, and reward model training objectives. Section 5 presents the datasets, models, and evaluation protocols used in our experiments. Appendix A provides implementation details for the rubric-based rewards, while Appendix B documents dataset statistics, preference data construction, and training/inference/evaluation configurations.

## REFERENCES

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 1952.

Hua Cai, Shuang Zhao, Liang Zhang, Xuli Shen, Qing Xu, Weilin Shen, Zihao Wen, and Tianke Ban. Unilaw-r1: A large language model for legal reasoning with reinforcement learning and iterative inference. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 18117–18131, 2025.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. LEGAL-BERT: The muppets straight out of law school. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2898–2904, 2020.

Jiujiu Chen, Yazheng Liu, Sihong Xie, and Hui Xiong. SCPRM: A schema-aware cumulative process reward model for knowledge graph question answering. arXiv preprint arXiv:2605.02819, 2026.

China Judgments Online. China Judgments Online, 2013. URL https://wenshu.court. gov.cn.

Xin Dai, Buqiang Xu, Zhenghao Liu, Yukun Yan, Huiyuan Xie, Xiaoyuan Yi, Shuo Wang, and Ge Yu. Legal∆: Enhancing legal reasoning in LLMs via reinforcement learning with chainof-thought guided information gain. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 16912–16916, 2026.

DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence, 2026. URL https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/ main/DeepSeek\_V4.pdf.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. LawBench: Benchmarking legal knowledge of large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7933–7962, 2024.

Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

Peter Goodrich. Legal discourse: Studies in linguistics, rhetoric and legal analysis. Springer, 1990.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Re, Adam Chilton, Aditya Narayana, Alex´ Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzen berger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems, pp. 44123–44279, 2023.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2023.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing finegrained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, pp. 29927–29962, 2024.

Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, and Yiqun Liu. Lexeval: A comprehensive chinese legal benchmark for evaluating large language models. arXiv preprint arXiv:2409.20288, 2024.

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. Skywork-reward-v2: Scaling preference data curation via human-ai synergy. arXiv preprint arXiv:2507.01352, 2025.

Llama Team. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Yiran Rex Ma, Yuxiao Ye, and Huiyuan Xie. Clase: A hybrid method for chinese legalese stylistic evaluation. In Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026), pp. 642–653, 2026.

Yon Maley. The language of the law. In Language and the Law, pp. 11–50. Routledge, 2014.

OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Mark K Osbeck. What is” good legal writing” and why does it matter? Drexel L. Rev., 2011.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, pp. 27730–27744, 2022.

Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Qiang Dong, Maosong Sun, and Zhendong Dong. Openhownet: An open sememe-based lexical knowledge base. arXiv preprint arXiv:1901.09957, 2019.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Jian Sun. jieba, 2012. URL https://github.com/fxsjy/jieba.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. arXiv preprint arXiv:2406.12845, 2024.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. Advances in Neural Information Processing Systems, pp. 59008–59033, 2023.

Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun. Lawformer: A pre-trained language model for chinese legal long documents. arXiv preprint arXiv:2105.03887, 2021.

Huiyuan Xie, Chenyang Li, Huining Zhu, Chubin Zhang, Yuxiao Ye, Zhenghao Liu, and Zhiyuan Liu. Lexchain: Modeling legal reasoning chains for chinese tort case analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 35913–35921, 2026.

Buqiang Xu, Xin Dai, Zhenghao Liu, Huiyuan Xie, Xiaoyuan Yi, Shuo Wang, Yukun Yan, Liner Yang, Yu Gu, and Ge Yu. LegalDuet: Learning fine-grained representations for legal judgment prediction via a dual-view contrastive learning. In International Conference on Advanced Data Mining and Applications, pp. 337–352, 2025.

Rujing Yao, Yang Wu, Chenghao Wang, Jingwei Xiong, Fang Wang, and Xiaozhong Liu. Elevating legal LLM responses: Harnessing trainable logical structures and semantic knowledge with legal reasoning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5630–5642, 2025.

Kepu Zhang, Guofu Xie, Weijie Yu, Mingyue Xu, Xu Tang, Yaxin Li, and Jun Xu. Legal mathematical reasoning with LLMs: Procedural alignment through two-stage reinforcement learning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pp. 1586–1598, 2025.

Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. JEC-QA: A legal-domain question answering dataset. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 9701–9708, 2020.

## A IMPLEMENTATION DETAILS OF RUBRIC-BASED REWARDS

## A.1 STYLE

Reference and Calibration Corpora. The lexical distributions and linguistic reference statistics are constructed from 1,000 authentic Chinese judgments in the CJO ms corpus (China Judgments Online, 2013). The normalization scales are estimated separately from the LY fields of 986 CJO ms documents. This separation ensures that each raw metric is normalized to a comparable numerical range.

Word Specificity. Legal terms are extracted using a domain-specific terminology lexicon and forward maximum matching (FMM). The maximum and minimum matching lengths are eight and two Chinese characters, respectively. For each term t in the legal vocabulary $V _ { \mathrm { l e g a l } }$ , the response distribution is calculated as

$$
Q _ { y } ^ { \mathrm { l e g a l } } ( t ) = \frac { N _ { y } ( t ) } { \sum _ { u \in V _ { \mathrm { l e g a l } } } N _ { y } ( u ) } ,\tag{8}
$$

where $N _ { y } ( t )$ is the number of occurrences of term t in y. Terms absent from the response are assigned $\bar { \epsilon } = 1 0 ^ { - 1 0 }$ before normalization to avoid undefined KL-divergence values. The complete divergence is

$$
d _ { \mathrm { w s } } ( y ) = \sum _ { t \in V _ { \mathrm { l e g a l } } } P ^ { \mathrm { l e g a l } } ( t ) \log \frac { P ^ { \mathrm { l e g a l } } ( t ) } { Q _ { y } ^ { \mathrm { l e g a l } } ( t ) } .\tag{9}
$$

<table><tr><td>Dimension</td><td>Raw deviation</td><td>Scale  $s _ { k }$ </td></tr><tr><td>WS</td><td>KL divergence</td><td>12.307</td></tr><tr><td>SW</td><td>KL divergence</td><td>12.696</td></tr><tr><td>COH</td><td>Conjunction-rate deviation</td><td>0.012</td></tr><tr><td>STR</td><td>Sentence-statistic distance</td><td>26.057</td></tr><tr><td>COL</td><td>n-gram coverage distance</td><td>0.282</td></tr></table>

Table 4: Normalization scales for the Style reward. WS: Word Specificity; SW: Subjective-Word Control; COH: Sentence Cohesion; STR: Sentence Structure; COL: Collocation Correctness.

Here, $P ^ { \mathrm { l e g a l } } ( t )$ is the normalized frequency of t in the reference corpus. The normalization scale for this dimension is $s _ { \mathrm { w s } } = 1 2 . 3 0 7 1$

Subjective Word Control. Subjective expressions are identified using the Chinese HowNet sentiment lexicon(Qi et al., 2019) and the same FMM procedure. The reference and response distributions, P<sup>subj</sup> and $Q _ { y } ^ { \mathrm { s u b j } }$ , are constructed in the same manner as the legal-term distributions. The resulting KL divergence is normalized using $s _ { \mathrm { s w } } = 1 2 . 6 9 5 7$

Sentence Cohesion. The response is tokenized and POS-tagged using jieba.posseg (Sun, 2012). Tokens with the POS tag c are treated as conjunctions. The reference conjunction rate is

$$
\rho _ { \mathrm { c o n j } } ^ { * } = 0 . 0 1 9 1 .\tag{10}
$$

Here, $\rho _ { \mathrm { c o n j } } ^ { * }$ is the mean conjunction-token proportion estimated from the reference documents. The absolute deviation from this value is normalized using $s _ { \mathrm { c o h } } = 0 . 0 1 1 5 7 3$

Pronoun frequency is not included because preliminary experiments showed that it provided little discrimination between authentic and generated responses.

Sentence Structure. Sentences are segmented using Chinese punctuation delimiters and newline characters. Sentence length is measured in Chinese characters. The reference statistics are

$$
\mu ^ { * } = 6 4 . 7 , \qquad \sigma ^ { * } = 4 5 . 3 ,\tag{11}
$$

where $\mu ^ { * }$ and $\sigma ^ { * }$ are the reference mean and standard deviation of sentence lengths. The Euclidean distance between the response and reference statistics is normalized using $s _ { \mathrm { s t r } } = 2 6 . 0 5 6 7$

Collocation Correctness. Both the reference documents and generated responses are tokenized using jieba, with punctuation tokens removed before n-gram extraction. A reference n-gram is retained only if it occurs in at least ten reference documents.

For each $n \in \{ 3 , 4 , 5 , 6 \}$ , let $G _ { n } ( y )$ denote the set of n-grams extracted from response y, and let $G _ { n } ^ { * }$ denote the retained reference set. The coverage statistic is

$$
c _ { n } ( y ) = { \frac { | G _ { n } ( y ) \cap G _ { n } ^ { * } | } { | G _ { n } ( y ) | } } , \qquad n \in \{ 3 , 4 , 5 , 6 \} .\tag{12}
$$

Here, $\left| G _ { n } ( y ) \cap G _ { n } ^ { * } \right|$ is the number of response n-grams found in the reference set, while $\left| G _ { n } ( y ) \right|$ | is the total number of response n-grams.

The collocation deviation is calculated as

$$
d _ { \mathrm { c o l } } ( y ) = \sqrt { \sum _ { n = 3 } ^ { 6 } \left( c _ { n } ( y ) - \hat { c } _ { n } ^ { * } \right) ^ { 2 } } ,\tag{13}
$$

where $\bar { c } _ { n } ^ { * }$ is the mean reference coverage for order n. We use 3-grams through 6-grams in the final metric. Bigrams are excluded because their high frequency and short length result in many generic or non-legal combinations, reducing their ability to characterize professional legal collocations.

Normalization Parameters. The complete set of normalization scales is summarized below:

For every dimension, the normalized reward is $r _ { k } ( y ) = - d _ { k } ( y ) / s _ { k }$ . The five normalized rewards are then combined using an equal-weight arithmetic mean.

## A.2 ELEMENT

Fig. 3 presents the prompt template used by the LLM-based Element evaluator. The prompt provides the task context, candidate response, and element-specific rubric criteria, and requires the judge to produce a structured assessment of the relevant legal elements. For readability, the prompt shown in the figure has been translated into English, and the original prompt used in all experiments was written in Chinese.

## A.3 CHAIN

The Chain reward uses rule-based scoring when task-specific scoring logic is available. We evaluate four criteria: Order, which detects violations of required step precedence; Completeness, which measures coverage of necessary reasoning steps; Correctness, which checks compliance with steplevel reasoning requirements; and Non-redundancy, which identifies repeated or conflicting steps. Detected omissions or violations are mapped to discrete criterion-level rewards, which are averaged to obtain the final Chain reward.

When task-specific scoring logic is unavailable, we use an LLM judge to score the response along the same four criteria using the prompt in Fig. 4. The criterion-level scores are averaged to obtain the final reward. The displayed prompt is translated into English; the original experimental prompt was written in Chinese.

## B EXPERIMENTAL SETTINGS

## B.1 DATASETS

For Style, we use the official CLASE test set of 1,000 response pairs and construct preference data from all 4,000 training queries using our five-model generation procedure. We reserve 2,000 training instances as the reference corpus for computing the objective component of CLASE-Mix and split the remaining 2,000 queries equally between SFT and GRPO. No model is trained on authentic judicial texts: SFT targets and preferred responses in preference pairs are model-generated candidates selected by their rubric scores, while GRPO uses the learned reward model to score online generations.

For Element, we use all 2,736 JEC-QA and LexChain instances for preference construction, yielding 2,244 preference pairs, and designate separate subsets for SFT and GRPO. Evaluation uses the 3,000 instances in the Legal∆ in-domain test set, comprising 500 instances each for statutory-article prediction, charge prediction, case analysis, and financial calculation, and 1,000 for sentence-length prediction.

For Chain, we allocate 1,000 of the 9,550 LexChain training instances to SFT, 1,000 to GRPO, and 4,000 to pairwise preference training, and evaluate on the official test set of 1,000 instances.

Across all three dimensions, we construct preference data by sampling one response per query from each model in the pool, scoring the five candidates with the corresponding rubric, and forming pairs from responses with different scores. Queries for which all candidates receive identical scores are discarded.

![](images/fd43c0b361efe0efba626e96dc259b00abeed2a2ba9f4a5ec2e50b602935546f.jpg)  
Figure 3: English translation of the rubric-guided prompt used for Element evaluation. The original experimental prompt was written in Chinese, while its evaluation criteria, scoring procedure, and output structure are preserved in the translation.

## B.2 HYPER-PARAMETERS

Inference. For LLM-based rubric evaluation we use the DeepSeek-V4-Flash(DeepSeek-AI, 2026) API with the default reasoning effort and a maximum input length of 8,192 tokens; all inputs fit within this limit, requiring no truncation. Candidate responses are sampled with temperature 0.8 and top-p 0.95, one response per model.

Evaluation. CLASE-Mix combines objective and subjective scores with equal weights. The objective component compares textual features of generated responses with those of authentic judicial texts, while the subjective component uses GPT-4o-mini to score responses with the benchmarkprovided prompt. For LexChain, we use GPT-4o-2024-05-13 (OpenAI, 2024) to score each evaluation dimension following the benchmark protocol. For Legal∆, we report F1 scores for statutoryarticle and charge prediction, and accuracy for case analysis, financial calculation, and sentencelength prediction. We follow the evaluation procedures specified in the respective benchmark papers; further details can be found therein.

![](images/0cb67447eaea115fb7e63984af821104ccdf5e183daf8feb2bfc51b64f0b793a.jpg)  
Figure 4: English translation of the reasoning-step extraction prompt used for Chain evaluation when predefined reasoning steps are unavailable. The LLM converts free-form reasoning into an ordered sequence of labeled steps, which is subsequently scored by rule-based rubrics. The original experimental prompt was written in Chinese.

Training. Reward models are trained with the Bradley–Terry objective (Bradley & Terry, 1952) for two epochs using DeepSpeed ZeRO-3, a learning rate of $\mathrm { i } \times \mathrm { i } 0 ^ { - 7 }$ and a batch size of 32; DPO uses the same configuration. We train policies separately for Style, Element, and Chain. For each dimension, we first perform SFT on 1,000 queries paired with their highest-scoring candidate responses under the corresponding rubric, then apply GRPO to another 1,000 queries. Both GRPO variants start from the same dimension-specific SFT checkpoint: GRPO (RM) uses the corresponding dimension-specific LexRM as its reward function, whereas GRPO (rule) uses the baseline reward described in Section 5.2.2. Each dimension block in Table 2 reports results from the respective policies. DPO is initialized directly from Qwen3-8B. SFT, DPO and GRPO all use low-rank adaptation (LoRA).

## C LIMITATIONS

Our taxonomy is grounded in Chinese legal materials input from the Chinese legal context, and our evaluation is therefore limited to Chinese-language datasets. Its applicability to other languages and legal systems remains untested. Future work will extend the framework to these settings and examine how the taxonomy and its reward criteria should be adapted.

This work primarily studies dimension-specific rewards. Combining multiple dimensions into a unified rubric reward or reward model remains an open challenge, including the choice of aggregation weights, the composition of preference data, and the training strategy. Developing and systematically evaluating such integration methods is a central direction for future work.