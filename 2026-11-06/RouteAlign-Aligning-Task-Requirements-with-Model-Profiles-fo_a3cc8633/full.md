# RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection

Jihao Zhao<sup>∗,1</sup>, Ding Chen<sup>∗,1</sup>, Yanfang Chen<sup>2</sup>, Bo Tang<sup>1</sup>, Feiyu Xiong<sup>1</sup>, Junchi Yan<sup>3</sup>, Zhiyu Li<sup>B,1</sup>

<sup>1</sup>MemTensor (Shanghai) Technology Co., Ltd., Shanghai, China, <sup>2</sup>Renmin University of China, Beijing, China, <sup>3</sup>Shanghai Jiao Tong University, Shanghai, China

## Abstract

LLM service selection requires performance evidence that reflects the demands of the current task. Aggregate scores obscure differences in capability and difficulty, while semantically related examples can involve different combinations of skills. We present RouteAlign, which uses task requirements to structure both the measurement of model performance and the selection of services. Task requirements describe capability demand and difficulty, enabling alignment with corresponding model performance measurements. Historical requirements determine how outcomes contribute to global profiles and local capability estimates; incoming requirements determine how those measurements inform the current decision. A shared scorer integrates the evidence, and a generation-cost preference controls selection among eligible services. Evaluation across diverse tasks and heterogeneous LLM services shows that RouteAlign extends the empirical qualitycost frontier toward higher task quality. Comparisons with open Jev-style decision backends show consistent quality gains, with lower generation cost than their strongest tested configuration. Ablations support the contribution of local performance evidence, while catalog-update experiments demonstrate effective reuse of the fitted scorer with refreshed model measurements.

Date: November 6, 2026   
Correspondence: lizy@memtensor.cn   
Author Legend: \* Co-equal primary author, B Corresponding author   
Code: https://github.com/MemTensor/RouteAlign   
Keywords: Large language models, service selection, task requirements, model profiles, quality of service.

## 1 Introduction

Applications increasingly draw on large language models (LLMs) as services for tasks that span knowledge, reasoning, programming, and interaction with external systems. The breadth of foundation-model applications [1] makes the choice of service consequential: the same candidate may be suitable for routine requests yet unreliable on a demanding combination of skills. Service-oriented computing provides the architectural basis for invoking interchangeable functionality [2], and classical service selection compares eligible alternatives under quality-of-service (QoS) constraints [3]. For LLMs, however, task quality must be estimated for the particular request before the selected service executes it. An interface establishes how a model can be invoked; effective selection additionally requires evidence of what that model can accomplish under the current task demands.

![](images/672e1f3c1a2ff4b1bc863b78105d29a89e05c641e982e47807ba287c0a8b485f.jpg)

![](images/4807679eddb1ce25923caace566d91f56eef20069df50430c9287b1ea634453f.jpg)

![](images/2614f5ca8be31908e9ffecbc40311eb9fa882e47cec5579c1b44b14103bed30a.jpg)  
Figure 1 Conceptual comparison of query-model matching and requirement-profile alignment for LLM service selection. The task combines coding, reasoning, and data analysis with weights of 0.50, 0.33, and 0.17 at difficulty L3. The model profiles of services A, B, and C express performance in the same capability coordinates.

The central challenge is determining which past outcomes support that estimate. Holistic evaluation reveals substantial variation across scenarios and quality dimensions [4], and the Benchmark Lottery shows that relative model rankings can change with the choice of evaluation tasks [5]. Dynabench further demonstrates the value of targeted challenge examples for exposing model failures [6]. Even within a topic, requests can require different combinations of capabilities. Consider identifying incorrect statistics in a Python script and repairing its implementation. Superficially similar programming examples may test routine syntax, algorithm design, or quantitative analysis, each providing different evidence about the requested repair. Instruction-following constraints introduce another source of variation, as illustrated by IFEval [7]; reasoning demands can also change the interpretation of measured performance, as shown on BIG-Bench Hard [8]. Figure 1 illustrates the resulting selection problem. A useful model description must preserve the conditions under which performance was observed, and a selector must consult that description according to the capabilities and difficulty of the incoming task.

Existing routing results establish the value of exploiting model complementarity [9]. RouteLLM learns pairwise preferences [10], RouterDC learns query-model compatibility [11], and nearest-neighbor routing estimates quality from similar tasks [12]. These advances motivate a practical question: which parts of a model’s performance history are informative for a request that combines several capabilities? Global measurements provide coverage across topics, whereas related tasks preserve local context. Combining them requires distinguishing semantic resemblance from the capability demands associated with an observed outcome. This issue persists when the catalog changes. Dependencies among models, data, and system configuration complicate maintenance [13]; empirical studies also identify data versioning and model reuse as distinct software-engineering challenges [14]. A selector should therefore accommodate updated candidate evidence while preserving a consistent rule for comparing services.

We introduce RouteAlign to retain the task conditions that make past performance informative for a new request. In the programming example, results on statistical reasoning and routine code editing inform different aspects of the requested repair, even when the task descriptions are semantically close. The method preserves these distinctions through profiles of performance under different capability demands and difficulty levels, together with estimates from related tasks. Within that neighborhood, each outcome’s influence depends on semantic proximity and the capability demands of both the historical and current tasks. The resulting estimate reflects how a candidate’s measured competence matches the request’s capability mixture. A shared scorer learns to combine broad performance coverage with this local context. A common measurement schema allows the learned scoring rule to be reused when candidate evidence is refreshed after catalog changes.

Across ten datasets and 13 candidate LLMs, RouteAlign achieves the highest Sample-Avg and Dataset-Avg among the evaluated routing baselines, exceeding RouteLLM by 1.38 and 2.22 percentage points at 1.05% higher generation cost. It also exceeds the strongest tested open Jev-style configuration by 5.22 and 6.44 points on the respective metrics at 9.7% lower generation cost. The quality gains persist across decision backends and scoring mixtures, supporting the combination of requirement-conditioned measurements and local performance evidence. Ablations identify local evidence as the most consequential feature group. Timing studies quantify cached selection and online requirement prediction. In controlled catalog updates, refreshed measurements let the retained scorer respond to changes in available services and prices.

The paper makes three contributions:

• A requirement-aware formulation of LLM service selection that separates task competence, service eligibility, and generation-cost preference. Shared capability and difficulty coordinates make the relationship between task demands and measured model performance explicit.

• A requirement-profile alignment mechanism that combines difficulty-conditioned model profiles with local estimates conditioned on the capabilities required by both historical and incoming tasks. A shared scorer integrates these complementary measurements across candidate services.

• An empirical assessment spanning established routers, open Jev-style decision backends, and direct candidate invocation. Quality, generation cost, ablations, execution overhead, and catalog updates jointly examine the effectiveness and reuse of the selection policy.

## 2 Related Work

## 2.1 Service Selection and Task-Dependent Quality

Service selection traditionally compares functionally suitable candidates through QoS attributes and application preferences. Zeng et al. [3] formulate utility-based selection under user constraints and examine task-level and global allocation in composite services. Their retrospective [15] identifies stochastic outputs and unstructured data as challenges for composition involving LLMs. For an LLM invocation, task-dependent competence is itself a quantity to be estimated before candidates can be compared. RouteAlign addresses this estimation problem within an available service catalog: requirements condition the performance evidence, metadata determines admissibility, and generation cost enters the selection preference. This formulation connects service descriptions to invocation-time selection

## 2.2 LLM Routing and Resource Allocation

Pre-generation routing estimates which model should answer a request from information available before candidate execution. Hybrid LLM [16] predicts quality differences, RouteLLM [10] learns from pairwise preferences, and RouterDC [11] uses contrastive query-model objectives. Reward-based supervision provides another source of such evidence: ZOOTER [17] distills rewards into a routing policy, while Routoo [18] couples predicted performance with cost-aware selection. RouteAlign also makes its decision before response generation, with a focus on how task requirements structure and weight the measurements supplied to that decision.

Other approaches extend the allocation decision across executions or adapt it from feedback. FrugalGPT [19] learns model cascades, and AutoMix [20] uses answer verification to decide when to invoke a larger model.

MixLLM [21] formulates allocation as a contextual bandit, while BEST-Route [22] jointly selects a model and the number of sampled responses. These settings introduce sequential or resource-allocation decisions beyond choosing a single service for the current invocation.

Evaluation resources separate policy design from the collection of model outcomes. RouterBench [23] supplies reusable inference records, RouterEval [9] examines model complementarity, and LLMRouterBench [24] supports comparison under a common routing protocol. LLMRouter [25] provides tools for implementing and deploying routing policies. Following this emphasis on comparable evidence, our evaluation uses a common held-out workload and reports task quality alongside recorded generation charges.

## 2.3 Task Requirements and Model Performance Evidence

Model documentation gives performance measurements an explicit context. Model Cards [26] record intended uses, evaluation conditions, and disaggregated results; Datasheets for Datasets [27] document data composition, collection, and intended use. Holistic evaluation [4] compares language models across scenarios and metrics, while CheckList [28] tests specific capabilities that aggregate accuracy can overlook. Benchmark design further specifies the demands under evaluation: FLORES-200 examines multilingual translation [29], and WritingBench evaluates diverse writing requirements [30]. WebArena measures interactive task completion [31], while HarmBench evaluates harmful behavior and robust refusal [32]. These studies motivate preserving the task conditions underlying model measurements. RouteAlign gives those conditions a computational role by using capability demand and difficulty to organize and consult empirical evidence.

Routing methods operationalize model descriptions in several ways. GraphRouter [33] models relationships among tasks, queries, and candidates. UniRoute [34] derives model features from predictions on representative prompts, whereas ICL-Router [35] conditions routing on in-context representations of query and model performance. AvengersPro [36] summarizes quality and efficiency within query clusters, and nearest-neighbor routing [12] estimates quality from similar tasks. These approaches motivate examining both the structure of candidate representations and the locality of supporting evidence.

Task structure further specifies what that evidence measures. IRT-Router [37] relates query difficulty to model ability, and InferenceDynamics [38] constructs capability and knowledge profiles. SkillOrchestra [39] derives skill-specific competence from execution experience for agent orchestration. DecoR [40] decomposes query requirements to guide matching against historical records. RouteAlign brings these forms of evidence into a common requirement representation. Model profiles retain capability and difficulty conditions, and local estimates adjust the influence of semantically related outcomes according to task demand. The shared scorer compares candidates through features with consistent meanings across the service catalog.

## 3 Requirement-Aware LLM Service Selection

## 3.1 Service Catalog and Selection Scope

Consider an application with access to a catalog M<sub>t</sub> of LLM services at deployment version t. Each candidate m represents configured access to a particular model version. Its model determines task competence, while its service configuration specifies execution attributes such as context capacity, supported modalities, tool access, and availability. A request x, together with its supplied context, requires one eligible candidate to be selected before response generation. The decision selects one invocation target from the available catalog.

The empirical study represents each service candidate by its model configuration and recorded responses. We use routing to refer to existing methods and to implementations of this selection decision.

## 3.2 Task Requirements and Model Profiles

Task requirements specify both the competence needed for successful completion and the conditions under which a service may be used:

$$
R ( x ) = { \big ( } r ( x ) , d ( x ) , u ( x ) { \big ) } ,\tag{1}
$$

where $r ( x ) \in \mathbb { R } ^ { K }$ contains nonnegative capability weights with $\begin{array} { r } { \sum _ { k } r _ { k } ( x ) = 1 , d ( x ) \in \{ 1 , \dots , L \} } \end{array}$ denotes difficulty, and u(x) contains execution requirements. The capability mixture describes the relative demands of the task, and difficulty specifies the level at which those capabilities are needed. Execution requirements are obtained from request metadata and deployment policy.

A profile $P _ { m , t }$ associates candidate m with capability measurements across difficulty levels and with its service attributes. Comparing these attributes with $u ( x )$ defines the eligible set $\mathcal { F } _ { t } ( x ) \subseteq \mathcal { M } _ { t }$ . For example, a candidate that lacks explicitly required tool access is excluded regardless of its estimated competence. Capability weights then govern quality estimation within the eligible set. Eligibility checks enforce known metadata requirements; missing attributes do not independently exclude a candidate.

## 3.3 Selection Objective

At a fixed deployment version, let $q ( x , m ) \in [ 0 , 1 ]$ denote expected task quality and $c ( x , m ) \geq 0$ expected generation cost. For a nonempty eligible set, the selection objective is

$$
\pi _ { t } ^ { * } ( x ) \in \arg \operatorname* { m a x } _ { m \in { \mathcal F } _ { t } ( x ) } \big [ q ( x , m ) - \lambda c ( x , m ) \big ] ,\tag{2}
$$

where $\lambda \geq 0$ specifies the application preference for cost relative to quality. This penalty expresses a soft preference, while eligibility determines which candidates may enter the comparison. The surrounding application policy handles requests for which no service is eligible.

Quality and cost must be estimated before the selected candidate executes the request. Historical evidence provides tasks $x _ { i } .$ , recorded responses $y _ { i m } ,$ , task scores $s _ { i m } \in [ 0 , 1 ]$ , generation costs $c _ { i m } \geq 0$ , and available metadata. The challenge is to use these observations according to their relevance to the incoming requirements, rather than treating each measurement as equally informative for every task.

A model profile must distinguish performance under different task demands and support comparison across candidates. Shared requirement coordinates condition measurements on capability demand and difficulty; a shared scorer maps the resulting evidence to a common quality scale. The catalog determines which candidates are available for comparison, while the requirement representation remains independent of their identities. This distinction provides the basis for reusing task descriptions as the candidate set changes.

## 4 RouteAlign: Requirement-Profile Alignment

## 4.1 Task Requirement Profiling

RouteAlign uses a shared requirement representation throughout evidence preparation and invocation-time selection, as shown in Figure 2. Offline preparation annotates historical tasks, constructs model profiles, indexes reference tasks, and fits a shared scorer. For an incoming request, the selector predicts requirements, checks eligibility, retrieves related records, and compares the eligible candidates. Each service enters this comparison through measurements of its underlying model.

A compact profiler $h _ { \phi }$ predicts capability weights and difficulty from the request text. Its ten capability coordinates cover task understanding, domain knowledge, reasoning, software engineering, complex information synthesis, data analysis, multilingual understanding, content generation, interactive execution, and risk control. Four difficulty levels range from direct facts and simple transformations to expert analysis and extended planning. A task can assign positive weight to several capabilities, preserving the mixture needed for completion.

We formulate profiling as instruction-conditioned generation of structured annotations, building on instruction tuning as a means of learning from task descriptions [41]. Qwen3-0.6B [42] receives full-parameter supervised fine-tuning (SFT) in LlamaFactory [43], using approximately 21,000 GPT-5.4-annotated task profiles. The source pool covers the ten capabilities, including program synthesis in HumanEval [44], numerical reasoning over financial data in FinQA [45], and document reasoning in LongBench v2 [46]. Train-test overlap can distort model evaluation [47]. We therefore first exclude every source dataset present in the routing test workload, then remove residual overlap using token 3-grams: a candidate is discarded when its overlap covers at least 10% of a compared test sample’s 3-grams. Training minimizes loss on the target profile only. The same deterministic profiler annotates historical and incoming tasks; execution requirements are supplied separately through metadata. The Supplementary Material gives the source inventory, filtering rule, and training settings.

![](images/4e526fe174b5b23e8094c805a37117fd3b8c34406e0350a13cec891fb170abce.jpg)  
Figure 2 RouteAlign for LLM service selection. Offline preparation uses training records to construct model profiles, a historical evidence bank, and a shared scorer. The same profiler supplies requirements for outcome profiles and incoming tasks. Response assessments use separate annotations in the common schema. At invocation time, requirements align with global profiles and local evidence, metadata determines eligibility, and the cost preference shapes selection.

The profiler runs once per incoming request, and all candidates reuse its output. Because this output describes task demand, it can be applied to any candidate with suitable performance evidence. Candidate additions do not change the capability coordinates or difficulty levels. The same task description therefore supports both candidate measurement and selection across the catalog.

## 4.2 Requirement-Conditioned Model Profiles

A global profile summarizes performance under specified capability demands and difficulty levels. For model m, capability k, and difficulty ℓ, let ${ \mathcal { T } } _ { m , k , \ell }$ index historical tasks with an observed score, profiler-assigned difficulty $d ( x _ { i } ) = \ell ,$ and positive capability weight $r _ { k } ( x _ { i } )$ . The outcome profile is

$$
A _ { m , k , \ell } ^ { \mathrm { o u t } } = \frac { \sum _ { i \in \mathbb { Z } _ { m , k , \ell } } r _ { k } ( x _ { i } ) s _ { i m } } { \sum _ { i \in \mathbb { Z } _ { m , k , \ell } } r _ { k } ( x _ { i } ) } .\tag{3}
$$

Observed successes and failures both enter this estimate, weighted by the demands of their tasks. An observation may contribute to several cells because a task can require several capabilities. Each cell describes performance on tasks that demand the corresponding capability at a specified difficulty. If a cell is unobserved, lookup uses the nearest observed difficulty level, preferring the higher level in a tie. A capability with no observed level contributes zero, and the remaining weights retain their original values.

LLM-based response evaluation provides another source of quality evidence [48]. RouteAlign constructs a second profile, $A ^ { \mathrm { j u d g e } }$ , from capability-specific assessments of recorded responses against references and task rubrics. These assessments use the common capability and difficulty schema with independently annotated requirement weights. Outcome scores remain in [0, 1], and response assessments remain in [0, 10]; separate coefficients in the shared scorer account for the two channels. The assessment protocol and aggregation rule are specified in the Supplementary Material.

For an incoming task, each profile yields a global compatibility estimate:

$$
Q ^ { a } ( x , m ) = \sum _ { k = 1 } ^ { K } r _ { k } ( x ) A _ { m , k , d ( x ) } ^ { a } ,\tag{4}
$$

The predicted difficulty selects the measurement level, and the incoming capability weights determine how the selected cells contribute. Historical demands thus govern the construction of outcome measurements, while current demands govern their use. Difficulty conditioning preserves the distinction between routine and demanding applications of a capability. Weighted aggregation also allows mixed tasks to draw on several cells, instead of forcing each observation or request into one exclusive skill category.

## 4.3 Capability-Conditioned Local Quality Estimation

Global profiles combine observations across topics, which can obscure distinctions relevant to a particular request. RouteAlign retains this local context by retrieving the $k _ { \mathrm { n n } }$ nearest historical tasks in a frozen embedding space. Let ${ \mathcal { N } } ( x )$ denote the neighborhood and $\gamma _ { j } ( x )$ its normalized weights, proportional to 6 $\displaystyle \operatorname { \ a x p } ( - \beta \delta ( x , x _ { j } ) )$ for cosine distance $\delta .$ The scalar quality and generation-cost estimates are

$$
\begin{array} { l } { { p _ { \mathrm { n n } } ( x , m ) = \displaystyle \sum _ { j \in \mathcal { N } ( x ) } \gamma _ { j } ( x ) s _ { j m } , } } \\ { { \widehat { c } ( x , m ) = \displaystyle \sum _ { j \in \mathcal { N } ( x ) } \gamma _ { j } ( x ) c _ { j m } . } } \end{array}\tag{5}
$$

These weights reflect semantic proximity but do not distinguish the capability demands underlying neighboring outcomes. RouteAlign therefore also estimates local performance for each capability and combines those estimates according to the incoming requirements:

$$
\begin{array} { l } { { p _ { k } ^ { \mathrm { c a p } } ( x , m ) = \displaystyle \frac { \sum _ { j \in \mathcal { N } ( x ) } \gamma _ { j } ( x ) r _ { k } ( x _ { j } ) s _ { j m } } { \sum _ { j \in \mathcal { N } ( x ) } \gamma _ { j } ( x ) r _ { k } ( x _ { j } ) } , } } \\ { { Q ^ { \mathrm { l o c a l } } ( x , m ) = \displaystyle \sum _ { k = 1 } ^ { K } r _ { k } ( x ) p _ { k } ^ { \mathrm { c a p } } ( x , m ) . } } \end{array}\tag{6}
$$

If the denominator is at most $1 0 ^ { - 6 }$ , the capability estimate falls back to $p _ { \mathrm { n n } } ( x , m )$ . Neighbors need not share the incoming difficulty level. Difficulty instead conditions the global profiles and the category-by-difficulty mean used by the scorer.

The historical weight $r _ { k } ( x _ { j } )$ controls how much a neighboring outcome informs capability $k ;$ the incoming weight $r _ { k } ( x )$ controls how much that capability estimate matters to the current decision. In the introductory example, a task dominated by code implementation can therefore contribute differently from one dominated by quantitative reasoning, even when both are semantically close to the request. Equations (3) and (6) apply the same requirement coordinates at different scopes: the former aggregates model evidence globally, and the latter adapts evidence within the retrieved neighborhood.

The resulting influence on each neighbor can be made explicit. Let $\begin{array} { r } { Z _ { k } ( x ) = \sum _ { j \in \mathcal { N } ( x ) } \gamma _ { j } ( x ) r _ { k } ( x _ { j } ) } \end{array}$ . When $Z _ { k } ( x ) > 1 0 ^ { - 6 }$ for every capability with positive incoming weight, Equation (6) can be written as

$$
\begin{array} { c } { { Q ^ { \mathrm { l o c a l } } ( x , m ) = \displaystyle \sum _ { j \in \mathcal { N } ( x ) } \omega _ { j } ( x ) s _ { j m } , } } \\ { { \omega _ { j } ( x ) = \gamma _ { j } ( x ) \displaystyle \sum _ { k : r _ { k } ( x ) > 0 } \frac { r _ { k } ( x ) r _ { k } ( x _ { j } ) } { Z _ { k } ( x ) } . } } \end{array}\tag{7}
$$

The effective weights are nonnegative and sum to one. Normalization by $Z _ { k } ( x )$ gives each capability a weighted mean within the neighborhood; its contribution to the final estimate is then set by the incoming requirement weight. A capability does not receive a larger share merely because it is more prevalent among the neighbors. Semantic retrieval determines which observations are available, and requirement alignment determines their relative influence. If all neighbors have the same supported capability mixture, the effective weights reduce to $\gamma _ { j } ( x )$ . When a required capability has insufficient support, its portion of the estimate uses the scalar fallback.

## 4.4 Shared Quality Scoring and Cost-Aware Selection

Global and local estimates summarize different aspects of performance but may share predictive information. RouteAlign fits one linear quality estimator, $\widehat { q } ( x , m ) = \theta ^ { \top } f ( x , m )$ , jointly across candidates. Its ten features include the two global compatibility scores, scalar and capability-conditioned local quality, the sample-category mean $b ,$ the category-by-difficulty mean $^ { g , }$ and four local comparison features. The latter comprise local quality multiplied by candidate separation, the margin over the strongest eligible alternative, an indicator of the highest local estimate, and disagreement with the category-by-difficulty mean. Joint fitting determines the contribution of each estimate in the presence of the others.

The terms sample-category prior and category-by-difficulty prior refer to empirical means of training outcomes, selected using an auxiliary sample-category prediction. An empty category-by-difficulty stratum falls back to the category mean. Candidate separation is the maximum local quality minus the mean across eligible candidates; in the benchmark, it is normalized by the mean separation in the corresponding feature-extraction batch. Selection ties follow a deterministic candidate order, and a single eligible candidate uses zero as the reference for its margin. These definitions are shared across models, so candidate identity indexes evidence without defining a learned output class. The Supplementary Material gives the complete feature vector and specifies how missing records are handled.

Ridge regression fits the scorer to recorded training scores:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \Bigg \{ \sum _ { ( i , m ) \in \mathcal { P } _ { \mathrm { t r a i n } } } \left( \theta ^ { \top } f ( x _ { i } , m ) - s _ { i m } \right) ^ { 2 } + \alpha \| \theta \| _ { 2 } ^ { 2 } \Bigg \} .\tag{8}
$$

Training uses native feature scales and no intercept, producing an unconstrained regression score. Retrieval for a training query excludes that query itself, while global profiles and category priors aggregate the training partition. At invocation time, selection follows

$$
\pi _ { t } ( x ) \in \arg \operatorname* { m a x } _ { m \in { \mathcal F } _ { t } ( x ) } \bigl [ \theta ^ { \top } f ( x , m ) - \lambda \widehat c ( x , m ) \bigr ] .\tag{9}
$$

The fitted coefficients estimate task quality from the available evidence, and λ determines how estimated generation expenditure affects the choice. Changing this preference changes the operating point without refitting the quality estimator. Requirement prediction and retrieval are shared across candidates; subsequent aggregation and scoring are linear in catalog size for a fixed schema and neighborhood size. Section 5.5 measures the cached selection stage and a separate endpoint that includes requirement prediction.

## 4.5 Updating Evidence as the Service Catalog Changes

The shared representation separates updates to candidate measurements from fitting the comparison rule. With task annotations fixed, each outcome-profile cell depends on a total requirement weight and a weighted sum of outcomes. Reference measurements for an added candidate populate its cells without changing those of existing candidates. A model replacement requires measurements for the new version, removal changes the active set, and repricing updates service metadata and the cost records used by the local estimator. Candidate-relative features are recomputed for the eligible set.

Requirement annotations and reference embeddings remain reusable as long as the profiler, schema, and reference tasks are unchanged. Newly measured candidates can then be expressed through the existing feature definitions and evaluated with retained scoring coefficients. Section 5.6 tests this property by fitting on an initial catalog, freezing the coefficients, and refreshing candidate evidence after each catalog update.

Table 1 Task quality and generation cost across ten benchmarks. S-Avg. and D-Avg. denote Sample-Avg and Dataset-Avg. Bold marks the highest quality or lowest generation cost in each column.
<table><tr><td></td><td colspan="9">Per-dataset score (%)</td><td colspan="2">Overall (%)</td><td></td></tr><tr><td>Selection policy</td><td>AIME</td><td>AH GPQA</td><td></td><td>HLE</td><td>LCB</td><td>LMB</td><td>MMP</td><td>SQA</td><td>SWE</td><td>τ2</td><td>S-Avg. D-Avg.</td><td></td><td>Cost ($)</td></tr><tr><td>Random Router</td><td>61.11</td><td>68.14</td><td>73.33</td><td>10.34</td><td>62.78</td><td>64.86</td><td>84.44</td><td>34.13</td><td>23.33</td><td>40.48</td><td>47.38</td><td>52.29</td><td>51.11</td></tr><tr><td>KNNRouter</td><td>61.11</td><td>71.46</td><td>58.33</td><td>13.73</td><td>63.41</td><td>64.86</td><td>82.89</td><td>53.93</td><td>28.67</td><td>57.14</td><td>55.07</td><td>55.55</td><td>34.96</td></tr><tr><td>SVMRouter</td><td>77.78</td><td>76.99</td><td>55.00</td><td>9.72</td><td>58.36</td><td>72.97</td><td>83.78</td><td>54.01</td><td>16.67</td><td>41.67</td><td>53.80</td><td>54.69</td><td>6.98</td></tr><tr><td>MLPRouter</td><td>72.22</td><td>70.80</td><td>61.67</td><td>18.52</td><td>67.82</td><td>59.46</td><td>82.78</td><td>50.69</td><td>24.00</td><td>70.24</td><td>55.24</td><td>57.82</td><td>42.44</td></tr><tr><td>MM-Embed-S</td><td>66.67</td><td>69.69</td><td>81.67</td><td>15.74</td><td>79.18</td><td>70.27</td><td>83.33</td><td>53.00</td><td>28.00</td><td>53.57</td><td>56.78</td><td>60.11</td><td>56.28</td></tr><tr><td>MM-Embed-L</td><td>72.22</td><td>74.34</td><td>80.00</td><td>16.51</td><td>76.34</td><td>75.68</td><td>83.56</td><td>53.08</td><td>25.33</td><td>48.81</td><td>56.88</td><td>60.59</td><td>58.48</td></tr><tr><td>mmBERT-32K</td><td>66.67</td><td>73.23</td><td>78.33</td><td>16.67</td><td>76.66</td><td>75.68</td><td>83.22</td><td>53.39</td><td>28.00</td><td>54.76</td><td>57.08</td><td>60.66</td><td>59.28</td></tr><tr><td>GraphRouter (cost)</td><td>55.56</td><td>70.13</td><td>70.00</td><td>8.33</td><td>64.04</td><td>67.57</td><td>83.56</td><td>46.84</td><td>17.33</td><td>41.67</td><td>51.19</td><td>52.50</td><td>7.12</td></tr><tr><td>GraphRouter (balance)</td><td>77.78</td><td>76.99</td><td>70.00</td><td>9.88</td><td>58.36</td><td>72.97</td><td>84.11</td><td>54.01</td><td>17.33</td><td>41.67</td><td>54.17</td><td>56.31</td><td>10.14</td></tr><tr><td>GraphRouter (quality)</td><td>66.67</td><td>75.22</td><td>58.33</td><td>14.97</td><td>64.67</td><td>40.54</td><td>83.33</td><td>54.01</td><td>30.67</td><td>39.29</td><td>55.22</td><td>52.77</td><td>146.68</td></tr><tr><td>RouteLLM</td><td>83.33</td><td>73.89</td><td>88.33</td><td></td><td>25.93 84.54</td><td></td><td>78.38 87.22</td><td>56.86</td><td>16.67</td><td>69.05</td><td>61.69</td><td>66.42</td><td>109.27</td></tr><tr><td>RouterDC</td><td>83.33</td><td>71.24</td><td>88.33</td><td></td><td>25.93 84.54</td><td></td><td>78.38 87.22</td><td>48.38</td><td>16.00</td><td>69.05</td><td>58.56</td><td>65.24</td><td>122.62</td></tr><tr><td>AvengersPro</td><td>83.33</td><td>71.24</td><td>76.67</td><td>22.84</td><td>77.92</td><td>78.38</td><td>85.67</td><td>53.54</td><td>26.00</td><td>60.71</td><td>58.91</td><td>63.63</td><td>61.35</td></tr><tr><td>RouteAlign</td><td></td><td>83.33 78.10</td><td></td><td>81.67 26.08</td><td>83.60 81.08</td><td></td><td>86.89</td><td>58.47</td><td>33.33</td><td>73.81</td><td>63.07</td><td>68.64</td><td>110.42</td></tr></table>

AH: ArenaHard; LCB: LiveCodeBench; LMB: LiveMathBench; MMP: MMLU-Pro; SQA: SimpleQA; SWE: SWE-bench.

## 5 Empirical Evaluation

The experiments assess selection quality and generation cost, the contributions of different feature groups, selection time, and scorer reuse after catalog changes. Four research questions organize the evaluation:

• RQ1: Quality and cost. How does RouteAlign compare with existing routers, open Jev-style decision backends, and direct candidate invocation in quality and generation expenditure?

• RQ2: Feature contributions. How do global, local, and requirement-conditioned estimates contribute to quality within the shared scorer?

• RQ3: Execution overhead. How long does selection take with cached requirements and embeddings, and what latency is observed at a separate endpoint that also predicts requirements?

• RQ4: Scorer reuse. Can coefficients fitted on an initial catalog remain effective after candidate evidence is refreshed to reflect catalog changes?

## 5.1 Experimental Setup

## 5.1.1 Datasets and Train-Test Split

We use the requests, model responses, task scores, and generation costs recorded in LLMRouterBench [24]. The workload comprises AIME, ArenaHard, GPQA, HLE, LiveCodeBench, LiveMathBench, MMLU-Pro, SimpleQA, SWE-bench, and τ<sup>2</sup>-bench. Partitioning by dataset and prompt with seed 42 and a 70% training fraction produces 8,710 training requests and 3,738 test requests. All responses to a request remain in the same partition. Model measurement, retrieval references, category priors, and scorer fitting use training records. A policy chooses a candidate before accessing its recorded test outcome.

The 13 model configurations represent alternative LLM services whose recorded outcomes permit comparison on a shared workload. Direct-candidate results use available records: GPT-5 covers the full test workload, whereas GPT-5-Chat lacks τ<sup>2</sup>-bench records. Dense retrieval features and regression targets use zero for absent records; global outcome profiles aggregate observed records only. RouteAlign selects an observed response for every test request. Dataset sizes, candidate coverage, and selection counts are reported in the Supplementary Material.

## 5.1.2 Baseline Policies and Implementations

The comparison includes preference-based RouteLLM [10], contrastive RouterDC [11], three utility configurations of GraphRouter [33], and cluster-based AvengersPro [36]. KNNRouter, SVMRouter, and MLPRouter provide classification-based comparisons [25]. We also evaluate three embedding-based retrieval policies, uniform random selection, and direct invocation of individual candidates. This set covers several ways of learning or aggregating historical performance evidence.

RouteLLM uses a multi-candidate matrix-factorization adaptation, and AvengersPro uses the benchmark implementation of cluster-based rank aggregation. Results refer to the implementations and training budgets detailed in the Supplementary Material. In particular, classification-based KNNRouter predicts a winning candidate label, whereas the retrieval-based policies estimate performance separately for each candidate.

## 5.1.3 Evaluation Metrics and Configuration

Sample-Avg weights requests equally, while Dataset-Avg weights dataset means equally. Both preserve graded task scores and are reported as percentages. Generation cost sums recorded candidate charges in US dollars, excluding requirement prediction, selector execution, offline profiling, and scorer fitting. Selector runtime is measured separately. Quality-cost comparisons use each policy’s observed expenditure at its specified operating point.

RouteAlign uses the profiler in Section 4.1 and sentence embeddings for semantic retrieval [49]. The retrieval encoder is the frozen all-MiniLM-L6-v2 checkpoint with 384-dimensional outputs. Defaults are $k _ { \mathrm { n n } } = 1 5 0$ $\beta = 5 ,$ , ridge penalty α = 1, and cost preference λ = 0.4. Feature ablations retain the partition and definitions of the remaining features and refit the scorer.

## 5.2 Selection Quality and Generation Cost

## 5.2.1 Aggregate and Per-Dataset Quality

RouteAlign achieves the highest aggregate quality among the evaluated policies in the fixed catalog, with 63.07% Sample-Avg and 68.64% Dataset-Avg (Table 1). The strongest competing router, RouteLLM, obtains 61.69% and 66.42%, respectively. The respective gains of 1.38 and 2.22 percentage points show improvement at the workload level and under equal weighting of datasets.

Per-dataset scores reveal how the advantage varies across task groups. Among the policies in Table 1, RouteAlign ranks first, including ties, on seven datasets, while RouteLLM and RouterDC achieve higher scores on GPQA, LiveCodeBench, and MMLU-Pro. The largest gain over RouteLLM occurs on SWE-bench, from 16.67% to 33.33%. This distribution of gains motivates examining the evidence used to select candidates for different task demands.

Table 2 tests the value of selection against committing to one candidate. GPT-5 has the highest individual Dataset-Avg and covers the same 3,738 requests as RouteAlign. Against this default, RouteAlign gains 4.70 Sample-Avg points and 3.53 Dataset-Avg points. It also exceeds the highest individual Sample-Avg, achieved by Gemini 2.5 Pro, by 3.89 points. RouteAlign assigns 1,852 requests to GPT-5 and the remaining 1,886 to other candidates. Its advantage over a strong default is therefore obtained through selective use of the catalog. Section 5.4 examines the feature contributions underlying these decisions.

## 5.2.2 Quality and Generation Cost Comparisons

Figure 3 shows the empirical quality-cost frontiers under both aggregate metrics. RouteAlign is Pareto nondominated in each panel and achieves the highest task quality among the plotted configurations. Its generation cost of \$110.42 exceeds RouteLLM’s \$109.27 by \$1.15, or 1.05%, for gains of 1.38 Sample-Avg and 2.22 Dataset-Avg points. RouteAlign thus extends the frontier toward higher quality with a modest increase in expenditure.

RouteAlign also dominates RouterDC and direct GPT-5 invocation in both panels, achieving higher quality at lower generation cost. Against GPT-5, it reduces expenditure from \$124.82 to \$110.42, a saving of 11.5%.

Table 2 Comparison with direct invocation of individual candidates. Scores (%) and generation costs (US dollars) use available records. GPT-5 covers all test requests; GPT-5-Chat excludes τ<sup>2</sup>-bench.
<table><tr><td rowspan="2">Candidate or policy</td><td colspan="2">Quality (%)</td><td rowspan="2">Cost ($)</td></tr><tr><td>S-Avg.</td><td>D-Avg.</td></tr><tr><td>Intern-S1</td><td>37.74</td><td>43.03</td><td>13.85</td></tr><tr><td>DeepSeek V3-0324</td><td>40.41</td><td>42.60</td><td>4.78</td></tr><tr><td>DeepSeek V3.1-Terminus</td><td>44.18</td><td>51.72</td><td>9.43</td></tr><tr><td>DeepSeek R1-0528</td><td>47.85</td><td>55.57</td><td>60.75</td></tr><tr><td>Qwen3-235B</td><td>53.81</td><td>54.70</td><td>6.98</td></tr><tr><td>Qwen3-235B Thinking</td><td>53.13</td><td>56.27</td><td>36.84</td></tr><tr><td>Kimi K2-0905</td><td>45.09</td><td>54.76</td><td>14.53</td></tr><tr><td>GLM-4.6</td><td>44.65</td><td>55.48</td><td>61.39</td></tr><tr><td>GPT-5-Chat</td><td>46.89</td><td>50.83</td><td>17.96</td></tr><tr><td>Gemini 2.5 Flash</td><td>42.64</td><td>45.74</td><td>32.97</td></tr><tr><td>Claude Sonnet 4</td><td>38.41</td><td>46.47</td><td>70.28</td></tr><tr><td>Gemini 2.5 Pro</td><td>59.18</td><td>59.55</td><td>233.97</td></tr><tr><td>GPT-5</td><td>58.37</td><td>65.11</td><td>124.82</td></tr><tr><td>AVG LLM</td><td>47.10</td><td>52.45</td><td>52.96</td></tr><tr><td>Random Router</td><td>47.38</td><td>52.29</td><td>51.11</td></tr><tr><td>RouteAlign</td><td>63.07</td><td>68.64</td><td>110.42</td></tr></table>

Table 3 Open Jev-style backends with direct decision scoring $( \rho = 1 )$ , plus the strongest configuration across all three mixtures. Cost covers recorded candidate generation. Bold marks the highest aggregate quality.
<table><tr><td>Decision backend</td><td>D-Avg.</td><td>S-Avg.</td><td>Cost ($)</td></tr><tr><td>Kev-0.8B</td><td>51.83</td><td>49.65</td><td>39.33</td></tr><tr><td>OpenJev-mjdileep-0.8B</td><td>47.44</td><td>45.85</td><td>52.57</td></tr><tr><td>Open-Jev-2B</td><td>48.93</td><td>48.86</td><td>39.52</td></tr><tr><td>Decider-2B</td><td>55.00</td><td>47.11</td><td>95.63</td></tr><tr><td>SemIf-4B</td><td>53.43</td><td>51.82</td><td>29.32</td></tr><tr><td>Kev-4B</td><td>61.43</td><td>55.43</td><td>102.97</td></tr><tr><td>Decider-4B</td><td>60.61</td><td>57.08</td><td>109.32</td></tr><tr><td>Open-Jev-9B</td><td>53.44</td><td>47.61</td><td>44.66</td></tr><tr><td>Kev-9B</td><td>60.50</td><td>56.51</td><td>111.32</td></tr><tr><td>Decider-4B (ρ = 0)</td><td>62.20</td><td>57.85</td><td>122.28</td></tr><tr><td>RouteAlign</td><td>68.64</td><td>63.07</td><td>110.42</td></tr></table>

## 5.3 Comparison with Jev-Style Decision Models

This comparison examines whether a general decision model supplied with task and model descriptions can replace the trained selection policy. We evaluate nine locally deployed, open Jev-style backends spanning 0.8B to 9B: three Kev sizes, two Open-Jev sizes, two Decider sizes, SemIf, and OpenJev-mjdileep. Each exposes typed choices and scores through a common local adapter.

Every configuration uses the same 3,738 requests, 13-model catalog, and recorded outcomes. Each backend predicts requirements and rates candidate profile summaries, retaining the adapter’s metadata filtering and capability-tier matching. The backends receive no additional routing-task fine-tuning. We vary the decision-score weight $\rho \in \{ 0 , 0 . 5 , 1 \}$ in a mixture with explicit requirement-profile compatibility. At $\rho = 0$ the backend still supplies the requirements and candidate set, while compatibility determines the candidate scores. The comparison evaluates complete selection policies: the Jev-style variants use backend predictions and supplied profiles, and RouteAlign combines its supervised profiler with local outcome evidence and a fitted scorer.

RouteAlign leads all 27 configurations on both aggregate metrics (Figure 4). The strongest configuration, Decider-4B with $\rho = 0 .$ , reaches 62.20% Dataset-Avg and 57.85% Sample-Avg. RouteAlign gains 6.44 and 5.22 percentage points while reducing generation cost by 9.7% (Table 3). Under direct decision scoring, Kev-4B has the highest Dataset-Avg and Decider-4B the highest Sample-Avg. Against the latter at similar expenditure, RouteAlign gains 8.03 and 5.99 points for 1.0% additional generation cost. Its advantage persists across all three mixtures, supporting the effectiveness of a policy trained to connect task requirements with global and local performance evidence. Complete mixture results, per-dataset scores, backend configurations, and project links appear together in the Supplementary Material.

![](images/10d2976b9691a7892a6a4e1c5386268917158a61dcc64aeb10b7cf8c0ea04793.jpg)  
Figure 3 Task quality versus generation cost for the fixed-catalog configurations. Grey circles denote direct candidate invocation; colored markers denote selection policies. Dashed curves are schematic guides to the Pareto frontiers. Cost includes answer generation only; direct-candidate summaries use available records.

## 5.4 Ablation Analysis

## 5.4.1 Ablation Protocol

RQ2 examines the fitted combination through the complete set of feature ablations in Table 4. Each ablation removes specified feature columns and refits ridge regression on the same training partition using leaveone-out retrieval features. The encoder and definitions of retained features remain fixed. Refitting permits compensation among estimates, so a measured change reflects the contribution of the removed columns conditional on the retained evidence. The cost-preference ablation instead retains the fitted quality estimator and sets λ = 0.

Removing all local quality evidence deletes scalar neighborhood quality, capability-conditioned local compatibility, and four candidate-comparison features, while retaining the generation-cost estimate. The requirementconditioned group contains Q<sup>out</sup>, Q<sup>judge</sup>, the category-by-difficulty prior g, and Q<sup>local</sup>. Global profile compatibility contains the two global scores. The Supplementary Material lists the removed columns and clarifies which derived features remain when an individual input is deleted.

## 5.4.2 Local Evidence versus Category Means

The largest quality loss follows removal of all local quality evidence: 3.12 Dataset-Avg points and 1.66 Sample-Avg points. Even after refitting, global profiles and category priors do not replace the information supplied by related task outcomes. Removing the scalar neighborhood estimate alone costs 0.61 Dataset-Avg points and 0.19 Sample-Avg points. Removing the four relative local features lowers Dataset-Avg by 0.46 points but raises Sample-Avg by 0.15 points. Local evidence is therefore consequential at the group level, while its relative features affect quality differently under the two task-weighting conventions.

Selection by category means provides a complementary check on whether coarse task groupings are sufficient. At similar observed generation expenditure, selection by the category-by-difficulty mean trails RouteAlign by

![](images/322a8a88951dc8dee84706eadfe58eb331d9afac10c6d04313ce1135662fca78.jpg)  
Figure 4 Aggregate quality of nine open Jev-style backends under three scoring mixtures. Each row uses one backend for requirement prediction; ρ weights its decision score, with explicit requirement-profile compatibility weighted by $1 - \rho .$ Vertical lines mark full RouteAlign. All 27 configurations use the same held-out requests and catalog.

Table 4 Full ablation results for RouteAlign. ∆ denotes the change from full RouteAlign in percentage points. Bold identifies the reference configuration.
<table><tr><td>Removed evidence or preference</td><td>D-Avg.</td><td>∆D</td><td>S-Avg.</td><td>∆S</td><td>Cost ($)</td></tr><tr><td>Full RouteAlign</td><td>68.64</td><td>0.00</td><td>63.07</td><td>0.00</td><td>110.42</td></tr><tr><td>All local quality evidence</td><td>65.52</td><td>-3.12</td><td>61.41</td><td>-1.66</td><td>106.24</td></tr><tr><td>Scalar neighborhood estimate</td><td>68.03</td><td>-0.61</td><td>62.88</td><td>-0.19</td><td>110.41</td></tr><tr><td>Sample-category prior</td><td>68.05</td><td>-0.59</td><td>62.93</td><td>-0.14</td><td>111.85</td></tr><tr><td>Local capability conditioning</td><td>68.12</td><td>-0.52</td><td>63.08</td><td>+0.01</td><td>110.96</td></tr><tr><td>Relative local evidence</td><td>68.18</td><td>-0.46</td><td>63.22</td><td>+0.15</td><td>109.42</td></tr><tr><td>Response assessment compatibility</td><td>68.26</td><td>-0.38</td><td>63.06</td><td>-0.01</td><td>110.00</td></tr><tr><td>Cost penalty (λ = 0)</td><td>68.27</td><td>-0.37</td><td>63.32</td><td>+0.25</td><td>124.82</td></tr><tr><td>Requirement-conditioned scores</td><td>68.39</td><td>-0.25</td><td>63.15</td><td>+0.08</td><td>114.73</td></tr><tr><td>Global profile compatibility</td><td>68.53</td><td>-0.11</td><td>62.99</td><td>-0.08</td><td>109.86</td></tr><tr><td>Category and difficulty prior</td><td>68.83</td><td>+0.19</td><td>62.92</td><td>-0.15</td><td>109.02</td></tr><tr><td>Outcome compatibility</td><td>68.64</td><td>0.00</td><td>63.07</td><td>0.00</td><td>110.59</td></tr></table>

2.22 Dataset-Avg points and 1.62 Sample-Avg points. The complete policy therefore improves on selecting the candidate with the highest mean score in the predicted category and difficulty group. The supplementary comparison reports the complete decision rules and their generation costs.

## 5.4.3 Requirement Conditioning and Feature Overlap

Local capability conditioning contributes most clearly under equal dataset weighting. Removing $Q ^ { \mathrm { l o c a l } }$ while retaining scalar neighborhood evidence lowers Dataset-Avg by 0.52 points, whereas Sample-Avg changes from 63.07% to 63.08%. This tests the explicit contribution of $Q ^ { \mathrm { l o c a l } }$ to the refitted scorer. Removing response-assessment compatibility lowers Dataset-Avg by 0.38 points and Sample-Avg by 0.01 points; removing outcome compatibility alone leaves both rounded averages unchanged.

Grouped removals reveal overlap among the estimates. Removing the four requirement-conditioned scores lowers Dataset-Avg by 0.25 points; removing the two global scores lowers it by 0.11 points. The samplecategory prior improves both metrics, while the category-by-difficulty prior has mixed effects. Because refitting redistributes weight across retained features, these changes characterize the fitted combination. Taken together, the ablations identify local context as the most consequential evidence group and show that local capability conditioning contributes under equal dataset weighting.

Table 5 Selection-stage runtime over 3,738 test requests. RouteAlign uses cached requirement profiles and embeddings. Candidate answer generation is excluded.
<table><tr><td>Selection policy</td><td>Total (s)</td><td>Per query (ms)</td></tr><tr><td>Random Router</td><td>0.0011</td><td>0.0003</td></tr><tr><td>KNNRouter</td><td>76.00</td><td>20.33</td></tr><tr><td>SVMRouter</td><td>83.86</td><td>22.43</td></tr><tr><td>MLPRouter</td><td>75.56</td><td>20.21</td></tr><tr><td>MM-Embed-S</td><td>4.23</td><td>1.13</td></tr><tr><td>MM-Embed-L</td><td>9.42</td><td>2.52</td></tr><tr><td>mmBERT-32K</td><td>9.23</td><td>2.47</td></tr><tr><td>GraphRouter (cost)</td><td>9.54</td><td>2.55</td></tr><tr><td>GraphRouter (balance)</td><td>9.67</td><td>2.59</td></tr><tr><td>GraphRouter (quality)</td><td>9.43</td><td>2.52</td></tr><tr><td>RouteLLM</td><td>85.97</td><td>23.00</td></tr><tr><td>RouterDC</td><td>42.33</td><td>11.32</td></tr><tr><td>AvengersPro</td><td>168.94</td><td>45.20</td></tr><tr><td>RouteAlign</td><td>119</td><td>31.9</td></tr></table>

## 5.4.4 Effect of the Cost Preference

Setting λ = 0 increases generation expenditure from \$110.42 to \$124.82, lowers Dataset-Avg by 0.37 points, and raises Sample-Avg by 0.25 points. The default preference reduces generation cost by 11.5% relative to this setting while changing the distribution of quality across tasks. Because the quality estimator is held fixed, the difference isolates the effect of applying the cost preference to the learned scores. The result shows that the preference changes candidate choices without refitting the scorer; its appropriate value depends on the workload and desired quality-cost balance.

## 5.5 Selection Time and Endpoint Latency

## 5.5.1 Selection with Cached Requirements and Embeddings

With requirement profiles and embeddings cached, RouteAlign selects candidates for 3,738 requests in approximately 119 seconds, averaging 31.9 milliseconds per request (Table 5). Its selection stage is 8.9 milliseconds longer than RouteLLM’s and shorter than the 45.20 milliseconds reported for AvengersPro. RouterDC and the embedding-based retrieval policies achieve lower selection times. Together with Section 5.2, these measurements locate RouteAlign’s quality gains within the execution costs of the evaluated policies.

The measured interval includes retrieval, feature construction, and candidate scoring. Requirement prediction, embedding construction, offline preparation, and candidate answer generation are excluded.

## 5.5.2 Endpoint Latency with Online Requirement Prediction

A separate endpoint executes the 0.6B profiler and selection with a 500-request reference bank. At concurrency 32, it processes 96 requests at 51.2 requests per second, with median latency of 0.60 seconds and 95th-percentile latency of 0.74 seconds. No errors or fallbacks occur. Timing ends with candidate selection. For request-level execution, this endpoint uses unnormalized candidate separation and alternative reference means when category metadata is absent. It characterizes latency with online requirement prediction; Table 5 measures the cached selection stage of the fixed-catalog quality configuration.

## 5.6 Scorer Reuse under Service Catalog Changes

Table 6 Service catalog update protocol. Each configuration is defined by its parent and update event; candidates are indexed by their model configurations.
<table><tr><td>Stage</td><td>Parent</td><td>Event</td><td>Candidates</td></tr><tr><td>T0</td><td>n/a</td><td>Initial candidate pool</td><td>7</td></tr><tr><td>T1</td><td>TO</td><td>Add DeepSeek R1-0528</td><td>8</td></tr><tr><td>T2</td><td>T1</td><td>Add Qwen3-235B Thinking-2507</td><td>9</td></tr><tr><td>T3</td><td>T2</td><td>Add Claude Sonnet 4</td><td>10</td></tr><tr><td>T4</td><td>T3</td><td>Add Gemini 2.5 Pro</td><td>11</td></tr><tr><td>T5</td><td>T4</td><td>Add GPT-5</td><td>12</td></tr><tr><td>T6</td><td>T5</td><td>Replace DeepSeek V3-0324 with V3.1-Terminus</td><td>12</td></tr><tr><td>T7</td><td>T6</td><td>Remove GPT-5</td><td>11</td></tr><tr><td>T8</td><td>T6</td><td>Gemini 2.5 Flash price multiplied by 0.5</td><td>12</td></tr></table>

Table 7 Selection quality under controlled catalog changes. Entries are Dataset-Avg (%); Mean weights the nine configurations equally. T7 and T8 independently branch from T6. Bold marks the highest score in each column.
<table><tr><td></td><td colspan="6">Main sequence</td><td colspan="3">T6 branches</td><td></td></tr><tr><td>Selection policy</td><td>TO</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>Mean</td></tr><tr><td>RouteLLM</td><td>54.70</td><td>54.70</td><td>54.70</td><td>54.70</td><td>54.70</td><td>66.42</td><td>66.42</td><td>54.70</td><td>66.42</td><td>58.61</td></tr><tr><td>RouterDC</td><td>55.79</td><td>55.56</td><td>58.69</td><td>58.69</td><td>58.69</td><td>65.24</td><td>65.24</td><td>58.69</td><td>65.24</td><td>60.20</td></tr><tr><td>AvengersPro</td><td>52.94</td><td>52.69</td><td>54.52</td><td>58.52</td><td>57.81</td><td>60.49</td><td>61.88</td><td>60.58</td><td>64.72</td><td>58.24</td></tr><tr><td>RouteAlign-Frozen</td><td>60.36</td><td>61.33</td><td>62.26</td><td>63.48</td><td>64.18</td><td>68.55</td><td>68.60</td><td>64.08</td><td>68.64</td><td>64.61</td></tr></table>

## 5.6.1 Catalog Update Protocol

RQ4 tests whether updating candidate measurements is sufficient to retain effective selection with a fixed scorer, as proposed in Section 4.5. T0 contains seven candidates: Intern-S1, DeepSeek V3-0324, Qwen3-235B-2507, Kimi K2-0905, GLM-4.6, GPT-5-Chat, and Gemini 2.5 Flash. T1 through T5 add candidates, and T6 replaces a model version. T7 and T8 independently branch from T6, respectively removing GPT-5 and halving the price of Gemini 2.5 Flash. Table 6 lists the events and candidate counts. These controlled updates change candidate availability, performance evidence, or generation prices.

RouteAlign-Frozen fits its coefficients on the seven T0 candidates and retains them for all subsequent configurations. Newly admitted candidates are evaluated on training-partition reference tasks to obtain their measurements and profiles. Requirement predictions remain fixed, while candidate evidence and relative comparison features are refreshed. This protocol tests whether one learned comparison rule remains effective as measured candidates enter or leave the catalog.

RouteLLM and RouterDC retain checkpoints trained on all 13 candidates and mask unavailable outputs. Their fitting budgets are 50,000 preference pairs and 6,000 requests, respectively. AvengersPro recomputes clusters and candidate rankings from training evidence for each catalog.

## 5.6.2 Service Addition and Model Replacement

RouteAlign-Frozen achieves the highest Dataset-Avg in all nine configurations (Table 7). Its equally weighted mean is 64.61%, compared with 60.20% for RouterDC, the strongest baseline by this mean. The difference is 4.41 percentage points before rounding. A supplementary figure plots the addition and replacement sequence separately from the removal and repricing branches.

Along the main sequence, RouteAlign-Frozen rises from 60.36% at T0 to 64.18% at T4. Adding GPT-5 at T5 contributes a further 4.37 points, reaching 68.55%; the replacement at T6 yields 68.60%. The coefficients remain unchanged throughout. The retained scorer can therefore use newly supplied model measurements within the existing requirement coordinates and feature definitions, providing empirical support for the separation between evidence updates and parameter fitting.

## 5.6.3 Service Removal and Repricing

The two branches distinguish a change in available competence from a change in cost evidence. Removing GPT-5 at T7 lowers RouteAlign-Frozen by 4.52 points to 64.08%, quantifying the loss from withdrawing this candidate. The retained scorer still achieves the highest Dataset-Avg among the evaluated policies on the remaining catalog. Halving the price of Gemini 2.5 Flash at T8 yields 68.64%, an increase of 0.04 points relative to T6, with GPT-5 retained in the catalog.

Across these updates, refreshed reference measurements allow the retained scorer to compare new model versions, select from a reduced catalog, and respond to revised prices. RouteAlign-Frozen achieves the highest quality in every configuration using coefficients fitted on the initial seven candidates.

## 6 Conclusion

RouteAlign makes task requirements the basis for connecting measured model competence to service selection. Historical capability demands determine how outcomes form global profiles and local estimates; incoming demands determine how the shared scorer uses this evidence to assess candidate suitability. The resulting policy attains the highest quality on both empirical Pareto frontiers and exceeds the strongest tested Jev-style configuration by 6.44 Dataset-Avg points at 9.7% lower generation cost. The largest ablation loss follows removal of local evidence, establishing its central role in the fitted policy. RouteAlign-Frozen also leads across the tested catalog changes, showing that refreshed candidate measurements remain useful with the original scoring coefficients. These findings support requirement alignment as a consistent basis for combining global measurements, related task outcomes, and a reusable comparison rule.

## References

[1] R. Bommasani et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021. [Online]. Available: https://arxiv.org/abs/2108.07258

[2] M. P. Papazoglou, P. Traverso, S. Dustdar, and F. Leymann, “Service-oriented computing: State of the art and research challenges,” Computer, vol. 40, no. 11, pp. 38–45, Nov. 2007. [Online]. Available: https://doi.org/10.1109/MC.2007.400

[3] L. Zeng, B. Benatallah, A. H. H. Ngu, M. Dumas, J. Kalagnanam, and H. Chang, “QoS-aware middleware for web services composition,” IEEE Transactions on Software Engineering, vol. 30, no. 5, pp. 311–327, 2004. [Online]. Available: https://doi.org/10.1109/TSE.2004.11

[4] P. Liang et al., “Holistic evaluation of language models,” Transactions on Machine Learning Research, Aug. 2023. [Online]. Available: https://openreview.net/forum?id=iO4LZibEqW

[5] M. Dehghani et al., “The benchmark lottery,” arXiv preprint arXiv:2107.07002, 2021. [Online]. Available: https://arxiv.org/abs/2107.07002

[6] D. Kiela et al., “Dynabench: Rethinking benchmarking in NLP,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021, pp. 4110–4124. [Online]. Available: https://aclanthology.org/2021.naacl-main.324/

[7] J. Zhou et al., “Instruction-following evaluation for large language models,” arXiv preprint arXiv:2311.07911, 2023. [Online]. Available: https://arxiv.org/abs/2311.07911

[8] M. Suzgun et al., “Challenging BIG-bench tasks and whether chain-of-thought can solve them,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 13 003–13 051. [Online]. Available: https://aclanthology.org/2023.findings-acl.824

[9] Z. Huang et al., “RouterEval: A comprehensive benchmark for routing LLMs to explore model-level scaling up in LLMs,” in Findings ofthe Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 2025, pp. 3860–3887. [Online]. Available: https://doi.org/10.18653/v1/2025.findings-emnlp.208

[10] I. Ong et al., “RouteLLM: Learning to route LLMs with preference data,” in The Thirteenth International Conference on Learning Representations. OpenReview.net, 2025. [Online]. Available: https://openreview.net/forum?id=8sSqNntaMr

[11] S. Chen, W. Jiang, B. Lin, J. T. Kwok, and Y. Zhang, “RouterDC: Query-based router by dual contrastive learning for assembling large language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2024/hash 7a641b8ec86162fc875fb9f6456a542f-Abstract-Conference.html

[12] Y. Li, “Rethinking predictive modeling for LLM routing: When simple kNN beats complex learned routers,” arXiv:2505.12601v2, 2026. [Online]. Available: https://arxiv.org/abs/2505.12601v2

[13] D. Sculley et al., “Hidden technical debt in machine learning systems,” in Advances in Neural Information Processing Systems, vol. 28, 2015. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2015/ hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

[14] S. Amershi et al., “Software engineering for machine learning: A case study,” in Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice, 2019, pp. 291–300. [Online]. Available: https: //www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/

[15] L. Zeng, B. Benatallah, M. Dumas, J. Kalagnanam, and A. H. H. Ngu, “QoS-aware service composition: A retrospective,” IEEE Transactions on Software Engineering, vol. 51, no. 3, pp. 836–841, 2025. [Online]. Available: https://doi.org/10.1109/TSE.2025.3533423

[16] D. Ding et al., “Hybrid LLM: cost-efficient and quality-aware query routing,” in The Twelfth International Conference on Learning Representations. OpenReview.net, 2024. [Online]. Available: https://openreview.net/forum?id=02f3mUtqnM

[17] K. Lu et al., “Routing to the expert: Efficient reward-guided ensemble of large language models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 2024, pp. 1964–1974. [Online]. Available: https://doi.org/10.18653/v1/2024.naacl-long.109

[18] A. Mohammadshahi, A. R. Shaikh, and M. Yazdani, “Routoo: Learning to route to large language models effectively,” arXiv preprint arXiv:2401.13979, 2024. [Online]. Available: https://arxiv.org/abs/2401.13979

[19] L. Chen, M. Zaharia, and J. Zou, “FrugalGPT: How to use large language models while reducing cost and improving performance,” Transactions on Machine Learning Research, 2024. [Online]. Available: https://openreview.net/forum?id=cSimKw5p6R

[20] P. Aggarwal et al., “AutoMix: Automatically mixing language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract-Conference.html

[21] X. Wang et al., “MixLLM: Dynamic routing in mixed large language models,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 10 912–10 922. [Online]. Available: https://doi.org/10.18653/v1/2025.naacl-long.545

[22] D. Ding et al., “BEST-Route: Adaptive LLM routing with test-time optimal compute,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 13 870–13 884. [Online]. Available: https://proceedings.mlr.press/v267/ding25d.html

[23] Q. J. Hu et al., “RouterBench: A benchmark for multi-LLM routing system,” arXiv preprint arXiv:2403.12031, 2024. [Online]. Available: https://arxiv.org/abs/2403.12031

[24] H. Li et al., “LLMRouterBench: A massive benchmark and unified framework for LLM routing,” in Findings of the Association for Computational Linguistics: ACL 2026. Association for Computational Linguistics, 2026, pp. 37 733–37 754. [Online]. Available: https://doi.org/10.18653/v1/2026.findings-acl.1881

[25] T. Feng et al., “LLMRouter: Unified infrastructure for developing, evaluating, and deploying LLM routers,” arXiv:2608.06867, 2026. [Online]. Available: https://arxiv.org/abs/2608.06867

[26] M. Mitchell et al., “Model cards for model reporting,” in Proceedings of the Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, 2019, pp. 220–229. [Online]. Available: https://doi.org/10.1145/3287560.3287596

[27] T. Gebru et al., “Datasheets for datasets,” Communications of the ACM, vol. 64, no. 12, pp. 86–92, 2021. [Online]. Available: https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets

[28] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 4902–4912. [Online]. Available: https://aclanthology.org/2020.acl-main.442/

[29] NLLB Team et al., “No language left behind: Scaling human-centered machine translation,” arXiv preprint arXiv:2207.04672, 2022. [Online]. Available: https://arxiv.org/abs/2207.04672

[30] Y. Wu et al., “WritingBench: A comprehensive benchmark for generative writing,” arXiv preprint arXiv:2503.05244, 2025. [Online]. Available: https://arxiv.org/abs/2503.05244

[31] S. Zhou et al., “WebArena: A realistic web environment for building autonomous agents,” in Proceedings of the Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=rmiwIL98uQ

[32] M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming and robust refusal,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 35 181–35 224. [Online]. Available: https://proceedings.mlr.press/v235/mazeika24a.html

[33] T. Feng, Y. Shen, and J. You, “GraphRouter: A graph-based router for LLM selections,” in The Thirteenth International Conference on Learning Representations. OpenReview.net, 2025. [Online]. Available: https://openreview.net/forum?id=eU39PDsZtT

[34] W. Jitkrittum et al., “Universal model routing for efficient LLM inference,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 10 169–10 218. [Online]. Available: https://proceedings.iclr.cc/ paper\_files/paper/2026/hash/113ae3a9762ca2168f860a8501d6ae25-Abstract-Conference.html

[35] C. Wang et al., “ICL-Router: In-context learned model representations for LLM routing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 39, 2026, pp. 33 413–33 421. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/40628

[36] Y. Zhang et al., “Beyond GPT-5: Making LLMs cheaper and better via performance-efficiency optimized routing,” in Proceedings of the 2025 7th International Conference on Distributed Artificial Intelligence, ser. DAI ’25. New York, NY, USA: Association for Computing Machinery, 2025, pp. 122–129. [Online]. Available: https://doi.org/10.1145/3772429.3772445

[37] W. Song et al., “IRT-Router: Effective and interpretable multi-LLM routing via item response theory,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 15 629–15 644. [Online]. Available: https://doi.org/10.18653/v1/2025.acl-long.761

[38] H. Shi et al., “InferenceDynamics: Efficient routing across LLMs through structured capability and knowledge profiling,” arXiv preprint arXiv:2505.16303, 2025. [Online]. Available: https://arxiv.org/abs/2505.16303

[39] J. Wang, Y. Ming, Z. Ke, S. Joty, A. Albarghouthi, and F. Sala, “SkillOrchestra: Learning to route agents via skill transfer,” arXiv preprint arXiv:2602.19672, 2026. [Online]. Available: https://arxiv.org/abs/2602.19672

[40] B. Lv et al., “Beyond query memorization: Large language model routing with query decomposition and historical matching,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2026, pp. 39 876–39 892. [Online]. Available: https://aclanthology.org/2026.acl-long.1852

[41] J. Wei et al., “Finetuned language models are zero-shot learners,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=gEZrGCozdqR

[42] A. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[43] Y. Zheng, R. Zhang, J. Zhang, Y. Ye, and Z. Luo, “LlamaFactory: Unified efficient fine-tuning of 100+ language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2024, pp. 400–410. [Online]. Available: https://aclanthology.org/2024.acl-demos.38

[44] M. Chen et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021. [Online]. Available: https://arxiv.org/abs/2107.03374

[45] Z. Chen et al., “FinQA: A dataset of numerical reasoning over financial data,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 3697–3711. [Online]. Available: https://aclanthology.org/2021.emnlp-main.300

[46] Y. Bai et al., “LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 3639–3664. [Online]. Available: https://aclanthology.org/2025.acl-long.183

[47] K. Lee et al., “Deduplicating training data makes language models better,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 8424–8445. [Online]. Available: https://aclanthology.org/2022.acl-long.577/

[48] L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 46 595–46 623. [Online]. Available: https://proceedings.neurips.cc/paper\_ files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets\_and\_Benchmarks.pdf

[49] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using siamese BERT-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2019, pp. 3982–3992. [Online]. Available: https://aclanthology.org/D19-1410/

[50] A. Parrish et al., “BBQ: A hand-built bias benchmark for question answering,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 2086–2105. [Online]. Available: https://aclanthology.org/2022.findings-acl.165/

## Appendix

## A Task Requirement Schema and Profiler Training

## A.1 Capability Definitions

The requirement schema expresses task demand through capability weights and a difficulty level. Its ten nonnegative weights sum to one, allowing several capabilities to contribute to the same request. Task profiles and model measurements use the following capabilities in a fixed order.

C1. Task understanding and constraint adherence. Identifying explicit and implicit task goals, interpreting instructions and conversational intent, and satisfying substantive constraints, priorities, procedures, and output requirements.

C2. Knowledge understanding and domain-specific question answering. Understanding and applying factual, commonsense, encyclopedic, and domain-specific knowledge to answer questions accurately.

C3. Reasoning and quantitative problem solving. Drawing deductive or inductive inferences, performing quantitative calculations, constructing proofs, planning, analyzing causal relations, and decomposing complex problems.

C4. Programming and software engineering. Generating, understanding, debugging, and testing code, implementing algorithms, and reasoning about software structure, repositories, and programming interfaces.

C5. Complex information understanding and long-context synthesis. Locating and reconciling evidence in complex documents, long texts, or extended conversational context, with emphasis on integration across passages and sources.

C6. Data analysis and structured information understanding. Understanding tables, statistics, and other structured representations; extracting, comparing, calculating, interpreting, or transforming their contents.

Table 8 Sources retained for profiler SFT after excluding datasets in the routing test workload. The groups indicate sampling coverage; each task receives its own capability mixture and difficulty annotation.

<table><tr><td>Capability coverage</td><td>Retained sources</td></tr><tr><td>C1: Task understanding</td><td>IFEval [7], EmoryNLP, MELD</td></tr><tr><td>C2: Domain knowledge</td><td>MedQA, C-Eval</td></tr><tr><td>C3: Reasoning</td><td>MATH-500, MathBench, KOR-Bench, K&amp;K (kandk), BBH [8]</td></tr><tr><td>C4: Software engineering</td><td>MBPP, HumanEval [44]</td></tr><tr><td>C5: Long-context synthesis</td><td>LongBench v2 [46]</td></tr><tr><td>C6: Structured information</td><td>FinQA [45]</td></tr><tr><td>C7: Multilingual tasks</td><td>INCLUDE, Global-MMLU, FLORES-200 [29]</td></tr><tr><td>C8: Content generation</td><td>WritingBench [30]</td></tr><tr><td>C9: Interactive execution</td><td>GAIA, WebArena [31]</td></tr><tr><td>C10: Risk control</td><td>HarmBench [32], StrongREJECT, XSTest, BBQ [50]</td></tr></table>

C7. Multilingual understanding, translation, and localization. Understanding and expressing content across languages, translating meaning, and adapting wording to linguistic and cultural contexts.

C8. Content generation and expression control. Generating, summarizing, rewriting, or expanding content while controlling organization, style, tone, and audience suitability.

C9. Tool use and interactive task execution. Executing tasks through retrieval, code execution, function calls, or interaction with external systems, including coordination across multiple steps.

C10. Safety, reliability, and risk control. Recognizing safety-sensitive requirements, handling consequential advice reliably, protecting privacy, and maintaining appropriate response and refusal boundaries.

## A.2 Difficulty Levels

Difficulty specifies the required level of knowledge, reasoning, constraint interaction, and execution complexity. L1 comprises direct facts and simple transformations; L2 covers bounded reasoning and standard knowledge application; L3 includes substantial multistep reasoning, complex code analysis, or synthesis; and L4 covers expert analysis, extended planning, or tasks that combine several demanding capabilities. These levels apply to the task as a whole, alongside its capability mixture. The introductory illustration in the main paper abbreviates C4, C3, and C6 as coding, reasoning, and data analysis, respectively.

## A.3 Training Sources and Test-Set Separation

The profiler training corpus is constructed for coverage of the ten capability dimensions, with GPT-5.4 providing structured requirement annotations. Source selection begins with 33 datasets. We exclude all nine sources that also occur in the routing test workload: AIME, LiveMathBench, LiveCodeBench, GPQA, HLE, ArenaHard, MMLU-Pro, SWE-bench, and $\tau ^ { 2 } { \mathrm { - b e n c h } }$ . Exclusion applies to each source dataset as a whole. The remaining 24 sources are organized in Table 8. This grouping guides sampling coverage; each retained task receives its own capability mixture and difficulty annotation.

After source-level exclusion, token-level 3-gram comparison checks the remaining candidate samples against routing test samples. This additional filtering addresses residual text overlap, which can distort language-model evaluation [47]. Let $G _ { 3 } ( v )$ be the collection of token 3-grams extracted from sample v. For a candidate training sample v and a test sample $t ,$ the overlap ratio is

$$
O _ { 3 } ( v , t ) = { \frac { | G _ { 3 } ( v ) \cap G _ { 3 } ( t ) | } { | G _ { 3 } ( t ) | } } .\tag{10}
$$

A candidate is filtered if $O _ { 3 } ( v , t ) \geq 0 . 1 0$ for any compared test sample. The denominator is the 3-gram count of the test sample, so the threshold measures how much of that sample is covered by the candidate. This stage found little residual overlap after dataset exclusion. The resulting supervision comprises approximately 21,000 task-profile pairs.

## A.4 Full-Parameter Supervised Fine-Tuning

We fine-tune Qwen3-0.6B [42] with LlamaFactory [43]. Each input contains the profiling instructions and the request with its supplied context. The target is a structured record containing all ten capability weights and one difficulty level. Annotation instructions assess the capabilities necessary for successful completion from the request and supplied context, with reference answers, candidate identities, and source-dataset labels excluded from the annotation input. The training objective is conditional token cross-entropy over the target profile; instruction and request tokens are masked from the loss. The supervised output thus describes the demands of the task.

Training updates all model parameters for one epoch on two NVIDIA A800 GPUs using bfloat16 and DeepSpeed ZeRO-3. The learning rate is $1 0 ^ { - 5 }$ with cosine decay and a 0.1 warmup ratio. A per-device batch size of one and 16 gradient-accumulation steps give an effective batch size of 32. The sequence limit is 12,000 tokens, and the qwen3\_nothink template disables thinking output. The validation fraction is 0.002, evaluation occurs every 20 steps, and the best checkpoint is loaded at the end. Inference uses deterministic decoding. An auxiliary sample-category prediction supplies the category used to retrieve empirical mean scores from routing training outcomes.

## A.5 Service Metadata and Eligibility

Execution requirements are obtained from request processing and deployment metadata, including context length, attachment modalities, and required tool support. Eligibility checks compare these requirements with availability and configured service attributes. Optional limits on price rates and first-token latency use the available metadata. Missing values do not independently exclude a candidate. The generation-cost preference is applied separately as a soft selection preference.

## B Model Profile Construction and Scoring Details

## B.1 Outcome Profiles and Missing-Cell Handling

Outcome profiles aggregate observed task scores according to capability weights and profiler-assigned difficulty, as defined in the main paper. Both successes and failures contribute, and graded scores retain their original scale. Because one task can require several capabilities, its outcome may inform several cells. Each cell describes performance on tasks that demand the corresponding capability at a specified difficulty.

For an unobserved cell, lookup uses the nearest observed difficulty level and prefers the higher level in a tie. If a capability has no observed level, it contributes zero. The other capability weights retain their original values rather than being renormalized.

## B.2 Response Assessment Profiles

The assessment channel follows the general use of LLMs to evaluate response quality [48]. Its reference bank is sampled from the routing training partition and contains the available model responses for those requests. DeepSeek-V4-Flash annotates item requirements, and GPT-5.4 assesses responses against references and task rubrics. The requirement annotations share the capability schema of the deployed profiler but assign their weights independently. The capability-specific rubrics and aggregation procedure below define the response-assessment channel used by RouteAlign.

Let $w _ { i k }$ denote the annotated requirement weight of item i for capability $k ,$ and $z _ { i m k } \in [ 0 , 1 0 ]$ its assessment score for model m. The index set $\mathcal { I } _ { m , k , \ell }$ contains valid assessments assigned to difficulty ℓ with positive requirement weight, evaluator relevance above 0.5, and a nonnegative response score. For nonempty support,

$$
A _ { m , k , \ell } ^ { \mathrm { j u d g e } } = \frac { \sum _ { i \in \mathcal { I } _ { m , k , \ell } } w _ { i k } z _ { i m k } } { \sum _ { i \in \mathcal { I } _ { m , k , \ell } } w _ { i k } } .\tag{11}
$$

The two measurement channels retain their native scales: [0, 1] for task outcomes and [0, 10] for response assessments. Separate learned coefficients combine them with the other evidence in the quality scorer. The common coordinates align both channels with task demand while preserving their distinct measurement scales.

## B.3 Retrieval and Feature Definitions

Retrieval uses sentence embeddings to compare tasks semantically [49]. We use the frozen al $1 - M i n i L M - L 6 - v 2$ checkpoint and normalize its 384-dimensional outputs. Each neighborhood contains $k _ { \mathrm { n n } } = 1 5 0$ training queries, with weights proportional to $\exp ( - \beta \delta ( x , x _ { j } ) )$ for cosine distance $\delta$ and $\beta = 5$ . Weights sum to one over the neighborhood. The same retrieved records supply scalar quality, capability-conditioned quality, and generation-cost estimates. A capability estimate falls back to scalar neighborhood quality when its weighted support is at most $1 0 ^ { - 6 }$

Write $p _ { m } = p _ { \mathrm { n n } } ( x , m )$ for scalar local quality, $b ( x , m )$ for the training mean in the predicted sample category, and $g ( x , m )$ for the mean in that category at difficulty $d ( x )$ . When the category-by-difficulty stratum is empty, $g ( x , m )$ falls back to $b ( x , m )$ . Together with the compatibility scores defined in the main paper, these quantities form the ten-feature vector

$$
\boldsymbol { f } ( \boldsymbol { x } , m ) = \left[ \boldsymbol { Q } ^ { \mathrm { o u t } } , \boldsymbol { Q } ^ { \mathrm { j u d g e } } , g , b , p _ { m } , \boldsymbol { Q } ^ { \mathrm { l o c a l } } , p _ { m } D , \delta _ { m } , \eta _ { m } , \left| p _ { m } - g \right| \right] ^ { \top } .\tag{12}
$$

Arguments $( x , m )$ are suppressed on the right-hand side. The margin $\delta _ { m }$ subtracts the highest local estimate among other eligible candidates from $p _ { m } ;$ the reference value is zero when only one candidate is eligible. The indicator $\eta _ { m }$ records whether a candidate attains the highest local estimate. Selection ties follow a deterministic candidate order.

Before normalization, candidate separation is the maximum local quality minus its mean across eligible candidates. The benchmark divides this statistic by its mean over the corresponding feature-extraction batch to obtain $D ( x )$ , separately for training and evaluation inputs. All constituent quality estimates come from training-partition evidence. The normalization therefore depends on the requests included in the feature-extraction batch; the serving endpoint uses the request-level configuration in Section E.

## B.4 Scorer Fitting and Missing Records

Ridge regression uses penalty $\alpha = 1$ , native feature scales, and no intercept, with the recorded task score as the target for each training query-model pair. Retrieval features use leave-one-out neighborhoods. Global profiles and category priors use the complete training partition.

Absent records receive zero in dense retrieval scores, generation costs, and regression targets. Global outcome profiles aggregate observed records only, so a missing response contributes no observation to a profile cell. Feature ablations retain these accounting rules.

## C Dataset Details and Baseline Implementations

This section reports the evaluation split, response coverage, and baseline configurations used in our experiments. Citations identify the source methods; the implementation choices and fitting budgets specify the comparisons reported in the main paper.

## C.1 Train-Test Split

Table 9 reports the shared split obtained with seed 42 and a 70% training fraction. All responses to a request remain in the same partition. Outcome profiles, response-assessment items, conditional means, retrieval references, and regression targets use training records. A policy selects a candidate before accessing the corresponding test outcome.

## C.2 Candidate Coverage and Selection Counts

Table 10 records available test responses and RouteAlign selections in the fixed catalog. Sample-Avg averages scores over requests, whereas Dataset-Avg weights dataset means equally. The benchmark adapter rounds dataset means to two decimals. Both metrics retain graded task scores.

Direct-candidate summaries use available records and datasets. GPT-5 covers all 3,738 test requests, GPT-5- Chat has no $\tau ^ { 2 } .$ -bench records, and several other candidates have one or two missing records. Every RouteAlign

Table 9 Training and test set sizes.
<table><tr><td>Dataset</td><td>Train</td><td>Test</td></tr><tr><td>AIME</td><td>42</td><td>18</td></tr><tr><td>ArenaHard</td><td>525</td><td>226</td></tr><tr><td>GPQA</td><td>138</td><td>60</td></tr><tr><td>HLE</td><td>1511</td><td>648</td></tr><tr><td>LiveCodeBench</td><td>738</td><td>317</td></tr><tr><td>LiveMathBench</td><td>84</td><td>37</td></tr><tr><td>MMLU-Pro</td><td>2100</td><td>900</td></tr><tr><td>SimpleQA</td><td>3028</td><td>1298</td></tr><tr><td>SWE-bench</td><td>350</td><td>150</td></tr><tr><td>τ2-bench</td><td>194</td><td>84</td></tr><tr><td>Total</td><td>8,710</td><td>3,738</td></tr></table>

Table 10 Candidate coverage and fixed-catalog selection counts. The selected candidates have observed outcomes for all 3,738 test requests.
<table><tr><td>Candidate checkpoint</td><td>Records</td><td>Datasets</td><td>Selections</td></tr><tr><td>intern-s1</td><td>3737</td><td>10</td><td>2</td></tr><tr><td>deepseek-v3-0324</td><td>3737</td><td>10</td><td>12</td></tr><tr><td>deepseek-v3.1-terminus</td><td>3737</td><td>10</td><td>37</td></tr><tr><td>deepseek-r1-0528</td><td>3737</td><td>10</td><td>181</td></tr><tr><td>qwen3-235b-a22b-2507</td><td>3737</td><td>10</td><td>807</td></tr><tr><td>qwen3-235b-a22b-thinking-2507</td><td>3737</td><td>10</td><td>89</td></tr><tr><td>kimi-k2-0905</td><td>3737</td><td>10</td><td>4</td></tr><tr><td>g1m-4.6</td><td>3737</td><td>10</td><td>14</td></tr><tr><td>gpt-5-chat</td><td>3653</td><td>9</td><td>1</td></tr><tr><td>gemini-2.5-flash</td><td>3737</td><td>10</td><td>2</td></tr><tr><td>claude-sonnet-4</td><td>3736</td><td>10</td><td>137</td></tr><tr><td>gemini-2.5-pro</td><td>3737</td><td>10</td><td>600</td></tr><tr><td>gpt-5</td><td>3738</td><td>10</td><td>1852</td></tr></table>

selection has an observed response, producing a total task score of 2,357.5 and generation expenditure of \$110.4199 before rounding. AVG LLM in the main paper averages the 13 individual-candidate summaries. Random Router selects uniformly from the candidates for each request using seed 42.

## C.3 Preference-Based and Contrastive Routers

RouteLLM [10] uses matrix factorization with frozen query embeddings and learned candidate vectors, fitted on 50,000 preference pairs. Our multi-candidate adaptation selects the highest candidate score; the original method applies a threshold to choose between a stronger and a weaker model. RouterDC [11] uses mDeBERTav3-base, candidate embeddings, sample-model and sample-sample contrastive objectives, KMeans semantic groups, and 6,000 training examples; inference selects by cosine similarity. Neither evaluated inference rule applies a separate monetary penalty.

## C.4 Graph-Based and Cluster-Based Routers

GraphRouter [33] represents tasks, queries, and candidates as nodes and uses two edge-aware graph convolution layers with masked-edge prediction. Its three configurations construct training targets from cost-oriented, balanced, and quality-oriented utility, respectively. AvengersPro [36] summarizes quality and efficiency within query clusters. Our evaluated implementation uses balance parameter 0.5 and aggregates reciprocal candidate ranks from nearby clusters using exponential distance weights. This is the rank-based adaptation of cluster utility scoring used in the reported comparison.

## C.5 Classification-Based and Retrieval-Based Routers

The classification baselines follow the KNNRouter, SVMRouter, and MLPRouter families provided by LLMRouter [25]. In our evaluation, they use frozen Longformer embeddings and labels identifying the highest-scoring training candidate. Their decision rules are neighbor-label voting, RBF-kernel classification, and a feed-forward classifier, respectively. Neighbor-label voting predicts a candidate identity; neighborhood performance estimation [12] instead averages scores separately for each candidate. This distinction separates the KNNRouter comparison from the local quality estimates used by RouteAlign.

The three embedding-based retrieval policies use the llm-semantic-router checkpoints multi-modal-embed-small, multi-modal-embed-large, and mmbert-embed-32k-2d-matryoshka. MM-Embed-S and MM-Embed-L use 384- and 768-dimensional text embeddings, respectively. The mmBERT-32K configuration uses 768 dimensions, all 22 layers, and a 2,048-token input limit. Each policy averages candidate scores over cosine neighbors and breaks ties using lower neighbor cost. All comparison policies select before response generation under the same benchmark partition.

## D Feature Ablations and Selection by Category Means

## D.1 Feature Groups and Removal Rules

The main paper includes every evaluated feature and cost-preference ablation. Feature ablations delete specified columns from Equation (12) and refit ridge regression on the same training partition with leave-oneout retrieval features. The encoder and definitions of retained features remain fixed, and global profiles and category priors continue to use the full training partition. The grouped removals are:

• All local quality evidence: $p _ { m } , Q ^ { \mathrm { l o c a l } } , p _ { m } D , \delta _ { m } , \eta _ { m }$ , and $| p _ { m } - g |$ . The local generation-cost estimate remains available.

• Relative local evidence: $p _ { m } D , \delta _ { m } , \eta _ { m }$ , and $| p _ { m } - g |$

• Requirement-conditioned scores: $Q ^ { \mathrm { o u t } } , Q ^ { \mathrm { j u d g e } } , g , \mathrm { a n d } Q ^ { \mathrm { l o c a l } }$

• Global profile compatibility: $Q ^ { \mathrm { o u t } }$ and $Q ^ { \mathrm { j u d g e } }$

Individual ablations of the scalar neighborhood estimate, sample-category prior, local capability conditioning, response-assessment compatibility, category-by-difficulty prior, and outcome compatibility delete $p _ { m } , b , Q ^ { \mathrm { l o c a l } }$ $Q ^ { \mathrm { j u d g e } } , \ g$ , and $Q ^ { \mathrm { o u t } }$ , respectively. Derived columns that remain in the feature vector keep their original definitions. In particular, deleting $p _ { m }$ retains the four relative local features, and deleting g retains $| p _ { m } - g |$ These ablations measure the conditional contribution of the designated input columns.

Refitting redistributes predictive weight across retained estimates, so changes for overlapping removal groups are conditional on the remaining features. The cost-preference ablation retains the fitted quality estimator and sets $\lambda = 0$

## D.2 Category-Based Selection with and without Difficulty Conditioning

These comparisons test whether empirical means for task categories are sufficient for selection. Using the same training evidence, the two reference policies select the candidate with the highest sample-category mean or category-by-difficulty mean, respectively, without the shared scorer. Table 11 compares these rules with the complete RouteAlign policy.

Conditioning on both category and difficulty yields 66.42% Dataset-Avg and 61.45% Sample-Avg at \$110.68 in generation charges. RouteAlign attains gains of 2.22 and 1.62 percentage points, respectively, at \$110.42. Selection by category alone yields 66.35% Dataset-Avg and 61.49% Sample-Avg at \$126.50. The complete policy therefore achieves higher quality than either rule based on category means, including the category-by-difficulty rule at similar observed generation cost.

These results support integrating complementary evidence beyond category means. The main-paper ablations examine how individual feature groups contribute within the fitted combination.

Table 11 Comparison with selection by category means. Each reference policy selects the candidate with the highest empirical mean score in the predicted category or category-and-difficulty group, without the shared scorer. Quality is in percent.
<table><tr><td>Selection rule</td><td>D-Avg.</td><td>S-Avg.</td><td>Cost ($)</td></tr><tr><td>Full RouteAlign</td><td>68.64</td><td>63.07</td><td>110.42</td></tr><tr><td>Sample-category mean</td><td>66.35</td><td>61.49</td><td>126.50</td></tr><tr><td>Category and difficulty mean</td><td>66.42</td><td>61.45</td><td>110.68</td></tr></table>

## E Runtime Configurations and Measurements

## E.1 Selection with Cached Requirements and Embeddings

The runtime table in the main paper measures selection over 3,738 test requests. RouteAlign reuses cached requirement profiles and embeddings; the measured operations are retrieval, feature construction, and candidate scoring. Offline preparation, requirement prediction, embedding construction, and candidate answer generation are excluded. The reported generation charges cover candidate responses and exclude the execution cost of this selection stage.

## E.2 Endpoint Latency with Online Requirement Prediction

The separate endpoint runs the 0.6B requirement profiler and selects candidates using a 500-request reference bank. It computes candidate separation independently for each request as the unnormalized maximum local quality minus the mean. When sample-category metadata is absent, difficulty-level and global reference means replace the category-by-difficulty and category means, respectively. These settings define the request-level serving configuration.

The reported run processes 96 requests at concurrency 32. Timing covers requirement prediction and selection, ending before candidate answer generation. The resulting median latency is 0.60 seconds, the 95th-percentile latency is 0.74 seconds, and throughput is 51.2 requests per second, with no observed errors or fallbacks.

## F Scorer Reuse under Catalog Changes

Figure 5 presents the catalog measurements reported in the main paper. Panel (a) follows sequential additions and replacement; panel (b) separates removal and repricing into independent branches. The figure complements the update protocol and result table by showing how each event changes selection quality under the specified adaptation procedures.

Along T0 through T6, RouteAlign-Frozen retains the coefficients fitted on the seven T0 candidates. Newly admitted candidates are evaluated on training-partition reference tasks to obtain their measurements and profiles. Dataset-Avg increases from 60.36% at T0 to 68.55% after GPT-5 is added at T5, and to 68.60% after the replacement at T6. These changes show that candidate measurements can be incorporated within the existing requirement coordinates and feature definitions while retaining the learned comparison rule.

The two events in panel (b) are measured against T6. Removing GPT-5 at T7 lowers RouteAlign-Frozen by 4.52 percentage points to 64.08%, quantifying the effect of withdrawing this candidate. Halving the price of Gemini 2.5 Flash at T8 raises Dataset-Avg by 0.04 points to 68.64%, with GPT-5 retained in the catalog.

RouteLLM and RouterDC retain checkpoints trained on all 13 candidates and mask unavailable outputs. RouteAlign-Frozen is fitted only on T0, while AvengersPro recomputes clusters and rankings for the current catalog. New candidate measurements come from reference evaluations performed before those candidates enter selection. All RouteAlign-Frozen results, including T8, use the T0-fitted coefficients; the fixed-catalog experiment uses coefficients fitted on all 13 candidates.

## G Decision-Model Implementations and Detailed Results

![](images/b9fe30d24082c073c01175b0937c770f43d1770b98f25fa76b445423c3b0f1be.jpg)

(b) Independent events  
![](images/cc4e272c0b92686bdfea1ed26b726f7921879d1b02ce322bc76fc0593c54d8b9.jpg)  
Figure 5 Selection quality under controlled catalog updates. (a) Dataset-Avg along T0 through T6, with candidate counts in parentheses. (b) Changes in percentage points for the independent T7 removal and T8 repricing branches relative to T6. RouteAlign-Frozen retains the scorer fitted on the seven T0 candidates and refreshes candidate evidence after each update.

Table 12 Open implementations used in the Jev-style comparison. Project names link to their repositories, and checkpoint identifiers specify the evaluated Qwen3.5-based models. SemIf and OpenJev-mjdileep use pretrained base checkpoints directly.
<table><tr><td>Project</td><td>Sizes</td><td>Decision mechanism</td><td>Checkpoint identifiers</td></tr><tr><td>Kev</td><td>0.8B, 4B, 9B</td><td>LoRA and pointer head</td><td>jaredpalmer/kev-0.8b, jaredpalmer/kev-4b, jaredpalmer/kev-9b</td></tr><tr><td>Open-Jev</td><td>2B, 9B</td><td>LoRA and scalar head</td><td>ZefanCai/Open-Jev-2B, ZefanCai/Open-Jev-9B</td></tr><tr><td>Decider</td><td>2B, 4B</td><td>Full fine-tuning and letter log- its</td><td>Mapika/decider-2b, Mapika/decider-4b</td></tr><tr><td>SemIf</td><td>4B</td><td>Pretrained letter logits</td><td>Qwen/Qwen3.5-4B</td></tr><tr><td>OpenJev-mjdileep</td><td>0.8B</td><td>Pretrained affirmative-answer logits</td><td>Qwen/Qwen3.5-0.8B</td></tr></table>

## G.1 Backend Implementations

The comparison uses local adaptations of the five open projects in Table 12, instantiated at nine model sizes. Kev uses LoRA adapters and a pointer head, Open-Jev uses LoRA with a scalar decision head, and Decider uses fully fine-tuned models with option-letter logit readout. SemIf and OpenJev-mjdileep read pretrained-model logits. The experiments use the Qwen3.5-based versions listed below. All nine backends are evaluated without additional fine-tuning on the routing workload.

The local adapter follows the state-and-questions structure of the Jev decision interface. Open-Jev uses a locally constructed candidate prompt following the published scoring interface; Decider follows its model-card protocol. Kev processes question branches as separate causal sequences in a batch to accommodate the recurrent components of its backbone. Its state and combined state-question limits are 384 and 1,024 tokens, respectively. Released temperature parameters are loaded with the checkpoints.

## G.2 Requirement Prediction and Candidate Scoring

One decision call contains 12 questions: ten four-level capability-demand scores, one L1–L4 difficulty choice, and one tool-requirement judgment. Each capability score is its expected ordinal level divided by three. Values below 0.05 are removed before the remaining weights are normalized. Tool use is required when its affirmative probability exceeds 0.5. The adapter then applies metadata filtering and matches candidates by the primary capability and predicted difficulty, using its configured tier fallback.

Table 13 All scoring mixtures for the open Jev-style backends. D and S denote Dataset-Avg and Sample-Avg (%); cost is in US dollars. The last column is mean requirement-prediction and scoring time on one A800, shared by the three offline replays.
<table><tr><td></td><td colspan="3"> $\rho = 0$ </td><td colspan="3"> $\rho = 0 . 5$ </td><td colspan="3"> $\rho = 1$ </td><td></td></tr><tr><td>Backend</td><td>D</td><td>S</td><td>Cost</td><td>D</td><td>S</td><td>Cost</td><td>D</td><td>S</td><td></td><td>Cost Time (ms)</td></tr><tr><td>Kev-0.8B</td><td>48.08</td><td>47.14</td><td>36.58</td><td>51.39</td><td>49.02</td><td>38.11</td><td>51.83</td><td>49.65</td><td>39.33</td><td>1268.9</td></tr><tr><td>OpenJev-mjdileep-0.8B</td><td>48.76</td><td>45.77</td><td>48.61</td><td>49.86</td><td>45.80</td><td>51.56</td><td>47.44</td><td>45.85</td><td>52.57</td><td>969.7</td></tr><tr><td>Open-Jev-2B</td><td>47.70</td><td>46.54</td><td>50.21</td><td>51.78</td><td>49.83</td><td>49.90</td><td>48.93</td><td>48.86</td><td>39.52</td><td>1544.0</td></tr><tr><td>Decider-2B</td><td>56.42</td><td>46.63</td><td>123.39</td><td>56.11</td><td>47.27</td><td>108.88</td><td>55.00</td><td>47.11</td><td>95.63</td><td>593.4</td></tr><tr><td>SemIf-4B</td><td>49.35</td><td>51.47</td><td>30.99</td><td>54.29</td><td>52.62</td><td>33.55</td><td>53.43</td><td>51.82</td><td>29.32</td><td>1727.8</td></tr><tr><td>Kev-4B</td><td>60.78</td><td>54.70</td><td>105.57</td><td>61.85</td><td>55.35</td><td>105.63</td><td>61.43</td><td>55.43</td><td>102.97</td><td>1665.8</td></tr><tr><td>Decider-4B</td><td>62.20</td><td>57.85</td><td>122.28</td><td>61.23</td><td>57.45</td><td>113.84</td><td>60.61</td><td>57.08</td><td>109.32</td><td>1727.5</td></tr><tr><td>Open-Jev-9B</td><td>48.78</td><td>45.65</td><td>79.49</td><td>50.20</td><td>46.11</td><td>65.00</td><td>53.44</td><td>47.61</td><td>44.66</td><td>3212.7</td></tr><tr><td>Kev-9B</td><td>59.48</td><td>55.93</td><td>121.22</td><td>61.00</td><td>56.45</td><td>110.96</td><td>60.50</td><td>56.51</td><td>111.32</td><td>1713.4</td></tr></table>

Candidate scoring receives the query truncated to 1,200 characters and a task summary. Each candidate description contains up to five capabilities selected by current demand, their difficulty-specific measurements, specialty tags, and service attributes, within a 700-character profile budget. The decision score maps the expected four-level performance rating to [0, 10]. For decision score $J ( x , m )$ and explicit profile compatibility $C ( x , m )$ , the final score is

$$
H _ { \rho } ( x , m ) = \rho J ( x , m ) + ( 1 - \rho ) C ( x , m ) , \qquad \rho \in \{ 0 , 0 . 5 , 1 \} .\tag{13}
$$

Here $C ( x , m )$ sums the adapter’s profile measurements weighted by the backend-predicted requirements at the predicted difficulty. At $\rho = 0$ , compatibility supplies the candidate score, while the backend determines the task requirements and resulting candidate set. The mixtures combine backend predictions with supplied global profile measurements; RouteAlign additionally uses retrieved local outcomes and its fitted ten-feature scorer. The shared workload and catalog support comparison of these complete selection policies.

The experiment saves both component scores and replays the three mixtures over the same candidate sets. Outcomes and generation charges are looked up only after selection. Every backend covers all 3,738 test requests, including successfully retried records. The implementation uses PyTorch 2.9.1, Transformers 5.3.0, PEFT 0.20.0, and bfloat16 inference on one A800 GPU.

## G.3 Complete Mixture Results

Table 13 provides quality and generation expenditure for all 27 configurations. The best configuration under both aggregate metrics is Decider-4B with explicit compatibility scoring. RouteAlign improves on it by 6.44 Dataset-Avg points and 5.22 Sample-Avg points at 9.7% lower generation expenditure. Intermediate mixtures help some backends, including Open-Jev-2B and SemIf-4B, while Decider-4B performs best at $\rho = 0$ RouteAlign’s aggregate advantage is sustained across this variation in backend and scoring mixture.

## G.4 Performance across Task Groups

Table 14 disaggregates direct decision scoring. RouteAlign achieves the highest score on seven of the ten datasets in this comparison, including information-intensive, code-generation, and interactive workloads. Decider is stronger on AIME, Kev-9B on $\mathrm { G P Q A }$ , and Kev-4B on SWE-bench. The aggregate advantage reflects gains across several task groups, complementing the comparison across scoring mixtures in Table 13.

Table 14 Per-dataset quality (%) with direct decision scoring (ρ = 1). Bold marks the highest score in each column among these configurations and RouteAlign.
<table><tr><td>Backend or policy</td><td>AIME</td><td>AH</td><td>GPQA</td><td>HLE</td><td>LCB</td><td>LMB</td><td>MMP</td><td>SQA</td><td>SWE</td><td> $\tau ^ { 2 }$ </td></tr><tr><td>Kev-0.8B</td><td>61.11</td><td>69.03</td><td>73.33</td><td>8.80</td><td>59.94</td><td>54.05</td><td>82.33</td><td>43.76</td><td>20.67</td><td>45.24</td></tr><tr><td>OpenJev-mjdileep-0.8B</td><td>55.56</td><td>69.03</td><td>66.67</td><td>12.35</td><td>56.15</td><td>62.16</td><td>82.67</td><td>34.05</td><td>16.67</td><td>19.05</td></tr><tr><td>Open-Jev-2B</td><td>55.56</td><td>67.92</td><td>63.33</td><td>8.18</td><td>59.94</td><td>62.16</td><td>81.67</td><td>44.30</td><td>15.33</td><td>30.95</td></tr><tr><td>Decider-2B</td><td>88.89</td><td>70.35</td><td>70.00</td><td>15.43</td><td>70.98</td><td>62.16</td><td>83.22</td><td>29.43</td><td>22.67</td><td>36.90</td></tr><tr><td>SemIf-4B</td><td>66.67</td><td>65.04</td><td>78.33</td><td>9.26</td><td>58.36</td><td>59.46</td><td>83.33</td><td>50.46</td><td>8.67</td><td>54.76</td></tr><tr><td>Kev-4B</td><td>83.33</td><td>73.89</td><td>81.67</td><td>13.89</td><td>77.29</td><td>64.86</td><td>83.22</td><td>49.08</td><td>34.67</td><td>52.38</td></tr><tr><td>Decider-4B</td><td>88.89</td><td>68.36</td><td>80.00</td><td>17.13</td><td>82.33</td><td>64.86</td><td>84.22</td><td>53.24</td><td>20.67</td><td>46.43</td></tr><tr><td>Open-Jev-9B</td><td>55.56</td><td>72.79</td><td>75.00</td><td>9.26</td><td>53.63</td><td>70.27</td><td>83.22</td><td>36.44</td><td>24.67</td><td>53.57</td></tr><tr><td>Kev-9B</td><td>72.22</td><td>74.12</td><td>85.00</td><td>15.28</td><td>76.03</td><td>70.27</td><td>83.22</td><td>52.70</td><td>27.33</td><td>48.81</td></tr><tr><td>RouteAlign</td><td>83.33</td><td>78.10</td><td>81.67 26.08</td><td></td><td>83.60</td><td>81.08</td><td>86.89</td><td>58.47</td><td>33.33</td><td>73.81</td></tr></table>

AH: ArenaHard; LCB: LiveCodeBench; LMB: LiveMathBench; MMP: MMLU-Pro; SQA: SimpleQA; SWE: SWE-bench.

## G.5 Backend Runtime

The final column of Table 13 measures sequential request processing on one A800, covering requirement prediction and candidate scoring before answer generation. The observed means range from 593.4 to 3,212.7 milliseconds. Runtime is recorded for the underlying inference run and shared by the three offline replays of its saved component scores. Section E reports RouteAlign’s cached selection and concurrent serving-endpoint measurements, with the operations and serving configuration specified for each.