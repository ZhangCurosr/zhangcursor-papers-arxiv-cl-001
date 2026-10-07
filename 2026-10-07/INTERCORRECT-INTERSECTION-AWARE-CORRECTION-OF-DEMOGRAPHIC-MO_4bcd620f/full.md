# INTERCORRECT: INTERSECTION-AWARE CORRECTION OF DEMOGRAPHIC MODELMERGING FOR FAIR ASR

Ashley E. Bravo-Bravo<sup>1</sup>, Yuchen Zhang<sup>2</sup>, Haralambos Mouratidis<sup>2</sup>, Ravi Shekhar<sup>3</sup>, Monorama Swain<sup>4</sup>

<sup>1</sup>PUCP <sup>2</sup>IADS, University of Essex <sup>3</sup>CSEE, University of Essex <sup>4</sup>JKU Linz ashley.bravo@pucp.edu.pe, {yuchen.zhang, h.mouratidis, r.shekhar}@essex.ac.uk, monorama.swain@jku.at

## ABSTRACT

Automatic Speech Recognition (ASR) systems often show uneven performance across demographic groups, and errors can be especially difficult to address for speakers belonging to multiple demographic groups. This work studies demographic-aware model merging for fair Speech-LLMbased ASR. Starting from a SLAM-ASR-based model, we fine-tune only the connector on demographic-specific subsets and merge the resulting subgroup-adapted connectors into a global model. We then identify critical cross-axis demographic pairs using subgroup WER and task-vector conflict, and apply intersection-specific correction vectors to the global merged model. Experiments on Fair-Speech show that global demographic merging improves overall WER over the base model, while intersection correction provides additional gains for several merging strategies. In particular, TIES with WER-based correction achieves the best overall WER, reducing it from 7.38% to 5.13%. Subgroup and disparity analyses further show that the proposed approach improves performance across demographic axes, while highlighting that lower average WER does not always imply reduced subgroup disparity.

Index Terms— ASR, Speech-LLM, Fair Speech, Model Merging

## 1. INTRODUCTION

Automatic Speech Recognition (ASR) systems can exhibit substantial performance disparities across demographic groups, including age, gender, accent, ethnicity, socioeconomic background, and first language [1, 2, 3, 4, 5]. These disparities limit the reliability of ASR systems for speakers whose speech patterns are underrepresented or poorly modeled. Prior work has explored several strategies for reducing demographic disparities in ASR, including demographic balancing, data augmentation, fine-tuning, contrastive learning, and adapter fusion [2, 6, 7, 8, 9]. These methods can reduce subgroup performance gaps, but they can depend on additional data, method-specific losses, explicit data balancing, or additional adaptation modules. Moreover, demographic attributes are not mutually exclusive, a speaker may belong to multiple demographic groups at the same time. As a result, improvements for individual demographic groups may not necessarily transfer to speakers at the intersection of multiple groups.

Model merging offers a complementary way to combine knowledge from separately fine-tuned models. Instead of jointly retraining on all adaptation data, model merging operates directly on model parameters and has shown promising results in ASR adaptation, including dysarthric speech recognition, child ASR, multi-domain ASR, and multilingual speech translation [10, 11, 12, 13]. Existing ASR model-merging studies focus primarily on robustness and task adaptation, while fairness-oriented ASR research has largely overlooked parameter-level merging as a mechanism for demographic bias mitigation. This gap motivates a central question of whether model merging can be used not only for ASR adaptation, but also to mitigate demographic and intersectional disparities.

We address this question using a Speech-LLM-based ASR model. First, we train a base model with all demographic data. Second, we build a global demographic model by fine-tuning the connector for each demographic subgroup and merging the resulting subgroup-specific models. Third, we build an intersection-corrected model by selecting crossaxis demographic pairs and adding their correction vectors to the global demographic model. This allows us to test whether demographic model merging improves ASR performance across groups, and whether intersection-specific correction can address cases that subgroup-level adaptation alone may miss. To our knowledge, this is the first study to connect model merging with demographic and intersectional fairness in Speech-LLM-based ASR. Our experiments on the Fair-Speech dataset show that global demographic merging consistently improves over the base model without using additional training data. Intersection-aware correction provides further gains, with WER-guided TIES reducing overall WER from 7.38% to 5.13%, a 30.5% relative reduction, while improving targeted intersections.

## 2. METHODOLOGY

## 2.1. Base ASR Model Training

Our base model follows the SLAM-ASR architecture [14, 15], which connects a speech encoder to an LLM through a lightweight connector. Specifically, we use Whisper-largev3 [16] as the encoder, TinyLlama 1.1B [17] as the LLM, and a linear connector to align the speech encoder outputs with the LLM embedding space. The speech encoder and LLM are kept frozen, and only the linear connector is trained. The base model $f _ { \theta _ { \mathrm { b a s e } } }$ is trained on the full training set $\mathcal { D } _ { \mathrm { t r a i n } } ,$ where $\theta _ { \mathrm { b a s e } }$ denotes the learned connector parameters.

## 2.2. Global Demographic Model Merging

Let $\mathcal { G } ~ = ~ \{ g _ { 1 } , g _ { 2 } , . . . , g _ { N } \}$ denote the set of demographic subgroups, organized under five demographic axes, including age, gender, ethnicity, socioeconomic background, and first language. For each subgroup $g _ { i }$ , we define the corresponding training subset $\mathcal { D } _ { g _ { i } } \subseteq \mathcal { D } _ { \mathrm { t r a i n } }$ using its demographic label.

We then fine-tune the base model on each subgroup:

$$
\theta _ { g _ { i } } = \mathrm { F i n e T u n e } \left( \theta _ { \mathrm { b a s e } } , \mathcal { D } _ { g _ { i } } \right) .\tag{1}
$$

This gives a set of subgroup-adapted models $\{ f _ { \theta _ { g _ { i } } } \} _ { i = 1 } ^ { N } .$ We then merge the subgroup-adapted models into a global demographic model:

$$
\theta _ { \mathrm { g l o b a l } } = \mathrm { M e r g e } \left( \theta _ { \mathrm { b a s e } } , \theta _ { g _ { 1 } } , \theta _ { g _ { 2 } } , \dots , \theta _ { g _ { N } } \right) .\tag{2}
$$

We adopt several model merging strategies. Linear merging, which is the only exception to Eq. 2, directly averages the subgroup-adapted connector parameters without including the base connector. Task Arithmetic first computes task vectors relative to the base model, then sums and scales these vectors before adding them back to the base parameters [18]. TIES resolves interference among task vectors by trimming small updates, electing parameter-wise signs, and merging only sign-consistent updates [19]. Model Breadcrumbs discards both low-magnitude and high-magnitude parameter updates before merging, retaining intermediate-magnitude updates that are expected to be more transferable [20]. DARE-TIES first applies random pruning and rescaling to task vectors before using TIES-style merging [21].

## 2.3. Intersection-Corrected Model

Global demographic merging combines independently adapted subgroup models but does not explicitly model cross-axis intersections. Because intersectional utterances occur in multiple subgroup subsets, their contributions may be counted repeatedly or reflected in misaligned subgroup updates, potentially leaving intersection-specific errors after merging. We therefore estimate intersection-specific correction vectors relative to selected pairwise merged models and transfer them to the global model.

We first identify the demographic intersections that are most likely to benefit from correction. We consider two criteria: subgroup-level WER and task-vector conflict. For the WER-based criterion, we evaluate the base model $f _ { \theta _ { \mathrm { b a s e } } }$ on each subgroup and select the worst-performing subgroup within each demographic axis. Candidate pairs are formed by taking cross-axis combinations of these selected subgroups, resulting in the candidate set $\mathcal { C } _ { \mathrm { W E R } }$ . For the conflict-based criterion, we measure the disagreement between subgroup updates. For each subgroup g<sub>i</sub>, we flatten the parameter difference between the subgroup-adapted connector and the base connector:

$$
\tau _ { g _ { i } } = \mathrm { f l a t t e n } \left( \theta _ { g _ { i } } - \theta _ { \mathrm { b a s e } } \right) .\tag{3}
$$

We then compute the cosine similarity $s _ { i , j }$ between $\tau _ { g _ { i } }$ and $\tau _ { g _ { j } }$ for each cross-axis pair $( g _ { i } , g _ { j } )$ . Lower similarity indicates stronger conflict, and the most conflicting pairs form ${ \mathcal { C } } _ { \mathrm { c o n f } } .$

For each candidate set ${ \mathcal { C } } \in \{ \mathcal { C } _ { \mathrm { W E R } } , \mathcal { C } _ { \mathrm { c o n f } } \}$ , we merge each candidate pair $( g _ { i } , g _ { j } ) \in \mathcal { C }$ to get the pairwise merged model:

$$
\begin{array} { r } { \theta _ { g _ { i } , g _ { j } } = \mathrm { M e r g e } \left( \theta _ { g _ { i } } , \theta _ { g _ { j } } \right) . } \end{array}\tag{4}
$$

Let $\mathcal { D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } = \mathcal { D } _ { g _ { i } } ^ { \mathrm { e v a l } } \cap \mathcal { D } _ { g _ { j } } ^ { \mathrm { e v a l } }$ . Let $\mathrm { W } ( \theta , { \mathcal { D } } )$ denote the WER of model $f _ { \theta }$ on D. We define the WER penalty as:

$$
\begin{array} { c } { { P _ { g _ { i } , g _ { j } } = W \left( \theta _ { g _ { i } , g _ { j } } , { \mathcal D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } \right) } } \\ { { - \operatorname* { m i n } \left( W \left( \theta _ { g _ { i } } , { \mathcal D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } \right) , W \left( \theta _ { g _ { j } } , { \mathcal D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } \right) \right) } } \end{array}\tag{5}
$$

Pairs are ranked by $P _ { g _ { i } , g _ { j } }$ within each candidate set. The selected pairs from $\mathcal { C } _ { \mathrm { W E R } }$ and $\mathcal { C } _ { \mathrm { c o n f } }$ are denoted by $S _ { \mathrm { W E R } }$ and $S _ { \mathrm { c o n f } }$

For each selected pair $( g _ { i } , g _ { j } ) \in \mathcal { S } _ { \mathrm { W E R } } \cup \mathcal { S } _ { \mathrm { c o n f } }$ , we further fine-tune the pairwise merged model on the intersection training set $\mathcal { D } _ { g _ { i } , g _ { j } } ^ { \mathrm { t r a i n } } = \mathcal { D } _ { g _ { i } } ^ { \mathrm { t r a i n } } \cap \bar { \mathcal { D } } _ { g _ { j } } ^ { \mathrm { t r a i n } }$ :

$$
\hat { \theta } _ { g _ { i } , g _ { j } } = \mathrm { F i n e T u n e } \left( \theta _ { g _ { i } , g _ { j } } , { \mathcal D } _ { g _ { i } , g _ { j } } ^ { \mathrm { t r a i n } } \right) .\tag{6}
$$

The correction vector is the model update introduced by this intersection fine-tuning:

$$
\delta _ { g _ { i } , g _ { j } } = \hat { \theta } _ { g _ { i } , g _ { j } } - \theta _ { g _ { i } , g _ { j } } .\tag{7}
$$

These corrections are estimated relative to pairwise merged models and transferred to the global model. We consider three ways to aggregate these corrections. Method 1 uses WER-selected pairs, Method 2 uses conflict-selected pairs, and Method 3 combines both:

$$
\theta _ { \mathrm { f i n a l } } ^ { c } = \left\{ \begin{array} { l l } { \theta _ { \mathrm { g l o b a l } } + \Delta _ { \mathrm { W E R } } , } & { c = \mathrm { M e t h o d 1 } , } \\ { \theta _ { \mathrm { g l o b a l } } + \Delta _ { \mathrm { c o n f } } , } & { c = \mathrm { M e t h o d 2 } , } \\ { \theta _ { \mathrm { g l o b a l } } + \Delta _ { \mathrm { W E R } } + \Delta _ { \mathrm { c o n f } } , } & { c = \mathrm { M e t h o d 3 } , } \end{array} \right.\tag{8}
$$

where $\begin{array} { r } { \Delta _ { \mathrm { W E R } } = \sum _ { ( g _ { i } , g _ { j } ) \in S _ { \mathrm { W E R } } } \lambda _ { g _ { i } , g _ { j } } ^ { \mathrm { W E R } } \delta _ { g _ { i } , g _ { j } } } \end{array}$ , and $\Delta _ { \mathrm { c o n f } } =$ $\sum _ { ( g _ { i } , g _ { j } ) \in S _ { \mathrm { c o n f } } } \lambda _ { g _ { i } , g _ { j } } ^ { \mathrm { c o n f } } \delta _ { g _ { i } , g _ { j } } \cdots$

## 3. EXPERIMENTS AND DISCUSSION

## 3.1. Experimental Setting

Dataset. We evaluate our approach on Fair-Speech [3], a command-based ASR dataset with demographic annotations. We use five demographic axes, including age, gender, ethnicity, socioeconomic background, and first language, and split the data into training, validation, and test sets using a stratified 70/15/15 split. In total, we use 18 demographic subgroups, including four age groups, two gender groups, seven ethnicity groups, three socioeconomic groups, and two first-language groups. All adaptations reuse subsets of $D _ { \mathrm { t r a i n } } ;$ no additional data are used.

Implementation Details. The base ASR model was trained for 10 epochs using AdamW with learning rate $1 \times 1 0 ^ { - 4 }$ batch size 64, 2000 warmup steps, linear decay, evaluation every 3000 steps, and early stopping with patience 5. For merging, Linear averages subgroup-adapted connectors, Task Arithmetic uses scaling $1 / 1 8 ,$ Model Breadcrumbs uses thresholds 0.90 and 0.99, TIES keeps the top 20% of taskvector updates with scaling 1, and DARE-TIES drops 20% of updates before applying TIES with the same top-20% rule and scaling 1. For intersection correction, selected pairs follow WER-penalty ranking. We vary K from 1 to 10 for Method 1 and from 1 to 12 for Method 2, and tune correction coefficients by grid search with coefficients summing to 1. Rand reports the mean WER over three runs of intersection correction using randomly selected pairs. All pairs and hyperparameters were selected exclusively on the validation set.

## 3.2. Overall Results

Table 1: Overall WER (%). Relative WER improvements over Base are shown as subscripts.
<table><tr><td>Merging Method</td><td>Global</td><td>Rand</td><td>M1</td><td>M2</td><td>M3</td></tr><tr><td>Base</td><td colspan="5"> $\overline { { 7 . 3 8 } }$ </td></tr><tr><td>Linear</td><td> $\overline { { 6 . 4 3 _ { 1 2 . 9 } } }$ </td><td> $\overline { { 6 . 4 1 _ { 1 3 . 1 } } }$ </td><td> $\overline { { { \bf 6 . 3 1 } _ { 1 4 . 5 } } }$ </td><td> $\underline { { 6 . 3 4 _ { 1 4 . 1 } } }$ </td><td> $\overline { { 6 . 3 7 _ { 1 3 . 7 } } }$ </td></tr><tr><td>TIES</td><td> $6 . 4 2 _ { 1 3 . 0 }$ </td><td> $6 . 4 8 _ { 1 2 . 2 }$ </td><td> ${ \bf 5 . 1 3 3 0 . 5 }$ </td><td> $6 . 3 8 1 3 . 6 $ </td><td> $\frac { \mathrm { 5 . 5 5 2 4 . 8 } } { c \mathrm { ~ . 0 2 } }$ </td></tr><tr><td>Task Arithmetic</td><td> $6 . 3 8 _ { 1 3 . 6 }$ </td><td> $6 . 4 0 _ { 1 3 . 3 }$ </td><td> ${ \bf 6 . 3 2 } _ { 1 4 . 4 }$ </td><td> $6 . 3 3 _ { 1 4 . 2 }$ </td><td> $\underline { { 6 . 3 2 } } _ { 1 4 . 4 }$ </td></tr><tr><td>Breadcrumbs</td><td> $6 . 7 7 _ { 8 . 3 0 }$ </td><td> $6 . 4 4 _ { 1 2 . 7 }$ </td><td> $5 . 7 5 _ { 2 2 . 1 }$ </td><td> $6 . 4 8 _ { 1 2 . 2 }$ </td><td>5.6323.7</td></tr><tr><td>DARE-TIES</td><td> $6 . 4 3 _ { 1 2 . 9 }$ </td><td> $6 . 4 5 1 2 . 6 $ </td><td> $6 . 3 4 1 4 . 1$ </td><td> ${ \bf 6 . 3 3 1 4 . 2 }$ </td><td> $\underline { { 6 . 3 4 } } _ { 1 4 . 1 }$ </td></tr></table>

Table 1 reports the overall WER of the base model, the global merged models, and the three intersection-corrected variants. Global merging consistently improves over the base model, reducing WER from 7.38% to 6.38%-6.77% (relative reductions of 8.3–13.6%) across all merging strategies. Subgroup merging therefore improves overall ASR performance.

Intersection correction provides further gains over global merging, although their magnitude varies across merging methods. TIES with Method 1 shows a clear improvement, achieving the best overall WER of 5.13%. Breadcrumbs also benefits from correction, with Method 3 reaching 5.63%. Random selection produced inconsistent gains and sometimes underperformed global merging, whereas the proposed pair selection consistently achieved lower WER. These results suggest that intersection correction is most effective when the global merge leaves stronger conflicts among subgroupspecific updates.

## 3.3. Subgroup Performance Analysis

Table 2: Demographic-level WER (%).
<table><tr><td>Merging method</td><td>Variant</td><td> $\mathrm { A g e }$ </td><td>Gender</td><td>Ethnicity</td><td>SES</td><td>First Lang.</td></tr><tr><td>Base</td><td>一</td><td>7.07</td><td>7.65</td><td>6.84</td><td>7.08</td><td>6.65</td></tr><tr><td>Linear</td><td>Global</td><td>6.31</td><td>6.63</td><td>6.04</td><td>6.23</td><td>5.87</td></tr><tr><td>Linear</td><td>M1</td><td>6.20</td><td>6.52</td><td>5.91</td><td>6.13</td><td>5.76</td></tr><tr><td>Linear</td><td>M2</td><td>6.28</td><td>6.58</td><td>5.89</td><td>6.19</td><td>5.82</td></tr><tr><td>Linear</td><td>M3</td><td>6.25</td><td>6.56</td><td>5.89</td><td>6.12</td><td>5.78</td></tr><tr><td>TIES</td><td>Global</td><td>6.27</td><td>6.62</td><td>6.08</td><td>6.27</td><td>5.88</td></tr><tr><td>TIES</td><td>M1</td><td>4.71</td><td>5.26</td><td>4.77</td><td>5.33</td><td>4.91</td></tr><tr><td>TIES</td><td>M2</td><td>6.31</td><td>6.59</td><td>5.96</td><td>6.20</td><td>5.72</td></tr><tr><td>TIES</td><td>M3</td><td>5.48</td><td>5.73</td><td>5.12</td><td>5.57</td><td>5.17</td></tr><tr><td>Task Arithmetic</td><td>Global</td><td>6.29</td><td>6.58</td><td>6.01</td><td>6.19</td><td>5.83</td></tr><tr><td>Task Arithmetic</td><td>M1</td><td>6.25</td><td>6.52</td><td>5.92</td><td>6.13</td><td>5.82</td></tr><tr><td>Task Arithmetic</td><td>M2</td><td>6.24</td><td>6.53</td><td>5.86</td><td>6.18</td><td>5.79</td></tr><tr><td>Task Arithmetic</td><td>M3</td><td>6.22</td><td>6.52</td><td>5.85</td><td>6.14</td><td>5.80</td></tr><tr><td>Breadcrumbs</td><td>Global</td><td>6.67</td><td>6.96</td><td>6.37</td><td>6.64</td><td>6.23</td></tr><tr><td>Breadcrumbs</td><td>M1</td><td>5.92</td><td>5.85</td><td>5.54</td><td>5.57</td><td>5.37</td></tr><tr><td>Breadcrumbs</td><td>M2</td><td>6.36</td><td>6.68</td><td>6.11</td><td>6.38</td><td>5.95</td></tr><tr><td>Breadcrumbs</td><td>M3</td><td>5.80</td><td>5.73</td><td>5.33</td><td>5.33</td><td>5.24</td></tr><tr><td>DARE-TIES</td><td>Global</td><td>6.37</td><td>6.63</td><td>6.13</td><td>6.22</td><td>5.80</td></tr><tr><td>DARE-TIES</td><td>M1</td><td>6.29</td><td>6.54</td><td>6.12</td><td>6.17</td><td>5.72</td></tr><tr><td>DARE-TIES</td><td>M2</td><td>6.26</td><td>6.53</td><td>6.03</td><td>6.18</td><td>5.68</td></tr><tr><td>DARE-TIES</td><td>M3</td><td>6.27</td><td>6.54</td><td>6.08</td><td>6.14</td><td>5.68</td></tr></table>

We further examine whether the overall WER improvements also appear across demographic groups. Table 2 reports the average subgroup WER for each demographic axis. Global merging improves subgroup performance over the base model across all five demographic axes. Among the global merged models, the largest reduction is achieved by Task Arithmetic on Gender, reducing WER from 7.65% to 6.58%. This suggests that demographic model merging improves ASR performance beyond the aggregate level.

Intersection correction brings the largest gain for TIES. Compared with the TIES global model, TIES+M1 reduces WER from 6.27% to 4.71% on age, 6.62% to 5.26% on gender, and 6.08% to 4.77% on ethnicity. Breadcrumbs also benefits from correction, with M3 reducing age WER from 6.67% to 5.80% and SES WER from 6.64% to 5.33%.

## 3.4. Demographic Disparity Analysis

We evaluate fairness by measuring whether each model reduces the WER disparity within each demographic axis. Let a denote a demographic axis and let ${ \mathcal { G } } _ { a }$ be the set of subgroups under that axis. For a model $f _ { \theta } { } _ { ; }$ , we define the WER gap along axis a as:

$$
\mathrm { G a p } _ { \theta , a } = \operatorname* { m a x } _ { g _ { i } \in \mathcal { G } _ { a } } \mathrm { W E R } \left( f _ { \theta } , \mathcal { D } _ { g _ { i } } ^ { \mathrm { e v a l } } \right) - \operatorname* { m i n } _ { g _ { i } \in \mathcal { G } _ { a } } \mathrm { W E R } \left( f _ { \theta } , \mathcal { D } _ { g _ { i } } ^ { \mathrm { e v a l } } \right) .\tag{9}
$$

We then compute the relative gap reduction with respect to the base model:

$$
\mathrm { R G R } _ { \theta , a } = 1 - \frac { \mathrm { G a p } _ { \theta , a } } { \mathrm { G a p } _ { \theta _ { \mathrm { b a s e } } , a } } .\tag{10}
$$

![](images/b05b22ccb467a038cad95024f227b5241b03b83c62d7354d40b4df9ea28b5875.jpg)  
Fig. 1: Relative reduction in max–min WER disparity compared with the base model.

![](images/10ee591155cf57086329900c8626995b77b7e646db2204143b8557979f3296cb.jpg)  
Fig. 2: Average WER relative to global merging. Lower than 100% is better.

A positive value indicates that the subgroup WER gap is smaller than that of the base model, while a negative value indicates that the gap increases. As shown in Fig. 1, most settings reduce subgroup disparity across demographic axes, but the effect is not uniform. In particular, TIES with Method 1 increases the WER gap for age and socioeconomic background, showing that intersection correction can improve average performance while still increasing within-axis disparity for some axes.

## 3.5. Intersection Analysis

To isolate the effect of intersection correction, we compare each corrected model with its corresponding global merged model on intersectional groups. Selected intersections are the cross-axis pairs whose correction vectors are included in the final model, while non-selected intersections are the remaining eligible cross-axis pairs. For a correction variant c ∈ {M1, M2, M3} and an intersection set S, we compute the average relative WER:

$$
R _ { c , S } = \frac { 1 } { | S | } \sum _ { ( g _ { i } , g _ { j } ) \in S } \frac { \mathrm { W E R } \left( f _ { \theta _ { \mathrm { f i n a l } } ^ { c } } , \mathcal { D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } \right) } { \mathrm { W E R } \left( f _ { \theta _ { \mathrm { g l o b a l } } } , \mathcal { D } _ { g _ { i } , g _ { j } } ^ { \mathrm { e v a l } } \right) } .\tag{11}
$$

As shown in Fig. 2, corrections yield larger gains on selected than non-selected intersections, indicating that they primarily benefit the groups they target. For selected intersections, most correction variants reduce WER relative to the global merged model, showing that the correction vectors improve the intersectional groups they are designed to target. The strongest gains appear with TIES and Model Breadcrumbs. TIES with Method 3 reduces the average relative

WER to 74.80%, while Model Breadcrumbs reaches 77.41% with Method 1 and 84.57% with Method 3. Linear merging also benefits from Method 2, reaching 70.00% on its selected intersections.

For non-selected intersections, the corrected models generally remain close to or below the global merged model. This suggests that the correction step does not broadly degrade intersections that were not directly selected. Several settings also improve non-selected intersections. TIES with Method 1 and Method 3 reach 87.91% and 88.67%, while Model Breadcrumbs with Method 3 reaches 84.96%. These results indicate that intersection correction can improve targeted intersectional performance while largely preserving performance on non-targeted intersections.

## 4. CONCLUSION

In this work, we studied demographic-aware model merging for fair Speech-LLM-based ASR. We considered five merging techniques and introduced intersection-aware corrections based on subgroup WER and task-vector conflict. Our results show that demographic model merging improves overall and demographic-level ASR performance, and that intersection correction can provide further gains for several merging strategies using the same training data. In particular, TIES with WER-based correction achieves the best overall WER. The subgroup and intersection analyses further show that the proposed corrections can improve targeted intersectional groups while largely preserving performance on non-targeted intersections.

## 5. ACKNOWLEDGMENTS

This research was supported by the IEEE SPS SigMA program. This work was partially funded by UKRI and the European Union’s Horizon Europe Research and Innovation Programme under the ELOQUENCE project (Grant Agreement No. 101135916).

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This research was conducted using human subject data from the open-access Fair-Speech Dataset released by Meta AI.

## 7. REFERENCES

[1] Irina-Elena Veliche and Pascale Fung, “Improving Fairness and Robustness in End-to-End Speech Recognition Through Unsupervised Clustering,” in ICASSP, 2023.

[2] Hend ElGhazaly, Bahman Mirheidari, Heidi Christensen, and Nafise Sadat Moosavi, “Fairness in Automatic Speech Recognition Isn’t a One-Size-Fits-All,” in EMNLP, 2025.

[3] Irina-Elena Veliche, Zhuangqun Huang, Vineeth Ayyat Kochaniyan, Fuchun Peng, Ozlem Kalinli, and Michael L. Seltzer, “Towards measuring fairness in speech recognition: Fair-Speech dataset,” in Interspeech, 2024.

[4] Allison Koenecke, Andrew Nam, Emily Lake, Joe Nudell, Minnie Quartey, Zion Mengesha, Connor Toups, John R. Rickford, Dan Jurafsky, and Sharad Goel, “Racial disparities in automated speech recognition,” PNAS, 2020.

[5] Mikel K. Ngueajio and Gloria Washington, “Hey ASR System! Why Aren’t You More Inclusive?,” in HCI International, 2022.

[6] Pranav Dheram, Murugesan Ramakrishnan, Anirudh Raju, I-Fan Chen, Brian King, Katherine Powell, Melissa Saboowala, Karan Shetty, and Andreas Stolcke, “Toward Fairness in Speech Recognition: Discovery and mitigation of performance disparities,” in Interspeech, 2022.

[7] Yuanyuan Zhang, Aaricia Herygers, Tanvina Patel, Zhengjun Yue, and Odette Scharenborg, “Exploring data augmentation in bias mitigation against non-nativeaccented speech,” in ASRU, 2023.

[8] Jongsuk Kim, Jaemyung Yu, Minchan Kwon, and Junmo Kim, “FairASR: Fair Audio Contrastive Learning for Automatic Speech Recognition,” in Interspeech, 2025.

[9] Monorama Swain, Anna Katrine Van Zee, and Anders Søgaard, “On Mitigating Performance Disparities in Multilingual Speech Recognition,” in EMNLP, 2024.

[10] Alexandre Ducorroy and Rachid Riad, “Robust finetuning of speech recognition models via model merging: application to disordered speech,” ArXiv, 2025.

[11] Natarajan Balaji Shankar, Zilai Wang, Eray Eren, and Abeer Alwan, “Selective Attention Merging for low resource tasks: A case study of Child ASR,” in ICASSP, 2025.

[12] Carlos Carvalho, Francisco Teixeira, Thomas Rolland, and Alberto Abad, “Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR,” ArXiv, 2026.

[13] Qiuming Zhao, Guangzhi Sun, and Chao Zhang, “Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation,” in ICASSP, 2026.

[14] Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, and Xie Chen, “An Embarrassingly Simple Approach for LLM with Strong ASR Capacity,” ArXiv, 2024.

[15] Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, and Xie Chen, “Speech Recognition Meets Large Language Model: Benchmarking, Models, and Exploration,” in AAAI, 2025.

[16] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in ICML, 2022.

[17] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu, “TinyLlama: An Open-Source Small Language Model,” ArXiv, 2024.

[18] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi, “Editing models with task arithmetic,” in ICLR, 2023.

[19] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal, “TIES-MERGING: resolving interference when merging models,” in NeurIPS, 2023.

[20] MohammadReza Davari and Eugene Belilovsky, “Model Breadcrumbs: Scaling Multi-task Model Merging with Sparse Masks,” in ECCV, 2024.

[21] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li, “Language models are super mario: absorbing abilities from homologous models as a free lunch,” in ICML, 2024.