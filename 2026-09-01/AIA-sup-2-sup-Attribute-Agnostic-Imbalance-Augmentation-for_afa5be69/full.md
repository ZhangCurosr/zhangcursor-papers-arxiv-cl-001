# AIA<sup>2</sup>: Attribute-Agnostic Imbalance Augmentation for Subgroup Robustness

Hanshu Rao University of Memphis hrao@memphis.edu

Guangzeng Han University of Memphis ghan@memphis.edu

Xiaolei Huang University of Missouri xiaolei.huang@missouri.edu

## Abstract

Attributes describing data content and context can induce diverse imbalance patterns that go beyond label imbalance alone. However, existing studies primarily address label imbalance while overlooking data attributes, such as topics and demographics, which can induce meaningful subgroup structure while causing model degradation on underrepresented subgroups. We propose Attribute-Agnostic Imbalance Augmentation (AIA<sup>2</sup>)<sup>1</sup>, a framework for improving model robustness under varying subgroup imbalances without explicit subgroup annotations. AIA<sup>2</sup> automatically discovers varying imbalances via latent semantic distributions, obtains slices with both learning difficulty and subgroup imbalance deficits, and deploys a large language model (LLM) for subgroup-aware imbalance augmentation. We have evaluated AIA<sup>2</sup> on 5 popular corpora with rich domains and their attribute values, covering social issues and diverse topics. Results show improved performance on the lowest-performing subgroups and consistent gains over competitive baselines. Ablation studies confirm complementary contributions from each component, and additional analyses show that AIA<sup>2</sup> provides a practical and consistent way to improve worst-group robustness under data subgroup imbalance.

## 1 Introduction

Data attributes explicitly characterize data with heterogeneous attribute values and categorize data into subgroups with distinct statistical properties, such as topic, demographic factors, and various sources (Koh et al., 2021; Chen et al., 2023a; Li et al., 2025; Liu et al., 2025). Those data attributes can induce diverse patterns of subgroup imbalance that are not reflected by general data and label imbalance alone. However, varying imbalance patterns by attribute values are usually overlooked in existing studies (Rudner et al., 2024), which can easily lead to unstable performance on underrepresented or minority subgroups and raise concerns on model robustness and reliability. For example, a model optimized for overall label distributions may still exhibit significant performance variations or disparities across groups of data attributes with varying distributions.

![](images/3cb046ef27f37c7efccebe4e369c94e15dda640af6bbf00d7f7b7e6c8828dc24.jpg)  
Figure 1: Subgroup-level distribution shifts in HumAID across different event sources (colors).

Figure 1 visualizes social media subgroup distributions by event source and label distributions: label imbalance widely exists across the subgroups, while imbalance patterns vary significantly across event sources. While a growing body of work focuses on data balance through re-weighting (Lin et al., 2020; Rao et al., 2026), sampling (Chawla et al., 2002), data augmentation (Zhang et al., 2023), and optimization (Ren et al., 2020), these studies usually consider only global-level label skew, leaving subgroup-level data skew underexplored. Additionally, existing approaches (Henning et al., 2023) implicitly or explicitly assume that attribute values and annotations are accessible; however, subgroup-defining attributes are often unknown, partially observed, sensitive, or unavailable. For example, demographic attributes may be restricted due to privacy or regulatory constraints. These gaps limit existing imbalance methods particularly when subgroup structure is implicit rather

![](images/5a01deb417c3261ccff87123ecc7875f25cc93e92ff381caf1f023aef0376710.jpg)  
Figure 2: Overview of the AIA<sup>2</sup> framework. (1) Latent Slice Discovery partitions examples into coherent slices using joint semantic-predictive representations; (2) Slice-Level Distribution Gap Analysis identifies deficit labels and selects high-loss seeds; (3) Constraint-Guided Augmentation and Filtering generates and filters candidates for consistency and diversity. Augmented data is merged into the training set for the next iteration.

than explicit.

In this work, we introduce Attribute-Agnostic Imbalance Augmentation $( \mathrm { A I A } ^ { 2 } )$ , a general framework for enhancing model robustness to subgroup imbalance induced by unknown or latent distributional variations. Instead of predefined attribute values or explicit subgroup labels, $\mathrm { \ A I A ^ { 2 } }$ automatically discovers latent slices with a joint semantic and learning space, identifies regions of distributional imbalance and misalignment, and constructs latent subgroup graphs to guide data and model augmentation. $\mathrm { \ A I A ^ { \bar { 2 } } }$ employs distributional-gap-aware and constraint-guided data augmentation by large language models (LLMs) to dynamically select and fix subgroup imbalance deficits. Because these latent slices evolve with the task model, $\mathrm { \ A I A ^ { 2 } }$ can track localized failure regions without requiring predefined group assignments. In sum, our contributions are threefold: 1) we propose an attributeagnostic imbalance-learning framework that improves worst-subgroup robustness without using subgroup metadata during training or model selection; 2) we present a subgroup-aware distributional gap analysis and imbalance-deficit-driven selection mechanism that targets minority patterns and labels within latent regions; and 3) we propose a constraint-guided filtering strategy and augmentation to ensure data diversity and label fidelity.

## 2 Method

Prior work on robustness under distribution shift and imbalance typically improves worst-case performance by reweighting or resampling training data across predefined groups, or by optimizing objectives that emphasize underperforming subpopulations (Sagawa et al., 2019; Liu et al., 2021). While these approaches effectively address global class imbalance or known group disparities, they implicitly assume that the relevant failure modes align with available metadata, either explicit group annotations or observable class boundaries. In practice, however, systematic errors often concentrate in latent regions where the local label distribution deviates sharply from the global one, and these regions evolve dynamically as the model learns, cutting across any predefined groupings.

This raises a fundamental question: when the model’s worst failures occur in such unobserved, locally-misaligned regions, how can we identify where the distribution is “broken” and intervene without access to the right grouping? To address this challenge, we propose an iterative slice-aware augmentation framework that i) discovers latent slices by clustering examples in a joint semanticpredictive space, ii) quantifies local distribution gaps to identify which labels are deficient in each slice, and iii) selectively generates targeted augmentations from the hardest examples while ensuring label consistency and semantic diversity. By iteratively repairing these local distribution defects, our method reduces concentrated slice-level errors without requiring known subgroups. Figure 2 shows the overview of our method.

## 2.1 Latent Slice Discovery

To identify where errors concentrate, we partition the data into slices based on two complementary signals: the input distribution $P ( X )$ captured by semantic embeddings, and the predictive distribution $P ( { Y | } X )$ captured by model outputs. Clustering on $P ( X )$ alone groups semantically similar examples but ignores model behavior; clustering on $P ( { Y \vert } X )$ alone captures prediction patterns but conflates unrelated inputs. By combining both, we discover regions where similar content elicits similar (often erroneous) predictions, the failure modes that single-distribution approaches miss.

We construct a joint representation space that captures both semantic content and predictive behavior. For each training example i, we combine its semantic embedding $\mathbf { z } _ { i } \in \mathbb { R } ^ { d _ { z } }$ (obtained from a pre-trained BGE encoder (Xiao et al., 2023)) with its prediction vector $\mathbf { p } _ { i }$ . For classification tasks, $\mathbf { p } _ { i } \in \mathbb { R } ^ { C }$ corresponds to softmax outputs for singlelabel or sigmoid outputs for multi-label settings. For sequence labeling tasks such as named entity recognition, where predictions occur at the token level, we instead construct a boundary behavior vector $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 C _ { E } }$ for $C _ { E }$ entity types. This vector encodes, for each type, the average prediction loss on gold entity tokens, the entropy at boundary regions including adjacent positions, and the negative margin between correct and competing label probabilities—collectively capturing how reliably the model identifies entity boundaries rather than aggregating raw token-level outputs. The unified representation is defined as:

$$
\mathbf { u } _ { i } = [ \bar { \mathbf { z } } _ { i } \parallel \boldsymbol { \alpha } \cdot \bar { \mathbf { p } } _ { i } ]
$$

, where $\bar { \bf z } _ { i }$ and $\bar { \bf p } _ { i }$ are L2-normalized vectors, || denotes concatenation, and α controls the relative importance of predictive information. This design ensures that slices capture coherent regions in the joint space of $P ( X )$ (via $\mathbf { z } _ { i } )$ and $P ( { Y \vert } X )$ (via $\mathbf { p } _ { i } )$ rather than relying on either signal alone.

To discover coherent slices from this joint space, we construct a k-nearest neighbor graph where edges connect similar examples, then apply the Leiden community detection algorithm (Traag et al., 2019) with resolution parameter $\gamma$ to control granularity. Unlike hierarchical clustering, Leiden optimization naturally handles varying cluster sizes and discovers communities at multiple scales, making it well-suited for finding both broad patterns and concentrated error pockets. The resulting slices provide a data-driven partitioning evolving with the model’s behavior across training iterations.

## 2.2 Slice-Level Distribution Gap Analysis

Having discovered slices that are coherent in $P ( X ) \times P ( Y | X )$ space, we now ask: which labels should we augment within each slice? The answer lies in the gap between a slice’s local label distribution $P ^ { ( s ) } ( Y )$ and the empirical global distribution $P ^ { \mathrm { g l o b a l } } ( Y )$ . For each slice $s ,$ we compute the local label distribution $\mathbf { q } ^ { ( s ) }$ and compare it with the global distribution $\mathbf { q } ^ { ( \mathrm { g l o b a l } ) }$ . For multi-class classification, these are simply the empirical class frequencies; for multi-label tasks, they represent per-label positive rates. For sequence labeling tasks such as NER, we define distributions at the span level rather than the token level— $\mathbf { - q } _ { c } ^ { ( s ) }$ represents the proportion of entity spans of type c within slice s, computed by counting contiguous B/I segments for each entity type. This span-based formulation better reflects the semantic composition of entities and avoids bias from varying entity lengths. We quantify the distribution gap using L1 distance:

$$
\mathrm { g a p } _ { s } = \lVert \mathbf { q } ^ { ( \mathrm { g l o b a l } ) } - \mathbf { q } ^ { ( s ) } \rVert _ { 1 }
$$

When $\mathrm { g a p } _ { s } > \tau ,$ , we select slice s for augmentation. More importantly, we compute the deficit vector ${ \bf d } ^ { ( s ) } = \bar { \mathrm { m a x } } ( { \bf 0 } , \bar { \bf q } ^ { ( \mathrm { g l o b a l } ) } - \bar { \bf q } ^ { ( s ) } )$ , which indicates which labels are under-represented relative to the global distribution and by how much.

To select seed examples for augmentation, we prioritize candidates based on both label deficits and individual difficulty. For each deficit label c with $d _ { c } ^ { ( s ) } > 0$ , we identify candidate examples in slice s with label $^ { c , }$ excluding any previously augmented samples. We then rank candidates by their cross-entropy loss $\ell _ { i } ,$ selecting the hardest examples as seeds. The number of seeds per label is proportional to its deficit mass, with the total augmentation budget for slice s determined by:

$$
n _ { s } = \operatorname* { m i n } \Bigg ( n _ { \operatorname* { m a x } } , \ : \lceil \lvert S _ { s } \rvert ^ { \beta } \cdot \sum _ { c } d _ { c } ^ { ( s ) } \cdot \rho \rceil \Bigg )
$$

, where $S _ { s }$ is a set of examples in the slice s, $\beta \in ( 0 , 1 )$ provides sublinear scaling to prevent large slices from dominating, $\rho$ controls the overall augmentation rate, and $n _ { \mathrm { m a x } }$ caps the per-slice budget.

## 2.3 Constraint-Guided Augmentation and Filtering

Given the deficit labels identified through $P ( Y )$ gap analysis within each $P ( X ) \times P ( Y | X )$ -defined slice, we now generate augmentations that maintain label fidelity while introducing sufficient diversity to improve local coverage. We leverage a large language model (LLM) with carefully designed constraints to generate meaningful candidates.

For classification tasks, we design prompts that enforce structural transformation and entity substitution while preserving label-determining semantics. Each prompt requires 1) sentencelevel restructuring beyond simple word swaps and 2) type-preserving entity replacement (e.g., person→person, location→location). Sequence labeling tasks like NER present a different challenge: direct text rewriting can misalign tokens with their labels. To address this, we use a placeholderbased mechanism. Before generation, we replace each entity span with a unique placeholder (e.g., $\_ { \_ } \mathtt { E N T } \varnothing _ { \_ - - } )$ and instruct the LLM to rewrite only the surrounding context while preserving all placeholders. After generation, we expand placeholders back to their original entity tokens with corresponding BIO tags, ensuring perfect token–label alignment. To guide the generation, we provide k reference examples—other high-loss samples from the same slice with the same label—with entities masked as type markers (e.g., <ENT:PER>) to prevent direct copying while allowing the model to borrow stylistic patterns.

To ensure quality, we generate m candidates per seed and apply a two-stage filtering pipeline. First, we apply hard constraints to verify format compliance, including length limits and single-line output; for NER tasks, we additionally enforce that each placeholder appears exactly once and that original entity surface forms do not appear in the rewritten context. Second, we score surviving candidates to balance prediction consistency and semantic diversity. Specifically, we pass each candidate through the current model and compute its score as:

$$
\begin{array} { r } { \mathrm { s c o r e } _ { j } = \lambda _ { 1 } \mathrm { s i m } ( \mathbf { p } _ { \mathrm { a u g } } ^ { ( j ) } , \mathbf { p } _ { \mathrm { s e e d } } ) + \lambda _ { 2 } \big ( 1 - \mathrm { s i m } ( \mathbf { z } _ { \mathrm { a u g } } ^ { ( j ) } , \mathbf { z } _ { \mathrm { s e e d } } ) \big ) } \end{array}
$$

, where p denotes predicted class probabilities from the task model and z denotes sentence embeddings from the pretrained BGE encoder, with both terms computed via cosine similarity. The first term rewards prediction consistency, while the second encourages semantic diversity relative to the seed. We then select the top-scoring candidates up to the perseed budget, typically retaining 2–3 augmentations per seed to balance quality and quantity.

Algorithm 1 $\mathrm { \ A I A ^ { 2 } }$ unified training procedure   
Input: Training set $\mathcal { D } _ { 0 }$ , validation set $\mathcal { D } _ { \mathrm { v a l } }$   
Parameter: Warmup rounds $R _ { w }$ , total rounds $T ,$   
gap threshold $\tau$   
Output: Best model $\theta ^ { * }$   
1: Initialize $\theta , \mathcal { D }  \mathcal { D } _ { 0 } , \theta ^ { * }  \theta$   
2: for $t = 1$ to $T$ do   
3: $\theta \gets \mathrm { T R A I N } ( \theta , \mathcal { D } )$   
4: Update $\theta ^ { * }$ if validation on $\mathcal { D } _ { \mathrm { v a l } }$ improves   
5: i $\mathbf { f } t \le R _ { w }$ or $t = T$ then   
6: continue   
7: end if   
8: Compute $\mathbf { u } _ { i } = [ \bar { \mathbf { z } } _ { i } \left| \left| \alpha \bar { \mathbf { p } } _ { i } \right. \right] , \forall ( x _ { i } , y _ { i } ) \in \mathcal { D }$   
9: Discover slices $\{ S _ { s } \}$ using Leiden on the   
kNN graph   
10: for each slice s with $\mathrm { g a p } _ { s } > \tau$ do   
11: Select high-loss seeds from deficit labels   
12: Generate augmentations $\mathcal { A } _ { s }$ via LLM   
13: end for   
14: $\textstyle { \mathcal { D } } \gets { \mathcal { D } } \cup \bigcup _ { s } { \mathcal { A } } _ { s }$   
15: end for   
16: return $\theta ^ { * }$

## 2.4 Iterative Training Framework

The latent slice discovery, distribution gap analysis, and targeted augmentation mechanisms integrate seamlessly into a unified iterative training procedure summarized in Algorithm 1. Training proceeds through $T$ rounds, beginning with $R _ { w }$ warmup rounds of standard training to establish initial model behavior before augmentation begins. At each subsequent round, the framework trains the model on the current dataset, re-discovers slices using updated joint representations, identifies deficit labels within misaligned slices, and generates targeted augmentations that are merged into the training set for the next round. Re-slicing at each round allows the discovery mechanism to adapt as the model evolves, enabling progressive correction of local distribution defects without predefined groups or manual intervention.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Domain</td><td rowspan="2">Task</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Size #Classes #Subgroups</td><td colspan="3">Splits</td><td rowspan="2">IR</td></tr><tr><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td>HateXplain</td><td>Social</td><td>CLS</td><td>14,995</td><td>3</td><td>8</td><td>11,996</td><td>1,499</td><td>1,500</td><td>1.6</td></tr><tr><td>HumAID</td><td>Crisis</td><td>CLS</td><td>15,311</td><td>7</td><td>5</td><td>10,717</td><td>1,559</td><td>3,035</td><td>14.8</td></tr><tr><td>WWW2015</td><td>Review</td><td>CLS</td><td>15,449</td><td>5</td><td>10</td><td>12,378</td><td>1,562</td><td>1,509</td><td>27.4</td></tr><tr><td>RE3D</td><td>Security</td><td>NER</td><td>931</td><td>5</td><td>7</td><td>634</td><td>50</td><td>247</td><td>16.3</td></tr><tr><td>CrossNER</td><td>Multi-domain</td><td>NER</td><td>5,327</td><td>5</td><td>5</td><td>4,261</td><td>532</td><td>534</td><td>2.0</td></tr></table>

Table 1: Dataset characteristics. #Subgroups indicates annotated metadata (e.g., country, target group) that define natural data partitions. IR (imbalance ratio) is the ratio of the largest class size to the smallest class size.

## 3 Experiment

## 3.1 Data

We evaluate our approach on five publicly available corpora across two tasks: text classification (CLS) and named-entity recognition (NER). These datasets exhibit both global class imbalance and heterogeneous subpopulations whose local label distributions may differ from the global pattern. For CLS, HateXplain (Mathew et al., 2021) provides hate speech annotations with target-community metadata; HumAID (Alam et al., 2021) contains humanitarian tweets from multiple disaster sources; and WWW2015 (Hovy et al., 2015) includes business reviews with company-level metadata. For NER, RE3D (Defence Science and Technology Laboratory, 2021) covers news, government, and encyclopedic documents annotated with defense and security entities, while CrossNER (Liu et al., 2020) spans five specialized domains with domainspecific entity labels. We quantify global imbalance by $\mathrm { I R } = | \mathrm { m a j } | / | \mathrm { m i n } |$ . Table 1 summarizes the dataset characteristics, with further details provided in Appendix A.

## 3.2 Baselines

We compare $\mathrm { \ A I A ^ { 2 } }$ against baselines spanning three paradigms: sample-level reweighting, group-level optimization, and LLM-based augmentation. For sample-level methods, Focal Loss (Lin et al., 2020) down-weights well-classified examples using the modulating factor $( 1 - p _ { t } ) ^ { \gamma }$ , while Just Train Twice (JTT) (Liu et al., 2021) upweights examples misclassified by an early ERM model. For grouplevel methods, Group DRO (Sagawa et al., 2020) optimizes the worst-group loss through group reweighting, and Diverse Prototypical Ensembles (DPE) (To et al., 2025) uses subgroup annotations to build attribute-balanced validation sets and train diverse prototypical classifiers. GEORGE (Sohoni et al., 2020) infers hidden subclasses by clustering learned representations and applies group-robust training over the resulting proxy groups. Following its original example-level formulation, we evaluate GEORGE only on the three classification datasets. AugGPT paraphrases random training samples; CB-LLM augments minority classes by the global class distribution. Unlike these baselines, $\mathrm { \ A I A ^ { 2 } }$ performs targeted augmentation by inferring latent subgroup-label structure. Additional baseline details are provided in Appendix B.

## 3.3 Experimental Settings

To ensure consistency with the baselines, we use DeBERTa-v3-base (He et al., 2021) as the default task model. For $\mathrm { \ A I A ^ { 2 } }$ , we use BGE-large-env1.5 (Xiao et al., 2023) for slice discovery and Qwen3-32B (Team, 2025) for sample augmentation (temperature 0.85). We run 8 iterative rounds and select the best checkpoint based on validation performance. We report Micro-F1 and Macro-F1. For NER, scores are computed at the entity level under the BIO scheme, excluding the O tag. We further evaluate worst-group performance using WGA for multi-class classification and WG-F1 for NER. WGA is the minimum accuracy across predefined subgroups, while WG-F1 is the minimum micro-F1 across subgroups. All results are averaged over five independent runs with different random seeds, and Appendix E reports standard deviations of the results in Table 2. Additional implementation details, hyperparameter sensitivity analyses, and computational overhead appear in Appendix C, Appendix F, and Appendix G, respectively.

## 4 Results

## 4.1 Overall Performance

Table 2 summarizes the overall performance of $\mathrm { \ A I A ^ { 2 } }$ against seven baselines, with GEORGE evaluated on the classification datasets only. Across datasets, $\mathrm { \ A I A ^ { 2 } }$ consistently achieves the best results, improving Macro-F1, Micro-F1, and worst-group metrics by up to 1.9, 2.2, and 1.5 points, respectively. On the severely imbalanced WWW2015 (IR = 27.4), $\mathrm { \ A I A ^ { 2 } }$ reaches 57.6 Macro-F1, outperforming GEORGE by 1.4 points, while improving WGA over the best baseline by 1.5 points. It also remains effective in less skewed settings, improving both Macro-F1 and WG-F1 on CrossNER (IR = 2.0). The comparisons with Aug-GPT and CB-LLM show that these gains are not simply due to generic LLM paraphrasing or global class-balanced expansion. On WWW2015, $\mathrm { \ A I A ^ { 2 } }$ improves over the best augmentation baseline by 3.2 points in Macro-F1, 2.0 points in Micro-F1, and 2.1 points in WGA. Compared with reweighting, group-aware optimization, error-based upweighting or ensembling, and LLM-based augmentation, $\mathrm { \ A I A ^ { 2 } }$ provides more consistent improvements. These results suggest that discovering latent slices and repairing slice-level label deficits through targeted augmentation is more effective than relying on predefined groups, loss-based signals, or untargeted LLM augmentation.

<table><tr><td rowspan="3">Method</td><td colspan="9">CLS</td><td colspan="6">NER</td></tr><tr><td colspan="3">HumAID</td><td colspan="3">WWW2015</td><td colspan="3">HateXplain</td><td colspan="3">RE3D</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td></td><td>WGA M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td></td><td>WGA M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td>WGA M-F1</td><td></td><td> $\mu { - } \mathrm { F } 1$ </td><td>WG-F1 M-F1</td><td></td><td> $\mu { - } \mathrm { F } 1$ </td><td>WG-F1</td></tr><tr><td>Focal</td><td>67.5</td><td>76.0</td><td>69.5</td><td>54.2</td><td>85.2</td><td>78.4</td><td>63.3</td><td>65.1</td><td>54.3</td><td>64.6</td><td>64.0</td><td>59.1</td><td>71.4</td><td>72.6</td><td>60.9</td></tr><tr><td>GroupDRO</td><td>66.8</td><td>75.9</td><td>71.5</td><td>53.1</td><td>85.4</td><td>76.7</td><td>63.1</td><td>64.8</td><td>54.9</td><td>64.9</td><td>63.9</td><td>60.0</td><td>73.7</td><td>72.0</td><td>61.9</td></tr><tr><td>GEORGE</td><td>67.2</td><td>75.9</td><td>70.4</td><td>56.2</td><td>84.5</td><td>78.8</td><td>63.4</td><td>64.4</td><td>56.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JTT</td><td>65.6</td><td>74.5</td><td>68.7</td><td>52.6</td><td>86.1</td><td>79.4</td><td>59.3</td><td>61.7</td><td>52.0</td><td>64.4</td><td>62.3</td><td>57.8</td><td>72.2</td><td>73.4</td><td>62.5</td></tr><tr><td>DPE</td><td>53.3</td><td>75.4</td><td>69.6</td><td>43.4</td><td>86.4</td><td>79.4</td><td>62.7</td><td>63.8</td><td>56.7</td><td>63.5</td><td>63.2</td><td>60.1</td><td>72.2</td><td>73.4</td><td>61.9</td></tr><tr><td>CB-LLM</td><td>66.4</td><td>75.6</td><td>67.5</td><td>53.5</td><td>85.1</td><td>78.8</td><td>62.0</td><td>64.1</td><td>55.6</td><td>64.7</td><td>63.7</td><td>59.5</td><td>71.7</td><td>72.3</td><td>60.9</td></tr><tr><td>AugGPT</td><td>67.5</td><td>76.0</td><td>71.0</td><td>54.4</td><td>85.5</td><td>77.0</td><td>60.4</td><td>63.3</td><td>53.7</td><td>64.9</td><td>64.1</td><td>59.6</td><td>72.1</td><td>72.5</td><td>62.2</td></tr><tr><td>AIA2 (Ours)</td><td>69.4</td><td>77.6</td><td>71.8</td><td>57.6</td><td>87.5</td><td>80.9</td><td>64.6</td><td>66.2</td><td>57.7</td><td>66.5</td><td>66.3</td><td>60.5</td><td>74.4</td><td>74.9</td><td>63.8</td></tr></table>

Table 2: Results of baselines and the proposed $\mathrm { \ A I A ^ { 2 } }$ (percentages). M-F1 and $\mu { - } \mathrm { F } 1$ denote macro-averaged and micro-averaged F1 scores, respectively. WGA denotes worst-group accuracy for CLS, and WG-F1 denotes worst-group F1 score for NER. GEORGE is evaluated only on the classification datasets. Best scores are in bold.

![](images/a8cafe675dc3ffec6222d430b2aa6028fe5bd46532dc167fbc5b706540ed454c.jpg)  
Figure 3: Subgroup-level Micro-F1 (%) across five datasets. Each radar plot compares $\mathrm { \ A I A ^ { 2 } }$ with two representative baselines, JTT and DPE. Each spoke denotes a subgroup, and radial values indicate Micro-F1.

Subgroup-level analysis. Figure 3 presents subgroup Micro-F1 comparisons using radar plots, contrasting $\mathrm { \ A I A ^ { 2 } }$ with two representative baselines, JTT and DPE. Each radar chart corresponds to one dataset, where each spoke denotes a subgroup and the radial value indicates Micro-F1. We choose JTT and DPE because they provide strong worstgroup performance while reflecting two distinct supervision settings: JTT does not rely on subgroup labels, whereas DPE leverages group annotations. Across all five datasets, $\mathrm { \ A I A ^ { 2 } }$ forms the outer envelope on most spokes, achieving the highest or near-highest Micro-F1 for the majority of subgroups. The gains are most evident on challenging subgroups where baselines exhibit sharp drops (e.g., several minority subgroups in HateXplain, specific disaster events in HumAID, and domainspecific subsets in RE3D/CrossNER). Overall, the radar plots indicate that $\mathrm { \ A I A ^ { 2 } }$ delivers broad and consistent improvements across diverse subgroups, rather than increasing aggregate scores by trading off performance on specific subgroups.

## 4.2 Slice Alignment and Deficit Repair

Slice–attribute alignment. To examine how the latent slices discovered without subgroup supervision relate to available dataset attributes, we compare their community assignments with metadata labels withheld from training and used only for post-hoc evaluation. For the four datasets in which each instance has a single metadata value, we compute Adjusted Mutual Information (AMI), a chance-adjusted measure of agreement between partitions (Vinh et al., 2009), over the original training instances after each community-discovery round and report the mean across rounds. HateXplain is excluded because an instance may be associated with multiple target groups, so its metadata does not define a single partition.

<table><tr><td rowspan="3" colspan="2">Blk Method</td><td colspan="10">CLS</td><td colspan="6">NER</td></tr><tr><td colspan="3">HumAID</td><td colspan="3">WWW2015</td><td colspan="3">HateXplain</td><td colspan="3">RE3D</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td></td><td>WGA M-F1</td><td> $\mu { - } \mathrm { F } 1$ </td><td></td><td>WGA M-F1</td><td> $\mu { - } \mathrm { F } 1$ </td><td>WGA M-F1</td><td></td><td> $\mu { - } \mathrm { F } 1$ </td><td>WG-F1 M-F1</td><td></td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td>WG-F1</td></tr><tr><td rowspan="4">MA</td><td>Full</td><td>69.4</td><td>77.6</td><td>71.8</td><td>57.6</td><td>87.5</td><td>80.9</td><td>64.6</td><td>66.2</td><td>57.7</td><td>66.5</td><td>66.3</td><td>60.5</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>w/o slice</td><td>67.8</td><td>76.8</td><td>70.8</td><td>54.3</td><td>86.5</td><td>80.0</td><td>62.3</td><td>64.1</td><td>55.2</td><td>63.9</td><td>63.9</td><td>58.7</td><td>73.2</td><td>71.7</td><td>61.1</td></tr><tr><td>w/o gap-aware</td><td>67.2</td><td>76.1</td><td>69.5</td><td>55.5</td><td>86.3</td><td>78.9</td><td>62.6</td><td>64.5</td><td>57.6</td><td>65.0</td><td>64.7</td><td>53.3</td><td>73.1</td><td>73.1</td><td>62.2</td></tr><tr><td>w/o LLM</td><td>67.7</td><td>76.7</td><td>70.4</td><td>53.8</td><td>86.5</td><td>78.9</td><td>63.5</td><td>65.5</td><td>56.7</td><td>65.6</td><td>65.4</td><td>58.2</td><td>72.8</td><td>71.8</td><td>64.2</td></tr><tr><td rowspan="8">TM</td><td>Focal</td><td>66.2</td><td>73.8</td><td>68.7</td><td>53.3</td><td>85.8</td><td>78.2</td><td>62.6</td><td>64.2</td><td>54.8</td><td>49.9</td><td>47.8</td><td>33.3</td><td>65.8</td><td>64.3</td><td>53.8</td></tr><tr><td>GroupDRO</td><td>67.2</td><td>75.8</td><td>69.5</td><td>52.2</td><td>85.5</td><td>77.8</td><td>60.4</td><td>61.7</td><td>52.9</td><td>49.5</td><td>46.8</td><td>27.0</td><td>66.2</td><td>65.2</td><td>54.5</td></tr><tr><td>GEORGE</td><td>66.5</td><td>75.1</td><td>68.7</td><td>52.4</td><td>84.0</td><td>77.7</td><td>62.0</td><td>63.5</td><td>53.2</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>JTT</td><td>66.2</td><td>75.8</td><td>70.2</td><td>53.8</td><td>85.9</td><td>77.1</td><td>62.2</td><td>63.7</td><td>53.9</td><td>50.7</td><td>51.7</td><td>43.3</td><td>65.7</td><td>64.8</td><td>54.1</td></tr><tr><td>DPE</td><td>40.4</td><td>67.9</td><td>59.7</td><td>17.9</td><td>81.0</td><td>67.7</td><td>44.3</td><td>50.9</td><td>34.3</td><td>45.0</td><td>43.7</td><td>28.6</td><td>24.7</td><td>29.9</td><td>16.1</td></tr><tr><td>CB-LLM</td><td>66.0</td><td>75.6</td><td>68.5</td><td>52.5</td><td>85.2</td><td>77.5</td><td>62.1</td><td>63.7</td><td>52.9</td><td>51.0</td><td>50.2</td><td>40.5</td><td>64.4</td><td>64.7</td><td>55.0</td></tr><tr><td> $\mathbf { A u g G P T }$ </td><td>65.5</td><td>75.4</td><td>68.1</td><td>51.9</td><td>84.6</td><td>77.6</td><td>62.3</td><td>63.9</td><td>51.0</td><td>51.8</td><td>50.2</td><td>42.7</td><td>63.9</td><td>64.1</td><td>57.3</td></tr><tr><td> $\mathbf { A I A } ^ { 2 } \mathbf { \Lambda } ( \mathbf { O u r s } )$ </td><td>67.0</td><td>76.1</td><td>69.5</td><td>54.9</td><td>86.4</td><td>78.2</td><td>63.2</td><td>64.5</td><td>54.8</td><td>53.8</td><td>51.2</td><td>43.6</td><td>66.8</td><td>66.1</td><td>59.1</td></tr><tr><td rowspan="3">BB</td><td>Qwen3-32B</td><td>69.4</td><td>77.6</td><td>71.8</td><td>57.6</td><td>87.5</td><td>80.9</td><td>64.6</td><td>66.2</td><td>57.7</td><td>66.5</td><td>66.3</td><td>60.5</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>Qwen3-14B</td><td>69.3</td><td>77.6</td><td>71.8</td><td>54.0</td><td>86.6</td><td>80.1</td><td>63.3</td><td>65.2</td><td>57.7</td><td>64.8</td><td>65.0</td><td>60.2</td><td>74.4</td><td>73.3</td><td>61.5</td></tr><tr><td>Phi-4</td><td>69.5</td><td>77.7</td><td>71.7</td><td>55.6</td><td>87.2</td><td>81.1</td><td>63.6</td><td>65.2</td><td>58.9</td><td>68.9</td><td>68.0</td><td>62.1</td><td>74.3</td><td>72.9</td><td>62.8</td></tr></table>

Table 3: Combined ablation results. Blocks correspond to: MA (module ablation of $\mathrm { \ A I A ^ { 2 } } )$ , TM (task model ablation on ModernBERT-base), and BB (LLM backbone ablation). In MA, w/o slice removes latent slice discovery, w/o gap-aware removes label-gap-aware seed selection, and w/o LLM removes LLM-based augmentation.

<table><tr><td>Dataset</td><td>#Groups</td><td>#Communities</td><td>AMI↑</td></tr><tr><td>HumAID</td><td>5</td><td>10-12</td><td>0.637</td></tr><tr><td>WWW2015</td><td>10</td><td>13-14</td><td>0.615</td></tr><tr><td>RE3D</td><td>7</td><td>8-10</td><td>0.268</td></tr><tr><td>CrossNER</td><td>5</td><td>12-15</td><td>0.559</td></tr></table>

Table 4: Alignment between discovered communities and metadata-defined groups. #Groups denotes the num ber of attribute values, and #Communities reports the observed range across discovery rounds.

As shown in Table 4, alignment varies across datasets. AMI reaches 0.637 on HumAID, 0.615 on WWW2015, and 0.559 on CrossNER, but decreases to 0.268 on RE3D. Across all four datasets, the number of discovered communities exceeds the number of metadata-defined groups, suggesting that a single group may contain multiple regions with distinct semantic content or model responses. Despite its weaker alignment, $\mathrm { \ A I A ^ { 2 } }$ still improves all three evaluation metrics on RE3D in Table 2, indicating that useful slices need not closely reproduce the available metadata partition. Overall, the discovered slices capture part of the metadatadefined structure while also reflecting finer-grained semantic–predictive variation.

Deficit repair and subgroup gains. To further examine whether $\mathrm { \ A I A ^ { 2 } }$ improves subgroup robustness through targeted deficit repair, we conduct a subset-level diagnostic analysis on classification datasets, where subgroup-label subsets and subsetlevel accuracy are directly defined. A subgrouplabel subset $( a , c )$ refers to examples of label c within subgroup a. For each subset, its label-deficit gap measures how much label c is underrepresented in subgroup a compared with the global label distribution of the original training set. We define the deficit reduction after augmentation as $\Delta d _ { a , c } = d _ { a , c } ^ { \mathrm { p r e } } - d _ { a , c } ^ { \mathrm { p o s t } }$ , where a larger value indicates that augmentation better reduces the local label deficit. We then compute the corresponding accuracy gain as $\Delta M _ { a , c } \dot { = } M _ { a , c } ^ { \mathrm { A I A } } - M _ { a , c } ^ { \mathrm { \hat { N } o A u g } }$ where $M _ { a , c } ^ { \mathrm { N o A u g } }$ is obtained from the same training setup without generated data.

![](images/2f2f990b315a6d332d39e45af971830af94f7b7c355a6853d97c41d66a2dda25.jpg)  
Figure 4: Subgroup-label accuracy gain versus deficit reduction on WWW2015, with an OLS fit.

On WWW2015, deficit reduction is strongly associated with accuracy gain across 39 subgrouplabel subsets: the Spearman correlation is 0.769 $( p = 1 . 1 \times 1 0 ^ { - 8 } )$ , and the ordinary least squares (OLS) fit explains a substantial portion of the variation $( R ^ { 2 } = 0 . 4 8 7 )$ . Figure 4 visualizes this pattern, showing that larger $\Delta d _ { a , c }$ generally corresponds to larger $\Delta M _ { a , c }$ . Among all subsets, 16 subsets (41.0%) fall into the positive-repair region with both reduced deficit and improved accuracy. This result suggests that $\mathrm { \ A I A ^ { 2 } }$ gains are concentrated in subsets where subgroup-label deficits are effectively reduced, supporting targeted deficit repair as a key mechanism behind the observed subgroup robustness improvements. Additional results on the other two CLS datasets are provided in Appendix H and show the same positive trend.

## 4.3 Ablation Study: Module Contributions

We conduct a module-level ablation study by selectively disabling one module at a time while keeping the rest of the iterative training pipeline unchanged: Latent Slice Discovery (“w/o slice”), Label-Gap-Aware Seed Selection (“w/o gap-aware”), and LLM-Based Augmentation (“w/o LLM”). In the first ablation, learned slices are replaced with random partitions of equal cardinality. In the second, seed examples are sampled uniformly at random within each slice rather than prioritized by typespecific gap scores. In the third, the LLM generator is replaced with a copy-based method that simply duplicates the seed instances.

As shown in the first block (MA) of Table 3, the full model performs best across benchmarks, achieving the top results on nearly all metrics. Removing Latent Slice Discovery leads to consistent degradation, with drops on WWW2015 Macro-F1 (57.6 to 54.3) and CrossNER Micro-F1 (74.9 to 71.7), indicating that learned slices support targeted augmentation. Removing Label-Gap-Aware Seed Selection weakens tail performance, with the largest drop on RE3D WG-F1 (60.5 to 53.3), showing that gap-based seed prioritization is critical for under-performing labels. Replacing LLM-Based Augmentation with simple duplication reduces most scores, especially on WWW2015 Macro-F1 (57.6 to 53.8), confirming that diverse examples provide richer supervision than duplicates.

## 4.4 Ablation Study: Impacts of Base Models

We investigate the robustness of $\mathrm { \ A I A ^ { 2 } }$ to two key model choices: 1) the task backbone used for classification and NER; 2) the LLM used for data augmentation. Additional ablations of slice representation, seed selection, and clustering methods are

provided in Appendix I.

Task model generalization. The TM block of Table 3 evaluates whether $\mathrm { \ A I A ^ { 2 } }$ remains effective when the task model is changed from DeBERTa-v3- base to ModernBERT-base (Warner et al., 2024). Although all methods show slightly lower absolute performance under ModernBERT, $\mathrm { \ A I A ^ { 2 } }$ still achieves the strongest or comparable results on most benchmarks. In particular, it obtains the best Macro-F1 on four of the five datasets and the best worst-group score on both NER datasets. For example, on CrossNER, $\mathrm { \ A I A ^ { 2 } }$ improves WG-F1 over the best baseline (AugGPT, 57.3) by 1.8 points. We also find that DPE is more sensitive to the backbone change, likely because it relies on frozen ERM features, whereas $\mathrm { \ A I A ^ { 2 } }$ remains comparatively stable across encoder architectures. These results suggest that slice-aware augmentation is not tied to a specific task backbone and generalizes across modern pretrained encoders.

Data Generator Impacts. The BB block of Table 3 examines sensitivity to augmentation LLM choice by comparing Qwen3-32B (default), Qwen3-14B, and Phi-4. Overall, results are stable across generators, with most datasets varying within 1–2 Macro-F1 points (e.g., HumAID: 69.3– 69.5). Phi-4 performs particularly well on RE3D (Macro-F1 68.9, WG-F1 62.1), exceeding Qwen3- 32B by 2.4 and 1.6 points, suggesting that smaller but well-aligned models can be effective for targeted augmentation. The main sensitivity occurs on WWW2015, where Qwen3-32B (57.6) outperforms Qwen3-14B (54.0) by 3.6 Macro-F1 points, likely due to the higher diversity required for business review generation. These findings indicate that $\mathrm { \ A I A ^ { 2 } }$ is not tightly coupled to a specific LLM and can leverage a range of open-source generators without substantial performance loss.

## 5 Related Work

Class imbalance and group robustness have been widely studied in machine learning. Re-weighting methods adjust loss contributions by class frequency or training difficulty, as in Focal Loss (Lin et al., 2020) and Class-Balanced Loss (Cui et al., 2019). Group-robust methods further target performance disparities by minimizing worst-case loss over predefined groups (Sagawa et al., 2020). Later methods reduce the need for full group annotations by using initial model errors, biased models, or inferred environments to identify hard examples (Liu et al., 2021; Nam et al., 2020; Creager et al., 2021). However, these methods usually rely on observed group information during training or validation, or address global class imbalance without modeling latent subgroup structure.

Another line of work discovers underperforming subgroups without explicit annotations. Domino identifies coherent error slices using cross-modal embeddings and produces natural language descriptions (Eyuboglu et al., 2022). GEORGE clusters learned representations to uncover hidden subclasses, then applies group-robust training with pseudo-labels (Sohoni et al., 2020). Spotlight searches representation space for regions with concentrated errors (d’Eon et al., 2022). These methods are useful for diagnosis, but they mainly identify failure modes after training. They do not integrate subgroup discovery into training to repair local distributional gaps.

Data augmentation offers a complementary way to improve robustness. Traditional text augmentation uses word-level perturbations or backtranslation to create paraphrases (Wei and Zou, 2019; Sennrich et al., 2016). Recent LLM-based methods generate more fluent augmented examples through sample mixing, rephrasing, or counterfactual prompting (Han et al., 2025; Yoo et al., 2021; Dai et al., 2025; Chen et al., 2023b). Mixupbased methods interpolate between training examples, while LISA selectively mixes samples using domain or label information to improve out-ofdistribution robustness (Guo et al., 2019; Yao et al., 2022). However, existing augmentation strategies usually apply transformations globally or depend on explicit domain annotations for selection. They lack a way to locate distributionally deficient latent subgroups and direct augmentation toward those regions.

## 6 Conclusion

This paper presented $\mathrm { \ A I A ^ { 2 } }$ , a framework for improving subgroup robustness without explicit attribute annotations. $\mathrm { \ A I A ^ { 2 } }$ discovers latent slices in a joint semantic-predictive space, identifies local label deficits, and uses LLMs for targeted augmentation. Across five text classification and named entity recognition datasets, $\mathrm { \ A I A ^ { 2 } }$ improves overall and worst-group performance over competitive baselines. Generated-data analysis shows that gains concentrate in subgroup-label subsets where augmentation reduces local deficits, supporting targeted deficit repair as the main mechanism. Ablation studies confirm the roles of slice discovery, gap-aware selection, and LLM-based generation, with stable results across design choices. Future work will explore more efficient slice discovery and augmentation strategies to reduce the computational overhead of iterative training. It will also be benefit to extend $\mathrm { \ A I A ^ { 2 } }$ to broader NLP tasks and systematically evaluate its effectiveness across subgroups defined by demographic attributes, such as health informatics, where rich demographic attributes exist and imbalance and augmentation may interact with demographic biases.

## Limitations

Despite its effectiveness, $\mathrm { \ A I A ^ { 2 } }$ has several limitations. First, the framework introduces extra computational cost due to iterative training, repeated slice discovery, distribution gap analysis, and LLMbased augmentation. Although the overhead is manageable in our experiments, scaling to larger corpora or stronger task models may require more efficient slice construction and generation strategies. Second, latent slice discovery depends on the quality of semantic embeddings and prediction signals. When these representations are noisy or weakly aligned with the target domain, the discovered slices may become less reliable, which can affect seed selection and augmentation quality. Third, LLM-generated examples may contain label noise, hallucinated details, or demographic and topical biases. Although our filtering procedure checks format, label consistency, and semantic diversity, it cannot fully eliminate these risks. Finally, our evaluation focuses on text classification and named entity recognition. The generalizability of $\mathrm { \ A I A ^ { 2 } }$ to other NLP tasks, such as relation extraction or question answering, remains to be further validated. Future work can improve the efficiency, stability, and applicability of slice-aware augmentation.

## Acknowledgments

The first and second authors were supported by the NSF CRII (IIS-2245920) and MRI (CNS-2318210) awards, respectively. We thank the computing resources provided by the iTiger GPU cluster (Sharif et al., 2026) supported by the NSF MRI program under the award CNS-2318210.

## References

Firoj Alam, Umair Qazi, Muhammad Imran, and Ferda Ofli. 2021. Humaid: Human-annotated disaster incidents data from twitter. In Proceedings ofthe Fifteenth International AAAI Conference on Web and Social Media, ICWSM ’21, Online. AAAI.

Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. 2002. Smote: synthetic minority over-sampling technique. Journal ofartificial intelligence research, 16:321–357.

Muxi Chen, YU LI, and Qiang Xu. 2023a. Hibug: On human-interpretable model debug. In Advances in Neural Information Processing Systems, volume 36, pages 4753–4766. Curran Associates, Inc.

Zeming Chen, Qiyue Gao, Antoine Bosselut, Ashish Sabharwal, and Kyle Richardson. 2023b. Disco: Distilling counterfactuals with large language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5514–5528.

Elliot Creager, Jörn-Henrik Jacobsen, and Richard Zemel. 2021. Environment inference for invariant learning. In International Conference on Machine Learning, pages 2189–2200. PMLR.

Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. 2019. Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9268–9277.

Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Yihan Cao, Zihao Wu, Lin Zhao, Shaochen Xu, Fang Zeng, Wei Liu, and 1 others. 2025. Auggpt: Leveraging chatgpt for text data augmentation. IEEE Transactions on Big Data, 11(3):907–918.

Defence Science and Technology Laboratory. 2021. Relationship and entity extraction evaluation dataset (documents). Accessed: 2026-01-17.

Greg d’Eon, Jason d’Eon, James R Wright, and Kevin Leyton-Brown. 2022. The spotlight: A general method for discovering systematic errors in deep learning models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1962–1981.

Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, KDD’96, page 226–231. AAAI Press.

Sabri Eyuboglu, Maya Varma, Khaled Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Ré. 2022. Domino: Discovering systematic errors with cross-modal embeddings. arXiv preprint arXiv:2203.14960.

Hongyu Guo, Yongyi Mao, and Richong Zhang. 2019. Augmenting data with mixup for sentence classification: An empirical study. arXiv preprint arXiv:1905.08941.

Guangzeng Han, Weisi Liu, and Xiaolei Huang. 2025. Attributes as textual genes: Leveraging LLMs as genetic algorithm simulators for conditional synthetic data generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 19367–19389, Suzhou, China. Association for Computational Linguistics.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021. Debertav3: Improving deberta using electra-style pretraining with gradient-disentangled embedding sharing. Preprint, arXiv:2111.09543.

Sophie Henning, William Beluch, Alexander Fraser, and Annemarie Friedrich. 2023. A survey of methods for addressing class imbalance in deep-learning based natural language processing. In Proceedings ofthe 17th Conference ofthe European Chapter of the Association for Computational Linguistics, pages 523–540.

Dirk Hovy, Anders Johannsen, and Anders Søgaard. 2015. User review sites as a resource for large-scale sociolinguistic studies. In Proceedings of the 24th International Conference on World Wide Web, WWW ’15, page 452–461, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

Abiodun M. Ikotun, Absalom E. Ezugwu, Laith Abualigah, Belal Abuhaija, and Jia Heming. 2023. K-means clustering algorithms: A comprehensive review, variants analysis, and advances in the era of big data. Information Sciences, 622:178–210.

Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M Beery, Jure Leskovec, Anshul Kundaje, and 4 others. 2021. Wilds: A benchmark of in-the-wild distribution shifts. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 5637–5664. PMLR.

Wang Li, Guangzeng Han, Yuexin Wu, I.-Chan Huang, and Xiaolei Huang. 2025. Joint Imbalance Adaptation for Radiology Report Generation. Journal of Healthcare Informatics Research, 9(4):720–742.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020. Focal loss for dense object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2):318–327.

Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. 2021. Just train twice: Improving group robustness without training group

information. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 6781–6792. PMLR.

Weisi Liu, Guangzeng Han, and Xiaolei Huang. 2025. Examining and adapting time for multilingual classification via mixture of temporal experts. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6151–6166, Albuquerque, New Mexico. Association for Computational Linguistics.

Zihan Liu, Yan Xu, Tiezheng Yu, Wenliang Dai, Ziwei Ji, Samuel Cahyawijaya, Andrea Madotto, and Pascale Fung. 2020. Crossner: Evaluating cross-domain named entity recognition.

Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. 2021. Hatexplain: A benchmark dataset for explainable hate speech detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 14867–14875.

Daniel Müllner. 2011. Modern hierarchical, agglomerative clustering algorithms. ArXiv, abs/1109.2378.

Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. 2020. Learning from failure: Debiasing classifier from biased classifier. Advances in Neural Information Processing Systems, 33:20673– 20684.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, and 1 others. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Hanshu Rao, Guangzeng Han, and Xiaolei Huang. 2026. Model-agnostic meta learning for class imbalance adaptation. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 10442–10456, San Diego, California, United States. Association for Computational Linguistics.

Jiawei Ren, Cunjun Yu, Xiao Ma, Haiyu Zhao, Shuai Yi, and 1 others. 2020. Balanced meta-softmax for long-tailed visual recognition. Advances in neural information processing systems, 33:4175–4186.

Tim GJ Rudner, Ya Shi Zhang, Andrew Gordon Wilson, and Julia Kempe. 2024. Mind the gap: Improving robustness to subpopulation shifts with group-aware priors. In International Conference on Artificial Intelligence and Statistics, pages 127–135. PMLR.

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. 2019. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731.

Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. 2020. Distributionally robust neural networks. In International Conference on Learning Representations.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Improving neural machine translation models with monolingual data. In Proceedings of the 54th annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 86–96.

Mayira Sharif, Guangzeng Han, Weisi Liu, and Xiaolei Huang. 2026. Cultivating multidisciplinary AI workforce development on itiger GPU cluster: Practices and challenges. In Proceedings ofthe Practice and Experience in Advanced Research Computing 2026: Resilient Roots + Empowered Communities, Minneapolis, MN, USA, July 26-30, 2026, pages 84:1– 84:6. ACM.

Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. 2020. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. Advances in Neural Information Processing Systems, 33:19339–19352.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Nguyen Nhat Minh To, Paul F R Wilson, Viet Nguyen, Mohamed Harmanani, Michael Cooper, Fahimeh Fooladgar, Purang Abolmaesumi, Parvin Mousavi, and Rahul Krishnan. 2025. Diverse prototypical ensembles improve robustness to subpopulation shift. In Forty-second International Conference on Machine Learning.

Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From louvain to leiden: guaranteeing wellconnected communities. Scientific reports, 9(1):1– 12.

Taha ValizadehAslani, Yiwen Shi, Jing Wang, Ping Ren, Yi Zhang, Meng Hu, Liang Zhao, and Hualou Liang. 2024. Two-stage fine-tuning with chatgpt data augmentation for learning class-imbalanced data. Neurocomputing, 592:127801.

Nguyen Xuan Vinh, Julien Epps, and James Bailey. 2009. Information theoretic measures for clusterings comparison: is a correction for chance necessary? In Proceedings ofthe 26th annual international conference on machine learning, pages 1073–1080.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. 2024. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. Preprint, arXiv:2412.13663.

Jason Wei and Kai Zou. 2019. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. Preprint, arXiv:2309.07597.

Huaxiu Yao, Yu Wang, Sai Li, Linjun Zhang, Weixin Liang, James Zou, and Chelsea Finn. 2022. Improving out-of-distribution robustness via selective augmentation. In International Conference on Machine Learning, pages 25407–25437. PMLR.

Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, and Woomyoung Park. 2021. Gpt3mix: Leveraging large-scale language models for text augmentation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2225–2239.

Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, and Jiashi Feng. 2023. Deep long-tailed learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 45(9):10795–10816.

## A Data Imbalance Details

HumAID (Alam et al., 2021) contains 15,311 annotated tweets collected from five different natural disasters (e.g., Hurricane Irma 2017, Cyclone Idai 2019). These disaster events serve as natural subgroups, with distinct label distributions across humanitarian categories as shown in Figure 5. The tweets are labeled into humanitarian categories (e.g., Infrastructure and utility damage, Rescue volunteering or donation effort). The moderate class imbalance (IR = 14.8) and social media context test a model’s robustness to noisy, informal text with uneven class distributions.

HateXplain (Mathew et al., 2021) contains 14,995 annotated posts collected from Twitter and Gab across eight target community groups (e.g., African, Caucasian, Homosexual). These target communities serve as natural subgroups, with distinct label distributions across hate speech categories as shown in Figure 5. The posts are labeled into three categories: hatespeech, offensive, and normal. The relatively balanced class distribution (IR = 1.6) and social media context test a model’s ability to distinguish fine-grained boundaries between hate speech and offensive language in informal, user-generated text.

WWW2015 (Hovy et al., 2015) contains 15,449 business reviews collected from ten different companies. These company sources serve as natural subgroups, with distinct label distributions across rating levels as shown in Figure 5. The reviews are labeled into five rating categories (1-5 stars). The substantial class imbalance (IR = 27.4) and review domain test a model’s robustness to skewed sentiment distributions and domain-specific language patterns across different business contexts.

RE3D (Defence Science and Technology Laboratory, 2021) contains 931 annotated documents collected from seven different sources spanning news, government, and encyclopedic domains (e.g., BBC\_Online, CENTCOM, UK\_Government). These document sources serve as natural subgroups, with distinct entity distributions across defense and security contexts as shown in Figure 5. The documents are annotated with five entity types (e.g., Temporal, Organisation). The moderate class imbalance (IR = 16.3) and security domain test a model’s ability to recognize specialized entities in formal, domain-specific text with limited training data.

CrossNER (Liu et al., 2020) contains 5,327 annotated sentences collected from five specialized domains (Politics, Natural Science, Music, Literature, and Artificial Intelligence). These domains serve as natural subgroups, with distinct entity distributions across knowledge areas as shown in Figure 5. The sentences are annotated with five entity types (e.g., Location, Quantity, Temporal, Organisation, Person). The relatively balanced class distribution (IR = 2.0) and multi-domain setting test a model’s robustness to domain shift and its ability to generalize entity recognition across diverse topical contexts.

## B Baseline Details

Focal Loss (Lin et al., 2020) adds a modulating factor to focus on data samples with hard minority classes and down-weight well-classified (easy) instances. The resulting loss is formulated as $\mathcal { L } _ { \mathrm { F o c a l } } = - \alpha ( 1 - p _ { t } ) ^ { \gamma }$ log $p _ { t }$ , where $p _ { t }$ refers to a predicted probability for the true class. Following prior practice, we set the focusing parameter $\gamma = 2$ and adopt a class-balanced weighting schema for α.

![](images/7e51ed02910b9e81d30b81568a38b20cfeecdb29a6aa997c55d42a72cb209a06.jpg)  
Figure 5: Label distributions across subgroups for five datasets. Each polygon represents a subgroup (G1, G2, ...), and axes correspond to label indices. Values are shown on a logarithmic scale $( \log _ { 1 0 } )$

Group DRO (Sagawa et al., 2020) is a group-aware robust baseline that uses observed group labels and optimizes a worst-case objective min<sub>θ</sub> $\operatorname* { m a x } _ { g \in { \mathcal { G } } } { \mathcal { L } } _ { g } ( \theta )$ over predefined groups. In practice, it maintains a distribution over groups q and upweights groups with higher minibatch loss via an exponentiated-gradient update, then updates model parameters by minimizing the resulting group-reweighted loss. We implement Group DRO on top of the same ERM backbone and use training-set group annotations to compute pergroup minibatch losses and update q.

GEORGE (Sohoni et al., 2020) discovers latent subclasses by clustering learned representations within each class and then applies group-robust training over the inferred subclass labels. We follow its original example-level formulation and evaluate it on the three classification datasets.

Just Train Twice (JTT) (Liu et al., 2021) improves worst-group robustness without requiring training group labels via a two-stage reweighting scheme. It first trains an ERM identification model for T steps and collects the examples it misclassifies into an error set E. It then trains the final model by upweighting (equivalently, upsampling) examples in E by a factor $\lambda _ { \mathrm { u p } }$ under standard ERM. Hyperparameters $( T , \lambda _ { \mathrm { u p } } )$ are selected by worstgroup performance on a small group-annotated validation set.

Diverse Prototypical Ensembles (DPE) (To et al., 2025) improves worst-group robustness under subpopulation shift by training a diversified ensemble of distance-based prototypical classifiers on top of frozen ERM features. It first fits an ERM backbone, then replaces the linear head with multiple prototype heads that assign class probabilities by a softmax over distances to class prototypes in the embedding space. To avoid redundant members, DPE regularizes similarity among prototypes across ensemble heads, encouraging complementary decision rules that better cover minority subpopulations.

Class-Balanced LLM Augmentation (CB-LLM) (ValizadehAslani et al., 2024) is based on class-balanced ChatGPT augmentation for classimbalanced text classification. The original method first trains on class-balanced augmented data before standard fine-tuning, so minority classes receive stronger supervision early in training. We adopt this class-balancing principle as an augmentation baseline by prioritizing classes with fewer training samples under the global class distribution. We use Qwen3-32B to generate label-preserving samples for these selected classes, so this baseline tests whether the gain mainly comes from increasing minority-class sample counts.

AugGPT (Dai et al., 2025) uses ChatGPT to rephrase training samples into label-preserving variants for text data augmentation. The method increases training data by generating multiple semantically related versions of each input sentence. Following this idea, we build an LLM paraphrasing baseline by randomly selecting training samples and generating paraphrases with the original labels. We use Qwen3-32B as the generation model instead of ChatGPT, so this baseline tests whether the gain comes from adding generic LLM-generated data.

## C Implementation Details

To validate our approach, we implement all methods using HuggingFace Transformers (Wolf et al., 2020) and PyTorch (Paszke et al., 2019) and conduct experiments on the same data splits. All experiments were conducted on a server with an NVIDIA H100 GPU. We fine-tuned the microsoft/debertav3-base<sup>2</sup> model as the backbone for all classification and NER tasks. We conduct iterative training for 8 rounds, with 1 warmup round for initial model training. Each round trains for 1 epoch using the AdamW optimizer with learning rate $2 \times 1 0 ^ { - 5 }$ batch size 16, and maximum sequence length 512. The best model is selected based on validation combined score (average of micro- and macro-F1) across all rounds.

<table><tr><td rowspan="3">Method</td><td colspan="10">CLS</td><td colspan="6">NER</td></tr><tr><td colspan="3">HumAID</td><td colspan="3">WWW2015</td><td colspan="3">HateXplain</td><td colspan="3">RE3D</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>Focal</td><td>0.53</td><td>0.38</td><td>0.77</td><td>0.74</td><td>0.47</td><td>1.02</td><td>0.45</td><td>0.34</td><td>0.59</td><td>0.92</td><td>0.58</td><td>1.24</td><td>0.60</td><td>0.45</td><td>0.81</td></tr><tr><td>GroupDRO</td><td>0.74</td><td>0.59</td><td>1.38</td><td>1.08</td><td>0.70</td><td>2.17</td><td>0.69</td><td>0.51</td><td>0.94</td><td>1.27</td><td>0.86</td><td>2.38</td><td>0.80</td><td>0.64</td><td>1.52</td></tr><tr><td>GEORGE</td><td>0.85</td><td>0.53</td><td>1.04</td><td>0.72</td><td>0.91</td><td>1.10</td><td>0.78</td><td>0.61</td><td>0.97</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JTT</td><td>1.01</td><td>0.80</td><td>1.59</td><td>1.46</td><td>1.01</td><td>2.44</td><td>0.90</td><td>0.66</td><td>1.11</td><td>1.78</td><td>1.14</td><td>2.89</td><td>1.19</td><td>0.81</td><td>1.84</td></tr><tr><td>DPE</td><td>0.60</td><td>0.45</td><td>0.81</td><td>0.80</td><td>0.54</td><td>1.11</td><td>0.51</td><td>0.34</td><td>0.66</td><td>0.96</td><td>0.62</td><td>1.40</td><td>0.63</td><td>0.50</td><td>0.95</td></tr><tr><td>CB-LLM</td><td>0.70</td><td>0.52</td><td>1.01</td><td>0.98</td><td>0.63</td><td>1.46</td><td>0.59</td><td>0.42</td><td>0.77</td><td>1.14</td><td>0.82</td><td>1.68</td><td>0.76</td><td>0.57</td><td>1.19</td></tr><tr><td>AugGPT</td><td>0.88</td><td>0.61</td><td>1.22</td><td>1.16</td><td>0.77</td><td>1.67</td><td>0.71</td><td>0.56</td><td>0.91</td><td>1.34</td><td>0.97</td><td>1.89</td><td>0.96</td><td>0.71</td><td>1.31</td></tr><tr><td>AIA2 (Ours)</td><td>0.72</td><td>0.53</td><td>1.10</td><td>0.99</td><td>0.65</td><td>1.24</td><td>0.61</td><td>0.47</td><td>0.83</td><td>1.02</td><td>0.84</td><td>1.94</td><td>0.81</td><td>0.56</td><td>1.12</td></tr></table>

Table 5: Standard deviation (±σ) over 5 runs. M-F1 and µ-F1 denote macro-averaged and micro-averaged F1 scores, respectively. WGA denotes worst-group accuracy for CLS, and WG-F1 denotes worst-group F1 score for NER. GEORGE is evaluated only on the classification datasets.

For slice discovery, we employ the bge-largeen-v1.5 model<sup>3</sup> to generate semantic embeddings, which are combined with predictive representations using α = 0.1 weighting. We construct a k-nearest neighbor graph (k = 15) in the joint semantic-predictive space and apply Leiden clustering (Traag et al., 2019) with resolution 1.0 to identify latent slices. Only slices with size ≥ 100 samples are considered for augmentation to ensure statistical reliability. For data augmentation, we use Qwen3-32B (Team, 2025) as our generative model, with temperature 0.85 and top-p 0.95 to balance diversity and quality in generated samples. For each identified slice with distribution gap exceeding threshold $\tau = 0 . 1$ , we select hard examples based on their loss values and generate augmentations with the following constraints: (i) maximum 80 augmentations per slice with 2 augmentations per seed example, (ii) generated texts must maintain label consistency as verified by the current classifier, (iii) semantic diversity is enforced through cosine similarity filtering. The augmentation budget is allocated proportionally to the deficit mass with augmentation coefficient ρ = 0.2 and size

exponent $\beta = 0 . 8 .$

## D Prompts and Filtering

This appendix provides the generation prompts and candidate-filtering rules used in $\mathrm { \ A I A ^ { 2 } }$ . Fields enclosed in braces are instantiated at runtime.

## Classification prompt

You are a careful assistant that follows   
instructions strictly .   
You are generating data augmentation for   
a text classification task .   
Label of the seed sample : { label }   
Hard constraints :   
- Output ONLY one final text .   
- One line only . No quotes . No prefixes .   
No explanations .   
Max length : { max\_chars\_cap } characters   
Required transformations ( must do BOTH ):   
1) Sentence - structure transformation (   
change syntax / order ; do NOT just swap a   
few words ).   
2) Entity substitution with same - type   
entities ( person - > person , city - > city ,   
org ->org , group ->group , etc .).   
Notes :   
- You MAY introduce new details , but the   
final text must still belong to label   
'{ label }'.   
- Use the reference examples to borrow   
entities / structures , but do NOT copy any   
full sentence .   
Seed text :   
{ seed }   
Reference examples ( same label , high -   
loss ; for style / entity ideas ) :   
{ refs }   
Write the new text now :

## NER prompt

You follow hard constraints strictly .   
You are generating data augmentation for   
a token - level NER task .   
You will be given a tokenized sentence (   
tokens are already split ).   
Reference examples ( masked entities ; <   
ENT :TYPE > tokens appear ONLY here ; do   
NOT output them ):   
{ refs\_block }   
Hard constraints ( must satisfy all ):   
- Output ONLY one line .   
- Output must be space - separated tokens   
( SINGLE spaces ).   
- You MUST keep every placeholder token   
EXACTLY unchanged .   
- Each placeholder token must appear   
EXACTLY once in the output .   
- Placeholders must be standalone tokens   
: do NOT attach any punctuation / quotes   
to them (e.g., " \_\_ENT0\_\_ ," not "   
\_\_ENT0\_\_ ,").   
- Entities MUST be represented ONLY by   
placeholders (no entity surface forms ).   
- Rewrite the non - placeholder context   
substantially ; avoid copying long   
consecutive spans from the seed .   
- Keep punctuation tokenization style (   
punctuation is its own token if shown ).   
Placeholders ( types only ) :   
{ placeholders\_block }   
Seed template tokens ( with placeholders )   
{ seed\_tokens\_block }   
Output tokens now (space - separated , one   
line ):

## Candidate filtering

For each selected seed, we generate 15 candidates and retain at most two. For classification, we discard empty outputs, outputs longer than 280 characters, and exact duplicates, and then select candidates according to predictive consistency with the seed and semantic diversity. For NER, we discard candidates that do not preserve every placeholder exactly once, contain reference markers or original entity surface forms, duplicate another candidate, or share at least 60% of their placeholder-stripped 5-grams with the seed; among the remaining candidates, we prioritize semantic diversity and use predictive similarity to break ties.

## E Stability Analysis

To assess run-to-run stability, we repeat the experiments in Table 2 five times with different random seeds (42, 43, 44, 45, 46) and report the standard deviation (±σ) of all metrics in Table 5. The main observation is that $\mathrm { \ A I A ^ { 2 } }$ maintains stable average performance while improving subgroup robustness. For Macro-F1 and Micro-F1, its standard deviation stays below 1.0 point on most datasets, indicating limited variation across seeds. Worst-group metrics show larger variance, as they depend on the most difficult subgroup and are more sensitive to small sample changes. Even so, $\mathrm { \ A I A ^ { 2 } }$ remains more stable than JTT on all worst-group metrics; for example, its standard deviation is 1.24 on WWW2015 and 1.94 on RE3D, compared with 2.44 and 2.89 for JTT. These results indicate that iterative slice discovery and LLM-based augmentation do not introduce excessive seed sensitivity.

<table><tr><td rowspan="2">Setting</td><td colspan="3">HumAID</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>Semantic-only</td><td>68.0</td><td>76.5</td><td>70.0</td><td>72.2</td><td>72.3</td><td>62.5</td></tr><tr><td>α = 0.05</td><td>68.8</td><td>77.2</td><td>71.0</td><td>73.2</td><td>73.5</td><td>63.9</td></tr><tr><td>α = 0.1</td><td>69.4</td><td>77.6</td><td>71.8</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>α = 0.2</td><td>69.1</td><td>77.0</td><td>72.1</td><td>74.4</td><td>74.1</td><td>64.0</td></tr><tr><td>α = 0.5</td><td>69.7</td><td>76.5</td><td>71.8</td><td>73.2</td><td>74.9</td><td>62.4</td></tr><tr><td>α = 1.0</td><td>66.5</td><td>75.2</td><td>68.5</td><td>72.8</td><td>73.2</td><td>62.9</td></tr><tr><td>Prediction-only</td><td>64.2</td><td>73.1</td><td>65.5</td><td>70.5</td><td>71.8</td><td>61.8</td></tr></table>

Table 6: Sensitivity analysis of the weighting coefficient α in the joint semantic–predictive representation.
<table><tr><td rowspan="2">k in kNN graph</td><td colspan="3">HumAID</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>5</td><td>68.6</td><td>77.4</td><td>70.1</td><td>72.8</td><td>74.0</td><td>62.2</td></tr><tr><td>10</td><td>68.9</td><td>77.2</td><td>71.4</td><td>73.6</td><td>73.9</td><td>63.3</td></tr><tr><td>15</td><td>69.4</td><td>77.6</td><td>71.8</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>20</td><td>69.7</td><td>77.5</td><td>71.6</td><td>74.4</td><td>74.5</td><td>63.2</td></tr></table>

Table 7: Sensitivity analysis of the number of nearest neighbors in the kNN graph.

## F Hyperparameter Sensitivity Analysis

We conduct a sensitivity analysis of $\mathrm { \ A I A ^ { 2 } }$ for four key hyperparameters: the weighting coefficient α in the joint semantic–predictive representation, the number of nearest neighbors k in the kNN graph, the Leiden resolution parameter, and the number of generated candidates per seed sample. The analysis is performed on two representative datasets: HumAID for CLS and CrossNER for NER. These hyperparameters cover the two main stages of $\mathrm { \ A I A ^ { 2 } }$ : latent slice discovery and LLMbased augmentation. For each experiment, we vary one hyperparameter while keeping all other settings fixed. The default setting used in the main experiments is $\alpha = 0 . 1 , k = 1 5$ , resolution = 1.0, and #Candidates = 20.

<table><tr><td rowspan="2">Resolution</td><td colspan="2">HumAID</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td>µ-F1 WGA</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>0.50</td><td>68.4</td><td>77.3 70.9</td><td>73.0</td><td>73.7</td><td>62.9</td></tr><tr><td>0.75</td><td>69.4</td><td>77.1 71.1</td><td>73.6</td><td>75.0</td><td>63.1</td></tr><tr><td>1.00</td><td>69.4</td><td>77.6 71.8</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>1.25</td><td>69.1</td><td>77.5 71.4</td><td>74.3</td><td>74.5</td><td>63.6</td></tr><tr><td>1.50</td><td>69.0</td><td>77.4 70.6</td><td>73.2</td><td>74.2</td><td>62.4</td></tr></table>

Table 8: Sensitivity analysis of the Leiden resolution parameter.
<table><tr><td rowspan="2">#Candidates</td><td colspan="3">HumAID</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td> $\mu { - } \mathrm { F } 1$ </td><td>WGA</td><td>M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td>WG-F1</td></tr><tr><td>10</td><td>68.4</td><td>77.5</td><td>70.8</td><td>72.9</td><td>74.2</td><td>62.5</td></tr><tr><td>15</td><td>69.2</td><td>77.3</td><td>71.3</td><td>73.8</td><td>73.6</td><td>63.4</td></tr><tr><td>20</td><td>69.4</td><td>77.6</td><td>71.8</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>30</td><td>69.9</td><td>77.6</td><td>71.5</td><td>74.5</td><td>75.0</td><td>63.5</td></tr></table>

Table 9: Sensitivity analysis of the number of generated candidates per seed sample.

Tables 6–9 show that $\mathrm { \ A I A ^ { 2 } }$ is stable under moderate hyperparameter changes. For α, semanticonly corresponds to $\alpha = 0$ , whereas predictiononly represents the $\alpha  \infty$ limit. Values between 0.05 and 0.2 outperform semantic-only slicing across all metrics on both datasets. The default $\alpha = 0 . 1$ remains competitive across tasks, while larger values yield less consistent results and prediction-only performs worst overall. For k, both datasets remain stable across $k \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ the default value $k = 1 5$ gives the best or near-best worst-group performance, while $k = 2 0$ slightly improves Macro-F1. For the Leiden resolution parameter, resolution = 1.0 gives the best results on HumAID and the best worst-group score on CrossNER. The tested resolutions lead to small performance changes, showing that $\mathrm { \ A I A ^ { 2 } }$ does not rely on a narrow clustering granularity. For the number of generated candidates, a larger candidate pool improves Macro-F1 in some cases, but $\# C a n d i d a t e s = 2 0$ gives the best worst-group performance on both datasets. Overall, the default setting remains competitive across tasks and usually provides the strongest worst-group results. These results suggest that the gains of $\mathrm { \ A I A ^ { 2 } }$ are not driven by dataset-specific tuning or a fragile hyperparameter choice.

## G Computational Overhead

We report training time and peak GPU-memory cost on one representative dataset per task:

<table><tr><td rowspan="2">Baselines</td><td colspan="2">WWW2015</td><td colspan="2">CrossNER</td></tr><tr><td>Time (s)</td><td>Memory (GB)</td><td>Time (s)</td><td>Memory (GB)</td></tr><tr><td>Focal</td><td>1143</td><td>17.6</td><td>1292</td><td>16.2</td></tr><tr><td>GroupDRO</td><td>1262</td><td>17.2</td><td>1666</td><td>18.6</td></tr><tr><td>JTT</td><td>2837</td><td>21.0</td><td>2389</td><td>23.8</td></tr><tr><td>DPE</td><td>1755</td><td>20.6</td><td>1973</td><td>21.0</td></tr><tr><td>CB-LLM</td><td>3193</td><td>74.2</td><td>3372</td><td>78.0</td></tr><tr><td> $\mathbf { A u g G P T }$ </td><td>3760</td><td>72.4</td><td>3789</td><td>75.4</td></tr><tr><td> $\mathrm { \ A I A ^ { 2 } }$ </td><td>2699</td><td>71.8</td><td>3022</td><td>76.6</td></tr></table>

Table 10: Efficiency comparison on different datasets.

WWW2015 for CLS and CrossNER for NER. All methods are evaluated with batch size 16 on an H100 GPU. Table 10 shows that $\mathrm { \ A I A ^ { 2 } }$ introduces additional overhead over lightweight training baselines such as Focal Loss and GroupDRO, due to latent slice discovery and LLM-based targeted augmentation. However, $\mathrm { \ A I A ^ { 2 } }$ is more efficient than generic LLM augmentation baselines in training time. On WWW2015, it reduces training time by 15.5% compared with CB-LLM and 28.2% compared with AugGPT; on CrossNER, it reduces training time by 10.4% and 20.2%, respectively. Its peak memory cost is also comparable to LLM augmentation baselines: $\mathrm { \ A I A ^ { 2 } }$ uses less memory than CB-LLM on both datasets, less than AugGPT on WWW2015, and slightly more than AugGPT on CrossNER. These results show that $\mathrm { \ A I A ^ { 2 } }$ provides a practical cost–benefit tradeoff: it adds expected overhead over simple training baselines while remaining more time-efficient than general-purpose LLM augmentation methods.

## H Additional Generated Data Analysis

To complement the WWW2015 analysis in Section 4.2, we further examine the relationship between deficit reduction and subgroup-label accuracy gain on the other two CLS datasets, HumAID and HateXplain. As shown in Figure 6, both datasets show a clear positive association between the reduction of subgroup-label deficits and the improvement of subset-level accuracy. On HumAID, the trend is significantly positive across 35 subgroup-label subsets, with Spearman’s $\rho =$ 0.700 $( p ~ = ~ 2 . 9 3 ~ \times ~ 1 0 ^ { - 6 } )$ and an OLS fit of $R ^ { 2 } = 0 . 3 2 2$ . On HateXplain, the association is also significant across 24 subsets, with Spearman’s $\rho = 0 . 7 1 1 ( p = 9 . 7 2 \times 1 0 ^ { - 5 } )$ and $R ^ { 2 } = 0 . 4 2 7$ . In addition, 14 subsets on HumAID and 15 subsets on HateXplain fall into the positive-repair region, where deficit reduction and accuracy gain are both positive. These results are consistent with the main

![](images/c8a376b84ab03dcd2b22e0f9efc320cc374d1bba7a6185e4266500eaae2d4b3d.jpg)  
(a) HumAID

![](images/9d6c6cf2353faaaf77c65014eb766c98e5eef1798f60525d0f534023a68cf0db.jpg)  
(b) HateXplain

Figure 6: Additional subgroup-label accuracy gain versus deficit reduction on HumAID and HateXplain.
<table><tr><td rowspan="2">Method</td><td colspan="2">HumAID</td><td colspan="2">CrossNER</td></tr><tr><td>Added</td><td>Increase</td><td>Added</td><td>Increase</td></tr><tr><td>AugGPT</td><td>1,493</td><td>13.9%</td><td>2,196</td><td>51.5%</td></tr><tr><td>CB-LLM</td><td>997</td><td>9.3%</td><td>1,913</td><td>44.9%</td></tr><tr><td>Semantic Cluster-Gap</td><td>1,281</td><td>12.0%</td><td>1,952</td><td>45.8%</td></tr><tr><td> $\mathrm { \ A I A ^ { 2 } }$ </td><td>1,073</td><td>10.0%</td><td>1,643</td><td>38.6%</td></tr></table>

Table 11: Retained augmentation volume across generative methods on HumAID and CrossNER.

WWW2015 finding and further support that $\mathrm { \ A I A ^ { 2 } }$ improves subgroup robustness by repairing local subgroup-label deficits, rather than relying only on untargeted data expansion.

Table 11 reports the filtered samples retained for training; raw candidates rejected during quality control are excluded, and the increase is measured relative to the original training-set size. Semantic Cluster-Gap corresponds to the semantic-only setting in Table 6: it forms semantic communities and targets labels that are underrepresented relative to the global label distribution. $\mathrm { \ A I A ^ { 2 } }$ retains 1,073 samples on HumAID and 1,643 on Cross-NER, corresponding to training-set increases of 10.0% and 38.6%, respectively. It retains fewer samples than AugGPT and Semantic Cluster-Gap on both datasets and fewer than CB-LLM on Cross-NER, while CB-LLM retains slightly fewer samples on HumAID. These comparisons indicate that the performance gains of $\mathrm { \ A I A ^ { 2 } }$ do not arise from consistently using a larger augmentation volume.

## I Additional Ablations

Representation and selection We further examine the representations used for latent slice discovery and the signals used for seed selection on WWW2015 and CrossNER, while keeping the remaining pipeline and augmentation budget unchanged. Semantic-only and prediction-only slicing use only semantic embeddings or model-output distributions, respectively. Deficit-only selection randomly samples within deficit labels, whereas high-loss-only selection ranks all instances within each slice without using deficit labels. As shown in Table 12, the full setting achieves the best results across all metrics on both datasets. It improves worst-group performance over the single-signal slice variants by 3.2–4.3 points on WWW2015 and 1.3–2.0 points on CrossNER, and over the singlesignal selection variants by 1.9–2.7 and 1.0–1.2 points, respectively. These results support the joint designs used in both stages.

<table><tr><td rowspan="2">Setting</td><td colspan="3">WWW2015</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>Full AIA²</td><td>57.6</td><td>87.5</td><td>80.9</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>Semantic-only slicing</td><td>55.4</td><td>85.3</td><td>77.7</td><td>72.2</td><td>72.3</td><td>62.5</td></tr><tr><td>Prediction-only slicing</td><td>54.3</td><td>85.8</td><td>76.6</td><td>70.5</td><td>71.8</td><td>61.8</td></tr><tr><td>Deficit-only selection</td><td>56.1</td><td>86.4</td><td>79.0</td><td>72.5</td><td>73.7</td><td>62.8</td></tr><tr><td>High-loss-only selection</td><td>56.5</td><td>86.3</td><td>78.2</td><td>72.0</td><td>73.3</td><td>62.6</td></tr></table>

Table 12: Ablations of slice representation and seed selection.

Clustering methods We include this experiment to examine whether the effectiveness of $\mathrm { \ A I A ^ { 2 } }$ depends on the specific clustering algorithm used for latent slice discovery. Table 13 compares the default Leiden community detection method (Traag et al., 2019) with K-Means (Ikotun et al., 2023), DBSCAN (Ester et al., 1996), and Agglomerative clustering (Müllner, 2011). Overall, Leiden provides the most consistent performance across datasets. K-Means performs comparably on classification tasks but drops more clearly on NER, suggesting that centroid-based clustering may be less effective for irregular error regions. DBSCAN is less stable, likely because the joint semantic– predictive space has varying local densities. Agglomerative clustering shows mixed results, with occasional gains on CrossNER but weaker performance on several other datasets. These results suggest that $\mathrm { \ A I A ^ { 2 } }$ is broadly robust to reasonable clustering choices, while graph-based community detection remains the most reliable default for iterative slice-aware augmentation.

<table><tr><td rowspan="3">Method</td><td colspan="9">CLS</td><td colspan="6">NER</td></tr><tr><td colspan="3">HumAID</td><td colspan="3">WWW2015</td><td colspan="3">HateXplain</td><td colspan="3">RE3D</td><td colspan="3">CrossNER</td></tr><tr><td>M-F1</td><td> $\mu { - } \mathrm { F } 1$ </td><td></td><td>WGA M-F1</td><td>µ-F1</td><td></td><td>WGA M-F1</td><td>µ-F1</td><td>WGA</td><td>M-F1</td><td>µ-F1</td><td></td><td>WG-F1 M-F1</td><td>µ-F1</td><td>WG-F1</td></tr><tr><td>Leiden</td><td>69.4</td><td>77.6</td><td>71.8</td><td>57.6</td><td>87.5</td><td>80.9</td><td>64.6</td><td>66.2</td><td>57.7</td><td>66.5</td><td>66.3</td><td>60.5</td><td>74.4</td><td>74.9</td><td>63.8</td></tr><tr><td>K-Means</td><td>69.4</td><td>77.6</td><td>71.7</td><td>55.4</td><td>87.1</td><td>80.3</td><td>63.4</td><td>65.0</td><td>57.7</td><td>63.9</td><td>63.4</td><td>55.3</td><td>73.0</td><td>72.5</td><td>61.8</td></tr><tr><td>DBSCAN</td><td>68.1</td><td>77.2</td><td>71.1</td><td>55.7</td><td>86.6</td><td>80.9</td><td>62.6</td><td>64.2</td><td>55.8</td><td>60.8</td><td>61.5</td><td>55.7</td><td>72.2</td><td>70.6</td><td>60.4</td></tr><tr><td>Agglomerative</td><td>68.4</td><td>76.9</td><td>70.7</td><td>52.7</td><td>86.1</td><td>79.4</td><td>61.5</td><td>63.4</td><td>55.8</td><td>62.9</td><td>63.5</td><td>59.3</td><td>74.3</td><td>73.1</td><td>64.1</td></tr></table>

Table 13: Clustering method robustness for latent slice discovery. Leiden is the default clustering method, while K-Means, DBSCAN, and Agglomerative clustering are used as alternatives.