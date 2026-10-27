# Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation

Haocheng Mei<sup>1</sup>, Yuchuan Tian<sup>1</sup>, Ye Yuan<sup>2</sup>, Enhua Wu<sup>3</sup> Chao Xu<sup>1</sup>, Allen Chen<sup>4</sup>, Kai Han<sup>5</sup>, Yunhe Wang<sup>5</sup>

<sup>1</sup>Peking University   
<sup>2</sup>SUIBE   
<sup>3</sup>University of Macau   
<sup>4</sup>Tsinghua University   
<sup>5</sup>TokenRhythm Technologies

## Abstract

JEV decision models pair a language model backbone with a lightweight scoring head to select among natural-language options. Direct selection compares candidate answers within one request, whereas entailment reranking evaluates each candidate in a separate natural language inference (NLI) request over fixed relation options. We examine whether the readout and adaptation choices used for direct selection remain effective for cross-request reranking. We combine a mean-span readout with selective backbone adaptation. Mean-span pooling replaces the option-ending pointer with an average of the contextualized states inside each NLI option, without adding head parameters. Selective adaptation retains low-rank updates in attention and gated DeltaNet projections while removing feed-forward adapters. On MC11, which averages accuracy over eleven reranking configurations from seven sources, selective adaptation improves NeoHorse-1-4B accuracy over the all-module baseline while cutting trainable parameters by more than half, and mean-span pooling yields further consistent gains at the same parameter count. Project page: https://github.com/Mei-TN/Jev-reranking.

## 1. Introduction

JEV models turn a natural-language state into a structured decision that software can act on. Their typed interfaces include Choice, which selects among described options; Noul, which answers a binary question; and Score, which assigns probabilities to ordered levels [43, 26]. Open implementations such as NeoHorse-Jev and Kev attach a small bilinear scoring head to an adapted language model backbone [41, 31], encoding the context and options in one forward pass and using the hidden state at each option’s closing delimiter to produce a distribution over the candidates.

These models can answer multiple-choice questions either by comparing all candidate answers within one request (direct selection) or by evaluating each candidate in a separate natural language inference (NLI) request (entailment reranking). Reranking provides a fixed scoring interface for externally supplied answers: their number and wording can vary while the head continues to score entailment, neutral, and contradiction. The question serves as the premise and each candidate answer enters the instruction as a hypothesis; selection then compares entailment probabilities across requests. Because the NLI option text is fixed while its contextualized states vary with the candidate hypothesis, designs suited to comparing answers within one request may be less effective when candidates are scored independently. We therefore combine mean-span pooling over the interior option states with selective backbone adaptation, gathering information from more positions while updating fewer backbone projections.

The two design choices jointly adjust how option representations are extracted and how they are learned. First, a mean-span readout replaces the option-ending delimiter pointer with the mean of the contextualized states inside each NLI option. Because the bilinear head is affine in the option states under a shared decision query, pooling computes an exact average of pre-softmax token-level scores and distributes the direct scoring gradient across the span without adding head parameters. Second, selective backbone adaptation retains low-rank (LoRA) updates in the attention and gated DeltaNet projections that participate in contextual interaction, removing feed-forward network (FFN) adapters while keeping the pretrained FFN weights in the forward pass. Because the head and adapters shape the option representations together, we evaluate both designs under a common one-epoch, lower-learning-rate recipe that avoids the accuracy loss observed with longer training or higher adapter rates.

Our experiments on NeoHorse-1-4B show that the scoring interface changes the relative performance of option readouts, while reducing adaptation capacity can improve reranking accuracy. With checkpoints and questions held fixed on PIQA and SocialIQA, the pointer leads on average under direct selection, while pooling leads under entailment reranking. On the eleven-configuration MC11 benchmark, the one-epoch, lower-rate recipe establishes an all-module pointer reference at 50.28% accuracy. Selective adaptation raises this to 52.49% while reducing trainable parameters by 55.2% (from 131.2M to 58.8M), and adding mean-span pooling lifts accuracy further to 53.39% at the same parameter count. Both steps improve MC11 across all four fold–seed settings. Under all-module adaptation, mean-span pooling (51.00%) also outperforms a pointer at the final content token (49.11%). The individual design choices also improve PIQA and CommonsenseQA reranking, whereas the combined model loses accuracy on the JevBench and Kev direct-decision suites.

## 2. Readout and Adaptation Design

## 2.1. Typed requests and candidate reranking

A request contains a state �, an instruction $u ,$ and named options $O = \left\{ o _ { 1 } , \dots , o _ { K } \right\}$ . Each option comprises a label and description between explicit delimiters. The instruction precedes the options; option isolation is disabled, so an option state can attend to the instruction and earlier options. The model returns a distribution over the options.

Direct selection chooses the highest-probability option in one request. For entailment reranking, the state supplies the premise, and each candidate answer � is rendered as a hypothesis in the instruction of a separate request. Its three options are entailment, neutral, and contradiction. We select

$$
\begin{array} { c } { { \hat { a } = \arg \operatorname* { m a x } _ { a } p _ { E } ( x , a ) , \hfill } } \\ { { \hfill } } \\ { { p _ { E } = \frac { e ^ { z _ { E } } } { e ^ { z _ { E } } + e ^ { z _ { N } } + e ^ { z _ { C } } } , } } \end{array}\tag{1}
$$

where $z _ { E } , z _ { N } , z _ { C }$ are the three option logits. The head reads the NLI labels and descriptions. Although this text is shared across candidates, its hidden states depend on the preceding

question in the state and the candidate hypothesis in the instruction. Each request therefore supplies a distinct contextual representation of the three relations (Figure 1).

Reranking uses the full probability of entailment, even when several candidates receive entailment as their highest-scoring label. Equivalently,

$$
p _ { E } = \sigma \big ( z _ { E } - \log ( e ^ { z _ { N } } + e ^ { z _ { C } } ) \big ) ,\tag{2}
$$

where $\sigma$ is the logistic function. Candidates are ordered by an entailment margin against the combined alternatives. A change to the three scores can alter candidate ordering without changing the winning NLI label within individual requests.

![](images/fb590b5c80bba1f895c8fac96bc5c9a9d83aeb3d0ff3934b92946a311faee875.jpg)  
Repeat for each candidate; select ˆ� = arg max<sub>�</sub> $p _ { E } ( x , a )$

Figure 1 | The final reranker. The candidate answer enters the instruction; pooling reads the three NLI option spans. The shared head scores those spans against the decision query, and the outer comparison selects an answer using its entailment probability.

## 2.2. Option-ending and mean-span readouts

Let $h _ { d } \in \mathbb { R } ^ { d }$ be the hidden state at the final decision token, which follows all options, and $h _ { i } \in \mathbb { R } ^ { d }$ the state at the delimiter closing option �. The pointer uses a shared affine bilinear head [44, 14]:

$$
q = W _ { q } h _ { d } + b _ { q } , \quad k _ { i } = W _ { k } h _ { i } + b _ { k } ,\tag{3}
$$

$$
s _ { i } = \frac { q ^ { \top } k _ { i } } { \sqrt { d _ { \mathrm { h e a d } } } } , \quad p _ { i } = \frac { e ^ { s _ { i } } } { \sum _ { j } e ^ { s _ { j } } } ,\tag{4}
$$

with $d _ { \mathrm { h e a d } } = 2 5 6$ . Training minimizes cross-entropy against the target option distribution, averaging questions within each record.

Mean-span pooling replaces $h _ { i }$ with

$$
\bar { h } _ { i } = \frac { 1 } { \vert S _ { i } \vert } \sum _ { j \in S _ { i } } h _ { j } ,\tag{5}
$$

where $S _ { i }$ contains the option’s interior label and description tokens, excluding both delimiters. The last-content control instead takes the final state in $S _ { i }$ . Both rules fall back to the closing delimiter for empty content. All three readouts use the same query and 512(� + 1) head parameters, and each model is trained and evaluated with its own readout.

The affine projection makes the pooled score an exact average of the contextual token scores. For a nonempty span,

$$
\begin{array} { r l r } { s _ { i } ^ { \mathrm { p o o l } } = \frac { q ^ { \top } \left( W _ { k } \bar { h } _ { i } + b _ { k } \right) } { \sqrt { d _ { \mathrm { h e a d } } } } } & { } & \\ { = \frac { 1 } { | S _ { i } | } \displaystyle \sum _ { j \in S _ { i } } \frac { q ^ { \top } \left( W _ { k } h _ { j } + b _ { k } \right) } { \sqrt { d _ { \mathrm { h e a d } } } } . } & { } & \end{array}\tag{6}
$$

The averaging takes place before the option softmax, with span-length normalization giving each option one mean score. All token contributions use the same key projection and decision query.

Pooling also changes the direct training signal. Holding the query fixed, each interior state receives the score gradient

$$
\frac { \partial s _ { i } ^ { \mathrm { p o o l } } } { \partial h _ { j } } = \frac { W _ { k } ^ { \top } q } { | S _ { i } | \sqrt { d _ { \mathrm { h e a d } } } } , \qquad j \in S _ { i } .\tag{7}
$$

The pointer supplies this direct path through one selected state; pooling supplies it through every interior state. Contextualization provides additional paths through the backbone in both cases.

## 2.3. Selective adaptation

We jointly train the decision head and LoRA adapters [18], comparing two scopes: all-module adaptation covers attention, gated DeltaNet, and feed-forward network (FFN) projections, whereas selective adaptation (no-FFN) retains adapters only in attention and DeltaNet. Both configurations keep every pretrained FFN layer in the forward computation. Crossing these scopes with the pointer and mean-span readouts lets us compare the readout choices under both forms of adaptation while keeping head size and adapter rank fixed.

At inference, pooling averages hidden states already produced by the backbone before applying the key projection (Equation 6). Each option therefore requires one projected key, and reranking continues to use one backbone request per answer. Selective adaptation reduces the parameters requiring gradients and optimizer states during training, while the pretrained backbone remains intact.

## 3. Experimental Setup

Models and training. Our main experiments use NeoHorse-1-4B [30], a post-trained, text-only repackaging of Qwen3.5-4B whose backbone combines attention and gated DeltaNet layers [49]. The reference recipe trains the decision head and rank-64 adapters jointly for one epoch with a OneCycle schedule [37], using a peak learning rate of 1 $. 2 5 { \times } \dot { 1 } 0 ^ { - 5 }$ for both parameter groups. The learning-rate diagnostics also test $2 . 5 { \times } 1 0 ^ { - 5 }$ , and all comparisons evaluate the final weights of each run. A separate pointer–pooling comparison on Qwen3-4B [48] examines the readout change with another backbone.

The training mixture combines web decisions and policy tasks with sentiment, topic, intent, and entailment classification. Public-source labels and model-generated reference answers are converted to a shared request format while retaining each question’s native options and target distribution. MC11 reuses the scorer learned from this mixture: each candidate answer becomes a separate NLI request, and its entailment probability determines its rank among the candidates.

To keep related records together, we assign whole groups to splits before forming eight training pools, which contain 17,661 records and 24,001 questions. Each fold trains on seven pools; crossing two folds with two random seeds gives four matched settings. Within a setting, each configuration is trained afresh with compatible initial tensors, data order, and augmentation streams matched across the comparison. The learning-rate, scope, and budget studies reuse the same lower-rate, one-epoch reference pointers, giving their contrasts a common reference within each setting. Appendix A gives the grouping rules, augmentation procedure, and complete training configuration.

Evaluation. MC11 averages candidate-reranking accuracy across ARC-Easy and ARC-Challenge [10]; four- and ten-option GSM8K [11]; GPQA and MMLU with and without demonstrations [33, 16]; HellaSwag [51]; WinoGrande [35]; and synthetic chess. Each configuration receives equal weight and scores its candidates through Equation 1. The paired configurations of GSM8K, GPQA, and MMLU reuse their respective question populations, changing the candidate count or demonstration context.

Candidate construction preserves the supplied answers for multiple-choice tasks. GSM8K pairs the numerical answer with three or nine generated distractors; synthetic chess pairs one legal move with three illegal moves. WinoGrande fills the sentence gap, and HellaSwag uses the candidate ending as the NLI hypothesis.

The additional reranking evaluations use PIQA and CommonsenseQA [6, 39]; fixedcheckpoint PIQA and SocialIQA tests compare both interfaces on the same questions [36]. JevBench equally averages 18 direct-decision families over 231 questions [3]. Kev equally averages six development/test selections from its decision suite and two transfer suites [31].

Reporting. We report accuracy in percent and paired changes in percentage points (pp). For each contrast, we first calculate the difference within each of the four settings and then report its mean and descriptive SE; SEs for absolute accuracy are calculated directly from the four model scores. These SEs summarize variation across settings that share training pools and evaluation questions. Appendix D examines sensitivity to source weighting, omission of individual sources, and repetition of the reference run.

## 4. Readout Behavior Across Decision Interfaces

We compare pointer and pooling models under both request interfaces to test whether the relative performance of the readouts changes with how answers are scored. Each checkpoint is held fixed across interfaces, and both interfaces use the same PIQA and SocialIQA questions. PIQA contains 1,838 validation questions and uses eight higher-rate checkpoints. SocialIQA contains 1,954 development questions and uses eight checkpoints at each of two training budgets.

<table><tr><td></td><td colspan="2">Direct accuracy (%)</td><td colspan="2">NLI accuracy (%)</td><td colspan="2">Pooling minus pointer (pp)</td></tr><tr><td>Population and training</td><td>Pointer</td><td>Pooling</td><td>Pointer</td><td>Pooling</td><td>Direct</td><td>NLI</td></tr><tr><td>PIQA, higher-rate models</td><td>79.33</td><td>79.20</td><td>65.15</td><td>73.22</td><td>-0.12±1.41</td><td>+8.07±3.76</td></tr><tr><td>SocialIQÃ, one epoch</td><td>73.76</td><td>71.43</td><td>61.12</td><td>61.40</td><td>-2.33±0.64</td><td>+0.28±0.72</td></tr><tr><td>SocialIQA, two epochs</td><td>73.67</td><td>71.81</td><td>56.82</td><td>58.84</td><td>-1.86±0.23</td><td>+2.02±1.20</td></tr></table>

Table 1 | The request interface changes the readout comparison on the same questions and fixed checkpoints. Accuracies are four-setting means; differences include descriptive SEs. Neither source enters MC11.

On PIQA, the readouts are nearly tied under direct selection, whereas pooling leads the pointer under reranking: the relative difference moves from −0.12 to +8.07 pp (Table 1). SocialIQA shows the same direction of interface interaction at both budgets, with a positive

interaction in every setting. Direct selection has higher mean absolute accuracy on both sources;   
within reranking, pooling becomes the stronger readout on average.

The readout comparison also depends on the NLI option wording. Across 18 combinations of three descriptions and six label orders on PIQA, pooling’s mean advantage is +3.18 ± 3.27 pp; one description set instead favors the pointer by 1.80 pp after averaging over orders (Appendix B.4).

## 5. Reranking Accuracy and Parameter Efficiency

The core comparisons use the same one-epoch recipe, with peak learning rates of $1 . 2 5 { \times } 1 0 ^ { - 5 }$ for both the head and adapters and adapter rank 64. We first examine the combined design, then use controls on adapter allocation, readout, and training to explain the choices.

## 5.1. Combining Selective Adaptation and Span Pooling

Combining the two choices gives the highest MC11 accuracy among the four matched configurations, with less than half the trainable parameters of the all-module models (Table 2). Starting from the all-module pointer at 50.28%, removing FFN adapters raises accuracy to 52.49%, a paired gain of $2 . 2 1 \pm 0 . 6 6$ pp. Mean-span pooling then raises it to 53.39% at the same parameter count, adding $0 . 9 1 \pm 0 . 3 5$ pp. Both steps improve MC11 in every fold–seed setting. Removing FFN adapters reduces trainable parameters from 131.17M to 58.82M, a 55.2% reduction (Table 3).

<table><tr><td rowspan="2">Readout</td><td rowspan="2">Adapter scope</td><td rowspan="2">No-FFN</td></tr><tr><td>All modules</td></tr><tr><td>Pointer</td><td>50.28±1.02</td><td>52.49±0.46</td></tr><tr><td>Mean-span</td><td>51.00±0.21</td><td> $5 3 . 3 9 \pm 0 . 2 4$ </td></tr><tr><td>Pool – pointer (pp)</td><td>+0.73±1.07</td><td>+0.91±0.35</td></tr><tr><td>Trainable parameters (M)</td><td>131.17</td><td>58.82</td></tr></table>

Table 2 | MC11 accuracy (%) and trainable parameter count for the core readout-byscope comparison. All four models use one epoch, lower head and adapter rates, and rank 64. Accuracies and paired changes are four-setting means with descriptive SEs. Both readouts have the same parameter count within each scope.

<table><tr><td>Parameter group</td><td>All modules</td><td>No-FFN</td></tr><tr><td>Attention adapters</td><td>12.58</td><td>12.58</td></tr><tr><td>DeltaNet adapters</td><td>44.92</td><td>44.92</td></tr><tr><td>FFN adapters</td><td>72.35</td><td>0</td></tr><tr><td>Bilinear head</td><td>1.31</td><td>1.31</td></tr><tr><td>Total trainable</td><td>131.17</td><td>58.82</td></tr></table>

Table 3 | Trainable parameters in millions at rank 64, rounded independently. The pretrained backbone weights are frozen. Both readouts use the same bilinear head.

The final combination gains about 3.1 pp over the reference pointer. Its readout-by-scope interaction is small (+0.18 ± 0.74 pp), consistent with approximately additive gains from span aggregation and scope restriction.

## 5.2. Adapter Scope and Parameter Budget

FFN adapters account for 72.35M trainable parameters, more than half of the all-module adapter budget. Removing them improves the pointer by $2 . 2 1 \pm 0 . 6 6$ pp and the pooling model by $2 . 3 9 \pm 0 . 3 5 \mathrm { p p }$ , showing that the scope change helps with either readout.

A similar parameter budget can instead be distributed across all projections at a lower rank. The all-module rank-28 pointer uses 58.12M trainable parameters and reaches 52.40%, close to the no-FFN rank-64 pointer’s 52.49%. Both allocations improve over the larger all-module reference. The no-FFN allocation uses the smaller budget while retaining rank 64 in attention and DeltaNet, and combines with pooling to give the strongest configuration in Table 2.

## 5.3. Option Aggregation and Scoring Head

To examine the role of aggregation, we compare mean-span pooling with a different single-token position and a larger scoring head. The last-content control reads the final token inside each option, retaining the same bilinear head and all-module training recipe. It reaches 49.11% MC11, below both the option-ending pointer and mean-span pooling at 51.00%. Pooling exceeds this control by $1 . 8 9 \pm 0 . 9 7$ pp, supporting aggregation across the span over either tested pointer position. The largest gains over last-content occur on HellaSwag (8.10 pp) and the two ARC tasks; Appendix B gives the complete task profile.

Keeping the head simple. Adding a two-layer residual correction to the all-module pointer increases its trainable parameter count by about 0.13M and yields 50.33% MC11, close to the reference pointer and its repeated run. Under this recipe, the residual branch provides little additional benefit, so the final design retains the bilinear head and changes the option representation supplied to it. Appendix C examines the residual branch through inferenceremoval and training controls.

## 5.4. Training Recipe and Reference Quality

Because the head and adapters are trained jointly, the reference model’s quality depends on how strongly the backbone is updated. We vary the head and adapter learning rates independently and compare one- and two-epoch schedules, keeping the all-module, rank-64 architecture fixed.

<table><tr><td></td><td colspan="2">Adapter peak rate</td><td>Higher</td></tr><tr><td>Head peak rate</td><td>Lower</td><td></td><td> $4 6 . 8 1 { \scriptstyle \pm 2 . 3 7 }$ </td></tr><tr><td>Lower Higher</td><td> $5 0 . 2 8 { \pm } 1 . 0 2 $   $4 9 . 6 6 { \pm } 1 . 2 4 $ </td><td></td><td> $4 7 . 4 6 { \pm } 1 . 3 5 $ </td></tr><tr><td></td><td></td><td>Training budget, both rates lower</td><td></td></tr><tr><td>Readout</td><td>One epoch</td><td></td><td>Two epochs</td></tr><tr><td>Pointer</td><td> $5 0 . 2 8 { \pm } 1 . 0 2 $ </td><td></td><td> $4 3 . 9 6 { \pm } 0 . 9 1$ </td></tr><tr><td>Mean-span</td><td> $5 1 . 0 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td></td><td> $4 8 . 5 7 { \pm } 1 . 1 6$ </td></tr></table>

Table 4 | Training-recipe diagnostics on MC11 (%, mean ± descriptive SE). The upper panel contains one-epoch pointers; the lower panel compares fresh runs with budget-specific schedule horizons. Lower/higher rates are $1 . 2 5 { \times } 1 0 ^ { - 5 } / 2 . 5 { \times } 1 0 ^ { - 5 }$

Learning rates. Lowering the adapter rate improves MC11 at both head rates, with an average gain of 2.83 ± 0.79 pp (Table 4). Changing the head rate has little average effect (+0.02 ± 0.43 pp). Among these rate settings, the pointer performs best with the lower rate for both parameter groups.

Training budget. Both readouts perform better after one epoch in every setting, with the largest losses from longer training on ARC and HellaSwag (Figure 2). The two-epoch pointer falls to 43.96%, while pooling falls to 48.57%. Pooling’s relative advantage therefore grows from 0.73 to 4.61 pp as both models become less accurate. The one-epoch, lower-rate recipe supplies the strongest of these all-module pointer references at 50.28% for evaluating the two designs.

![](images/5fcefa4f884dacc42044b39f43f64617e90eba53027d93d5e2c5942246c0d190.jpg)  
Figure 2 | Accuracy loss from one to two epochs under the lower-rate all-module recipe. Both readouts favor one epoch on ARC and HellaSwag.

Update schedule. At the higher learning rate, training only the head for the first 30% of updates before enabling the adapters yields 53.54% MC11 with a no-FFN pointer (Appendix B.3). The core readout-by-scope comparison keeps joint training and the lower rates fixed, so each design change can be assessed under the same recipe. Combining the head-first schedule with mean-span pooling remains untested.

## 5.5. Task-Level Sources of Improvement

HellaSwag and synthetic chess account for most of the net MC11 gain (Figure 3). On HellaSwag, pooling adds about 4.2 pp under either adapter scope, while removing FFN adapters adds about 9.1 pp with either readout, giving the final model an advantage of about 13.3 pp. On chess, the pooling gain grows from 1.95 pp under all-module adaptation to 7.35 pp without FFN adapters.

The final model also improves both ARC configurations, both MMLU configurations, and WinoGrande, alongside declines on GSM8K and GPQA. Appendix B gives the component contrasts for every task. The mean pooling and scope gains remain positive under both weighting schemes when any one source is omitted (Appendix D).

![](images/f89a9b53fa2f9a9b2469be73ea1784724c60be4d2de26590c6267325d4c9cd09.jpg)

![](images/a0740901797301fbe9289a8508ad330560b7eeab837d7894d14392d0c90ad4cf.jpg)  
Figure 3 | Task-level effects of the two design choices. Left: mean-span pooling minus the pointer under each adapter scope. Right: removing FFN adapters for each readout. Bars are means over four paired settings under the one-epoch, lower-rate recipe. All configurations retain their original weight in MC11.

## 6. Additional Reranking Sources and Backbone Comparison

## 6.1. PIQA and CommonsenseQA

We test the individual design choices beyond MC11 on PIQA and CommonsenseQA. PIQA asks the model to select between two solutions to a physical goal, whereas CommonsenseQA offers five candidate answers. We apply the same three-option NLI scorer to their 1,838 and 1,221 questions, respectively, using separate matched model groups to compare learning rates, readouts, and adapter scope (Table 5).
<table><tr><td>Factor</td><td>Paired comparison</td><td>PIQA (pp)</td><td>CommonsenseQA (pp)</td></tr><tr><td>Learning rate</td><td>Lower-rate pointer – higher-rate pointer</td><td>+12.17±2.49</td><td>+11.65±1.57</td></tr><tr><td>Readout</td><td>Pooling – pointer, both lower-rate</td><td>+1.26±0.85</td><td>+4.01±1.83</td></tr><tr><td>Adapter scope</td><td>No-FFN pointer – all-module pointer</td><td>+5.21±1.15</td><td>+9.56±2.81</td></tr><tr><td colspan="2">Scope comparison accuracy: all-module pointer</td><td>75.04%</td><td>54.24%</td></tr><tr><td colspan="2">Scope comparison accuracy: no-FFN pointer</td><td>80.25%</td><td>63.80%</td></tr></table>

Table 5 | Additional-source support for individual design choices. Changes are paired foursetting means with descriptive SEs. The learning-rate comparison changes head and adapter rates together; the readout comparison uses the same model group. Adapter scope is tested separately with lower-rate, one-epoch pointers and matched initializations. These groups are separate from the core readout-by-scope models in Table 2; the final no-FFN pooling model has not been evaluated on these sources.

Lowering both learning rates strengthens the pointer on both sources, and pooling improves these lower-rate models by a further 1.26 pp on PIQA and 4.01 pp on CommonsenseQA. The separate scope comparison follows the same direction: omitting FFN adapters improves both sources in every setting, reaching 80.25% and 63.80%, respectively.

## 6.2. Readout on Qwen3-4B

The Qwen3-4B comparison tests the readout change under lower-rate, all-module adaptation. Pooling improves HellaSwag by 3.91 pp, close to the 4.23 pp gain on NeoHorse, and also improves the ARC configurations. Declines elsewhere, including a 4.28 pp loss on four-option GSM8K, outweigh these gains: overall MC11 accuracy is 51.70% for pooling and 52.17% for the pointer. The readout change thus retains gains on particular tasks without improving aggregate accuracy on this backbone. Appendix B reports the complete task comparison.

## 7. Trade-offs on Direct Selection

We also test the same core checkpoints under direct selection to see how the reranking choices affect their original use (Table 6). The no-FFN pointer raises MC11 while maintaining JevBench accuracy, at 77.52% versus 77.11% for the all-module pointer. Adding pooling improves MC11 further, with JevBench and Kev accuracies of 73.79% and 78.12%. These are 3.32 and 3.25 pp below the all-module pointer.

Across both adapter scopes, the pointer gives higher direct-selection accuracy than pooling on these suites. ${ \mathrm { K e v } } ^ { \prime } { \mathrm { s } }$ declines include decision-test as well as the transfer subsets; Appendix B gives the six-subset profile. Together with the fixed-checkpoint interface comparisons in Section 4, these results support choosing the readout with the intended scoring task in mind.
<table><tr><td>Core model</td><td>JevBench (%) Kev (%)</td></tr><tr><td>All-module pointer  $7 7 . 1 1 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $8 1 . 3 7 { \pm } 0 . 5 9 $ </td></tr><tr><td>All-module pooling</td><td> $7 5 . 3 7 { \scriptstyle \pm 0 . 9 6 }$   $7 9 . 9 1 { \scriptstyle \pm 0 . 3 5 }$ </td></tr><tr><td>No-FFN pointer</td><td> $7 7 . 5 2 { \pm } 0 . 9 8 $   $7 9 . 7 8 { \pm } 0 . 3 5 $ </td></tr><tr><td>No-FFN pooling</td><td> $7 3 . 7 9 { \scriptstyle \pm 1 . 1 8 }$   $7 8 . 1 2 { \scriptstyle \pm 0 . 2 6 }$ </td></tr></table>

Table 6 | Direct-decision results for the core readout-by-scope comparison. Entries are four-setting means with descriptive SEs. MC11 results for these same models appear in Table 2.

## 8. Related Work

Studies of specialized JEV decision models report uneven performance across knowledge, reasoning, and multilingual tasks [24]. Interventions on labels, definitions, and context also change their behavior without changing model weights [1, 47]. LLM-as-Jev examines joint suffix scoring without training and fine-tuning with KL anchors [25]. Related work on prompted language models shows that multiple-choice scores depend on the scoring interface and option identifiers [34, 53]. We focus on improving the trained JEV model when it is used to compare candidate answers through separate entailment requests.

Pointer and biaffine architectures score contextual representations with shared projections [44, 14]. Research on sentence and span representations examines which token states to aggregate [32, 42]; recent work extends these choices to language models used as text encoders [45, 2]. Tang and Yang [40] compare pooling and attention while holding the embedding model’s data and backbone fixed. Our comparisons examine option-span readouts trained jointly with adapters, including whether pooling remains useful under a smaller adapter scope.

LoRA restricts adaptation through low-rank updates to selected projections [18]. Module selection under distribution shift [22], forgetting during adaptation [5], and learning rates across adapter factors and ranks [15, 8] provide reasons to examine placement, capacity, and optimization together. Gradual unfreezing [17], head fitting before fine-tuning [20], and parameterefficient head fitting [50] offer other ways to control adaptation. Fine-tuning stability also depends on initialization, data order, and optimization [13, 29]. Our paired comparisons bring readout design and adaptation scope together in a typed decision model used for cross-request entailment reranking.

## 9. Conclusion

Fixed-checkpoint comparisons show that direct selection and entailment reranking can favor different option readouts, indicating that design choices for single-request selection need not transfer unchanged to cross-request reranking. Motivated by this observation, we pair meanspan pooling over each NLI option with selective backbone adaptation restricted to attention and DeltaNet projections, jointly adjusting how option states are extracted and how the backbone is updated. Matched comparisons under a tuned one-epoch recipe establish the gains from both choices, while pointer-position, parameter-budget, and head-capacity controls clarify their respective contributions.

On NeoHorse-1-4B, selective adaptation improves MC11 from 50.28% to 52.49%, and adding mean-span pooling reaches 53.39%, with both steps positive in all four fold–seed settings. Together, these changes improve reranking accuracy by about 3.1 pp while reducing trainable parameters by 55.2%.

## References

[1] Seyedarmin Azizi, Erfan Baghaei Potraghloo, and Massoud Pedram. 2026. Labels override definitions in Jev-style typed decision models. Preprint, arXiv:2610.02586.

[2] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large language models are secretly powerful text encoders. In First Conference on Language Modeling.

[3] Benchmark Heaven. 2026. JevBench: Benchmark for AI decision models. Benchmark repository. Accessed 11 October 2026.

[4] Bespoke Labs and Maheswaran Sathiamoorthy. 2026. Nimble. Software and dataset repository. Revision f136b3f75721fda4ea961f73993cc50b08488835; accessed 2026-10-11.

[5] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John P. Cunningham. 2024. LoRA learns less and forgets less. Transactions on Machine Learning Research.

[6] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439.

[7] Iñigo Casanueva, Tadas Temˇcinas, Daniela Gerz, Matthew Henderson, and Ivan Vuli´c. 2020. Efficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processingfor Conversational AI, pages 38–45, Online. Association for Computational Linguistics.

[8] Nan Chen, Soledad Villar, and Soufiane Hayou. 2026. Learning rate scaling across LoRA ranks and transfer to full finetuning. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 17039–17084. PMLR.

[9] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

[10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. Preprint, arXiv:1803.05457.

[11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

[12] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems, volume 36, pages 28091–28114. Curran Associates, Inc.

[13] Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah A. Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. Preprint, arXiv:2002.06305.

[14] Timothy Dozat and Christopher D. Manning. 2017. Deep biaffine attention for neural dependency parsing. In International Conference on Learning Representations.

[15] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024. LoRA+: Efficient low rank adaptation of large models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17783–17806. PMLR.

[16] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

[17] Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 328–339. Association for Computational Linguistics.

[18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

[19] Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. 2020. The multilingual Amazon reviews corpus. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4563–4568, Online. Association for Computational Linguistics.

[20] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. 2022. Fine-tuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations.

[21] Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An evaluation dataset for intent classification and out-of-scope prediction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1311–1316. Association for Computational Linguistics.

[22] Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. 2023. Surgical fine-tuning improves adaptation to distribution shifts. In International Conference on Learning Representations.

[23] Xin Li and Dan Roth. 2002. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, pages 556–562.

[24] Xing Li, Qingcheng Chang, Jinzhong Ning, Changfeng Xu, Shenlong Zhang, Yijia Zhang, Ling Luo, and Hongfei Lin. 2026. Specialized decision models vs. general-purpose LLMs: Benchmarking Jev across knowledge, reasoning, and multilingual tasks. Preprint, arXiv:2610.11978.

[25] Yinheng Li and Justin Wagle. 2026. LLM-as-Jev: LLMs are already Jev-style decision models—when and how to fine-tune them. Preprint, arXiv:2610.02076. Version 3.

[26] Guoming Ling, Muen Xue, and Zijian Ye. 2026. Jev in the wild: A data-driven analysis of the Jev model’s functionality, applications and ecosystem. Preprint, arXiv:2609.30216.

[27] Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

[28] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

[29] Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. 2021. On the stability of finetuning BERT: Misconceptions, explanations, and strong baselines. In International Conference on Learning Representations.

[30] NeoHorse Team, Guoliang Cao, Guohao Dai, Tianyu Guo, Kai Han, Hailin Hu, Zihan Jiang, Xiang Kuang, Boxun Li, Yulong Li, Zehua Pei, Yuchuan Tian, Jiamin Wang, Yu Wang, Yunhe Wang, Yihong Wu, Haiyang Xu, Shuo Zhang, Hang Zhou, and 18 others. 2026. NeoHorse-1: Towards recursive self-improvement via agentic post-training with routing harness. Preprint, arXiv:2609.08183.

[31] Jared Palmer. 2026. Kev: A family of decision models. Software repository. Accessed 11 October 2026.

[32] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992. Association for Computational Linguistics.

[33] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level Google-proof Q&A benchmark. In First Conference on Language Modeling.

[34] Joshua Robinson, Christopher Michael Rytting, and David Wingate. 2023. Leveraging large language models for multiple choice question answering. In The Eleventh International Conference on Learning Representations.

[35] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial Winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740.

[36] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense reasoning about social interactions. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, Hong Kong, China. Association for Computational Linguistics.

[37] Leslie N. Smith. 2018. A disciplined approach to neural network hyper-parameters: Part 1—learning rate, batch size, momentum, and weight decay. Preprint, arXiv:1803.09820.

[38] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

[39] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158. Association for Computational Linguistics.

[40] Yixuan Tang and Yi Yang. 2024. Pooling and attention: What are effective designs for LLM-based embedding models? Preprint, arXiv:2409.02727.

[41] TokenRhythm. 2026. NeoHorse-Jev: Prefill-only decisions for agent workflows. Software and model documentation. Accessed 11 October 2026.

[42] Shubham Toshniwal, Haoyue Shi, Bowen Shi, Lingyu Gao, Karen Livescu, and Kevin Gimpel. 2020. A cross-task analysis of text span representations. In Proceedings of the 5th Workshop on Representation Learningfor NLP, pages 166–176. Association for Computational Linguistics.

[43] TypeSafe. 2026. Jev: Introduction. Model and API documentation. Accessed 13 October 2026.

[44] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. In Advances in Neural Information Processing Systems, volume 28.

[45] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving text embeddings with large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11897–11916, Bangkok, Thailand. Association for Computational Linguistics.

[46] Adina Williams, Nikita Nangia, and Samuel R. Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

[47] Zixiang Xu, Zirui Song, Chiyu Zhang, Xiuying Chen, Xi Liu, Xiyang Hu, and Yue Zhao. 2026. JevOut: Natural context can flip decision models. Preprint, arXiv:2609.30243. Version 2.

[48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

[49] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025b. Gated delta networks: Improving Mamba2 with delta rule. In International Conference on Learning Representations.

[50] Zhuoyi Yang, Ming Ding, Yanhui Guo, Qingsong Lv, and Jie Tang. 2022. Parameter-efficient tuning makes a good classification head. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7576–7586. Association for Computational Linguistics.

[51] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Associa tion for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

[52] Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

[53] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. Large language models are not robust multiple choice selectors. In International Conference on Learning Representations, volume 2024, pages 19426–19454.

## A. Protocol and Reproducibility

## A.1. MC11 populations and scoring

MC11 averages the eleven configuration accuracies listed in Table 7, each computed by ranking candidates with Equation 1. These configurations contain 46,475 question occurrences per model: GSM8K, GPQA, and MMLU repeat their question populations under two formats, whereas ARC-Easy and ARC-Challenge use different populations from the same source. An occurrence identifies a question within a configuration, so a repeated question contributes once in each configuration. Appendix D examines how the results change when sources receive equal weight or are omitted individually.
<table><tr><td>Configuration</td><td>Questions</td><td>Split</td><td>External candidates and demonstrations</td></tr><tr><td>ARC-Easy</td><td>2,376</td><td>Test</td><td>Choices supplied with each source question.</td></tr><tr><td>ARC-Challenge</td><td>1,172</td><td>Test</td><td>Choices supplied with each source question.</td></tr><tr><td>WinoGrande</td><td>1,267</td><td>Validation</td><td>Supplied alternatives, inserted into the incomplete sentence.</td></tr><tr><td>GSM8K, four options</td><td>1,319</td><td>Test</td><td>Correct numerical answer and three generated dis- tractors.</td></tr><tr><td>GSM8K, ten options</td><td>1,319</td><td>Test</td><td>Correct numerical answer and nine generated dis- tractors.</td></tr><tr><td>GPQA</td><td>198</td><td>Diamond</td><td>Supplied answers, with the correct answer first in loaded order.</td></tr><tr><td>GPQA, few-shot</td><td>198</td><td>Diamond</td><td>Same answers; five solved examples from other Diamond questions.</td></tr><tr><td>Chess</td><td>500</td><td>Generated</td><td>One legal move and three illegal moves.</td></tr><tr><td>HellaSwag</td><td>10,042</td><td>Validation</td><td>Supplied candidate endings.</td></tr><tr><td>MMLU</td><td>14,042</td><td>Test</td><td>Supplied answer choices.</td></tr><tr><td>MMLU, few-shot</td><td>14,042</td><td>Test</td><td>Same choices; five examples from the subject&#x27;s de- velopment split.</td></tr></table>

Table 7 | MC11 populations, candidate construction, and demonstration sources.

The question supplies the premise, and the default candidate hypothesis is “The correct answer is: [option]”. GSM8K uses “The answer is [option]”, WinoGrande substitutes the candidate into the sentence, HellaSwag uses the candidate ending, and few-shot configurations use “Answer: [option]”. The instruction asks how this hypothesis relates to the premise, with three option descriptions: “The hypothesis follows from the premise”, “The hypothesis may or may not be true given the premise”, and “The hypothesis contradicts the premise”.

Evaluation selects the first maximal candidate in the recorded order and requires three finite probabilities in [0, 1] whose sum differs from one by at most 10<sup>−5</sup>; an invalid output stops the evaluation. For GSM8K, the distractors perturb the correct numerical answer, and reproducing the evaluation requires the saved candidate order. Chess questions are generated with chess version 1.11.2 and the recorded randomization settings. Source inventories identify the dataset revisions, splits, and row order for the evaluated populations.

## A.2. Training corpus and grouped folds

The predominantly English corpus contains 22,035 records and 29,897 questions drawn from web decisions, policy tasks, and public classification datasets. It combines 4,642 JevForge/Mind2Web records [12], 2,676 Nimble records [4], 1,680 compositional-policy records, ten public-source groups of 1,000 records each, and generated policy, date, insufficient-evidence, control, and assertion records. The ten public sources are Yelp full-star reviews, DBpedia14, and AG News [52]; Banking77 [7]; TREC [23]; MultiNLI [46]; IMDb [28]; sentence-level SST-5 [38]; BoolQ [9]; and the SetFit English packaging of MARC [19], with all source labels converted to the shared decision schema.

JevForge uses the cleaned Mind2Web resource at revision 080a4b67 and the JevForge conversion at revision 8c6fa6c9. The conversion preserves labels and distributions, including probability mass shared by several positive candidates. Its website partition differs from the official Mind2Web out-of-distribution split. The retained Nimble reference is revision f136b3f75721fda4ea961f73993cc50b08488835. Its publisher describes the reference answers as synthetic and model-checked. Training consumes record contents directly without opening external private-file or image references.

To keep related examples in the same split, we link records through shared families, paired or control examples, upstream rows, websites, and normalized states, then assign each connected group as a whole. This produces 12,472 groups, allocated with approximate feature balance: 80% to eight training pools, 5% to locked development, 5% to calibration, and 10% to a separate test split. The eight pools contain 17,661 records and 24,001 questions, with fold � holding out pool � and training on the remaining seven. Our four settings combine folds 2 and 3 with seeds 44 and 45: fold 2 contains 15,456 training records and requires 1,932 updates per epoch, while fold 3 contains 15,442 records and requires 1,931 updates.

The corpus contains 12,486 Choice, 12,176 binary, and 5,235 Score questions. Most supplied target distributions are one-hot (9,142 of 9,457), while the remaining distributions express insufficient evidence through 173 uniform targets or multiple valid solutions through 142 JevForge targets. Training preserves these distributions and augments compatible requests by shuffling options or adding distractor and none-of-the-above options, with augmentation probabilities ranging from 0.10 to 0.25.

## A.3. Serialization, backbone, and optimization

The request format follows NeoHorse-Jev’s option-conditioned implementation of typed JEV decisions [43, 41]. A record contains a state and one or more questions, each with an instruction, a type, and option descriptions or ordered levels. The encoder places the state after <|fim\_prefix|>, the instruction after <|fim\_middle|>, each option between <|box\_start|> and <|box\_end|>, and a final <|fim\_suffix|>. Text segments are tokenized separately without added special tokens; delimiter IDs are inserted explicitly, with no intervening spaces or newlines. The instruction precedes all option tokens. Option isolation is disabled, allowing an option to attend to earlier options in its request.

Training packs up to 4,096 tokens, with a 2,048-token limit on each state or branch segment; the inference server permits 8,192 tokens and rejects longer requests. Target distributions index Choice options, false/true answers, or Score levels, assigning zero mass to missing keys before normalization. When only an integer label is supplied, it defines a one-hot target. Training minimizes $- \sum _ { i } t _ { i } \log p _ { i }$ averaged over questions within a record, with the ordered-score auxiliary loss disabled.

We use TokenRhythm/NeoHorse-1-4B at revision 56f0584b [30], with a width-256 readout and zero head dropout. Rank-64 LoRA [18] uses � = 128 and dropout 0.05 across attention projections q\_proj, k\_proj, v\_proj, and o\_proj; gated DeltaNet projections in\_proj\_qkv, in\_proj\_z, in\_proj\_a, in\_proj\_b, and out\_proj; and, for all-module adaptation, feedforward projections gate\_proj, up\_proj, and down\_proj. The no-FFN configuration omits these last three adapter targets while keeping the pretrained feed-forward modules.

Parameters are stored in FP32 and computation uses BF16 autocast, gradient checkpointing, and gradient clipping at 1.0. AdamW [27] uses weight decay 0.01, microbatch size one, and accumulation over eight records. The OneCycle schedule [37] warms up for 10% of updates, begins at one twenty-fifth of the peak learning rate, and cosine-anneals to one ten-thousandth of that initial rate. Its first-moment coefficient cycles inversely between 0.95 and 0.85. Evaluation uses the final weights of each run. The two-epoch runs restart from saved initialization with a two-epoch schedule horizon: 3,864 updates on fold 2 and 3,862 on fold 3.

## A.4. Configurations, matching, and reuse

Each configuration in Tables 2 and 4 is trained independently from its corresponding saved initialization and uses its assigned readout throughout training and evaluation.

The pointer and mean-span heads have identical parameter counts: 1,311,232 parameters. All-module LoRA adds 129,859,584 parameters; attention and DeltaNet adapters alone add 57,507,840. The auxiliary all-module rank-28 control uses � = 56 and totals 58,124,800 trainable parameters, keeping $\alpha / r _ { \mathrm { L o R A } } = 2$ . Its adapter shapes differ from rank 64.

Within each fold–seed setting, the core learning-rate models start from common saved parameters. The adapter-scope and rank comparisons copy reference parameters wherever their dimensions agree and align the initial CPU/CUDA random-generator states. Separate generators control record shuffling and augmentation, with matching recorded orders, inputs, and generator traces across the comparisons. A repeated pointer executes the reference recipe again under the same recorded starting conditions.

Training the four learning-rate combinations and the repeated-pointer, trained-residual, and frozen-branch controls across four settings gives 28 models. The adapter-scope and rank comparisons add 20 models while reusing the four reference pointers; eight two-epoch models then reuse those pointers and the four all-module pooling models as their one-epoch references, giving 56 distinct trainings. The joint learning-rate experiments on H100, the additional adapterscope comparison, and the Qwen3 experiments each use their own references.

For each named comparison, we first subtract scores within a fold–seed setting and then average the four differences. The descriptive standard error is the sample standard deviation of those differences divided by two. All arithmetic uses unrounded values; displayed means may therefore differ by 0.01 percentage points from the displayed paired difference. Factorial main effects average over both levels of the other factor, and interactions combine the corresponding four arms.

## A.5. Other endpoints

JevBench [3] equally weights 18 families over 231 questions: 48 easy, 72 original, and 111 hard decisions. Invalid probability outputs count as incorrect, and ties follow the sorted option-key order. Kev [31] averages question-level accuracy equally over the development and test partitions of decision-v7, transfer-v4, and transfer-v9. After excluding the source named unknowable, these partitions contain 1,264/1,200, 656/656, and 1,046/1,046 questions, respectively; the testonly analysis averages the three test partitions. The fixed-checkpoint interface comparisons use all 1,838 PIQA validation questions and all 1,954 SocialIQA development questions, both outside MC11.

## A.6. Evidence coverage and reproduction

Model development used the primary evaluation suites and fold-development data. Because the four reported settings share six training pools and the same evaluation questions, their standard errors describe variation within this design. The original GPQA few-shot configuration draws demonstrations from the Diamond evaluation population; Appendix B.4 reports a separate evaluation with disjoint demonstrations.

Overlap screening found no exact or normalized state matches between the training corpus and 231 JevBench records or 6,436 Kev record occurrences, using Unicode NFKC normalization, case folding, and whitespace normalization. Available source identities likewise showed no overlap. Further screening covered 30,503 reranking source rows, including 285 MMLU demonstration rows, and the 198-row GPQA file. This coverage is limited to the recorded source text and identities; semantic paraphrases, substrings, pretraining exposure, synthetic chess, and rendered evaluation requests remain unassessed. The intermediate SetFit Amazon conversion has not been recovered.

The saved predictions support independent rescoring of 72 checkpoints: the 56 core Neo-Horse models, eight Qwen3 models, and eight higher-rate pointer and pooling checkpoints used in the fixed-interface comparison. Candidate probabilities are stored with answer identities, candidate order, population information, and eligibility rules, allowing predictions and aggregate scores to be recomputed while preserving repeated questions and duplicate answer texts. Some additional learning-rate and adapter-scope results are available only as per-model summaries.

Reproduction requires the base weights, training records, evaluation caches, framework forks, and compatible runtimes specified by the code and configurations. The retained inputs supported reconstruction of the corpus mixture and fold exports, followed by one separate pointer/pooling pair from training through scoring on H20. The OpenCompass and RunWeaver forks are unavailable anonymously, and redistribution permissions for RunWeaver and other assets remain unresolved. The initial source-generation and tokenizer-based eligibility steps remain unreplayed, and the tested environment retains a declared Torch dependency conflict.

CUDA determinism was disabled, allowing repeated training runs to diverge despite matched initial parameters and data streams. Dropout trajectories were not saved for the experiments that varied the head and adapter rates together. Appendix C examines execution variation through repeated-training controls. The 56 core NeoHorse models were trained on individual GPUs of a shared eight-H20 server, using 94.46 GPU hours over the measured training intervals with allocated-memory peaks of 18.32–19.81 GiB. These measurements include concurrent-job contention and diagnostic capture; evaluation and the additional experiments fall outside the timing intervals.

## B. Supplementary Comparisons

## B.1. Additional training comparisons

Additional experiments test simultaneous changes to the head and adapter learning rates, training schedules, and adapter placement. The learning-rate and schedule comparisons use H100-trained models; the separate adapter-scope comparison uses eight H20-trained pointers. Their lower-rate all-module references reach 50.33% and 49.34% MC11, respectively, compared with 50.28% for the core reference. Paired differences use each experiment’s own matched reference models under the same four fold–seed settings.

## B.2. Task effects, adapter count, and backbone

Table 8 shows where pooling and scope restriction help. On NeoHorse, pooling improves HellaSwag under both adapter scopes. Removing FFN adapters from the pooling model adds 9.10 pp on HellaSwag and 15.20 pp on chess, but lowers both GSM8K configurations and both GPQA configurations. The corresponding pointer scope changes are also largest on HellaSwag and chess, at 9.13 and 9.80 pp.
<table><tr><td rowspan="2"></td><td colspan="4">NeoHorse</td><td rowspan="2">Qwen3-4B Pool-ptr.</td></tr><tr><td>Pool-ptr. all</td><td>Pool-ptr. no-FFN</td><td>No-FFN-all pooling</td><td>Pool-last-content all</td></tr><tr><td>Configuration</td><td></td><td></td><td></td><td></td><td>all</td></tr><tr><td>MMLU, few-shot</td><td>-0.34</td><td>-0.38</td><td>+1.32</td><td>+0.94</td><td>+0.16</td></tr><tr><td>MMLU</td><td>+0.49</td><td>+0.46</td><td>+1.42 +9.10</td><td>+1.75 +8.10</td><td>+0.01</td></tr><tr><td>HellaSwag</td><td>+4.23</td><td>+4.21 -0.66</td><td>-0.30</td><td>-0.55</td><td>+3.91</td></tr><tr><td>GSM8K, ten options</td><td>-1.48</td><td>+0.99</td><td>+1.73</td><td>+4.08</td><td>-1.84 +0.03</td></tr><tr><td>ARC-Easy GSM8K, four options</td><td>+1.60</td><td>-1.82</td><td>-2.12</td><td>+0.13</td><td>-4.28</td></tr><tr><td>WinoGrande</td><td>-1.40 +0.57</td><td>+1.46</td><td>+1.07</td><td>+0.39</td><td>-0.36</td></tr><tr><td>ARC-Challenge</td><td>+3.14</td><td>+1.79</td><td>+1.92</td><td>+5.38</td><td>+0.26</td></tr><tr><td>Chess</td><td>+1.95</td><td>+7.35</td><td>+15.20</td><td>+0.10</td><td>-1.15</td></tr><tr><td>GPQA, few-shot</td><td>+0.88</td><td>-0.88</td><td>-2.15</td><td>+1.01</td><td>-1.39</td></tr><tr><td>GPQA</td><td>-1.64</td><td>-2.53</td><td>-0.88</td><td>-0.51</td><td>-0.51</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8 | Mean within-setting task differences in percentage points. Each column uses four paired settings; all models use one epoch and the lower rates. The second NeoHorse column compares readouts with no-FFN adapters; the third compares adapter scopes with pooling fixed. The Qwen3-4B column uses a separate pointer–pooling comparison. Numbers are computed before rounding.

<table><tr><td>Kev subset</td><td></td><td></td><td>Pool-ptr., all Pool-ptr., no-FFN Ptr.: no-FFN-all Pool: no-FFN-all</td><td></td></tr><tr><td>Decision development</td><td> $+ 0 . 2 0 \pm 0 . 2 4$ </td><td> $+ 0 . 2 4 \pm 0 . 2 5$ </td><td> $- 0 . 3 4 \pm 0 . 1 7$ </td><td> $- 0 . 3 0 \pm 0 . 3 3$ </td></tr><tr><td>Decision test</td><td> $- 0 . 0 6 \pm 0 . 2 4$ </td><td> $- 0 . 9 0 \pm 0 . 4 9$ </td><td> $- 0 . 9 2 \pm 0 . 0 3$ </td><td> $- 1 . 7 5 \pm 0 . 4 3$ </td></tr><tr><td>Transfer-v4 development</td><td> $- 2 . 4 4 \pm 0 . 6 9$ </td><td> $- 2 . 1 3 \pm 0 . 6 5$ </td><td> $- 2 . 7 1 \pm 1 . 1 0$ </td><td> $- 2 . 4 0 \pm 0 . 2 7$ </td></tr><tr><td>Transfer-v4 test</td><td> $- 1 . 5 6 \pm 0 . 2 0$ </td><td> $- 1 . 6 8 \pm 0 . 6 2$ </td><td> $- 1 . 8 7 \pm 0 . 7 0$ </td><td> $- 1 . 9 8 \pm 0 . 4 7$ </td></tr><tr><td>Transfer-v9 development</td><td> $- 3 . 2 3 \pm 0 . 5 3$ </td><td> $- 3 . 3 0 \pm 0 . 1 3$ </td><td> $- 2 . 2 5 \pm 0 . 5 1$ </td><td> $- 2 . 3 2 \pm 0 . 3 7$ </td></tr><tr><td>Transfer-v9 test</td><td> $- 1 . 7 0 \pm 0 . 0 6$ </td><td> $- 2 . 2 0 \pm 0 . 2 9$ </td><td> $- 1 . 4 8 \pm 0 . 3 3$ </td><td> $- 1 . 9 8 \pm 0 . 3 9$ </td></tr></table>

Table 9 | The six Kev selections in the core readout-by-scope comparison. Entries are paired mean changes in pp with descriptive SEs. The direct-decision cost includes changes on decision-test as well as the transfer suites.

The Qwen3-4B comparison uses four lower-rate, one-epoch pairs with rank-64 attention/FFN adapters and identical shared initial tensors. Qwen3 has no DeltaNet targets. Pooling changes MC11 from 52.17% to 51.70%, a paired difference of $- 0 . 4 7 \pm 0 . 5 9$ pp, with one positive setting. HellaSwag retains a gain, but the ARC differences are small and GSM8K declines. JevBench changes by $+ 0 . 1 7 \pm 1 . 1 4$ pp and Kev by $- 1 . 4 4 \pm 0 . 4 2 \mathrm { p p }$ . These models received no additionalsource evaluation.

## B.3. Learning-rate scaling and staged adaptation

The learning-rate comparison lowers the head and adapter rates together while retaining saved initial tensors and the one-epoch schedule. Pointer MC11 increases from 45.90% to 50.33%; the residual-head and mean-span pooling models increase by 1.60 and 1.75 pp. Consequently, their advantages over the same-rate pointer shrink from 3.41 to 0.57 pp and from 3.95 to 1.26 pp. Here both rates change together; the adapter-rate main effect in Table 4 averages over two head rates.

Head-first adaptation freezes the backbone for the first ⌈0.3�⌉ updates and then enables adapters without restarting the full-horizon schedule. The backbone runs in evaluation mode during the initial phase. Head-only-tail training instead places the same number of head-only updates at the end. Both schedules omit 580 adapter updates per setting but expose the adapters to different examples, learning rates, momentum histories, and dropout phases.
<table><tr><td>Pointer recipe or paired change</td><td>MC11</td><td>JevBench</td><td>Kev</td></tr><tr><td>Joint training, higher rate (accuracy, %)</td><td>45.90</td><td>75.62</td><td>81.47</td></tr><tr><td>Lower-rate joint – higher-rate joint (pp)</td><td> $+ 4 . 4 3 \pm 1 . 0 3$ </td><td> $+ 2 . 3 5 \pm 0 . 6 7$ </td><td> $- 0 . 4 3 \pm 0 . 5 3$ </td></tr><tr><td>Head-first – higher-rate joint (pp)</td><td> $+ 6 . 3 8 \pm 2 . 1 7$ </td><td> $+ 2 . 0 3 \pm 0 . 8 4$ </td><td> $- 1 . 0 2 \pm 0 . 2 8$ </td></tr><tr><td>No-FFN – higher-rate joint (pp)</td><td> $+ 3 . 6 7 \pm 1 . 3 3 $ </td><td> $+ 1 . 8 0 \pm 1 . 5 3 $ </td><td> $- 0 . 0 2 \pm 0 . 3 4$ </td></tr><tr><td>Head-first + no-FFN – higher-rate joint (pp)</td><td> $+ 7 . 6 4 \pm 1 . 9 1 $ </td><td> $+ 0 . 9 6 \pm 0 . 6 6$ </td><td> $- 1 . 9 5 \pm 0 . 4 0$ </td></tr><tr><td>Head-only tail – higher-rate joint (pp)</td><td> $+ 3 . 8 4 \pm 0 . 8 8$ </td><td> $- 0 . 3 1 \pm 1 . 4 0$ </td><td> $- 0 . 3 0 \pm 0 . 2 3$ </td></tr></table>

Table 10 | Effects of update timing and adapter placement in a separately trained set of pointers. All recipes use the option-ending pointer. Except for the explicit lower-rate row, both peak rates are $2 . 5 \dot { \times } 1 0 ^ { - 5 }$ . Differences are means ± descriptive SE across four settings. Head initialization can vary when adapter scope changes in these comparisons.

Head-first reaches 52.28% MC11 and exceeds head-only tail by $2 . 5 4 \pm 1 . 4 9$ pp. Combining head-first and no-FFN reaches 53.54%, but its scope-by-schedule interaction is $- 2 . 4 1 \pm 1 . 6 1$ pp: the combined gain is smaller than the sum of the separate gains. The combined head-first/no-FFN model uses the higher-rate pointer recipe; adding pooling to this recipe remains untested. The separate adapter-scope comparison starts from identical head and attention/DeltaNet adapter tensors.

A further seven-pair retraining checks a different rate interval, $5 \times 1 0 ^ { - 5 }$ to $2 . 5 \times 1 0 ^ { - 5 }$ for both parameter groups. The mean MC11 change is −0.34 pp: five pairs improve, while two decline by 2.46 and 6.10 pp. JevBench and Kev means rise by 1.69 and 0.74 pp. Those pairs share their seeded construction procedure, but saved initial-tensor identity is not established. The rate result therefore depends on the interval and comparison being tested.

## B.4. Additional sources and inference execution

Fixed-question requests. The PIQA state is “Goal: [goal]”, followed by a newline and “Which solution best achieves this goal?”. Direct selection presents both solutions under the instruction “Select the solution that best achieves the goal.”, whereas reranking evaluates each solution in a separate NLI request with the hypothesis “The correct solution is: [solution]”. For the same 1,838 questions, these interfaces require one and two requests per question, respectively.

SocialIQA version 1.4 combines the original context, two newlines, and the question as its state. Direct selection presents the raw $\mathrm { A / B / C }$ answer texts under “Select the correct answer to the question.”; NLI reranking places each raw answer in quotation marks within “Hypothesis: [raw answer] How does it relate to the premise?”. Answer fragments remain in their original form, and the three NLI options use the descriptions in Appendix A.1. Comparing entailment probabilities across answers requires three requests per question, against one for direct selection. All 1,954 questions are evaluated with eight one-epoch checkpoints from the core all-module comparison and eight two-epoch checkpoints from the budget comparison.

The SocialIQA source, checkpoint selection, candidate order, request templates, and overlap screening were specified before scoring, after the main evaluations had begun. Exact matching, normalized lexical containment, and available source-lineage checks found no overlap with the training inputs and derivation files, so all questions were retained. These checks address recorded text and provenance; semantic overlap and pretraining exposure remain unmeasured.

NLI descriptions and option order. To examine sensitivity to the wording of the NLI options, we use the default descriptions in Appendix A.1 and two alternatives. The second set is “The premise supports the hypothesis as true.”, “The premise does not determine whether the hypothesis is true.”, and “The premise supports the hypothesis as false.” The third is “The hypothesis must be true if the premise is true.”, “The hypothesis could be either true or false if the premise is true.”, and “The hypothesis must be false if the premise is true.” For each fixed checkpoint, we cross these descriptions with all six label orders, giving 18 templates while keeping external solution order fixed and retrieving entailment by label identity.

Additional-source protocol. PIQA and CommonsenseQA evaluate 1,838 validation and 1,221 development questions through NLI reranking. CLINC150 [21] evaluates all 4,500 in-scope test utterances through a single 150-option direct request. These evaluations cover 36 models varying learning rates, training schedules, or adapter placement, plus eight separately trained pointers comparing adapter scopes with matched initialization. The final no-FFN pooling configuration is evaluated on MC11, JevBench, and Kev.

CommonsenseQA uses the question stem without its question-concept field; each of its five candidate answers becomes “The correct answer is: [candidate]”. The 24 questions with duplicate answer texts retain distinct candidate identities. Reranking candidate order follows the SHA-256 of canonical JSON triples containing the source revision, sample identity, and candidate identity, with candidate identity as a secondary key. Gold labels enter neither requests nor ordering. Exact ties use the first maximum. Uniform-tie credit leaves the reported pooling and scope effects unchanged. Registered training and development-selection materials contain no exact or normalized PIQA or CommonsenseQA question matches under Unicode NFKC, case, and whitespace normalization; semantic overlap and pretraining exposure remain unmeasured.

On CLINC150, pooling lowers accuracy by 3.64 ± 0.54 pp in the lower-rate readout comparison, and omitting FFN adapters lowers it by 1.78 ± 0.41 pp in the separate scope comparison. These direct-selection losses accompany the PIQA and CommonsenseQA gains in Table 5. The known CLINC overlap item is retained and correctly classified by every model, contributing zero to each paired difference.

The fixed-checkpoint PIQA and additional-source comparisons evaluate the same eight higher-rate models in two execution environments. The original inference service on H100 gives pointer/pooling accuracies of 65.27/73.12%, while the packaged-model service on H20 gives 65.15/73.22%. These close aggregate scores conceal changes in about 2.6% of the 14,704 model– question selections, with nearly balanced correct-to-wrong and wrong-to-correct changes. We therefore keep each comparison within one environment: the interface tests use the packagedmodel service, and the additional-source tests use the original service. The two executions match weights, tokenizer, model and loading code, requests, labels, and candidate order; both use a BF16 backbone and FP32 head in evaluation mode. Hardware and service environment vary together.

Wording changes the readout comparison even with the checkpoints fixed. Across the 18 templates, the mean pooling effect is $+ 3 . 1 8 \pm 3 . 2 7 \mathrm { p p }$ , but averaging label orders within each description gives +6.31, −1.80, and +5.03 pp, and the smallest fixed-template mean is −5.51 pp. The sign changes show why the default-template gain should be interpreted alongside this sensitivity to how the NLI relations are described.

For GPQA, a separate five-shot evaluation excludes Diamond identities, normalized question matches, and revision aliases from the Main-split demonstration pool. On the same 198 Diamond questions and eight higher-rate checkpoints, pooling scores 36.87% against the pointer’s $3 3 . 4 6 \% , a + 3 . 4 1 \pm 2 . 7 5$ pp effect with two positive settings. This additional evaluation changes demonstration selection, candidate order, and identity handling together. Its scores are reported separately, with MC11 retaining the original GPQA configurations.

## C. Residual Corrections and Training Controls

The residual head augments the option-ending pointer with a learned decision–option correction. For projected query and key vectors �, $k _ { i } \in \mathbb { R } ^ { \bar { 2 } \bar { 5 } 6 }$ and the bilinear pointer score $s _ { i } ,$ it uses

$$
\begin{array} { r l } & { \quad \nu _ { i } = [ q ; k _ { i } ; q \odot k _ { i } ; | q - k _ { i } | ] , } \\ & { \quad s _ { i } ^ { \mathrm { p a i r } } = s _ { i } + w ^ { \top } \mathrm { G E L U } ( A \nu _ { i } + a ) + b . } \end{array}\tag{8}
$$

The correction adds a 128-unit hidden layer and 131,329 parameters, with output weights � and bias � initialized to zero. To examine its role during training, we compare optimization of all four branch tensors with a frozen branch whose output remains zero while the shared projections and adapters train. To examine its contribution after training, we remove the correction from a trained residual model and apply the softmax to $s _ { i }$

## C.1. Removing the correction from a trained model

For each of the four higher-rate residual models, we compare corrected and pre-correction logits from the same forward pass, keeping the trained backbone and query/key projections fixed. The head computes both sets of logits in FP32 without autocast, so the comparison isolates the correction’s contribution at inference within an already trained model.

Removing the correction changes only four of 185,900 selected answers, with no correct answer becoming incorrect, and increases mean MC11 by less than 0.001 pp. This mean uses the same configuration weighting as the main evaluation. The learned bilinear component therefore recovers virtually all answer selections at these checkpoints.

## C.2. Frozen branches, repeated training, and divergence

The training comparisons show a stronger dependence on the recipe. At the higher rate, the frozen-branch model exceeds the pointer by $1 . 3 9 \pm 0 . 3 8$ pp on MC11, and the trained residual model exceeds the frozen-branch model by a further $2 . 0 2 \pm 0 . 9 7 \mathrm { p p }$ . Under the lower-rate recipe, the core comparisons are small and depend on the reference run (Table 11), while the separate comparison with jointly reduced head and adapter rates retains a $+ 0 . 5 7 \pm 0 . 1 8 ~ \mathrm { p p }$ residual advantage.

<table><tr><td>Lower-rate training comparison</td><td>MC11</td><td>JevBench</td><td>Kev</td></tr><tr><td>Repeated pointer – reference</td><td> $+ 0 . 1 7 \pm 0 . 7 4$ </td><td> $- 0 . 6 6 \pm 0 . 5 2$ </td><td> $- 0 . 1 5 \pm 0 . 3 7$ </td></tr><tr><td>Frozen branch – reference</td><td> $+ 0 . 4 0 \pm 0 . 9 3$ </td><td> $- 0 . 1 3 \pm 1 . 6 0$ </td><td> $- 0 . 0 6 \pm 0 . 5 2$ </td></tr><tr><td>Trained residual – reference</td><td> $+ 0 . 0 5 \pm 0 . 5 7$ </td><td> $- 0 . 8 2 \pm 1 . 6 1$ </td><td> $- 0 . 1 1 \pm 0 . 4 1$ </td></tr><tr><td>Trained residual – frozen branch</td><td> $- 0 . 3 5 \pm 0 . 5 9$ </td><td> $- 0 . 6 9 \pm 0 . 8 7$ </td><td> $- 0 . 0 4 \pm 0 . 2 2$ </td></tr><tr><td>Trained residual – repeated pointer</td><td> $- 0 . 1 2 \pm 0 . 5 8$ </td><td> $- 0 . 1 6 \pm 1 . 8 2$ </td><td> $+ 0 . 0 4 \pm 0 . 2 0$ </td></tr></table>

Table 11 | Lower-rate residual controls, in percentage points (mean ± descriptive SE over four settings). All arms use one epoch and all-module rank-64 adapters. The repeated pointer reruns the reference recipe with the same nominal seed.

Repeating the pointer from the same saved initial parameters, nominal seed, and recorded data streams changes its MC11 score enough to reverse the small residual-versus-pointer contrast (Table 11). Short matched probes help explain why identical starting conditions are insufficient: even pairs of ordinary pointers develop different gradients and parameters when CUDA determinism is disabled. The operations responsible and their contribution to the full-training differences remain unresolved.

During lower-rate residual training, corrected and pre-correction BF16 logits select the same option on all recorded training questions. BF16 addition can round away a nonzero correction. These traces concern option choices during lower-rate training, whereas the FP32 intervention measures reranking selections at higher-rate checkpoints.

## D. Aggregation and Reference Sensitivity

MC11 gives equal weight to eleven configurations. The seven-source alternative first averages configurations within ARC, GSM8K, GPQA, and MMLU, then gives these four means and HellaSwag, WinoGrande, and chess equal weight. Configuration-weighted leave-one-source-out (LOSO) removes all configurations of one source and averages the remainder; source-weighted LOSO averages the remaining six source means. Every omission retains all four model settings. The ranges in Table 12 summarize how the mean effects change with benchmark composition.
<table><tr><td></td><td></td><td> ${ \mathrm { S e v e n } }$ </td><td>Config.-weighted LOSO</td><td>Source-weighted</td></tr><tr><td>Contrast</td><td>MC11</td><td>sources</td><td></td><td>LOSO</td></tr><tr><td>Head-rate main effect</td><td> $+ 0 . 0 2 \pm 0 . 4 3$ </td><td> $- 0 . 1 1 \pm 0 . 4 6$ </td><td>[-0.132, +0.285]</td><td>[-0.345, +0.314]</td></tr><tr><td>Adapter-rate main effect</td><td> $- 2 . 8 3 \pm 0 . 7 9$ </td><td> $- 2 . 9 6 \pm 0 . 6 5$ </td><td>[-3.393,-2.240]</td><td>[-3.408,-2.478]</td></tr><tr><td>Pooling – pointer, all</td><td> $+ 0 . 7 3 \pm 1 . 0 7$ </td><td> $+ 1 . 0 5 \pm 1 . 1 1$   $+ 1 . 6 4 \pm 0 . 5 7$ </td><td>[+0.363, +1.209]</td><td>[+0.524, +1.470]</td></tr><tr><td>Pooling – pointer, no-FFN No-FFN – all, pointer</td><td> $+ 0 . 9 1 \pm 0 . 3 5$   $+ 2 . 2 1 \pm 0 . 6 6$ </td><td> $+ 3 . 1 0 \pm 0 . 6 4$ </td><td>[+0.264, +1.489] [+1.451,+3.015]</td><td>[+0.693, +2.202]</td></tr><tr><td>No-FFN – all, pooling</td><td> $+ 2 . 3 9 \pm 0 . 3 5$ </td><td> $+ 3 . 6 9 \pm 0 . 4 0$ </td><td>[+1.110, +3.259]</td><td>[+1.985, +3.854]</td></tr><tr><td>Readout-by-scope interaction</td><td> $+ 0 . 1 8 \pm 0 . 7 4$ </td><td> $+ 0 . 5 9 \pm 0 . 6 6$ </td><td>[-0.341,+0.515]</td><td>[+1.773, +4.558] [-0.212, +0.909]</td></tr><tr><td>No-FFN rank 64 – all rank 28</td><td> $+ 0 . 0 9 \pm 0 . 4 2$ </td><td> $+ 0 . 5 1 \pm 0 . 5 9$ </td><td>[-0.200, +0.514]</td><td>[+0.102, +0.900]</td></tr><tr><td>Mean-span – last-content</td><td> $+ 1 . 8 9 \pm 0 . 9 7$ </td><td> $+ 2 . 1 0 \pm 0 . 7 7$ </td><td>[+1.265, +2.362]</td><td>[+1.103, +2.488]</td></tr><tr><td>Trained residual – reference</td><td> $+ 0 . 0 5 \pm 0 . 5 7$ </td><td> $- 0 . 0 9 \pm 0 . 6 4$ </td><td>[-0.343, +0.338]</td><td>[-0.407, +0.366]</td></tr><tr><td>Budget interaction</td><td> $+ 3 . 8 8 \pm 1 . 1 3 $ </td><td> $+ 3 . 6 6 \pm 1 . 3 7$ </td><td>[+2.530, +4.553]</td><td>[+2.608, +4.428]</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 12 | Sensitivity of learning-rate, readout, adapter-placement, and training-budget effects to benchmark aggregation (pp). The first two columns are paired means ± descriptive SE. Each LOSO range spans seven omission-specific means. Main effects use higher minus lower rate; the budget interaction is the two-epoch pooling advantage minus the one-epoch advantage.

The head-by-adapter rate interaction is +1.28 ± 1.65 pp on MC11. The pooling and no-FFN main comparisons retain positive four-setting means under both weighting rules and every source omission. The interaction between them is less stable: omitting chess reverses its MC11- weighted mean. Likewise, no-FFN rank 64 versus all-module rank 28 can change sign under configuration-weighted omission. The near-zero head-rate and trained-residual means change sign under equal source weighting. These outcomes support the individual recipe comparisons more strongly than claims about their interactions or the ranking of near-equal arms.

The budget result also persists under reweighting: longer training lowers both models’ accuracy, with the pointer losing more. ARC and HellaSwag account for about 63% of the wider gap. Under seven-source weighting, the mean interaction remains +3.66 pp, although one setting has a small negative interaction of −0.20 pp.

Replacing the reference pointer by its repeated training changes the all-module pooling effect to $+ 0 . 5 5 \pm 1 . 1 1$ pp and the trained-residual effect to −0.12 ± 0.58 pp. No-FFN and rank-28 pointers retain gains of $+ 2 . 0 4 \pm 0 . 6 3$ and $+ 1 . 9 5 \pm 0 . 7 7$ pp, each positive in all four settings. These larger recipe gains are less dependent on which reference execution is used.

Pooling and scope restriction also lower Kev under the three-test-subset average, with declines on the decision test as well as transfer subsets. The smaller residual-control contrasts change sign under this weighting.