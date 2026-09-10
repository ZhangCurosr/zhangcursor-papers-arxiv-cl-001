# Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling

Tingshuo Fan Hongtao Mu Tianyu Zhou Hansen Liu Tao Ji<sup>#</sup> College of Foreign Languages and Literature, Fudan University tsfan24@m.fudan.edu.cn taoji@fudan.edu.cn

## Abstract

When training data are limited, increasing parameter count is not the only way to improve language-model performance. A small parameter set, when repeatedly applied, can also deliver comparable performance. We study Looped GPT-BERT in the BabyLM 2026 Strictsmall setting, combining GPT-BERT’s masked next-token and causal language-modeling objectives with depth-wise parameter sharing. We train on a preprocessed 7.48M-word English corpus and compare objective ratios, nonlooped and looped architectures, and loop counts. Our final 4 × 12 model uses four physical layers for twelve recurrent traversals and contains 12.18M parameters. The BabyLM 2026 leaderboard reports an Overall Average of 35.42 and an NLP Average of 48.48. Compared with public BabyLM 10M Strict-small GPT-2 and GPT-BERT baselines, it achieves comparable performance on selected linguistic and downstream metrics, including BLiMP and GLUE, with fewer parameters. The loop ablations show that additional recurrent computation can improve training and preserve strong performance on selected linguistic tasks, whereas poorer performance on other tasks may reveal an inherent limitation of the looped design: using only a few physical layers restricts the model’s representational space.

## 1 Introduction

Language-model performance has commonly improved through jointly scaling model size, data, and training computation. Scaling-law studies describe predictable power-law relations between loss, parameter count, data, and compute (Kaplan et al., 2020), while compute-optimal analyses emphasize that parameter and token budgets should be allocated jointly (Hoffmann et al., 2022). These results largely concern much larger autoregressive models, where model capacity and per-example computation grow together. Under constrained data and parameter budgets, however, they need not: repeatedly applying the same parameters can create a deeper computational path without proportionally increasing the number of learned weights. We therefore ask whether recurrent computation induced by shared parameters can compensate for reduced physical depth and parameter capacity in low-resource language-model pretraining.

The BabyLM Challenge provides a controlled setting for this question by restricting pretraining text to a developmentally plausible scale and evaluating linguistic, downstream, and human-like behavior (Warstadt et al., 2023). We work in the BabyLM 2026 Strict-small track on a cleaned 7.48M-word English corpus. Our starting point is GPT-BERT (Charpentier and Samuel, 2024), which trains masked next-token prediction (MNTP) and causal language modeling (CLM) in one Transformer parameter stack. It consequently supports both bidirectional masked prediction and autoregressive scoring or generation. This hybrid objective gives a further dimension to study: when the backbone is looped, does the relative amount of masked and causal supervision change the capabilities the architecture acquires?

We introduce depth-wise recurrent parameter sharing on top of GPT-BERT. Universal Transformers repeatedly share transformations along depth (Dehghani et al., 2019), and ALBERT shows that cross-layer sharing can substantially reduce encoder parameters (Lan et al., 2020). Most directly, Looped Transformers inject a fixed input and the prior loop state into a shared backbone, increasing effective computational depth through iteration (Yang et al., 2024). Their strongest evidence, however, comes from in-context algorithm-learning tasks such as linear functions, decision trees, and neural-network regression. Natural-language pretraining must simultaneously acquire lexical knowledge, syntactic regularities, world knowledge, and discourse state; whether those computations can be supported by repeated shared layers remains an open empirical question.

We address three questions: how the BERT:GPT training ratio changes the training objective and downstream abilities; whether effective depth formed by repeatedly applying a small number of physical layers can retain language ability with substantially fewer parameters; and whether training and downstream tasks continue to benefit as the number of loops increases. We summarize our contributions as follows:

• We introduce Looped GPT-BERT, combining GPT-BERT’s hybrid MNTP/CLM objective with depth-wise recurrent parameter sharing.

• We study how the BERT:GPT objective ratio affects training and downstream behavior in the looped architecture.

• We compare a 12-layer non-looped model with four physical layers applied 1/3/6/12 times, showing how recurrent computation closes the gap from fewer learned layers and where additional loops yield diminishing returns.

• We evaluate the final model in the BabyLM Challenge, where its 12.18M parameters achieve comparable performance to public Strict-small baselines on linguistic and downstream tasks.

## 2 Related Work

Data-Efficient LM Pretraining BabyLM shifts attention from performance under web-scale training to learning efficiency under a controlled data budget. The first challenge showed that architecture, pretraining objective, preprocessing, and curriculum can all substantially affect small-model performance, with no single approach dominating every downstream task (Warstadt et al., 2023; Hu et al., 2024). Subsequent work further found that the effect of data composition depends on model scale: a genre mixture suitable for one model size need not be optimal for another (Yam and Paek, 2024). Other BabyLM studies explore variation sets, child-inspired data and vocabulary choices, explicit linguistic information, and self-distillation as complementary routes to sample-efficient pretraining (Haga et al., 2024; Ghanizadeh and Dousti, 2024; Edman et al., 2024; Nair et al., 2024). These approaches primarily improve sample efficiency through data selection, ordering, or objectives. Complementarily, we study parameter efficiency: keeping the training corpus and model width fixed, we replace part of the independently parameterized depth with repeated computation through shared physical layers.

Hybrid Causal and Masked Modeling CLM predicts the next token from left context, whereas conventional MLM predicts selected tokens at their own positions from bidirectional context. The two objectives therefore assign different meanings to the same output position in a decoder-style model. GPT-BERT resolves this mismatch with MNTP (Charpentier and Samuel, 2024): when token $x _ { k + 1 }$ is masked, supervision is shifted left so that the hidden state at position k predicts the original $x _ { k + 1 }$ Consequently, causal and masked examples share a single next-token vocabulary projection and loss interface. GPT-mode rows retain their original tokens and use a lower-triangular attention mask; MNTPmode rows corrupt selected input tokens, permit bidirectional attention, and compute loss only at the corresponding shifted next-token positions. Each training row is assigned one mode, rather than receiving both losses. AntLM similarly combines causal and masked language-modeling objectives in a BabyLM setting, alternating between them during training (Yu et al., 2024).

Recurrent Depth and Parameter Sharing Universal Transformers update representations recurrently along depth while retaining parallel computation over sequence positions (Dehghani et al., 2019); ALBERT uses cross-layer parameter sharing primarily for parameter efficiency and shows that the sharing strategy affects downstream behavior (Lan et al., 2020). Looped Transformers use the recurrence $z _ { t + 1 } = f ( x + z _ { t } )$ , where x is a fixed input embedding, $z _ { t }$ is the loop state, and $f$ is a shared Transformer backbone (Yang et al., 2024). This makes effective computational depth grow with loop count while the parameter count is governed mainly by the number of physical layers. Kohli et al. (2026) provide a more direct naturallanguage reference: recurrent-depth Transformers can improve systematic generalization and depth extrapolation in implicit multi-hop reasoning, but excessive loops can also reduce prediction quality through overthinking. Neither line of work establishes that shared loops are uniformly suitable for lexical, syntactic, world-knowledge, and discourse-state learning. We therefore compare equal-application and equal-parameter configurations across linguistic and state-tracking tasks.

<table><tr><td>Cleaning operation</td><td>Before</td><td>After</td></tr><tr><td>Style and grammar normalization</td><td>Where are you at?</td><td>Where are you?</td></tr><tr><td>Disfluency removal</td><td>a pound, or a hundred pounds today, is not the A hundred pounds today is not worth the same same</td><td>amount later.</td></tr><tr><td>Short-content filtering</td><td>Right.</td><td>Removed when it lacks independent semantic content.</td></tr><tr><td>Case normalization</td><td>LET&#x27;S GIVE IT UP FOR DANNY TANNER</td><td>Let&#x27;s give it up for Danny Tanner.</td></tr><tr><td>Structural-noise removal</td><td>=== Bullet the Blue Sky ===</td><td>Header removed; article text retained.</td></tr><tr><td>Stage-direction removal</td><td>I&#x27;ll be right back. [leaves room.]</td><td>I&#x27;ll be right back</td></tr><tr><td>Speaker normalization</td><td>*MOT: more what? *CHI: more tapioca.</td><td>A: More what? B: More tapioca</td></tr><tr><td>Consecutive-turn merging</td><td>B: Yeah. B: I&#x27;m in Texas</td><td>B: Yeah, I&#x27;m in Texas.</td></tr><tr><td>Duplicate removal</td><td>Repeated or near-identical sentences</td><td>One normalized instance.</td></tr></table>

Table 1: Representative corpus-cleaning operations. Examples are shortened for presentation.

## 3 Method

## 3.1 Corpus Preparation and Tokenization

We construct our training corpus by selecting and cleaning six source corpora: BNC Spoken, CHILDES, Project Gutenberg, OpenSubtitles, Simple English Wikipedia, and Switchboard. The resulting new\_data1 corpus is produced by our rule-based normalization and filtering pipeline, which removes redundancy, short or low-quality content, case and structural noise, and normalizes dialogue formatting. The preprocessing code and source files are available in our preprocessing repository. Table 1 summarizes the operations and representative input–output examples. After cleaning, the six selected corpora yield 7,482,189 whitespace-delimited words. We use this complete post-cleaning corpus directly for pretraining rather than adding or padding data to reach the 10M-word upper bound. Their source-wise counts are reported in Table 2.

We use two byte-pair encoding (BPE) tokenizers. The official 16k GPT-BERT tokenizer is used only for early non-looped ratio experiments, while the 8k tokenizer trained on the new\_data1 corpus is used for the main and loop experiments. BPE provides a subword representation that can handle words beyond a fixed vocabulary (Sennrich et al., 2016). We choose the 8k vocabulary as a parameterallocation trade-off for this small-data setting: with hidden size 384, its tied embedding table contains 8,192 × 384 ≈ 3.15M parameters. In our corpus, reducing the vocabulary from 16k to 8k increases the measured fertility from 1.4381 to 1.4787 tokens per whitespace-delimited word, a 2.82% increase. This indicates a modest change in word-level context coverage for the fixed 128-token input, while avoiding a larger embedding allocation.

<table><tr><td>Source</td><td>Words</td></tr><tr><td>BNC Spoken</td><td>619,847</td></tr><tr><td>CHILDES</td><td>1,779,027</td></tr><tr><td>Gutenberg OpenSubtitles</td><td>2,502,709 1,174,391</td></tr><tr><td>Simple Wikipedia</td><td>1,386,065</td></tr><tr><td>Switchboard</td><td>20,150</td></tr><tr><td>Total</td><td>7,482,189</td></tr></table>

Table 2: Word counts after preprocessing.

## 3.2 Hybrid GPT-BERT Objective

Let $\boldsymbol { x } ~ = ~ ( x _ { 1 } , \dots , x _ { T } )$ be a token sequence and θ the shared model parameters. For GPT-mode rows, a causal attention mask is used and every non-padding next token is supervised:

$$
\mathcal { L } _ { \mathrm { G P T } } = - \sum _ { t = 1 } ^ { T - 1 } \log p _ { \theta } ( x _ { t + 1 } \mid x _ { \le t } ) .\tag{1}
$$

For BERT-mode rows, predictable positions are sampled with a masking probability that decreases linearly from 0.3 to 0.15. Of the selected tokens, 80% are replaced by the mask token, 10% by a random token, and 10% are left unchanged. Bidirectional attention is permitted. The target is shifted to the preceding hidden-state position so that it remains aligned with the decoder-style next-token head. If x˜ denotes the corrupted input, the objective is

$$
\mathcal { L } _ { \mathrm { B E R T } } = - \sum _ { t \in M } \log p _ { \theta } ( x _ { t } \mid \tilde { x } _ { \backslash t } ) .\tag{2}
$$

Only shifted positions corresponding to selected masks contribute to this loss; all other labels are set to the ignore index. Thus, unmasked BERT positions never enter the loss. A batch-level ratio of r<sub>B</sub> : r<sub>G</sub> assigns complete rows to the two modes, and the optimized objective averages valid supervised tokens with a z-loss regularizer weighted by

10<sup>−4</sup>. Our final setting uses 1:3, so one quarter of rows use MNTP and three quarters use CLM.

## 3.3 Looped Transformer Backbone

The non-looped baseline contains twelve independent Transformer layers. The looped backbone instead contains four independently parameterized physical layers, each with attention and feed-forward sublayers. Each loop adds the static token representation x to the preceding loop state z<sub>t</sub> and applies the same four-layer stack $F _ { \theta } { } _ { \mathrm { : } }$ :

$$
\begin{array} { r } { z _ { 0 } = 0 , \qquad } \\ { z _ { t + 1 } = F _ { \theta } ( x + z _ { t } ) , \quad t = 0 , \ldots , L - 1 . } \end{array}\tag{3}
$$

Thus, $z _ { t + 1 }$ is the final representation produced by the current traversal and is simply carried into the next pass; it is not an additional network operation. Within physical layer i, attention and feed-forward transformations are applied sequentially:

$$
\begin{array} { r } { u _ { t } ^ { ( i ) } = \mathrm { D W A } _ { i , A } \left( h _ { t } ^ { ( i ) } + \mathrm { A t t n } _ { i } ( h _ { t } ^ { ( i ) } ) \right) , } \\ { h _ { t } ^ { ( i + 1 ) } = \mathrm { D W A } _ { i , F } \left( u _ { t } ^ { ( i ) } + \mathrm { F F N } _ { i } ( u _ { t } ^ { ( i ) } ) \right) . } \end{array}\tag{4}
$$

GPT-BERT’s dynamic weighted accumulation (DWA) is a learnable short-range residual mixer, not an additional recurrent state. We restrict DWA to representation fusion within each physical layer and reinitialize it on every loop; information across loops is carried exclusively by z<sub>t</sub>. A 4 × L model therefore has four sets of Transformer parameters but performs 4L layer applications. The 4 × 3, 4 × 6, and 4 × 12 variants perform 12, 24, and 48 layer applications, respectively, while keeping the same parameter count. This “effective depth” counts layer applications only: repeatedly applying shared parameters is not equivalent to a non-shared Transformer with the same number of layers.

## 3.4 Training Configuration

Table 3 lists the main configuration. Beyond the structural hyperparameters shown there, optimization uses LAMB (You et al., 2020), a peak learning rate of 0.0141, a minimum learning rate of 0.00141, and cosine decay for 2,600 steps followed by a constant minimum rate. We select the peak learning rate and decay horizon through the controlled studies in Section 5. All pretraining and fine-tuning runs use random seed 42.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Physical layers</td><td>4</td></tr><tr><td>Loop iterations</td><td>12</td></tr><tr><td>Hidden size</td><td>384</td></tr><tr><td>FFN size Attention heads</td><td>1,280 6</td></tr><tr><td>Vocabulary</td><td>8,192</td></tr><tr><td>Maximum sequence length</td><td>128</td></tr><tr><td>Batch size per GPU Number of GPUs</td><td>32</td></tr><tr><td>Global batch (tokens)</td><td>8</td></tr><tr><td>Epochs</td><td>32,768</td></tr><tr><td>Optimizer</td><td>10</td></tr><tr><td>Peak learning rate</td><td>LAMB</td></tr><tr><td></td><td>0.0141</td></tr><tr><td>Decay steps</td><td>2,600</td></tr><tr><td>Minimum learning rate</td><td>0.00141</td></tr><tr><td>Gradient clipping</td><td>2.0</td></tr><tr><td>Parameters</td><td>12.18M</td></tr></table>

Table 3: Main pretraining configuration.

## 4 Experiments

## 4.1 Evaluation Protocol

We follow the BabyLM 2026 Strict-small evaluation pipeline. The final checkpoint receives the full evaluation: the zero-shot suite includes BLiMP and BLiMP Supplement minimal-pair syntax judgments (Warstadt et al., 2020), EWoK world knowledge (Ivanova et al., 2024), Entity Tracking, COMPS conceptual properties, GlobalPIQA commonsense reasoning, Reading, and Age of Acquisition (AoA). Fine-tuning covers BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, and WSC, the GLUE/SuperGLUE-style naturallanguage-understanding subset used by the challenge (Wang et al., 2019). We report the leaderboard aggregates, where NLP Average pools NLP tasks and Human-like Average includes Reading, AoA, and related measures. We use the official causal backend for all reported GPT-BERT results.

Beyond the final main revision, we retain intermediate checkpoints at one-million-word intervals from chck\_1M through chck\_10M, and at tenmillion-word intervals from chck\_20M through chck\_70M. We run the official fast evaluation on each of these checkpoints. The ten-epoch training run contains approximately 74.8M word presentations, so chck\_70M is the last complete ten-millionword milestone. The final model and all intermediate revisions are released in one public Hugging Face repository.

## 4.2 BabyLM 2026 Leaderboard Results

Table 4 reports the final 4 × 12 Looped GPT-BERT alongside two publicly released BabyLM Strictsmall references: the official GPT-2 baseline and the GPT-BERT causal-focus baseline. The leaderboard snapshot was accessed on July 21, 2026.<sup>1</sup>

<table><tr><td>Model</td><td>Params</td><td></td><td></td><td></td><td></td><td></td><td>Context Overall NLP BLiMP Sup. EWoK Entity COMPS GPIQA GLUE</td></tr><tr><td>Official GPT-2 baseline</td><td>124M</td><td></td><td></td><td></td><td></td><td>512 37.38 48.99 65.23 57.25 50.63 19.10</td><td>51.81 35.09 63.80</td></tr><tr><td>Official GPT-BERT causal-focus</td><td></td><td>31M 128→512 28.4635.6471.6663.21 49.49</td><td></td><td></td><td></td><td></td><td>65.13</td></tr><tr><td>Looped GPT-BERT (ours)</td><td>12.18M</td><td></td><td></td><td></td><td></td><td>128 35.42 48.48 71.19 55.06 49.10 15.78</td><td>51.01 34.67 62.55</td></tr></table>

Table 4: BabyLM 2026 Strict-small leaderboard comparison on aggregate and selected task-level metrics. “Context” denotes the maximum training sequence length, “Sup.” denotes BLiMP Supplement, “Entity” denotes Entity Tracking, “GPIQA” denotes GlobalPIQA, and “GLUE” denotes the official (Super)GLUE aggregate. Dashes indicate results not reported for the corresponding public entry. Aggregate scores for entries with unreported tasks are not directly comparable.

With 12.18M parameters, the final model is substantially smaller than both public references. It reaches 71.19 on BLiMP, compared with 65.23 for GPT-2 and 71.66 for GPT-BERT causal-focus, while its GLUE aggregate is 62.55, compared with 63.80 and 65.13. This pattern illustrates the tradeoff studied here: at the cost of additional computation, repeated application of shared layers can retain strong performance on selected tasks with a smaller independently parameterized model, with gains varying across tasks.

## 4.3 Effect of the BERT:GPT Ratio

We examine the BERT:GPT ratio under two tokenizers. The official 16k tokenizer is used for early non-looped ratio exploration, whereas the 8k tokenizer trained on the new\_data1 corpus is used for all main and loop experiments. The 15:1 configuration is BERT-heavy, 1:1 is balanced, and 1:3 is GPT-heavy. The two settings test whether the ratio trend is consistent; they are not used to compare tokenizer quality. Tables 5 and 6 report the principal 8k comparison, including training endpoints and available zero-shot results.

<table><tr><td>BERT:GPT</td><td>Params</td><td>Loss</td><td>Acc.</td></tr><tr><td>1:1</td><td>29.9M</td><td>3.9925</td><td>30.03</td></tr><tr><td>1:3</td><td>29.9M</td><td>3.3625</td><td>36.47</td></tr></table>

Table 5: Non-looped training results across objective ratios with the custom 8k tokenizer.

For context, Table 7 reports the corresponding early experiments with the official 16k tokenizer. Increasing the proportion of GPT rows lowers the final mixed training loss in both tokenizer settings, with 1:3 giving the strongest endpoint.

<table><tr><td>BERT:GPT</td><td>BLiMP</td><td>Sup.</td><td>COMPS</td><td>Entity</td></tr><tr><td>1:1</td><td>63.77</td><td>55.06</td><td>50.87</td><td>38.66</td></tr><tr><td>1:3</td><td>70.54</td><td>57.19</td><td>51.41</td><td>22.87</td></tr></table>

Table 6: Non-looped zero-shot results across objective ratios with the custom 8k tokenizer.

<table><tr><td>BERT:GPT</td><td>Final loss</td><td>Token accuracy</td></tr><tr><td>15:1</td><td>4.0039</td><td>33.42</td></tr><tr><td>1:1</td><td>3.9711</td><td>31.50</td></tr><tr><td>1:3</td><td>3.7925</td><td>32.37</td></tr></table>

Table 7: Non-looped training results across objective ratios with the official 16k tokenizer.

The 1:3 ratio improves BLiMP, BLiMP Supplement, and COMPS under the 8k tokenizer, but reduces Entity Tracking from 38.66 to 22.87. This pattern suggests that objective mixing changes the distribution of learned capabilities rather than merely the aggregate loss. More CLM supervision repeatedly trains left-to-right next-token prediction and is therefore closely aligned with sequential generation, local dependencies, and causal syntactic scoring. More MNTP supervision requires reconstructing a target from both sides of its context and may better support integrating multiple positions and relations, which is useful for tracking entities and their states. We therefore use 1:3 as the main configuration for subsequent loop experiments targeting generative and syntactic ability.

## 4.4 Effect of Looped Depth

We first examine training loss and token accuracy under the 1:3 objective. Figure 1 plots how these two measures change over training for the 12-layer non-looped model and for four-layer models with 1, 3, 6, or 12 applications per forward pass. The 4 × 3 and 12-layer non-looped models both execute twelve layer applications per forward pass, while the looped model uses only four independently parameterized layers.

<table><tr><td colspan="5">BERT:GPT = 1:3</td></tr><tr><td>Metric</td><td>Non-loop</td><td>4 × 1</td><td>4 × 3</td><td>4× 6</td><td>4 × 12</td></tr><tr><td>BLiMP</td><td>70.54</td><td>65.08</td><td>71.21</td><td>70.49</td><td>71.19</td></tr><tr><td>BLiMP Sup.</td><td>57.19</td><td>55.90</td><td>55.73</td><td>55.37</td><td>55.06</td></tr><tr><td>COMPS</td><td>51.41</td><td>51.21</td><td>51.40</td><td>51.69</td><td>51.01</td></tr><tr><td>Entity Tracking</td><td>22.87</td><td>17.38</td><td>16.47</td><td>28.66</td><td>15.78</td></tr><tr><td>Eye Tracking</td><td>8.61</td><td>8.62</td><td>7.85</td><td>8.62</td><td>8.25</td></tr><tr><td>Self-paced Reading</td><td>4.03</td><td>3.88</td><td>3.68</td><td>4.16</td><td>4.24</td></tr><tr><td colspan="6">BERT:GPT = 1:1</td></tr><tr><td>Metric</td><td>Non-loop</td><td>4 × 1</td><td>4× 3</td><td>4× 6</td><td>4 × 12</td></tr><tr><td>BLiMP</td><td>63.77</td><td>62.28</td><td>64.45</td><td>69.38</td><td>64.21</td></tr><tr><td>BLiMP Sup.</td><td>55.06</td><td>57.03</td><td>54.96</td><td>56.11</td><td>56.07</td></tr><tr><td>COMPS</td><td>50.87</td><td>50.44</td><td>50.95</td><td>51.42</td><td>50.03</td></tr><tr><td>Entity Tracking</td><td>38.66</td><td>17.52</td><td>16.60</td><td>13.10</td><td>17.15</td></tr><tr><td>Eye Tracking</td><td>7.86</td><td>7.64</td><td>7.88</td><td>7.62</td><td>7.55</td></tr><tr><td>Self-paced Reading</td><td>3.80</td><td>3.84</td><td>3.57</td><td>3.56</td><td>3.63</td></tr></table>

Table 8: Zero-shot comparison across objective ratios and loop counts.

The 4 × 1 control is visibly less stable at the beginning of training: it maintains a higher loss and lower token accuracy than the models with repeated layer applications. Increasing the loop count lowers the training loss and generally raises token accuracy, bringing the looped models closer to the 12-layer non-looped trajectory. The largest improvement occurs when moving from $4 \times 1$ to $4 \times 3 ;$ the changes from 4 × 3 to $4 \times 6$ and from 4 × 6 to 4 × 12 are smaller. Thus, repeated computation recovers much of the training behavior of the non-looped model, while additional loops provide diminishing returns.

We next turn to the zero-shot task results in Table 8. The table compares the 12-layer non-looped model with four physical layers applied 1, 3, 6, or 12 times, using BERT:GPT ratios of 1:3 in the upper block and 1:1 in the lower block. Under 1:3, increasing the loop count from 4 × 1 to 4 × 3 substantially improves BLiMP, from 65.08 to 71.21, slightly exceeding the non-looped score of 70.54. Increasing the depth beyond 4×3 does not produce a consistent additional gain: BLiMP is 70.49 for 4 × 6 and 71.19 for $4 \times 1 2$ . Under 1:1, 4 × 6 gives the highest BLiMP score (69.38), compared with 63.77 for the non-looped model and 62.28–64.45 for the other looped settings. Overall, the looped models preserve or improve several syntactic and reading-related scores despite using fewer independently parameterized layers, while the effect of recurrent depth varies across tasks.

Unlike the syntactic measures above, Entity Tracking is more sensitive to parameter sharing.

Overall, the looped variants are weaker than the corresponding non-looped model on this task, with one exception: under 1:3, 4 × 6 reaches 28.66, compared with 22.87 for the non-looped model. Even in this setting, the score falls to 15.78 with $4 \times 1 2$ , so the additional computation from six to twelve applications does not preserve the $4 \times 6$ improvement. Under 1:1, all looped variants remain below the non-looped score of 38.66, with scores between 13.10 and 17.52. One possible explanation is that entity tracking requires distinguishing and updating multiple entities, their states, and state changes (Kim and Schuster, 2023). Reusing four attention/FFN parameter sets rather than twelve independent sets may leave less representational space for the layer-specific transformations needed to maintain multiple entities and relations. This contrast suggests that recurrent depth can compensate for reduced parameterization on some linguistic tasks, but cannot uniformly replace the representational flexibility of independently parameterized layers. Considering both downstream performance and inference cost, $4 \times 6$ is a practical intermediate configuration, but the table also shows that the preferred loop depth depends on the objective ratio and task.

## 4.5 Inference Cost

We measure inference cost for the 12-layer nonlooped model and the $4 \times 3 , 4 \times 6 ,$ , and 4×12 looped models with the same batch size of 32, sequence length of 128, 20 warmup steps, and 100 timed forward passes on one RTX 3090. As shown in Table 9, $4 \times 3$ is better than the 12-layer non-looped model on every measured systems metric: it has lower latency, higher token throughput, and lower peak memory use. Increasing the loop count from 4×3 to 4×6 and 4×12 roughly doubles the latency at each step, while throughput decreases and memory use increases. Considering both downstream performance and inference speed, 4 × 6 provides a practical compromise: it is slower than $4 \times 3$ but performs better on some downstream tasks, while remaining substantially faster than $4 \times 1 2$

## 5 Optimization Ablations

## 5.1 Selecting the Peak Learning Rate

We first select a peak learning rate that lowers the training objective quickly without producing clear instability. Before full training, we run fourepoch non-looped sweeps at 0.0075, 0.01, 0.0141, and 0.02 with all other settings fixed. As Table 10 shows, the first three settings decrease stably, whereas 0.02 degrades after approximately step 600 and produces gradient-norm spikes. Although 0.01 has the highest final token accuracy, 0.0141 achieves the lowest final loss (3.6847) and is selected as the peak learning rate for the main experiments. This sweep provides a preliminary learningrate choice based on the non-looped model and does not fully optimize the rate for looped models. We examine the effect of decay speed on training stability separately below.

![](images/1caa7b5e63e1351dc293020ee5f0cfb13b12d2f482c42b8418ea87f4245eca34.jpg)  
Figure 1: Training loss and token accuracy for the 8k-tokenizer, 1:3 objective across loop depths. The looped models use four physical layers; the non-looped model uses twelve independently parameterized layers.

<table><tr><td>Model</td><td>Latency (ms) Tokens/s Memory (GiB)</td><td></td><td></td></tr><tr><td>12-layer non-loop</td><td>58</td><td>70,571</td><td>0.68</td></tr><tr><td>4 × 3</td><td>55</td><td>74,128</td><td>0.51</td></tr><tr><td>4× 6</td><td>108</td><td>37,966</td><td>0.72</td></tr><tr><td>4 × 12</td><td>212</td><td>19,295</td><td>1.15</td></tr></table>

Table 9: Inference speed measured with batch size 32, sequence length 128, 20 warmup steps, and 100 timed forward passes on one RTX 3090.

<table><tr><td>Peak LR</td><td>Final loss</td><td>Final accuracy</td></tr><tr><td>0.0075</td><td>3.7442</td><td>32.66</td></tr><tr><td>0.0100</td><td>3.6931</td><td>33.07</td></tr><tr><td>0.0141</td><td>3.6847</td><td>32.95</td></tr><tr><td>0.0200</td><td>4.5864</td><td>22.84</td></tr></table>

Table 10: Four-epoch peak learning-rate sweep.

## 5.2 Learning-Rate Decay and Stability

We select two candidate decay lengths from the training budget: 4,080 steps spans the full optimizer-step budget of the run, whereas 2,600 steps reaches the minimum earlier and remains there for the rest of training. With total training steps fixed, lr\_schedule\_steps determines how quickly cosine decay reaches its minimum: a smaller value decays earlier and remains at the minimum longer, while a larger value retains a higher learning rate later in training. We compare these decay rates in 4 × 6 and 4 × 12 models. The main 2,600-step schedule reaches 10% of the peak learning rate and then holds 0.00141; the slower 4,080- step schedule maintains a higher learning rate into the later stages. Because shared physical layers are repeatedly applied across loops, the effect of one parameter update can be propagated through multiple layer applications. A high early learning rate may therefore amplify update oscillations, while earlier decay may help shared parameters enter a useful region more smoothly.

Figure 2 shows that the 2,600-step schedule has its clearest advantage in early and middle training: loss declines faster and token accuracy rises sooner. Earlier decay, however, can reduce the opportunity to explore alternative parameter regions and leave the model near a suboptimal solution. The 4,080- step schedule gradually catches up later, consistent with its higher late-stage learning rate preserving additional exploration. Nevertheless, the 2,600- step curves are smoother overall and show no clear final-metric disadvantage. We therefore choose 2,600 steps as a practical compromise between stability and late-stage exploration under the current compute budget and single-seed setting, not as a schedule that is uniformly superior at every point in training.

![](images/06f689ba527330a0c96766cc4f8833741fecdad8a1e286e88f3ad12765d112a5.jpg)  
Figure 2: Training loss and token accuracy for the learning-rate decay ablation. Solid and dashed lines denote 2,600-step and 4,080-step decay, respectively; colors distinguish 4 × 6 and 4 × 12.

## 5.3 Qualitative Generation Examples

The causal interface also supports open-ended completion. Table 11 presents selected completions from the final 1:3, 4 × 12 model using temperature 0.9, top-k 40, top-p 0.95, and seed 42. The examples illustrate locally coherent continuation and appropriate termination in short responses; they are qualitative examples rather than a controlled generation evaluation.

<table><tr><td>Prompt</td><td>Sampled completion</td></tr><tr><td>I want to eat some</td><td>dinner, all right?</td></tr><tr><td>After the rain stopped, the children</td><td>began to run for a walk.</td></tr><tr><td>She put the toy in the</td><td>drawer, and gave the doll a glass of milk.</td></tr><tr><td>The bird flew over the</td><td>mountains and in the morning the rain came in.</td></tr></table>

Table 11: Selected sampled completions from the final 1:3, 4 × 12 model.

## 6 Conclusion

We combine GPT-BERT’s hybrid CLM/MNTP training with depth-wise recurrent parameter sharing to study a “fewer parameters, more computation” language-model design under BabyLM Strictsmall constraints. Reusing four physical layers for twelve loops yields a 12.18M-parameter model whose training objective is close to that of a 29.9M non-looped model. The final model reaches 71.19 on BLiMP and 62.55 on GLUE, compared with 71.66 and 65.13 for the official GPT-BERT causalfocus reference, despite its smaller parameter budget. On the reported task-level metrics, looped models preserve or improve selected syntactic abilities, while Entity Tracking becomes weaker as a result of parameter sharing. The ablations further show that gains from additional loops saturate quickly and differ across tasks. Looped GPT-BERT is therefore an architecture with clear advantages and limitations: under constrained data and parameter budgets, it can support syntactic understanding and generation, but its objective ratio and recurrent depth require further adjustment.

## Limitations

• Data and tokenizer. The main experiments use one cleaned corpus and one 8k tokenizer. We do not independently ablate raw data, rule-based cleaning, alternative cleaning procedures, and tokenization.

• Training variance. Each model is trained with a single random seed, so sub-percentage-point differences may fall within training variance.

• Compute matching. The 4×12 model performs substantially more layer applications and computation than the 12-layer non-looped baseline. This is a parameter-efficiency study, not a FLOPmatched comparison.

• Objective mixing. Changing the BERT:GPT ratio also changes supervision density and attention visibility; mixed training losses are therefore not perfectly homogeneous across ratios.

• Capability boundary. The final model is weak on AoA and Entity Tracking, indicating limited cognitive-similarity and state-tracking ability.

• Scope. Our conclusions are limited to approximately 12M parameters, a 7.48M-word corpus, ten epochs, and a maximum sequence length of

128; they should not be directly extrapolated to larger models or longer contexts.

## Ethics Statement

The model uses English text supplied by, or derived from, the BabyLM Challenge and introduces no newly collected personal data. The sources include books, subtitles, encyclopedic text, and dialogue, so they may retain social biases or inappropriate content from the original material; automated cleaning cannot guarantee their removal. Cleaning may also remove dialectal, conversational, or minoritylanguage patterns and introduce stylistic preferences through normalization. This small model is intended for research and should not be deployed directly in high-stakes applications.

## Reproducibility Statement

The Hugging Face repository contains the final model and sixteen intermediate checkpoints. The main revision stores the final model; chck\_1M–chck\_10M are one-million-word checkpoints, and chck\_20M–chck\_70M are ten-millionword checkpoints. It also includes configuration\_gpt\_bert.py, modeling\_gpt\_bert.py, and tokenizer files, allowing the model to be loaded with trust\_remote\_code=True.<sup>2</sup> Training configurations and scripts are available in the accompanying code repository,<sup>3</sup> and the data-cleaning workflow is documented separately.

## Acknowledgments

The authors thank the reviewers for their helpful comments and suggestions, as well as the BabyLM organizers for maintaining the datasets, evaluation pipeline, and leaderboard. This work was partially funded by the National Natural Science Foundation of China (No. 62506079).

## References

Lucas Georges Gabriel Charpentier and David Samuel. 2024. GPT or BERT: Why not both? In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 262– 283, Miami, FL, USA. Association for Computational Linguistics.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. 2019. Universal transformers. In International Conference on Learning Representations.

Lukas Edman, Lisa Bylinina, Faeze Ghorbanpour, and Alexander Fraser. 2024. Are BabyLMs second language learners? In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 166–173, Miami, FL, USA. Association for Computational Linguistics.

Mohammad Amin Ghanizadeh and Mohammad Javad Dousti. 2024. Towards data-efficient language models: A child-inspired approach to language learning. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 22–27, Miami, FL, USA. Association for Computational Linguistics.

Akari Haga, Akiyo Fukatsu, Miyu Oba, Arianna Bisazza, and Yohei Oseki. 2024. BabyLM challenge: Exploring the effect of variation sets on language model training efficiency. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 252–261, Miami, FL, USA. Association for Computational Linguistics.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. In Advances in Neural Information Processing Systems, volume 35, pages 30016–30030.

Michael Y. Hu, Aaron Mueller, Candace Ross, Adina Williams, Tal Linzen, Chengxu Zhuang, Leshem Choshen, Ryan Cotterell, Alex Warstadt, and Ethan Gotlieb Wilcox. 2024. Findings of the second BabyLM challenge: Sample-efficient pretraining on developmentally plausible corpora. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 1–21, Miami, FL, USA. Association for Computational Linguistics.

Anna A. Ivanova, Aalok Sathe, Benjamin Lipkin, Unnathi Kumar, Setayesh Radkani, Thomas H. Clark, Carina Kauf, Jennifer Hu, R. T. Pramod, Gabriel Grand, et al. 2024. Elements of world knowledge (EWoK): A cognition-inspired framework for evaluating basic world knowledge in language models. arXiv preprint arXiv:2405.09605.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Najoung Kim and Sebastian Schuster. 2023. Entity tracking in language models. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages

3835–3855, Toronto, Canada. Association for Computational Linguistics.

Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, and Yuekun Yao. 2026. Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. arXiv preprint arXiv:2604.07822.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations.

Aakarsh Nair, Alina Hancharova, Mayank Kumar, and Ali Gharaee. 2024. BabyLM challenge: Experimenting with self-distillation and reverse-distillation for language model pre-training on constrained datasets. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 28–36, Miami, FL, USA. Association for Computational Linguistics.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725. Association for Computational Linguistics.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations.

Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjape, Adina Williams, Tal Linzen, and Ryan Cotterell. 2023. Findings of the BabyLM challenge: Sample-efficient pretraining on developmentally plausible corpora. In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 1–34, Singapore. Association for Computational Linguistics.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions of the Association for Computational Linguistics, 8:377– 392.

Hong Meng Yam and Nathan Paek. 2024. What should baby models read? exploring sample-efficient data composition on model performance. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 284– 291, Miami, FL, USA. Association for Computational Linguistics.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. 2024. Looped transformers are better at learning learning algorithms. In International Conference on Learning Representations.

Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. 2020. Large batch optimization for deep learning: Training BERT in 76 minutes. In International Conference on Learning Representations.

Xinru Yu, Bin Guo, Shiwei Luo, Jie Wang, Tao Ji, and Yuanbin Wu. 2024. AntLM: Bridging causal and masked language models. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 324–331, Miami, FL, USA. Association for Computational Linguistics.