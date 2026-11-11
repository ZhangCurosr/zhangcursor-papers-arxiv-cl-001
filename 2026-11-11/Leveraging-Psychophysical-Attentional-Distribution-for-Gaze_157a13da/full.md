# Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling

Chaeho Lee<sup>1</sup>, Junyup Kim<sup>1</sup>, Wansoo Kim<sup>2</sup>, Young Min Jung<sup>1</sup>, Sang Ho Lee<sup>1,2,3</sup> <sup>†</sup>

<sup>1</sup>School of Digital Humanities and Computational Social Sciences, KAIST

<sup>2</sup>Department of Brain & Cognitive Sciences, KAIST

<sup>3</sup>Graduate School of Data Science, KAIST

{chaeholee, ytrewq271828, wsk37, youngmin, sangholee}@kaist.ac.kr

## Abstract

Integrating human feedback into language models has gained increasing attention as a means to align model outputs with human preferences, with Reinforcement Learning from Human Feedback (RLHF) serving as a prominent framework for preference-based alignment. Recent studies have incorporated eye-tracking (ET) data as an additional supervisory signal for reward model training, grounding reward learning in human reading behavior. Building on this approach, we extend prior fixation-centric models by leveraging established findings from cognitive psychology on human letter recognition. Specifically, we model visual attention as a spatially graded distribution centered on the current gaze location and incorporate this representation into reward model training. Our results show that the reward model trained with the proposed gaze distribution achieves higher preference prediction accuracy than a baseline model. Moreover, some results from our experiments are qualitatively consistent with established findings on human attention during reading, suggesting that they potentially capture cognitively meaningful aspects of human attention allocation.

## 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) is a widely used paradigm for aligning large language models (LLMs) with human preferences. In a typical RLHF pipeline, a reward model (RM) is trained to predict human preferences over pairs of responses from a generative LLM by assigning a higher reward to a chosen response than to a rejected one. Since the RM serves as a proxy for human judgment, its ability to faithfully capture human preferences is central to alignment quality.

A limitation of standard RLHF, however, is that such supervision is typically restricted to the final preference judgment. The model observes which response a human prefers, but not how the human cognitively processes the responses when deciding their preferred response. This limitation points to an opportunity to enrich reward modeling with human cognitive signals that arise during text processing. Among such signals, gaze signals derived from eye-tracking data are particularly relevant because gaze behavior reflects how readers allocate visual attention and cognitive effort across a text (Just & Carpenter, 1980). These data can be represented as gaze features (e.g., gaze location, fixation duration) mapped to individual words or tokens, and such gaze features can be used as additional inputs for reward modeling.

Recent studies suggest that gaze features can improve reward modeling for language model alignment. For example, López-Cardona et al. (2025) found that GazeReward, a gaze-augmented reward modeling framework that incorporates gaze features as additional inputs, increased the accuracy of human preference prediction. Galliamov et al. (2025) further investigated gaze-informed reward modeling and showed that incorporating gaze features can improve alignment efficiency while maintaining performance.

Although these studies highlight the potential of gaze features for reward modeling, their approaches underrepresent a key property of human information processing during reading. They represent human gaze as a discrete token-wise feature associated with the directly fixated token. However, humans can acquire visual information beyond the point of fixation, because visual attention is spatially distributed rather than confined to a fixation (Schotter et al., 2012). The spatial region from which such information can be acquired during a fixation is often referred to as the perceptual span. In left-to-right reading, this region typically extends further to the right of fixation than to the left (Rayner, 2014).

Based on these findings, we hypothesize that redistributing token-wise gaze features across neighbor ing tokens according to an asymmetric profile will improve reward modeling by capturing aspects of human cognitive processing relevant to preference judgment. Specifically, we apply a profile inspired by the spatial distribution of human attention, assigning lower weights to tokens farther from the fixated token. The spread of the distribution on each side of fixation is controlled by separate param eters learned during reward model training. Across experimental settings, our approach improves preference prediction accuracy relative to a baseline. We also find that the optimized shape of the asymmetric distribution tends to extend more broadly to the right than to the left, which is consistent with the perceptual span reported in psychophysical research.

## 2 Background and Related Work

## 2.1 Visual Information Processing and Attention During Reading

Human reading is not a process in which visual information is acquired from one isolated word at a time. Instead, reading proceeds through a sequence of fixations, and readers obtain information from a spatially limited region around each fixation. Importantly, this region is not confined to the directly fixated word. Although visual processing is most precise at the fixated word, readers also extract useful information from neighboring text (Schotter et al., 2012). This suggests that a computational representation of gaze should extend across the region surrounding the fixated token to better reflect human information processing.

A study on the visual span during reading provides a more detailed account of how visual processing varies across the region surrounding fixation. Legge et al. (2001) found that letter-recognition accuracy within the visual span is spatially graded, with accuracy being highest near fixation and decreasing with distance from fixation. They modeled this pattern as an asymmetric Gaussian function, allowing the spread of recognition accuracy to differ between the left and right sides of fixation. This asymmetric function is analogous to a spatial distribution of visual attention, assigning different weights to locations around fixation. In left-to-right reading, visual attention is consistently found to be asymmetric, extending only about 3–4 letter spaces to the left of fixation but about 14–15 letter spaces to the right (Rayner, 2014). Figure 1 schematically illustrates this asymmetric attentional distribution during reading.

Computational models of reading also describe how human visual attention guides word recognition and eye movements across text. For example, sequential attention shift (SAS; Inhoff et al., 2005) models, such as E-Z Reader (Reichle, 2011; Reichle et al., 1998, 2006), describe attention as shifting from one word to the next as lexical processing progresses. In contrast, processing gradient (PG; Inhoff et al., 2005) models, such as SWIFT (Engbert et al., 2005), Glenmore (Reilly & Radach, 2006), and OB1-reader (Snell et al., 2018), assume parallel processing of multiple adjacent words under a spatially distributed attentional gradient. Notably, OB1-reader is particularly relevant to the current approach because it assumes an asymmetric attentional distribution.

## 2.2 Gaze-Augmented Natural Language Processing

Gaze information from eye-tracking data has been increasingly used to improve various downstream natural language processing (NLP) tasks. Mishra et al. (2016) augmented traditional linguistic features with gaze-derived cognitive features for sentiment analysis and sarcasm detection. Sood et al. (2020) and Zhang et al. (2024) incorporated gaze information into neural attention mechanisms for

![](images/b6488ebecf9972b4372d3be7868cf76c4529bb35f1cf9193ed782845248ce43b.jpg)

#

Figure 1: A schematic illustration of visual attention allocation during each fixation while reading.

paraphrase generation and code summarization, respectively. Together, these studies show that gaze features can be a useful resource for improving model performance across NLP tasks.

Recent research has extended the use of gaze information to language model post-training. For instance, Kiegeland et al. (2024) incorporated human eye-tracking data as direct feedback within Direct Preference Optimization (DPO). In addition, Deng et al. (2024) proposed a fine-tuning framework that reorders token embeddings based on simulated gaze trajectories. Gaze prediction models such as ScanTextGAN (Khurana et al., 2023) simulate human-like gaze patterns for new texts, enabling the use of gaze features without collecting additional eye-tracking data. GazeReward (López-Cardona et al., 2025), which serves as the foundation of our approach, uses gaze prediction models proposed by Huang and Hollenstein (2023) and Li and Rudzicz (2021).

## 2.3 GazeReward

In this section, we provide an overview of GazeReward (López-Cardona et al., 2025), the gazeaugmented reward modeling framework on which our approach directly builds.

GazeReward extends standard reward modeling by incorporating eye-tracking (ET) features as additional inputs to the reward model. In conventional reward modeling, the model receives a prompt-response pair and learns to assign a higher scalar reward to the response preferred by human annotators. GazeReward keeps this basic preference-learning objective, but augments the input representation with gaze-derived features.

Because human ET data are costly to collect and unavailable for most preference datasets, GazeReward uses pretrained ET prediction models to generate synthetic gaze features for each promptresponse pair. Given the same text that is passed to the reward model, the ET generator predicts token-level gaze features such as total reading time (TRT), first fixation duration (FFD), go-past time (GPT), number of fixations (nFix), and fixation proportion (fixProp). These features are then projected into the reward model’s embedding space through a neural feature projector, allowing gaze-derived information to be combined with the model’s text embeddings.

GazeReward considers two strategies for incorporating ET features into the reward model. In the GazeConcat setting, the projected ET embeddings are concatenated with the text embeddings, with special separator tokens marking the gaze feature segment. In the GazeAdd setting, the projected ET embeddings are added directly to the corresponding text embeddings. Since the tokenizer used by the ET prediction model may differ from the tokenizer used by the reward model, GazeReward also performs token remapping so that gaze features can be aligned with the reward model’s input tokens before they are combined with text representations.

After constructing the gaze-augmented input, the reward model is trained with the standard pairwise preference loss. For a prompt and two candidate responses, the model produces scalar rewards for the chosen and rejected responses, and training encourages the reward for the chosen response to be higher than that for the rejected response.

![](images/d6a4ca18999d4762e9531b656f20cd5ad3e5710e23cfc1ed58d8cdcc1b4cb3a7.jpg)  
Figure 2: An overview of the GazeReward framework with our redistribution module. First, we obtain original TRT values for a given sentence and redistribute them using an asymmetric Gaussian kernel with learnable parameters. The TRT values are then transformed into eye-tracking embeddings through the ET feature projector.

## 3 Methods

Our method modifies the gaze-feature construction stage of GazeReward. In the original GazeReward pipeline, token-level ET features are first generated by a pretrained ET generator and then directly projected into the reward model’s embedding space through the ET feature projector. In contrast, we insert an additional redistribution module between these two steps. Figure 2 shows the implementation of our redistribution module in the GazeReward pipeline.

GazeReward addresses five eye tracking features. Among these features, we use total reading time (TRT) as the core input to our formulation. TRT measures the cumulative duration of all fixations on a word and thus reflects the total attention allocated by readers.

Specifically, after obtaining token-level TRT values from the pretrained ET generator model proposed by Huang and Hollenstein (2023), we redistribute each token’s TRT value across its neighboring tokens using an asymmetric Gaussian kernel. A sequence of redistributed TRT values is then passed to the ET feature projector, which generates the eye-tracking embedding. The eye-tracking embedding is subsequently concatenated with (GazeConcat) or added to (GazeAdd) text embeddings. Our method does not alter the pairwise reward modeling objective or the overall reward model architecture; it changes the representation of gaze features before they are incorporated into the reward model.

While TRT provides information about which tokens were directly fixated, human text recognition is not confined to the fixated region alone. Cognitive and psychophysical studies of reading suggest that visual information is acquired over a visual span encompassing both central and peripheral vision (Legge et al., 2001). Accordingly, we redistribute token-level TRT values to adjacent tokens to reflect this perceptual span.

Formally, let $\{ \mathrm { t o k } _ { 1 } , \dots , \mathrm { t o k } _ { S } \}$ denote a sequence of S tokens, and let $\mathrm { T R T } _ { j }$ be the predicted TRT associated with token $\mathrm { \ t o k } _ { j }$ . For each fixation at $\operatorname { t o k } _ { j } ,$ we define a perceptual field over neighboring tokens using an asymmetric Gaussian kernel centered at tok<sub>j</sub>, where distances are measured in token indices. The weight from fixation j to token tok is:

$$
k _ { j \to i } = \exp \left( - \frac { ( i - j ) ^ { 2 } } { 2 \sigma ^ { ( \mathrm { s i d e } ) ^ { 2 } } } \right) ,\tag{1}
$$

where $\sigma ^ { \mathrm { ( s i d e ) } }$ $\sigma _ { \mathrm { l e f t } }$ when $i < j$ and $\sigma _ { \mathrm { r i g h t } }$ when $i \geq j$ , reflecting the asymmetric nature of human reading. Both $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { r i g h t } }$ are learnable parameters jointly optimized with the reward model during training, allowing the perceptual span to adapt to the downstream alignment objective rather than being fixed a priori.

![](images/fcb43e1cbc1beac5c58d32634cae8bb8be9f47075102d7f564f6b4519f94aa07.jpg)  
Figure 3: Asymmetric Gaussian redistribution of total reading time (TRT). Example of redistributing per-token TRT values for the sentence “The quick brown fox jumps over the lazy dog.” (Top) Original TRT values predicted per token. (Middle) Each cell indicates how much of source token’s TRT is redistributed to neighboring tokens weighted by an asymmetric Gaussian kernel with $\sigma _ { \mathrm { l e f t } } = 1 . 0$ and $\sigma _ { \mathrm { r i g h t } } = 4 . 0$ . (Bottom) Redistributed TRT values obtained by summing each column of the contribution matrix.

The weights are normalized per fixation to preserve the total TRT mass:

$$
\hat { k } _ { j  i } = \frac { k _ { j  i } } { \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { s } k _ { j  i ^ { \prime } } } .\tag{2}
$$

The final redistributed TRT for token tok<sub>i</sub> aggregates contributions from all fixated tokens:

$$
g _ { i } = \sum _ { j = 1 } ^ { S } T R T _ { j } \cdot \hat { k } _ { j  i } .\tag{3}
$$

The resulting sequence $\{ \mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { S } \}$ forms a dense token-level TRT distribution that captures both direct fixations and peripheral information acquired during reading.

Figure 3 illustrates this redistribution process using a simple sentence as an example. The top row shows the original token-level TRT values predicted by the ET generator. The middle matrix shows how each source token distributes its TRT value to neighboring target tokens according to the asymmetric Gaussian kernel. Each row therefore represents the redistribution pattern from a single source token. The bottom row shows the final redistributed TRT values obtained by summing the contributions in each column. This example visualizes how the proposed module transforms sparse token-local TRT values into a smoother, spatially distributed gaze representation before projection into the reward model.

## 4 Experiments

Our experiments are designed to address two central questions. First, we examine whether redistributing token-level TRT according to an asymmetric distribution assumption improves reward modeling

accuracy over the baseline, which uses token-wise gaze representation. Second, we analyze whether the optimized asymmetric Gaussian parameters $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { r i g h t } }$ show a directional pattern consistent with human visual attention during left-to-right reading.

## 4.1 Experimental Setup

## 4.1.1 Practical Adjustments and Datasets

We run our experiments on the OpenAssistant Conversations dataset (OASST1; Köpf et al., 2023), HelpSteer2 (Wang et al., 2024), and UltraFeedback (Cui et al., 2024). Due to consistency with the publicly released code of the original GazeReward implementation and to our computational resource constraints, we cap the maximum input length across the datasets and exclude samples exceeding this threshold. Table 1 shows the input length cap imposed on each dataset and the corresponding amount of data loss.

Table 1: Data loss under the input length cap across the three datasets.
<table><tr><td></td><td>OASST1</td><td>HelpSteer2</td><td>UltraFeedback</td></tr><tr><td>Input Length Cap (characters)</td><td>5,000</td><td>10,000</td><td>10,000</td></tr><tr><td>Total Data Samples</td><td>8,161</td><td>7,592</td><td>63,966</td></tr><tr><td>Data Loss</td><td>307 (3.76%)</td><td>241 (3.17%)</td><td>1,632 (2.55%)</td></tr></table>

Additionally, we set the per-device batch size to 1 and use gradient accumulation over 8 steps, preserving an effective batch size of 8, which matches the batch size used in the prior study.

## 4.1.2 Model Choice and Gaze Integration Settings

For the experiments on OASST1, we focus on two variants of Llama-3-based reward models: Llama 3-8B and Llama-3-8B-Instruct (Grattafiori et al., 2024). We include both variants to assess whether the effect of the redistribution is consistent across the two Llama-3 initializations, rather than restricting the evaluation to a single model variant. We also evaluate both gaze integration settings (GazeConcat and GazeAdd) introduced in GazeReward.

## 4.1.3 Training Configurations for Redistribution Parameters

The $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { r i g h t } }$ parameters are initialized to 1.0 and optimized jointly with the base parameters during training. Because these parameters operate at a different scale from the main reward model parameters, we use a separate learning rate of $5 \times 1 0 ^ { - 2 }$ for $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { r i g h t } }$ , while the remaining model parameters are trained with a learning rate of $5 \times 1 0 ^ { - 5 }$ . These hyperparameter values were selected through empirical tuning. After training, we analyze the $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { { r i g h t } } }$ values from the best checkpoint.

For each experimental condition, we conduct six runs using different random seeds and report the mean reward modeling accuracy with the standard deviation. Further details on the experimental setups are provided in Appendix B.

## 5 Results

Table 2 reports the reward modeling accuracy on OASST1 across six random seeds. Overall, our approach improves performance over the corresponding baseline in all four experimental settings. In the GazeConcat setting, our approach achieves 74.73% accuracy with Llama-3-8B-Instruct, compared to 68.38% for the baseline, corresponding to a 9.29% relative improvement. For Llama-3-8B, the GazeConcat setting also improves from 69.23% to 74.81%, yielding an 8.06% relative improvement. These results indicate that asymmetrically redistributing gaze features can improve reward preference prediction across both gaze integration methods and both Llama-3 model variants.

Table 2: Mean reward modeling accuracy on OASST1 across six random seeds. Diff. denotes the relative improvement of our approach over the corresponding baseline (mean ± standard deviation).
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Llama-3-8B-Instruct</td><td colspan="2">Llama-3-8B</td></tr><tr><td>Accuracy (%)</td><td>Diff. (%)</td><td>Accuracy (%)</td><td>Diff. (%)</td></tr><tr><td rowspan="2">GazeConcat</td><td>Baseline</td><td> $6 8 . 3 8 \pm 3 . 8 8$ </td><td></td><td> $6 9 . 2 3 \pm 4 . 9 2$ </td><td></td></tr><tr><td>Our Approach</td><td> $7 4 . 7 3 \pm 1 . 5 5$ </td><td>9.29</td><td> $7 4 . 8 1 \pm 1 . 3 2$ </td><td>8.06</td></tr><tr><td rowspan="2">GazeAdd</td><td>Baseline</td><td> $7 0 . 3 0 \pm 1 . 2 5$ </td><td></td><td> $6 9 . 2 7 \pm 0 . 8 1$ </td><td></td></tr><tr><td>Our Approach</td><td> $7 1 . 8 9 \pm 1 . 2 2$ </td><td>2.26</td><td> $7 1 . 1 7 \pm 1 . 2 1$ </td><td>2.74</td></tr></table>

Table 3 summarizes the asymmetric Gaussian parameters observed at the best checkpoints. In most settings, the right-side spread is larger than the left-side spread, indicating that the optimized redistribution tends to assign a wider influence range to tokens on the right side of the fixated point. This tendency is especially clear in the GazeConcat setting, where $\sigma _ { \mathrm { r i g h t } }$ exceeds $\sigma _ { \mathrm { l e f t } }$ for both Llama-3-8B-Instruct and Llama-3-8B. The GazeAdd setting shows a similar pattern for Llama-3-8B-Instruct, while the difference is smaller for Llama-3-8B.

Table 3: Asymmetric Gaussian parameters from the best checkpoints on OASST1 across six random seeds (mean ± standard deviation).
<table><tr><td rowspan="2"></td><td colspan="2">Llama-3-8B-Instruct</td><td colspan="2">Llama-3-8B</td></tr><tr><td> $\sigma _ { \mathrm { l e f t } }$ </td><td> $\sigma _ { \mathrm { r i g h t } }$ </td><td> $\sigma _ { \mathrm { l e f t } }$ </td><td> $\sigma _ { \mathrm { r i g h t } }$ </td></tr><tr><td>GazeConcat</td><td> $1 . 6 2 \pm 1 . 1 8$ </td><td> $2 . 2 5 \pm 1 . 7 1$ </td><td> $0 . 8 0 \pm 0 . 5 6$ </td><td> $3 . 4 7 \pm 2 . 1 3$ </td></tr><tr><td>GazeAdd</td><td> $0 . 5 8 \pm 0 . 2 1$ </td><td> $1 . 9 9 \pm 1 . 1 7$ </td><td> $0 . 6 4 \pm 0 . 2 3$ </td><td> $0 . 8 0 \pm 0 . 3 1$ </td></tr></table>

Figure 4 visualizes the token-level distribution of attention defined by the estimated $\sigma _ { \mathrm { l e f t } }$ and $\sigma _ { \mathrm { { r i g h t } } }$ over a schematic perceptual span during reading. The upper panel (a) shows a perceptual span around the fixated point (on the character o in the wordfox) based on the finding that the span extends 4 letter spaces to the left (up to the character w in the word brown) and 15 letter spaces to the right (up to the

(a)  
(b)  
![](images/3041488697787babb6d59d5611e67b1e63581754939e9915d545a2f29abc0081.jpg)  
Figure 4: An illustration of the human perceptual span and the proposed token-level gaze redistribution. (a) The asymmetric human perceptual span in left-to-right reading. (b) TRT values redistributed to the tokens adjacent to the token $f o x .$ The red dotted lines indicate the hypothetical boundaries of the human perceptual span mapped onto the tokenized sentence.

character h in the word the) from the fixated point during left-to-right reading (Rayner, 2014). The lower panel (b) illustrates the asymmetric Gaussian distribution derived from the best-performing Llama-3-8B model in the GazeConcat setting, where mean $\sigma _ { \mathrm { l e f t } } = 0 . 8 0$ and mean $\sigma _ { \mathrm { r i g h t } } = 3 . 4 7$ . The blue curved line shows the redistributed TRT values across the adjacent tokens given that the token fox has a TRT value of 4.221. This example is not intended as a direct quantitative mapping from letter spaces to tokens, but as a qualitative illustration of the learned redistribution pattern across letters.

We further evaluate our method on two additional datasets, HelpSteer2 and UltraFeedback, using Llama-3-8B as the reward model backbone and GazeConcat as the gaze integration method, as this configuration achieved the highest accuracy on OASST1. Table 4 summarizes the results.

Table 4: Reward modeling performance on additional preference datasets. Results are reported as mean ± standard deviation across six random seeds. Diff. denotes the relative improvement of our approach over the corresponding baseline.
<table><tr><td>Dataset</td><td>Method</td><td>Accuracy (%)</td><td>Diff. (%)</td><td> $\sigma _ { \mathrm { l e f t } }$ </td><td> $\sigma _ { \mathrm { r i g h t } }$ </td></tr><tr><td rowspan="2">HelpSteer2</td><td>Baseline</td><td> $5 9 . 4 8 \pm 3 . 6 6$ </td><td></td><td></td><td></td></tr><tr><td>Our Approach</td><td> $6 2 . 6 4 \pm 1 . 6 8$ </td><td>5.31</td><td> $3 . 1 8 \pm 2 . 3 4$ </td><td> $1 . 8 5 \pm 2 . 0 1$ </td></tr><tr><td rowspan="2">UltraFeedback</td><td>Baseline</td><td> $6 5 . 6 5 \pm 2 . 1 2$ </td><td></td><td></td><td></td></tr><tr><td>Our Approach</td><td> $7 4 . 2 0 \pm 0 . 4 6$ </td><td>13.02</td><td> $1 . 4 8 \pm 1 . 6 9$ </td><td> $3 . 7 9 \pm 1 . 9 4$ </td></tr></table>

Our approach improves reward model performance on both additional datasets. On HelpSteer2, our approach achieves 62.64% accuracy compared with 59.48% for the baseline, corresponding to a 5.31% relative improvement. A substantially larger gain is observed on UltraFeedback, where accuracy increases from 65.65% to 74.20%, yielding a 13.02% relative improvement.

HelpSteer2 showed the smallest improvement among the datasets and was the only dataset on which the learned profile extended further to the left than to the right. We speculate that the differences in the number of candidate responses available for chosen–rejected pair construction may help explain the distinct results on HelpSteer2. In our processed data, OASST1 contains an average of 2.82 candidate responses per prompt, while UltraFeedback contains 4 responses for nearly every prompt. For these datasets, our preprocessing selects the highest- and lowest-rated responses as the chosen–rejected pair. By contrast, HelpSteer2 provides only 2 responses per prompt in almost all cases, and our preprocessing designates the response with the higher rating as chosen and the other as rejected. Thus, HelpSteer2 generally requires using the only available pair, whereas OASST1 and UltraFeedback allow selecting maximally separated pairs from larger candidate pools. This constraint may yield weaker preference contrasts, potentially making preference prediction more challenging and providing a less informative training signal for learning the redistribution profile.

## 6 Ablation Study

In this section, we examine the contributions of different design choices in our method, using Llama-3-8B as the reward model backbone in the GazeConcat setting on the OASST1 dataset. We compare the original pinpoint representation against uniform redistribution, a fixed asymmetric Gaussian, a learnable symmetric Gaussian, and our learnable asymmetric Gaussian across six random seeds. Table 5 presents the results.

Our ablation study supports the following observations. First, the fixed 1:4 condition achieved higher mean accuracy (71.46%) than the uniform 1:4 condition (70.18%). Although the difference was not large enough to draw a firm conclusion given the variability across seeds, the results suggest that implementing a Gaussian-shaped distribution may be more effective than uniformly distributing the TRT values over the neighboring tokens.

Second, the learnable symmetric and asymmetric Gaussian conditions achieve higher accuracy than the uniform and fixed Gaussian conditions. These results support the importance of learning the redistribution parameters for achieving the improvement.

Table 5: Reward modeling accuracy across five conditions: Pinpoint baseline, Uniform 1:4, Fixed 1:4, Symmetric, and Asymmetric. A description for each condition and its accuracy (mean ± standard deviation) across the six random seeds are provided. Diff. denotes the relative improvement over the pinpoint baseline. Where applicable, the learned values of the σ parameters are shown.
<table><tr><td>Method</td><td>Description</td><td>Accuracy (%)</td><td>Diff. (%)</td><td>Learned σ</td></tr><tr><td>Baseline</td><td>No redistribution; each pre- dicted TRT value remains assigned to its original to- ken</td><td> $6 9 . 2 3 \pm 4 . 9 2$ </td><td></td><td>N/A</td></tr><tr><td>Uniform 1:4</td><td>TRT is uniformly redis- tributed over one token to the left and four tokens to the right</td><td> $7 0 . 1 8 \pm 1 . 1 5$ </td><td>1.37</td><td>N/A</td></tr><tr><td>Fixed 1:4</td><td>Predetermined asymmetric Gaussian with a fixed 1:4 left-to-right spread</td><td> $7 1 . 4 6 \pm 2 . 2 8$ </td><td>3.22</td><td>N/A</td></tr><tr><td>Symmetric</td><td>A single learnable σ param- eter shared by the left and right sides</td><td> $7 4 . 1 9 \pm 1 . 8 6$ </td><td>7.16</td><td> $\sigma = 3 . 2 8 \pm 0 . 8 4$ </td></tr><tr><td>Asymmetric</td><td>Independently learned left- and right-side σ parameters</td><td> $7 4 . 8 1 \pm 1 . 3 2$ </td><td>8.06</td><td> $\sigma _ { \mathrm { l e f t } } = 0 . 8 0 \pm 0 . 5 6$   $\sigma _ { \mathrm { r i g h t } } = 3 . 4 7 \pm 2 . 1 3$ </td></tr></table>

Lastly, the asymmetric Gaussian condition achieved the highest mean accuracy at 74.81%. However, its advantage over the symmetric condition (74.19%) was small relative to the variability across seeds. Hence, the strong performance improvement of the symmetric Gaussian indicates that the overall gain cannot be attributed solely to the asymmetric design of the redistribution.

## 7 Discussion

The current work has three main implications. First, the accuracy improvements in Table 2 suggest that gaze features are more informative when they are modeled as spatially distributed attention rather than as strictly token-local signals. Across all evaluation settings, the proposed redistribution approach outperforms the corresponding pinpoint gaze baseline, suggesting that treating gaze location as the center of an attentional distribution during text processing is a promising approach.

Second, the optimized asymmetric Gaussian parameters may serve as a bridge between reward modeling in LLMs and human attention modeling during reading. The fitted model shows rightward asymmetry across all experimental settings in our evaluation on OASST1 and UltraFeedback, with a larger average right-side width than left-side width (Table 3). This pattern is directionally consistent with the asymmetric perceptual span reported in left-to-right reading, where visual information is acquired from a broader region to the right of fixation than to the left. We further examine this correspondence through a comparison with the OB1-reader model; detailed procedures and results are provided in Appendix A. This suggests that the gaze-derived attentional distribution optimized for LLMs may reflect certain spatial properties of human attention, although this does not imply that the model provides a direct estimate of human attention.

The results of the ablation study provide further insight into how this may operate. The symmetric Gaussian condition learns $\sigma = 3 . 2 8$ , close to the asymmetric Gaussian condition’s right-side width $( \sigma _ { \mathrm { r i g h t } } = 3 . 4 7 )$ , whereas the asymmetric condition learns a substantially smaller left-side width $( \sigma _ { \mathrm { l e f t } } = 0 . 8 0 )$ . This pattern suggests that information to the right of the fixated token contributes more strongly to preference prediction. A symmetric Gaussian with a width comparable to $\sigma _ { \mathrm { r i g h t } }$ may capture a similar range of predictive information, but less efficiently because it necessarily distributes weight equally in both directions.

Third, the proposed gaze redistribution method generally yields more stable performance across random seeds than the pinpoint gaze baseline, as reflected in the smaller standard deviations reported in Table 2. We hypothesize that this stability improvement arises because pinpoint gaze features are sparse and token-local: a high TRT value assigned to a single token can disproportionately affect the learned representation, making the model sensitive to noise in the predicted gaze features and to seed-dependent optimization dynamics. The proposed redistribution mechanism smooths such information over neighboring tokens, reducing the influence of isolated noisy values and providing a denser representation.

Overall, these findings suggest that modeling gaze as a spatially distributed feature provides a useful extension to fixation-centric gaze features in reward modeling. More broadly, the current study highlights how grounding machine learning frameworks in established findings from cognitive science offers a principled way to improve model performance and stability.

## 8 Limitations and Future Work

We acknowledge several limitations of this work that provide avenues for future research. First, our redistribution mechanism remains a simplified approximation of human visual attention during reading. Although asymmetric Gaussian redistribution captures several important properties of the attentional distribution, including distance-based decay and left–right asymmetry, human attention is more complex and context-dependent. The attentional distribution during reading may vary depending on reading skill, text difficulty, typography, task demands, and linguistic context.

More importantly, the Gaussian kernel used in the current study imposes a monotonic decay from the fixation center. This assumption is useful as a simple approximation, but it may not fully capture the structure of visuospatial attention. For example, prior work has suggested that the attentional field may not simply decrease with distance, but can show a Mexican-hat-like pattern, where attention is enhanced near the attended location and suppressed in surrounding regions (Müller et al., 2005). Recent evidence also suggests that visual spatial attention can be nonmonotonic and may vary across individuals (Lee & Pitt, 2022). These findings imply that future gaze-augmented reward models may benefit from more flexible redistribution functions beyond a single asymmetric Gaussian, such as distributions with inhibitory surround effects, nonmonotonic profiles, or individually adaptive span parameters.

Additionally, the current implementation redistributes TRT values based on token-level distance. This is a design choice for compatibility with the ET prediction model, which predicts token-level TRT values. Future work could explore redistribution based on letter-level distance, which may better reflect the spatial scale of human visual attention.

Lastly, although TRT can serve as a proxy for the amount of attention allocated to each token, other gaze features not considered in the current study may provide complementary information. Future work may extend the framework to multiple gaze features, such as first fixation duration, fixation probability, and the number of fixations, which may capture different aspects of human visual attention.

## Acknowledgments and Disclosure of Funding

This work was supported by the New Faculty Settlement Research Program through Korea Advanced Institute of Science & Technology (Grant No. G04240059 [to SHL]); the Data Science Convergence Human Resource Development Program through the National Research Foundation (NRF) of Korea funded by the Ministry of Science and ICT (Grant No. RS-2022-NR068758 [to SHL]); the AI Science Hub Program through the Ministry of Science and ICT (Grant No. N10260136 [to SHL]); and the Industry Research and Development Program funded by FMMENTALCARE (Grant No. G01250572 [to SHL]).

## References

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., Liu, Z., & Sun, M. (2024). ULTRAFEEDBACK: Boosting language models with scaled AI feedback. Proceedings of the 41st International Conference on Machine Learning, 235, 9722–9744.

Deng, S., Prasse, P., Reich, D., Scheffer, T., & Jäger, L. (2024). Fine-tuning pre-trained language models with gaze supervision. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 217–224. https://doi.org/10.18653/ v1/2024.acl-short.21

Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: A dynamical model of saccade generation during reading. Psychological Review, 112(4), 777–813.

Galliamov, K., Titov, I., & Pershin, I. (2025). Enhancing RLHF with human gaze modeling. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 30637–30643.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., . . . Ma, Z. (2024). The Llama 3 herd of models. arXiv preprint arXiv:2407.21783. https://doi.org/10.48550/arXiv.2407.21783

Huang, X., & Hollenstein, N. (2023). Long-range language modeling with selective cache. Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, 4838–4858.

Inhoff, A. W., Eiter, B. M., & Radach, R. (2005). Time course of linguistic information extraction from consecutive words during eye fixations in reading. Journal of Experimental Psychology: Human Perception and Performance, 31(5), 979–995. https://doi.org/10.1037/0096- 1523.31.5.979

Just, M. A., & Carpenter, P. A. (1980). A theory of reading: From eye fixations to comprehension. Psychological Review, 87(4), 329–354. https://doi.org/10.1037/0033-295X.87.4.329

Khurana, V., Kumar, Y., Hollenstein, N., Kumar, R., & Krishnamurthy, B. (2023). Synthesizing human gaze feedback for improved NLP performance. Proceedings ofthe 17th Conference of the European Chapter of the Association for Computational Linguistics, 1895–1908. https://doi.org/10.18653/v1/2023.eacl-main.139

Kiegeland, S., Reich, D. R., Cotterell, R., Jäger, L. A., & Wilcox, E. (2024). The pupil becomes the master: Eye-tracking feedback for tuning LLMs. ICML 2024 Workshop on LLMs and Cognition.

Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z. R., Stevens, K., Barhoum, A., Nguyen, D., Stanley, O., Nagyfi, R., ES, S., Suri, S., Glushkov, D., Dantuluri, A., Maguire, A., Schuhmann, C., Nguyen, H., & Mattick, A. (2023). OpenAssistant conversations – democratizing large language model alignment. Advances in Neural Information Processing Systems, 36, 47669–47681.

Lee, S. H., & Pitt, M. A. (2022). Individual differences in selective attention reveal the nonmonotonicity of visual spatial attention and its association with working memory capacity. Journal of Experimental Psychology: General, 151(4), 749–762. https://doi.org/10.1037/xge0000801

Legge, G. E., Mansfield, J. S., & Chung, S. T. L. (2001). Psychophysics of reading: XX. linking letter recognition to reading speed in central and peripheral vision. Vision Research, 41(6), 725–743.

Li, B., & Rudzicz, F. (2021, June). TorontoCL at CMCL 2021 shared task: RoBERTa with multistage fine-tuning for eye-tracking prediction. In E. Chersoni, N. Hollenstein, C. Jacobs, Y. Oseki, L. Prévot, & E. Santus (Eds.), Proceedings ofthe workshop on cognitive modeling and computational linguistics (pp. 85–89). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.cmcl-1.9

Lopes Rego, A. T., Snell, J., & Meeter, M. (2024). Language models outperform cloze predictability in a cognitive model of reading. PLOS Computational Biology, 20(9), e1012117. https: //doi.org/10.1371/journal.pcbi.1012117

López-Cardona, Á., Segura, C., Karatzoglou, A., Abadal, S., & Arapakis, I. (2025). Seeing eye to AI: Human alignment via gaze-based response rewards for large language models [arXiv:2410.01532]. International Conference on Learning Representations (ICLR).

Mishra, A., Kanojia, D., Nagar, S., Dey, K., & Bhattacharyya, P. (2016). Leveraging cognitive features for sentiment analysis. Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, 156–166.

Müller, N. G., Mollenhauer, M., Rösler, A., & Kleinschmidt, A. (2005). The attentional field has a Mexican hat distribution. Vision Research, 45(9), 1129–1137.

Rayner, K. (2014). The gaze-contingent moving window in reading: Development and review. Visual Cognition, 22(3–4), 242–258.

Reichle, E. D. (2011). Serial-attention models of reading. In S. P. Liversedge, I. D. Gilchrist, & S. Everling (Eds.), The oxford handbook of eye movements (pp. 767–786). Oxford University Press.

Reichle, E. D., Pollatsek, A., Fisher, D. L., & Rayner, K. (1998). Toward a model of eye movement control in reading. Psychological Review, 105(1), 125–157.

Reichle, E. D., Pollatsek, A., & Rayner, K. (2006). E-Z Reader: A cognitive-control, serial-attention model of eye-movement behavior during reading. Cognitive Systems Research, 7(1), 4–22.

Reilly, R. G., & Radach, R. (2006). Some empirical tests of an interactive activation model of eye movement control in reading. Cognitive Systems Research, 7(1), 34–55.

Schotter, E. R., Angele, B., & Rayner, K. (2012). Parafoveal processing in reading. Attention, Perception, & Psychophysics, 74(1), 5–35.

Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements in text reading. Psychological Review, 125(6), 969–984. https://doi.org 10.1037/rev0000119

Sood, E., Tannert, S., Müller, P., & Bulling, A. (2020). Improving natural language processing tasks with human gaze-guided neural attention. Advances in Neural Information Processing Systems, 33, 6327–6341.

Wang, Z., Dong, Y., Delalleau, O., Zeng, J., Shen, G., Egert, D., Zhang, J. J., Sreedhar, M. N., & Kuchaiev, O. (2024). HelpSteer 2: Open-source dataset for training top-performing reward models. Advances in Neural Information Processing Systems, 37, 1474–1501.

Zhang, Y., Li, J., Karas, Z., Bansal, A., Li, T. J.-J., McMillan, C., Leach, K., & Huang, Y. (2024). Eye-Trans: Merging human and machine attention for neural code summarization. Proceedings ofthe ACM on Software Engineering, 1(FSE), 115–136.

## A Comparison with the OB1-reader Model

To further examine whether the learned token-space redistribution aligns quantitatively with an established cognitive model of reading, we compare our learned asymmetric Gaussian kernel against simulated attentional trajectories from the OB1-reader model (Lopes Rego et al., 2024; Snell et al., 2018). OB1-reader is a cognitive model designed to simulate how readers guide their gaze movements and allocate visual attention across words. Rather than processing text strictly one word at a time, the model assumes that readers process multiple neighboring words simultaneously, with the amount of visual attention allocated to each word decaying smoothly with distance from the attentional focus. Specifically, visual input across a five-word window is modulated by an asymmetric attentional gradient that extends further in the reading direction (i.e., to the right side).

## A.1 Experimental Setup

For this analysis, we select the Llama-3-8B reward model under the GazeConcat setting on OASST1, as it achieved higher mean downstream accuracy than Llama-3-8B-Instruct across six random seeds (Table 2). Across the six checkpoints, the learned parameter values averaged $\sigma _ { \mathrm { l e f t } } = 0 . 8 0 4$ and $\sigma _ { \mathrm { r i g h t } } = 3 . 4 7 1$ , corresponding to a 1:4.32 left-to-right width ratio. These parameters were optimized solely through downstream preference learning without any exposure to the Provo Corpus or OB1- reader simulations.

Using 100 OB1-reader trajectories generated on the Provo Corpus with the implementation provided by Lopes Rego et al. (2024), we project OB1-reader’s letter-level attentional weights onto the relative T5-token positions used by our redistribution module.

In OB1-reader, the attentional weight $a _ { i }$ assigned to a letter i at a signed letter distance $f _ { i }$ from the focus of attention is modulated by an asymmetric Gaussian distribution formally defined as follows (Snell et al., 2018, Equation 2):

$$
a _ { i } = \frac { 1 } { \mathrm { w i d t h } } \cdot \exp \left( - \frac { f _ { i } ^ { 2 } } { 2 \cdot ( \mathrm { w i d t h } \cdot \mathrm { a s y m } ) ^ { 2 } } \right) + c _ { a } .\tag{4}
$$

We evaluate OB1-reader under the 1:4 left-to-right attentional asymmetry $( \mathrm { a s y m } = 0 . 2 5$ toward the left and asym = 1.0 toward the right). This 1:4 configuration directly models established psychophysical estimates of the human perceptual span in left-to-right reading, where effective processing extends approximately four times further to the right of fixation (14–15 letter spaces) than to the left (3–4 letter spaces; Rayner, 2014; Snell et al., 2018).

## A.2 Comparison Conditions and Metrics

We evaluate unit-TRT allocation across three conditions over OB1-reader’s mapped five-word window (comprising one word to the left and three words to the right of the fixated word n in the Provo simulation setup of Lopes Rego et al., 2024):

• No redistribution (Pinpoint): All weight remains strictly assigned to the directly fixated token without spatial spreading.

• Symmetric Gaussian: For each checkpoint, we construct a symmetric baseline with equal left and right Gaussian widths $( \sigma _ { \mathrm { l e f t } } = \sigma _ { \mathrm { r i g h t } } )$ calibrated to match the overall token-space spread of the learned asymmetric kernel. Because the five-word window (one word to the left and three words to the right of fixation) is itself asymmetrically bounded, equal Gaussian widths do not necessarily yield equal total weight on the two sides.

• Learned asymmetric Gaussian (Ours): Token-level redistribution using each checkpoint’s independently learned parameter values $( \sigma _ { \mathrm { l e f t } } , \sigma _ { \mathrm { r i g h t } } )$ as defined in Equations (1)–(2).

All comparison metrics are computed separately for each of the six random-seed checkpoints and then averaged. We report our results using two metrics: (1) Spearman rank correlation, which measures agreement in the relative ranking of token positions across the window, and (2) Jensen–Shannon (JS) divergence, which evaluates the discrepancy between the complete normalized attentional allocation distributions.

Table 6: Quantitative comparison between redistributed token-level TRT under different conditions and attention simulated by OB1-reader on the Provo Corpus under the 1:4 left-to-right attentional asymmetry $( \mathrm { a s y m _ { l e f t } = 0 . 2 5 , a s y m _ { r i g h t } = 1 . 0 ) }$ . Values represent the mean across six random seeds.
<table><tr><td>Redistribution Condition</td><td>Spearman Correlation (↑)</td><td>JS Divergence (↓)</td></tr><tr><td>No redistribution (Pinpoint)</td><td>0.4645</td><td>0.2756</td></tr><tr><td>Symmetric Gaussian</td><td>0.8958</td><td>0.1584</td></tr><tr><td>Learned asymmetric Gaussian (Ours)</td><td>0.9095</td><td>0.1473</td></tr></table>

## A.3 Results and Analysis

Table 6 summarizes the results. Both Gaussian redistribution conditions correspond substantially better to OB1-reader’s attentional allocation than the pinpoint baseline. Specifically, the symmetric Gaussian condition increases Spearman correlation from 0.4645 to 0.8958 and reduces JS divergence from 0.2756 to 0.1584. As for the learned asymmetric Gaussian, it outperforms the paired symmetric condition on both Spearman correlation (0.9095 vs. 0.8958) and JS divergence (0.1473 vs. 0.1584) under OB1-reader’s 1 : 4 asymmetry setting.

The numerical differences between the learned asymmetric and symmetric conditions are modest by design because their overall spatial spread was matched. However, they differ primarily in their directional left–right mass allocation. A breakdown of the pooled attentional weights across the left, center (fixated), and right regions of the window clarifies this effect:

• Center allocation: The asymmetric and symmetric redistributions assign nearly identical weight to the fixated center position (26.83% vs. 26.69%, respectively).

• Leftward allocation: The asymmetric redistribution substantially reduces leftward weight from 26.65% (symmetric) to 11.39%, bringing it much closer to OB1-reader’s sharp leftward decay.

• Rightward allocation: The asymmetric kernel assigns more weight toward the right side (61.77% vs. 46.66% for the symmetric kernel).

These results suggest that Gaussian spatial spreading accounts for much of the shared local structure between redistributed TRT and attention simulated by OB1-reader, while the learned rightward asymmetry provides additional directional alignment with the 1 : 4 attentional profile of OB1-reader.

## B Experiment Details

## B.1 Dataset Processing

We conduct experiments on the OpenAssistant Conversations dataset (OASST1; Köpf et al., 2023), HelpSteer2 (Wang et al., 2024), and UltraFeedback (Cui et al., 2024). Following the reward modeling setup, each example is formatted as a prompt–chosen–rejected triplet, where the chosen response is expected to receive a higher reward than the rejected response. We use the English subset of the dataset, since the eye-tracking prediction model used in our experiments was trained on English reading data.

## B.2 Training Configuration

Table 7: Training configuration
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Datasets</td><td>OASST1, HelpSteer2, UltraFeedback</td></tr><tr><td>Backbones</td><td>Llama-3-8B, Llama-3-8B-Instruct</td></tr><tr><td>Training epochs</td><td>2</td></tr><tr><td>Per-device batch size</td><td>1</td></tr><tr><td>Gradient accumulation steps</td><td>8</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate scheduler</td><td>cosine</td></tr><tr><td>Minimum learning-rate ratio</td><td>0.7</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>LoRA rank</td><td>8</td></tr><tr><td>LoRA alpha</td><td>32</td></tr><tr><td>LoRA dropout</td><td>0.1</td></tr><tr><td>Gaze projector dropout</td><td>0.1 / 0.3</td></tr><tr><td>Initial  $\sigma _ { \mathrm { l e f t } }$ </td><td>1.0</td></tr><tr><td>Initial  $\sigma _ { \mathrm { r i g h t } }$ </td><td>1.0</td></tr><tr><td>Random seeds</td><td>41, 42, 43, 44, 45, 46</td></tr></table>

## B.3 Evaluation Protocol

Reward modeling performance is evaluated using pairwise preference accuracy. For each prompt, the model assigns scalar reward scores to the chosen and rejected responses. A prediction is counted as correct when the chosen response receives a higher reward than the rejected response:

$$
r _ { \theta } ( x , y _ { w } ) > r _ { \theta } ( x , y _ { l } ) .
$$

Accuracy is computed as the percentage of correctly ranked response pairs. This is the same evaluation criterion used in prior reward modeling work and in GazeReward, where accuracy is defined by whether the preferred response receives a higher reward score than the less preferred response.

## B.4 Computing Environment

We trained the models on three local servers.

The first server was equipped with an AMD Ryzen Threadripper PRO 5965WX 24-Core CPU, 503 GiB of RAM, and 4 NVIDIA RTX A6000 48 GB GPUs.

The second server was equipped with an AMD Ryzen Threadripper PRO 7975WX 32-Core CPU, 251 GiB of RAM, and 1 NVIDIA RTX A5000 24 GB GPU.

The third server was equipped with 2 Intel Xeon Gold 6226R 16-Core CPUs, 376 GiB of RAM, and 4 NVIDIA RTX A5000 24GB GPUs.

Each training was done on a single GPU at a time.