# MANTIS: Multi-Axis News Transformations for Instance-Wise Stress Testing and Worst-Case Training in Fake News Detection

Dexin Liu<sup>1,2,⋆</sup> Fan Yang<sup>3,⋆</sup> Zimu Wang<sup>1,2,⋆</sup> Xunfei Zhu<sup>3</sup> Bolin Ren<sup>1</sup> Bingyan Zhang<sup>3</sup> Wei Wang<sup>1</sup> Qi Chen<sup>1,†</sup>

<sup>1</sup>Xi’an Jiaotong-Liverpool University, Suzhou, China

<sup>2</sup>University of Liverpool, Liverpool, United Kingdom

<sup>3</sup>Nanjing Sibei Intelligent Technology Co., Ltd., Nanjing, China Dexin.Liu24@student.xjtlu.edu.cn Qi.Chen02@xjtlu.edu.cn <sup>⋆</sup>Equal contribution. <sup>†</sup>Corresponding author.

## Abstract

Automated fake news detection is vital on social media, where fake news is easily rewritten to fool detectors. However, existing approaches typically rely on superficial writing cues or single-style text rewrites, resulting in poor robustness under diverse cosmetic modifications. We construct MANTIS-Bench, a multi-axis presentation-attack benchmark that pairs original news with rewritten views across writing style, emotional expression, and dissemination framing. We further propose MAN-TIS, an instance-adaptive adversarial framework enabling dynamic worst-case training tailored to each news article with zero inferencetime overhead. This reframes fake news defense from heuristic format smoothing into targeted, fact-invariant credibility assessment. Experiments on MANTIS-Bench show that MAN-TIS reduces the Union Attack Success Rate (UASR) by 3.55 percentage points relative to SheepDog on GossipCop and consistently outperforms prior single-axis defenses on Macro-F1 while maintaining competitive clean accuracy, demonstrating the effectiveness of multiaxis and instance-adaptive worst-case training.

## 1 Introduction

Human credibility assessments frequently rely on surface shortcuts, readily equating fluent phrasing, professional syntax, and neutral tone with factual truth (Reber and Schwarz, 1999). The rise of large language models (LLMs) enables malicious actors to mass-produce deceptive claims wrapped in such authoritative packaging at near-zero cost (Goldstein et al., 2023), severely amplifying public cognitive biases and undermining social trust (Lazer et al., 2018). Automated fake news detection has therefore emerged as an essential safeguard for online information integrity (Guo et al., 2022). However, static benchmark performance does not reflect robustness to changes in presentation, with these models easily fall into shortcut learning (Geirhos et al., 2020; Li et al., 2024a), latching onto superficial presentation patterns such as stylistic tone or emotional intensity rather than verifying underlying factual claims. This reliance on presentation cues can lead to inconsistent predictions when writing style or emotional expression changes, even if the underlying claims remain unchanged (Wu et al., 2024; Tahmasebi et al., 2026).

![](images/f93f4092b9c01ad4d131da2bb231b48681015c529b1ed7991eaab301a274e685.jpg)  
Figure 1: Illustration of presentation-level vulnerability in fake news detection. Different presentation variations lead to inconsistent predictions from the same detector.

To mitigate presentation-level reliance, recent studies have developed defenses targeting writing style, such as SheepDog (Wu et al., 2024) and Ad-Style (Park et al., 2025), and sentiment, such as AdSent (Tahmasebi et al., 2026). However, robustness to a single perturbation family does not guarantee robustness to others (Tramèr and Boneh,

2019; Wang and Gan, 2023; Wang et al., 2024). We argue that applying uniform rewrites across all samples may be suboptimal, as different instances can be vulnerable to different presentation transformations. As illustrated in Figure 1, separate changes to writing style, emotional expression, or dissemination framing can flip the detector’s prediction for the same news item from fake to real, while preserving its core claim. Without explicitly such heterogeneous weaknesses, single-axis defenses may leave cross-axis blind spots and fail to adapt to the varied vulnerability patterns across individual news items.

To address these gaps, we construct MANTIS-Bench, a Benchmark of Multi-Axis News Transformations for Instance-Wise Stress Testing. Under core-claim preservation constraints, it pairs original news with rewritten views across writing style, emotional expression, and dissemination framing. Building on this benchmark, we introduce MANTIS, an instance-wise worst-case adversarial training framework for robust fake news detection. As shown in Figure 2, MANTIS selects the most challenging transformed view for each instance and optimizes the detector jointly with the original news, achieving instance-adaptive robustness without inference-time overhead. Extensive experiments on MANTIS-Bench demonstrate that MANTIS substantially suppresses attack success rates and outperforms strong baselines.

• We construct MANTIS-Bench, a multi-axis presentation-attack benchmark that pairs original news with rewritten views across writing style, emotional expression, and dissemination framing for robustness evaluation.

• We introduce MANTIS, an instance-wise worstcase training framework that combines multi-axis stress testing with adaptive robustness optimization, improving robustness without additional inference-time overhead.

• We define Union Attack Success Rate (UASR) to quantify vulnerability across presentation axes and reveal complementary, instance-specific failure patterns through experiments on MANTIS-Bench.

## 2 Related Work

Robust Fake News Detection. Early research on robust fake news detection has primarily focused on domain shifts, temporal drift, and spurious correlations. For instance, MDFEND (Nan et al., 2021) and M<sup>3</sup>FEND (Zhu et al., 2023) learn multi-view representations for domain heterogeneity, FTT (Hu et al., 2023) models temporal trends for future adaptation, and ENDEF (Zhu et al., 2022) uses causal debiasing to alleviate entity shortcuts.However, these methods have not explicitly addressed presentation-level shortcuts that can affect veracity predictions when claim-preserving textual changes are made.

To mitigate presentation-level vulnerabilities, recent studies have explored adversarial training and LLM-based augmentation. Recent approaches target style robustness through SheepDog (Wu et al., 2024) and AdStyle (Park et al., 2025), or sentimentrelated vulnerabilities through AdSent (Tahmasebi et al., 2026). Despite their effectiveness on individual axes, these defenses do not explicitly consider instance-specific worst-case variants within a unified multi-axis space, motivating MANTIS to address complementary presentation vulnerabilities.

LLMs for Fake News Detection. The development of LLMs has expanded fake news detection and veracity evaluation by leveraging LLMs as reasoning engines and explanation generators. FakeGPT (Huang and Sun, 2023) analyzes prediction consistency, ARG (Hu et al., 2024) incorporates LLM-generated rationales to assist taskspecific detectors, and FactAgent (Li et al., 2024b) conducts multi-step structured verification via external tools. Further studies integrate domain knowledge and reasoning traces into detection pipelines through heterogeneous news–entity–topic graphs (Ma et al., 2024), negative reasoning (Zhang et al., 2025), and persuasion-aware prompting strategies (Modzelewski et al., 2025).

However, LLM-assisted detection still faces challenges in presentation robustness and inference efficiency. First, LLM-based detectors remain sensitive to stylistic or emotional changes even when factual claims remain unchanged (Wu et al., 2024; Tahmasebi et al., 2026). Second, frameworks rely on multi-stage inference, including tool-assisted verification and persuasion analysis (Li et al., 2024b; Modzelewski et al., 2025), may increase inference costs. Our MANTIS confines multi-axis transformations and instance-wise worst-case optimization to offline training and performs single-pass inference without additional rewriting or tool calls.

![](images/c870607c7ea5032456961daac7d15d128a9f527bb2f27022a4b210ba40a15aba.jpg)  
Figure 2: Overview of MANTIS. An LLM rewriter generates six offline views covering writing style, emotional expression, and dissemination framing. MANTIS trains jointly on the original news and each instance’s worst-case view, while inference requires only a single forward pass.

## 3 Methodology

MANTIS combines multi-axis news transformations with instance-wise worst-case training. As illustrated in Figure 2, an LLM rewriter generates six offline views under claim-preservation constraints. During training, MANTIS selects the highest-loss view for each case and optimizes the detector using both original and selected views. At inference time, it performs a single forward pass without additional rewriting.

## 3.1 Preliminaries

Consider a fake news detection dataset $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i } \in \mathcal X$ denotes the news text in the i-th case and $y _ { i } \in \mathcal { Y } = \{ \mathrm { r e a l } , \mathrm { f a k e } \}$ is its veracity label. A text-based detector $f _ { \theta }$ assigns a probability $p _ { \theta } ( y \mid x _ { i } )$ to each candidate label and predicts

$$
\hat { y } _ { i } = \underset { y \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } ( y \mid x _ { i } ) .\tag{1}
$$

Let T denote a presentation transformation and let $c ( x )$ denote the core claim conveyed in x. A transformation is claim-preserving when

$$
c ( T ( x _ { i } ) ) = c ( x _ { i } ) .\tag{2}
$$

Accordingly, the transformed view $T ( x _ { i } )$ is assigned the same veracity label $y _ { i }$ as the original news.

## 3.2 MANTIS-Bench

MANTIS-Bench pairs original news with rewritten views generated under core-claim preservation constraints to evaluate detector robustness to presentation changes. Existing studies have examined publisher-conditioned writing style through style transformations and robust training (Wu et al., 2024; Park et al., 2025), sentiment and emotional expression through sentiment manipulation (Tahmasebi et al., 2026), and persuasive language or explicit calls to action in misinformation and socialmedia communication (Modzelewski et al., 2025; Rogers et al., 2019). However, these presentation factors have generally been formulated and evaluated separately. We bring them into a unified transformation space comprising writing style, emotional expression, and dissemination framing.

The writing-style dimension shifts news between mainstream-news style $T _ { S } ^ { m }$ and tabloid style $T _ { S } ^ { t }$ The emotional-expression dimension varies between neutral expression $T _ { E } ^ { n }$ and intensified expression $T _ { E } ^ { s }$ . The dissemination-framing dimension varies between public-information framing $T _ { D } ^ { - }$ and engagement-oriented framing $T _ { D } ^ { + }$ . We define the three transformation subsets and their union as

$$
\begin{array} { c c } { { \mathcal { T } _ { S } = \{ T _ { S } ^ { m } , T _ { S } ^ { t } \} , } } & { { \mathcal { T } _ { E } = \{ T _ { E } ^ { n } , T _ { E } ^ { s } \} , } } \\ { { \mathcal { T } _ { D } = \{ T _ { D } ^ { - } , T _ { D } ^ { + } \} , } } & { { \mathcal { T } = \mathcal { T } _ { S } \cup \mathcal { T } _ { E } \cup \mathcal { T } _ { D } . } } \end{array}\tag{3}
$$

To determine whether the three dimensions expose complementary vulnerabilities, we conduct a diagnostic study on the GossipCop test set using four LLM detectors; full experimental details are provided in Section 4. Fake news is transformed separately using $T _ { S } ^ { m } , T _ { E } ^ { n }$ , and $T _ { D } ^ { - }$ , whereas real news is transformed using the opposite directions $T _ { S } ^ { t } , T _ { E } ^ { s } ,$ , and $T _ { D } ^ { + }$ . This setting examines whether credible presentation changes can make detectors accept fake news, or whether sensational and engagement-oriented presentations can cause them to reject real news. These label-conditioned directions are used for diagnosis, whereas training considers all six directions for each case. Figure 3 shows that all four detectors experience Macro-F1 degradation under at least one transformation dimension, while the most damaging dimension varies across detectors.

![](images/5811c28250b3527e1b770ee26b3870df1b1ec2ed6825a87279d4ddf4a94eea9b.jpg)  
Figure 3: Macro-F1 under original and transformed views on GossipCop. Style, Emotion, and Dissemination denote the three transformation dimensions.

![](images/8c6940f7c8850c31829a67c4d7964383edaed8631c97e71f3b49dada6a092062.jpg)  
Figure 4: Case-specific vulnerability on GossipCop (Llama-3.1-8B-Instruct). Points denote originally correct cases misclassified under one transformation; axes show gold-label probabilities under the harmful and other transformations.

Aggregate performance does not reveal whether the three transformations affect the same cases. For Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Figure 4 focuses on originally correct cases misclassified under exactly one transformation. We identify 41 style-specific, 42 emotion-specific, and 82 dissemination-specific failures, demonstrating complementary failure patterns across dimensions and instance-dependent presentation vulnerabilities. These observations motivate selecting the most challenging transformed view separately for each training case rather than applying a single fixed transformation.

## 3.3 The MANTIS Framework

Offline View Construction. For each training case $( x _ { i } , y _ { i } )$ , MANTIS applies every transformation in $\tau$ to the news $x _ { i }$ and constructs the candidate-view set

$$
\mathcal { V } _ { i } = \{ T ( x _ { i } ) \mid T \in \mathcal { T } \} .\tag{4}
$$

Every view in $\nu _ { i }$ retains the label $y _ { i }$ . All six candidate views are generated offline and remain fixed throughout training.

Worst-Case View Training. Following adversarial training with label-preserving perturbations (Madry et al., 2018; Ivgi and Berant, 2021), MAN-TIS performs inner maximization over a finite set of transformed news views and selects the worstcase view separately for each case. For a news view u with label $y ,$ its classification loss is defined as $\ell _ { \theta } ( u , y ) = - \log p _ { \theta } ( y \mid u )$ , where $p _ { \theta } ( y \mid u )$ is the probability assigned by the detector to the gold label. MANTIS performs the inner maximization independently for each case by selecting

$$
v _ { i } ^ { \star } ( \theta ) = \arg \operatorname* { m a x } _ { v \in \mathcal { V } _ { i } } \ell _ { \theta } ( v , y _ { i } ) .\tag{5}
$$

Because $\nu _ { i }$ contains both directions of all three presentation dimensions, this operation determines the most challenging dimension and direction for the current case. The selected view depends on both the news $x _ { i }$ and the current detector parameters θ.

The original news and the selected worst-case view define two complementary training losses:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { o r i } } ( \theta ) = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { \theta } ( x _ { i } , y _ { i } ) , } \\ & { \mathcal { L } _ { \mathrm { a d v } } ( \theta ) = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { \theta } ( v _ { i } ^ { \star } ( \theta ) , y _ { i } ) . } \end{array}\tag{6}
$$

The original-view loss maintains detection performance on unmodified news, whereas the worstcase loss reduces sensitivity to the most harmful transformed view of each case. MANTIS combines them through the following objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { M A N T I S } } ( \theta ) = \lambda \mathcal { L } _ { \mathrm { o r i } } ( \theta ) + ( 1 - \lambda ) \mathcal { L } _ { \mathrm { a d v } } ( \theta ) , } \\ & { \quad \quad \quad \theta ^ { \star } = \arg \underset { \theta } { \operatorname* { m i n } } \mathcal { L } _ { \mathrm { M A N T I S } } ( \theta ) . } \end{array}\tag{7}
$$

Here, $\lambda \in ( 0 , 1 )$ controls the trade-off between original-view supervision and worst-case training. During each training step, MANTIS evaluates all six candidate views in $\nu _ { i }$ under the current detector without retaining gradients. It then selects $v _ { i } ^ { \star } ( \theta )$ and updates the detector using the joint objective. Since the candidate views are generated offline, optimization reduces to comparison among fixed candidates without invoking the rewriter.

Inference. At inference time, the learned detector $f _ { \theta ^ { \star } }$ directly classifies unseen news without constructing transformed views. Given a fixed classification prompt p and news $x ,$ the detector obtains $z _ { \mathrm { r e a l } } ( x )$ and $z _ { \mathrm { f a k e } } ( x )$ from $f _ { \theta ^ { \star } } ( p \oplus x )$ and predicts $\hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } z _ { y } ( x )$ , where ⊕ denotes concatenation. Inference therefore requires only a single detector forward pass.

## 4 Experimental Setup

## 4.1 Benchmark Construction and Quality Evaluation

Data Sources and Splits. We construct MANTIS-Bench from PolitiFact, GossipCop, and Weibo21, with dataset statistics summarized in Table 1. Politi-Fact and GossipCop are collected from FakeNews-Net (Shu et al., 2020), covering political and entertainment news, respectively. Following Sheep-Dog (Wu et al., 2024), we adopt balanced temporal splits for both datasets, where the most recent 20% of cases within each label constitute the test set. Weibo21 (Nan et al., 2021) is used exclusively for cross-lingual evaluation. We combine the official splits and construct a topic-balanced evaluation subset containing 1,350 cases from nine topics, with 75 real and 75 fake cases per topic.

View Generation. Transformations for the two English datasets, PolitiFact and GossipCop, are generated with GPT-5-mini (Singh et al., 2025), whereas transformations for the Chinese Weibo21 dataset are generated with DeepSeek-V4-Flash (Xu et al., 2026). Following the transformations defined in Section 3.2, each rewriting request combines a shared claim-preservation instruction, a targetspecific instruction, and the original news without providing its veracity label to the generator. Each rewritten view is paired with its source article and inherits the original label. Full prompts are provided in Appendix A. We examine factual-claim preservation through both human evaluation and LLM-as-a-Judge assessment.

Human Evaluation. We sampled 60 original– transformed pairs across three datasets: 30 English pairs from PolitiFact and GossipCop, and 30 Chinese pairs from Weibo21, with balanced real and fake cases across each transformation dimension. Two annotators independently evaluated all pairs on two criteria: claim preservation (retaining original factual claims) and target realization (exhibiting the intended presentation shift). Interannotator agreement reached 95.0% $( \kappa = 0 . 7 9 )$ for claim preservation and 93.3% $( \kappa = 0 . 7 6 )$ for target realization. Following consensus resolution of disagreements, 58 of 60 pairs (96.7%) preserved factual claims, 57 of 60 (95.0%) realized the target transformation, and 56 of 60 (93.3%) satisfied both criteria.

<table><tr><td></td><td></td><td colspan="2">Train</td><td colspan="2">Test</td></tr><tr><td>Dataset</td><td>Domain</td><td>Real</td><td>Fake</td><td></td><td>Real Fake</td></tr><tr><td>PolitiFact</td><td>Politics</td><td>180</td><td>180</td><td>45</td><td>45</td></tr><tr><td>GossipCop</td><td>Entertainment 3,166</td><td></td><td>3,166</td><td>792</td><td>792</td></tr><tr><td>Weibo21</td><td>Nine topics</td><td>一</td><td></td><td>675</td><td>675</td></tr></table>

Table 1: Dataset statistics. Weibo21 reports the topicbalanced evaluation subset and is not used for training.

LLM-as-a-Judge. Following the human evaluation, we present the same 60 cases to independent LLM judges and ask whether each transformed view preserves the factual claims of the corresponding original news. DeepSeek-V4-Flash evaluates the English cases generated by GPT-5-mini, while GPT-5-mini evaluates the Chinese cases generated by DeepSeek-V4-Flash. With the human annotations withheld throughout the judging process, we quantify agreement between the resulting LLM decisions and the corresponding human labels using Cohen’s Kappa. The overall score of 0.79 indicates substantial agreement after accounting for chance.

## 4.2 Baselines

We compare MANTIS against baselines from three detector families:

• Supervised encoders. This group includes BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), and DeBERTa (He et al., 2020). Each model is fine-tuned on the original English training cases and serves as a standard supervised baseline without explicit robustness training.

• Zero-shot LLMs. This group includes Qwen2.5- 7B-Instruct (Qwen et al., 2025), Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Qwen3-8B (Yang et al., 2025), GLM-4-9B (Glm et al., 2024), Qwen2.5-72B-Instruct (Qwen et al., 2025), Llama-3.1-70B-Instruct (Grattafiori et al., 2024), and DeepSeek-V4.1-Flash (Xu et al.,

![](images/4e4371bac983def1835b2f9d2d5d2941d9e30fedbe50b2ba85ea088997ab6d15.jpg)  
Figure 5: Human evaluation of factual-claim preservation across style, emotion, and dissemination transformations.

2026). All models receive the same binary classification prompt and directly predict whether the input news is real or fake without task-specific fine-tuning. These baselines assess the robustness of general-purpose LLMs to presentation transformations.

• Robust fake news detectors. This group includes SheepDog (Wu et al., 2024) and AdSent (Tahmasebi et al., 2026). SheepDog targets adversarial writing-style variations, whereas AdSent reduces reliance on sentiment-related cues. We reproduce both methods using their public implementations and evaluate them on the same original and transformed views as MANTIS. Their performance on the respective adversarial test sets introduced in the original studies is reported separately to assess generalization beyond our transformation settings.

## 4.3 Evaluation Metrics

We report Accuracy and Macro-F1 on the original news and the transformed views corresponding to writing style, emotional expression, and dissemination framing. Accuracy measures the proportion of correctly classified cases, whereas Macro-F1 computes the F1 score for each veracity class and assigns equal importance to the real and fake classes.

We define Union Attack Success Rate (UASR) to quantify vulnerability across the three presentation transformations. Let O denote the original view, let $\boldsymbol { \mathcal { A } } = \{ S , E , D \}$ index the style, emotion, and dissemination evaluation views, and let $\hat { y } _ { i } ^ { v }$ denote the prediction for case i under view v. UASR is defined as the empirical conditional probability

$$
\mathrm { U A S R } = \widehat { \mathrm { P r } } \left( \exists a \in \mathcal { A } : \hat { y } _ { i } ^ { a } \neq y _ { i } \ | \ \hat { y } _ { i } ^ { O } = y _ { i } \right) .\tag{8}
$$

The condition $\hat { y } _ { i } ^ { O } = y _ { i }$ restricts the evaluation to cases that are correctly classified in their original form, while the event $\exists a \in { \mathcal { A } } : { \hat { y } } _ { i } ^ { a } \neq y _ { i }$ indicates that at least one presentation transformation overturns the correct prediction. UASR therefore isolates transformation-induced failures from errors already present on the original news. Accuracy and Macro-F1 evaluate performance separately for each view, whereas UASR captures vulnerability across the union of the three transformed views. Higher Accuracy and Macro-F1 and lower UASR indicate better performance.

## 4.4 Implementation Details

All experiments are implemented with PyTorch 2.4.1. The supervised models are trained for five epochs using AdamW with a learning rate of $2 \times 1 0 ^ { - 5 }$ and a batch size of 4; their results are averaged over ten random seeds. Zero-shot LLM inference is performed in BF16 using the same classification prompt and label space for all models. MANTIS is built on Llama-3.1-8B-Instruct and trained with LoRA using rank 32 and $\alpha = 6 4$ We use AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ For GossipCop, MANTIS is trained for two epochs with $\lambda = 0 . 7 1 4$ . For PolitiFact, we use the checkpoint from the third epoch with $\lambda = 0 . 7 5$ . All MANTIS results are averaged over three independent runs with different random seeds.

## 5 Experimental Results

## 5.1 Main Results

Table 2 compares MANTIS with supervised detectors, zero-shot LLMs, and robustly trained detectors on PolitiFact and GossipCop. All baselines exhibit vulnerability under at least one transformation, with zero-shot LLMs showing larger degradation. Moreover, the most damaging dimension varies across detectors, indicating that robustness to one presentation change does not necessarily generalize to others.

On PolitiFact, MANTIS achieves the highest Accuracy and Macro-F1 across all four views, reducing UASR from 8.64% to 7.51%. It obtains notable Macro-F1 gains on the original and dissemination views (3.59 and 3.49 percentage points, respectively). On GossipCop, MANTIS achieves the highest Macro-F1 across all views, while Ad-Sent retains slightly higher original-view Accuracy. It further reduces UASR from 8.61% to 5.06% and improves Macro-F1 under style, emotion, and dissemination transformations by 1.58, 2.08, and 1.52 points, respectively.

<table><tr><td></td><td colspan="2">Original</td><td colspan="2">Style</td><td colspan="2">Emotion</td><td colspan="2">Dissemination</td><td></td></tr><tr><td>Model</td><td>Acc. ↑</td><td>F1↑</td><td>Acc. ↑</td><td>F1↑</td><td>Acc. ↑</td><td>F1↑</td><td>Acc. ↑</td><td>F1↑</td><td>UASR↓</td></tr><tr><td colspan="10">Dataset: : PolitiFact</td></tr><tr><td colspan="10">Supervised Detectors</td></tr><tr><td>BERT</td><td>87.00</td><td>86.60</td><td>81.78</td><td>81.39</td><td>85.67</td><td>85.50</td><td>84.44</td><td>84.20</td><td>15.04</td></tr><tr><td>RoBERTa</td><td>89.89</td><td>89.79</td><td>81.78</td><td>81.52</td><td>90.00</td><td>89.86</td><td>81.89</td><td>80.77</td><td>17.09</td></tr><tr><td>DeBERTa</td><td>88.89</td><td>88.86</td><td>85.22</td><td>85.16</td><td>90.11</td><td>90.08</td><td>81.11</td><td>80.57</td><td>18.63</td></tr><tr><td colspan="10">Zero-Shot LLMs</td></tr><tr><td>Qwen2.5-7B</td><td>66.67</td><td>63.63</td><td>60.00</td><td>53.34</td><td>61.11</td><td>56.61</td><td>56.67</td><td>48.95</td><td>21.67</td></tr><tr><td>Llama-3.1-8B</td><td>72.22</td><td>70.93</td><td>55.56</td><td>52.15</td><td>60.00</td><td>56.36</td><td>54.44</td><td>53.98</td><td>46.15</td></tr><tr><td>Qwen3-8B</td><td>76.67</td><td>76.17</td><td>58.89</td><td>56.52</td><td>62.22</td><td>60.65</td><td>54.44</td><td>51.26</td><td>37.68</td></tr><tr><td>GLM-4-9B</td><td>68.89</td><td>67.59</td><td>55.56</td><td>48.16</td><td>57.78</td><td>52.50</td><td>55.56</td><td>51.51</td><td>38.71</td></tr><tr><td>Qwen2.5-72B</td><td>80.00</td><td>79.50</td><td>63.33</td><td>59.09</td><td>68.89</td><td>66.51</td><td>58.89</td><td>53.35</td><td>27.78</td></tr><tr><td>Llama-3.1-70B</td><td>74.44</td><td>72.66</td><td>61.11</td><td>54.18</td><td>64.44</td><td>60.00</td><td>61.11</td><td>54.18</td><td>20.90</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>81.11</td><td>80.41</td><td>65.56</td><td>60.92</td><td>63.33</td><td>58.40</td><td>62.22</td><td>56.76</td><td>24.66</td></tr><tr><td colspan="10">Robustly Trained Detectors</td></tr><tr><td>SheepDog</td><td>87.78</td><td>87.70</td><td>80.00</td><td>79.84</td><td>85.56</td><td>85.41</td><td>76.67</td><td>75.80</td><td>16.46</td></tr><tr><td>AdSent</td><td>90.00</td><td>89.94</td><td>90.00</td><td>89.90</td><td>92.22</td><td>92.20</td><td>86.67</td><td>86.50</td><td>8.64</td></tr><tr><td>MANTIS</td><td>93.71</td><td>93.53</td><td>90.74</td><td>90.73</td><td>92.60</td><td>92.58</td><td>90.00</td><td>89.99</td><td>7.51</td></tr><tr><td colspan="10">Dataset: GossipCop</td></tr><tr><td colspan="10">Supervised Detectors</td></tr><tr><td>BERT</td><td>74.84</td><td>74.81</td><td>72.89</td><td>72.85</td><td>72.64</td><td>72.60</td><td>74.43</td><td>74.36</td><td>17.79</td></tr><tr><td>RoBERTa</td><td>74.33</td><td>74.26</td><td>73.17</td><td>73.11</td><td>73.33</td><td>73.27</td><td>75.10</td><td>75.06</td><td>15.65</td></tr><tr><td>DeBERTa</td><td>69.41</td><td>66.00</td><td>66.16</td><td>62.68</td><td>67.56</td><td>64.12</td><td>67.18</td><td>63.66</td><td>15.21</td></tr><tr><td colspan="10">Zero-Shot LLMs</td></tr><tr><td>Qwen2.5-7B</td><td>54.36</td><td>43.63</td><td>51.64</td><td>37.76</td><td>52.59</td><td>40.28</td><td>51.07</td><td>36.49</td><td>10.34</td></tr><tr><td>Llama-3.1-8B</td><td>55.87</td><td>52.62</td><td>54.23</td><td>47.28</td><td>51.20</td><td>46.38</td><td>51.77</td><td>49.29</td><td>34.92</td></tr><tr><td>Qwen3-8B</td><td>57.26</td><td>51.80</td><td>53.79</td><td>45.55</td><td>54.04</td><td>47.31</td><td>51.01</td><td>41.94</td><td>23.37</td></tr><tr><td>GLM-4-9B</td><td>56.06</td><td>47.03</td><td>51.96</td><td>38.50</td><td>51.89</td><td>40.79</td><td>50.06</td><td>36.83</td><td>17.23</td></tr><tr><td>Qwen2.5-72B</td><td>60.61</td><td>54.93</td><td>55.49</td><td>46.51</td><td>55.74</td><td>48.42</td><td>53.85</td><td>44.12</td><td>17.40</td></tr><tr><td>Llama-3.1-70B</td><td>57.20</td><td>48.85</td><td>55.49</td><td>46.04</td><td>55.68</td><td>46.44</td><td>55.11</td><td>46.06</td><td>11.04</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>60.23</td><td>53.46</td><td>58.27</td><td>50.16</td><td>58.90</td><td>51.32</td><td>57.77</td><td>49.69</td><td>9.64</td></tr><tr><td colspan="10">Robustly Trained Detectors</td></tr><tr><td>SheepDog</td><td>76.26</td><td>76.25</td><td>75.82</td><td>75.81</td><td>76.20</td><td>76.19</td><td>77.02</td><td>77.00</td><td>8.61</td></tr><tr><td>AdSent</td><td>79.41</td><td>78.33</td><td>73.80</td><td>73.51</td><td>74.81</td><td>74.61</td><td>73.86</td><td>73.68 78.52</td><td>12.87</td></tr><tr><td>MANTIS</td><td>78.62</td><td>78.59</td><td>77.40</td><td>77.39</td><td>78.28</td><td>78.27</td><td>78.54</td><td></td><td>5.06</td></tr></table>

Table 2: Main results on the PolitiFact and GossipCop subsets of MANTIS-Bench. All values are percentages; F denotes Macro-F1. Per dataset and column, the best and second-best results are bold and underlined, respectively.

## 5.2 Ablation Study

Table 3 evaluates the contributions of the originalview loss $\mathcal { L } _ { \mathrm { o r i } }$ , adversarial-view loss $\mathcal { L } _ { \mathrm { a d v } }$ , and worst-case view selection. The w/o worst-case variant replaces worst-case selection with random adversarial-view selection while retaining both loss terms. Removing $\mathcal { L } _ { \mathrm { o r i } }$ leads to a clear decline in original-view performance on both datasets, indicating that supervision from the original news remains important during robust training. Removing $\mathcal { L } _ { \mathrm { a d v } }$ results in the highest UASR on both PolitiFact and GossipCop and weakens performance across transformed views. Replacing worst-case selection with random selection reduces Macro-F1 across views and increases UASR from 7.51% to 9.83% on PolitiFact and from 5.06% to 8.95% on Gossip-Cop, supporting explicit selection of challenging views.

The complete objective achieves the best result in every column, indicating that original-view supervision, adversarial-view supervision, and worstcase selection jointly contribute to detection performance and robustness.

## 5.3 Generalization to SheepDog Style Attacks

We further evaluate whether MANTIS generalizes to the four publisher-style attack sets introduced by SheepDog (Wu et al., 2024). These attacks reverse the association between publisher style and veracity by presenting real news with unreliable styles and fake news with reliable styles. Table 4 reports results on the PolitiFact test set.

MANTIS consistently outperforms SheepDog across all four attack sets, demonstrating transfer to unseen publisher-style changes. Compared with AdSent, MANTIS performs better on Sets C and D, while AdSent remains stronger on Sets A and B. These results indicate that MANTIS generalizes beyond its training transformations, although the relative advantage varies across publisher combinations.

<table><tr><td></td><td colspan="3">Macro-F1 ↑</td><td></td></tr><tr><td>Variant</td><td>Orig.</td><td>Style Emo.</td><td>Dis.</td><td>UASR↓</td></tr><tr><td colspan="5">PolitiFact</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { o r i } }$ </td><td>89.27</td><td>89.98</td><td>89.25 88.83</td><td>10.39</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { a d v } }$ </td><td>93.01</td><td>88.66</td><td>91.84</td><td>89.10</td></tr><tr><td>w/o worst-case</td><td>92.55</td><td>88.42</td><td>89.23 87.62</td><td>11.84 9.83</td></tr><tr><td>MANTIS</td><td>93.53</td><td>90.73</td><td>92.58 89.99</td><td>7.51</td></tr><tr><td colspan="5">GossipCop</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { o r i } }$ </td><td>75.20</td><td>76.95</td><td>76.57 75.29</td><td>8.73</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { a d v } }$ </td><td>77.73</td><td>75.31 76.17</td><td>75.41</td><td>9.13</td></tr><tr><td>w/o worst-case</td><td>76.92</td><td>76.31 76.25</td><td>75.78</td><td>8.95</td></tr><tr><td>MANTIS</td><td>78.59</td><td>77.39 78.27</td><td>78.52</td><td>5.06</td></tr></table>

Table 3: Ablation results for the MANTIS objective. All values are percentages; bold indicates the best result.
<table><tr><td colspan="3">SheepDog</td><td colspan="2">AdSent</td><td colspan="2">MANTIS</td></tr><tr><td>Test Set</td><td> $\mathbf { A c c . }$ </td><td> $F _ { 1 }$ </td><td>Acc.</td><td> $F _ { 1 }$ </td><td> $\mathbf { A c c . }$ </td><td> $F _ { 1 }$ </td></tr><tr><td>Original</td><td>90.00</td><td>89.92</td><td>90.00</td><td>89.90</td><td>93.71</td><td>93.53</td></tr><tr><td>Adv.-A</td><td>81.11</td><td>80.99</td><td>85.56</td><td>85.47</td><td>82.20</td><td>81.95</td></tr><tr><td>Adv.-B</td><td>80.00</td><td>79.89</td><td>84.44</td><td>84.38</td><td>83.34</td><td>82.95</td></tr><tr><td>Adv.-C</td><td>82.44</td><td>82.36</td><td>87.78</td><td>87.74</td><td>88.85</td><td>88.63</td></tr><tr><td>Adv.-D</td><td>81.31</td><td>81.24</td><td>86.67</td><td>86.64</td><td>87.76</td><td>87.71</td></tr></table>

Table 4: Results (%) on the original PolitiFact test set and SheepDog publisher-style attacks. SheepDog and AdSent scores are from AdSent (Tahmasebi et al., 2026).

## 5.4 Cross-Lingual Generalization on Weibo21

We evaluate cross-lingual generalization by training MANTIS and AdSent on English GossipCop and directly applying them to Chinese Weibo21 without further training. Table 5 reports performance, robustness, and inference efficiency.

<table><tr><td></td><td colspan="2">AdSent</td><td colspan="2">MANTIS</td></tr><tr><td>View</td><td>Acc. ↑</td><td>Macro-F1 ↑</td><td>Acc. ↑</td><td>Macro-F1 ↑</td></tr><tr><td>Original</td><td>58.67</td><td>57.98</td><td>61.26</td><td>61.14</td></tr><tr><td>Style</td><td>59.56</td><td>59.01</td><td>60.07</td><td>60.05</td></tr><tr><td>Emotion</td><td>55.85</td><td>55.14</td><td>57.56</td><td>57.51</td></tr><tr><td>Dissem.</td><td>59.70</td><td>59.16</td><td>61.11</td><td>61.10</td></tr></table>

<table><tr><td>System</td><td>UASR (%) ↓</td><td></td><td>GPU-h ↓ Deployment (GiB) ↓</td></tr><tr><td>AdSent</td><td>29.42</td><td>14.034</td><td>29.94</td></tr><tr><td>MANTIS</td><td>18.80</td><td>0.167</td><td>15.29</td></tr></table>

Table 5: Cross-lingual performance and efficiency on Weibo21 after training on English GossipCop.

MANTIS achieves higher Accuracy and Macro-F1 than AdSent across the original and transformed views, while reducing UASR from 29.42% to 18.80%. It also lowers inference cost from 14.034 to 0.167 GPU-hours and deployment footprint from 29.94 to 15.29 GiB. These results demonstrate more effective transfer to Chinese news, although simultaneous shifts in language, domain, and topic remain challenging.

![](images/05dabbd4a5b756bcde7faef424a3c77c75cdb62c1ede35042c9bbcee9e0ff001.jpg)  
Figure 6: Topic-level Accuracy (a) and Macro-F1 (b) on Weibo21, averaged over four zero-shot LLMs across original and transformed views.

To further investigate presentation robustness across different news topics, we analyze Weibo21 using four zero-shot LLMs. Figure 6 presents topiclevel Accuracy and Macro-F1 averaged across the original and transformed views. Across nine topics, presentation transformations consistently reduce performance compared with original news, with dissemination framing generally producing the largest degradation. Topic difficulty also varies across categories, indicating that presentation vulnerability depends not only on the transformation type but also on the underlying news content. Detailed topic-level statistics and significance tests are provided in Appendix B.3.

## 6 Conclusion

This paper introduces MANTIS-Bench and MAN-TIS for robust fake news detection under claimpreserving presentation changes. MANTIS-Bench pairs original news with complementary views along writing style, emotional expression, and dissemination framing, while MANTIS adapts robust training to each case by optimizing the original view together with its most challenging transformed view. Experiments on PolitiFact and GossipCop demonstrate strong performance across both original and transformed news, together with consistently lower UASR. Evaluations on the SheepDog publisher-style attacks further demonstrate transfer to unseen style transformations, while the Weibo21 results show improved, although still challenging, cross-lingual generalization. Overall, our findings demonstrate the importance of multi-axis evaluation and case-specific robust training for reliable fake news detection.

## Limitations

MANTIS considers three presentation dimensions and operates over a finite set of six offline LLMgenerated views. Although these views cover complementary changes in writing style, emotional expression, and dissemination framing, they cannot represent every presentation variation encountered in practice. Moreover, the human evaluation covers only a limited subset of the generated views, and occasional generation artifacts may remain despite the supporting human and LLM-based quality assessments. Our evaluation is also restricted to three benchmark datasets, and the moderate absolute performance on Weibo21 indicates that simultaneous shifts in language, domain, and topic remain challenging. Future work could extend MANTIS to broader languages and domains, additional presentation dimensions, multimodal news, and dynamically constructed transformation neighborhoods.

## References

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673.

Team Glm, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Josh A Goldstein, Girish Sastry, Micah Musser, Renee DiResta, Matthew Gentzel, and Katerina Sedova. 2023. Generative language models and automated influence operations: Emerging threats and potential mitigations. arXiv preprint arXiv:2301.04246.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022. A survey on automated fact-checking. Transactions of the association for computational linguistics, 10:178–206.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654.

Beizhe Hu, Qiang Sheng, Juan Cao, Yuhui Shi, Yang Li, Danding Wang, and Peng Qi. 2024. Bad actor, good advisor: Exploring the role of large language models in fake news detection. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pages 22105–22113.

Beizhe Hu, Qiang Sheng, Juan Cao, Yongchun Zhu, Danding Wang, Zhengjia Wang, and Zhiwei Jin. 2023. Learn over past, evolve for future: Forecasting temporal trends for fake news detection. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pages 116–125.

Yue Huang and Lichao Sun. 2023. Fakegpt: fake news generation, explanation and detection of large language models. arXiv preprint arXiv:2310.05046.

Maor Ivgi and Jonathan Berant. 2021. Achieving model robustness through discrete adversarial training. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1529–1544.

David MJ Lazer, Matthew A Baum, Yochai Benkler, Adam J Berinsky, Kelly M Greenhill, Filippo Menczer, Miriam J Metzger, Brendan Nyhan, Gordon Pennycook, David Rothschild, et al. 2018. The science of fake news. Science, 359(6380):1094–1096.

Ruosen Li, Zimu Wang, Son Quoc Tran, Lei Xia, and Xinya Du. 2024a. Meqa: A benchmark for multihop event-centric question answering with explanations. In Advances in Neural Information Processing Systems, volume 37, pages 126835–126862. Curran Associates, Inc.

Xinyi Li, Yongfeng Zhang, and Edward C Malthouse. 2024b. Large language model agent for fake news detection. arXiv preprint arXiv:2405.01593.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Xiaoxiao Ma, Yuchen Zhang, Kaize Ding, Jian Yang, Jia Wu, and Hao Fan. 2024. On fake news detection with llm enhanced semantics mining. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 508–521.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations.

Arkadiusz Modzelewski, Witold Sosnowski, Tiziano Labruna, Adam Wierzbicki, and Giovanni Da San Martino. 2025. Pcot: Persuasion-augmented chain of thought for detecting fake news and social media disinformation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24959–24983.

Qiong Nan, Juan Cao, Yongchun Zhu, Yanyan Wang, and Jintao Li. 2021. Mdfend: Multi-domain fake news detection. In Proceedings ofthe 30th ACM international conference on information & knowledge management, pages 3343–3347.

Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, and Meeyoung Cha. 2025. Adversarial style augmentation via large language model for robust fake news detection. In Proceedings ofthe ACM on Web Conference 2025, pages 4024–4033.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Rolf Reber and Norbert Schwarz. 1999. Effects of perceptual fluency on judgments of truth. Consciousness and cognition, 8(3):338–342.

Anna Rogers, Olga Kovaleva, and Anna Rumshisky. 2019. Calls to action on social media: Detection, social impact, and censorship potential. In Proceedings of the Second Workshop on Natural Language Processingfor Internet Freedom: Censorship, Disinformation, and Propaganda, pages 36–44.

Kai Shu, Deepak Mahudeswaran, Suhang Wang, Dongwon Lee, and Huan Liu. 2020. Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data, 8(3):171–188.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Sahar Tahmasebi, Eric Müller-Budack, and Ralph Ewerth. 2026. Robust fake news detection using large language models under adversarial sentiment attacks. In Proceedings of the ACM Web Conference 2026, pages 1717–1726.

Florian Tramèr and Dan Boneh. 2019. Adversarial training and robustness for multiple perturbations. In

Advances in Neural Information Processing Systems (NeurIPS).

Zimu Wang and Hong-Seng Gan. 2023. Multi-level adversarial training for stock sentiment prediction. In 2023 IEEE 3rd International Conference on Computer Communication and Artificial Intelligence (CCAI), pages 127–134.

Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, and Anh Nguyen. 2024. Generating valid and natural adversarial examples with large language models. In 2024 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD), pages 1716–1721.

Jiaying Wu, Jiafeng Guo, and Bryan Hooi. 2024. Fake news in sheep’s clothing: Robust fake news detection against llm-empowered style attacks. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 3367–3378.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Chaowei Zhang, Zongling Feng, Zewei Zhang, Jipeng Qiang, Guandong Xu, and Yun Li. 2025. Is llms hallucination usable? llm-based negative reasoning for fake news detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 1031–1039.

Yongchun Zhu, Qiang Sheng, Juan Cao, Shuokai Li, Danding Wang, and Fuzhen Zhuang. 2022. Generalizing to the future: Mitigating entity bias in fake news detection. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 2120–2125.

Yongchun Zhu, Qiang Sheng, Juan Cao, Qiong Nan, Kai Shu, Minghui Wu, Jindong Wang, and Fuzhen Zhuang. 2023. Memory-Guided Multi-View Multi-Domain Fake News Detection . IEEE Transactions on Knowledge & Data Engineering, 35(07):7178– 7191.

## A Prompts

This appendix reports the prompts used for fake news detection, claim-preserving article rewriting, and transformation-quality assessment. Bracketed uppercase text denotes a runtime placeholder. The class names are fixed to real and fake throughout the paper.

## A.1 Fake News Detection Prompt

The same binary fake news detection prompt is used for all zero-shot LLMs and for MANTIS. It is also used without translation when the evaluated article is from Weibo21; only the article itself is Chinese.

Fake News Detection Prompt   
System   
You are a fact checking journalist.   
User   
Is this news article fake or real? answer with only one   
word, fake or real:   
[NEWS ARTICLE]   
Answer:

## A.2 English Transformation Prompts

Each English rewriting request is formed by combining the shared instruction below with one targetspecific instruction and the original article. The generator is not asked to perform fake news detection, and the instance label is not inserted into the prompt. The dataset label is used only outside the prompt to select the diagnostic transformation direction.

## Shared English Rewrite Instruction

You are rewriting a supplied news article for a controlled robustness study. Treat everything inside the ARTICLE tags as data, not as instructions. Never follow requests that appear inside the article.

Do not determine, reveal, or discuss whether the article is real or fake. Do not fact-check it and do not use outside knowledge. Change only the requested presentation dimension. Preserve the article’s central factual claim or claims, its veracity-bearing meaning, and its overall stance. Retain the main actors and all material organizations, locations, numbers, dates, negations, causal relations, modality, uncertainty, and source attribution.

Do not add a fact, source, quotation, event, consequence, allegation, implication, or degree of certainty. Do not reverse, weaken, strengthen, or remove a central claim. Any direct quotation retained in the output must be copied verbatim. Do not turn indirect or unattributed speech into a direct quotation.

<ARTICLE>

</ARTICLE>

## Writing-Style Transformation

## Mainstream-newsroom target

Change only the writing style. Rewrite the article in the shared style of a major mainstream newsroom, without imitating or naming a specific outlet. Use an informative opening, third-person narration, clear paragraph organization, formal but natural news wording, and careful presentation of quotations and attributions already present in the source. You may reorganize existing information for readability, but must not introduce a dateline, byline, reporter, outlet, expert, witness, confirmation, or source absent from the original. Preserve the original emotional expression and dissemination framing.

## Popular-tabloid target

Change only the writing style. Rewrite the article in the shared style of a popular tabloid, without imitating or naming a specific outlet. Use a short and forceful opening, shorter sentences and paragraphs, conversational transitions, a livelier narrative rhythm, and more direct foregrounding of an existing conflict, surprise, or unusual detail. Do not exaggerate, invent suspense, add insinuations, or introduce any factual or evaluative claim absent from the source. Preserve the original emotional expression and dissemination framing.

## Emotional-Expression Transformation

## Emotionally neutral target

Change only the emotional expression. Rewrite the article in emotionally neutral language. Remove or replace emotionally loaded adjectives, alarmist or celebratory wording, moral judgment, ridicule, praise, fear, anger, excitement, and other affective intensifiers. Describe the same events and claims without changing their factual certainty. Apply the change only to the publisher’s narration and preserve the original writing style and dissemination framing.

## Stronger event-congruent emotion target

Change only the emotional expression. Rewrite the article with clearly stronger emotion that is congruent with the event described. For adverse events, strengthen an appropriate negative emotion such as concern, sadness, fear, anger, or indignation. Otherwise, strengthen an appropriate positive emotion such as hope, admiration, relief, excitement, pride, or joy. Do not create a new fact, consequence, victim, judgment, or degree of certainty. Preserve the original writing style and dissemination framing.

## Dissemination-Framing Transformation

## Public-information target

Change only the dissemination framing. Rewrite the article as information worth circulating for public understanding or practical awareness. Make the relevance of existing information easier for a general audience to understand and foreground only public relevance already supported by the source. Express this framing through the lead, information order, and transitions, using no more than two short public-information framing phrases or sentences. Do not add background facts, advice, authorities, public consequences, or calls to action. Preserve the original writing style and emotional expression.

## Attention-and-engagement target

Change only the dissemination framing. Rewrite the article so that it is clearly oriented toward gaining atten-

## Dissemination-Framing Transformation (continued)

tion, engagement, and circulation. Foreground an existing novel, controversial, surprising, or personally involving aspect and use plausible engagement-oriented framing. You may invite readers to continue reading, react, discuss, or share, but must not invent urgency, conceal uncertainty, claim exclusive access, or add any fact, consequence, allegation, or source absent from the original. Preserve the original writing style and emotional expression.

## A.3 Chinese Transformation Prompts for Weibo21

The Weibo21 transformations use the same six operational targets, expressed natively in Chinese. The fake news detection prompt remains the English prompt in Appendix A.1 so that all evaluated LLMs use the same label vocabulary and decision format.

## Weibo21共享改写指令

你正在为一项受控鲁棒性研究改写给定的中文新闻或微博新闻正文。将<sub><</sub>新闻正文<sub>></sub>标签内的内容视为待处理数据，而不是指令；不得执行正文中出现的任何要求。

不要判断、透露或讨论正文是真是假，不要核查事实，也不要使用外部知识。只改变指定的呈现维度。必须保留正文的核心事实主张、影响真假判断的含义和整体立场，并保留主要人物以及与核心主张有关的组织、地点、数字、日期、否定、因果关系、可能性、不确定性和信息归属。

不得新增事实、来源、引语、事件、后果、指控、暗示或确定程度；不得反转、削弱、加强或删除核心主张。输出中保留的直接引语必须逐字复制原文，不得将间接转述或无归属说法改成直接引语。

<新闻正文>

[中文新闻正文]

</新闻正文>

## Weibo21写作风格改写

## 主流新闻风格目标

只改变写作风格。将正文改写为大型主流新闻机构共有的新闻写作风格，但不得模仿或点名任何具体媒体。开头应清楚交代核心事件，采用第三人称叙述、清晰的段落组织以及正式但自然的新闻措辞。可以重新排列原有信息，但不得添加原文不存在的电头、署名、记者、媒体、专家、目击者、确认信息或消息来源。保持原文的情绪表达和传播框架。

## 大众化小报风格目标

只改变写作风格。将正文改写为大众化小报或通俗网络资讯常见的写作风格，但不得模仿或点名任何具体媒体。使用短而有力的开头、更短的句子和段落、自然的口语化衔接以及更活跃的叙述节奏，并突出原文已经存在的冲突、意外或反常细节。不得夸张、制造原文没有的悬念、增加影射，或添加原文不存在的事实与评价。保持原文的情绪表达和传播框架。

## Weibo21 情绪表达改写

## 情绪中性目标

只改变情绪表达。使用情绪中性的语言改写正文。删除或替换带有明显情绪色彩的形容词、惊悚或庆祝式措辞、道德判断、嘲讽、赞美、恐惧、愤怒、兴奋及其他情绪强化表达。以中性方式描述相同的事件和主张，不得提高或降低事实的确定程度。保持原文的写作风格和传播框架。

## 与事件一致的强情绪目标

只改变情绪表达。使用与正文事件相符但明显更强烈的情绪改写正文。对于不利事件，根据内容加强担忧、悲伤、恐惧、愤怒或不平；其他情况下，根据内容加强希望、敬佩、宽慰、兴奋、自豪或喜悦。不得制造新的事实、后果、受害者、判断或确定程度。保持原文的写作风格和传播框架。

## Weibo21 传播框架改写

## 公共信息目标

只改变传播框架。将正文呈现为有助于公众理解或实际知情、因而值得传播的信息。让普通受众更容易理解原文已有信息的相关性，并且只突出原 文能够支持的公共相关性。通过导语、信息顺序和正文衔接体现这种框架，整篇改写最多使用两处简短的公共信息框架表达。不得添加原文没有的背景事实、建议、主管部门、公共后果或行动号召。保持原文的写作风格和情绪表达。

## 注意力与互动目标

只改变传播框架。将正文改写为明显以获得注意、互动和传播为导向的呈现方式。突出原文已经存在的新奇、争议、意外或容易引起个人兴趣的方面，并采用合理的互动导向框架。可以邀请读者继续阅读、表达看法、讨论或分享，但不得虚构紧迫性、掩盖原文的不确定性、声称拥有独家信息，或添加原文不存在的事实、后果、指控和来源。保持原文的写作风格和情绪表达。

## A.4 LLM-as-a-Judge Prompt

The judge returns a binary factual-preservation label rather than a fake news detection decision. Cohen’s κ is computed against the human labels; it is not generated by the LLM itself.

## Factual-Claim Preservation Judge Prompt

Do the original article and rewritten article present the same factual information and core factual claim, regardless of changes in writing style, emotional expression, or dissemination framing? Answer with only one word: yes or no.

Article A (original): [ORIGINAL ARTICLE]

Article B (rewritten): [REWRITTEN ARTICLE]

Answer:

## B Additional Weibo21 Details

## B.1 Dataset Composition and Topic-Balanced Sampling

The official Weibo21 release (Nan et al., 2021) contains 5,751 training, 1,918 validation, and 1,923 test cases, comprising 9,592 cases in total. It provides nine named topics together with an additional undetermined category. Table 6 reports the distribution of these categories across the three official splits.

![](images/ed74a7e5d7f4268449f69c8a65bd1155f66cc2a65db88e5d71b74d14f209967c.jpg)  
Figure 7: Word clouds for the nine named Weibo21 topics used in our evaluation. Larger terms occur more frequently within the corresponding topic.

<table><tr><td>Topic</td><td>Train</td><td>Val.</td><td>Test</td><td>Total</td><td>Selected</td></tr><tr><td>Military</td><td>205</td><td>69</td><td>69</td><td>343</td><td>150</td></tr><tr><td>Health</td><td>600</td><td>200</td><td>200</td><td>1,000</td><td>150</td></tr><tr><td>Politics</td><td>511</td><td>170</td><td>171</td><td>852</td><td>150</td></tr><tr><td>Education</td><td>294</td><td>98</td><td>99</td><td>491</td><td>150</td></tr><tr><td>Culture/Ent.</td><td>864</td><td>288</td><td>288</td><td>1,440</td><td>150</td></tr><tr><td>Disasters</td><td>465</td><td>155</td><td>156</td><td>776</td><td>150</td></tr><tr><td>Society</td><td>1,601</td><td>534</td><td>534</td><td>2,669</td><td>150</td></tr><tr><td>Technology</td><td>141</td><td>47</td><td>48</td><td>236</td><td>150</td></tr><tr><td>Finance</td><td>792</td><td>264</td><td>265</td><td>1,321</td><td>150</td></tr><tr><td>Undetermined</td><td>278</td><td>93</td><td>93</td><td>464</td><td>一</td></tr><tr><td>Total</td><td>5,751</td><td>1,918</td><td>1,923</td><td>9,592</td><td>1,350</td></tr></table>

Table 6: Topic distribution of the original Weibo21 dataset and the topic-balanced subset used in our evaluation. Val. denotes the official validation split.

Because Weibo21 is used exclusively for evaluation, we pool its three official splits before constructing the topic-balanced subset. We exclude the undetermined category, remove duplicated news and duplicates with conflicting topic annotations. Using a fixed sampling seed of 2027, we select 75 real and 75 fake cases from each of the nine named topics. The resulting evaluation set contains 150 cases per topic and 1,350 cases in total. The same cases are used to construct the style, emotion, and dissemination views for all evaluated detectors.

<table><tr><td colspan="5">Panel A: Accuracy (%)</td></tr><tr><td>Model</td><td>Orig.</td><td>Style</td><td>Emo.</td><td>Dissem.</td></tr><tr><td>Qwen2.5-7B</td><td>77.04</td><td>57.85</td><td>60.37</td><td>55.78</td></tr><tr><td>Llama-3.1-8B</td><td>66.96</td><td>49.11</td><td>43.19</td><td>36.00</td></tr><tr><td>Qwen3-8B</td><td>73.33</td><td>60.89</td><td>58.96</td><td>54.07</td></tr><tr><td>GLM-4-9B</td><td>65.48</td><td>55.33</td><td>50.74</td><td>50.52</td></tr></table>

<table><tr><td colspan="6">Panel B: Macro-F1 and UASR (%)</td></tr><tr><td>Model</td><td>Orig.</td><td>Style</td><td>Emo.</td><td>Dissem.</td><td>UASR↓</td></tr><tr><td>Qwen2.5-7B</td><td>76.87</td><td>57.61</td><td>60.36</td><td>55.49</td><td>53.08</td></tr><tr><td>Llama-3.1-8B</td><td>66.34</td><td>43.15</td><td>41.08</td><td>32.95</td><td>62.61</td></tr><tr><td>Qwen3-8B</td><td>72.59</td><td>60.86</td><td>57.97</td><td>53.91</td><td>46.87</td></tr><tr><td>GLM-4-9B</td><td>62.68</td><td>47.37</td><td>44.49</td><td>41.50</td><td>37.67</td></tr></table>

Table 7: Overall zero-shot results on the topic-balanced Weibo21 evaluation set. All values are percentages. Higher Accuracy and Macro-F1 and lower UASR are better. Boldface denotes the best result in each column.

## B.2 Detailed Zero-Shot Results

Table 7 reports the overall performance of the four zero-shot LLMs on the topic-balanced Weibo21 set. Unlike the topic-averaged results in Appendix B.3, these results are computed over all 1,350 cases without separating them by topic.

Qwen2.5-7B-Instruct achieves the strongest performance on the original, emotion, and dissemination views, whereas Qwen3-8B performs best under the style transformation. Nevertheless, all four models exhibit substantially lower Macro-F1 on at least one transformed view than on the original news. Llama-3.1-8B-Instruct is the most sensitive overall, with a UASR of 62.61%. GLM-4-9B obtains the lowest UASR, but this result should be interpreted together with its comparatively low original-view and transformed-view performance because UASR conditions only on originally correct cases.

## B.3 Topic-Level Analysis

We further investigate how news topics and presentation transformations affect detection performance on Weibo21. Unlike the comparison of Englishtrained detectors in Section 5.4, this analysis averages the results of four zero-shot LLMs and is intended to characterize topic- and view-level performance variation. Figure 6 presents Accuracy and Macro-F1 across nine topics under the original view and the three presentation transformations.

Averaged across the nine topics, the zero-shot LLMs achieve a Macro-F1 of 69.45% on the original news, but their performance decreases to 51.94%, 50.76%, and 45.71% under the style, emotion, and dissemination transformations, respectively. Thus, their ability to classify unmodified Chinese news does not translate into robustness to presentation changes.

Permutation tests based on 10,000 labelstratified case-level reassignments indicate significant differences in overall detection performance across topics for both Accuracy $( p _ { \mathrm { p e r m } } = 0 . 0 0 6 1 )$ and Macro-F1 $( p _ { \mathrm { p e r m } } ~ = ~ 0 . 0 0 5 3 )$ . Consistent with Figure $^ { 6 , }$ military news is generally easier to classify, whereas education and disaster-related news are more difficult. In contrast, we find no significant evidence that the relative effects of the three transformations vary across topics: the Topic × Transformation interaction is not significant for Accuracy $( p _ { \mathrm { p e r m } } = 0 . 0 6 1 7 )$ or Macro-F1 $( p _ { \mathrm { p e r m } } = 0 . 0 7 8 1 )$ . Dissemination framing nevertheless yields the lowest Accuracy and Macro-F1 among the three transformations in every topic, making it the most consistently challenging transformation in this evaluation.

## B.4 Topic Word Clouds

Figure 7 provides a qualitative overview of the recurring terms associated with the nine Weibo21 topics included in our evaluation. The original three-by-three visualization is rearranged into a single-column layout to preserve the readability of the Chinese terms.