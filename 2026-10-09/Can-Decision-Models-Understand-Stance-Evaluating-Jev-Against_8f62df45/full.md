# Can Decision Models Understand Stance? Evaluating Jev Against General-Purpose LLMs

Xing Li<sup>1</sup>, Jinzhong Ning<sup>1</sup>\*, Yijia Zhang<sup>1</sup>, Liang Yang<sup>2</sup>, Hongfei Lin<sup>2</sup>

<sup>1</sup>Dalian Maritime University

<sup>2</sup>Dalian University of Technology

lllxxx@dlmu.edu.cn, ningjinzhong@dlmu.edu.cn liang@dlut.edu.cn, hflin@dlut.edu.cn

## Abstract

Stance detection requires identifying an author’s attitude toward a given target, sometimes based on conversational context. Jev, a specialized decision model designed for structured decision-making, offers an alternative to general-purpose large language models (LLMs). In this work, we evaluate Jev on two stance detection datasets, VAST (English texts) and ZS-CSD (Chinese conversations), comparing it with four general-purpose LLMs and two fine-tuned models. Results show that Jev achieves competitive performance on VAST, matching GPT-5.6 and outperforming the other general-purpose LLMs. However, it falls behind stronger LLMs on ZS-CSD, particularly in distinguishing favor from against. Further analysis suggests that this limitation may be related to understanding reply relationships and stance direction rather than conversation length alone. These findings highlight both the potential and limitations of Jev for stance detection.

## 1 Introduction

Stance detection aims to determine whether an author is in favor of, against, or neutral toward a given target (Mohammad et al., 2016). It is widely used in public opinion analysis (Stefanov et al., 2020), rumor detection (Wei et al., 2019), and argument mining (Bar-Haim et al., 2017). Unlike sentiment classification, stance detection requires not only identifying the attitude expressed in a text but also determining whether it is directed toward the given target. In practice, authors may express their stance through implicit opinions, indirect evaluations, or disagreement with opposing views, making stance detection more challenging than simply identifying the sentiment of a text. When moving from individual texts to conversations, stance detection also requires understanding the dialogue context (Niu et al., 2024). For example, a reply may express its stance by agreeing or disagreeing with an earlier utterance, making it difficult to determine the stance from the reply alone.

In recent years, large language models (LLMs) have provided new ways to address stance detection through their language understanding and contextual modeling abilities (Zhang et al., 2024; Weinzierl and Harabagiu, 2024). Unlike conventional approaches that rely on task-specific labeled data, general-purpose LLMs can perform stance detection directly through task instructions, reducing the need for task-specific training (Weinzierl and Harabagiu, 2024; Nguyen and Kim, 2025). However, stance detection typically has a fixed set of labels, such as favor, against, and neutral (Mohammad et al., 2016). The model only needs to select the appropriate label rather than generate openended text. This raises an interesting question: can models designed specifically for structured decision-making achieve performance comparable to general-purpose LLMs on stance detection?

Recently, Jev, a specialized decision model developed by TypeSafe AI, has offered a new way to explore this question (TypeSafe AI, 2026). Jev is designed to make structured decisions, and its Choice interface selects an answer directly from a predefined set of options. Recent studies have explored its use in scientific, legal, clinical, and rubric-based decision tasks (Deng et al., 2026; Zhang et al., 2026; Huang et al., 2026; Rao and Callison-Burch, 2026), with some reporting advantages in efficiency (Rao and Callison-Burch, 2026). Although this decision format fits the output requirements of stance detection, it remains unclear whether Jev can accurately identify an author’s stance toward a given target, especially in conversations where support or opposition may depend on earlier utterances. Therefore, evaluating Jev across different stance detection settings and comparing it with general-purpose LLMs can help reveal its performance and limitations.

In this work, we conduct an empirical evaluation of Jev on both individual-text and conversational stance detection. Specifically, we use VAST (Allaway and McKeown, 2020), an English stance detection dataset, and ZS-CSD (Ding et al., 2025), a Chinese zero-shot conversational stance detection dataset, to evaluate Jev in two different settings. We compare Jev with four general-purpose LLMs: DeepSeek V4.1 Flash, Qwen3.8-Flash, GPT-5.6, and Claude Sonnet 5. We also include Qwen2.5- 3B and Qwen2.5-7B fine-tuned with LoRA as supervised reference models. Through these experiments, we examine whether Jev can achieve competitive stance detection performance without taskspecific training, and analyze its performance differences and error patterns across the two settings.

Our results show that Jev performs differently across the two stance detection settings. On VAST, Jev achieves performance comparable to GPT-5.6 and outperforms the other general-purpose LLMs in our comparison, showing its potential for stance detection on individual texts. However, on ZS-CSD, Jev falls well behind GPT-5.6 and Claude Sonnet 5. Further analysis shows that Jev performs relatively well at distinguishing neutral from nonneutral stances but makes more errors when deciding whether a stance is in favor of or against the target. In addition, the performance gap between Jev and the stronger LLMs does not increase noticeably with conversation length. These findings suggest that Jev’s limitations in conversational stance detection may be related to its difficulty in understanding reply relationships and stance direction, rather than conversation length alone. Overall, our study shows that Jev can achieve competitive results in some stance detection settings but still faces challenges when stance judgments depend on conversational context. These findings provide initial evidence of its strengths and limitations in stance detection.

## 2 Related Work

## 2.1 Stance detection

Early work focuses on in-target stance detection, where training and evaluation use the same targets (Mohammad et al., 2016). These methods model the relationship between the target and the author’s text. For example, Du et al. (2017) use target-specific attention to identify relevant parts of the text instead of relying on a target-independent representation.

Cross-target research reduces the need for labeled data for each target by transferring knowledge between related targets. Xu et al. (2018) propose CrossNet, using self-attention to extract shared domain features. Zero-shot research evaluates generalization to unseen targets. Allaway and McKeown (2020) introduce VAST and use generalized topic representations to capture similarities between topics. Liang et al. (2022) propose JointCL, combining stance contrastive learning with targetaware prototypical graph contrastive learning to transfer stance features to unseen targets.

Recent work also uses LLMs to supply knowledge for stance prediction. Zhang et al. (2024) extract text–target relationships with an LLM and inject this information into BART. This differs from our direct comparison of Jev with general-purpose LLMs, with fine-tuned models as supervised references.

## 2.2 Conversational stance detection

CSD studies stance expressed through interactions between speakers. Early work such as SRQ distinguishes replies from quoted tweets and evaluates stance prediction for these response types (Villa-Cox et al., 2020). Multi-turn conversations require modeling dependencies beyond the immediate reply. Niu et al. (2024) introduce MT-CSD and GLAN, which combines global attention with local convolutional and graph-based branches to capture long- and short-range dependencies. These methods use dialogue structure to recover information that may be absent from the current utterance.

Target generalization is also important in conversations, where new discussion topics may lack labeled examples. Ding et al. (2025) introduce ZS-CSD, a Chinese benchmark with unseen test targets covering noun phrases and claims. Their SITPCL model combines a speaker interaction network with target-aware prototypical contrastive learning. The interaction network models dependencies within and between speakers, while contrastive learning supports generalization across targets. This line of work brings together two requirements: interpreting the current speaker’s stance in context and transferring that interpretation to unfamiliar targets. Our evaluation includes both through the full ZS-CSD test set.

## 2.3 Evaluations of Jev

Recent studies evaluate Jev on scientific (Deng et al., 2026), legal (Zhang et al., 2026), clinical (Huang et al., 2026), and rubric-based (Rao and Callison-Burch, 2026) judgments. These evaluations concern decisions made under domainspecific criteria. We use stance detection to examine decisions conditioned on a target and, for conversational records, on the history needed to interpret the current utterance.

## 3 Method

We use Jev’s Choice interface to solve the tasks in Section 3.1. Each request combines one record’s structured input, a task instruction, and a written criterion for each stance label.

## 3.1 Task formulation

## 3.1.1 Stance detection

Given an author’s text u and target t, the task is to identify the author’s expressed stance:

$$
\hat { y } = f ( t , u ) , \qquad \hat { y } \in \mathcal { V } ,\tag{1}
$$

where Y = {favor, against, neutral}. The target t may be an entity or a claim. Favor and against indicate support for and opposition to t. Neutral follows the benchmark annotations, including cases with no expressed stance or text unrelated to the target (Allaway and McKeown, 2020; Ding et al., 2025).

## 3.1.2 Conversational stance detection

Conversational stance detection uses the same stance label set and adds conversation history to the input. Let $C _ { n } = ( ( s _ { i } , u _ { i } ) ) _ { i = 1 } ^ { n }$ be a conversation, where $s _ { i }$ identifies the speaker of utterance $u _ { i }$ , and let $H _ { n } \ = \ ( ( s _ { i } , u _ { i } ) ) _ { i = 1 } ^ { n - 1 }$ be the history preceding the current utterance $u _ { n }$ . The task is

$$
\hat { y } _ { n } = f \big ( t , ( s _ { n } , u _ { n } ) , H _ { n } \big ) , \quad \hat { y } _ { n } \in \mathcal { V } .\tag{2}
$$

The output labels the current utterance $u _ { n }$ toward t; history and speaker information provide context for interpreting its stance (Ding et al., 2025).

## 3.2 Structured input representation

Let v denote the focus item, H its history, and τ the supplied target type. For VAST, $v \ = \ u$ and $H = \tau = \emptyset$ . For ZS-CSD, $\boldsymbol { v } = \left( s _ { n } , u _ { n } \right)$ , H = $H _ { n }$ , and τ indicates a noun-phrase or claim target. We construct the request state as

$$
x = \phi ( t , v , H , \tau ) ,\tag{3}
$$

where ϕ serializes these fields into a structured object. VAST inputs identify the target and focus text.

ZS-CSD inputs retain all turns in order with speaker identifiers and explicitly designate the last turn as the focus; one-turn conversations have empty history. We strip surrounding whitespace, preserve the original language, and exclude reference labels and sample-selection metadata.

## 3.3 Label-specific selection criteria

We associate each label with a fixed written criterion:

$$
{ \mathcal { O } } = \{ ( y , c _ { y } ) : y \in { \mathcal { V } } \} ,\tag{4}
$$

Here $y$ is the option identifier and $c _ { y }$ its criterion. Favor covers support or approval toward the target, against covers opposition or criticism, and neutral covers no discernible stance, unrelated or informational text, and ambiguity. Both datasets use the same criteria.

The instruction I asks Jev to choose exactly one label for the focus text, using history only to interpret that text. Appendix A gives the full wording. We supply no in-context demonstrations.

## 3.4 Prediction from the Choice response

Let $\mathcal { I } _ { \mathrm { C h o i c e } }$ denote the Jev Choice request and $\pi _ { \mathrm { c h o i c e } }$ extract its selected option identifier. Our prediction is

$$
\hat { y } = \pi _ { \mathrm { c h o i c e } } ( \mathcal { I } _ { \mathrm { C h o i c e } } ( I , x , \mathcal { O } ) ) ,\tag{5}
$$

with $\hat { y } \in \mathcal { V }$ . The client validates that the returned choice is a stance label and uses it directly, without confidence filtering or adjustment using returned probabilities. Each request classifies one record without task-specific fine-tuning.

## 4 Experiments

## 4.1 Datasets and evaluation metrics

VAST (Allaway and McKeown, 2020) contains English text–target pairs for stance detection. We evaluate all 3,006 official test records, including the seen-target (few-shot) and unseen-target (zeroshot) subsets, and retain repeated records.

ZS-CSD (Ding et al., 2025) evaluates Chinese conversational stance toward unseen targets. We use all 2,584 official test records, covering nounphrase and claim targets. Exact target strings are disjoint across training, development, and test splits. For context, SITPCL, the best model in the original paper, reaches 43.81% macro-F1 on this test set.

<table><tr><td>Dataset</td><td>Model / setting</td><td>Accuracy</td><td>Macro-F1</td><td>Input tokens</td><td>Output tokens</td><td>Cost</td></tr><tr><td rowspan="7">VAST</td><td>Jev</td><td>78.14</td><td>77.92</td><td>1,763,143</td><td>114,228</td><td>$0.0741</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>74.82</td><td>73.99</td><td>865,478</td><td>18,708</td><td>$0.2133</td></tr><tr><td>Qwen3.8-Flash</td><td>74.52</td><td>73.79</td><td>763,609</td><td>28,874</td><td>$0.1021</td></tr><tr><td>GPT-5.6</td><td>78.51</td><td>78.08</td><td>1,178,659</td><td>33,439</td><td>$3.9010</td></tr><tr><td>Claude Sonnet 5</td><td>74.82</td><td>74.51</td><td>2,279,907</td><td>251,109</td><td>$4.7051</td></tr><tr><td>Qwen2.5-3B-Instruct (SFT)†</td><td>74.16</td><td>74.52</td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B-Instruct (SFT)†</td><td>79.32</td><td>79.91</td><td></td><td></td><td></td></tr><tr><td rowspan="7">ZS-CSD</td><td>Jev</td><td>58.40</td><td>58.43</td><td>1,846,931</td><td>98,192</td><td>$0.0776</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>58.28</td><td>56.89</td><td>815,813</td><td>15,920</td><td>$0.1829</td></tr><tr><td>Qwen3.8-Flash</td><td>54.61</td><td>53.16</td><td>707,481</td><td>27,236</td><td>$0.0948</td></tr><tr><td>GPT-5.6</td><td>74.65</td><td>73.62</td><td>1,140,780</td><td>35,592</td><td>$3.9230</td></tr><tr><td>Claude Sonnet 5</td><td>72.52</td><td>72.42</td><td>2,402,197</td><td>325,790</td><td>$5.5582</td></tr><tr><td>Qwen2.5-3B-Instruct (SFT)†</td><td>55.76</td><td>56.08</td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B-Instruct  $( \mathrm { S F T } ) ^ { \dag }$ </td><td>60.98</td><td>60.85</td><td></td><td></td><td></td></tr></table>

Table 1: Performance and resource usage on the full test sets of VAST and ZS-CSD. Bold indicates the best results among prompted models. Dashes denote unavailable measurements. <sup>†</sup>SFT results are from single runs without per-instance predictions.

We report accuracy and three-class macro-F1:

$$
F _ { 1 , k } = \frac { 2 T P _ { k } } { 2 T P _ { k } + F P _ { k } + F N _ { k } } ,\tag{6}
$$

$$
F _ { \mathrm { m a c r o } } = { \frac { 1 } { 3 } } \sum _ { k = 1 } ^ { 3 } F _ { 1 , k } .\tag{7}
$$

All five prompted runs per dataset use the same test IDs and reference labels. A failed final response receives no accuracy credit and counts as a false negative for its reference class. Paired target-bootstrap intervals use 5,000 resamples, seed 42, and percentile 95% bounds without multiple-comparison adjustment. Each sampled target contributes all its records to both models, with 759 target clusters in VAST and 45 in ZS-CSD.

## 4.2 Models and inputs

We call a system prompted when it is used without task-specific fine-tuning or in-context demonstrations. All five prompted systems receive the target and the focus text, with speaker-tagged history for ZS-CSD. Jev follows the Choice procedure in Section 3, while the LLMs receive the label definitions in a shorter prose instruction (Appendix A).

Jev. We request jev-latest, returned as jev-1.13.0, and classify one record per Choice request.

Flash models. DeepSeek V4.1 Flash (DeepSeek, 2026) is called through the Responses API with a schema-constrained stance field, and Qwen3.8- Flash (Alibaba Cloud, 2026) through an OpenAIcompatible Chat Completions API with JSON output. Both use temperature zero and a 32-token output limit, with thinking disabled.

GPT-5.6 and Claude Sonnet 5. GPT-5.6 (OpenAI, 2026) runs through Codex CLI with no tools and reasoning effort set to none. Claude Sonnet 5 (Anthropic, 2026) is also evaluated with thinking disabled. Unlike the other three systems, which classify one record per request, GPT-5.6 and Claude Sonnet 5 receive up to 100 and 25 records per request, respectively, and are instructed to classify each record independently.

SFT references. As supervised fine-tuning (SFT) references, we train Qwen2.5-3B-Instruct and 7B-Instruct (Qwen Team, 2024) with LoRA (Hu et al., 2021) on each training split, with learning rate $5 \times 1 0 ^ { - 5 }$ , five epochs, and seed 42. Only their single-run scores are available, so the paired and subgroup analyses cover the five prompted systems.

## 4.3 Experimental settings

The single-record runs of Jev, DeepSeek V4.1 Flash, and Qwen3.8-Flash were collected on September 21–24, 2026, with 16 concurrent requests and up to three retries with backoff. GPT-5.6 processes each dataset in sequential batches, and Claude Sonnet 5 resumes from checkpoints, splitting failed batches until all records are recovered. Each record is scored once on its final outcome. Jev, DeepSeek V4.1 Flash, GPT-5.6, and Claude Sonnet 5 return valid final predictions throughout both tests. Qwen3.8-Flash has one VAST request rejected by content screening; its effect on the score is small, but it remains in the full-test denominator. Appendix B summarizes the reasoning settings.

<table><tr><td>Dataset</td><td>Model</td><td>Against</td><td>Favor</td><td>Neutral</td></tr><tr><td rowspan="3">VAST</td><td>Jev</td><td>77.13</td><td>72.71</td><td>83.92</td></tr><tr><td>DeepSeek Qwen</td><td>74.15 73.88</td><td>66.83 65.52</td><td>81.00 81.97</td></tr><tr><td>GPT-5.6 Claude</td><td>78.25 75.35</td><td>73.95 70.26</td><td>82.05 77.91</td></tr><tr><td rowspan="4">ZS-CSD</td><td>Jev</td><td>56.66</td><td>59.58</td><td>59.04</td></tr><tr><td>DeepSeek</td><td>54.60</td><td>51.18</td><td>64.89</td></tr><tr><td>Qwen</td><td>55.59</td><td>42.45</td><td>61.42</td></tr><tr><td>GPT-5.6 Claude</td><td>79.49 75.78</td><td>82.70 77.50</td><td>58.67 63.99</td></tr></table>

Table 2: Per-class F1 (%). Bold indicates the best result for each class within each dataset.

![](images/0bc5698ef9b0b1d12a3203c484156e8abb439e41fac8f78514d33d99c5cddb12.jpg)  
Figure 1: Macro-F1 differences between Jev and each LLM (Jev minus LLM, in percentage points), with paired target-bootstrap 95% intervals.

In addition to the scores, Table 1 lists the input and output tokens recorded for retained successful requests or completed batches, together with their cost in US dollars.

## 5 Results

## 5.1 Main Results

Table 1 presents the overall performance and resource usage of the evaluated models on VAST and ZS-CSD. Figure 1 further shows the paired macro-F1 differences between Jev and the four generalpurpose LLMs. We summarize the main findings as follows.

(1) Jev performs competitively on VAST but shows a substantial performance gap on ZS-CSD. On VAST, Jev achieves 77.92% macro-F1, only 0.16 percentage points below GPT-5.6 (78.08%) and outperforming the other three general-purpose LLMs. The paired confidence interval for the difference between Jev and GPT-5.6 includes zero, indicating that the observed difference is not statistically significant under our evaluation. However, this competitive performance does not extend to ZS-CSD. Jev achieves 58.43% macro-F1, falling 15.19 and 13.99 points behind GPT-5.6 and Claude Sonnet 5, respectively. Both differences are supported by the paired confidence intervals. These results suggest that Jev can handle stance classification in individual English texts effectively, but remains less competitive in the more challenging Chinese conversational setting.

(2) Jev offers a favorable performance–cost trade-off, particularly on VAST. Jev outperforms both DeepSeek V4.1 Flash and Qwen3.8-Flash in macro-F1 on the two datasets, although its advantage over DeepSeek V4.1 Flash on ZS-CSD is not statistically established. More importantly, Jev achieves these results at a relatively low inference cost. On VAST, its recorded API cost is only \$0.0741, compared with \$3.9010 for GPT-5.6, despite their nearly identical macro-F1 scores. Similar cost differences are observed on ZS-CSD, although Jev performs substantially worse than the stronger LLMs. These findings indicate that Jev can be a cost-effective option for stance detection when its accuracy meets task requirements, especially for individual-text classification.

(3) Jev achieves comparable performance to task-specific fine-tuned models without additional training. On VAST, Jev achieves 77.92% macro-F1, outperforming the LoRA-tuned Qwen2.5-3B (74.52%) but falling below Qwen2.5- 7B (79.91%). A similar pattern appears on ZS-CSD, where Jev achieves 58.43%, compared with 56.08% and 60.85% for the two fine-tuned models. Although Jev does not outperform the larger finetuned model, it achieves competitive results without requiring task-specific training. This provides an additional reference point for evaluating the potential of specialized decision models on stance detection tasks.

## 5.2 Class-level Performance

On VAST, Jev achieves performance comparable to GPT-5.6 across all three stance categories (Table 2). GPT-5.6 performs slightly better on favor (73.95% vs. 72.71%) and against (78.25% vs. 77.13%), while Jev achieves the highest F1 on neutral (83.92%). These results indicate that Jev maintains competitive performance across different stance categories on individual English texts, consistent with the overall results.

<table><tr><td>Target type</td><td>N</td><td>Jev</td><td>DeepSeek</td><td>Qwen</td><td>GPT-5.6</td><td>Claude</td></tr><tr><td>Noun phrase</td><td>967</td><td>63.49</td><td>61.61</td><td>60.80</td><td>78.02</td><td>76.19</td></tr><tr><td>Claim</td><td>1,617</td><td>55.26</td><td>54.08</td><td>48.58</td><td>70.72</td><td>70.13</td></tr></table>

Table 3: Macro-F1 scores (%) across different target types on ZS-CSD. Bold indicates the best result for each target type.

On ZS-CSD, however, the performance differences are more pronounced. GPT-5.6 and Claude Sonnet 5 substantially outperform Jev on favor and against, while their neutral F1 scores remain relatively close. For example, GPT-5.6 exceeds Jev by more than 22 percentage points on both favor and against, but achieves a similar F1 on neutral (58.67% vs. 59.04%). The confusion analysis further shows that Jev makes 373 favor–against errors, compared with only 35 for GPT-5.6 and 101 for Claude Sonnet 5 (Table 5). These findings suggest that Jev’s main difficulty on ZS-CSD lies in distinguishing support from opposition, rather than in the neutral category. This may reflect limitations in interpreting stance direction when opinions are expressed through interactions between speakers.

## 5.3 Performance across Target Types

Table 3 reports the macro-F1 scores on the two target types in ZS-CSD. All five models perform better on noun-phrase targets than on claim targets, suggesting that stance detection becomes more challenging when targets are expressed as complete propositions. For Jev, macro-F1 decreases from 63.49% on noun phrases to 55.26% on claims. A similar trend is observed for the other models. This may be because claim targets often contain more complex semantic information, such as negation or qualifying conditions, which requires more precise interpretation of the target.

Jev outperforms the two Flash models on both target types but remains substantially behind GPT-5.6 and Claude Sonnet 5. In particular, the gap between Jev and GPT-5.6 is 14.53 percentage points on noun phrases and 15.46 points on claims. These results indicate that Jev’s performance limitations on ZS-CSD are consistent across different target types, rather than being restricted to a particular type of target.

## 5.4 Performance across Conversation Lengths

Table 4 reports the macro-F1 scores across different conversation lengths on ZS-CSD. Jev achieves the highest score on single-turn inputs, although this group contains only 45 samples. For conversations with two or more turns, GPT-5.6 and Claude Sonnet 5 consistently outperform Jev. Notably, the performance gap does not increase steadily with conversation length. For example, the gap between Jev and GPT-5.6 decreases from 22.69 percentage points for two-turn conversations to 14.20 points for conversations with six or more turns. These results suggest that Jev’s limitations in conversational stance detection cannot be explained by conversation length alone. Instead, its difficulty may also involve interpreting stance relationships between speakers, even in relatively short conversations.

<table><tr><td>Turns</td><td>N</td><td>Jev</td><td>DeepSeek</td><td>Qwen</td><td>GPT-5.6</td><td>Claude</td></tr><tr><td>1</td><td>45</td><td>66.67</td><td>53.55</td><td>62.69</td><td>64.34</td><td>59.80</td></tr><tr><td></td><td>2 445</td><td>56.99</td><td>52.78</td><td>51.37</td><td>79.68</td><td>74.77</td></tr><tr><td></td><td>3877</td><td>57.45</td><td>54.27</td><td>50.52</td><td>74.95</td><td>70.44</td></tr><tr><td></td><td>4 585</td><td>58.11</td><td>57.27</td><td>50.83</td><td>69.05</td><td>70.77</td></tr><tr><td></td><td>5 351</td><td>54.64</td><td>57.55</td><td>52.65</td><td>68.23</td><td>69.97</td></tr><tr><td></td><td>6+ 281</td><td>54.00</td><td>55.80</td><td>57.92</td><td>68.20</td><td>70.06</td></tr></table>

Table 4: Macro-F1 scores (%) across different conversation lengths on ZS-CSD. Bold indicates the best result for each group.

## References

Alibaba Cloud. 2026. Qwen3.8-Flash: Model information and pricing. Vendor documentation. Pricing recorded September 24 and checked September 27, 2026.

Emily Allaway and Kathleen McKeown. 2020. Zeroshot stance detection: A dataset and model using generalized topic representations. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8913– 8931. Association for Computational Linguistics.

Anthropic. 2026. Claude Sonnet 5: Model overview. Vendor documentation. Accessed September 28, 2026.

Roy Bar-Haim, Indrajit Bhattacharya, Francesco Dinuzzo, Amrita Saha, and Noam Slonim. 2017. Stance classification of context-dependent claims. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 251–261. Association for Computational Linguistics.

DeepSeek. 2026. DeepSeek API: Model identifiers and first API call. Vendor documentation. Accessed September 27, 2026.

Boyuan Deng, Shuyi Fan, Hongyang Zhang, and Xinhong Xie. 2026. Jev for scientific decisions: Evaluating semantic choices and their consequences. arXiv preprint arXiv:2609.24965.

Yuzhe Ding, Kang He, Bobo Li, Li Zheng, Haijun He, Fei Li, Chong Teng, and Donghong Ji. 2025. Zeroshot conversational stance detection: Dataset and approaches. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 3221–3235. Association for Computational Linguistics.

Jiachen Du, Ruifeng Xu, Yulan He, and Lin Gui. 2017. Stance classification with target-specific neural attention. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, pages 3988–3994. International Joint Conferences on Artificial Intelligence Organization.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Jiaju Huang, Hao Yang, Xinyu Ma, Xinglong Liang, Kunyan Cai, Junqiang Ma, Shaobin Chen, Yue Sun, and Tao Tan. 2026. Can Jev judge radiology reports? Evaluating a System One model for clinical factuality. arXiv preprint arXiv:2609.27607.

Bin Liang, Qinglin Zhu, Xiang Li, Min Yang, Lin Gui, Yulan He, and Ruifeng Xu. 2022. JointCL: A joint contrastive learning framework for zero-shot stance detection. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 81–91. Association for Computational Linguistics.

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. SemEval-2016 task 6: Detecting stance in tweets. In Proceedings ofthe 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 31–41. Association for Computational Linguistics.

Quang Minh Nguyen and Taegyoon Kim. 2025. Is external information useful for stance detection with LLMs? In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 14798–14807. Association for Computational Linguistics.

Fuqiang Niu, Min Yang, Ang Li, Baoquan Zhang, Xiaojiang Peng, and Bowen Zhang. 2024. A challenge dataset and effective models for conversational stance detection. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 122–132. ELRA and ICCL.

OpenAI. 2026. GPT-5.6 model guidance. Vendor documentation. Accessed September 28, 2026.

Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Delip Rao and Chris Callison-Burch. 2026. JEV vs. LLMs as rubric judges: Cheaper, faster, and wrong in the same places. arXiv preprint arXiv:2609.29769.

Peter Stefanov, Kareem Darwish, Atanas Atanasov, and Preslav Nakov. 2020. Predicting the topical stance and political leaning of media using tweets. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 527–537. Association for Computational Linguistics.

TypeSafe AI. 2026. Jev: Introduction and Choice API. Vendor documentation. Accessed September 27, 2026.

Ramon Villa-Cox, Sumeet Kumar, Matthew Babcock, and Kathleen M. Carley. 2020. Stance in replies and quotes (SRQ): A new dataset for learning stance in Twitter conversations. arXiv preprint arXiv:2006.00691.

Penghui Wei, Nan Xu, and Wenji Mao. 2019. Modeling conversation structure and temporal dynamics for jointly predicting rumor stance and veracity. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 4787–4798. Association for Computational Linguistics.

Maxwell Weinzierl and Sanda Harabagiu. 2024. Treeof-counterfactual prompting for zero-shot stance detection. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 861–880. Association for Computational Linguistics.

Chang Xu, Cécile Paris, Surya Nepal, and Ross Sparks. 2018. Cross-target stance classification with selfattention networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 778–783. Association for Computational Linguistics.

Fan Zhang, Yankai Chen, Zhuohan Xie, Yixi Zhou, Sijia Peng, Lei Fan, Xinhua Ji, Cunyuan Zheng, Huangyong Shan, Philip S. Yu, Xue Liu, Yu Chen, Preslav Nakov, and Songwei He. 2026. Same scores, different decisions: Evaluating JEV and language models for legal document understanding. arXiv preprint arXiv:2609.27678.

Zhao Zhang, Yiming Li, Jin Zhang, and Hui Xu. 2024. LLM-driven knowledge injection advances zero-shot and cross-target stance detection. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 371–378. Association for Computational Linguistics.

## A Prompts and Input Construction

Inputs retain their original language. Reference labels and sample-selection metadata are not included in model inputs. The instructions and label definitions are given below.

## Jev Choice instruction.

Determine the stance expressed by the focus text toward the target. Use conversation context only to resolve the focus text. Choose exactly one label according to the criteria.

The criteria of its three options are:

• favor: The focus text supports, approves of, argues for, or expresses a positive attitude toward the target.

• against: The focus text opposes, rejects, criticizes, argues against, or expresses a negative attitude toward the target.

• neutral: The focus text has no discernible stance toward the target, is unrelated, merely reports information, or is too ambiguous to label as favor or against.

Single-record LLM instruction. The DeepSeek V4.1 Flash and Qwen3.8-Flash requests begin with this text:

You are a stance classifier. Determine the stance expressed by the focus text toward the target. Use earlier conversation only to interpret the focus text. Favor means support or approval; against means opposition, rejection, or criticism; neutral means no discernible stance, unrelated, merely informational, or too ambiguous.

DeepSeek V4.1 Flash constrains the response with a stance-enum schema. Qwen3.8-Flash requests a single-key JSON object, and its label is validated in the client.

Batched LLM instruction. GPT-5.6 and Claude Sonnet 5 share the same stance criteria, with explicit instructions to classify each record independently, use earlier turns only to interpret its focus text, avoid tools, and return one prediction per input ID in the same order as JSON. Each batch contains test IDs and inputs, without reference labels.

Input construction. Section 3.2 specifies the structured input construction. We add no translation, retrieval, manual text editing, or few-shot examples.

## B Reasoning Settings

Thinking is disabled for all four LLMs.

## C Pairwise Confusions on ZS-CSD

Table 5 counts the ZS-CSD errors between each pair of labels. GPT-5.6 and Claude Sonnet 5 make far fewer favor–against errors than Jev, whereas their errors involving neutral are much closer to Jev’s.

<table><tr><td>System</td><td>Fav.-Ag.</td><td>Fav.-Neu. Ag.-Neu.</td><td></td></tr><tr><td>Jev</td><td>373</td><td>304</td><td>398</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>275</td><td>368</td><td>435</td></tr><tr><td>Qwen3.8-Flash</td><td>462</td><td>232</td><td>479</td></tr><tr><td>GPT-5.6</td><td>35</td><td>301</td><td>319</td></tr><tr><td>Claude Sonnet 5</td><td>101</td><td>306</td><td>303</td></tr></table>

Table 5: Number of ZS-CSD test records whose reference and predicted labels form each pair, in either direction. Bold indicates the fewest errors for each label pair.