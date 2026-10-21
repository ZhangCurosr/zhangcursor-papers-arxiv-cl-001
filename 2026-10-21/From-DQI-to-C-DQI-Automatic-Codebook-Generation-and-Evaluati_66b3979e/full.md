# From DQI to C-DQI: Automatic Codebook Generation and Evaluation for Measuring Computational Discourse Quality Index

Esther Shizgal<sup>1∗</sup> Ron Keinan<sup>1∗</sup> Daniel Shtauber<sup>2</sup>

Roee Tzvi Esquira<sup>1</sup> Matan Toledano<sup>1</sup> Or Rappel-Kroyzer<sup>3</sup> Liron Lavi<sup>4</sup> Ori Aronson<sup>5</sup> Renana Keydar<sup>3,6</sup> Ittai Bar-Siman-Tov<sup>5</sup> Reut Tsarfaty<sup>1</sup>

<sup>1</sup>Department of Computer Science, Bar-Ilan University

<sup>2</sup>Max Planck Center for Democracy, Security and Human Rights, Faculty of Law, Hebrew University

<sup>4</sup>Department of Political Studies, Bar-Ilan University <sup>5</sup>Faculty of Law, Bar-Ilan University <sup>6</sup>Faculty of Law, Hebrew University

<sup>1,4,5</sup>{first\_name}.{last\_name}@biu.ac.il <sup>2,3,6</sup>{first\_name}.{last\_name}@mail.huji.ac.il

## Abstract

Effective democratic deliberation is vital for civic engagement and the health of legislative institutions. However, assessing parliamentary discourse quality is traditionally done using expert-scored, theory-based codebooks (CBs), which are rigid, labor-intensive, and prone to biases. While Large Language Models (LLMs) offer a scalable alternative for both schema induction and discourse annotation, ensuring schema quality and mitigating models inherent bias remains a clear challenge. In this work, we propose a complete protocol for LLM-based codebook generation and evaluation. Starting from the widely accepted discourse quality Index (DQI), we compare topdown (theory-based), bottom-up (data-driven), and hybrid approaches for generating contextadapted schemas. To evaluate these schemas, we define an extrinsic proxy measure, validated by manual annotations, alongside an intrinsic inter-model agreement (IMA) metric, which leverages differing model biases to signal overall reliability. Our experiments on 100 Israeli parliament transcripts reveal that inductive, data-driven schemas achieve higher IMA than top-down methods while maintaining validity, demonstrating their potential for reliable, large-scale analysis of political discourse.

## 1 Introduction

Deliberation is the interactive process where individuals engage in discussion to reach rational conclusions, occurring in both formal political bodies and local civic settings. At its core lies the premise that political decision-making should be talk-centric rather than vote-centric (Steenbergen et al., 2003). Effective democratic deliberation is crucial for civic engagement and the health of legislative institutions (Serrano-Contreras et al., 2020; Schroeder et al., 2024; Tschentscher et al., 2010). Prominent literature identifies thoughtful analysis, open-mindedness, mutual respect, and participants’ willingness to revise their initial preferences through discourse as prerequisites for this efficacy (Steenbergen et al., 2003; Ercan et al., 2022).

Quantifying deliberation quality enables its application to downstream tasks, such as identifying anomalous deliberative behavior, evaluating the performance of legislators, and alerting civic facilitators when participant satisfaction could be improved (Schroeder et al., 2024). But while numerous factors have been argued to affect discourse quality, such as domination, issue polarization, and agenda setting (Kumar et al., 2025; Serrano-Contreras et al., 2020), quantifying this quality remains challenging due to the inherent competing interests between interlocutors, rhetorical ambiguity, and the variance in acceptable standards across societies and domains.

Various methods, including computational frameworks, have been proposed for measuring multi-party discourse quality (Steenbergen et al., 2003; Behrendt et al., 2024; Bosley, 2025). A common protocol for quality assessment involves experts evaluating live multi-party discourse using a theoretical-scientific scoring schema. This schema, known as a codebook (CB), comprises categories that reflect rationality, reciprocity, and civility, and uses ordinal scales to individually assess each criterion (Behrendt et al., 2024; Graham, 2010; Esau et al., 2021; Bächtiger et al., 2010; Steenbergen et al., 2003). However, this paradigm has various limitations. First, rigid, theory-based schemas often struggle to generalize across topic-dependent or culture-dependent nuances. Second, collecting human annotation is labor-intensive and inherently subject to individual biases (Romberg et al., 2025; Vecchi et al., 2025; Brown et al., 2025).

![](images/b51e3cec5159fcc1ee77eb68ec407504639b2605599c40ad14cf28a798ab924e.jpg)  
Figure 1: The four workflows for deriving CBs for deliberation quality assessment: (a) the standard DQI CB (baseline), and three LLM-generated CBs — (b) theory-led (top-down), (c) data-driven (bottom-up), and (d) a hybrid CB combining the two previous approaches.

![](images/d81006f58895c102c654b077b0d037591d9dc2b73205a9bb1334a5defc29978a.jpg)  
Figure 2: The evaluation pipeline: each discussion is segmented and scored against a given CB by multiple models, whose outputs are averaged. Repeating this across discussions yields the CB’s reliability (inter-model agreement) and validity (rank correlation with proxy C-DQI).

To automate the application of such schemas, researchers have employed supervised machine learning (Fournier-Tombs and Di Marzo Serugendo, 2020; Fournier-Tombs and MacKenzie, 2021; Berkovitz et al., 2023) and Large Language Models (LLMs) (Bosley, 2025; Langlais et al., 2025; Chen et al., 2026). However, both approaches still rely upon the same rigid frameworks.

In this work, we propose Computational Discourse Quality Index (C-DQI), a methodology for the definition and application of CBs for assessing discourse quality. We explore four distinct workflows for C-DQI CB generation. We start with a baseline adapted from Steenbergen et al. (2003)’s Discourse Quality Index (DQI) and implement three pipelines that deploy LLMs for CB induction, differing in their balance of top-down (theorybased) and bottom-up (data-driven) approaches.

To evaluate the CBs, we propose both intrinsic and extrinsic measures. Following Steenbergen et al. (2003), who discuss the importance of robustness in a discourse quality index, and acknowledging that LLMs are inherently pre-trained with internal biases (Mor-Lan et al., 2026), we argue that the consensus of multiple models indicates that the CBs’ criteria are reliable for LLM annotation, rather than artifacts of a single model. We thus propose to define Inter-Model Agreement (IMA) scores across multiple LLMs to capture intrinsic reliability. A higher IMA score indicates a clearer, more objective, and robust framework for automated protocol annotation. However, because agreement does not equate to validity, we further compare the CBs’ overall assessments against human-validated proxies for deliberation quality, utilizing both manual and automatic annotations. Figure 1 demonstrates the full C-DQI pipeline and evaluation process.

To assess the efficacy of our proposed workflows, we apply different LLMs to annotate a corpus of 100 committee meeting transcripts from the

Israeli parliament, the Knesset, according to each CB. The data spans multiple governments and a diverse range of bills, providing a dynamic, culturedependent environment for examining discourse quality. Utilizing this region-specific data demonstrates the cultural alignment challenges inherent in analyzing localized political discourse.

Our results reveal that bottom-up CBs exhibit higher IMA, while all CBs correlate comparably with the quality proxy $( \rho > 0 . 5 8 )$ , demonstrating that data-driven schema generation yields more reliable evaluations without sacrificing validity. Through this work, our primary contributions are twofold: (1) we develop a computational tool that automatically defines and evaluates discourse quality CBs for multi-party discussions and scores the data accordingly; and (2) we provide empirical insights into the optimal balance between top-down theoretical frameworks and bottom-up, inductive approaches for yielding reliable evaluation criteria.

## 2 Background

Discourse Quality Measures. A widely applied measure for deliberative quality is the Discourse Quality Index (DQI) originally developed for the analysis of parliamentary debates (Steenbergen et al., 2003). The DQI is theoretically grounded in established discourse ethics and its schema includes criteria representing norms that discourse is expected to satisfy. Its unit of analysis is an individual speech, defined as “the public discourse by a particular individual delivered at a particular point in a debate” (§4.2 provides further details).

Yet, this de facto standard is constrained by drawbacks tied directly to the DQI framework, as well as broader limitations inherent to predefined schemas. First, despite well-defined coding categories, the DQI does not prescribe a standardized procedure for aggregating category-level annotations into a single discourse-level score, complicating its use in large-scale automated evaluation (Steenbergen et al., 2005). Second, according to the DQI, offtopic discourse and clarifying questions are excluded from the final score, and interruptions are only accounted for if participants complain about them, ignoring the potential for these factors to contribute to the overall quality assessment.

More generally, hard-coded CBs target specific settings and contexts, and therefore struggle to generalize to topic-dependent or culture-dependent contexts; and manually annotating them is laborintensive and inherently subject to individuals’ biases (Romberg et al., 2025; Vecchi et al., 2025; Brown et al., 2025).

Computational Discourse Quality Evaluation. Multiple efforts have operationalized discourse quality assessment by applying supervised machine learning to the DQI (Fournier-Tombs and Di Marzo Serugendo, 2020; Fournier-Tombs and MacKenzie, 2021; Berkovitz et al., 2023). Furthermore, Kumar et al. (2025) recently combined argumentation graphs with the DQI for more objective assessment. Deep learning approaches like the AQuA framework successfully bridge NLP with deliberation theory to analyze discourse semantics, though they remain resource-intensive due to the overhead of running multiple fine-tuned BERT adapters (Behrendt et al., 2024).

Currently, the field is actively moving toward using LLMs as evaluative agents, bypassing the limitations of manual annotation and rigid linguistic features (Langlais et al., 2025; Chen et al., 2026; Bosley, 2025). Specifically, Bosley (2025) evaluated the DQI using a few-shot prompting framework, achieving IMA comparable to human standards. Their work laid the foundation for our methodological study on C-DQI schemas.

Automatic Schema Discovery. To address the limitations of rigid, predefined schemas, researchers have utilized LLMs for automatic schema definition. Earlier work deploys LLMs to construct literature-review tables by prompting the model to generate a schema and then populating the table from the underlying data (Newman et al., 2024), optionally conditioned on a user-provided intent (Padmakumar et al., 2025).

More recently, human-in-the-loop systems have been developed to automate schema definition over large corpora of unstructured text, including ScheMatiQ (Levy et al., 2026) and Schemaminer<sup>pro</sup> (Sadruddin et al., 2025, 2026). These systems combine prompting pipelines with interactive interfaces and agentic tooling.

To our knowledge, however, the outputs of these systems have been evaluated mainly for their agreement with expert-annotated schemas, rather than being systematically assessed for intrinsic reliability, which we emphasize in our current work.

## 3 The Challenge

Without systematically evaluating codebook efficacy, the question remains: How can we derive new or adapted schemas for new settings and domains? How can we determine whether a schema is good? And, how can we avoid the undermining effects of inherent biases of individual LLMs?

## 4 A Complete Protocol for LLM-based Codebook Generation and Evaluation

To remove the reliance on rigid CBs that require manual annotation, we propose an LLMbased methodology for effective and efficient definition and application of a Computational Discourse Quality Index (C-DQI). This methodology addresses the core limitations of traditional schemabased annotations by allowing for flexible schema induction across different contexts and replacing human annotators with LLMs to mitigate subjective human biases.

In §4.1 we introduce C-DQI, used to quantify quality based on a codebook. In §4.2 we then define the computation task of codebook induction and its various variants, and finally in §4.3 we define metrics that aim to assess the reliability and validity of the generated codebook.

## 4.1 Computational Discourse Quality Index

We next define the notion of a Computational Discourse Quality Index (C-DQI) which is our means to define discourse quality in quantitative terms. Let D be an input transcript and C be a given CB. The input transcript D is split into multiple segments $D = ( s _ { 1 } , s _ { 2 } , \ldots , s _ { n } )$ , and the CB C defines a set of evaluation categories $C = \{ c _ { 1 } , c _ { 2 } , \dots , c _ { m } \}$ where each category is scored on a predefined scale.<sup>1</sup> For every segment-category pair $( s _ { i } , c _ { j } )$ an LLM annotator M evaluates the text on the required scale, yielding a score $y _ { i , j }$ . The per-category discussion score, $S ( c _ { j } )$ , is calculated as the average across all segments:

$$
S ( c _ { j } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } y _ { i , j }
$$

This final index, representing the overall discourse quality for the transcript, is the average over all

categories:

$$
\mathrm { C \mathrm { - D Q I } } ( D , M ) = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } S ( c _ { j } )
$$

When a single comprehensive score per transcript is required, we report the mean of the C-DQI scores produced by the ensemble of annotator models.

## 4.2 LLM-based Codebook Generation

We define the task of CB generation as a function from texts to a set of categories. Formally, let T be a collection of texts and let $C = \{ c _ { 1 } , \ldots , c _ { n } \}$ be a finite set of categories describing dimensions for evaluating discourse; the CB generation task is defined as $f ( T ) = C$

Specifically, we implement four workflows to derive CBs. One is a standard baseline, and three are LLM-based — one receives academic literature as the input text (top-down), the second reviews transcripts from the domain being studied (bottomup), and the third combines both inputs (hybrid) (see Figure 1). All prompts are available in Appendix A.

Vanilla C-DQI. The baseline workflow adapts the original Discourse Quality Index (DQI), which features seven coding categories: (1) participation, (2) level of justification, (3) content of justification, (4) respect for the groups to be helped by particular policies, (5) respect toward the demands of others, (6) respect toward counterarguments, and (7) constructive politics.

Deriving the final score involves coding all speeches and interpreting their criteria scores. The authors define relevant speeches as ones containing a policy proposal (originally termed a ‘demand’).

The coding process requires initial filtering to distinguish relevant excerpts, which are then scored per category. For this workflow, we adapt the authors’ original coding guidelines (Steenbergen et al., 2005) by assigning numerical values to their scales and adjust the unit of analysis from individual speeches to segments of up to 1,500 words to maintain consistency with our other CBs.

Top-Down C-DQI. To define the theory-driven (Top-Down) CB, we provide an LLM with a selection of academic literature on deliberation to generate a set of categories. We begin by prompting the literature-survey tool developed by Gupta (2026) with the instruction: “List 10 foundational academic papers on democratic deliberation, in order to develop a discourse quality index”, followed by an iterative process of refining the results according to our prior knowledge and available sources.

The resulting list ranges from sources studying Habermas’s discourse ethics (Heath, 2019) to methodological articles (Steenbergen et al., 2003; Stromer-Galley, 2007). The selection also includes frameworks for evaluating deliberative quality (Mansbridge et al., 2012) and sources providing dimensions for evaluation rooted in deliberative theory (Friess and Eilders, 2015).

We then instruct Gemini-3.1-Pro<sup>2</sup> to define metrics for evaluating the quality of discourse in the selected dataset, based on the frameworks from the selected sources. The response serves as the Top-Down CB.

Bottom-Up C-DQI. To define the inductive (Bottom-Up) CB, we prompt Gemini-3.1-Pro with 10 random transcripts to generate evaluation metrics from empirical data. To prevent contamination, these transcripts are excluded from later experiments. Ultimately, this workflow yields three CB variants: Bottom-Up-English, generated from an English prompt; Bottom-Up-Hebrew, generated from a Hebrew prompt; and Bottom-Up-Corrected, which we, as native Hebrew speakers, merged from the two, edited for content, and standardized in structure (Appendix D).

Hybrid C-DQI. The Hybrid C-DQI CB combines both the theory-led and inductive approaches. We generate this CB by instructing Google’s Gemini Notebook <sup>3</sup> to synthesize the previously selected academic sources with the empirical transcripts, to create a unified set of metrics for evaluating discourse quality in general committee meetings.

## 4.3 LLM-based Codebook Evaluation

CB Reliability. We define the measure of reliability as the consistency of the generated CBs, calculated based on IMA scores and averaged over each CB’s criteria.

Steenbergen et al. (2003) name four properties that a measure of discourse quality must exhibit: theoretical grounding, reliance on observable phenomena, generalizability, and reliability. To evaluate reliability, they use the ratio of agreement among coders, Cohen’s κ, Spearman’s $\rho ,$ and the standardized α.

Inter-annotator agreement serves as a lens into the complexity of the task and the quality of the annotations. Higher values indicate that the definitions are clear for different coders, whereas lower values suggest both annotator performance issues and poorly defined categories.

We introduce Inter-Model Agreement (IMA) to quantify consensus across different models. Because these LLMs are developed by diverse organizations on distinct training data (Shmidman et al., 2026; Anthropic, 2026; Gemma Team, 2026; Jiang et al., 2023; Bai et al., 2023), we hypothesize they exhibit divergent inherent biases; prior research on cultural biases validates this premise (Mor-Lan et al., 2026). This diversity contrasts with human annotators, who often share demographic backgrounds or subjective opinions. Consequently, high IMA serves as a compelling measure of robustness. If multiple models agree on the scoring criteria, it indicates that the CBs are reliable for LLM annotation, rather than reflecting the artifacts of a singular model. Thus, a higher agreement score indicates a clearer, more objective, and robust framework for automated protocol annotation.

Following standard practices for evaluating the DQI, we report the percent agreement as a baseline measure of consensus. To rigorously account for chance agreement, we include Krippendorff’s α (Krippendorff, 2011); values range from -1 to 1; negative values indicate systematic disagreement, while values above zero indicate a degree of agreement. Furthermore, we use Spearman’s ρ to capture annotator agreement on the relative ranking of discussion quality, and the Standardized α to evaluate the overall internal consistency of the aggregated index.

CB Validity. To validate the CBs’ overall performance, we compare their final scores against proxies for discourse quality. For each discussion transcript, we calculate this proxy score by prompting an independent LLM to holistically assess sequences of turns, using a short-context sliding window. These LLM assessments are validated by human annotators to ensure they adequately capture conversational dynamics. This procedure serves as an extrinsic measure of how accurately each schema captures overall discourse quality. We define this automated proxy for every transcript, represented as a sequence of turns $D = \left( t _ { 1 } , \ldots , t _ { k } \right)$ The task scores the conversation dynamics of an n-turn exchange on a binary scale, where 0 reflects negative deliberation and 1 indicates positive deliberation (see Appendix B for the detailed instructions). We use a sliding window of size $n = 3$ and score each window independently. The final proxy score for a conversation is the average of all its triplet scores (p<sub>i</sub>):

<table><tr><td>Statistic</td><td>Mean</td><td>Median</td><td>Std.</td><td>Min</td><td>Max</td></tr><tr><td>Transcript length</td><td>9,393.7</td><td>8,414.5</td><td>6,545.9</td><td>170</td><td>27,113</td></tr><tr><td>Turns / transcript</td><td>365.4</td><td>267.5</td><td>337.8</td><td>5</td><td>1,736</td></tr><tr><td>Turn length</td><td>25.7</td><td>10.0</td><td>53.5</td><td>1</td><td>1,364</td></tr><tr><td>Speakers per transcript</td><td>16.9</td><td>17.0</td><td>7.9</td><td>2</td><td>39</td></tr><tr><td>Segments per transcript</td><td>7.0</td><td>6.0</td><td>4.5</td><td>1</td><td>19</td></tr><tr><td>Segment length</td><td>1,342.0</td><td>1,466.0</td><td>299.6</td><td>45</td><td>1,500</td></tr></table>

Table 1: Structural statistics of the 100-transcript meeting corpus. Length statistics are reported in words.

$$
P ( D ) = { \frac { 1 } { k - 2 } } \sum _ { i = 1 } ^ { k - 2 } p _ { i }
$$

## 5 Data

The data for our empirical assessment consists of over 15,000 hours of Hebrew recordings and corresponding transcripts. These transcripts are derived from more than 30,000 human-generated committee meeting protocols of the Knesset.<sup>4</sup> This corpus spans from November 2018 to January 2026, encompassing six consecutive Knesset terms. The data was curated from government-released material and made available as a Hugging Face dataset by Ivrit.ai (Marmor et al., 2025).<sup>5</sup> For this study, we restrict the corpus to legislative sessions by retaining only protocols in which every agenda item listed in the session metadata is marked as a bill.<sup>6</sup>

We randomly sample 100 protocols from this subset, stratified across Knesset terms and committees. Table 1 describes the selection’s statistics. To demonstrate the corpus’ diversity we highlight here items with notable variance:

Most Intensive Meeting: The longest session regarded the implementation of the Economic Policy for 2023 and 2024. It lasted nearly 3 hours, with 27 participants speaking, generating 1,736 turns and 27,113 words (session\_id=2204258).

Most Chair-Dominated Session: During the Economic Affairs Committee’s discussion on the bill to Increase Competition in the Broadcasting Market, the chair alone spoke 60.5% of all words (session\_id=2220120).

Most Evenly Distributed: The most balanced discussion occurred during the National-Civic Service Bill, yielding a Gini coefficient<sup>7</sup> of 0.20 among five speakers and eight participants (session\_id=2209191). Furthermore, the most broadly inclusive large discussion focused on the bill for reducing the use of disposable shopping bags, which featured 24 speakers, a Gini score of 0.40, and no single speaker accounting for more than 15.8% of the words (session\_id=2193138).

This analysis confirms our selection’s potential to examine discourse quality across diverse axes of length, dominance, and inclusivity.

## 6 Experimental Setup

Goal. We aim to empirically assess the proposed LLM-based CB generation pipeline using our proposed metrics. We evaluate each of the four CB generation strategies, comparing the top-down, bottom-up, and hybrid scenarios.

Data. Drawing on the 100 transcript selection (§5), we segment the discussion into smaller units of analysis, defined as text excerpts of up to 1,500 words and ensuring that individual turns are not split. This text length provides sufficient context for analyzing interaction dynamics and responsiveness, as the median turn length is 10 words. This results in a total of 700 segments for the full corpus, with the majority exceeding 1,466 words (Table 1).

Models. Our selection includes a proprietary LLM (Claude Sonnet 5.5<sup>8</sup>), and open-weight models: a Hebrew-centric model (DictaLM3-24B-Thinking (Shmidman et al., 2026)), and multilingual models (Gemma4-31B (Gemma Team, 2026), Qwen3.6-27B (Qwen Team, 2026), GPT-OSS-20B (OpenAI, 2025), and Mistral-Small-3.2-24B<sup>9</sup>).

For the open-weight models, we use default hyperparameters with temperature 0.6. We run each model using two seeds (11, 22) and report the mean. For Claude, due to cost considerations, we perform a single run with the temperature 1.0.

For estimating the proxy C-DQI, we utilize GPT-6-luna<sup>10</sup> with its default hyperparameters.

CB Application. We prompt each model with every category-segment pair. Apart from the Vanilla C-DQI CB, which utilizes category-specific scales, all categories are scored on a four-point scale. To create a final, representative vector of deliberation criteria for an entire discussion, the segment-level scores for each category are averaged. The final C-DQI score for a given discussion is determined according to the definitions in §4.1.

Proxy Validation. With a team of domain experts, we manually annotated 162 exchanges to serve as a human benchmark, attaining a Krippendorff’s α of 0.79 for inter-annotator agreement in a complete overlap setting where each exchange was reviewed by exactly two annotators. Exchanges on which the annotators disagreed were discarded.

We then deployed GPT-6-luna<sup>11</sup> — an independent LLM not utilized in the CBs’ reliability evaluation — to compute proxy scores across the dataset. Against the human benchmark, this LLM achieved an accuracy of 82% and a macro F1 score of 77% on a held-out test set of 50 samples (see Appendix F). Deploying an LLM as a judge not only allows us to scale the validation across the entire dataset (a total of 36,341 3-turn exchanges), avoiding costly time and resource constraints, but it also provides a consistent scoring mechanism that mitigates the noise and subjectivity inherent in human annotations. Consequently, we rely on these automated proxy values to validate our experiments. The distribution of the proxy scores across the corpus is provided in Appendix G.

Evaluation Metrics. To assess the generated CBs, we employ the metrics defined in §4.3. For CB reliability, we compute IMA scores for each category.<sup>12</sup> For validity, we calculate Spearman correlation coefficients between the mean C-DQI scores and the proxy scores.

## 7 Results and Analysis

## 7.1 CB Reliability

Table 2 presents the reliability evaluation, structured to mirror the evaluation metrics of the original DQI by Steenbergen et al. (2003). Overall, the IMA results of the CB annotation experiments show some level of agreement for every CB $( \alpha > 0 )$ $\rho > 0 . 5 )$ . The data-driven CBs are consistently the most reliable across workflows, with all three inductive CBs reaching Krippendorff α scores in the range of 0.57–0.58, compared with $\alpha = 0 . 4 6$ and $\alpha = 0 . 4 9$ for the Hybrid and Top-Down CBs, respectively. In contrast, the baseline Vanilla DQI CB yields the lowest reliability on both rank-based measures (α = 0.40, $\rho ~ = ~ 0 . 5 4$ , and Standardized $\alpha = 0 . 5 5 )$ . hitting the threshold for moderate agreement $( \alpha = 0 . 4 0 )$ (Olsson et al., 2023).

Furthermore, when evaluating category-level IMA, the categories associated with the highest values primarily relate to respect and decorum, whereas those at the lower end of the reliability spectrum correspond to justification, counterargument engagement, and emotional regulation. For example, Respect and Civility in the Top-Down CB (α = 0.79) and Procedural Decorum in the Bottom-Up-English CB $( \alpha = 0 . 8 3 )$ exhibit the strongest consensus. Conversely, Justification Level yields poor agreement in both the Vanilla DQI $( \alpha = 0 . 0 8 )$ and Top-Down (α = 0.20) CBs, and Emotional Regulation is the least reliable category of both the Bottom-Up-English (α = 0.15) and Bottom-Up-Corrected (α = 0.19) CBs (see Table 4 in Appendix C).

## 7.2 CB Validity

In Table 3, we present the Spearman correlations between the discussion-level CB scores and the C-DQI proxy scores across the 100 transcripts. All correlations exceed 0.55 and are statistically significant $( p < 1 0 ^ { - 9 } )$ . Notably, the top-down CB aligns closest to the proxy $( \rho = 0 . 6 6 )$ . This suggests that although the proxy annotators received only minimal binary guidelines, the concept of deliberative quality they apply is itself rooted in the literature, which may favor theory-led CB in the resulting rankings.

<table><tr><td>CB</td><td>Krippendorff&#x27;s α</td><td>Percent Agreement</td><td>Spearman&#x27;s  $\rho$ </td><td>Standardized α</td></tr><tr><td>Bottom-Up-Hebrew</td><td>0.58</td><td>0.20</td><td>0.75</td><td>0.78</td></tr><tr><td>Bottom-Up-English</td><td>0.57</td><td>0.20</td><td>0.72</td><td>0.73</td></tr><tr><td>Bottom-Up-Corrected</td><td>0.57</td><td>0.17</td><td>0.74</td><td>0.77</td></tr><tr><td>Top-Down</td><td>0.49</td><td>0.13</td><td>0.70</td><td>0.72</td></tr><tr><td>Hybrid</td><td>0.46</td><td>0.10</td><td>0.70</td><td>0.73</td></tr><tr><td>Vanilla DQI</td><td>0.40</td><td>0.20</td><td>0.54</td><td>0.55</td></tr></table>

Table 2: Inter-Model Agreement (IMA) across six LLMs over 100 transcripts, computed per category on transcriptlevel scores and averaged within each CB. Percent Agreement counts only exact matches, which are rare between averaged scores; Krippendorff’s α weights disagreements by size, Spearman’s ρ compares rankings, and Standardized α compares relative scores.

<table><tr><td>CB GPT-6-luna (ρ)</td></tr><tr><td>Top-Down 0.66</td></tr><tr><td>Bottom-Up-English 0.62 Bottom-Up-Corrected 0.61</td></tr><tr><td>Hybrid 0.61</td></tr><tr><td>Vanilla DQI 0.59</td></tr><tr><td>Bottom-Up-Hebrew 0.59</td></tr></table>

Table 3: Spearman correlation between each C-DQI CB’s discussion-level score and the proxy C-DQI computed by GPT-6-luna, with the default hyperparameters. CB scores are averaged over the six annotator LLMs per transcript; items are all test transcripts (n = 100). All correlations are significant at $p < 1 0 ^ { - 9 }$

## 7.3 CB Analysis

We examine the criteria of each CB to identify their similarities and divergences.

The only category shared across all CBs is Respect, which appears as three separate dimensions in the original DQI alone. According to the DQI, respect divides into respect toward groups in society, respect toward others, and respect toward contradicting arguments. A dominant category in the theory-led CBs is Constructive Politics, defined as the ability to bridge disagreements and reach practical solutions, consensus, or compromise. The most similar pair of CBs is the Hybrid and Top-Down, sharing five out of six metrics.

Notably, specific categories exist exclusively in the deductive workflows, such as participation, level and content of justification, and reciprocity. Conversely, categories appearing solely in the inductive workflows include procedural decorum, emotional regulation, and moderator efficiency.<sup>13</sup>

## 7.4 Discussion

As shown, the data-driven CBs are more stable across LLM annotation than the theory-based CBs. Specific criteria that appear in the inductive CBs such as emotional regulation and moderator efficiency are likely a product of local cultural nuances in the input data of the CB generation phase.

To observe how participation dynamics manifest, we re-examine the transcripts from §5 using the inductive CBs.<sup>14</sup> The most intensive meeting is scored 2.5 (Bottom-Up-English) and 2.8 (Bottom-Up-Corrected), while the most chair-centered meeting is scored 2.6 and 2.8, respectively. In contrast, the most evenly distributed meetings attain the same scores, 3.0 and 3.2 for the respective CBs. This suggests these CBs favor balanced participation over heavy dominance, inviting further investigation to systematically explore the interplay between structural meeting dynamics and automated discourse quality assessments.

Future directions include applying the C-DQI to analyze fluctuations in deliberative quality across the parliamentary cycle, investigating whether sessions held immediately before recesses exhibit different scores than mid-session meetings and how it systematically varies across the legislative term.

Additionally, the C-DQI enables the comparison of discourse quality analysis across different deliberative environments, such as other parliaments, alternative Israeli legislative bodies, and varying types of Knesset committee protocols.

The results validate the promise of LLM-driven bottom-up strategies for defining and applying discourse quality schemas. Serving as a proof of concept, our study motivates scaling the generation and annotation of CBs. By maneuvering LLM biases to serve as a feature that enhances robustness, rather than viewing them as unwanted artifacts, future research can uncover new analytical schemes that advance concrete legislative goals.

## 8 Conclusion

In this study, we proposed a scalable, LLM-driven methodology for generating and validating discourse quality codebooks. Specifically, we examined four approaches for schema definition, deploying an ensemble of diverse LLMs as zero-shot annotators to evaluate the resulting categories. Our results underscore the value of a bottom-up strategy for constructing reliable discourse evaluation frameworks, with validity comparable to theorybased alternatives. Taken together, this work highlights promising avenues for applying our proposed workflow rather than predefined schemas to diverse practical, political, and social scientific investigations.

## 9 Limitations

We acknowledge the limitations in our study. Although we rely on LLM-inherent biases in our methodology design, we have not examined how these artifacts manifest in annotating the CBs in practice. Additionally, regarding our choice to use segments of 1,500-words: this aims to capture reciprocity across multiple turns, yet larger segments risk reducing interpretability and introducing noise into holistic scoring. This highlights an inherent tension: higher-level abstractions capture broader dynamics but are harder to evaluate against specific criteria. Despite this, holistic evaluation remains valuable for assessing overarching processes and solutions. In future work, we will explore this in more depth.

Regarding the Top-Down CB, the instruction ‘to develop a discourse quality index’ may have biased the model toward the DQI and Habermasian discourse ethics. Further experiments could explore alternative phrasing for the prompt.

Furthermore, the CBs are evaluated separately as a proof of concept; we did not examine their combinations and other balances besides the hybrid. Finally, for validity, we rely on statistical correlations. Although a positive correlation indicates the direction of agreement, it doesn’t directly translate to accuracy. We acknowledge this as a limitation. Future research may address this by validating the distributional similarity between CB scores and the proxy C-DQI, and can benefit from a manual gold standard as well.

## 10 Acknowledgments

We wish to thank Nadav Oved and Noam Golan from our annotators team for their dedicated research assistance. This research was supported by the Bar-Ilan University Data Science and Artificial Intelligence Institute, and by the Rector’s grant which funded the 2026 interdisciplinary research group on computational analysis of the legislative process. Furthermore, this research was funded by the Israel Science Foundation (grant No. 670/23 and grant No. 2352/26), a KAMIN grant from the Israel Innovation Authority (IIA), and a VATAT grant from the Israeli Planning and Budgeting Committee (PBC), for which we are grateful. We finally wish to thank Google for their generous grant award supporting our lab efforts on AI for Science.

## References

Anthropic. 2026. System card: Claude Sonnet 5.5. https://www.anthropic.com/ claude-sonnet-5-5-system-card. Accessed: 2026-10-06.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Maike Behrendt, Stefan Sylvius Wagner, Marc Ziegele, et al. 2024. Aqua – combining experts’ and nonexperts’ views to assess deliberation quality in online discussions using llms. In Proceedings of the First Workshop on Language-Driven Deliberation Technology (DELITE) @ LREC-COLING 2024. ELRA and ICCL.

Shai Berkovitz, Amit Mazuz, and Michael Fire. 2023. Open framework for analyzing public parliaments data. Journal ofBig Data, 10(1).

Mitchell Bosley. 2025. Towards qualitative measurement at scale: A prompt-engineering framework for large-scale analysis of deliberative quality in parliamentary debates. Journal ofPolitical Institutions and Political Economy, 6:1–29.

Megan A Brown, Shubham Atreja, Libby Hemphill, and Patrick Y Wu. 2025. Evaluating how llm annotations represent diverse views on contentious topics. arXiv preprint arXiv:2503.23243.

André Bächtiger, Simon Niemeyer, Michael Neblo, Marco R. Steenbergen, and Jürg Steiner. 2010. Disentangling diversity in deliberative democracy: Com-

peting theories, their blind spots and complementarities\*. Journal ofPolitical Philosophy, 18:32–63.

Xunyu Chen, Yuanxia Li, One-Ki Daniel Lee, and Amita Goyal Chin. 2026. Large language modelempowered debate outcome prediction via argument quality assessment. Group Decision and Negotiation, 35(2).

Selen A. Ercan, Hans Asenbaum, Nicole Curato, and Ricardo Fabrino Mendonça. 2022. Researching Deliberative Democracy: Methods and Approaches. Oxford University Press.

Katharina Esau, Dannica Fleuß, and Sarah-Michelle Nienhaus. 2021. Different arenas, different deliberative quality? using a systemic framework to evaluate online deliberation on immigration policy in germany. Policy & Internet, 13(1):86–112.

Eleonore Fournier-Tombs and Giovanna Di Marzo Serugendo. 2020. Delibanalysis: Understanding the quality of online political discourse with machine learning. Journal ofInformation Science, 46(6):810–822.

Eleonore Fournier-Tombs and Michael K MacKenzie. 2021. Big data and democratic speech: Predicting deliberative quality using machine learning techniques. Methodological Innovations, 14(2):20597991211010416.

Dennis Friess and Christiane Eilders. 2015. A systematic review of online deliberation research. Policy & Internet, 7(3):319–339.

Gemma Team. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Todd Graham. 2010. The use of expressives in online political talk: Impeding or facilitating the normative goals of deliberation? In Electronic Participation, pages 26–41, Berlin, Heidelberg. Springer Berlin Heidelberg.

Arpit Gupta. 2026. literature-survey-skill: A claude code skill for structured literature surveys.

Joseph Heath. 2019. Discourse Ethics, page 104–109. Cambridge University Press.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Timothée Wang, Thomas Lacroix, and William El Sayed. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Klaus Krippendorff. 2011. Computing krippendorff’s alpha-reliability.

Sanjay Kumar, Jane Suiter, and Luca Longo. 2025. Advancing deliberative discourse measurement: The intersection with computational abstract argumentation in discourse quality evaluations. Systems, 13(3):204.

Pierre-Carl Langlais, Annina Claesson, Manon Berriche, Andreï Mogoutov, and Jean-Philippe Cointet. 2025. Legibot: A comprehensive theory-driven annotated corpus and slm for french legislative data.

Shahar Levy, Eliya Habba, Reshef Mintz, Barak Raveh, Renana Keydar, and Gabriel Stanovsky. 2026. Schematiq: From research question to structured data through interactive schema discovery. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 3: System Demonstrations), pages 220–230.

Jane Mansbridge, James Bohman, Simone Chambers, et al. 2012. A systemic approach to deliberative democracy. In Jane Mansbridge and John Parkinson, editors, Deliberative Systems: Deliberative Democracy at the Large Scale, Theories of Institutional Design. Cambridge University Press.

Yanir Marmor, Yair Lifshitz, Yoad Snapir, and Kinneret Misgav. 2025. Building an accurate open-source hebrew asr system through crowdsourcing. In Proceedings of the Annual Conference of the International Speech Communication Association, INTER-SPEECH, pages 723–727.

Guy Mor-Lan, Omer Goldman, Matan Eyal, Adi Mayrav Gilady, Sivan Eiger, Idan Szpektor, Avinatan Hassidim, Yossi Matias, and Reut Tsarfaty. 2026. Location not found: Exposing implicit local and global biases in multilingual llms. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 29130–29150.

Benjamin Newman, Yoonjoo Lee, Aakanksha Naik, Pao Siangliulue, Raymond Fok, Juho Kim, Daniel S Weld, Joseph Chee Chang, and Kyle Lo. 2024. Arxivdigestables: Synthesizing scientific literature into tables using language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9612–9631.

Henrik Olsson, Kristina Åhlund, Joakim Alfredsson, David Andersson, Anne-Marie Boström, Susanne Guidetti, Mattias Prytz, and Niklas Ekerstad. 2023. Cross-cultural adaption and inter-rater reliability of the swedish version of the updated clinical frailty scale 2.0. BMC Geriatrics, 23.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

Vishakh Padmakumar, Joseph Chee Chang, Kyle Lo, Doug Downey, and Aakanksha Naik. 2025. Intentaware schema generation and refinement for literature review tables. EMNLP (Findings), pages 23450– 23472.

Qwen Team. 2026. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Julia Romberg, Maximilian Maurer, Henning Wachsmuth, and Gabriella Lapesa. 2025. Towards a perspectivist turn in argument quality assessment.

In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics.

Sameer Sadruddin, Jennifer D’Souza, Eleni Poupaki, Alex Watkins, Hamed Babaei Giglou, Anisa Rula, Bora Karasulu, Sören Auer, Adrie Mackus, and Erwin Kessels. 2025. Llms4schemadiscovery: A human-in-the-loop workflow for scientific schema mining with large language models. In European Semantic Web Conference, pages 244–261. Springer.

Sameer Sadruddin, Jennifer D’Souza, Eleni Poupaki, Alex Watkins, Bora Karasulu, Sören Auer, Adrie Mackus, and Erwin Kessels. 2026. schemaminerpro: Agentic ai for ontology grounding over llm-discovered scientific schemas in a human-in-the-loop workflow. Semantic Web, 17(3):22104968261431521.

Hope Schroeder, Deb Roy, and Jad Kabbara. 2024. Fora: A corpus and framework for the study of facilitated dialogue. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13985–14001, Bangkok, Thailand. Association for Computational Linguistics.

Ignacio-Jesús Serrano-Contreras, Javier García-Marín, and Óscar G. Luengo. 2020. Measuring online political dialogue: Does polarization trigger more deliberation? Media and Communication, 8(4):63–72.

Shaltiel Shmidman, Avi Shmidman, Amir DN Cohen, and Moshe Koppel. 2026. Dicta-lm 3.0: Advancing the frontier of hebrew sovereign llms. arXiv preprint arXiv:2602.02104.

Marco R. Steenbergen, André Bächtiger, Markus Spörndli, and Jürg Steiner. 2003. Measuring political deliberation: A discourse quality index. Comparative European Politics, 1(1):21–48.

Marco R. Steenbergen, André Bächtiger, Jürg Steiner, and Markus Spörndli. 2005. Appendix: Discourse quality index (dqi): Instructions for coders. In Deliberative Politics in Action: Analyzing Parliamentary Discourse, Theories of Institutional Design. Cambridge University Press.

Jennifer Stromer-Galley. 2007. Measuring deliberation’s content: A coding scheme. Journal ofDeliberative Democracy, 3(1).

Axel Tschentscher, André Bächtiger, Jürg Steiner, and Marco Steenbergen. 2010. Deliberation in parliaments: Research objectives and preliminary results of the bern center for interdisciplinary deliberation studies (bids). Legisprudence, 4(1):13–34.

Eva Maria Vecchi, Neele Falk, Carlotta Quensel, Iman Jundi, and Gabriella Lapesa. 2025. Perspectivemod: A perspectivist resource for deliberative moderation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

## A Code-Book Generation Prompts

## Bottom-Up CB. - English

Attached are 10 raw transcripts from committee meetings in the Israeli parliament, the Knesset. Using a bottom-up approach, analyze the discourse of these specific transcripts to define metrics that evaluate the quality of discourse in general committee meetings.

Provide:

1. A list of the metrics.

2. The definition of each metric.

## Bottom-Up CB. - Hebrew

<sup>לפני</sup>K ק<sup>בצי</sup> <sup>ט</sup>ק<sup>סט</sup> <sup>שמכיליM</sup> <sup>פרוטו</sup>ק<sup>וליM</sup> <sup>מוועדות</sup> הכנסת.   
<sub>הגדר</sub> <sub>מטריקות</sub> ל<sub>הערכת</sub> <sub>איכות</sub> <sub>השיח</sub> <sub>ש</sub>ל <sub>כ</sub>ל וועדה .   
<sub>תציג</sub> <sub>את</sub> <sub>המטריקות</sub> <sub>ואת</sub> <sub>ההגדרה</sub> <sub>ש</sub>ל <sub>כ</sub>ל <sub>אחת</sub> מהM.

## Top-Down CB.

Based on the frameworks in these papers, define metrics to evaluate the quality of discourse in committee transcripts from the Israeli parliament, the Knesset.

Provide:

1. A list of the metrics.

2. The definition of each metric.

## Hybrid CB.

Attached are 20 files: Half are academic literature on democratic deliberation and discourse quality (Theoretical Foundation). The other half are raw committee meeting transcripts from the Israeli parliament, the Knesset (Empirical Context).

Using a combined theory and bottom-up approach, synthesize the academic frameworks and analyze the discourse of these specific transcripts to define metrics that evaluate the quality of discourse in general committee meetings.

Provide:

1. A list of the metrics.

2. The definition of each metric.

## B Proxy Generation Prompts

For the proxy generation prompts see Figure 3 and Figure 4 for the system and user prompts.

## C Extreme Category IMA

Table 4 describes the categories in each CB that exhibit the highest and lowest IMA scores.

## D Constructing the Bottom-Up-Corrected Codebook

We constructed the corrected inductive codebook by merging the English and Hebrew codebooks, retaining every distinct metric from each. Where the two versions overlapped, we consolidated them into a single metric; Hebrew’s Decorum and Tone category was split between Procedural Decorum and Respectful Engagement. We removed most illustrative examples (“such as. . . ”) to avoid biasing human or model annotators toward specific cases. We also adopted the bullet structure of the Hebrew codebook and standardized it so that every metric has two labeled poles, “High” and “Low”. Where a source codebook described only one pole, we defined the missing one to be the parallel opposite of the existing one. Finally, for the Procedural Decorum metric, we replaced “unanimous agreement” from the English codebook with “respectful disagreement”, since agreement is not a condition for good decorum.

## E CodeBooks

## Vanilla DQI

## 0. Demand

A proposal about what should or should not be decided or done. A demand may support changing, adopting, rejecting, or preserving a policy or course of action.

## 1. Participation

Definition: Evaluates whether a participant can complete a relevant contribution without being prevented from speaking by disruption or formal intervention.

## Coding Categories / Scale:

Code 0 (Participation Interrupted): The speaker explicitly indicates that an interruption is disturbing or preventing the contribution, or the contribution is terminated by a formal decision. Do not assign this code merely because the allocated speaking time has expired.

Code 1 (Normal Participation Possible): The participant is able to make the relevant contribution under ordinary speaking conditions.

## 2. Level of Justification for Demands

Definition: Measures the completeness of the reasons supplied for a demand. The substantive truth or persuasiveness of the argument is not judged. A complete justification contains (1) a conclusion about what should or should not be done, (2) a supporting reason, and (3) an explicit or unmistakably implied inferential link between the reason and conclusion.

## Coding Categories / Scale:

Code 0 (No Justification): A demand or evaluation is stated, but no supporting reason is provided.

Code 1 (Inferior Justification): A supporting consideration is mentioned, but the link between the reason and the demand is incomplete or unclear. A conclusion supported only by illustration also receives Code 1.

Code 2 (Qualified Justification): At least one complete justification connects a demand to a reason through an explicit or clearly implied inferential link.

Code 3 (Sophisticated Justification): At least two distinct complete justifications are supplied, whether for the same demand or for different demands within the speech act.

Decision Rule: Assign the highest level whose requirements are fully met. An implicit linkage counts only when its intended meaning would be beyond reasonable doubt to participants in the debate.

<table><tr><td rowspan="2">Codebook</td><td colspan="2">Most reliable feature</td><td colspan="2">Least reliable feature</td></tr><tr><td>Feature</td><td>α</td><td>Feature</td><td>α</td></tr><tr><td>Vanilla DQI</td><td>Respect toward Groups</td><td>0.60</td><td>Justification Level</td><td>0.08</td></tr><tr><td>Top-Down</td><td>Respect and Civility</td><td>0.79</td><td>Justification Level</td><td>0.20</td></tr><tr><td>Bottom-Up-English</td><td>Procedural Decorum</td><td>0.83</td><td>Emotional Regulation</td><td>0.15</td></tr><tr><td>Bottom-Up-Corrected</td><td>Procedural Decorum</td><td>0.80</td><td>Emotional Regulation</td><td>0.19</td></tr><tr><td>Bottom-Up-Hebrew</td><td>Decorum and Tone</td><td>0.80</td><td>Chair Management Efficiency</td><td>0.36</td></tr><tr><td>Hybrid</td><td>Civility and Conflict Mgmt.</td><td>0.69</td><td>Moderator Efficacy</td><td>0.22</td></tr></table>

Table 4: Most and least reliable features per C-DQI codebook, determined by highest and lowest Krippendorff’s α scores.

## 3. Content of Justification for Demands

Definition: Identifies whose interests are explicitly invoked when a speaker justifies a demand: a constituency or particular group, society collectively, or disadvantaged groups under a fairnessbased conception of the common good.

## Coding Categories:

Code 0 (Constituency or Group Interest): At least one explicit appeal is made to the interest of a particular constituency, organization, sector, party, region, or social group.

Code 1 (Neutral Statement): There is no explicit reference to a constituency or group interest and no explicit appeal to either form of the common good.

Code 2 (Common-Good Orientation): The speaker explicitly justifies a demand using at least one of the following forms of commongood reasoning:

Utilitarian or Collective Common Good: An appeal to the best outcome for the greatest number, society as a whole, or a larger collective such as Europe or humanity.

Difference-Principle Common Good: An appeal to protecting or helping the least advantaged, poorest, most discriminated against, or otherwise vulnerable members of society, including children and future generations, or an appeal to fairness and justice in this sense.

Decision Rule: Assign the highest level whose requirements are fully met.

4. Respect toward Groups to Be Helped (Em-

## pathy)

Definition: Evaluates how the speaker characterizes the people or groups who would be affected or helped by the demand.

## Coding Categories / Scale:

Code 0 (No Respect): At least one negative or degrading statement is made about a relevant group, and no positive statement is made. A single negative statement is sufficient.

Code 1 (Neutral): Neither negative nor explicitly positive statements are made about the relevant groups.

Code 2 (Explicit Respect): At least one explicitly positive, valuing, empathetic, or dignity-affirming statement is made about a relevant group. This code still applies when negative statements also occur.

Priority Rule: Explicit respect takes priority over negativity: mixed positive and negative statements receive Code 2.

## 5. Respect toward the Demands of Others

Definition: Evaluates whether a speaker degrades, treats neutrally, or explicitly values demands advanced by other participants.

## Coding Categories / Scale:

Code 0 (No Respect): At least one negative or degrading statement is made toward another demand, or toward the person or group advancing it, and no positive statement is made.

Code 1 (Neutral): Other demands receive neither negative nor explicitly positive treatment.

Code 2 (Explicit Respect): At least one positive statement values any aspect of another participant’s demand. This code still applies when negative statements also occur.

N/A (Not Applicable): Only one demand is on the agenda and the speaker supports it; consequently, there is no distinct competing demand to evaluate.

Priority Rule: Explicit respect takes priority over negativity: mixed positive and negative statements receive Code 2. Criticism of another demand may additionally be coded under Level and Content of Justification when it functions as part of the speaker’s own argument.

## 6. Respect toward Counterarguments

Definition: Evaluates how a speaker responds to arguments that contradict the speaker’s conclusion concerning a demand. The category concerns engagement with opposing reasons, not merely interpersonal politeness.

## Coding Categories / Scale:

Code 0 (Counterarguments Ignored): One or more counterarguments are already available, but the speaker does not address any of them.

Code 1 (Included but Explicitly Degraded): Counterarguments are acknowledged, but at least one is degraded, mocked, or attacked, and none is explicitly valued. Attacks on the people or groups advancing the counterarguments also count.

Code 2 (Included Neutrally): Counterarguments are acknowledged or addressed without either negative degradation or explicit positive valuation.

Code 3 (Included and Explicitly Valued): At least one aspect of at least one counterargument is explicitly recognized as valid, correct, useful, or worthy of consideration. This code still applies when other negative statements occur.

N/A (Not Applicable): No counterargument has yet been introduced and the speaker does not anticipate one.

Totality Rule: When several counterarguments exist, assign one code to the speaker’s treatment of the set as a whole. Explicit valuation takes priority over negative treatment; without valuation, any degradation leads to Code 1. An anticipated counterargument should receive an actual code rather than N/A.

## 7. Constructive Politics

Definition: Measures whether the speaker remains fixed in a position or advances a mediating proposal, distinguishing proposals that address the present agenda from alternatives that belong to another policy agenda.

## Coding Categories / Scale:

Code 0 (Positional Politics): The speaker maintains a position without offering a mediating or compromise proposal.

Code 1 (Alternative Proposal): The speaker advances a proposal that could mediate the broader conflict but does not fit the current agenda or resolve the decision presently under debate. It would need to be considered under another agenda.

Code 2 (Mediating Proposal): The speaker offers a concrete compromise or accommodation within the current agenda that attempts to bridge competing concerns or modify the proposal under consideration.

Coding Note: A proposal is not mediating merely because it differs from the current proposal. It must attempt to accommodate competing concerns; the distinction between Codes 1 and 2 depends on whether it can be acted upon within the present agenda.

## Bottom-Up CB. - English

## Metric 1: Procedural Decorum

This metric evaluates the extent to which participants follow parliamentary rules, wait for their turn to speak, and avoid shouting or talking over one another. In some committees, the chairperson must repeatedly threaten to remove members to maintain order. Conversely, other meetings proceed with unanimous agreement, polite praises, and uninterrupted presentations.

## Metric 2: Topic Adherence

This metric measures the ratio of substantive discussion regarding the agenda versus unrelated political grandstanding or personal attacks. High topic adherence is observed when members strictly debate specific legal definitions, such as defining security offenses versus treason. Low adherence occurs when speaking time meant for legislative reservations is entirely used to attack political opponents on unrelated national issues.

Metric 3: Constructive Problem-Solving This metric assesses the ability of committee members and government officials to bridge disagreements and reach practical legislative compromises. This metric is positive when members negotiate temporary orders to test a law’s impact before making it permanent. It is also evident when legal advisors and ministries successfully negotiate timelines for administrative tasks, such as extending a deadline from three months to seven months to accommodate workload constraints.

## Metric 4: Respectful Engagement

This metric analyzes the tone and language used when interacting with civil servants, academic experts, and members of the public. Discourse quality drops when elected officials use dismissive language, such as calling opposing arguments “bullshit”. It also drops when there is friction between politicians and civil servants, such as when an official allegedly calls a member a “populist”. High quality is shown when chairpersons actively listen to and validate the emotional testimonies of local mayors.

## Metric 5: Emotional Regulation

This metric examines how the committee handles highly charged emotional topics or national tragedies while conducting legislative business. Poor emotional regulation is visible when procedural duties, like budget discussions, clash with days of national mourning, leading to accusations of callousness and “disgusting” behavior between members. High emotional tension is also present when bereaved parents confront lawmakers directly during security-related debates.

## Bottom-Up CB. - Hebrew

## 1. ענייניות ומיקוד בדיוN Relevance( and Focus)

הגדרה:מדדהבוחNעדכמההדובריM(חבריהכנסת והמוזמניM) מתמקדיM בנושא המקצועי <sup>ל</sup>שמו התכנסה <sub>הוועדה,</sub>ל<sub>עומתג</sub>ל<sub>ישה</sub>ל<sub>פי</sub>ל<sub>יבסטריM,הע</sub>ל<sub>אתנושאיM</sub> שאינM קשוריM ,או נאומיM פו<sup>ל</sup>יטייM.

• איכות שיח גבוהה: התמקדות בסעיפי החוק, <sup>ניסוחיM</sup> <sup>משפטייM</sup> <sup>ופתרו</sup>N <sup>חסמיM</sup> <sup>בירו</sup>ק<sup>רטייM</sup>.

•איכותשיחנמוכה:שימושבזמNהוועדה<sup>ל</sup>נאומיM פוליטייMארוכיMשאינM<sub>ק</sub>שוריMלסעי<sub>P</sub>הנדו<sub>N</sub>כלל,כפי <sup>שנית</sup>N<sup>לראותבהגשתעשרותהסתייגויותונימו</sup>קN<sup>עלידי</sup> <sub>תקיפתהשר</sub>ל<sub>ביטחוN</sub>ל<sub>אומיבנושאיMשוניM</sub>ל<sub>ח</sub>ל<sub>וטיN</sub> )כגוN תאונות דרכיM או גזענות (במה<sup>ל</sup>K דיוN ע<sup>ל</sup> חוק <sup>ל</sup>שכתעורכיהדיN.

## 2. תרבות הדיוN וסגנוN ההתבטאות (Decorum and Tone)

הגדרה: מדד המעריK את רמת הכבוד ההדדי, <sup>הה</sup>ק<sup>שבה</sup> <sup>ואופ</sup>N <sup>ההתבטאות</sup> <sup>בי</sup>N <sup>המשתתפיM</sup> .<sup>המדד</sup> <sup>נבח</sup>N על ידי ספירת <sub>ק</sub>ריאות בינייM ,התפרצויות לדברי אחריM ,הרמת קו<sup>ל</sup> ,ושימוש בשפה בוטה או תקיפות אישיות.

• איכות שיח גבוהה: הקשבה מ<sup>ל</sup>אה <sup>ל</sup>דובריM ,הבעת הערכה הדדית גM ביN קוא<sup>ל</sup>יציה <sup>ל</sup>אופוזיציה ,ודיוN ש<sup>ל</sup>יו ומקצועי.

• איכות שיח נמוכה: התפרצויות תכופות ,קושי <sup>ל</sup>הש- ליMמשפט,ושימושבביטוייMפוגענייMוהאשמותאישיות )כגוN חי<sup>ל</sup>ופי ההאשמות "תומכי טרור "ו"-גיס חמישי "א<sup>ל</sup> מו<sup>ל</sup> "מפעי<sup>ל</sup> טרור "ו"-פאשיסטיM.")

## 3.יעי<sup>ל</sup>ותניהו<sup>ל</sup>הישיבהofEfficiency( Chair Management)

הגדרה: מדד הבוחN את יכו<sup>ל</sup>תו ש<sup>ל</sup> יושב-ראש הוועדה ל<sub>ש</sub>ל<sub>וטבדיוN,</sub>ל<sub>נתבאותו</sub>ל<sub>הכרעותמעשיות,</sub>ל<sub>קטוע</sub> <sub>הפרעות</sub> <sub>ו</sub>ל<sub>מנוע</sub> <sub>מהדיוN</sub> ל<sub>צאת</sub> <sub>מכ</sub>לל <sub>ש</sub>ל<sub>יטה</sub> <sub>,תוK</sub> <sub>ש-</sub> <sub>מירהע</sub>ל<sub>הוגנותכ</sub>ל<sub>פיהמשתתפיM.</sub>

•ניהו<sup>ל</sup>הח<sup>ל</sup>טיואקטיבי:יכו<sup>ל</sup>תש<sup>ל</sup>היושב-ראש <sup>ל</sup>קטועויכוחיMבירוקרטייMארוכיM,<sup>ל</sup>הנחותאתגורמי <sub>המקצוע</sub>ל<sub>הגיע</sub>ל<sub>פשרותפרקטיות,ו</sub>ל<sub>קב</sub>ל<sub>הח</sub>ל<sub>טותע</sub>ל <sup>המ</sup>ק<sup>וMתו</sup>K<sup>נזיפהבגורמיMשלאמבצעיMאתעבודתM</sup> כראוי)כמוהדרישה<sup>ל</sup>קנוסרשויותמקומיותששו<sup>ל</sup>חות קנסות שווא <sup>ל</sup>אמבו<sup>ל</sup>נסיM.)

• קושי בניהו<sup>ל</sup> הדיוN: אובדN ש<sup>ל</sup>יטה ע<sup>ל</sup> סדר הדו- בריM ,ניסיונות חוזריM ונשניM( ו<sup>ל</sup>עיתיM כוש<sup>ל</sup>יM) <sup>ל</sup>השקיט <sub>ק</sub>ריאות בינייM ,ואיומיM תכופיM בהוצאת משתתפיM ללא הצלחה.

## 4. מקצועיות והסתמכות ע<sup>ל</sup> נתוניM -Pro( fessionalism and Data Reliance)

הגדרה: מדד הבודק את מידת ההישענות ש<sup>ל</sup> משתתפי הוועדהע<sup>ל</sup>עובדות,נתוניM,מחקריM(כגוNמרכזהמידע והמחקר ש<sup>ל</sup> הכנסת ,(והקשבה <sup>ל</sup>ייעוZ המשפטי ו<sup>ל</sup>דרגיM המקצועייMבמשרדיהממש<sup>ל</sup>ה.

•<sup>איכותשיחגבוהה</sup>:<sup>ניתוחמדו</sup>ק<sup>ד</sup>ק<sup>שלהחו</sup>ק<sup>סעי</sup>P <sup>אחר</sup> <sup>סעי</sup>P <sup>יחד</sup> <sup>עM</sup> <sup>הייעו</sup>Z <sup>המשפטי</sup> ,<sup>תו</sup>K <sup>הבנת</sup> <sup>ההש-</sup> ל<sub>כותהמשפטיותש</sub>ל<sub>כ</sub>ל<sub>מי</sub>ל<sub>הבחוק)כגוNההבד</sub>ל<sub>ביN</sub> "חוקיחו<sup>ל</sup>"<sup>ל</sup>"-הגב<sup>ל</sup>ות<sup>ל</sup>איחו<sup>ל</sup>ו.("

•פשרהמקצועיתמבוססתנתוניM:קביעת<sup>ל</sup>וחות זמניMריאלייM(כגו<sub>N</sub>ויכוחעלמספרהחודשיMהנדרש ל<sub>וועדתשחרוריM</sub>ל<sub>עבורע</sub>ל<sub>120תיקיMפ</sub>ל<sub>י</sub>ל<sub>ייM)מו</sub>ל גורמי המקצוע בהנה<sup>ל</sup>ת בתי המשפט ומשרד המשפטיM.

## 5. אק<sup>ל</sup>יM רגשי וגי<sup>ל</sup>וי אמפתיה -Emo( tional Climate and Empathy)

הגדרה:מדדהבוחNאתיכו<sup>ל</sup>תהוועדה<sup>ל</sup>הכי<sup>ל</sup>ו<sup>ל</sup>הגיב ל<sub>מצביMרגשייMמורכביM,נוכחותNש</sub>ל<sub>משפחותשכו</sub>ל<sub>ות</sub> ומשפחותחטופיM,וההתאמהש<sup>ל</sup>קיוMהדיוN<sup>ל</sup>מציאות החיצונית )אירועיM ביטחונייM קשיM.)

•חוסרה<sup>ל</sup>ימהרגשית/מתיחותקשה:קיוMדיוN "רגי<sup>ל</sup>"שמעוררזעMבקרבמשתתפיMע<sup>ל</sup>רקעאירועיM <sup>ביטחונייM</sup>ק<sup>שיMבזמ</sup>N<sup>אמת</sup>,<sup>מהשמוביללפיצו</sup>Z<sup>הדיו</sup>N ו<sup>ל</sup>הטחתהאשמותע<sup>ל</sup>אטימותוניתוק")ער<sup>ל</sup>ות<sup>ל</sup>ב.("

• <sup>מת</sup>N <sup>מ</sup>ק<sup>וM</sup> <sup>לכאב</sup>: <sup>עצירת</sup> <sup>שט</sup>P <sup>הח</sup>ק<sup>י</sup>ק<sup>ה</sup> <sup>כדי</sup> ל<sub>אפשר</sub>ל<sub>משפחותחטופיMומשפחותשכו</sub>ל<sub>ות</sub>ל<sub>השמיע</sub> <sup>את</sup> ק<sup>ול</sup>N <sup>ואת</sup> <sup>זע</sup>ק<sup>ת</sup>N ,<sup>והבעת</sup> <sup>הזדהות</sup> <sup>ואמפתיה</sup> <sup>מצד</sup> חבריהכנסתוהיושב-ראש<sup>ל</sup>מצוקתM.

## Bottom-up CB - Corrected

## Metric 1: Procedural Decorum

Evaluates the extent to which participants follow parliamentary rules, wait for their turn to speak, and avoid shouting or talking over one another.

High Decorum: Observed in meetings that proceed with polite exchanges, uninterrupted presentations, and respectful disagreement.

Low Decorum: Associated with committees where the chairperson must repeatedly threaten to remove members to maintain order.

## Metric 2: Topic Adherence

Measures the ratio of substantive discussion regarding the agenda versus unrelated political grandstanding or personal attacks.

High Adherence: Observed when the majority of participants strictly stick to the discussion agenda, debating specific outcomes and legal definitions.

Low Adherence: Observed when discussions frequently go off-topic, or when speaking time meant for legislative reservations is entirely used to attack political opponents on unrelated national issues.

## Metric 3: Constructive Problem-Solving

Assesses the ability of committee members and government officials to bridge disagreements and reach practical legislative compromises.

High Constructiveness: Indicated by members negotiating temporary orders to test a law’s impact before making it permanent, or when legal advisors and ministries successfully negotiate timelines for administrative tasks.

Low Constructiveness: Indicated by members adopting zero-sum ideological positions that lead to legislative deadlocks, or government ministries engaging in jurisdictional turf wars and blame-shifting instead of finding actionable paths forward.

## Metric 4: Respectful Engagement

Analyzes the tone and language used when interacting with civil servants, academic experts, and members of the public.

High Respect: Demonstrated when participants utilize active listening, validate each other’s opinions, and maintain a professional tone.

Low Respect: Characterized by friction between participants and the use of dismissive or vulgar language (e.g., calling opposing arguments “bullshit”).

## Metric 5: Emotional Regulation

Examines how the committee handles highly charged emotional topics or national tragedies while conducting legislative business.

High Regulation: Demonstrated when lawmakers maintain composure, empathy, and respect during emotionally charged testimonies. The committee successfully balances sensitivity to national tragedies with its legislative duties without allowing the discussion to devolve into chaos.

Low Regulation: Demonstrated when routine business clashes with sensitive events, resulting in insults, accusations of callousness, or uncontrolled tension when lawmakers are confronted by grieving citizens.

Metric 6: Meeting Management Effectiveness Examines the committee chairperson’s ability to control the discussion, steer it toward practical decisions, cut off disruptions, and prevent the debate from spiraling out of control, all while maintaining fairness.

High Effectiveness: The chairperson successfully cuts short lengthy bureaucratic arguments, guides professional entities to reach practical compromises, makes on-the-spot decisions, and holds parties accountable for failing to perform their duties.

Low Effectiveness: Characterized by a loss of control over the speaking order, repeated attempts to quiet heckling, and frequent but unsuccessful threats to remove participants from the room.

## Metric 7: Professionalism and Reliance on Data

Assesses the extent to which committee participants rely on facts, data, and research, and their attentiveness to professional echelons and legal counsel.

High Professionalism: Characterized by a meticulous, section-by-section analysis of the law alongside legal counsel, demonstrating an understanding of the legal implications of the legislation. Compromises set realistic timetables in coordination with professional authorities.

Low Professionalism: Characterized by participants ignoring research and data, disregarding professional or legal advice, and arguing over legislation without understanding its technical or practical implications.

## Top-Down CB.

## 1. Level of Justification (Argumentative Rationality)

Theoretical Origin: Discourse Quality Index (Steenbergen et al., 2003); Stromer-Galley (2007); Friess & Eilders (2015).

Definition: Evaluates the logical structure and completeness of reasons provided by Members of the Knesset (MKs) or invited witnesses when advancing a policy proposal, amendment, or claim. Coding Categories / Scale:

Level 0 (No Justification): A demand/claim is stated without any supporting reason.

Level 1 (Inferior Justification): A reason is stated, but the logical link between the demand and the reason is incomplete or relies strictly on illustration.

Level 2 (Qualified Justification): At least one complete, explicit, or clearly implied inferential link connects the demand to a supporting reason.

Level 3 (Sophisticated Justification): At least two distinct, complete inferential links are provided (either multiple complete reasons for one demand or reasons spanning multiple demands).

## 2. Content of Justification (Common Good Orientation)

Theoretical Origin: Steenbergen et al. (2003);   
Habermas (1996); Cohen (1989).

Definition: Assesses whether a speaker justifies claims solely through narrow sectoral/party/constituency interests or frames arguments in terms of broader societal benefit.

Coding Categories / Scale:

Group/Constituency Interest: Explicit appeals to narrow political, regional, or constituency benefits.

Neutral Statement: No explicit reference to either narrow group interests or the broader common good.

Common Good (Utilitarian): Explicit appeal to the best outcome for Israeli society as a whole or the greatest benefit for the majority.

Common Good (Difference Principle): Explicit appeal to protecting or helping the least advantaged, vulnerable, or marginalized groups in society (Rawlsian difference principle).

## 3. Sourcing and Evidence Base

Theoretical Origin: Stromer-Galley (2007);   
Friess & Eilders (2015).

Definition: Measures the extent to which participants support their positions using verifiable external references, expert testimonies, or official materials rather than unsupported assertions.

Coding Categories:

Official/Committee Documents: Reference to Knesset Research and Information Center (MMM) reports, ministry briefings, or draft bill texts.

External/Expert Sources: Reference to academic research, statistical data, court rulings, or international benchmarks.

Mass Media: Reference to news reports, journalism, or public media broadcasts.

Personal Experience / Narrative: Reference to first-hand accounts, constituent stories, or personal anecdotes.

No Source: Claim made without grounding in external or experiential sources.

## 4. Interactivity and Reciprocity

Theoretical Origin: Stromer-Galley (2007);   
Friess & Eilders (2015).

Definition: Evaluates whether committee dialogue constitutes a genuine, reciprocal exchange of ideas where speakers listen and respond to one another, or consists of disconnected monologues

(“soap-boxing”).

Coding Indicators:

Formal Interactivity: Whether a speech turn explicitly addresses a prior speaker or turn.

Substantive Interactivity: Whether a speaker directly addresses, rebuts, clarifies, or builds upon the specific arguments made by a prior participant.

Information-Seeking / Questioning: Presence of genuine questions aimed at eliciting clarification or facts from witnesses, government officials, or fellow MKs (distinguished from rhetorical questions).

Clarification (Self/Other): Speech acts that explicitly clarify a speaker’s own prior position or summarize another participant’s argument to verify understanding.

## 5. Respect and Civility toward Actors and Demands

Theoretical Origin: Steenbergen et al. (2003);   
Young (1996); Friess & Eilders (2015).

Definition: Measures the interpersonal tone and level of mutual recognition accorded to other participants, social groups, and policy proposals during committee proceedings.

Coding Categories / Scale:

Respect toward Groups: Evaluates whether social, ethnic, or political groups affected by the bill are degraded (0), treated neutrally (1), or explicitly valued/respected (2).

Respect toward Demands: Evaluates whether policy proposals/amendments raised by fellow participants are degraded/insulted (0), treated neutrally (1), explicitly valued (2), or agreed with (3).

Civility / Disruption: Tracks presence of personal insults, ad hominem attacks, or unprocedural interruptions.

## 6. Engagement with Counterarguments

Theoretical Origin: Steenbergen et al. (2003);   
Bächtiger et al. (2010).

Definition: Assesses how speakers handle opposing viewpoints or counterarguments raised by coalition/opposition members or invited experts. Coding Categories / Scale:

Level 0 (Ignored): Counterarguments exist on the table, but the speaker completely ignores them.

Level 1 (Included but Degraded): Counterarguments are acknowledged but explicitly degraded, mocked, or dismissed without substantive engagement.

Level 2 (Included Neutrally): Counterarguments are acknowledged and restated neutrally without personal or thematic degradation.

Level 3 (Included and Valued): Counterarguments are acknowledged and explicitly praised or treated as valid considerations, even if ultimately rejected or integrated.

## 7. Constructive Politics and Compromise Orientation

Theoretical Origin: Steenbergen et al. (2003); Bächtiger et al. (2010); Friess & Eilders (2015). Definition: Measures whether committee participants remain locked in rigid, zero-sum partisan stances or actively attempt to build consensus and formulate acceptable compromises.

Coding Categories / Scale:

Positional Politics (Level 0): Speakers rigidly maintain initial party lines without attempt at compromise, accommodation, or reconciliation.

Alternative Proposal (Level 1): A mediating proposal is offered that belongs to a different agenda or jurisdiction, failing to resolve the immediate conflict on the current draft text.

Mediating Proposal (Level 2): A speaker advances a concrete, integrative compromise or amendment that directly fits the current agenda and seeks to bridge coalition-opposition or stakeholder divides.

## 8. Topic Coherence and Agenda Focus

Theoretical Origin: Stromer-Galley (2007);   
Friess & Eilders (2015).

Definition: Evaluates whether committee discussion stays focused on the legislative agenda (structuring topic) or drifts into uncoordinated, off-topic partisan disputes (interactional topic drift).

Coding Indicators:

On-Topic (Structuring): Statements directly addressing the bill, policy options, or specific committee agenda items.

On-Topic (Related Problem): Statements addressing the broader policy issue (e.g., educational quality, defense budget) without directly addressing the specific legislative choices at hand.

Off-Topic / Drift: Statements digressing into unrelated political scandals, electoral grandstanding, or personal disputes unlinked to the agenda.

## 9. Participatory Equality and Vocal Distribution

Theoretical Origin: Stromer-Galley (2007); Knight & Johnson (1997); Friess & Eilders (2015).

Definition: Measures whether speaking opportunities and floor time are equitably distributed among participants or monopolized by committee chairs, coalition members, or dominant speakers.

## Coding Indicators:

Turn & Word-Count Distribution: Quantitative balance of speech turns and word volume across coalition vs. opposition MKs, and committee members vs. invited external experts/citizens.

Procedural Equal Access: Absence of arbitrary silencing, unequal time enforcement by the chair, or procedural coercion that prevents backbenchers or opposition members from presenting amendments.

## 10. Inclusion of Alternative Communicative Modes

Theoretical Origin: Young (1996); Bächtiger et al. (2010) (Type II Deliberation).

Definition: Captures non-canonical speech forms—such as narrative/storytelling, rhetoric, and formal greeting—that complement formal argumentation by bridging cultural, ideological, or experiential divides in parliamentary committees.

## Coding Categories:

Narrative / Testimony: Use of situated personal stories, constituent accounts, or lived experiences to demonstrate legislative impact or illuminate marginalized perspectives.

Rhetoric / Expressive Language: Use of emotional resonance, figurative language, or metaphor to contextualize policy claims and engage listeners.

Greeting and Deference: Use of formal politeness, honorifics, or conciliatory gestures that acknowledge the standing of opponents and establish deliberative trust.

## Hybrid CB.

## Metric 1: Interactivity and Engagement

Definition: This metric measures the degree to which participants directly respond to the claims of others, rather than engaging in parallel monologues. In a healthy deliberation, participants listen, ask genuine questions to solicit information, and build upon previous speakers’ points. Empirically, Knesset committees exhibit high levels of interactivity, but it frequently devolves into disruptive interruptions or cross-talk (e.g., shouting over one another during debates about terror and academia). Therefore, this metric evaluates both positive interactivity (substantive responses and reciprocal dialogue) and negative interactivity (disruptive interruptions that derail the speaker’s thought process).

## Metric 2: Level of Justification and Epistemic Reliance

Definition: This metric assesses whether participants support their demands with evidence, logic, and expert knowledge, rather than mere assertion. High-quality discourse includes “sophisticated justifications” where multiple complete inferences are made. Furthermore, in a systemic deliberative approach, good discourse relies on the appropriate integration of expert knowledge. In the Knesset transcripts, this is observed through the heavy reliance on legal advisors (e.g., from the Ministry of Justice or the committee’s own counsel) to define legal boundaries, correct factual inaccuracies, and structure the consequences of legislation. The metric evaluates how well politicians incorporate this expert data into their arguments versus dismissing it for purely partisan reasons.

## Metric 3: Civility, Respect, and Conflict Management

Definition: This evaluates the extent to which participants treat opposing groups, demands, and counterarguments with respect, which is a fundamental prerequisite for serious listening. In ideal deliberation, counterarguments are acknowledged and valued. The empirical Knesset data shows that committee meetings often struggle with this metric, featuring ad hominem attacks, sarcasm, and severe disrespect (e.g., calling opposing members a “fifth column” or comparing political opponents to terrorist organizations). This metric measures the frequency of degrading language versus respectful disagreement, tracking whether the discourse maintains a baseline of mutual recognition even during fierce ideological disputes.

## Metric 4: Inclusiveness and Communicative Pluralism

Definition: This metric assesses whether diverse voices, particularly those of marginalized groups and non-politicians, are given an equal and fair opportunity to influence the debate. Drawing on “Type II” deliberative theory, it recognizes that rational, dispassionate argumentation can sometimes exclude the less privileged, meaning that rhetoric, emotion, and storytelling are valid and necessary forms of democratic communication. Empirically, this is vital in the Knesset, where bereaved parents use highly emotional, personal narratives to urge legislative action, and minority Arab MKs express the lived reality of their constituencies regarding soaring crime rates. This metric evaluates how well the committee makes space for, and seriously considers, these diverse perspectives and expressive styles.

## Metric 5: Constructive Politics and Problem-Solving

Definition: This metric measures the committee’s ability to move beyond entrenched, positional politics toward generating alternative or mediating proposals that aim for consensus or practical solutions. True deliberation in a legislative setting must ultimately aim to reach a binding, workable decision. The Knesset transcripts frequently display this metric in action: despite intense partisan arguments, members, legal advisors, and government representatives engage in real-time bargaining to tweak the wording of a law, adjust implementation timelines, or compromise on budget allocations. The metric captures the density of these constructive, solutionoriented exchanges.

Metric 6: Moderator (Chairperson) Efficacy Definition: This metric evaluates the behavior and effectiveness of the committee chairperson in facilitating the deliberative process. A good moderator acts as a “promoter of deliberation” by managing the agenda, ensuring marginalized voices are heard, bringing the discussion back on topic, and intervening in conflicts. In the Knesset, the Chairperson (HaYo"r - originally in Hebrew letters) plays a highly active role in maintaining order, threatening to remove disruptive members, pushing for compromise, and clarifying procedural next steps. This metric tracks the chair’s success in balancing the enforcement of rules with the allowance of free and open debate.

## Metric 7: Common Good vs. Sectorial Framing

Definition: This metric evaluates the content of the justifications used by participants, specifically whether they appeal to narrow group interests or to the broader common good. Appeals to the common good can be utilitarian (the greatest good for the greatest number) or based on the difference principle (helping the least advantaged). In the empirical data, we see constant tension between sectorial lobbies (e.g., the agricultural lobby fighting to protect local honey production, or specific municipal authorities demanding resources) and the overarching national interest (e.g., lowering the cost of living or ensuring national security). This metric tracks how frequently and successfully participants link their specific, narrow claims to generalizable principles that benefit the entire society.

## F Proxy Judge Model Selection

Human benchmark. Two annotators independently rated 162 three-turn exchanges as high (1) or low (0) deliberative quality; scoring an interannotator agreement of 0.79 Krippendorff’s α. The benchmark consists of the 148 exchanges on which both annotators agreed (40 positive, 108 negative); the 14 disagreements were excluded. The judge received the same instrument as the annotators: the coder definitions and clarifications, the session metadata, and the context turns surrounding the exchange. The clarifications were refined on a separate set of 21 expert-reviewed exchanges, none of which appear in the benchmark.

Model selection. We evaluated seven OpenAI model settings (GPT-5.6-luna, GPT-6-luna, GPT-6-luna-pro (reasoning effort set to ‘pro’), GPT-5.6-sol, GPT-6-sol, GPT-6.1-sol and GPT-6-astra), none of which were used in the CBs’ reliability evaluation. The benchmark was split, stratified by label, into a validation set of 98 exchanges (26 positive) and a held-out test set of 50 exchanges (14 positive). We ran a stratified 5-fold crossvalidation and selected the model with the highest mean macro F1. Validation runs were issued through the OpenRouter chat API on October 12, 2026; after validation, the selected model scored the full corpus through the OpenRouter Batch API with the identical prompt. On the held-out test set, GPT-6-luna achieved an accuracy of 82% and a macro F1 of 77%, against the human labels.

## G Proxy C-DQI Distributions

In §7.2, we evaluate the validity of the generated CBs by correlating their scores with an extrinsic proxy measure of discourse quality. To ensure this proxy serves as a valid and discriminatory baseline, we analyzed the distribution of its scores across the 100 evaluation transcripts.

Figure 5 illustrates the distribution of the proxy C-DQI scores as evaluated by GPT-6-luna. The scores exhibit a wide variance across the corpus, confirming that the proxy captures a broad spectrum of deliberative quality rather than collapsing into a narrow, uniform range. This distribution underscores the structural and topical diversity of the selected Knesset committee transcripts (as highlighted in Section 5) and provides a robust numerical foundation for the rank correlation analyses presented in Table 3.

## H Per-Model Correlation with the Proxy

Table 5 reports Spearman’s ρ between each $\mathbf { C B } ^ { \prime } \mathbf { s }$ discussion-level score and the proxy judge (GPT-6-luna), broken out per annotator model rather than averaged. The top-down CB aligns stronger with the proxies, for four out of the six models. The top-Down and Vanilla DQI CBs exhibit the widest spread across models $( \Delta ~ = ~ 0 . 1 4 )$ whereas the Bottom-Up CBs are the most stable $( \Delta \in [ 0 . 0 9 , 0 . 1 ] )$ ).

<table><tr><td>CB</td><td>Claude</td><td>Gemma4</td><td>DictaLM</td><td>Qwen3.6</td><td>GPT-OSS</td><td>Mistral</td></tr><tr><td>Bottom-Up-Corrected</td><td>0.57</td><td>0.55</td><td>0.64</td><td>0.59</td><td>0.58</td><td>0.58</td></tr><tr><td>Bottom-Up-English</td><td>0.59</td><td>0.56</td><td>0.63</td><td>0.54</td><td>0.58</td><td>0.64</td></tr><tr><td>Bottom-Up-Hebrew</td><td>0.57</td><td>0.53</td><td>0.54</td><td>0.55</td><td>0.53</td><td>0.62</td></tr><tr><td>Top-Down</td><td>0.55</td><td>0.64</td><td>0.59</td><td>0.69</td><td>0.65</td><td>0.64</td></tr><tr><td>Hybrid</td><td>0.54</td><td>0.56</td><td>0.52</td><td>0.61</td><td>0.59</td><td>0.63</td></tr><tr><td>Vanilla DQI</td><td>0.48</td><td>0.53</td><td>0.41</td><td>0.55</td><td>0.49</td><td>0.53</td></tr></table>

Table 5: Spearman’s ρ between each C-DQI CB’s discussion-level score and the proxy (GPT-6-luna), broken out by annotator model rather than averaged across models. All correlations significant at $p < 1 0 ^ { - 4 }$

![](images/943e0480bf5f5a2cd638a670ec3e0a6ea948f69524d6e5061801f1577a6a5f52.jpg)  
Figure 3: System prompt used by the LLM judge.

![](images/46448de516f51a36bb87d70aeccb8951b2e446986048521a1077b6c8e4374a5a.jpg)  
Figure 4: User prompt used by the LLM judge (continued on the next page).

User prompt (2/2)   
• Merely reporting an administrative position, explaining implementation, announcing a vote, expressing agreement,   
or closing the discussion does not qualify as substantive deliberation unless it directly addresses a contested policy   
issue with reasons or justification. A respectful but purely procedural or informational exchange therefore receives 0   
• Distinguish why from how. Explaining why a bill, provision, or request is needed (its purpose, the problem it   
solves, the consequences of not acting) is justification and can support a 1. Describing how a provision will be   
implemented, what an agency will check, or what should be calculated or requested from another body (action   
items) is informational and scores 0.   
• An explanation that resolves a concern or confusion raised by another participant, and is acknowledged by them, is   
responsive engagement and can support a 1.   
Output format   
Return only a valid JSON object with this exact structure:   
{   
"rationale": "<1-2 sentences justifying the score>",   
"score": <1 or 0>   
}   
Metadata   
Knesset {number} | {committee} | {date} | {topics discussed}   
Context before the exchange   
{context before}   
Exchange to rate   
{exchange}   
Context after the exchange   
{context after}  
Figure 4: User prompt used by the LLM judge (continued). Text in {blue} is filled in per exchange.

![](images/02a1cf42f81c139485ab9915fadd77e3f5aa42b14163853c18d430180c926212.jpg)  
Figure 5: Distribution of the proxy C-DQI across the 100 evaluation transcripts, computed by averaging each transcript’s 3-turn exchange scores (36,341 in total), judged by GPT-6-luna.