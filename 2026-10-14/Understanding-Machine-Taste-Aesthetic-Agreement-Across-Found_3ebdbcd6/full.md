# Understanding Machine Taste: Aesthetic Agreement Across Foundation Models in K-pop

Kaishen Wang, Yucheng Pan, Xuehao Cui, Hanling Tian, Heng Huang

University of Maryland, College Park

{kaishen, ycpan, cedrus, hanling, heng}@umd.edu

## Abstract

As foundation models enter creative industries, understanding their aesthetic preferences becomes an important step toward AI-assisted creative decision making. In this work, we investigate how foundation models evaluate creative content and characterize their “machine taste" using K-pop songs and music videos as a testbed. We formulate aesthetic evaluation as a pairwise ranking problem and systematically examine pairwise inter-model agreement, group-level consensus, and crossmodal preference alignment. Our results reveal stronger pairwise inter-model agreement and group-level consensus in visual evaluation than in audio evaluation, while preferences across the two modalities remain weakly aligned. We further examine how visual consensus rankings relate to real-world popularity using historical YouTube viewership data, finding positive but varying correlations across artist groups. These findings provide initial insights into machine aesthetic preferences and lay the groundwork for future AI-assisted creative decision making that considers creative content, artist identity, audience preferences, and market dynamics.

## 1 Introduction

Recent foundation models (Ma et al., 2024; Yang et al., 2025; Xu et al., 2025; Yang et al., 2026; Ghosh et al., 2026; Ye et al., 2026; Wang et al., 2026) have become increasingly capable across language, vision, and audio, and are gradually being integrated into real-world workflows beyond conventional benchmark-oriented tasks. This transition is particularly important in domains where decisions are not determined by a single correct answer. In creative industries, for example, choices such as selecting a song, developing a visual concept, or evaluating creative content are inherently subjective and often depend on aesthetic judgment rather than objective correctness (Zhao et al., 2025; Fein et al., 2026; Chen et al., 2026).

The entertainment industry provides a timely example of this transition, as major companies are increasingly exploring AI applications in their creative and production workflows. SM Entertainment (SM Entertainment, 2026) has identified AI as a key component of its SM NEXT 3.0 strategy, aiming to improve production efficiency and support more sophisticated fan- and market-informed decision making. HYBE (HYBE, 2026) has expanded its AI capabilities through Supertone, which develops AI-based audio technologies for generating realistic singing and acting voices. JYP Entertainment (JYP Entertainment, 2025) has similarly expanded its AI-related initiatives, including plans to develop an AI artist as part of its future content portfolio. These developments reflect the growing role of AI in creative industries and point toward broader opportunities for AI-assisted creative and strategic decision making. An important first step toward this vision is to understand how foundation models themselves evaluate creative content and what aesthetic preferences they exhibit.

Unlike factual question answering or mathematical reasoning, aesthetic evaluation does not admit a unique ground-truth answer (Marco et al., 2025). While prior studies have investigated the aesthetic evaluation capabilities and biases of foundation models (Li et al., 2024; Yao and Zhuang, 2025; Li et al., 2025), understanding how these models evaluate creative content remains challenging without a universally accepted reference for aesthetic quality. In this context, examining aesthetic preference agreement provides a useful starting point for investigating machine taste, allowing us to examine whether shared preferences emerge between individual models, among models evaluating the same artist group, and across evaluation modalities.

To investigate machine taste through aesthetic preference agreement, we use K-pop as a testbed that offers diverse creative content across both audio and visual media. Songs and music videos span a wide range of musical styles, production choices, and visual concepts, providing rich material for examining aesthetic preferences. Moreover, the availability of audio tracks and corresponding music videos enables systematic comparisons of model preferences within and across modalities.

![](images/0c792a2e9f12cf921e70354af5dc5f840eeb4212436716c1f1ab555265c4e326.jpg)  
Figure 1: Overview of our study of machine taste in K-pop. ALMs and VLMs perform pairwise aesthetic evaluations of songs and music videos, respectively. The resulting rankings are analyzed to examine inter-model agreement, group-level consensus, and cross-modal preference alignment.

In this work, we address three research questions, as illustrated in Figure 1: (1) how strongly individual models agree in their aesthetic rankings across all artist groups, (2) whether a broader consensus emerges among all evaluated models within each artist group, and (3) whether aesthetic preferences align across audio and visual modalities. To investigate these questions, we formulate aesthetic evaluation as a pairwise ranking problem, considering both audio-based song ranking and visual-only music-video ranking. We analyze the resulting preferences using Spearman rank correlation and Kendall's coefficient of concordance. Our results show stronger agreement between individual vision-language models (VLMs) and consistently higher group-level consensus in visual evaluation across all 14 artist groups, compared with audio evaluation by audio-language models (ALMs). Meanwhile, audio and visual preferences remain weakly aligned. We also examine how model-derived visual preferences relate to realworld popularity using historical YouTube viewership data, finding positive but varying correlations across artist groups.

More broadly, this study represents a first step toward AI-assisted decision making in the Kpop industry. Understanding machine aesthetic preferences can help inform future systems that jointly consider creative content, artist identity, audience preferences, and market dynamics to support broader creative and strategic decisions. Our contributions are summarized as follows:

• We introduce a systematic evaluation framework for investigating machine aesthetic preferences through pairwise ranking, inter-model agreement, group-level consensus, and crossmodal preference alignment.

• We conduct a large-scale empirical study of foundation models using K-pop songs and music videos, revealing weaker and more variable aesthetic agreement in audio evaluation, stronger and more consistent agreement in visual evaluation, and limited alignment between the two modalities.

• We further examine the relationship between model-derived visual preferences and realworld popularity using historical YouTube viewership data, finding positive but varying correlations across artist groups.

## 2 Related Work

## 2.1 Multimodal Models for Creative Content

Recent foundation models have extended multimodal understanding from text and images to audio and video, enabling direct perception and reasoning over increasingly complex forms of creative content (Ma et al., 2024; Xu et al., 2025; Yang et al., 2026; Ghosh et al., 2026). Audio-language models (ALMs) such as Qwen3-Omni (Xu et al., 2025) and MOSS (Yang et al., 2026) can directly process acoustic signals, including music, while recent vision-language models (VLMs) such as Qwen3-VL (Bai et al., 2025), InternVL3.5 (Wang et al., 2025), GLM-4.6V-Flash (Hong et al., 2025), MiniCPM-V-4.5 (Yu et al., 2026), and Qwen3.5 (Qwen Team, 2026), provide strong capabilities for understanding videos. Existing evaluations of these models primarily focus on perceptual understanding, grounding, question answering, and multimodal reasoning. In contrast, our work studies a different aspect of multimodal intelligence: whether independently developed foundation models form similar subjective preferences when evaluating the same creative content.

## 2.2 Aesthetic Evaluation and Model Agreement

Recent studies have investigated the ability of foundation models to evaluate subjective creative content, including visual aesthetic sensitivity, alignment with human aesthetic preferences, and the evaluation of creative writing (Yao and Zhuang, 2025; Li et al., 2025; Feng et al., 2026; Kim and Oh, 2025; Jiang et al., 2026). These studies suggest that foundation models can capture non-trivial aesthetic signals, but their judgments remain imperfect and may differ substantially from human or expert preferences. Other studies have examined the reliability of LLMs and MLLMs as evaluators (Chen et al., 2024; Ye et al., 2025; Shi et al., 2025; Feng et al., 2025), identifying systematic biases such as position bias, authority and presentation effects, and model-specific preferences. These studies also highlight inconsistencies in model judgments when evaluating the same content. Other studies have explored ways to improve the consistency of modelbased evaluation through stronger reasoning or specialized evaluation models (Saha et al., 2025; Xiong et al., 2025, 2026). These efforts primarily focus on making model judgments more reliable and better aligned with human preferences, rather than understanding the disagreement itself. In contrast, our work focuses on inter-model aesthetic agreement rather than evaluation accuracy. We examine whether independently developed foundation models share similar aesthetic preferences and whether these preferences remain consistent across audio and visual modalities.

## 3 Method

Our goal is to investigate how foundation models evaluate creative content through their expressed aesthetic preferences. Rather than relying on absolute quality scores, which may provide limited discrimination when models assign similar scores to different creative items, we formulate aesthetic evaluation as a pairwise ranking problem to directly capture models' relative preferences. Models compare candidate items, and their pairwise preferences are aggregated into model-specific rankings. We then examine how consistently models rank creative content within each modality and whether their preferences align across audio and visual evaluation.

## 3.1 Pairwise Aesthetic Ranking

Consider an evaluation set

$$
{ \mathcal { C } } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { N } \} ,\tag{1}
$$

where each $c _ { i }$ denotes a creative item to be evaluated. Depending on the evaluation setting, a candidate corresponds to either an audio track or a music video.

For each unordered pair $\{ c _ { i } , c _ { j } \}$ , we evaluate model m under both presentation orders:

$$
{ \big ( } c _ { i } , c _ { j } { \big ) } \qquad { \mathrm { a n d } } \qquad { \big ( } c _ { j } , c _ { i } { \big ) } .\tag{2}
$$

The two evaluations are treated as independent comparisons. Each comparison requires a binary preference decision, and the model must select exactly one of the two candidates. The preferred candidate receives one point. Evaluating both presentation orders reduces the influence of positional preference while retaining the model's individual decision in each comparison.

Let

$$
y _ { m } ( c _ { i } , c _ { j } ) \in \{ 0 , 1 \}\tag{3}
$$

denote the model judgment when $c _ { i }$ is presented before $c _ { j }$ , where $y _ { m } ( c _ { i } , c _ { j } ) = 1$ indicates that $c _ { i }$ is preferred over $c _ { j }$ . The total score assigned to candidate $c _ { i }$ is

$$
s _ { m } ( c _ { i } ) = \sum _ { j \neq i } \left[ y _ { m } ( c _ { i } , c _ { j } ) + 1 - y _ { m } ( c _ { j } , c _ { i } ) \right] .\tag{4}
$$

The final ranking produced by model m is obtained by sorting all candidates according to their

aggregate preference scores:

$$
R _ { m } = \operatorname { R a n k } \left( s _ { m } ( c _ { 1 } ) , s _ { m } ( c _ { 2 } ) , \ldots , s _ { m } ( c _ { N } ) \right) .\tag{5}
$$

This procedure provides a common ranking framework across models while avoiding direct comparison of model-specific absolute scoring scales.

## 3.2 Aesthetic Preference Agreement

Given the rankings produced by different models, we analyze aesthetic agreement from two complementary perspectives: how consistently individual models rank creative content across all artist groups, and whether a broader consensus emerges among models within each group.

Pairwise Inter-model Agreement. For two models $m _ { i }$ and $m _ { j }$ , we measure the similarity between their induced rankings using Spearman's rank correlation:

$$
{ \rho } _ { i , j } ^ { ( g ) } = \rho _ { \mathrm { S } } \left( R _ { m _ { i } , g } , R _ { m _ { j } , g } \right) ,\tag{6}
$$

where $g$ indexes an artist group. Given G artist groups, we report the mean pairwise agreement across groups:

$$
\bar { \rho } _ { i , j } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \rho _ { i , j } ^ { ( g ) } .\tag{7}
$$

This metric characterizes how similarly a particular pair of models ranks creative content across different artist groups.

Group-level Model Consensus. While pairwise correlation compares individual model pairs, we also examine whether all evaluated models exhibit a shared preference ranking within each artist group. For each group $^ { g , }$ we therefore compute Kendall's coefficient of concordance over the rankings produced by all M models:

$$
W _ { g } = W \left( R _ { m _ { 1 } , g } , R _ { m _ { 2 } , g } , \ldots , R _ { m _ { M } , g } \right) ,\tag{8}
$$

where $W _ { g } ~ \in ~ [ 0 , 1 ]$ . A larger value indicates stronger collective agreement among models, with $W _ { g } = 1$ corresponding to identical rankings.

## 3.3 Modality-specific and Cross-modal Preference Analysis

We apply the pairwise ranking framework in Section 3.1 to audio-based song evaluation and musicvideo-based evaluation, both conducted within individual artist groups. The resulting rankings are analyzed using the agreement metrics defined in Section 3.2. We further examine preference alignment between audio and visual evaluation. In all evaluations, models are not provided with song titles, popularity statistics, chart performance, or other external market information. Detailed prompts are presented in Appendix D.

<table><tr><td>Group</td><td># Audio</td><td># Video</td></tr><tr><td>RED VELVET</td><td>30</td><td>28</td></tr><tr><td>TWICE</td><td>30</td><td>30</td></tr><tr><td>BLACKPINK</td><td>30</td><td>30</td></tr><tr><td>I-DLE</td><td>30</td><td>25</td></tr><tr><td>ITZY</td><td>30</td><td>26</td></tr><tr><td>AESPA</td><td>30</td><td>23</td></tr><tr><td>IVE</td><td>30</td><td>23</td></tr><tr><td>NMIXX</td><td>30</td><td>16</td></tr><tr><td>LE SSERAFIM</td><td>30</td><td>17</td></tr><tr><td>NEWJEANS</td><td>20</td><td>16</td></tr><tr><td>BABYMONSTER</td><td>24</td><td>16</td></tr><tr><td>ILLIT</td><td>30</td><td>12</td></tr><tr><td>MEOVV</td><td>13</td><td>7</td></tr><tr><td>HEARTS2HEARTS</td><td>16</td><td>9</td></tr><tr><td>Total</td><td>373</td><td>278</td></tr></table>

Table 1: Statistics of the K-pop evaluation dataset across 14 artist groups. The audio collection includes up to 30 songs per group, while the video collection contains the subset with eligible official music videos.

Audio-based Preference Analysis. For each Kpop artist group $^ { g , }$ we construct a song set ${ \cal S } _ { g } =$ $\{ s _ { 1 } , \ldots , s _ { N _ { g } } \}$ and evaluate audio-language models (ALMs) using only the audio content. Models compare songs based on overall musical quality, including melody, vocal performance, production, arrangement, and overall cohesion. We obtain the ranking $R _ { m , g } ^ { \mathrm { a u d i o } }$ for each ALM m following Section 3.1, and analyze agreement among ALMs using the metrics in Section 3.2.

Visual Preference Analysis. For each artist group g, we construct a set of official music videos $\mathcal { V } _ { g } ~ = ~ \{ v _ { 1 } , . . . , v _ { M _ { g } } \}$ Vision-language models (VLMs) receive only the visual stream of each music video and are not given access to the corresponding audio track. Models compare videos based on visual and creative execution, including visual concept, cinematography, styling, editing, and overall visual coherence. Following the same procedure, we obtain the ranking $R _ { m , g } ^ { \mathrm { M V } }$ for each VLM m and analyze agreement among VLMs.

Cross-modal Preference Alignment. We examine whether audio and visual aesthetic preferences align for the same underlying songs. For each artist group g, we restrict the analysis to songs shared by the audio and music-video evaluation sets. We recompute the ALM rankings using only pairwise decisions between songs in this shared subset, ensuring that both modalities are compared over the same candidates. At the model level, we compute Spearman rank correlation between each ALM and VLM ranking. At the consensus level, we construct separate audio and visual rankings by averaging candidate ranks across ALMs and VLMs, respectively, and re-ranking the candidates. We then compute Spearman rank correlation between the two consensus rankings.

<table><tr><td>Group</td><td>Kendall&#x27;s W</td><td>p-value</td></tr><tr><td>RED VELVET</td><td>0.334</td><td>&lt; 0.001</td></tr><tr><td>TWICE</td><td>0.240</td><td>0.043</td></tr><tr><td>BLACKPINK</td><td>0.319</td><td>&lt; 0.001</td></tr><tr><td>I-DLE</td><td>0.356</td><td>&lt; 0.001</td></tr><tr><td>ITZY</td><td>0.272</td><td>0.010</td></tr><tr><td>AESPA</td><td>0.263</td><td>0.015</td></tr><tr><td>IVE</td><td>0.419</td><td>&lt; 0.001</td></tr><tr><td>NMIXX</td><td>0.297</td><td>0.003</td></tr><tr><td>LE SSERAFIM</td><td>0.314</td><td>&lt; 0.001</td></tr><tr><td>NEWJEANS</td><td>0.375</td><td>&lt; 0.001</td></tr><tr><td>BABYMONSTER</td><td>0.226</td><td>0.104</td></tr><tr><td>ILLIT</td><td>0.427</td><td>&lt; 0.001</td></tr><tr><td>MEOVV</td><td>0.191</td><td>0.321</td></tr><tr><td>HEARTS2HEARTS</td><td>0.137</td><td>0.675</td></tr></table>

Table 2: Group-level aesthetic consensus among ALMs. Kendall's coefficient of concordance W is computed from the rankings of all six model configurations for each artist group in the audio-based evaluation.

## 4 Experiments

## 4.1 Experimental Setup

Dataset Construction. We construct our evaluation dataset from 14 K-pop girl groups spanning different debut periods and artist identities. For each group, we first collect songs with official music videos, with at most 30 videos retained per group. The corresponding audio tracks of these songs are included in the audio evaluation set, ensuring that the audio and video settings share a substantial set of common content.

Because many groups have fewer than 30 eligible official music videos, we further expand the audio collection with additional popular songs from the same artist, up to a maximum of 30 audio tracks per group. For groups whose available discography contains fewer than 30 eligible songs, we retain all available tracks. As a result, the video set forms a subset of the corresponding audio collection for each artist, while the audio setting generally contains a broader selection of songs.

The resulting dataset contains 373 audio tracks and 278 music videos. Detailed statistics for each artist group are reported in Table 1, with the complete candidate lists provided in Appendix B.

![](images/380996b278ce31809567e6cc7ae96d5e18db19acd6e3f8e11d31af32dfd15f82.jpg)  
Figure 2: Pairwise ranking agreement between ALMs. Each cell reports the mean Spearman rank correlation across 14 artist groups. Inst and Think denote Instruct and Thinking configurations, respectively; Qwen refers to Qwen3-Omni-30B-A3B.

Audio Models. For audio-based evaluation, we consider six audio-language models (ALMs) from two audio-capable foundation model families: MOSS-4B-Instruct (Yang et al., 2026), MOSS-4B-Thinking (Yang et al., 2026), MOSS-8B-Instruct (Yang et al., 2026), MOSS-8B-Thinking (Yang et al., 2026), Qwen3-Omni-30B-A3B-Instruct (Xu et al., 2025), and Qwen3-Omni-30B-A3B-Thinking (Xu et al., 2025). We select these models based on both their strong audio understanding capabilities and their support for processing multiple audio inputs within a single context, which enables direct pairwise comparison between two songs.

Video Models. For music-video evaluation, we consider 28 model configurations from five visionlanguage model (VLM) families. Specifically, we evaluate Qwen3.5 (Qwen Team, 2026) at four model scales (2B, 4B, 9B, and 27B), InternVL3.5 (Wang et al., 2025) at six scales (1B, 2B, 4B, 8B, 14B, and 38B), GLM-4.6V-Flash (Hong et al., 2025), and MiniCPM-V-4.5 (Yu et al., 2026) For Qwen3.5, InternVL3.5, GLM-4.6V-Flash, and MiniCPM-V-4.5, we evaluate both non-thinking and thinking modes whenever supported. We additionally include Qwen3-VL-Instruct (Bai et al., 2025) at four model scales (2B, 4B, 8B, and 32B). This diverse model set allows us to examine aesthetic agreement across model families, model scales, and inference modes.

<table><tr><td>Group</td><td>Kendall&#x27;s W</td><td>p-value</td></tr><tr><td>RED VELVET</td><td>0.656</td><td>&lt; 0.001</td></tr><tr><td>TWICE</td><td>0.649</td><td>&lt; 0.001</td></tr><tr><td>BLACKPINK</td><td>0.769</td><td>&lt; 0.001</td></tr><tr><td>I-DLE</td><td>0.681</td><td>&lt; 0.001</td></tr><tr><td>ITZY</td><td>0.516</td><td>&lt; 0.001</td></tr><tr><td>AESPA</td><td>0.584</td><td>&lt; 0.001</td></tr><tr><td>IVE</td><td>0.692</td><td>&lt; 0.001</td></tr><tr><td>NMIXX</td><td>0.555</td><td>&lt; 0.001</td></tr><tr><td>LE SSERAFIM</td><td>0.503</td><td>&lt; 0.001</td></tr><tr><td>NEWJEANS</td><td>0.536</td><td>&lt; 0.001</td></tr><tr><td>BABYMONSTER</td><td>0.506</td><td>&lt; 0.001</td></tr><tr><td>ILLIT</td><td>0.470</td><td>&lt; 0.001</td></tr><tr><td>MEOVV</td><td>0.532</td><td>&lt; 0.001</td></tr><tr><td>HEARTS2HEARTS</td><td>0.557</td><td>&lt; 0.001</td></tr></table>

Table 3: Group-level aesthetic consensus among VLMs. Kendall's coefficient of concordance W is computed from the rankings of all 28 model configurations for each artist group in the music-video setting.

Evaluation Protocol. For both modalities, comparisons are performed only among candidates belonging to the same artist group. ALMs receive only the audio content, while VLMs receive only the visual content of the corresponding music videos. Song titles, chart performance, popularity statistics, and other external metadata are excluded from the model input.

All candidate pairs are evaluated under both presentation orders following the pairwise ranking procedure described in Section 3.1. To reduce the impact of randomness, each pairwise comparison is independently repeated five times under each presentation order. The resulting preferences are aggregated to construct within-group rankings, which are analyzed using the aesthetic agreement metrics defined in Section 3.2. For cross-modal analysis, we restrict the comparison to songs shared by both modalities and recompute the audio rankings using pairwise decisions within this shared subset, following Section 3.3. We use the same evaluation prompt and decoding configuration across candidates within each model setting.

## 4.2 Experimental Results

Audio-based Aesthetic Agreement. We first examine whether ALMs exhibit similar aesthetic preferences when evaluating songs based solely on their audio content. Figure 2 presents the pairwise Spearman correlations averaged across 14 artist groups. Overall, agreement is limited, with the strongest model pair reaching a correlation of only 0.526, while several pairs exhibit near-zero or negative correlations. Notably, Qwen3-Omni-Thinking shows consistently weak agreement with the other configurations, illustrating substantial variation in musical preferences across models.

<table><tr><td>Group</td><td>Spearman ρ</td><td>p-value</td></tr><tr><td>Red Velvet</td><td>0.023</td><td>0.905</td></tr><tr><td>TWICE</td><td>0.475</td><td>0.007</td></tr><tr><td>BLACKPINK</td><td>0.178</td><td>0.344</td></tr><tr><td>I-DLE</td><td>0.377</td><td>0.064</td></tr><tr><td>ITZY</td><td>0.356</td><td>0.074</td></tr><tr><td>AESPA</td><td>0.006</td><td>0.977</td></tr><tr><td>IVE</td><td>0.279</td><td>0.193</td></tr><tr><td>NMIXX</td><td>-0.219</td><td>0.413</td></tr><tr><td>LE SSERAFIM</td><td>-0.010</td><td>0.972</td></tr><tr><td>NewJeans</td><td>0.177</td><td>0.512</td></tr><tr><td>BABYMONSTER</td><td>0.399</td><td>0.128</td></tr><tr><td>ILLIT</td><td>-0.084</td><td>0.793</td></tr><tr><td>MEOVV</td><td>0.107</td><td>0.840</td></tr><tr><td>HEARTS2HEARTS</td><td>0.167</td><td>0.680</td></tr></table>

Table 4: Cross-modal consensus alignment between audio and visual evaluations. Spearman rank correlation is computed between the audio and visual consensus rankings for each artist group, using only songs shared by both modalities.

We further assess group-level consensus using Kendall's W (shown in Table 2). Concordance ranges from 0.137 to 0.427, with 11 of the 14 groups showing statistically significant agreement $( p \ : < \ : 0 . 0 5 )$ . Although the overall consensus remains relatively modest, these results indicate that shared ranking tendencies can still emerge within individual artist groups despite considerable differences between models.

Visual Aesthetic Agreement. We next examine aesthetic agreement among VLMs in the musicvideo setting. As shown in Figure 3, pairwise Spearman correlations are substantially higher than those observed in audio-based evaluation. Strong positive correlations appear not only within the same model family but also across independently developed families, suggesting broadly shared visual preferences. Meanwhile, the visible block structure indicates that model family and inference configuration still influence the resulting rankings.

This stronger agreement is also reflected in the group-level analysis (shown in Table 3), where Kendall's W ranges from 0.470 to 0.769. All 14 artist groups exhibit statistically significant concordance $( p < 0 . 0 0 1 )$ , with the highest agreement observed for BLACKPINK (W = 0.769) and the lowest for ILLIT (W = 0.470). Together, these findings indicate that visual aesthetic preferences are more consistently shared among the evaluated models than audio-based preferences, although modelspecific differences remain.

![](images/2bda9fabd1b37e9ac6057d005be080a05902bdd8135b4a0f0849702d6f828702.jpg)  
Figure 3: Pairwise ranking agreement between VLMs. Each cell reports the mean Spearman rank correlation across 14 artist groups in the music-video setting. Models are grouped by family, showing strong within-family agreement and substantial cross-family correlations.

Cross-modal Preference Alignment. We next examine whether aesthetic preferences derived from audio remain aligned with those obtained from the corresponding music videos. For each artist group, we restrict the analysis to songs shared by both evaluation sets and compute the Spearman correlation between each audio-model ranking and each video-model ranking. Figure 4 reports the resulting correlations averaged across 14 artist groups.

Overall, cross-modal agreement is weak, with a mean Spearman correlation of only 0.060 across all audio-video model pairs. Most correlations are close to zero, and even the strongest model pair reaches only 0.288. This indicates that audio-based preferences are generally not preserved in visual evaluation.

We further compare the audio and visual consensus rankings for each artist group (shown in Table 4). The correlations range from —0.219 to 0.475, with only TWICE showing statistically significant alignment (ρ = 0.475, p = 0.007). BABY-MONSTER (ρ = 0.399), I-DLE (ρ = 0.377), and ITZY (ρ = 0.356) also exhibit relatively higher positive correlations, although none reaches statistical significance (p < 0.05). Several other groups show near-zero or negative correlations.

Together, these results reveal a clear contrast: although visual evaluation yields substantially stronger inter-model consensus, this agreement does not necessarily extend to audio-based preferences. The two evaluation settings therefore exhibit largely distinct aesthetic preference patterns.

Alignment with Real-world Popularity. Finally, we examine how model-derived visual preferences relate to real-world popularity. We compare VLM consensus rankings with historical YouTube view counts for official MVs from AESPA, IVE, and TWICE. Using Internet Archive snapshots taken approximately seven days after release (±2 days), we obtain historical view counts for 64 of the 76 MVs and compute Spearman rank correlations within each artist group. Further details are provided in Appendix F.

![](images/10ca546bdbe90d1979140ab48bd24dab4db93b1f38290dbed15f502341793694.jpg)  
Figure 4: Cross-modal ranking agreement between ALMs and VLMs. Each cell reports the mean Spearman rank correlation across 14 artist groups, computed using only songs shared by both evaluation sets.

<table><tr><td>Artist</td><td>#MVs</td><td>Spearman ρ</td><td>p-value</td></tr><tr><td>AESPA</td><td>19/23</td><td>0.661</td><td>0.002</td></tr><tr><td>IVE</td><td>19/23</td><td>0.363</td><td>0.127</td></tr><tr><td>TWICE</td><td>26/30</td><td>0.406</td><td>0.040</td></tr></table>

Table 5: Association between VLM consensus rankings and early YouTube viewership. Historical view counts are collected approximately seven days after release (±2 days). We report within-group Spearman correlations and two-sided p-values.

As shown in Table 5, all three groups exhibit positive correlations, with the strongest association observed for AESPA $( \rho ~ = ~ 0 . 6 6 1 )$ , followed by TWICE $( \rho = 0 . 4 0 6 )$ and IVE $( \rho = 0 . 3 6 3 )$ . Only AESPA remains statistically significant after correcting for multiple comparisons. Similar positive trends are observed under alternative observation windows. These findings suggest a possible association between machine visual preferences and early YouTube viewership.

## 5 Discussion

This study represents an initial step toward AIassisted creative decision making in the K-pop industry. By examining how foundation models evaluate K-pop songs and music videos, we begin to understand the role of machine aesthetic preferences in future creative decision-making systems. Our findings reveal differences in aesthetic agreement across audio and visual evaluation, as well as limited alignment between the two modalities. These observations suggest that machine taste cannot be fully characterized by a single aesthetic perspective. Moreover, the varying relationships between visual preferences and real-world popularity highlight the importance of considering factors beyond model-derived aesthetic evaluations when studying creative content in real-world settings.

Building on this initial understanding, our longterm goal is to progressively integrate machine aesthetic preferences with artist identity, audience feedback, historical performance, and market dynamics into a unified framework. Ultimately, we envision a K-pop world model that captures how creative content, artists, audiences, and the broader market interact and evolve, enabling AI systems to explore potential market responses to different creative strategies. Rather than treating aesthetic evaluation or historical popularity as isolated indicators, such a framework could provide a broader understanding of how creative choices interact with artist characteristics and changing market conditions. In the long term, this could support tasks such as song selection, comeback planning, artist positioning, and strategic development, while complementing the expertise of creative professionals.

## 6 Conclusion

In this work, we investigate machine aesthetic preferences through pairwise evaluations of Kpop songs and music videos. Visual evaluation shows stronger inter-model agreement and grouplevel consensus than audio evaluation, while crossmodal preferences remain weakly aligned. We also find positive but varying correlations between visual consensus rankings and historical YouTube viewership. These findings offer initial insights into machine taste and future AI-assisted creative decision making.

## Limitations

Our study focuses on 14 K-pop girl groups and a selected set of foundation models, so the findings may not fully generalize to other creative domains. The audio and video evaluations also involve different model populations, and the video setting considers visual content only. In addition, YouTube viewership provides a proxy for real-world popularity rather than direct human aesthetic judgments. Future work could extend the evaluation to broader datasets, additional model families, and human preference studies.

## Ethical Considerations

Our study examines aesthetic preferences expressed by foundation models rather than establishing objective standards of creative quality. Modelderived rankings may reflect biases in training data and should not be treated as substitutes for human aesthetic judgment. In future AI-assisted creative decision-making systems, excessive reliance on such rankings could reinforce existing popularity patterns and potentially reduce creative diversity. We therefore emphasize that machine aesthetic preferences should complement, rather than replace, the perspectives of artists, audiences, and creative professionals.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or llms as the judge? a study on judgement bias. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8301–8327.

Han Chen, Ming Li, Chenguang Wang, Yijun Liang, Dawei Zhou, Tianyi Zhou, et al. 2026. Llms struggle to measure what distinguishes students of different proficiency levels: A study of item discrimination in reading comprehension assessment. arXiv preprint arXiv:2606.18709.

Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. 2026. Litbench: A benchmark and dataset for reliable evaluation of creative writing. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7740–7755.

Yichen Feng, Yuetai Li, Chunjiang Liu, Yuanyuan Chen, Fengqing Jiang, Yue Huang, Hang Hua, Zhengqing Yuan, Kaiyuan Zheng, Luyao Niu, et al. 2026. Visual aesthetic benchmark: Can frontier models judge beauty? arXiv preprint arXiv:2605.12684.

Yuanning Feng, Sinan Wang, Zhengxiang Cheng, Yao Wan, and Dongping Chen. 2025. Are we on the right way to assessing llm-as-a-judge? arXiv preprint arXiv:2512.16041.

Sreyan Ghosh, Arushi Goel, Lasha Koroshinadze, Sanggil Lee, Zhifeng Kong, Joao Santos, Ramani Duraiswami, Dinesh Manocha, Wei Ping, Mohammad Shoeybi, et al. 2026. Music flamingo: Scaling music understanding in audio language models. In International Conference on Learning Representations, volume 2026, pages 78958–78996.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. 2025. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006.

HYBE. 2026. TECHNOLOGY. https://careers. hybecorp.com/ko/business3.

Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, and Yejin Choi. 2026. Artificial hivemind: The open-ended homogeneity of language models (and beyond). Advances in Neural Information Processing Systems, 38.

JYP Entertainment. 2025. FY25 Q1 Earnings Note. https://www.jype.com/board/ir-data/ 3t5vhexm.

Sungeun Kim and Dongsuk Oh. 2025. Evaluating creativity: can llms be good evaluators in creative writing tasks? Applied Sciences, 15(6):2971.

Junlong Li, Fan Zhou, Shichao Sun, Yikai Zhang, Hai Zhao, and Pengfei Liu. 2024. Dissecting human and 1lm preferences. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1790– 1811.

Kun Li, Lai Man Po, Hongzheng Yang, Xuyuan Xu, Kangcheng Liu, and Yuzhi Zhao. 2025. Aesbiasbench: Evaluating bias and alignment in multimodal language models for personalized image aesthetic assessment. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7618–7631.

Yinghao Ma, Anders Øland, Anton Ragni, Bleiz Mac-Sen Del Sette, Charalampos Saitis, Chris Donahue, Chenghua Lin, Christos Plachouras, Emmanouil Benetos, Elona Shatri, et al. 2024. Foundation models for music: A survey. arXiv preprint arXiv:2408.14340.

Guillermo Marco, Julio Gonzalo, and Víctor Fresno. 2025. The reader is the metric: How textual features and reader profiles explain conflicting evaluations of ai creative writing. In Findings of the Association for Computational Linguistics: ACL 2025, pages 25432– 25449.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. 2025. Learning to plan & reason for evaluation with thinking-llm-as-ajudge. arXiv preprint arXiv:2501.18099.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in llmas-a-judge. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292-314.

SM Entertainment. 2026. 4Q25 CEO Message Script.https://cdn2.smentertainment.com/ wp-content/uploads/2026/02/11152627/4Q25\_ CEO-Message\_Script\_Eng\_vF.pdf.

Kaishen Wang, Dongdi Zhao, Yijun Liang, Dingqiang Ye, Ruibo Chen, Heng Huang, and Di Fu. 2026. Where to look matters: On-policy self-distillation for long-video understanding. arXiv preprint arXiv:2608.25356.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Tianyi Xiong, Yi Ge, Ming Li, Zuolong Zhang, Pranav Kulkarni, Kaishen Wang, Qi He, Zeying Zhu, Chenxi Liu, Ruibo Chen, et al. 2026. Multi-crit: Benchmarking multimodal judges on pluralistic criteriafollowing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8641–8652.

Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. 2025. Lllava-critic: Learning to evaluate multimodal models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13618–13628. IEEE.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. 2025. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Chen Yang, Chufan Yu, Hanfu Chen, Jie Zhu, Jingqi Chen, Ke Chen, Wenxuan Wang, Yang Wang, Yaozhou Jiang, Yi Jiang, et al. 2026. Moss-audio technical report. arXiv preprint arXiv:2606.01802.

Yi Yao and Qiu Zhuang. 2025. Do multimodal large language models have “good taste"? an evaluation of mllms via the visual aesthetic sensitivity test. In Proceedings of the 39th Pacific Asia Conference on Language, Information and Computation, pages 413– 418.

Dingqiang Ye, Dongdi Zhao, Kaishen Wang, Qingqiao Hu, Jingchen Sun, Yijun Liang, Yuqi Jia, Yiqiao Huang, Yunjie Tian, Jiaxing Zhang, et al. 2026. Videoresearcher: Self-improving tool design for long-video understanding. arXiv preprint arXiv:2609.19664.

Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. 2025. Justice or prejudice? quantifying biases in llm-as-a-judge. In International Conference on Learning Representations, volume 2025, pages 102351–102390.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Ranchi Zhao, et al. 2026. Minicpmv 4.5: Cooking efficient mllms via architecture, data, and training recipe. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11704–11715.

Justin Zhao, Flor Miriam Plaza-del Arco, Benjamin Genchel, and Amanda Cercas Curry. 2025. Language model council: Democratically benchmarking foundation models on highly subjective tasks. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 12395–12450.

## Appendix

## A Data Usage and Copyright

Our evaluation uses publicly released K-pop songs and official music videos solely for academic research. All original audio and video content remains the property of its respective copyright holders.

## B Full Song List

Tables 12 and 13 provide the complete lists of songs and music videos used in our audio-based and video-based evaluations, respectively, covering all 14 K-pop artist groups.

## C Detailed Experimental Results

We provide the complete pairwise Spearman correlation results underlying the agreement heatmaps in the main paper. Table 6 reports the audio-based results, while Table 14 presents the music-videobased results, with model abbreviations listed in Table 7.

## D Prompts

We provide the complete system and question prompts used for audio-based song ranking and music-video-based evaluation. The placeholder {group} is replaced with the corresponding artist group name during evaluation.

## D.1 Audio Evaluation

## System Prompt

You are a highly experienced K-pop music critic and a devoted, long-time fan of {group}. You are comparing two songs by {group} and must judge which one is the BETTER SONG based on musical quality and craftsmanship – not simply which one gives you the strongest nostalgia or hype in the moment.

## Important instructions:

• Track labels A and B are arbitrary and carry no meaning.

• Playback order must NOT influence your judgment.

• Evaluate each track independently and carefully before comparing them. Pay close attention to concrete musical quality dimensions you can actually judge from listening: melody and hook strength, vocal performance and harmonies, production and mixing quality, arrangement and song structure, and overall cohesion. Do not judge originality – you cannot reliably tell from a single listen whether something is derivative of other songs, so do not factor that in.

• Do not favor the first or second track simply because of presentation order.

• Base your judgment strictly on the audio quality and craftsmanship you actually hear in each track, not on which one you recognize, remember more vividly, or assume is more popular.

Do not identify or guess the song titles. Base your judgment only on the audio.

## Question Prompt

Having carefully and independently evaluated the musical quality of Track A and Track B – melody, vocals, production, and arrangement – which track is the better song overall?

Answer with EXACTLY one line and nothing else:

PREFERENCE: A

or

PREFERENCE:B

## D.2 Music-Video Evaluation

## System Prompt

You are a highly experienced K-pop music video critic and a devoted, long-time fan of {group}. You are comparing two official music videos by {group} and must judge which one is the BETTER MUSIC VIDEO as a visual production – not simply which one gives you the strongest nostalgia or hype in the moment, and not based on the song itself.

Important instructions:

• Video labels A and B are arbitrary and carry no meaning.

• Playback order must NOT influence your judgment.

• Evaluate each video independently and carefully before comparing them. Pay close attention to concrete visual quality dimensions you can actually judge from watching: choreography and dance performance, cinematography and camera work, visual concept and styling (costumes, sets, color grading), editing and pacing, and overall visual cohesion and production value. Do not judge originality – you cannot reliably tell from a single viewing whether something is derivative of other MVs, so do not factor that in.

• Do not favor the first or second video simply because of presentation order.

• Base your judgment strictly on the visual quality and craftsmanship you actually see in each video, not on which one you recognize, remember more vividly, or assume is more popular.

Do not identify or guess the song titles. Base your judgment only on what you see.

## Question Prompt

Having carefully and independently evaluated the visual quality of Video A and Video B – choreography, cinematography, styling/concept, and editing – which video is the better music video overall?

Answer with EXACTLY one line and nothing else:

PREFERENCE: A

or

PREFERENCE:B

<table><tr><td>Model</td><td>MOSS-4B-Inst</td><td>MOSS-4B-Think</td><td>MOSS-8B-Inst</td><td>MOSS-8B-Think</td><td>Qwen-Inst</td><td>Qwen-Think</td></tr><tr><td>MOSS-4B-Inst</td><td></td><td>0.018</td><td>0.091</td><td>0.176</td><td>0.106</td><td>-0.043</td></tr><tr><td>MOSS-4B-Think</td><td></td><td></td><td>0.276</td><td>0.425</td><td>0.320</td><td>-0.063</td></tr><tr><td>MOSS-8B-Inst</td><td></td><td></td><td></td><td>0.433</td><td>0.354</td><td>-0.029</td></tr><tr><td>MOSS-8B-Think</td><td></td><td></td><td></td><td></td><td>0.526</td><td>-0.144</td></tr><tr><td>Qwen-Inst</td><td></td><td></td><td></td><td></td><td></td><td>-0.086</td></tr><tr><td>Qwen-Think</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 6: Pairwise ranking agreement between ALMs. Each cell reports the mean Spearman rank correlation across 14 artist groups. Higher values indicate stronger agreement between model rankings.

<table><tr><td>Abbreviation</td><td>Model</td></tr><tr><td>G-N / G-T</td><td>GLM-4.6V-Flash, non-thinking / thinking</td></tr><tr><td>I1-N/I1-T</td><td>InternVL3.5-1B, non-thinking / thinking</td></tr><tr><td>I2-N / I2-T</td><td>InternVL3.5-2B, non-thinking / thinking</td></tr><tr><td>I4-N / I4-T</td><td>InternVL3.5-4B, non-thinking / thinking</td></tr><tr><td>I8-N / I8-T</td><td>InternVL3.5-8B, non-thinking / thinking</td></tr><tr><td>I14-N/I14-T</td><td>InternVL3.5-14B, non-thinking / thinking</td></tr><tr><td>I38-N / I38-T</td><td>InternVL3.5-38B, non-thinking / thinking</td></tr><tr><td>M-N/M-T</td><td>MiniCPM-V-4.5, non-thinking / thinking</td></tr><tr><td>Q2-N / Q2-T</td><td>Qwen3.5-2B, non-thinking / thinking</td></tr><tr><td>Q4-N / Q4-T</td><td>Qwen3.5-4B, non-thinking / thinking</td></tr><tr><td>Q9-N / Q9-T</td><td>Qwen3.5-9B, non-thinking / thinking</td></tr><tr><td>Q27-N / Q27-T</td><td>Qwen3.5-27B, non-thinking / thinking</td></tr><tr><td>V2 / V4 / V8 / V32</td><td>Qwen3-VL-2B/4B/8B/32B-Instruct</td></tr></table>

Table 7: Abbreviations used for VLMs in Table 14.

## E Case Studies

To complement the aggregate results, we present three song-level case studies corresponding to the main evaluation settings. We examine inter-model agreement in AESPA's audio evaluation, grouplevel consensus in IVE's music-video evaluation, and cross-modal preference alignment in TWICE. All reported rankings are derived from the original pairwise evaluation results, where a lower rank indicates a stronger preference.

## E.1 Inter-model Agreement: AESPA

Table 8 illustrates substantial variation in the audio preferences of individual models. Savage receives favorable rankings from most configurations, particularly MOSS-Audio-8B-Thinking (rank 1), although Qwen3-Omni-Thinking places it at rank 16. The disagreement is more pronounced for several other songs. For example, Hold On Tight ranks second under MOSS-Audio-4B-Instruct but last under both MOSS-Audio-4B-Thinking and MOSS-Audio-8B-Thinking. Similarly, Next Level is ranked fourth by MOSS-Audio-4B-Thinking but 27th by MOSS-Audio-4B-Instruct and Qwen3- Omni-Thinking. In contrast, Can't Help Myself receives consistently low rankings across all six configurations. These examples illustrate how individual model preferences can differ substantially, even when evaluating the same songs under an identical protocol.

## E.2 Group-level Consensus: IVE

Table 9 presents representative consensus rankings from IVE's visual evaluation. HEYA receives the highest consensus ranking, with a mean rank of 2.54 and a relatively small standard deviation of 1.82, indicating that most evaluated configurations place it near the top. I AM and LOVE DIVE also receive favorable rankings with comparatively limited variation. In contrast, BLACKHOLE ranks fourth overall but exhibits greater dispersion (SD = 4.49), suggesting less agreement on its exact position despite its favorable average ranking. At the other end, Either Way receives the lowest consensus ranking with a similarly small SD of 1.85. These examples illustrate that group-level consensus can capture both shared high and low preferences, while individual songs may still exhibit different degrees of inter-model variation.

## E.3 Cross-modal Preference Alignment: TWICE

TWICE exhibits relatively stronger cross-modal consensus alignment $( \rho = 0 . 4 7 5 )$ , as illustrated in Table 10. Fancy and Feel Special receive identical rankings across audio and visual evaluations, while Celebrate also ranks highly in both. Similar agreement appears among lower-ranked songs, including TT and Cheer Up. Nevertheless, notable differences remain. More & More ranks first visually but tenth in audio, while Scientist receives a considerably higher visual ranking (9 versus 23). Conversely, YES or YES ranks first in audio but twentieth visually, and Doughnut ranks sixth in audio but 26th visually. These examples demonstrate that stronger aggregate cross-modal agreement does not necessarily imply consistent preferences for every individual song.

<table><tr><td>Song</td><td>M4-I</td><td>M4-T</td><td>M8-I</td><td>M8-T</td><td>Q-I</td><td>Q-T</td><td>Consensus</td></tr><tr><td>Savage</td><td>10</td><td>2</td><td>2</td><td>1</td><td>6</td><td>16</td><td>1</td></tr><tr><td>Welcome To MY World</td><td>7</td><td>3</td><td>1</td><td>5</td><td>5</td><td>24</td><td>2</td></tr><tr><td>Drama</td><td>5</td><td>14</td><td>8</td><td>2</td><td>3</td><td>23</td><td>3</td></tr><tr><td>Count On Me</td><td>15</td><td>10</td><td>16</td><td>6</td><td>15</td><td>6</td><td>6</td></tr><tr><td>Spicy</td><td>9</td><td>25</td><td>10</td><td>2</td><td>2</td><td>24</td><td>8</td></tr><tr><td>Next Level</td><td>27</td><td>4</td><td>6</td><td>9</td><td>12</td><td>27</td><td>13</td></tr><tr><td>Better Things</td><td>30</td><td>9</td><td>2</td><td>18</td><td>24</td><td>7</td><td>16</td></tr><tr><td>Hold On Tight</td><td>2</td><td>30</td><td>15</td><td>30</td><td>21</td><td>13</td><td>23</td></tr><tr><td>Black Mamba</td><td>21</td><td>26</td><td>23</td><td>21</td><td>8</td><td>29</td><td>28</td></tr><tr><td>Can&#x27;t Help Myself</td><td>28</td><td>19</td><td>26</td><td>23</td><td>22</td><td>30</td><td>30</td></tr></table>

Table 8: Song-level audio rankings of all 30 AESPA songs across six ALM configurations. M4, M8, and Q denote MOSS-Audio-4B, MOSS-Audio-8B, and Qwen3-Omni-30B-A3B, respectively; I and T indicate Instruct and Thinking. Consensus rankings are obtained by averaging the six model rankings and re-ranking the songs. Lower ranks indicate stronger preferences.

<table><tr><td>Song</td><td>Mean Rank</td><td>Rank SD</td><td>Consensus</td></tr><tr><td>HEYA</td><td>2.54</td><td>1.82</td><td>1</td></tr><tr><td>I AM</td><td>4.50</td><td>2.79</td><td>2</td></tr><tr><td>ELEVEN</td><td>4.55</td><td>3.62</td><td>3</td></tr><tr><td>BLACKHOLE</td><td>5.39</td><td>4.49</td><td>4</td></tr><tr><td>LOVE DIVE</td><td>5.71</td><td>2.82</td><td>5</td></tr><tr><td>BANG BANG</td><td>7.46</td><td>4.02</td><td>6</td></tr><tr><td>After LIKE</td><td>8.27</td><td>3.35</td><td>9</td></tr><tr><td>Accendio</td><td>13.77</td><td>4.01</td><td>14</td></tr><tr><td>Off The Record</td><td>19.00</td><td>3.82</td><td>21</td></tr><tr><td>Either Way</td><td>20.46</td><td>1.85</td><td>23</td></tr></table>

Table 9: Selected IVE songs illustrating visual consensus among 28 VLM configurations over 23 music videos. Mean Rank and Rank SD denote the mean and sample standard deviation of model rankings, respectively. Consensus is obtained by re-ranking the mean ranks. Lower ranks indicate stronger preferences, while smaller SD values indicate greater agreement.

## F Real-world Popularity Analysis

Data Collection. To examine the relationship between model-derived aesthetic preferences and realworld popularity, we collect historical YouTube view counts for official music videos from AESPA, IVE, and TWICE. These three groups contain 23, 23, and 30 music videos in our evaluation dataset, respectively, resulting in 76 videos in total.

We first match each song to its corresponding official YouTube video. Candidate videos are identified through Kworb and web search, and subsequently verified using their titles and uploading channels. We retain official music videos while excluding alternative content such as dance practices, performance videos, lyric videos, teasers, and remixes. All 76 music videos are successfully matched to their official YouTube video IDs.

<table><tr><td>Song</td><td>Audio Rank</td><td>Visual Rank</td></tr><tr><td>Fancy</td><td>2</td><td>2</td></tr><tr><td>Feel Special</td><td>5</td><td>5</td></tr><tr><td>Celebrate</td><td>3</td><td>6</td></tr><tr><td>Heart Shaker</td><td>17</td><td>17</td></tr><tr><td>TT</td><td>25</td><td>24</td></tr><tr><td>Cheer Up</td><td>30</td><td>29</td></tr><tr><td>More &amp; More</td><td>10</td><td>1</td></tr><tr><td>Scientist</td><td>23</td><td>9</td></tr><tr><td>YES or YES</td><td>1</td><td>20</td></tr><tr><td>Doughnut</td><td>6</td><td>26</td></tr></table>

Table 10: Selected TWICE songs illustrating crossmodal preference alignment. Audio and visual consensus rankings are obtained from six ALM and 28 VLM configurations, respectively, using the same 30 songs with official music videos. Lower ranks indicate stronger preferences.

Since the YouTube Data API provides current cumulative view counts but does not offer publicly accessible historical statistics for third-party channels, we retrieve historical observations from the Internet Archive's Wayback Machine. Specifically, we identify archived snapshots of the corresponding YouTube watch pages and extract the cumulative view counts displayed at the time of capture.

Observation Window. Directly comparing current cumulative view counts would introduce substantial differences in exposure time, as the evaluated songs were released in different years. We therefore focus on early YouTube viewership within a fixed post-release observation window.

Our primary analysis uses archived view counts observed within two days of the seventh day after each video's publication. For each video, we select the available snapshot closest to day seven and retain its original view count without interpolation or extrapolation. This yields usable historical observations for 64 of the 76 music videos: 19 from AESPA, 19 from IVE, and 26 from TWICE. The actual capture ages range from 5.2 to 8.8 days after publication.

<table><tr><td>Setting</td><td>AESPA</td><td>IVE</td><td>TWICE</td></tr><tr><td>Day 7 (±2 days)</td><td>0.661 (19)</td><td>0.363 (19)</td><td>0.406 (26)</td></tr><tr><td>Day 7 (±1 day)</td><td>0.704 (15)</td><td>0.354 (14)</td><td>0.423 (22)</td></tr><tr><td>Day 30 (±3 days)</td><td>0.782 (11)</td><td>0.423 (13)</td><td>0.367 (22)</td></tr></table>

Table 11: Robustness analysis of the association between VLM consensus rankings and historical YouTube viewership across different observation windows. Each cell reports Spearman's ρ, with the number of available music videos indicated in parentheses.

We additionally consider two alternative settings for robustness analysis: a stricter seven-day window (±1 day) and a 30-day window (±3 days), covering 51 and 46 music videos, respectively. Videos without an archived observation within the corresponding window are excluded from that analysis.

Evaluation Protocol. We use the existing visual consensus rankings obtained from the 28 VLM configurations, without performing any additional model inference. For each artist group, we restrict the comparison to songs with available historical viewership data and rank these songs by their observed view counts, with higher view counts indicating greater popularity within the observation window.

We then compute Spearman's rank correlation between the model-derived visual consensus ranking and the corresponding viewership ranking. Correlations are calculated independently for each artist group rather than pooling songs across groups with different audience sizes. We report two-sided p-values and additionally verify the results using permutation tests with 100,000 random shuffles.

Robustness Analysis. Table 11 reports the results under different observation windows. All correlations remain positive, with AESPA consistently showing the strongest association. The stricter seven-day window produces results similar to the primary analysis, while the 30-day analysis preserves the same overall pattern despite its lower data coverage.

We further conduct an exploratory analysis controlling for publication time. The resulting partial Spearman correlations are 0.60, 0.32, and 0.55 for

AESPA, IVE, and TWICE, respectively, compared with the original correlations of 0.66, 0.36, and 0.41. These results suggest that the observed associations are not solely explained by the chronological ordering of releases.

## G AI Assistance Disclosure

ChatGPT was used solely for language polishing. All research design, experiments, data collection, analysis, and conclusions were independently completed by the authors.

<table><tr><td>Group</td><td>Songs</td></tr><tr><td>AESPA</td><td>Lemonade; Switchblade (Feat. Ty Dolla $ign); Supernova; Kiss N Tell; WDA (Whole Different Animal); Whiplash; Dirty Work; Drama; Live My Life; Armageddon; Rich Man; Spicy; Hot Mess; Better Things; Welcome To My World (Feat. Nævis); Life's Too Short (English Ver.); Savage; Next Level; Dreams Come True; Girls; Black Mamba; Forever; Hold On Tight; Count On Me; 'Til We Die; Aenergy; Yeppi Yeppi; Can't Help Myself; Just Another Girl; Prologue.</td></tr><tr><td>IVE</td><td>Eleven; Love Dive; After Like; Kitsch; I Am; Wave; I Want; Either Way; Off The Record; Baddie; All Night (Feat. Saweetie); Heya; Accendio; Summer Festa; Crush; Rebel Heart; Attitude; XOXZ; Bang Bang; Blackhole; Supernova Love; Lucid Dream; Be Alright; Take It; Royal; Blue Blood; Holy Moly; OTT; Payback; Ice Queen.</td></tr><tr><td>LE SSERAFIM</td><td>Hot; Crazy; Made My Night; Fearless; Perfect Night; Unforgiven; Smart; Antifragile; Boompala; Impurities; Celebration; Different; Easy; Come Over; Spaghetti; Eve, Psyche &amp; The Bluebeard's Wife; Kawaii; Sour Grapes; Blue Flame; The Great Mermaid; No Celestial; Good Parts (When The Quality Is Bad But I Am); Fire In The Belly; 1-800-Hot-N-Fun; Pierrot; Crazier; Ash; So Cynical (Badum); Iffy Iffy; Trust Exercise.</td></tr><tr><td>NEWJEANS</td><td>Attention; Hype Boy; Hurt; Cookie; Ditto; OMG; Zero; New Jeans; Super Shy; Cool With You (Side A); ETA; Bubble Gum; How Sweet; Right Now; Supernatural; ASAP; Get Up; Gods; Beautiful Restriction; Our Night Is More Beautiful Than Your Day.</td></tr><tr><td>I-DLE</td><td>Nxde; Tomboy; Mono; I Want That; Queencard; Oh My God; Gimme Dat Love; Good Thing; Girlfriend; Crow; Lion; Klaxon; Super Lady; Hwaa; Wife; Latata; Senorita; Uh-Oh; Hann (Alone); Allergy; Dumdi Dumdi; Revenge; I Do; Last Dance; Blow Your Mind; My Bag; Fate; I'm The Trend; Bloom; Nightmare.</td></tr><tr><td>TWICE</td><td>This Is For; Strategy; What Is Love?; The Feels; Fancy; Feel Special; Talk That Talk; I Got You; Set Me Free; One Spark; I Can't Stop Me; Alcohol-Free; Me+You; More &amp; More; Scientist; TT; Knock Knock; Yes Or Yes; Dance The Night Away; Likey; Cheer Up; Like Ooh-Ahh; Heart Shaker; Signal; Doughnut; Perfect World; Kura Kura; Celebrate; Hare Hare; Moonlight Sunrise.</td></tr><tr><td>BLACKPINK</td><td>As If It's Your Last; How You Like That; Playing With Fire; Go; Pink Venom; Lovesick Girls; Jump; Kill This Love; Shut Down; Boombayah; Ddu-Du Ddu-Du; Ice Cream; Whistle; Ready For Love; Stay; Typa Girl; Forever Young; Really; See U Later; Don't Know What To Do; Kick It; Hope Not; Pretty Savage; Bet You Wanna; Crazy Over You; Love To Hate Me; You Never Know; Yeah Yeah Yeah; The Happiest Girl; Tally.</td></tr><tr><td>RED VELVET</td><td>Russian Roulette; Cosmic; Feel My Rhythm; Surfin Boy; Chill Kill; Psycho; Peek-A-Boo; Dumb Dumb; Zimzalabim; Red Flavor; Bad Boy; RBB (Really Bad Boy); Queendom; Umpah Umpah; Ice Cream Cake; Happiness; Cookie Jar; Automatic; One Of These Nights; Rookie; Power Up; Would U; Rebirth; I Just; Sappy; Wildside; Be Natural; Wish Tree; Look; Kingdom Come.</td></tr><tr><td>ITZY</td><td>Wannabe; Tunnel Vision; Loco; Bet On Me; ICY; Motto; In The Morning; Dalla Dalla; Girls Will Be Girls; Cheshire; Untouchable; Not Shy; Born To Be; Imaginary Friend; Cake; Boys Like You; Gold; None Of My Business; Sneakers; Swipe; Mr. Vampire; Voltage; Blah Blah Blah; Ringo; Algorhythm; Rock &amp; Roll; Shoot!; That's A No No; #Twenty; Sorry Not Sorry.</td></tr><tr><td>NMIXX</td><td>Blue Valentine; Heavy Serenade; Know About Me; Crescendo; Dice; Love Me Like This; Dash; Party O'clock; Tic Tic (Feat. Pabllo Vittar); See That?; Soñar (Breaker); Roller Coaster; O.O; Young, Dumb, Stupid; Funky Glitter Christmas; Feifei; Run For Roses; Cool (Your Rainbow); Passionfruit; High Horse; Spinnin' On It; Papillon; Boom; Love Is Lonely; XOXO; My Gosh; Paxxword; Beat Beat; Red Light Sign, But We Go; Break The Wall.</td></tr><tr><td>BABYMONSTER</td><td>Batter Up; Stuck In The Middle; Sheesh; Like That; Dream; Forever; Clik Clak; Drip; Love, Maybe; Really Like You; Billionaire; Love In My Heart; Woke Up In Tokyo; Ghost; Hot Sauce; We Go Up; Psycho; Supa Dupa Luv; Wild; Moon; Choom; I Like It; Locked In; Sugar Honey Ice Tea.</td></tr><tr><td>MEOVV</td><td>Meow; Toxic; Body; Hands Up; Drop Top; Lit Right Now; Me Me Me; Burning Up; Hit 'Em; DDI Ro Ri; In My Hands; Favorite Song; Revenge.</td></tr><tr><td>ILLIT</td><td>My World; Magnetic; Midnight Fiction; Lucky Girl Syndrome; I'll Like You; Cherish (My Love); IYKYK; Pimple; Tick-Tack; Almond Chocolate; I Got Your Back; Little Monster; Billyeoon Goyangi (Do The Dance); Jellyous; Oops!; Bamsopoong; Topping; Toki Yo Tomare; Secret Quest; Not Cute Anymore; Not Me; All For You; Sunday Morning; GRWM (Get Ready With Me); It's Me; Paw, Paw!; Mamihlapinatapai; Love, Older You; Swingin' Magic; Bubee.</td></tr><tr><td>HEARTS2HEARTS</td><td>The Chase; Butterflies; Style; Pretty Please; Focus; Apple Pie; Flutter; Blue Moon; Rude!; Lemon Tang; 15-Love; Baby Steps; Heart Emoji ($\{ }heartsuit$); Secret Recipe; Iconic Heart; Moonride.</td></tr><tr><td>AESPA</td><td>Lemonade; Switchblade (Feat. Ty Dolla $ign); Supernova; Kiss N Tell; WDA (Whole Different Animal); Whiplash; Dirty Work; Drama; Live My Life; Armageddon; Rich Man; Spicy; Hot Mess; Better Things; Welcome To My World (Feat. Nævis); Life's Too Short (English Ver.); Savage; Next Level; Dreams Come True; Girls; Black Mamba; Forever; Hold On Tight.</td></tr><tr><td>IVE</td><td>Eleven; Love Dive; After Like; Kitsch; I Am; Wave; I Want; Either Way; Off The Record; Baddie; All Night (Feat. Saweetie); Heya; Accendio; Summer Festa; Crush; Rebel Heart; Attitude; XOXZ; Bang Bang; Blackhole; Supernova Love; Lucid Dream; Be Alright.</td></tr><tr><td>LE SSERAFIM</td><td>Hot; Crazy; Made My Night; Fearless; Perfect Night; Unforgiven; Smart; Antifragile; Boompala; Impurities; Celebration; Different; Easy; Come Over; Spaghetti; Eve, Psyche &amp; The Bluebeard's Wife; Kawaii.</td></tr><tr><td>NEWJEANS</td><td>Attention; Hype Boy; Hurt; Cookie; Ditto; OMG; Zero; New Jeans; Super Shy; Cool With You (Side A); ETA; Bubble Gum; How Sweet; Right Now; Supernatural; ASAP.</td></tr><tr><td>I-DLE</td><td>Nxde; Tomboy; Mono; I Want That; Queencard; Oh My God; Gimme Dat Love; Good Thing; Girlfriend; Crow; Lion; Klaxon; Super Lady; Hwaa; Wife; Latata; Senorita; Uh-Oh; Hann (Alone); Allergy; Dumdi Dumdi; Revenge; I Do; Last Dance; Blow Your Mind.</td></tr><tr><td>TWICE</td><td>This Is For; Strategy; What Is Love?; The Feels; Fancy; Feel Special; Talk That Talk; I Got You; Set Me Free; One Spark; I Can't Stop Me; Alcohol-Free; Me+You; More &amp; More; Scientist; TT; Knock Knock; Yes Or Yes; Dance The Night Away; Likey; Cheer Up; Like Ooh-Ahh; Heart Shaker; Signal; Doughnut; Perfect World; Kura Kura; Celebrate; Hare Hare; Moonlight Sunrise.</td></tr><tr><td>BLACKPINK</td><td>As If It's Your Last; How You Like That; Playing With Fire; Go; Pink Venom; Lovesick Girls; Jump; Kill This Love; Shut Down; Boombayah; Ddu-Du Ddu-Du; Ice Cream; Whistle; Ready For Love; Stay; Typa Girl; Forever Young; Really; See U Later; Don't Know What To Do; Kick It; Hope Not; Pretty Savage; Bet You Wanna; Crazy Over You; Love To Hate Me; You Never Know; Yeah Yeah Yeah; The Happiest Girl; Tally.</td></tr><tr><td>RED VELVET</td><td>Russian Roulette; Cosmic; Feel My Rhythm; Surfin Boy; Chill Kill; Psycho; Peek-A-Boo; Dumb Dumb; Zimzalabim; Red Flavor; Bad Boy; RBB (Really Bad Boy); Queendom; Umpah Umpah; Ice Cream Cake; Happiness; Cookie Jar; Automatic; One Of These Nights; Rookie; Power Up; Would U; Rebirth; I Just; Sappy; Wildside; Be Natural; Wish Tree.</td></tr><tr><td>ITZY</td><td>Wannabe; Tunnel Vision; Loco; Bet On Me; ICY; Motto; In The Morning; Dalla Dalla; Girls Will Be Girls; Cheshire; Untouchable; Not Shy; Born To Be; Imaginary Friend; Cake; Boys Like You; Gold; None Of My Business; Sneakers; Swipe; Mr. Vampire; Voltage; Blah Blah Blah; Ringo; Algorhythm; Rock &amp; Roll.</td></tr><tr><td>NMIXX</td><td>Blue Valentine; Heavy Serenade; Know About Me; Crescendo; Dice; Love Me Like This; Dash; Party O'clock; Tic Tic (Feat. Pabllo Vittar); See That?; Soñar (Breaker); Roller Coaster; O.O; Young, Dumb, Stupid; Funky Glitter Christmas; Feifei.</td></tr><tr><td>BABYMONSTER</td><td>Batter Up; Stuck In The Middle; Sheesh; Forever; Clik Clak; Drip; Really Like You; Love In My Heart; Hot Sauce; We Go Up; Psycho; Supa Dupa Luv; Moon; Choom; I Like It; Sugar Honey Ice Tea.</td></tr><tr><td>MEOVV</td><td>Meow; Toxic; Hands Up; Drop Top; Burning Up; DDI Ro Ri; In My Hands.</td></tr><tr><td>ILLIT</td><td>Jellyous; Not Cute Anymore; It's Me; Tick-Tack; Cherish (My Love); Billyeoon Goyangi (Do The Dance); Lucky Girl Syndrome; Magnetic; Sunday Morning; I Got Your Back; Not Me; Toki Yo Tomare.</td></tr><tr><td>HEARTS2HEARTS</td><td>The Chase; Butterflies; Pretty Please; Style; Focus; Rude!; Iconic Heart; Lemon Tang; Moonride.</td></tr></table>

Table 12: Complete list of audio tracks included in the audio evaluation (373 items across 14 K-pop groups).

Table 13: Complete list of music videos included in the mv evaluation (278 items across 14 K-pop groups).

Lasdo s  a e e oe s as e sodes e ng es e-s t   ee e s s : Hier    s a s gs
<table><tr><td rowspan=1 colspan=15>Model G-NG-TI1-N11-TI2-NI2-TI4-NI4-TI8-NI8-TI14-NI14-TI38-NI38-TM-NM-TQ2-NQ2-TQ4-NQ4-TQ9-NQ9-TQ27-NQ27-T</td><td rowspan=1 colspan=3>V2  V4  V8</td><td rowspan=1 colspan=1>V32</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>G-N        0.8950.3260.3070.4760.4740.6480.6290.5820.7090.655</td><td rowspan=1 colspan=2>0.721 0.662</td><td rowspan=1 colspan=2>0.7220.600</td><td rowspan=1 colspan=2>0.6230.423</td><td rowspan=1 colspan=1>0.637</td><td rowspan=1 colspan=1>0.630</td><td rowspan=1 colspan=2>0.6500.507</td><td rowspan=1 colspan=1>0.585</td><td rowspan=1 colspan=2>0.554 0.506</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.541</td><td rowspan=1 colspan=1>0.615</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>G-T              0.3510.3350.4780.5230.7080.7040.5870.7510.674</td><td rowspan=1 colspan=2>0.7110.678</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.469</td><td rowspan=1 colspan=1>0.721</td><td rowspan=1 colspan=1>0.626</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1>0.661</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>I1-N                    0.6040.5650.5910.5380.5590.5300.5070.413</td><td rowspan=1 colspan=2>0.3320.523</td><td rowspan=1 colspan=1>0.389</td><td rowspan=1 colspan=1>0.189</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=1>0.438</td><td rowspan=1 colspan=1>0.258</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.051</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>-0.019</td><td rowspan=1 colspan=1>0.415</td><td rowspan=1 colspan=1>0.357</td><td rowspan=1 colspan=1>0.254</td><td rowspan=1 colspan=1>0.249</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>I1-T                           0.6540.8250.5790.6810.7310.6450.593</td><td rowspan=1 colspan=2>0.4530.617</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=3>0.3060.3180.620</td><td rowspan=1 colspan=1>0.353</td><td rowspan=1 colspan=1>0.221</td><td rowspan=1 colspan=1>0.181</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=2>0.106 -0.033</td><td rowspan=1 colspan=1>0.529</td><td rowspan=1 colspan=1>0.534</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.279</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>I2-N                                 0.8320.7060.7210.6980.6880.570</td><td rowspan=1 colspan=2>0.5100.643</td><td rowspan=1 colspan=2>0.5620.382</td><td rowspan=1 colspan=2>0.4360.477</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.308</td><td rowspan=1 colspan=1>0.300</td><td rowspan=1 colspan=1>0.210</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.091</td><td rowspan=1 colspan=1>0.5660</td><td rowspan=1 colspan=1>.419</td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.320</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>I2-T                                        0.7510.8180.8280.7950.678</td><td rowspan=1 colspan=2>0.5780.720</td><td rowspan=1 colspan=2>0.6380.451</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=2>0.5870.441</td><td rowspan=1 colspan=1>0.332</td><td rowspan=1 colspan=1>0.325</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.089</td><td rowspan=1 colspan=1>0.646</td><td rowspan=1 colspan=1>0.557</td><td rowspan=1 colspan=1>0.412</td><td rowspan=1 colspan=1>0.348</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=2>I4-N                                              0.9200.7360.8320.717I4-T                                                     0.8070.8810.757I8-N                                                           0.8750.852</td><td rowspan=1 colspan=2>0.7140.714</td><td rowspan=1 colspan=3>0.7140.5050.544</td><td rowspan=1 colspan=2>0.5680.603</td><td rowspan=1 colspan=1>0.512</td><td rowspan=1 colspan=1>0.505</td><td rowspan=1 colspan=1>0.401</td><td rowspan=1 colspan=1>0.447</td><td rowspan=1 colspan=2>0.380 0.330</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.529</td><td rowspan=1 colspan=1>0.508</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>0.7350.749</td><td rowspan=1 colspan=5>0.7490.5090.5560.6340.590</td><td rowspan=1 colspan=1>0.499</td><td rowspan=1 colspan=1>0.498</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.430</td><td rowspan=1 colspan=2>0.351 0.303</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.545</td><td rowspan=1 colspan=1>0.518</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>0.7830.847</td><td rowspan=1 colspan=2>0.7370.532</td><td rowspan=1 colspan=3>0.5440.6860.525</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>0.427</td><td rowspan=1 colspan=1>0.449</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=2>0.312 0.184</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.645</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.530</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=2>I8-T                                                                  0.866I14-NI14-T</td><td rowspan=1 colspan=2>0.8410.852</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.623</td><td rowspan=1 colspan=3>0.6660.6490.659</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.499</td><td rowspan=1 colspan=1>0.481</td><td rowspan=1 colspan=2>0.458 0.353</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>0.9310.829</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=2>0.6130.674</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.574</td><td rowspan=1 colspan=1>0.557</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=2>0.480 0.348</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.798</td><td rowspan=1 colspan=1>0.816</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=2>0.6180.632</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.538</td><td rowspan=1 colspan=1>0.551</td><td rowspan=1 colspan=1>0.427</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.711</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>I38-N</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.816</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=2>0.6420.585</td><td rowspan=1 colspan=1>0.554</td><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.516</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1>0.458</td><td rowspan=1 colspan=1>0.342</td><td rowspan=1 colspan=1>0.711</td><td rowspan=1 colspan=1>0.630</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=14 colspan=20>I38-T                                                                                             0.694                   0.692      0.559                            0.633M-N                                                                                                    0.9320.4540.7220.7350.7870.6260.7400.730 0.6580.4890.5740.7880.617M-T                                                                                                           0.4390.7260.7050.8060.5730.7340.695 0.6630.4990.5830.7830.600Q2-N                                                                                                                0.5970.4190.4060.4900.3460.318 0.2490.5010.6020.5060.445Q2-T                                                                                                                       0.7080.8130.596       0.666 0.6770.5400.7130.765Q4-N                                                                                                                                    0.7350.771 0.767 0.6780.5480.5290.7570.703Q4-T                                                                                                                                    0.671       0.799 0.8110.4610.6010.7870.650Q9-N                                                                                                                                                  0.712 0.6210.4410.5780.7270.688Q9-T                                                                                                                                                  0.803 0.8430.3670.5600.7310.642Q27-N                                                                                                                                                        0.8510.4520.4960.7130.682Q27-T                                                                                                                                                                0.2810.4890.6670.567V2                                                                                                                                                                          0.5580.5730.629V4                                                                                                                                                                                0.7230.595V8V32</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.932</td><td rowspan=1 colspan=2>0.454 0.722</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.626</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=2>0.730 0.658</td><td rowspan=1 colspan=3>0.489 0.574 0.788</td><td rowspan=1 colspan=1>0.617</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.439</td><td rowspan=1 colspan=1>0.726</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.806</td><td rowspan=1 colspan=1>0.573</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=2>0.695 0.663</td><td rowspan=1 colspan=1>0.499</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.597</td><td rowspan=1 colspan=1>0.419</td><td rowspan=1 colspan=1>0.406</td><td rowspan=1 colspan=1>0.490</td><td rowspan=1 colspan=1>0.346</td><td rowspan=1 colspan=2>0.318 0.249</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.708</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.713</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=1>0.623</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.846</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>0.767</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.529</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.703</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.914</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>0.461</td><td rowspan=1 colspan=1>0.601</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.707</td><td rowspan=1 colspan=1>0.712</td><td rowspan=1 colspan=1>0.621</td><td rowspan=1 colspan=1>0.441</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.727</td><td rowspan=1 colspan=1>0.688</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.843</td><td rowspan=1 colspan=1>0.367</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.642</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.713</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.281</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.567</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.558</td><td rowspan=1 colspan=1>0.573</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.723</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>0.750</td></tr></table>