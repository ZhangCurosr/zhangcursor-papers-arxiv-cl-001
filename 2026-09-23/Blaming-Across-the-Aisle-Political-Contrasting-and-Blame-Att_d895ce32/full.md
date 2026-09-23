# Blaming Across the Aisle: Political Contrasting and Blame Attribution in the Danish Parliament

Markus Lundsfryd Jensen<sup>∗</sup> School of Communication and Culture Aarhus University markuslundsfryd@hotmail.dk

Kenneth Christian Enevoldsen Center for Humanities Computing kenneth.enevoldsen@cas.au.dk

## Abstract

Political discourse is widely perceived to be growing more hostile, yet robust evidence remains scarce. This study examines blame attribution in the Danish Parliament from 1997 to 2026, combining a purpose-built classifier, BlameBERT ( F1: 0.80), with multilevel statistical modeling. The classifier is constructed using an annotation-efficient pipeline for blame attribution in low-to-mid resource languages. The results reveal a banana-shaped trajectory, with blame declining until around 2016 before entering a significant and sustained increase in recent years (2019–2026). Government status consistently influenced blame attribution – an effect we term political contrasting – with opposition parties blaming substantially more than governing parties. This effect was moderated by ideology: The blame-dampening effect of governing was less pronounced among right-wing parties, and ideological extremity amplified blame more strongly on the right. In recent years, the interaction between political wing and ideological extremity intensified, suggesting an ideological hardening of the blame rhetoric concentrated on the right of the political spectrum. Taken together, these patterns suggest that the perceived rise in harsh political language reflects not merely a general rhetorical drift but an ideologically asymmetric hardening of political discourse. A sensitivity analysis showed that the conclusions were robust to varying classification thresholds.<sup>1</sup>

## 1 Introduction

Discourse in the media, and society at large, has been concerned with communication within politics taking a turn toward sharper tones (Bilotta et al., 2025; Frisch and Kelly, 2013; Montanaro, 2018;

Rune Egeskov Trust<sup>∗</sup> School of Communication and Culture Aarhus University rune.trust@gmail.com

Sara Kolding Center for Humanities Computing sarakolding@clin.au.dk

Shandwick and Tate, 2019; Frisch, 2022). This perceived shift toward more hostile communication (Bøggild and Jensen, 2025; Muddiman, 2017; Theocharis et al., 2020; Van’t Riet and Van Stekelenburg, 2022), is disputed both internationally and specifically in Danish settings (Barton, 2023; Brusgaard, 2022; Kenski and Jamieson, 2017; Shandwick and Tate, 2019). Much of the empirical evidence comes from social media (Stie, 2021; Frisch, 2022; Becker, 2021; Hansen, 2024). Parliamentary discourse, by contrast, remains comparatively underexamined, despite arguably carrying greater institutional weight. Parliamentary proceedings offer a unique window into policy positions and rhetorical strategies (Proksch and Slapin, 2014), where blame attribution plays a pivotal role, as political actors routinely assign responsibility and criticize opponents to shape public perception and influence debates (Hinterleitner, 2020; Heinkelmann-Wild and Zangl, 2020).

Analyzing blame dynamics requires close attention to the evaluative tone of discourse, as expressions of disapproval and criticism are key indicators of how responsibility is constructed and communicated (Chilton, 2004; Powell Jr, 2004). Emotional language is not merely rhetorical embellishment but a fundamental component of political communication, contributing to processes such as affective polarization, and, potentially, the stability of democratic systems (Garrett et al., 2014; Mason, 2015; Young and Soroka, 2012; Bøggild and Jensen, 2025).

## 1.1 Blame and Political Communication

Blame is inherently evaluative, as it combines causal attribution with negative sentiment, distinguishing it from mere criticism or disagreement (Bilotta et al., 2025). This duality is theoretically grounded in moral psychology, where blame functions as a signal that the blaming party adheres to norms which the blamed party violates (Shoemaker and Vargas, 2021a). In political contexts, this norm-signaling logic generates a clear prediction, where actors in opposition, who position themselves against a governing party, have the strongest incentive to blame, as doing so simultaneously highlights their opponents’ policy failures and differentiates their own position. We refer to this as political contrasting, a rhetorical strategy in which opposition parties systematically blame more than governing parties (Bilotta et al., 2025; Frimer et al., 2023; Van’t Riet and Van Stekelenburg, 2022; Muddiman, 2017). An extension of this logic applies to ideological extremity, which we refer to as wingness; the farther a party is from the political center, the more policies it disputes. Since ideologically extreme parties dispute policies from more parties than ideologically centered parties, and assuming this disagreement translates into rhetorical attacks, we expect an increase in blame attribution from these more extreme parties (Elmelund-Præstekær, 2010).

## 1.2 Emotional Tone in Parliamentary Discourse

Cross-national analyses of European parliaments find that negative and neutral sentiment dominate, that positive sentiment is consistently the least frequent category, and that shifts in emotional tone mirror country-specific political conditions rather than reflecting a single universal pattern (Mochtak et al., 2025; Lehtosalo and Nerbonne, 2024; Rheault et al., 2016). Research specifically on Danish and Dutch intra-party speeches finds that emotional arousal has increased over time, even as the overall valence of sentiment has remained stable (Schumacher et al., 2019). This suggests a growing intensity of political language rather than a directional shift – an intensity we expect extends to blame. However, sentiment is not equivalent to blame: a sentence can be negative without attributing responsibility to any actor.

Given the impact of political discourse on the health of democracy, we investigate how blame attribution varies with structural political characteristics and how this relates to contested empirical claims about the recent rise of hostile communication in politics. In order to ensure that the blame identification reflects the rhetorical act itself, rather than prior assumptions about which parties typically blame, this paper follows precedent in the literature that blame can be contained exclusively in linguistic markers (Park et al., 2021; Liang et al.,

2019).

## 1.3 Contributions and Hypotheses

The contributions of this paper are three-fold:

Firstly, we annotate a dataset of direct utterances from politicians from the Danish parliament (1997- 2026) using an annotation-efficient pipeline computationally assisted by the NLI model DEBATE (Burnham et al., 2026) and computationally validated using downstream performance on a manually constructed test-set.

Secondly, we publish the first model for blame detection in a Danish political context.

Thirdly, we estimate general temporal and political aspects of blame attribution and contrast these results with more recent developments (2019-2026) to quantify the evolution of blame attribution in the Danish Parliament. This analysis is guided by the following hypotheses:

## Hypotheses

H1: The rate of blame attribution in Danish Parliament has increased over time — (H1.1) particularly so in recent years (2019–2026).

H2: Political extremity and opposition-status (political contrasting) increase blame attribution — (H2.1) particularly so in recent years (2019–2026).

## 2 Methods

The following methods section is presented in three parts: dataset curation, model development, and analysis. See Figure 1 for a visual guide of the process and Appendix A.0.1 for a more detailed version.

## 2.1 Dataset

The ParlSpeechV2 dataset covers transcripts from the Danish Parliament from 07/10-1997 to 20/12- 2018 and was collected by scraping the Danish Parliaments’ website (Rauh and Schwalbach, 2020). We fetched more recent transcripts from the Danish Parliament’s SFTP server. At the time of the fetch, the earliest transcribed debate was from 06/10- 2009, and the most recent from 26/02-2026. The datasets were merged from 20/12-2018, which is the last date in ParlSpeechV2, with the next available date from our fetch, 09/01-2019, for a full dataset covering 07/10-1997 to 26/02-2026. Paragraphs spoken by a chairman were excluded. Minimal standardization was required between the sets, but ParlSpeechV2 contains certain metadata that our fetch did not, which was removed. See Appendix A.1 for a consistency check of temporal data characteristics.

![](images/6ea394bd7b1c90de0fe0e1b974a8a96beb23664f04b4713e1fa2edaa96602a04.jpg)  
Figure 1: Flowchart of creation of datasets and models (more detailed version in Appendix A.0.1).

Preprocessing and Labeling: In order to create an annotation-efficient pipeline computationally assisted by DEBATE, two of its architectural properties require upstream preprocessing of the input data. First, being a fine-tuned DeBERTa-V3 based zero-shot classifier (Laurer et al., 2023), it is not multilingual by design – necessitating machine translation. Second, input is restricted to 512 tokens, which requires splitting the data into sentences instead of full paragraphs. Sentence segmentation was done using DaCy<sup>2</sup> (Enevoldsen et al., 2021). Sentences shorter than five characters or containing parentheses were excluded, reducing the number of sentences from 6,553,133 to 5,598,994 (85%).

500,500 sentences were then randomly sampled from the cleaned dataset for use in model training, validation, and testing, while the remaining 5,098,494 sentences were held out for inference. Only the training data was machine translated using Opus-MT-da-en (Tiedemann et al., 2023) and labeled with DEBATE, done by passing each sentence through a set of hypothesis templates:

T1. Based on this text, the author’s attitude towards

others is best described as {}.

T2. Overall, the author’s stance toward others in this passage is {}.

T3. The overall feeling that the author communicates toward others in this text is best described as {}.

T4. According to this passage, the author’s reaction to others’ conduct can be described as {}.

T5. From the way others are described, the author’s expression towards them is best described as {}.

The candidate labels passed to the hypothesis templates were “blame”, “praise”, and “neutral”. The choice of “blame” versus “praise” was based on a general consensus in the literature that these two concepts can be considered antonyms (Tognazzini and Coates, 2024; Williams). The "neutral" label was added to account for edge cases, such as a high probability of both blame and praise occurring in the same sentence.

Absolute probabilities for each label were extracted by DEBATE. A threshold for the classification of blame in a sentence was determined as the probability of blame being ≥ .80 and greater than both “praise” and “neutral”. These labels were then mapped back onto the original Danish sentence.

Training Data: Based on the five hypothesis templates, five separate datasets which we call "Datasets of Increasing Agreement Levels" (DI-ALs) were constructed. For each DIAL-n, a sentence was given a positive label if at least n templates agreed, ranging from the most conservative, n = 5 (DIAL-5), to the least conservative n = 1 (DIAL-1). The prevalence of blame was 1.71%, 1.18%, 0.91%, 0.70%, and 0.50% for DIAL-1 to DIAL-5, respectively.

Gold Label Test Set: Due to the class imbalance, upsampling of blame was performed by randomly sampling 250 blame and 250 non-blame sentences from DIAL-1, following established practices (Bigoulaeva et al., 2022; Rathpisey and Adji, 2019; Anju et al., 2024).

These 500 samples were manually annotated by two of the authors (males, Danish, age 24-25), who were instructed to follow the definition of Bilotta et al. (2025) of blame as a causal utterance with negative sentiment. For examples of such sentences, see Appendix A.2. Inter-annotator agreement was 84.8% (Cohen’s Kappa = .676). Only sentences where both annotators agreed were included in the test set, totaling 424 samples (148 blame, 34.9%).

## 2.2 Model Development

20,000 sentences extracted from each DIAL were used for model training. In each subset, true labels were upsampled by including all true labels from each template.

The training pipeline was a full precision LoRA fine-tune (Hu et al., 2021) with a rank of 64 and alpha scaling at 128 using focal loss. A hyperparameter grid search was performed over the five DIAL subsets, with three learning rates; $1 e ^ { - 5 }$ $1 e ^ { - 4 } , 5 e ^ { - 4 }$ , all with a linear learning rate decay using the Huggingface trainer API (Wolf et al., 2020). Three alpha scaling constants were applied for the focal loss function: raw class weights, class weights to the power of two-thirds, and the square root of class weights. The Gamma focusing parameter was kept constant at 2.0.

mmBERT (Marone et al., 2025) was trained and validated on an 80-20 split of each DIAL subset, and performance was measured by maximizing the Matthews Correlation Coefficient (MCC) on the validation split.

All experiments were tracked and shared using Weights and Biases, see Appendix A.3.

Model Performance The model with the highest MCC for each of the DIALs was tested on the goldlabeled test set. Both DIAL-5 and DIAL-4 yielded models with the same macro-F1. However, due to a better trade-off between precision and recall, we chose to use the DIAL-5 model, which was trained with a learning rate of $1 e ^ { - 4 }$ and an Alpha parameter of the square root of the class weights. This model, called BlameBERT, obtained an average recall score of .81, an average precision score of .80, and an macro-averaged F1 score of .80. For performance metrics across all DIALs, see Appendix A.3 and A.4. To investigate if errors were partyspecific, we examined performance across parties on the test set, see Appendix A.6. No systematic error rate was found.

Model Comparison: A baseline for zero-shot blame classification was established using the Qwen 3:0.6B embedding model (Zhang et al., 2025) and the generative Qwen 3.5:9B (Qwen Team, 2026). The results of this classification are shown in Table 1.

Based on these results, we argue that Blame-BERT is best suited to the task, even before taking into account the computational cost of the generative Qwen model. More details about computation of baselines can be found in Appendix A.5.

<table><tr><td colspan="4">Class 1 (Blame)</td><td rowspan="2">Average Macro F1</td></tr><tr><td></td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Embedding</td><td>0.52</td><td>0.87</td><td>0.65</td><td>0.67</td></tr><tr><td>Generative</td><td>1.00</td><td>0.42</td><td>0.60</td><td>0.75</td></tr><tr><td>BlameBERT</td><td>0.72</td><td>0.79</td><td>0.75</td><td>0.80</td></tr></table>

Table 1: Classification performance for Qwen embedding and -generative models against BlameBERT.

## 2.3 Analysis

Preprocessing: Only parties that still existed and actively practiced politics within continental Denmark at the time of the analysis were included. Additionally, all utterances from non-attached members of parliament were excluded, reducing the number of unique sentences to 4,938,119 (96.9%). The final 13 parties are noted in Table 2 along with their political wing and degree of wingness.

Wing is defined based on the general ideological placement of the parties from the Chapel Hill Expert Survey (Rovny et al., 2025). Parties with positive standardized scores are placed on the right wing, while parties with negative scores are placed on the left wing. The zero threshold reflects the sample mean of the standardized ideology index. Wingness is defined as the absolute standardized distance from the mean. See Appendix A.7 for the calculation.

All remaining sentences were classified using blameERT, and aggregated by month, year, and party, resulting in each row of a dataset containing a month-wise count of total sentences and sentences containing blame for each party. The resulting dataset contained a total of 2,529 observations. For recent years (2019-2026), the number of observations was 810.

Analysis H1: To investigate how the blame rate changes over time (1997-2026), we fit a series of negative binomial mixed-effects models with linear parameterization of increasing complexity; intercept-only, linear, and quadratic for time, using glmmTMB (McGillycuddy et al., 2025), and compared using likelihood ratio tests (LRT) using anova (R Core Team, 2025).

Formally, let $Y _ { i t }$ denote the blame count for party i at scaled time t, modeled as $\begin{array} { r l } { Y _ { i t } } & { { } \sim } \end{array}$ $\mathrm { N e g B i n } ( \mu _ { i t } , \phi )$ . The linear predictor is given by:

$$
\log ( \mu _ { i t } ) = \log ( S _ { i t } ) + \beta _ { 0 } + \beta _ { 1 } t + \beta _ { 2 } { \bf G o v } _ { i t } + u _ { i }\tag{1}
$$

<table><tr><td colspan="2">Party Wingness</td></tr><tr><td>Left wing</td><td>(-)</td></tr><tr><td> $\mathrm { E L }$  Enhedslisten</td><td>1.74</td></tr><tr><td> $\mathring \mathrm { A }$  Alternativet</td><td>1.39</td></tr><tr><td> $\mathrm { S F }$  Socialistisk Folkeparti</td><td>1.15</td></tr><tr><td> $\mathbf { S }$ </td><td>Socialdemokratiet 0.60</td></tr><tr><td>RV Radikale Venstre</td><td>0.15</td></tr><tr><td>Right wing</td><td>(+)</td></tr><tr><td>KD Kristendemokraterne</td><td>0.02</td></tr><tr><td>M Moderaterne</td><td>0.10</td></tr><tr><td>V Venstre</td><td>0.59</td></tr><tr><td>K Konservative</td><td>0.63</td></tr><tr><td>DD Danmarksdemokraterne</td><td>0.80</td></tr><tr><td>DF Dansk Folkeparti</td><td>0.91</td></tr><tr><td>LA Liberal Alliance</td><td>0.96</td></tr><tr><td>NB Nye Borgerlige</td><td>1.34</td></tr></table>

Table 2: Overview of parties and their wingness score, derived from the Chapel Hill Survey, grouped by wing.

Where log $( S _ { i t } )$ is an offset for the sentences uttered by party i at time t, Gov is a binary indicator of whether party i is in government at time t included as a controlling fixed-effect variable, and $u _ { i } ~ \sim$ ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ is a random intercept at the party level. The coefficient of primary interest is $\beta _ { 1 }$ , which captures whether the overall tendency to attribute blame has shifted linearly throughout the period.

Analysis H1.1: To assess whether the temporal dynamics of blame attribution differed in more recent years, an equivalent model was estimated on observations from 2019-2026.

Analysis H2: To investigate whether structural political characteristics predict blame attribution over the entire period (1997-2026), we fit a series of negative binomial mixed-effects models with linear parameterization of increasing complexity using glmmTMB (McGillycuddy et al., 2025), compared via likelihood ratio tests using anova (R Core Team, 2025). The temporal trend from Analysis H1 was included as a fixed-effect control variable.

Formally, let $Y _ { i t }$ denote the blame count for party i at scaled time t, modeled as $Y _ { i t } \sim$ NegBin $( \mu _ { i t } , \phi )$ . The linear predictor is given by:

$$
\begin{array} { r l } & { \log ( \mu _ { i t } ) = \log ( S _ { i t } ) + \beta _ { 0 } + f ( t ) } \\ & { + \beta _ { 1 } { \bf G o v } _ { i t } + \beta _ { 2 } { \bf W i n g } _ { i } + \beta _ { 3 } W i n g n e s s _ { i } + u _ { i } } \end{array}\tag{2}
$$

Where log $( S _ { i t } )$ is an offset for sentences uttered by party i at time $t , f ( t )$ denotes the temporal trend from Analysis H1 included as a controlling variable, $\operatorname { G o v } _ { i t }$ is a binary indicator of whether party i is in government at time t, Wing<sub>i</sub> is a categorical indicator of party i for political wing affiliation, $W i n g n e s s _ { i }$ is a continuous measure of distance from the political center for party $i ,$ and $u _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ is a party-level random intercept. The coefficients of primary interest are $\beta _ { 1 } , \beta _ { 2 }$ , and $\beta _ { 3 }$ , capturing the effects of government status, ideological wing affiliation, and wingness on the blame rate, respectively.

Analysis H2.1: To investigate political predictors of blame attribution in more recent years, a separate model was estimated on observations from 2019-2026, using the same approach as Analysis H2, except the temporal change in blame rate from Analysis H1.1, which was included as a fixed-effect control variable.

Sensitivity and Political Agenda: The blame labels produced by BlameBERT carry classification uncertainty that is not accounted for in the statistical models, potentially reducing statistical power and obscuring true effect sizes. Inspection of BlameBERT performance indicated that the model overpredicts blame (Table 1). However, if overprediction is balanced across all focal predictors, the relative effect of political characteristics will still hold. To inspect the robustness of results, a sensitivity analysis was conducted by applying increasingly conservative probability thresholds. For full analysis, see Appendix A.9

Additionally, the models did not control for the topic or agenda of the parliamentary proceedings, which can affect sentiment and emotional language in political settings (Pätz et al., 2025; Ristilä et al., 2026). Government status of Danish parties has also been found to differ topically in their speeches (Navarretta and Haltrup Hansen, 2024). A minimalistic analysis was implemented using ManifestoBERTa (Burst et al., 2024) to classify political topics on a sentence level, and investigate if parties in government and in opposition generally differed in their agenda, and if controlling for such topics alters the effect of government status on blame attribution. For full analysis, see Appendix A.10

## 3 Results

In the following, we present the main results of our analysis. For extended results, we refer to appendix $_ { \mathrm { A } . 8 }$

Results H1: The LRT indicated that adding time as a linear predictor (M1.1) significantly improved model fit compared to an intercept-only model $( \mathbf { M } 1 . 0 ) ~ \chi ^ { 2 } ( 1 ) ~ = ~ 6 . 4 9 , p ~ = ~ . 0 1 0 9$ . Including a quadratic term (M1.2) for time further significantly improved model fit $\chi ^ { 2 } ( 1 ) = 4 . 4 4 , p = . 0 3 5 2$ compared to M1.1.

For M1.2, a significant positive quadratic effect of scaled time was found, $b ~ = ~ 0 . 0 1 3 5 , S E ~ =$ $0 . 0 0 6 4 0 , z = 2 . 1 1 , p = . 0 3 4 7$ . The linear term for time was negative but not significant, b = $- 0 . 0 0 9 8 0 , S E = 0 . 0 0 6 0 0 , z = - 1 . 6 3 , p > . 0 5$ The intercept was negative and significant b = $- 2 . 3 8 , S E = 0 . 0 6 4 5 , z = - 3 6 . 9 , p < . 0 0 1$ . (Fig. 2, left).

![](images/62231a50bea460836e4431562a6fae7cd70728b6a6d042c92b08aeb2466d27a5.jpg)  
Figure 2: Estimated effect of time on blame for both time periods with 95% CI. Left: Analysis H1, Time post October 1997, Right: Analysis H1.1, Time post January 2019

Results H2: LRT of the models that evaluated predictors of blame throughout the entire period indicated that including government status (M2.1) significantly improved model fit compared to the intercept-only model $( \mathbf { M } 2 . 0 ) , \chi ^ { 2 } ( 1 ) = 6 6 8 , p \ <$ .001. Adding ideological wing affiliation as a predictor (M2.2) did not provide a significantly better fit, $\chi ^ { 2 } ( 1 ) = 0 . 9 5 1 , p > . 0 5$ compared to M2.1. Adding wingness as a predictor (M2.3) did not provide a significantly better fit compared to M2.1, $\chi ^ { 2 } ( 2 ) = 5 . 0 8 , p > . 0 5$ . However, adding an interaction between wing affiliation and wingness (M2.4) yielded a significantly better fit than M2.1, $\chi ^ { 2 } ( 3 ) = 1 0 . 6 2 , p = . 0 1 4 0$ . Furthermore, the addition of an interaction between government status and wing affiliation (M2.5) further improved model fit, $\chi ^ { 2 } ( 1 ) = 9 . 3 0 , p = . 0 0 2 2 9$ compared to M2.4.

For M2.5, with left wing as the reference category, a significant negative effect of government status emerged, $b = - 0 . 4 7 8 , S E = 0 . 0 2 0 5 , z =$ $- 2 3 . 3 , p < . 0 0 1$ . The main effect of right wing was not significant, $b = - 0 . 3 1 6 , S E = 0 . 1 6 9 , z =$ 二 $- 1 . 8 7 , p > . 0 5 ,$ nor was the main effect of wing-$n e s s , b = 0 . 0 8 5 0 , S E = 0 . 0 9 6 7 , z = 0 . 8 7 8 , p >$ .05. The interaction between government status and right wing was positive and significant, $b = 0 . 1 0 5 , S E = 0 . 0 3 4 6 , z = 3 . 0 4 , p = . 0 0 2 3 6 ,$ as was the interaction between right wing and wing-$n e s s , \ : b \ : = \ : 0 . 5 5 4 , S E \ : = \ : 0 . 1 8 4 , z \ : = \ : 3 . 0 1 , p \ : =$ .00265. The intercept was also significant, b = $- 2 . 5 2 , S E = 0 . 1 0 3 , z = - 2 4 . 4 , p < . 0 0 1 \ : ( \mathrm { F i g } .$ 3 and 4, left).

![](images/efded90cb7f2f1b6d181144f93545c87437bbaa6092c45adfcd906010b0be2c7.jpg)  
Figure 3: Estimated effect of Government status on the two political wings for both time periods with 95% CI. Left: Analysis H2, Right: Analysis H2.1

Results H1.1: LRT of the models that evaluated counts of blame as predicted by time in more recent years found that the linear model (M3.1) provided a significantly better fit than the intercept model $( \mathbf { M } 3 . 0 ) , \chi ^ { 2 } ( 1 ) = 1 0 7 , p < . 0 0 1$ . Adding a quadratic term (M3.2) did not further improve model fit $\chi ^ { 2 } ( 1 ) = 3 . 7 4 , p > . 0 5$

For M3.1, the estimated effect of the linear term of scaled time on the blame count in recent years was significant and positive, $b = 0 . 0 9 2 1 , S E =$ 0.00861, z = 10.7, p < .001. The intercept was also significant, $b = - 2 . 2 9 , S E = 0 . 0 6 8 6 , z =$ $- 3 3 . 5 , p < . 0 0 1 . \ : ( \mathrm { F i g . } \ : 2 , \mathrm { r i g h t } )$

Results H2.1: LRT of the predictors of blame in recent years found that adding government status as a predictor (M4.1) significantly improved model fit compared to an intercept-only model $( \mathbf { M } 4 . 0 ) \chi ^ { 2 } ( 1 ) = 1 1 9 , p < . 0 0 1$ . Adding ideological wing affiliation (M4.2) did not further improve model fit, $\chi ^ { 2 } ( 1 ) = 0 . 5 5 6 , p > . 0 5$ , compared to M4.1. Neither did adding wingness as a predictor (M4.3) compared to $\mathbf { M } 4 . 1 , \chi ^ { 2 } ( 2 ) \ : = \ : 0 . 9 8 1 , p \ : >$ .05. However, the addition of the interaction between wing affiliation and wingness (M4.4) did improve model fit significantly compared to M4.1, $\chi ^ { 2 } ( 3 ) = 1 5 . 5 , p = . 0 0 1 4 3$ . Adding an interaction between government status and wing (M4.5) did not further improve model fit compared to M4.4, $\chi ^ { 2 } ( 1 ) = 0 . 8 5 7 , p > . 0 5$

![](images/d2e3396ac8ce92b30303e71d53b0a12072e2ec23253d72a424878ee9a0c85f33.jpg)  
Figure 4: Estimated effect of wing affiliation and wingness for both time periods with 95% CI. Left: Analysis H2, Right: Analysis H2.1

For M4.4, with left wing as the reference category, the effect of government status was negative and significant, $\textit { b } = \ - 0 . 5 1 5 , S E$ $0 . 0 4 4 2 , z ~ = ~ - 1 1 . 7 , p ~ < ~ . 0 0 1$ The main effect of right wing was negative and significant, $b = - 0 . 5 2 2 , S E = 0 . 1 5 1 , z = - 3 . 4 5 , p < . 0 0 1$ while the main effect of wingness was not significant, $b = - 0 . 1 3 6 , S E = 0 . 0 9 0 8 , z = - 1 . 4 9 , p >$ .05. The interaction between right wing and wingness was positive and significant, $b = 0 . 8 1 7 , S E =$ $0 . 1 6 2 , z \ = \ 5 . 0 5 , p \ < \ . 0 0 1$ . The intercept was also significant, $b = - 2 . 4 1 , S E = 0 . 1 0 1 , z =$ $- 2 3 . 8 , p < . 0 0 1$ (Fig. 3 and 4, right).

Sensitivity and Political Topic: The results of sensitivity analysis and an investigation of structural differences in agendas across governing parties can be found in Appendix A.9 and A.10, respectively.

## 4 Discussion

Blame Attribution: Examining the effect of time on blame attribution (H1) revealed a banana-shaped effect throughout the 29-year period, suggesting that blame rhetoric initially declined – reaching a global minimum in ∼ April 2016 – then increased at an accelerating rate (Fig. 2, left). This pattern is consistent with the public belief of an increasingly harsh political scene. The timing coincides with Denmark’s 2015 immigration crisis, after which rhetoric toward immigrants reportedly hardened across the political spectrum, even among parties not typically associated with conservative immigration views (Navarretta et al., 2022). A significant positive linear trend was also found in recent years (H1.1), mirroring the overall-period increase in Analysis H1 (Fig. 2, right).

Political Contrasting: Analysis of structural political characteristics and their relationship with blame (H2) showed that government status had a substantial blame-dampening effect, with governing left-wing parties only blaming ∼ 62% compared to left-wing opposition parties – a finding consistent with the idea of political contrasting. This blame-dampening effect was ∼ 11% weaker among right-wing parties (Fig. 3, left). The main effects of wing affiliation and wingness were not significant. Given the interactions, these reflect the difference between wings for opposition parties at zero wingness and the effect of wingness among left-wing parties, respectively. The significant interactions thus underline that the effects of ideology are not absent, but differentiated across political affiliation.

The effect of wingness also differed across opposing beliefs. The increase in blame was stronger for the right wing, with each one unit increase resulting in $a \sim 7 4 \%$ increase in blame attribution compared to parties on the left wing (Fig. 4, left). These findings partially support those of Elmelund-Præstekær (2010), as more ideologically extreme parties seem to be associated with an increase in blame. However, our findings only support this effect being true for right-wing parties. The blame dampening effect of government held in recent years (H2.1), where government status remained a significant negative predictor of blame, with governing parties blaming only 60% compared to the opposition. As the interaction between government status and wing did not improve model fit in this period, this effect applies across both wings (Fig.

3, right).

In contrast, the main effect of ideological wing affiliation was significant in this period, with rightwing parties attributing less blame than left-wing parties (Fig. 3, right). However, given the interaction with wingness, this effect is estimated at wingness being zero, and thus reflects the estimated difference between wings at the arbitrary state of all parties being politically placed at the center, see low wingness on the right-hand side of Figure 4. As such, the results indicate that when computing wing differences across the mean value of wingness, which is more representative of the actual parties associated with each wing, left-wing parties show substantially reduced blame attribution compared to right-wing parties, see right side of Figure 3. The main effect of wingness, reflecting its effect among left-wing parties, was not significant, indicating no systematic relationship between blame and ideological extremity on the left. However, the interaction between ideological wing and wingness was again significant, showing the same trend as for the entire period, where parties on the right show a substantially greater positive relationship between their degree of wingness and increase in blame attribution (Fig. 4, right). Crucially, this effect was intensified, with each unit increase in wingness resulting in a 126% increase in blame attribution among right-wing parties compared to left-wing parties.

Collectively, these findings describe a parliament in which blame is not only rising but is increasingly structured by ideology, intensifying most among ideologically extreme right-wing parties. As the findings of Hinterleitner (2020) also underline, these ideologically extreme right wing parties could potentially be pulling moderate parties toward more blame generation in order not to be overshadowed in the debates. This dynamic of blame becoming a more frequent occurrence within politics could spread, as parliamentary debate is at the core of democratic institutions: The rhetoric exchanged here helps set the stage for how political conflict is handled throughout society, and such an asymmetric hardening may therefore carry consequences that extend beyond the chamber itself.

Sensitivity and Political Topic: The sensitivity analysis showed strong robustness, as the direction and significance of findings hold at all thresholds, with the interaction effect between government status and ideological wing in Analysis H2 being the one exception. This effect was the same direction, but not statistically significant at the most conservative threshold, see appendix A.9. The sensitivity analysis underlines the robustness of the relative effects, while acknowledging that blame is a difficult construct to identify – a point corroborated by the Cohen’s Kappa of .676 on the test set, and the many different definitions of blame in the literature (Portmore, 2022; Malle et al., 2014; Tognazzini and Coates, 2024; Susan et al., 2011; Bilotta et al., 2025; Shoemaker and Vargas, 2021b; Shoemaker, 2015; Wallace, 2011; Smith, 2012; Malle et al., 2013; Wang, 2024).

Governing and opposition parties did not notably differ in their topical agendas, and controlling for topic did not alter the blame-dampening effect of government status. For the full analysis, see Appendix A.10.

## 5 Conclusion

This study examined blame attribution in the Danish Parliament from 1997 to 2026, combining a purpose-built classifier with multilevel statistical modeling across nearly five million parliamentary sentences. Our results reveal a banana-shaped temporal trajectory of blame declining through 2016 before entering a sustained and accelerating increase in recent years, a trend corroborated by a standalone analysis of the temporal drift in blame from 2019 to 2026. Government status consistently suppresses blame attribution across both periods, consistent with political contrasting. Throughout the full period, this effect is moderated by ideological wing affiliation, with right-wing parties showing a weaker reduction in blame when in government and a stronger amplification of blame as wingness increases. In the more recent period, the government effect applied across both wings, while the interaction between wing and wingness intensified compared to the full period. A sensitivity analysis on the classification threshold confirmed that the direction and significance of the core findings are robust across thresholds. Taken together, these findings offer partial empirical support for the perceived rise in political harshness in Denmark, while underlining that this shift is nonuniform across the Danish political landscape.

## Limitations

Measurement limitations: Our blame labels rest on a multi-step pipeline with each step potentially introducing noise. Training labels were generated by machine-translating Danish sentences into English before applying DEBATE. This was a necessary step given the architectural constraints of DEBATE, but the translation model may distort the signal DEBATE itself responds to, consequently propagating this into the training data. Furthermore, translation artifacts might be particularly pronounced for ironic, rhetorically indirect phrasing, or language specific idioms. DEBATE itself was primarily developed and validated on English political text, and translation from Danish political text to English might not fully encapsulate rhetorical language-specific differences. As BlameBERT is trained on these silver labels, systematic biases in DEBATE’s zero-shot classifications could be inherited rather than corrected by the fine-tuned model.

Modeling limitations: The predictors covering wing affiliation and wingness are time-invariant characteristics that vary across only 13 parties, resulting in limited between-party variance. Although the random intercept permits their inclusion, the effective sample size for these estimates is closer to 13 than to the full observation count, and standard errors should be interpreted accordingly. Relatedly, wing and wingness are both derived from the same underlying "lrgen" measure from the Chapel Hill Expert Survey (Rovny et al., 2025). Although we treat them as capturing distinct (i.e. categorical versus continuous) aspects of ideological position, some information leakage between them is unavoidable, and should be kept in mind when interpreting their independent contributions. A mild zero-deflation was also detected across all models, as the negative binomial family allocates more probability mass to zero counts than observed in the data. Given its negligible scale and the otherwise well-specified dispersion structure, this is unlikely to materially affect substantive conclusions.

Support parties: Due to the tradition of minority governments in Denmark, parties outside government often act as formal support parties, and are therefore not strictly in opposition, although our binary government indicator treats them as such. Since such arrangements are typically formed within the same political wing, with the notable exception of the cross-wing government formed in

2022, the inclusion of wing affiliation and its interaction with government status partially accounts for this.

Scope limitations: The models estimated for the recent period (2019-2026) draw on a smaller sample and span a period marked by considerable political disruption. This includes the COVID-19 pandemic and multiple government transitions, each of which could confound the observed temporal and political trends independent of any genuine shift in rhetorical norms. Our classifier also identifies that blame occurs, but not whom it targets. A party could register high blame-rates while primarily directing blame at external or non-partisan targets like the EU, the pandemic, or global markets, rather than domestic political rivals. This is a meaningfully different phenomenon from the rivaldirected political contrasting that our theoretical framework emphasizes.

## Acknowledgments

Part of the computation for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark.

## References

Anju, Inderdeep Kaur Aulakh, and Raj Kumari. 2024. The impact of employing sampling-based strategies to address the issue of an imbalanced dataset in hate speech and offensive language categorization. In 2024 15th International Conference on Computing Communication and Networking Technologies (ICC CNT), pages 1–7.

Camilla Alexandra Barton. 2023. Er tonen i Folketinget for hård? Det er der delte meninger om - TV 2. nyheder.tv2.dk.

Emma Joisten Becker. 2021. "Bagsiden af medaljen ved sociale medier er ved at være større end forsiden".

Irina Bigoulaeva, Viktor Hangya, Iryna Gurevych, and Alexander Fraser. 2022. Addressing the challenges of cross-lingual hate speech detection. arXiv preprint arXiv:2201.05922.

Francesco Bilotta, Alberto Binetti, and Giacomo Manferdini. 2025. Blameocracy: Causal Attribution in Political Communication. arXiv preprint. ArXiv:2504.06550 [econ].

Maiken Brusgaard. 2022. Er tonen i dansk politik for hård? Her er otte eksempler på, at den altid har været det - TV 2. nyheder.tv2.dk.

Michael Burnham, Kayla Kahn, Ryan Yang Wang, and Rachel X. Peng. 2026. Political debate: Efficient zero-shot and few-shot classifiers for political text. Political Analysis, 34(3):329–343.

Tobias Burst, Pola Lehmann, Simon Franzmann, Denise Al-Gaddooa, Christoph Ivanusch, Sven Regel, Felicia Riethmüller, Bernhard Weßels, Lisa Zehnter, Wissenschaftszentrum Berlin Für Sozialforschung (WZB), and Institut Für Demokratieforschung Göttingen (IfDem). 2024. Manifestoberta.

Troels Bøggild and Carsten Jensen. 2025. When politicians behave badly: Political, democratic, and social consequences of political incivility. American Journal ofPolitical Science, 69(3):1064–1081.

Paul Chilton. 2004. Analysing political discourse: Theory and practice. routledge.

Christian Elmelund-Præstekær. 2010. Beyond american negativity: Toward a general understanding of the determinants of negative campaigning. European Political Science Review, 2(1):137–156.

Kenneth Enevoldsen, Lasse Hansen, and Kristoffer Nielbo. 2021. DaCy: A Unified Framework for Danish NLP. arXiv preprint. ArXiv:2107.05295.

Jeremy A Frimer, Harinder Aujla, Matthew Feinberg, Linda J Skitka, Karl Aquino, Johannes C Eichstaedt, and Robb Willer. 2023. Incivility is rising among american politicians on twitter. Social Psychological and Personality Science, 14(2):259–269.

Nathalie Damgaard Frisch. 2022. Hvis politikerne ønsker en bedre tone i debatten, skal de måske starte med sig selv, viser forskningen.

S Frisch and S Kelly. 2013. Politics to the extreme: American political institutions in the twenty-first century. Springer.

R Kelly Garrett, Shira Dvir Gvirsman, Benjamin K Johnson, Yariv Tsfati, Rachel Neo, and Aysenur Dal. 2014. Implications of pro-and counterattitudinal information exposure for affective polarization. Human communication research, 40(3):309–332.

Søren Peter Hansen. 2024. Tænketanken Prospekt: Vi har brug for at genfinde etikken i politik - Altinget. Section: Etik og Tro.

Tim Heinkelmann-Wild and Bernhard Zangl. 2020. Multilevel blame games: Blame-shifting in the european union. Governance, 33(4):953–969.

Markus Hinterleitner. 2020. Blame Games in the Political Sphere, page 18–39. Cambridge Studies in Comparative Public Policy. Cambridge University Press.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint. ArXiv:2106.09685.

Kate Kenski and Kathleen Hall Jamieson. 2017. The Oxford handbook ofpolitical communication. Oxford University Press.

Moritz Laurer, Wouter van Atteveldt, Andreu Casas, and Kasper Welbers. 2023. Building Efficient Universal Classifiers with Natural Language Inference. arXiv preprint.

Suvi Lehtosalo and John Nerbonne. 2024. Detecting emotional polarity in Finnish parliamentary proceedings. In Proceedings ofthe 4th Workshop on Computational Linguisticsfor the Political and Social Sciences: Long and short papers, pages 90–100, Vienna, Austria. Association for Computational Linguistics.

Shuailong Liang, Olivia Nicol, and Yue Zhang. 2019. Who Blames Whom in a Crisis? Detecting Blame Ties from News Articles Using Neural Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):655–662.

Bertram Malle, Steve Guglielmo, and Andrew Monroe. 2014. A Theory of Blame. Psychological Inquiry, 25:147–186.

Bertram F. Malle, Steve Guglielmo, and Andrew E. Monroe. 2013. Moral, cognitive, and social: The nature of blame. In Social thinking and interpersonal behavior, Sydney symposium of social psychology, pages 313–331. Psychology Press, New York, NY, US.

Marc Marone, Orion Weller, William Fleshman, Eugene Yang, Dawn Lawrie, and Benjamin Van Durme. 2025. mmbert: A modern multilingual encoder with annealed language learning. Preprint, arXiv:2509.06888.

Lilliana Mason. 2015. “i disrespectfully agree”: The differential effects of partisan sorting on social and issue polarization. American journal of political science, 59(1):128–145.

Maeve McGillycuddy, David I. Warton, Gordana Popovic, and Benjamin M. Bolker. 2025. Parsimoniously fitting large multivariate random effects in glmmTMB. Journal ofStatistical Software, 112(1):1– 19.

Michal Mochtak, Peter Rupnik, Taja Kuzman, and Nikola Ljubešic. 2025. Parlasent: mapping senti-´ ment in political discourse with large language models. Political Research Exchange, 7(1):2508377.

Domenico Montanaro. 2018. Poll: Nearly 4 in 5 voters concerned incivility will lead to violence. NPR.

Ashley Muddiman. 2017. Personal and public levels of political incivility. International Journal of Communication, 11:21–21.

Costanza Navarretta and Dorte Haltrup Hansen. 2024. Government and opposition in Danish parliamentary debates. In Proceedings of the IV Workshop on Creating, Analysing, and Increasing Accessibility ofParliamentary Corpora (ParlaCLARIN) @ LREC-COLING 2024, pages 154–162, Torino, Italia. ELRA and ICCL.

Costanza Navarretta, Dorte Haltrup Hansen, and Bart Jongejan. 2022. Immigration in the manifestos and parliament speeches of Danish left and right wing parties between 2009 and 2020. In Proceedings ofthe Workshop ParlaCLARIN III within the 13th Language Resources and Evaluation Conference, pages 71–80, Marseille, France. European Language Resources Association.

Kunwoo Park, Zhufeng Pan, and Jungseock Joo. 2021. Who blames or endorses whom? entity-to-entity directed sentiment extraction in news text. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4091–4102, Online. Association for Computational Linguistics.

Lukas Pätz, Moritz Beyer, Jannik Späth, Lasse Bohlen, Patrick Zschech, Mathias Kraus, and Julian Rosenberger. 2025. Analyzing German Parliamentary Speeches: A Machine Learning Approach for Topic and Sentiment Classification. Preprint, arXiv:2508.03181.

Douglas W Portmore. 2022. A comprehensive account of blame: Self-blame, non-moral blame, and blame for the non-voluntary.

G Bingham Powell Jr. 2004. Political representation in comparative politics. Annu. Rev. Polit. Sci., 7(1):273– 296.

Sven-Oliver Proksch and Jonathan B Slapin. 2014. The politics ofparliamentary debate: Parties, rebels and representation. Cambridge University Press.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

R Core Team. 2025. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.

Heng Rathpisey and Teguh Bharata Adji. 2019. Handling imbalance issue in hate speech classification using sampling-based methods. In 2019 5th International Conference on Science in Information Technology (ICSITech), pages 193–198.

Christian Rauh and Jan Schwalbach. 2020. Corp\_folketing\_v2.rds. In The ParlSpeech V2 data set: Full-text corpora of 6.3 million parliamentary speeches in the key legislative chambers of nine representative democracies. Harvard Dataverse. Version Number: V1.

Ludovic Rheault, Kaspar Beelen, Christopher Cochrane, and Graeme Hirst. 2016. Measuring emotion in parliamentary debates with automated textual analysis. PloS one, 11(12):e0168843.

Anna Ristilä, Otto Tarkka, Veronika Laippala, and Kimmo Elo. 2026. Hopes and Fears – Emotion Distribution in the Topic Landscape of Finnish Parliamentary Speech 2000-2020. Preprint, arXiv:2601.20424.

Jan Rovny, Jonathan Polk, Ryan Bakker, Liesbet Hooghe, Seth Jolly, Gary Marks, Marco Steenbergen, and Milada Anna Vachudova. 2025. The 2024 chapel hill expert survey on political party positioning in europe: Twenty-five years of party positional data. Electoral Studies, 97:102981.

Gijs Schumacher, Daniel Hansen, Mariken A.C.G. van der Velden, and Sander Kunst. 2019. A new dataset of Dutch and Danish party congress speeches. Research & Politics, 6(2):2053168019838352.

Weber Shandwick and Powell Tate. 2019. Civility in america 2019: Solutions for tomorrow. Weber Shandwick We Solve.

David Shoemaker. 2015. Responsibilityfrom the Margins. Oxford University Press (UK).

David Shoemaker and Manuel Vargas. 2021a. Moral torch fishing: A signaling theory of blame. Noûs, 55(3):581–602.

David Shoemaker and Manuel Vargas. 2021b. Moral torch fishing: A signaling theory of blame. Noûs, 55(3):581–602.

Angela Smith. 2012. Moral blame and moral protest.

Hans-Henrik Busk Stie. 2021. Danskere siger fra og blokerer eller unfriender ’venner’ på Facebook - TV 2. nyheder.tv2.dk. Section: Stop hadbeskeder.

Wolf Susan, Jay Wallace, Kumar Rahul, and Freeman Samuel. 2011. Blame, Italian Style. Reasons and Recognition: Essays on the Philosophy ofTM Scanlon, pages 332–47.

Yannis Theocharis, Pablo Barberá, Zoltán Fazekas, and Sebastian Adrian Popa. 2020. The dynamics of political incivility on twitter. Sage Open, 10(2):2158244020919447.

Jörg Tiedemann, Mikko Aulamo, Daria Bakshandaeva, Michele Boggia, Stig-Arne Grönroos, Tommi Nieminen, Alessandro Raganato, Yves Scherrer, Raul Vazquez, and Sami Virpioja. 2023. Democratizing Neural Machine Translation with OPUS-MT. arXiv preprint. ArXiv:2212.01936 [cs].

Neal Tognazzini and D. Justin Coates. 2024. Blame. In Edward N. Zalta and Uri Nodelman, editors, The Stanford Encyclopedia ofPhilosophy, fall 2025 edition. Metaphysics Research Lab, Stanford University.

Jonathan Van’t Riet and Aart Van Stekelenburg. 2022. The effects of political incivility on political trust and political participation: A meta-analysis of experimental research. Human Communication Research, 48(2):203–229.

Jay Wallace. 2011. Dispassionate opprobrium: On blame and the reactive sentiments.

Shawn Tinghao Wang. 2024. Rethinking Functionalist Accounts of Blame. The Journal of Ethics, 28(4):607–623.

Williams. Praise and Blame | Internet Encyclopedia of Philosophy.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. HuggingFace’s Transformers: State-of-theart Natural Language Processing. arXiv preprint. ArXiv:1910.03771 [cs].

Lori Young and Stuart Soroka. 2012. Affective news: The automated coding of sentiment in political texts. Political communication, 29(2):205–231.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176.

Jingming Zhuo, Songyang Zhang, Xinyu Fang, Haodong Duan, Dahua Lin, and Kai Chen. 2024. ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1950–1976, Miami, Florida, USA. Association for Computational Linguistics.

## A Appendix

## A.0.1 Data Processing Flowchart

A more in-depth flow chart of the procedure described in the Methods section (§ 2) is seen in Figure 5.

## A.1 Assessment of Dataset Compatibility

To evaluate whether merging the pre-2019 and post-2019 parliamentary speech datasets introduces systematic bias, we examine continuity in sentence volume and test for distributional differences at the merge point (January 2019). The objective is to assess whether the two datasets can be treated as originating from a continuous data-generating process.

## A.1.1 Temporal Continuity, Distributional Comparisons and Discontinuity

Monthly sentence counts show no visible discontinuity at the 2019 merge point. Although some variation is present over time, these changes do not align with the boundary of the datasets. The same conclusion holds when adjusting for the number of active parliamentary days per month, suggesting that fluctuations reflect natural variation in activity rather than a structural break (Fig. 6 and 7).

![](images/5a4aac13a68221be9920796a629f4bf18a7d69ce2e591fe0182c8afde9017868.jpg)  
Figure 6: Monthly counts of sentences toughout the entire period. Time of dataset merge is shown by red dashed line.

![](images/1f60bb6a932f62e823fe08fc981d6687f88ad7b42170fcccc69569cc19767666.jpg)  
Figure 7: Mean monthly counts of sentences per active day in parliament. Time of dataset merge is shown by the dashed red line.

Comparing the distribution of monthly sentence counts before and after 2019 reveals substantial overlap, with no systematic differences apparent in either density or boxplot representations (Fig. 8 and 9).

![](images/b945202fef038cc8a4925c5898a01acd3bff7c6b532944c162411542cc863190.jpg)  
Figure 8: Density plot of number of monthly sentences in the dataset split by merge date

![](images/06021b2f2ba7bb49296025ff8f399d454633cba4faed8b89d12b16561465cbaf.jpg)  
Figure 5: Detailed flow chart of data processing steps.

![](images/711af8c37d225861b26aad6cd314a35e08cfbf7d15bf1bf80d45a68d563dc71c.jpg)  
Figure 9: Boxplot of number of monthly sentences in the dataset split by merge date.

To account for temporal trends, we further examine the residuals from a linear time model. The resulting distributions remain highly similar across periods. A Kolmogorov–Smirnov test confirms no detectable difference between pre- and post-2019 residuals, $D = 0 . 0 8 4 7 , p = 0 . 8 1 0$

We estimate a regression model including a linear time trend and an indicator for the post-2019 period. The post-2019 coefficient is small and statistically insignificant $b = - 5 8 . 2 , S E = 2 1 7 5 , p >$ 0.05, indicating no evidence of a discontinuity at the merge point.

## A.1.2 Overall Compatibility

Across all analyses, we find no indication of an abrupt shift associated with the merge. The results support treating the combined dataset as a continuous time series without evidence of structural artifacts introduced by merging.

## A.2 Blame Examples

Table 3 showcases a few randomly sampled sentences that were classified by BlameBERT as either containing blame or no blame. These random examples show the general tendency of lower confidence on sentences classified as containing blame.

<table><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Translation</td><td rowspan=1 colspan=1>Label</td><td rowspan=1 colspan=1>Confidence</td></tr><tr><td rowspan=1 colspan=1>Man indrømmede så her i salen, atdet nok var for dårligt, at man havdeskåret ned på det, men så løb man indi en anden udemokratisk ting, nemligat når man bag lukkede døre først harvedtaget sådan en aftale, så kan denkun laves om, hvis samtlige partier,der er med i aftalen, ønsker at lave denom.</td><td rowspan=1 colspan=1>It was then admitted in this Chamberthat it was probably too bad to have cutit down, but then you ran into anotherundemocratic thing, namely that whenyou have first adopted such an agree-ment behind closed doors, it can onlybe changed if all the parties involvedin the agreement want to change it.</td><td rowspan=1 colspan=1>Blame</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1>Jeg må indrømme, at jeg finder, aten kønsopdelt fagbevægelse er medtil at fastholde og fortsætte et køn-sopdelt arbejdsmarked og dermed eren selvstændig årsag til kønsbetingedelønforskelle.</td><td rowspan=1 colspan=1>I must admit that I believe that agender-separated trade union move-ment is helping to maintain and con-tinue a gender-based labour marketand is thus an independent cause ofgender pay differences.</td><td rowspan=1 colspan=1>Blame</td><td rowspan=1 colspan=1>0.58</td></tr><tr><td rowspan=1 colspan=1>Når FN&#x27;s Sikkerhedsråd har besluttet,at Saddam Hussein skal afvæbnes, atSaddam Hussein skal destruere sinekemiske og biologiske våben, at hanikke må udvikle atomvåben, at hanikke må samarbejde med terrorister, athan skal skrotte sine langtrækkendemissiler, og han alligevel ikke gør det,må der være nogle, der siger: På eteller andet tidspunkt skal der altsåmagt bag ordene, hvis der skal stå re-spekt om FN&#x27;s Sikkerhedsråd.</td><td rowspan=1 colspan=1>When the United Nations SecurityCouncil has decided that Saddam Hus-sein must be disarmed, that SaddamHussein must destroy his chemical andbiological weapons, that he must notdevelop nuclear weapons, that he mustnot cooperate with terrorists, that hemust scrap his long-range missiles,and yet he does not, there must besome who say: at some point, then,if there is to be respect for the UN Se-curity Council, there must be powerbehind the words.</td><td rowspan=1 colspan=1>Blame</td><td rowspan=1 colspan=1>0.59</td></tr><tr><td rowspan=1 colspan=1>I dag har man 20.000 arbejdspladserplaceret rundtomkring i verden.</td><td rowspan=1 colspan=1>Today there are 20,000 jobs locatedaround the world.</td><td rowspan=1 colspan=1>Noblame</td><td rowspan=1 colspan=1>0.89</td></tr><tr><td rowspan=1 colspan=1>Vi er glade for, at et enigt Lagtingbakker op om den her ændring, og Lib-eral Alliance støtter det selvfølgeligogså her fra Folketinget med vores jatil lovforslaget.</td><td rowspan=1 colspan=1>We are pleased that a united Layingsupports this change, and of course theLiberal Alliance also supports it herefrom the Folketing with our &#x27;yes’tothe bill.</td><td rowspan=1 colspan=1>Noblame</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>Og det er også fuldstændig rigtigt, atder i sandhed er en langsigtet strategimed regeringens boligpolitik</td><td rowspan=1 colspan=1>And it is also absolutely true that thereis indeed a long-term strategy with thegovernment&#x27;s housing policy.</td><td rowspan=1 colspan=1>Noblame</td><td rowspan=1 colspan=1>0.96</td></tr></table>

Table 3: Randomly sampled examples of sentences classified by BlameBERT.

## A.3 Experiment Tracking

All created models in this paper were tracked with Weights & Biases. For selected performance metrics for the best performing model within each of the five DIALs can be found by following this link: https://api.wandb.ai/ links/markuslundsfryd-aarhus-university/ uj5xetsx. For training and validation metrics for all models, see https://api.wandb.ai/ links/markuslundsfryd-aarhus-university/ t24vt5k9

## A.4 Performance Metrics

Performance metrics of the five different temporary classification models, each based on DIAL’s with increasing levels of labeling conservatism, are noted in Table 4. Due to shared highest macroaveraged F1 and more balanced Precision/Recall distribution, the final model was based on DIAL-5.

<table><tr><td></td><td>Precision</td><td>Recall</td><td>F1</td><td>Macro F1</td></tr><tr><td>DIAL-1</td><td>0.62</td><td>0.88</td><td>0.73</td><td>0.77</td></tr><tr><td>DIAL-2</td><td>0.67</td><td>0.85</td><td>0.75</td><td>0.79</td></tr><tr><td>DIAL-3</td><td>0.68</td><td>0.69</td><td>0.68</td><td>0.76</td></tr><tr><td>DIAL-4</td><td>0.70</td><td>0.80</td><td>0.75</td><td>0.80</td></tr><tr><td>DIAL-5</td><td>0.72</td><td>0.79</td><td>0.75</td><td>0.80</td></tr></table>

Table 4: Test set performance across datasets. All metrics for blame class.

## A.5 Comparing Against Other Models

Qwen 3-Embedding: The method chosen for the embedding strategy was to use representative “anchor” sentences (Zhuo et al., 2024). These anchors consisted of eight representative sentences for both blame and non-blame, which were randomly sampled from the full dataset where the confidence score associated with the label was ≥ 0.90.

Qwen 3.5 Generative: As the Qwen 3.5 model is generative, it was constrained with the Ollama package to output into a structured json schema. This was combined with a temperature of 0 for more deterministic output. Even with this framework, there is no guarantee of a valid response, as the model sometimes breaks down. For this reason, the model was allowed one extra try for each sentence. Otherwise, it was skipped. This null output can be interpreted in three ways: The first option is to treat it as a “true” null by not counting it in model performance and acknowledging failure. The second option is to treat a null as a false, or no blame, label (default no blame). The third option is to treat it as a true, or blame, label (default blame).

In practice, a system prompt (“Du er en ekspert i at identificere hvornår politikere anklager hinandenfor at være skyld i et negativt udfald. Identificér om der er nogle der anklager hinanden i sætningen. /no\_think”) ["You are an expert in identifying when politicians blame each other for causing a negative outcome. Identify whether someone is blaming others in the sentence."] was passed to the model together with the sentence to be classified. Reasoning was disabled with the /no\_think flag for computational reasons. Macro averaged performance metrics can be seen in Table 5.

<table><tr><td></td><td>Precision</td><td>Recall</td><td>Macro F1</td></tr><tr><td>Embedding</td><td>0.70</td><td>0.72</td><td>0.67</td></tr><tr><td>Generative</td><td>0.91</td><td>0.71</td><td>0.75</td></tr><tr><td>BlameBERT</td><td>0.80</td><td>0.81</td><td>0.80</td></tr></table>

Table 5: Qwen 3-Embeddings, Generative Qwen 3.5, and BlameBERT compared on macro averaged metrics.

Choosing a null-handling strategy plays a significant role, as the model produced 68/424 (16%) null labels. The consequences of treating null labels differently can be seen in Table 6.

<table><tr><td></td><td>Precision</td><td>Recall</td><td>Macro F1</td></tr><tr><td>Abstain</td><td>0.91</td><td>0.71</td><td>0.75</td></tr><tr><td>Default no blame</td><td>0.86</td><td>0.64</td><td>0.64</td></tr><tr><td>Default blame</td><td>0.82</td><td>0.77</td><td>0.79</td></tr></table>

Table 6: Generative Qwen 3.5 macro-averaged zero-shot sensitivity to null-handling strategy.

These results indicate that metrics are most balanced if null labels are treated as blame labels (default blame), but this assumption is hard to justify. Looking at the classification report for the abstain strategy (Table 7), it can be seen that this produces a very conservative model, trading precision (1.00) for recall (0.42).

For these reasons, and to introduce the least possible author-bias, the abstain strategy was chosen for the comparative baseline. This means that null labels were removed when metrics were computed.

For and Against: If only considering the macroaveraged metrics as seen in Table 5, the generative model is tempting, with a reasonable F1 score of .75 and a precision of .91. However, as the classification report reveals, the generative approach severely underpredicts blame.

<table><tr><td colspan="2">Condition</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Support</td></tr><tr><td rowspan="5">Default no blame</td><td>No blame</td><td>0.72</td><td>1.00</td><td>0.84</td><td>278</td></tr><tr><td>Blame</td><td>1.00</td><td>0.28</td><td>0.44</td><td>148</td></tr><tr><td>Accuracy</td><td></td><td></td><td>0.75</td><td>426</td></tr><tr><td>Macro avg</td><td>0.86</td><td>0.64</td><td>0.64</td><td>426</td></tr><tr><td>Weighted avg</td><td>0.82</td><td>0.75</td><td>0.70</td><td>426</td></tr><tr><td rowspan="5">Default blame</td><td>No blame</td><td>0.82</td><td>0.93</td><td>0.87</td><td>278</td></tr><tr><td>Blame</td><td>0.83</td><td>0.61</td><td>0.71</td><td>148</td></tr><tr><td>Accuracy</td><td></td><td></td><td>0.82</td><td>426</td></tr><tr><td>Macro avg</td><td>0.82</td><td>0.77</td><td>0.79</td><td>426</td></tr><tr><td>Weighted avg</td><td>0.82</td><td>0.82</td><td>0.81</td><td>426</td></tr><tr><td rowspan="5">Abstain</td><td>No blame</td><td>0.82</td><td>1.00</td><td>0.90</td><td>259</td></tr><tr><td>Blame</td><td>1.00</td><td>0.42</td><td>0.60</td><td>99</td></tr><tr><td>Accuracy</td><td></td><td></td><td>0.84</td><td>358</td></tr><tr><td>Macro avg</td><td>0.91</td><td>0.71</td><td>0.75</td><td>358</td></tr><tr><td>Weighted avg</td><td>0.87</td><td>0.84</td><td>0.82</td><td>358</td></tr></table>

Table 7: Qwen 3.5:9B zero-shot classification performance under different null-handling strategies.

Another thing to keep in mind when considering using a generative LLM is the computational cost. Inference with Qwen 3.5 on the small dataset in this paper took ∼ 19 hours on an RTX 4070 Super. Inference using BlameBERT, once fine-tuned for ∼ 20 minutes, took a few minutes on an Nvidia L40 GPU. While these are far apart in computational power (L40 being significantly faster), and therefore not directly comparable, BlameBERT is only 307 million parameters, and the version of Qwen 3.5 used here is 9 billion.

Sentence embeddings provide a simple, cheap, and easy to implement alternative to the other models. The anchor strategy performs reasonably well and is quite balanced between precision and recall.

Collating these findings, the metrics point toward BlameBERT as the best option for blame classification in a political context.

## A.6 Per Party Performance

To investigate whether BlameBERT has an easier or harder time identifying blame between the different parties, we have plotted confusion matrices for parties represented with more than 10 sentences in the test-set (Fig. 10). These indicate no systematic difference in blame classification between parties.

![](images/7417b377b63afa33b201640f6a972fce8d14feccbbd9ab809ab30787ba27dd4e.jpg)  
Figure 10: Confusion matrix per party in gold-labeled test-set. (Only parties with more than 10 sentences included.)

## A.7 Wing and Wingness variables

We adopt an index of a party’s overall left–right ideological stance, from the "lrgen" variable of the Chapel Hill Expert Survey (CHES) (Rovny et al., 2025).<sup>3</sup> This variable was standardized (z-scored) across all parties included in this study. Parties with negative values are classified as belonging to the left wing, while parties with positive values are classified as belonging to the right wing. The zero threshold corresponds to the sample mean of the standardized index. The absolute standardized value was represented by the wingness variable.

## A.8 Analysis Summaries

Full model summaries for analysis H1-2.1. The following summaries are the four best-fitting models according to the LRTs:

<table><tr><td colspan="2">M1.2, Time and Blame (1997–2026)</td></tr><tr><td>(Intercept)</td><td> $\phantom { + } 2 . 3 8 ^ { * * * }$   $( 6 . 4 5 \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>scale(month_index)</td><td> $- 9 . 8 0 \times 1 0 ^ { - 3 }$   $( 6 . 0 0 \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>scale(month_index)²</td><td> $1 . 3 5 \times 1 0 ^ { - 2 * }$   $( 6 . 4 0 \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>in_gov1</td><td> $- 4 . 4 4 \times 1 0 ^ { - 1 * * }$   $( 1 . 6 2 \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>AIC Log Likelihood</td><td> $\overline { { 2 . 4 9 \times 1 0 ^ { 4 } } }$   $- 1 . 2 4 \times 1 0 ^ { 4 }$ </td></tr><tr><td>Num. obs.</td><td>2529</td></tr><tr><td>Num. groups: party</td><td>13</td></tr><tr><td>Var: party (Intercept)</td><td> $5 . 2 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td></td></tr><tr><td>*** p&lt;0.001; **p&lt;0.01; *p&lt;0.05</td><td></td></tr></table>

Table 8: Full summary with effect and SE (b (SE)) of all included paramteres of model M1.2 of Analysis 2.3. Here time (scaled month index) is treated as the focal predictor, while government status is included as a controlling fixed-effect variable.

<table><tr><td colspan="2">M2.5: Political Predictors (1997–2026)</td></tr><tr><td>(Intercept)</td><td> $- 2 . 5 2 ^ { * * * }$   $( 1 . 0 3 \times 1 0 ^ { - 1 } )$ </td></tr><tr><td>in_gov1</td><td> $- 4 . 7 8 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>blocright_bloc</td><td> $( 2 . 0 5 \times 1 0 ^ { - 2 } )$   $- 3 . 1 6 \times 1 0 ^ { - 1 }$   $( 1 . 6 9 \times 1 0 ^ { - 1 } )$ </td></tr><tr><td>wingness</td><td> $8 . 5 0 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>scale(month_index)</td><td> $( 9 . 6 7 \times 1 0 ^ { - 2 } )$   $- 7 . 6 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>scale(month_index)²</td><td> $( 6 . 0 0 \times 1 0 ^ { - 3 } )$   $1 . 7 5 \times 1 0 ^ { - 2 * * }$ </td></tr><tr><td>in_gov1:blocright_bloc</td><td> $( 6 . 4 9 \times 1 0 ^ { - 3 } )$   $1 . 0 5 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>blocright_bloc:wingness</td><td> $( 3 . 4 6 \times 1 0 ^ { - 2 } )$   $5 . 5 4 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>AIC</td><td> $( 1 . 8 4 \times 1 0 ^ { - 1 } )$   $\overline { { 2 . 4 9 \times 1 0 ^ { 4 } } }$ </td></tr><tr><td>Log Likelihood</td><td> $- 1 . 2 4 \times 1 0 ^ { 4 }$ </td></tr><tr><td>Num. obs.</td><td>2529</td></tr><tr><td>Num. groups: party</td><td>13</td></tr><tr><td>Var: party (Intercept)</td><td> $2 . 1 5 \times 1 0 ^ { - 2 }$ </td></tr></table>

<sup>∗∗∗</sup>p<0.001; <sup>∗∗</sup>p<0.01; <sup>∗</sup>p<0.05

Table 9: Full summary with effect and SE (b (SE)) of all included parameters of model M2.5 of Analysis 2.3. Here, government status (in\_gov), wing affiliation (bloc), political extremity (wingness), and the interaction between these are treated as focal predictors. The quadratic relationship between blame and time for this period (see Table 8) was included as a controlling fixedeffect variable.
<table><tr><td colspan="2">M3.1: Time and Blame (2019–2026)</td></tr><tr><td>(Intercept)</td><td> $- 2 . 2 9 ^ { * * * }$   $( 6 . 8 6 \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>in_gov1 scale(month_index)</td><td> $- 5 . 3 4 \times 1 0 ^ { - 1 * * }$   $( 4 . 5 9 \times 1 0 ^ { - 2 } )$   $9 . 2 1 \times 1 0 ^ { - 2 * * * }$ </td></tr><tr><td>AIC</td><td> $( 8 . 6 1 \times 1 0 ^ { - 3 } )$   $\overline { { 7 . 3 3 \times 1 0 ^ { 3 } } }$ </td></tr><tr><td>Log Likelihood Num. obs.</td><td> $- 3 . 6 6 \times 1 0 ^ { 3 }$  810</td></tr><tr><td>Num. groups: party</td><td>13</td></tr><tr><td></td><td> $5 . 8 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Var: party (Intercept)</td><td></td></tr></table>

<sup>∗∗∗</sup>p<0.001; <sup>∗∗</sup>p<0.01; <sup>∗</sup>p<0.05

Table 10: Full summary with effect and SE (b (SE)) of all included paramteres of model M3.1 of Analysis 2.3. Here time (scaled month index) is treated as the focal predictor, while government status is included as a controlling fixed-effect variable.

<table><tr><td colspan="2">M4.4, Political predictors (2019–2026)</td></tr><tr><td>(Intercept)</td><td> $- 2 . 4 1 ^ { * * * }$   $( 1 . 0 1 \times 1 0 ^ { - 1 } )$ </td></tr><tr><td>in_gov1</td><td> $5 . 1 5 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>blocright_bloc</td><td> $( 4 . 4 2 \times 1 0 ^ { - 2 } )$   $5 . 2 2 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>wingness</td><td> $( 1 . 5 1 \times 1 0 ^ { - 1 } )$   $- 1 . 3 6 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>month_index</td><td> $( 9 . 0 8 \times 1 0 ^ { - 2 } )$   $3 . 7 7 \times 1 0 ^ { - 3 * * }$ </td></tr><tr><td>blocright_bloc:wingness</td><td> $( 3 . 5 4 \times 1 0 ^ { - 4 } )$   $8 . 1 7 \times 1 0 ^ { - 1 * * }$ </td></tr><tr><td>AIC</td><td> $( 1 . 6 2 \times 1 0 ^ { - 1 } )$   $\overline { { 7 . 3 3 \times 1 0 ^ { 3 } } }$ </td></tr><tr><td>Log Likelihood Num. obs.</td><td> $- 3 . 6 5 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Num. groups: party</td><td>810 13</td></tr><tr><td>Var: party (Intercept)</td><td> $1 . 5 7 \times 1 0 ^ { - 2 }$ </td></tr></table>

<sup>∗∗∗</sup>p<0.001; <sup>∗∗</sup>p<0.01; <sup>∗</sup>p<0.05

Table 11: Full summary with effect and SE (b (SE)) of all parameters included in model M4.4 of Analysis 2.3. Here government status (in\_gov), wing affiliation (bloc), political extremity (wingness) and the interaction between these are treated as focal predictors. The linear relationship between blame and time which was found for this period (see Table 10) was included as controlling fixed-effects variables.

## A.9 Sensitivity Analysis: Classification Threshold Robustness

The blame labels produced by BlameBERT carry classification uncertainty that is not propagated into the statistical models. To assess whether the substantive findings depend on the decision threshold used to classify a sentence as containing blame, we refit the best-fitting model from each analysis using three increasingly conservative probability thresholds: $\tau = 0 . 6 2 5 , \tau = 0 . 7 5$ , and $\tau = 0 . 8 7 5$ relative to the original $\tau = 0 . 5 0$ . These correspond to blame prevalence rates of approximately 4.8%, 2.1%, and 0.4% respectively, compared to 8.5% at the original threshold. Figure 11 shows the density distribution of probabilities in the inference data. Table 12 reports the estimated coefficients and standard errors for each focal predictor across thresholds.

<table><tr><td rowspan="2">Threshold Predictor</td><td colspan="2"> $\tau = 0 . 5 0$ </td><td colspan="2"> $\tau = 0 . 6 2 5$ </td><td colspan="2"> $\tau = 0 . 7 5$ </td><td colspan="2"> $\tau = 0 . 8 7 5$ </td></tr><tr><td>b</td><td>(SE)</td><td>b</td><td>(SE)</td><td>b</td><td>(SE)</td><td>b</td><td>(SE)</td></tr><tr><td colspan="9">Analysis H1: Temporal trend (1997–2026), M1.2</td></tr><tr><td>Month (linear)</td><td>-0.00980 (0.00600)</td><td></td><td> $- 0 . 0 0 9 9 5 \left( 0 . 0 0 7 2 0 \right)$ </td><td></td><td></td><td></td><td>-0.0121 (0.00888) -0.00113 (0.0134)</td><td></td></tr><tr><td> $\mathbf { M o n t h ^ { 2 } \left( q u a d r a t i c \right) }$ </td><td></td><td> $0 . 0 1 3 5 ^ { * } \left( 0 . 0 0 6 4 0 \right)$ </td><td> $0 . 0 1 8 7 ^ { * } \left( 0 . 0 0 7 6 7 \right)$ </td><td></td><td> $0 . 0 2 3 7 ^ { * } \left( 0 . 0 0 9 4 5 \right)$ </td><td></td><td></td><td> $0 . 0 4 0 4 ^ { * * } \left( 0 . 0 1 4 3 \right)$ </td></tr><tr><td colspan="9">Analysis H2: Political predictors (1997–2026), M2.5</td></tr><tr><td>Government Status</td><td> $- 0 . 4 7 8 ^ { * * * } \left( 0 . 0 2 0 5 \right)$ </td><td></td><td> $- 0 . 4 9 7 ^ { \ast \ast \ast } \left( 0 . 0 2 4 7 \right)$ </td><td></td><td> $- 0 . 5 1 8 ^ { * * * } \left( 0 . 0 3 0 8 \right)$ </td><td></td><td> $- 0 . 4 9 1 ^ { * * * } \left( 0 . 0 4 7 3 \right)$ </td><td></td></tr><tr><td>Wing: right</td><td>−0.316· (0.169)</td><td></td><td> $- 0 . 4 0 7 ^ { \ast } \left( 0 . 1 9 2 \right)$ </td><td></td><td> $- 0 . 4 7 8 ^ { * } \left( 0 . 2 1 8 \right)$ </td><td></td><td> $- 0 . 6 8 9 ^ { * } \left( 0 . 2 8 3 \right)$ </td><td></td></tr><tr><td>Wingness</td><td></td><td>0.0850 (0.0967)</td><td>0.0767 (0.109)</td><td></td><td>0.0774 (0.122)</td><td></td><td></td><td>0.0868 (0.153)</td></tr><tr><td>Gov × Wing: right</td><td></td><td> $0 . 1 0 5 ^ { * * } \left( 0 . 0 3 4 6 \right)$ </td><td> $0 . 1 1 8 ^ { * * } \left( 0 . 0 4 1 8 \right)$ </td><td></td><td> $0 . 1 4 4 ^ { * * } \left( 0 . 0 5 1 8 \right)$ </td><td></td><td></td><td>0.0756 (0.0792)</td></tr><tr><td>Wing: right × Wingness</td><td> $0 . 5 5 4 ^ { * * } \left( 0 . 1 8 4 \right)$ </td><td></td><td> $0 . 7 0 8 ^ { * * * } \left( 0 . 2 1 0 \right)$ </td><td></td><td> $0 . 8 7 6 ^ { * * * } \left( 0 . 2 3 7 \right)$ </td><td></td><td></td><td> $1 . 2 4 9 ^ { * * * } \left( 0 . 3 0 7 \right)$ </td></tr><tr><td colspan="9">Analysis H1.1: Temporal trend (2019–2026), M3.1</td></tr><tr><td>Time (linear)</td><td></td><td> $0 . 0 9 2 1 ^ { * * * } \left( 0 . 0 0 8 6 1 \right)$ </td><td> $0 . 1 1 9 ^ { * * * } \left( 0 . 0 1 0 5 \right)$ </td><td></td><td> $0 . 1 4 9 ^ { * * * } \left( 0 . 0 1 3 7 \right)$ </td><td></td><td></td><td> $0 . 1 6 8 ^ { * * * } \left( 0 . 0 2 2 7 \right)$ </td></tr><tr><td colspan="9">Analysis H2.2: Political predictors (2019–2026), M4.4</td></tr><tr><td>Government</td><td> $- 0 . 5 1 5 ^ { * * * } \left( 0 . 0 4 4 2 \right)$ </td><td></td><td> $- 0 . 5 6 7 ^ { \ast \ast \ast } \left( 0 . 0 5 4 8 \right)$ </td><td></td><td> $- 0 . 6 1 4 ^ { * * * } \left( 0 . 0 7 1 3 \right)$ </td><td></td><td> $- 0 . 7 3 0 ^ { * * * } \left( 0 . 1 1 6 \right)$ </td><td></td></tr><tr><td>Wing: right</td><td></td><td> $- 0 . 5 2 2 ^ { * * * } \left( 0 . 1 5 1 \right)$ </td><td> $- 0 . 6 4 5 ^ { * * * } \left( 0 . 1 7 7 \right)$ </td><td></td><td> $- 0 . 7 4 2 ^ { * * * } \left( 0 . 2 0 6 \right)$ </td><td></td><td> $- 0 . 9 8 5 ^ { * * * } \left( 0 . 2 7 9 \right)$ </td><td></td></tr><tr><td>Wingness</td><td></td><td> $- 0 . 1 3 6 \left( . 0 9 0 8 \right)$ </td><td> $- 0 . 1 7 6 \cdot ( 0 . 1 0 6 )$ </td><td></td><td> $- 0 . 2 0 8 \cdot ( 0 . 1 2 3 )$ </td><td></td><td></td><td>-0.223 (0.163)</td></tr><tr><td>Wing: right × Wingness</td><td></td><td> $0 . 8 1 7 ^ { * * * } \left( 0 . 1 6 2 \right)$ </td><td> $1 . 0 1 ^ { * * * } \left( 0 . 1 9 0 \right)$ </td><td></td><td></td><td> $1 . 2 2 ^ { * * * } \left( 0 . 2 2 0 \right)$ </td><td></td><td> $1 . 6 1 ^ { * * * } \left( 0 . 2 9 4 \right)$ </td></tr></table>

Note. All models include a <sub>p</sub>art<sub>y</sub>-level random interce<sub>p</sub>t and a sentence-count offset. The tem<sub>p</sub>oral trend (M 1 . 2 or M3 . 1 ) is included as a controllin<sub>g</sub> fi<sub>xe</sub>d <sub>e</sub>ff<sub>ec</sub>t i<sub>n</sub> A<sub>na</sub>l<sub>yses</sub> 2 <sub>an</sub>d 4 <sub>respec</sub>ti<sub>ve</sub>l<sub>y</sub>. Th<sub>res</sub>h<sub>o</sub>ld <sub>τ</sub> d<sub>eno</sub>t<sub>es</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>mum pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f bl<sub>ame requ</sub>i<sub>re</sub>d t<sub>o</sub> l<sub>a</sub>b<sub>e</sub>l <sub>a sen</sub>t<sub>ence as con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> blame. Blame <sub>p</sub>revalence at each threshold: 8 .5 % (τ = 0 . 50) 4. 8 % (τ = 0 . 62 5) 2. 1 % (τ = 0 . 75) 0.4% (τ = 0 . 875) . The Gov × Win<sub>g</sub> interaction i<sub>n</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub> 2 l<sub>oses s</sub>i<sub>gn</sub>ifi<sub>cance a</sub>t <sub>τ</sub> = 0 . 8 75 th<sub>e on</sub>l<sub>y excep</sub>ti<sub>on</sub> t<sub>o an o</sub>th<sub>erw</sub>i<sub>se cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>pa</sub>tt<sub>ern</sub>.  
T<sub>a</sub>bl<sub>e</sub> 1 2 <sub>:</sub> S<sub>ens</sub>iti<sub>v</sub>it<sub>y</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s :</sub> fi<sub>xe</sub>d-<sub>e</sub>ff<sub>ec</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> <sub>across</sub> <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s .</sub> St<sub>an</sub>d<sub>ar</sub>d <sub>errors</sub> i<sub>n</sub> <sub>paren</sub>th<sub>eses .</sub> Si<sub>gn</sub>ifi<sub>cance :</sub> $\cdot p < . 1 0 , ^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1$

![](images/621501acc0a0176fed457c19c69807ede50be37ed59bf918e379632e07d9c477.jpg)  
Figure 11: Density distribution of probabilities of BlameBERT in the inference data.

The results demonstrate substantial robustness across all four analyses. The direction of every reported effect is preserved at all thresholds, and the vast majority of originally significant effects remain significant. Notably, effect sizes generally speaking increase with the threshold. This pattern is consistent with the interpretation that the more permissive threshold admits noisier classifications that attenuate the underlying political signal: when restricted to sentences classified with higher confidence, the structural patterns in blame attribution become more pronounced rather than less.

One exception warrants attention. The interaction between government status and ideological wing in Analysis H2 (M2.5), while significant at $\tau = 0 . 5 0 , \tau = 0 . 6 2 5$ , and $\tau = 0 . 7 5$ , loses significance at the strictest threshold $( b = 0 . 0 7 6$ $S E = 0 . 0 7 9 , p = . 3 4 0 )$ . This is the one finding that should be treated with some caution, as it does not entirely survive the most conservative classification criterion. All other focal effects, including the dominant government suppression effect and the wing × wingness interaction in both time windows, remain stable across all thresholds.

As the threshold increases, the number of blamepositive sentences decreases, and some partymonth cells transition to zero counts. This progressive zero inflation reduces statistical power, particularly for smaller parties with fewer monthly observations, and partly explains the widening standard errors at $\tau = 0 . 8 7 5$ . Where a coefficient loses significance at the strictest threshold, this could be interpreted as a power limitation rather than an unstable effect.

## A.10 Controlling for Political Topics

The temporal and political predictors of blame examined in the main analyses do not account for the topical content of parliamentary speech, even though some subjects plausibly invite sharper rhetoric than others. If governing and opposition parties also differ systematically in which topics they speak about, the government-status effect reported in Analysis H2 (M2.5) and Analysis H2.1 (M4.4) could partly reflect agenda composition rather than a genuine difference in rhetorical framing. We therefore conduct a supplementary analysis to assess whether the blame-suppressing effect of government status survives the inclusion of topic as a covariate.

Data Preparation: Because the inference data used throughout the paper can contain multiple sentences from the same speaker on the same day, applying a topic classifier to every sentence would yield a set of observations with strong withinspeaker-day dependence. To obtain a set of approximately independent observations for this supplementary check, one sentence was randomly sampled for each unique speaker-day combination, yielding a reduced dataset of 84,581 observations.

To assign topics, the classifier manifestoproject/manifestoberta-xlm-roberta-56policytopics-sentence-2024-1-1 (Burst et al., 2024) was applied to each sampled sentence. This model is a fine-tuned XLM-RoBERTa-large classifier trained on 38 languages, including Danish, that assigns text to one of 56 fine-grained political topics. Following the guidelines of the Manifesto-project handbook (Burst et al., 2024), the 56 topics were collapsed into seven broader domains: External Relations, Freedom and Democracy, Political System, Economy, Welfare and Quality of Life, Fabric of Society, and Social Groups.

Topic and Blame Rate: We first examined whether blame rate varies by topic domain, and whether governing and opposition parties differ in the topics they discuss. Both checks are descriptive and intended to contextualize the subsequent regression analysis rather than serve as statistical tests in themselves.

The proportion of blame-labeled sentences varied substantially across domains. The rate of blame in Fabric of Society was more than three times that observed in Economy, see Figure 12. This indicates that topic is plausibly related to emotional tone, consistent with the intuition that some subjects invite harsher language than others.

![](images/365736f01282b93390f713a96d54940190e11452eafd99f9a4fb1eb8ed1eea49.jpg)  
Figure 12: Proportion of sentences labeled as blame, by political topic domain.

In contrast, the distribution of topics discussed by governing parties closely resembled that of opposition parties, see Figure 13. While substantial differences existed in how much attention each topic domain received overall, this allocation did not differ meaningfully by government status. This is consistent with the structure of parliamentary debate, where the agenda for a given proceeding is generally fixed in advance and applies symmetrically to all speakers.

![](images/fb483ef015e8ce29e7c20b07a0b29f9b181df0db2ab2326107a4bd2af57cc03e.jpg)  
Figure 13: Topic domain distribution for parties in government and in opposition.

Taken together, these patterns suggest that although some topic domains are associated with elevated or reduced blame rate, governing and opposition parties are not differentially exposed to high- versus low-blame topics. This makes it unlikely that the government-status effect found in the main analyses is purely an artifact of agenda composition, a possibility we test directly below.

Model Specification: To formally evaluate whether topic domain confounds the relationship between government status and blame, two mixedeffects logistic regression models were fit at the sentence level on the downsampled dataset. Let $Y _ { i j m }$ denote whether sentence j spoken by party i in month m was labeled as blame, modeled as $Y _ { i j m } \sim$ Bernoulli $\left( p _ { i j m } \right)$ . The baseline model (M1) is specified as:

$$
\mathrm { l o g i t } ( p _ { i j m } ) = \beta _ { 0 } + \beta _ { 1 } \mathrm { G o v } _ { i j m } + u _ { i } + v _ { m }\tag{3}
$$

where $\mathbf { G o v } _ { i j m }$ is a binary indicator of government status, and $u _ { i } \sim \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } )$ and $v _ { m } \sim$ $\mathcal { N } ( 0 , \sigma _ { v } ^ { 2 } )$ are random intercepts for party and month, respectively, included to account for withinparty and within-month clustering. The extended model (M2) adds topic domain as a fixed effect:

$$
\begin{array} { r } { \log \mathrm { i t } ( p _ { i j m } ) = \beta _ { 0 } + \beta _ { 1 } \mathrm { G o v } _ { i j m } + \beta _ { 2 } \mathrm { T o p i c } _ { i j m } + u _ { i } + v _ { m } } \end{array}\tag{4}
$$

with Welfare and Quality of Life as the reference category, as it was the most frequently occurring domain for both governing and opposition parties. The two models were compared using a likelihood ratio test, and the coefficient for government status was inspected for stability between M1 and M2 as the focal diagnostic of confounding.

Results: The LRT indicated that adding topic domain significantly improved model fit, $\chi ^ { 2 } ( 6 ) =$ 1082, $p ~ < ~ . 0 0 1$ , confirming that topic explains meaningful variance in blame beyond government status alone.

For M1, the effect of government status was negative and significant, $b = - 0 . 3 9 2 , S E = 0 . 0 3 8 8 ,$ $z = - 1 0 . 1 , p < . 0 0 1$ . The intercept was also significant, $b = - 2 . 4 1 , S E = 0 . 0 6 4 1 , z = - 3 7 . 7 .$ $p < . 0 0 1$

For M2, the effect of government status remained negative and significant, $~ b ~ = ~ - 0 . 4 1 0 .$ $S E ~ = ~ 0 . 0 3 9 0 , ~ z ~ = ~ - 1 0 . 5 , ~ p ~ < ~ . 0 0 1$ All topic domains differed significantly from the reference category. Fabric of Society showed the strongest positive association with blame, $b = 1 . 1 0 .$ $S E = 0 . 0 3 9 1 , \ z \ = \ 2 8 . 1 , \ p \ < \ . 0 0 1$ , followed by External Relations, $b = 0 . 6 5 4 , S E = 0 . 0 5 3 0 .$ $z = 1 2 . 3 , p < . 0 0 1$ , and Political System, $b =$ $0 . 5 3 1 , S E = 0 . 0 3 5 1 , z = 1 5 . 1 , p < . 0 0 1$ . Freedom and Democracy, $b = 0 . 4 7 6 , S E = 0 . 0 5 9 5 .$ $z = 8 . 0 1 , p < . 0 0 1$ , and Social Groups, $b = 0 . 3 0 7$ $S E = 0 . 0 6 2 3 , z = 4 . 9 2 , p < . 0 0 1$ , were also associated with significantly elevated blame relative to the reference. Economy was the only domain associated with lower odds of blame than Welfare and Quality of Life, $b = - 0 . 4 1 7 , S E = 0 . 0 6 4 0 ,$ $z = - 6 . 5 2 , p < . 0 0 1$

Critically, the coefficient for government status was nearly unchanged after including topic domain. M1 estimates that a sentence spoken by a governing party has a 67.5% relative probability of containing blame compared to an opposition party. The corresponding estimate from M2 is 66.3%. This near-identical estimate indicates that controlling for topical agenda does not meaningfully alter the suppressive effect of government status on blame attribution.

Interpretation: The descriptive and modelbased results converge on the same conclusion. While topic domain is itself associated with blame rate, and some domains are considerably more conducive to blame than others, governing and opposition parties are not differentially exposed to these domains, and accounting for topic does not erode the government-status effect. This supports the interpretation that the political-contrasting effect identified in Analyses H2 and H2.1 reflects a genuine difference in rhetorical framing between governing and opposition parties, rather than an artifact of which subjects each side happens to discuss more.