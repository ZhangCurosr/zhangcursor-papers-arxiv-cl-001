# From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X

Zhiyang Qi, Kazuhiro Ito, Jinghui Chen, Hibiki Nakamura, Zhangxuan Chen,

Erina Murata, Masaki Chujyo, and Fujio Toriumi

Department of Systems Innovation, School of Engineering, The University of Tokyo

7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8656, Japan

{zhiyangqi, kazuhiro-it, jinghuichen, harmonie-nkmr-00, chengmg, erinamurata, mchujyo}@g.ecc.u-tokyo.ac.jp tori@sys.t.u-tokyo.ac.jp

Abstract—Ragebait refers to online content intentionally designed to provoke anger or outrage and thereby increase attention and engagement. However, reliable large-scale detection and systematic analysis of ragebait remain limited, hindering efforts to understand its prevalence, impact, and mitigation. This study aims to develop an effective ragebait detection framework and to clarify the characteristics of ragebait at scale, providing a basis for understanding and mitigating emotionally provocative content online. We constructed a labeled dataset with the assistance of a large language model (LLM) and trained several Japanese language models for ragebait detection. The resulting ensemble classifier was then applied to a large-scale dataset of Japaneselanguage posts on X. Our analysis shows that ragebait is more prevalent in politically and socially contentious topics, including politics, discrimination, public health, and interpersonal conflict. Ragebait posts also spread faster and receive more negative reactions than non-ragebait posts, particularly anger, fear, disgust, sadness, and surprise. These findings demonstrate the utility of the proposed detector and provide a large-scale characterization of ragebait in Japanese online discourse.

Index Terms—ragebait, social media, large language models, text classification, emotion analysis

## I. INTRODUCTION

Ragebait refers to online content deliberately designed to provoke anger or outrage in order to attract attention and increase engagement. The growing prominence of this practice was reflected in the selection of rage bait as the Oxford Word of the Year 2025, following a threefold increase in the use of the term over the preceding year [1]. Unlike conventional clickbait, which often exploits curiosity gaps, ragebait captures attention by eliciting frustration, offense, or interpersonal conflict. Its increasing visibility highlights a broader shift in online attention strategies from stimulating curiosity to actively manipulating users’ emotions [2].

A substantial body of research has demonstrated that anger and moral outrage play an important role in online engagement and information diffusion. Moral-emotional language has been associated with increased diffusion of political messages in social networks [3], while social feedback, such as likes and reposts, can reinforce users’ subsequent expressions of outrage [4]. Anger has also been found to spread efficiently across weak social ties [5] and to generate larger and deeper information cascades than anxiety [6]. These findings suggest that anger is not merely an individual response to online content: it can be amplified through social and platform-level mechanisms, potentially increasing polarization and degrading the quality of online discourse.

Nevertheless, research directly examining ragebait remains limited. Prior language-oriented work has discussed ragebait mainly in relation to intentionality and social-media discourse rather than as a scalable detection problem [7]. A recent study distinguished ragebait from information-oriented clickbait in news headlines and found that ragebait was associated with higher audience engagement [8]. However, its focus was on professionally produced news headlines, leaving ragebait in general user-generated social media posts largely unexplored.

Detecting ragebait is also challenging because it cannot be reduced to negative sentiment, offensive language, or anger expression alone. A post may contain negative language without attempting to provoke its audience, while ragebait may use irony, exaggeration, selective framing, or deliberately unreasonable claims to trigger a reaction. Identifying ragebait therefore requires consideration of the communicative intent of the poster and the likely emotional response of readers. The lack of a scalable detection method makes it difficult to estimate how prevalent ragebait is, identify the users and topics associated with it, and examine how it spreads and influences audience reactions.

To address these limitations, we develop a ragebait detector for Japanese-language posts on X and conduct a large-scale analysis. Specifically, following the definition of ragebait, we first use a large language model (LLM) to determine whether individual posts constitute ragebait, thereby constructing a pseudo-labeled dataset for training text classification models. The resulting dataset contains 18,558 instances, with an equal number of ragebait and non-ragebait posts. We further ask two annotators to independently label a randomly sampled subset and confirm that the generated pseudo-labels exhibit a reasonable degree of reliability. Using this dataset, we then train text classifiers for ragebait detection. To improve robustness beyond any single model, we train multiple classifiers and combine the three best-performing models through majority voting. The resulting ensemble achieves an accuracy of 84.05% and a Macro-F1 score of 84.04%.

We apply the ensemble classifier to posts extracted from a

1% sample of X data collected between October 2022 and June 2023, comprising more than 150 million Japanese-language event records. In addition to estimating the prevalence of ragebait, we characterize the detected posts along multiple dimensions, including lexical characteristics, thematic content, diffusion dynamics, and the emotional reactions they receive.

Our analysis reveals several distinctive patterns. Ragebait posts are more prevalent in politically and socially contentious topics, including party politics, gender and discrimination, vaccines and infectious diseases, and moral conflict. They also tend to spread faster than non-ragebait posts. Moreover, replies and quote posts directed at ragebait contain substantially more negative emotions, particularly anger, fear, disgust, sadness, and surprise. Taken together, these findings provide converging evidence that the classifier captures a meaningful category of intentionally provocative content rather than merely negative language in general. Resources related to this study are publicly available<sup>1</sup> for further research on ragebait detection and analysis.

## II. RELATED WORK

## A. Ragebait and Related Content

Ragebait is related to clickbait, offensive language, hate speech, and toxic content. Prior work has developed datasets and detection methods for clickbait in news and social media [2], [9]–[11], as well as for offensive and abusive language [12]–[14]. However, ragebait is defined not only by its linguistic form but also by the intention to provoke and the anticipated reaction of readers. Recent studies have begun to examine ragebait as a distinct phenomenon [7], [8], but largescale detection and analysis of user-generated posts remain limited.

## B. Anger and Online Engagement

Anger and moral outrage are strongly associated with online engagement and diffusion. Digital platforms can amplify moral outrage [15], while moral-emotional language and social feedback increase its expression and diffusion [3], [4]. Anger also exhibits stronger social correlation and wider diffusion than other emotions [5], [6], [16]. Political out-group animosity and outrage-inducing misinformation are likewise associated with increased sharing [17], [18]. These studies mainly examine emotions expressed in posts or their consequences, whereas our work focuses on detecting content intended to elicit such reactions.

## C. LLM-Assisted Annotation

LLMs have increasingly been used for scalable text annotation [19], [20]. Classifiers trained on LLM-generated labels can approach the performance of models trained on human annotations [19], although subjective labels may be affected by annotation bias and model suggestions [21]. We therefore use an LLM to construct pseudo-labeled training data, validate a sampled subset through independent human annotation, and train separate encoder-based classifiers for large-scale ragebait detection.

## III. DATASET CONSTRUCTION

## A. Motivation and Overview

Constructing a ragebait dataset presents two major challenges. First, identifying ragebait requires assessing not only the surface wording of a post but also the author’s likely intention to provoke and the emotional reactions that the post may elicit from readers, making large-scale manual annotation costly. Second, ragebait accounts for only a small proportion of randomly sampled social media posts, and simple random sampling therefore yields too few positive instances for training a reliable classifier. To address these challenges, we adopt a two-stage dataset construction procedure. We first use an LLM to assign pseudo-labels to a randomly sampled subset and train an initial ragebait detector. We then apply this detector to the remaining data to retrieve likely ragebait candidates, which are subsequently relabeled by the LLM. Fig. 1 provides an overview of the complete pipeline.

## B. Source Data and Filtering

We use a 1% sample of Japanese-language X data collected between October 2022 and June 2023. The raw data contain 153,849,869 event records, including original posts and engagement events such as reposts, replies, and quotes. Because each repost event preserves the corresponding original post together with its engagement counts at the time the event was recorded, we first extract the original posts associated with these events. We then retain posts containing at least 20 characters and require that at least one of the repost, like, reply, or quote counts exceeds 50. This filtering criterion allows us to focus on posts that received a certain level of public attention and for which engagement-based analyses are feasible. We further remove posts containing keywords associated with campaigns, lotteries, advertisements, promotions, and similar commercial content. After filtering, 5,792,059 original posts remain.

## C. First-Stage GPT-Based Labeling

We randomly sample 20,000 posts from the filtered data and use GPT-5.4 mini<sup>2</sup> to determine whether each post constitutes ragebait. The labeling prompt is based on the definition of ragebait and asks the model to consider whether the post appears intended to anger, upset, or provoke readers and whether it is likely to elicit anger, indignation, disgust, or strong discomfort. GPT-5.4 mini assigns a binary label of YES or NO to each post. Among the 20,000 sampled posts, 822 are labeled as ragebait and 19,178 as non-ragebait. The resulting positive rate is only 4.11%, illustrating that random sampling alone is inefficient for obtaining a sufficient number of ragebait examples.

To train an initial detector, we retain all 822 ragebait posts and randomly sample the same number of non-ragebait posts, resulting in a balanced seed dataset of 1,644 posts. We allocate 1,444 posts to the training set and 200 posts to the test set, with both sets containing equal numbers of ragebait and nonragebait instances. We then fine-tune Rinna RoBERTa base<sup>3</sup> as a binary classifier. The initial detector achieves an accuracy of 0.74 on the balanced test set. This detector is used only to retrieve additional candidate posts and is not used as the final classifier in our subsequent analyses.

![](images/b6eef2f5e4a9e8cb077d9dc79eba2cd390d9daace5bf35485251b647f0fe35ce.jpg)  
Fig. 1. Overview of the ragebait dataset construction pipeline.

## D. Detector-Assisted Data Expansion

We apply the initial detector to the remaining filtered posts and obtain a predicted ragebait probability for each post. To efficiently collect additional positive examples, we select 30,000 posts with high predicted ragebait probabilities and high engagement counts. Incorporating engagement information enables us to prioritize posts that are both likely to be ragebait and sufficiently visible to generate observable diffusion and audience reactions.

The selected 30,000 candidates are then classified again by GPT-5.4 mini using the same definition and labeling criteria as in the first stage. In this second labeling stage, 8,747 posts are labeled as ragebait and 21,253 as non-ragebait. Compared with the initial random sample, the proportion of positive instances increases substantially, demonstrating that the initial detector is effective for retrieving likely ragebait candidates.

## E. Final Dataset

We combine the labeled instances obtained from the two GPT-based labeling stages and construct a balanced dataset containing equal numbers of ragebait and non-ragebait posts. The final dataset contains 18,558 posts, consisting of 9,279 ragebait and 9,279 non-ragebait instances. We allocate 16,558 posts to the training set and 2,000 posts to the held-out test set. Both sets maintain a balanced class distribution, with ragebait and non-ragebait each accounting for 50% of the instances. The held-out test set is used exclusively to evaluate the final ragebait classifiers described in Section IV.

## F. Human Validation of GPT-Generated Labels

Because ragebait judgments involve subjective assessments of communicative intention and anticipated emotional impact, we conduct an independent human evaluation of the GPTgenerated labels. We randomly sample 200 posts from the pseudo-labeled dataset, including 100 posts labeled as ragebait and 100 labeled as non-ragebait. Two annotators independently determine whether each post constitutes ragebait without access to the GPT-generated labels.

The annotators are instructed to consider three criteria. First, they assess whether the post appears to have been written with the intention of angering, upsetting, or provoking readers. Second, they consider whether reading the post personally elicits anger, indignation, disgust, or strong discomfort. Third, they judge whether other readers would be likely to experience similar reactions. These criteria reflect both the presumed intention of the author and the anticipated emotional effects of the post.

The agreement rate between Annotator A and GPT is 75.0%, with Cohen’s $\kappa = 0 . 5 0 0$ , while the agreement rate between Annotator B and GPT is 74.5%, with $\kappa = 0 . 4 9 8$ . The two human annotators achieve an agreement rate of 78.5%, with $\kappa = 0 . 5 7 0$ . Thus, the agreement between each human annotator and GPT is close to the agreement observed between the two human annotators. Although the GPT-generated labels should not be regarded as equivalent to manually established gold-standard labels, these results indicate that they possess a reasonable degree of reliability for constructing training data for ragebait detection. Table I shows English translations of representative Japanese posts for which GPT and both human annotators assigned the same labels.

## IV. RAGEBAIT DETECTION MODELS

## A. Experimental Setup

Using the dataset described in Section III, we train six Japanese pretrained language models for binary ragebait classification: Tohoku BERT base $\mathrm { v } 3 ^ { 4 }$ , Rinna RoBERTa base, LINE DistilBERT base<sup>5</sup>, Waseda RoBERTa base<sup>6</sup>, KU-NLP DeBERTa base<sup>7</sup>, and KU-NLP DeBERTa large<sup>8</sup>. The 16,558 training instances are further divided into training and validation subsets using stratified random sampling based on the Ragebait and Non-Ragebait labels, with 90% used for training and 10% for validation. The held-out test set of 2,000 instances is used only for final evaluation.

TABLE I  
EXAMPLES OF RAGEBAIT AND NON-RAGEBAIT POSTS. ENGLISH TRANSLATIONS IN PARENTHESES WERE TRANSLATED USING GPT-5.5.
<table><tr><td>Label</td><td>Post</td></tr><tr><td>Ragebait</td><td>「右でも左でもなく」って言ってる人ほぽ右なんだよな。なんでかな。 (People who say that they are &quot;neither right-wing nor left-wing&quot; are almost always right-wing. I wonder why.)</td></tr><tr><td>Ragebait</td><td>いまだに政府の発表を信じて、それに反する意見には反ワクとか雑なレッテルを貼ってる人を見ると、以前は腹が立ったが、今はもうアホは 一生アホなんだなとしか思わない。 (When I see people who still believe government announcements and casually label opposing views as anti-vaccine, I used to get angry, but now I simply</td></tr><tr><td>Ragebait</td><td>think that fools remain fools for life.) この世の中に、良い人、優しい人、正しい人、義理堅い人、筋が通っている人、ちゃんとしている人はほとんど居ません。みんな自分は大丈 夫と見せかけているだけで、多くの人か糞で味噌でカスです。 (There are almost no good, kind, righteous, loyal, principled, or decent people in this world. Everyone merely pretends to be decent, while many people are nothing but filth and scum.)</td></tr><tr><td>Non-Ragebait</td><td>立憲民主党の泉代表が、小西洋之氏の憲法審野党筆頭幹事の職を解くと発表。事実上の更迭とは言っているから、更迭ではない。ずいぷん甘 い処分だ。 (Izumi, leader of the Constitutional Democratic Party, announced that Hiroyuki Konishi would be removed from his position as the opposition&#x27;s lead director on the Commission on the Constitution. Although it was described as a de facto dismissal, it was not an actual dismissal. The punishment was</td></tr><tr><td>Non-Ragebait</td><td>far too lenient.) 人がちょっと精神的に荒れてるからって、人格に問題があるとか言ったら本人が傷つくだろ。自分がただでさえ苦しんでるときに、泣き叫ん でも誰にも理解されず、逆に頭のおかしい奴みたいに扱われたらどういう気分になるか考えてみろよ。そういうのか結果的に差別やいじめに つながったりするんだよ。 (Saying that someone has a personality problem simply because they are experiencing mental distress will hurt them. Imagine how it would feel to already be suffering, to cry out without being understood, and instead to be treated as if you were mentally unstable. Such treatment can ultimately lead to</td></tr><tr><td>Non-Ragebait</td><td>またホットクックがトレンド入りしてるんですが、「片付けるのが苦手な人」や「流し台か狭い人」や「数時間前から料理の計画を立てられ  $\begin{array} { r }  \mathcal { Z } ( \mathtt { a } ) , \mathtt { b } , | \mathtt { Z } | \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { e } _ { - } \gamma _ { \mathtt { c } } \le | \mathtt { b } | \mathtt { d } \mathtt { p } _ { + } \gamma _ { \mathtt { c } } \mathtt { d } \mathtt { d } \gamma _ { \mathtt { c } } \mathtt { d } \mathtt { d } \mathtt { e } _ { \mathtt { c } } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { p } _ { + } \gamma \le \langle \mathtt { f } _ { \mathtt { c } } \rVert \mathtt { d } \mathtt { e } _ { \mathtt { c } } \mathtt { d } \mathtt { d } \mathtt { s } _ { + } \ \mathtt { s } _ { + } \mathtt { s } \mathtt { m } _ { + } \mathtt { s } \mathtt { f } \mathtt { h } \mathtt { d } \mathtt { e } _ { - } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { p } _ { + } \mathtt { d } \mathtt { d } \mathtt { e } _ { - } \mathtt { h } \mathtt { e } _ { \mathtt { d } } \mathtt { d } \mathtt { d } \mathtt { p } _ { - } \mathtt { b } \mathtt { e } _ { \mathtt { c } } \mathtt { d } \mathtt { p } _ { - } \mathtt { f } \mathtt { e } _ { - } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { p } _ { \mathtt { d } } \mathtt { e } _ { \mathtt { c } } \mathtt { m } _ { + } \mathtt { p } \mathtt { d } \mathtt { e } _ { - } \mathtt { f } \mathtt { d } \mathtt { d } \mathtt { e } _ { \mathtt { c } } \mathtt { d } \mathtt { p } _ { + } \mathtt { e } _ { \mathtt { c } } \mathtt { d } \mathtt { p } _ { - } \mathtt { f } \mathtt { d } \mathtt { d } \mathtt { e } _ { - } \mathtt { d } \mathtt { d } \mathtt { p } _ { \mathtt { c } } \mathtt { d } \mathtt { q } _ { \mathtt { d } } \mathtt { e } _ { - } \mathtt { f } \mathtt { d } \mathtt { d } \mathtt { e } _ { - } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { p } _ { \mathtt { c } } \mathtt { d } \mathtt  q  \end{array}$  (Hot Cook is trending again, but it is completely unsuitable for people who are not good at cleaning up, have a small kitchen sink, or cannot plan their meals several hours in advance. Please be careful. I paid nearly 50,000 yen for it but used it only during the first year. Even washing the inner pot is a chore.)</td></tr></table>

All models are fine-tuned for three epochs using AdamW. The training batch size is set to 16, while the validation batch size is set to 32. We search learning rates from $\lbrace 1 \times 1 0 ^ { - 5 } , 2 \times$ $1 0 ^ { - 5 } , 3 \times 1 0 ^ { - 5 } \}$ and select the checkpoint with the highest Macro-F1 score on the validation set. We report Accuracy, Precision, Recall, and Macro-F1 on the held-out test set.

## B. Classification Results

Table II presents the classification performance of the six individual models. Among the single models, Tohoku BERT base v3 achieves the highest Accuracy and Macro-F1, reaching 83.40% and 83.38%, respectively. It also obtains the highest Recall of 87.00%. Rinna RoBERTa base achieves the highest Precision of 83.10%, while LINE DistilBERT base also shows competitive overall performance.

## C. Ensemble Classifier

Although the three best-performing individual models achieve similar overall performance, they are based on different architectures and pretraining resources and may therefore exhibit different error patterns. Relying on a single model could make the final predictions more sensitive to modelspecific biases and occasional misclassifications. To improve robustness and reduce the influence of any one model, we combine the binary predictions of Tohoku BERT base v3,

Rinna RoBERTa base, and LINE DistilBERT base using majority voting.

The resulting ensemble achieves an Accuracy of 84.05% and a Macro-F1 score of 84.04%, outperforming all individual models on both metrics. Its Precision and Recall are 82.65% and 86.20%, respectively, indicating a favorable balance between detecting ragebait posts and limiting false positives. Based on its stronger overall performance and improved robustness, we adopt the majority-voting ensemble as the final ragebait detector for the large-scale analyses described in the following sections.

## V. LARGE-SCALE ANALYSIS OF POSTS ON X

We apply the ensemble detector described in Section IV to a large-scale data derived from a 1% sample of Japaneselanguage X collected between October 2022 and June 2023. From 153,849,869 event records, we extract and deduplicate the original posts associated with repost events, because these records preserve the engagement counts of the original posts at the time each repost was observed, enabling subsequent analyses of diffusion. To focus the analysis on general public discourse and reduce contamination from commercial or promotional content, we filter out posts matching a predefined list of keywords related to advertisements, lotteries, campaigns, and promotions. This results in 17,435,881 posts, of which 1,118,931 posts (6.42%) are classified as ragebait and 16,316,950 as non-ragebait. We use this data to analyze lexical characteristics, thematic patterns, and diffusion speed.

TABLE II  
PERFORMANCE OF THE RAGEBAIT DETECTION MODELS ON THE TEST SET. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td>Model</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>Macro-F1 (%)</td></tr><tr><td>Tohoku BERT base v3</td><td>83.40</td><td>81.16</td><td>87.00</td><td>83.38</td></tr><tr><td>Rinna RoBERTa base</td><td>82.90</td><td>83.10</td><td>82.60</td><td>82.90</td></tr><tr><td>LINE DistilBERT base</td><td>82.65</td><td>81.79</td><td>84.00</td><td>82.65</td></tr><tr><td>Waseda RoBERTa base</td><td>80.65</td><td>77.99</td><td>85.40</td><td>80.61</td></tr><tr><td>KU-NLP DeBERTa large</td><td>80.60</td><td>81.29</td><td>79.50</td><td>80.60</td></tr><tr><td>KU-NLP DeBERTa base</td><td>79.70</td><td>77.15</td><td>84.40</td><td>79.66</td></tr><tr><td>LINE + Rinna + Tohoku Majority Vote</td><td>84.05</td><td>82.65</td><td>86.20</td><td>84.04</td></tr></table>

![](images/7adb50e9820aa98623da683007ba48dc0c6b7131069301e70bf6f5903b64627d.jpg)  
Fig. 2. Word clouds of ragebait and non-ragebait posts. Word size represents within-group frequency. English translations are from GPT-5.5.

## A. Lexical Characteristics

We first examine whether ragebait and non-ragebait posts exhibit distinct lexical patterns. This analysis provides an interpretable overview of the language associated with each group and helps assess whether the detector identifies semantically coherent differences rather than merely general negative sentiment.

We preprocess the post text and perform morphological analysis using fugashi<sup>9</sup>. We retain nouns, verbs, adjectives, and adverbs, while removing highly general terms. We then calculate token frequencies separately for ragebait and nonragebait posts and construct the word clouds shown in Fig. 2. Larger words indicate higher frequencies within each group. Considering that raw frequency can be dominated by terms that are common to both groups, we further compute weighted log-odds z-scores [22]. This measure identifies terms that are overrepresented in one group relative to the other while accounting for differences in data size and overall token frequency. Fig. 3 presents the 15 most distinctive terms for each group.

Taken together, the word clouds and weighted log-odd results reveal clear lexical differences between the two groups. Ragebait posts more frequently contain terms related to politics, social identity, public health, and social conflict, including “Japan,” “women,” “LDP,” “vaccine,” “discrimination,” “China,” “crime,” and “LGBT.” By contrast, non-ragebait posts more frequently contain terms associated with everyday communication, creative activities, and recreational content, such as “wish,” “today,” “best regards,” “cute,” “photo,” “enjoyable,” “draw,” and “happy.” These patterns indicate that, within the analyzed data, posts classified as ragebait are more lexically concentrated around contentious public issues, whereas non-ragebait posts more often concern everyday interaction and leisure-related topics.

## B. Differences in Topic Prevalence

Lexical analysis captures individual words, but it does not directly reveal broader thematic structures. We therefore analyze the thematic differences between ragebait and nonragebait posts using Structural Topic Modeling (STM) [23]. STM estimates latent topics in a document collection and can incorporate document-level metadata, enabling us to compare topic prevalence between the two groups while considering temporal variation.

For this analysis, we use 1,073,827 ragebait posts and 13,028,190 non-ragebait posts after additional preprocessing for topic modeling. Because the two groups differ greatly in size, we sample 100,000 posts from each group for STM training while preserving the monthly distribution of the original data. We use the ragebait label and posting month as document metadata. The ragebait label is used to compare topic prevalence between the two groups, while posting month is included to reduce the influence of temporal shifts in topic composition.

![](images/3aa17a4087cd75714cfb3b1bea5143b041ed2f5301c58353ef44b6c44bcfa372.jpg)

![](images/fb8b3b9854dfbc3f3bc0b1585638a82e0f75180c46ff80a5ad976e602c6e32a5.jpg)  
Fig. 3. Top 15 distinctive words for ragebait and non-ragebait posts ranked by weighted log-odds z-score.

We compare models with $K ~ \in ~ \{ 1 5 , 2 0 , 2 5 \}$ topics and select the final number of topics based on held-out likelihood, exclusivity, and semantic coherence. Based on these criteria, we adopt $K = 2 5$ and apply the trained STM to the full posts.

Fig. 4 shows that ragebait posts have higher topic prevalence for party politics and national governance, gender and discrimination, vaccines and infectious diseases, interpersonal conflict and moral condemnation, and public spending, welfare, and litigation. In contrast, non-ragebait posts show higher prevalence for topics related to sports, daily interaction, creative production, anime and games, and information exchange.

The four most prevalent ragebait-associated topics—party politics and national governance, interpersonal conflict and moral condemnation, vaccines and infectious diseases, and gender and discrimination—account for 52.22% of the mean topic prevalence in ragebait posts, compared with 14.78% for the same topics in non-ragebait posts. This result indicates that ragebait is not evenly distributed across topics, but is concentrated in a limited set of socially contentious themes where disagreement and emotional reactions are more likely to occur.

## C. Diffusion Dynamics

We next compare the diffusion dynamics of ragebait and non-ragebait posts. This analysis is motivated by the possibility that ragebait may not only differ in content, but also in how quickly it elicits engagement after being posted.

To enable temporal analysis, we restrict the data to original posts that were observed at least five times in the repost-event records. This allows us to track changes in engagement counts across multiple time points for the same post. We examine four engagement indicators: reposts, likes, replies, and quotes.

![](images/c7fb50ecf00bf509fae4e228412de1edea7003a086364ba7bab3d6ba31b27637.jpg)  
Fig. 4. Mean topic prevalence in ragebait and non-ragebait posts. Points above the diagonal are more prevalent in ragebait posts, while points below it are more prevalent in non-ragebait posts.

For each post, we sort its observations chronologically and compute the increase in each engagement count between two consecutive observations. We then divide this increment by the elapsed time between the observations to obtain an hourly growth rate. To reduce the influence of account size, we normalize the growth rate by the author’s follower count and report the median growth per 10,000 followers. Finally, we group the results by elapsed time since posting and compare ragebait and non-ragebait posts within each time bin.

![](images/20dab7545a62cb358644693c12d241d4073e4b85eee40c09434d236de170727f.jpg)  
Fig. 5. Median engagement growth per 10,000 followers over time for ragebait and non-ragebait posts: (a) reposts, (b) likes, (c) replies, and (d) quotes.

Fig. 5 shows that the growth rate is highest immediately after posting for all four engagement indicators and gradually declines over time. For reposts, ragebait posts show slightly higher growth than non-ragebait posts after the first hour and continue to maintain an advantage across several later time bins. For likes, the two groups follow very similar trajectories, with only small differences throughout the observed period.

More pronounced differences appear for replies and quotes. In both cases, ragebait posts show consistently higher growth rates than non-ragebait posts, and this gap remains visible from the first few hours after posting to several days later. These results suggest that ragebait posts are not only reposted at a slightly faster rate, but more importantly, they are more likely to trigger replies and quotes, which often reflect discussion, disagreement, or reactive engagement. In this sense, ragebait appears to facilitate interaction patterns that extend beyond passive exposure and are more likely to sustain public response over time.

## D. Emotional Reactions in Replies and Quotes

Finally, we examine the emotional characteristics of replies and quote posts directed at ragebait and non-ragebait posts. Unlike the preceding analyses, which focus on the detected original posts themselves, this analysis uses a separate reaction dataset consisting of original posts that received at least one reply or quote. After deduplication, we obtain 34,523,063 original posts. We classify these original posts using the threemodel majority-voting detector, resulting in 846,962 ragebait posts (2.45%) and 33,676,101 non-ragebait posts (97.55%). Based on this classification, we compare the sentiment and fine-grained emotions expressed in replies and quote posts.

1) Binary Sentiment: We first conduct binary sentiment classification to examine whether reactions to ragebait are more negative than reactions to non-ragebait. We use a Japanese sentiment classification model<sup>10</sup> trained on WRIME [24] and classify each reply and quote post as either positive or negative. Table III shows that reactions to ragebait are substantially more negative than reactions to non-ragebait. Among replies to ragebait posts, 52.02% are classified as negative, compared with 21.44% for replies to non-ragebait posts. The difference is even larger for quote posts: 63.49% of quotes directed at ragebait are negative, compared with 22.74% for quotes directed at non-ragebait. This suggests that ragebait posts are more likely to elicit negative audience reactions, particularly through quote posts.

2) Fine-Grained Emotion Analysis: We further analyze fine-grained emotions using a Japanese emotion classification model fine-tuned on WRIME<sup>11</sup>. We examine eight basic emotions: joy, trust, anticipation, surprise, sadness, fear, anger, and disgust. Because the model treats each emotion independently, a single reply or quote post may be assigned multiple emotions. Following the model setting, we apply ROC-optimized thresholds for each emotion and regard an emotion as present when its predicted score exceeds the corresponding threshold.

Table III shows that replies and quote posts directed at ragebait contain higher rates of sadness, surprise, anger, fear, and disgust than those directed at non-ragebait. The contrast is especially clear for anger and disgust. For example, anger appears in 26.58% of replies and 35.53% of quotes directed at ragebait, whereas it appears in only 4.27% of replies and 6.40% of quotes directed at non-ragebait. Similarly, disgust appears in 57.18% of replies and 69.46% of quotes directed at ragebait, compared with 19.41% and 21.18% for non-ragebait.

By contrast, joy, anticipation, and trust are more frequent in reactions to non-ragebait posts. These results indicate that ragebait does not merely increase negative sentiment in general; rather, it is associated with specific negative emotions such as anger, fear, and disgust. The tendency is particularly pronounced in quote posts, suggesting that ragebait is more likely to trigger reactive or critical forms of engagement.

## VI. CONCLUSION

This study developed a ragebait detector for Japaneselanguage posts on X and conducted a large-scale analysis of ragebait. We constructed an LLM-generated pseudo-labeled dataset of approximately 18K posts, validated the annotation reliability through independent human evaluation, and trained an ensemble classifier that achieved 84.05% accuracy and a Macro-F1 score of 84.04%. Applying the detector to largescale X data, we found that ragebait accounted for only a small proportion of the analyzed posts. Ragebait was more prevalent in politically and socially contentious topics and showed stronger growth in replies and quotes. Reactions to ragebait also contained more negative sentiment and higher rates of anger, fear, and disgust than reactions to non-ragebait.

Overall, these findings suggest that ragebait is a distinct form of emotionally provocative content rather than merely negative or offensive language. Although relatively uncommon, it appears to have a disproportionate capacity to stimulate conflict-oriented interaction and negative emotional responses. The proposed detector provides a practical basis for large-scale analysis and future mitigation efforts.

TABLE III  
BINARY SENTIMENT AND FINE-GRAINED EMOTION OCCURRENCE RATES IN REPLIES AND QUOTE POSTS. ALL COLUMNS EXCEPT N ARE PERCENTAGES;SHADED CELLS INDICATE THE HIGHER VALUE WITHIN EACH COMPARISON.
<table><tr><td rowspan="2">Reaction</td><td rowspan="2">Target</td><td rowspan="2">N</td><td colspan="2">Binary Sentiment</td><td colspan="8">Fine-Grained Emotions</td></tr><tr><td>Positive</td><td>Negative</td><td>Joy</td><td>Trust</td><td>Ant.</td><td>Sur.</td><td>Sad.</td><td>Fear</td><td>Ang.</td><td>Disg.</td></tr><tr><td rowspan="2">Replies</td><td>Ragebait</td><td>819,406</td><td>47.98</td><td>52.02</td><td>48.62</td><td>19.41</td><td>55.91</td><td>64.08</td><td>42.47</td><td>59.93</td><td>26.58</td><td>57.18</td></tr><tr><td>Non-ragebait</td><td>33,256,332</td><td>78.56</td><td>21.44</td><td>81.64</td><td>34.78</td><td>70.58</td><td>50.53</td><td>25.52</td><td>29.33</td><td>4.27</td><td>19.41</td></tr><tr><td rowspan="2">Quotes</td><td>Ragebait</td><td>215,287</td><td>36.51</td><td>63.49</td><td>43.12</td><td>22.28</td><td>61.37</td><td>75.17</td><td>42.04</td><td>64.00</td><td>35.53</td><td>69.46</td></tr><tr><td>Non-ragebait</td><td>3,605,330</td><td>77.26</td><td>22.74</td><td>82.72</td><td>38.19</td><td>79.02</td><td>62.85</td><td>21.47</td><td>24.97</td><td>6.40</td><td>21.18</td></tr></table>

## ACKNOWLEDGMENT

This work was supported by JST ERATO (JPMJER2502) and MEXT SPReAD (JPMXP1726302875).

## REFERENCES

[1] Oxford University Press, “The Oxford Word of the Year 2025 is rage bait,” Dec. 1, 2025. [Online]. Available: https://corp.oup.com/ news/the-oxford-word-of-the-year-2025-is-rage-bait/ [Accessed: Jul. 6, 2026].

[2] M. Potthast, T. Gollub, K. Komlossy, S. Schuster, M. Wiegmann, E. P. Garces Fernandez, M. Hagen, and B. Stein, “Crowdsourcing a large corpus of clickbait on Twitter,” in Proceedings of the 27th International Conference on Computational Linguistics, Santa Fe, NM, USA, 2018, pp. 1498–1507.

[3] W. J. Brady, J. A. Wills, J. T. Jost, J. A. Tucker, and J. J. Van Bavel, “Emotion shapes the diffusion of moralized content in social networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 28, pp. 7313–7318, 2017, doi: 10.1073/pnas.1618923114.

[4] W. J. Brady, K. McLoughlin, T. N. Doan, and M. J. Crockett, “How social learning amplifies moral outrage expression in online social networks,” Science Advances, vol. 7, no. 33, Art. no. eabe5641, 2021, doi: 10.1126/sciadv.abe5641.

[5] R. Fan, K. Xu, and J. Zhao, “Weak ties strengthen anger contagion in social media,” arXiv preprint arXiv:2005.01924, 2020, doi: 10.48550/arXiv.2005.01924.

[6] J. Han, S. E. Lee, and M. Cha, “The secret to successful evocative messages: Anger takes the lead in information sharing over anxiety,” Communication Monographs, vol. 90, no. 4, pp. 545–565, 2023, doi: 10.1080/03637751.2023.2236183.

[7] E. S. Ohman and A. Liimatta, “Text length and the function of intentionality: A case study of contrastive subreddits,” in Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities, Miami, USA, 2024, pp. 1–8, doi: 10.18653/v1/2024.nlp4dh-1.1.

[8] J. Shin, C. DeFelice, and S. Kim, “Emotion sells: Rage bait vs. information bait in clickbait news headlines on social media,” Digital Journalism, vol. 13, no. 7, pp. 1271–1290, 2025, doi: 10.1080/21670811.2025.2505566.

[9] P. Biyani, K. Tsioutsiouliklis, and J. Blackmer, “8 amazing secrets for getting more clicks: Detecting clickbaits in news streams using article informality,” in Proceedings of the 30th AAAI Conference on Artificial Intelligence, Phoenix, AZ, USA, 2016, pp. 94–100, doi: 10.1609/aaai.v30i1.9966.

[10] A. Chakraborty, B. Paranjape, S. Kakarla, and N. Ganguly, “Stop clickbait: Detecting and preventing clickbaits in online news media,” in Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, San Francisco, CA, USA, 2016, pp. 9–16, doi: 10.1109/ASONAM.2016.7752207.

[11] V. Indurthi, B. Syed, M. Gupta, and V. Varma, “Predicting clickbait strength in online social media,” in Proceedings ofthe 28th International Conference on Computational Linguistics, Barcelona, Spain, 2020, pp. 4835–4846, doi: 10.18653/v1/2020.coling-main.425.

[12] M. Zampieri, S. Malmasi, P. Nakov, S. Rosenthal, N. Farra, and R. Kumar, “SemEval-2019 Task 6: Identifying and categorizing offensive language in social media (OffensEval),” in Proceedings of the 13th International Workshop on Semantic Evaluation, Minneapolis, MN, USA, 2019, pp. 75–86, doi: 10.18653/v1/S19-2010.

[13] T. Davidson, D. Warmsley, M. Macy, and I. Weber, “Automated hate speech detection and the problem of offensive language,” in Proceedings of the 11th International AAAI Conference on Web and Social Media, Montreal, Canada, 2017, pp. 512–515, doi: 10.1609/icwsm.v11i1.14955.

[14] Z. Waseem, T. Davidson, D. Warmsley, and I. Weber, “Understanding abuse: A typology of abusive language detection subtasks,” in Proceedings of the First Workshop on Abusive Language Online, Vancouver, Canada, 2017, pp. 78–84, doi: 10.18653/v1/W17-3012.

[15] M. J. Crockett, “Moral outrage in the digital age,” Nature Human Behaviour, vol. 1, no. 11, pp. 769–771, 2017, doi: 10.1038/s41562-017- 0213-3.

[16] R. Fan, J. Zhao, Y. Chen, and K. Xu, “Anger is more influential than joy: Sentiment correlation in Weibo,” PLOS ONE, vol. 9, no. 10, Art. no. e110184, 2014, doi: 10.1371/journal.pone.0110184.

[17] S. Rathje, J. J. Van Bavel, and S. van der Linden, “Out-group animosity drives engagement on social media,” Proceedings of the National Academy of Sciences, vol. 118, no. 26, Art. no. e2024292118, 2021, doi: 10.1073/pnas.2024292118.

[18] K. L. McLoughlin, W. J. Brady, A. Goolsbee, B. Kaiser, K. Klonick, and M. J. Crockett, “Misinformation exploits outrage to spread online,” Science, vol. 386, no. 6725, pp. 991–996, 2024, doi: 10.1126/science.adl2829.

[19] N. Pangakis and S. Wolken, “Knowledge distillation in automated annotation: Supervised text classification with LLM-generated training labels,” in Proceedings of the Sixth Workshop on Natural Language Processing and Computational Social Science, Mexico City, Mexico, 2024, pp. 113–131, doi: 10.18653/v1/2024.nlpcss-1.9.

[20] F. Gilardi, M. Alizadeh, and M. Kubli, “ChatGPT outperforms crowd workers for text-annotation tasks,” Proceedings of the National Academy of Sciences, vol. 120, no. 30, Art. no. e2305016120, 2023, doi: 10.1073/pnas.2305016120.

[21] H. Schroeder, D. Roy, and J. Kabbara, “Just put a human in the loop? Investigating LLM-assisted annotation for subjective tasks,” in Findings of the Association for Computational Linguistics: ACL 2025, Vienna, Austria, 2025, pp. 25771–25795, doi: 10.18653/v1/2025.findings-acl.1323.

[22] B. L. Monroe, M. P. Colaresi, and M. J. Quinn, “Fightin’ words: Lexical feature selection and evaluation for identifying the content of political conflict,” Political Analysis, vol. 16, no. 4, pp. 372–403, 2008, doi: 10.1093/pan/mpn018.

[23] M. E. Roberts, B. M. Stewart, and D. Tingley, “stm: An R package for structural topic models,” Journal of Statistical Software, vol. 91, no. 2, pp. 1–40, 2019, doi: 10.18637/jss.v091.i02.

[24] T. Kajiwara, C. Chu, N. Takemura, Y. Nakashima, and H. Nagahara, “WRIME: A new dataset for emotional intensity estimation with subjective and objective annotations,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Online, 2021, pp. 2095– 2104, doi: 10.18653/v1/2021.naacl-main.169.