# Nobody Truly Agrees on Sentiment: Humans, Bespoke Tools, and LLMs Struggle with Social Media Texts

Himarsha R. Jayanetti<sup>1[0000−1111−2222−3333]</sup>, Sivakanesan Dhanushkanda<sup>1[0000−0001−6452−5576]</sup> , Shuai Hao<sup>1[0000−0001−7483−5252]</sup>, Michael L. Nelson<sup>1[0000−0003−3749−8116]</sup> , and Michele C. Weigle<sup>1[0000−0002−2787−7166]</sup>

Old Dominion University, Norfolk, Virginia, USA {hjaya002,sdhan005,shao}@odu.edu {mln,mweigle}@cs.odu.edu

Abstract. Social media is a rich source of real-time public sentiment, but widely used sentiment analysis tools are often applied without understanding their limitations. In this study, we evaluate the inter-rater reliability of three bespoke sentiment analysis tools (TextBlob, VADER, and Twitter-roBERTa-base) and three large language models (LLMs: Qwen3-32B, GPT-OSS-120B, Llama-4-Maverick-17B) against six human raters across 100 tweets. We measured agreement using two statistical measures: Cohen’s kappa for pairwise comparisons and Fleiss’ kappa for multiple raters. Even among the human raters, our results showed only fair agreement, highlighting the subjectivity of sentiment analysis. Higher agreement was observed under the binary sentiment classification (negative vs. non-negative and positive vs. non-positive) than under the three-class classification across both humans and automated tools. The Twitter-roBERTa-base model showed the strongest alignment with human ratings, outperforming both bespoke sentiment tools and LLMs, particularly in distinguishing negative versus non-negative sentiment. LLMs showed substantial agreement among themselves and moderate to substantial alignment with humans, performing better in positive vs. non-positive classifications. Our findings underscore that domain-specific fine-tuning remains crucial for reliable social media sentiment analysis, and human-centered evaluation remains essential for establishing goldstandard labels.

Keywords: Sentiment Analysis · Opinion · Sentiment Analysis Tools · TextBlob · VADER · Twitter-roBERTa-base model · TweetEval · GPT-OSS · Llama · Qwen

## 1 Introduction

In recent years, social media has become a valuable tool to capture real-time public sentiment. Sentiment analysis, a key task in natural language processing (NLP), enables the classification of such text as positive, negative, or neutral. However, the accessibility of existing bespoke sentiment analysis tools has led to their use without fully understanding their limitations. Running text through a tool and accepting its output at face value can be misleading, especially when the nuances of human interpretation are overlooked. Through our study we learned that even humans do not consistently agree on the sentiment of short, contextpoor posts, and automated tools inherit the same ambiguity.

In this study, we provide an empirical examination of sentiment analysis reliability by comparing both bespoke tools and modern Large Language Models (LLMs) directly against human judgment. Specifically, we analyzed the sentiment of 100 randomly selected tweets, small enough for detailed, qualitative review using three bespoke sentiment analysis tools (TextBlob [23], VADER [15], and Twitter-roBERTa-base model [3]) as well as three large language models (Qwen3-32B [30, 14], GPT-OSS-120B [12], and Llama-4-Maverick-17B [13]), and compared their outputs with the ratings of six human raters. The deliberately small dataset enabled detailed qualitative review and independent manual annotation by all six raters. To measure agreement, we calculated Cohen’s kappa [7] for each pair of raters (including both humans and tools) and Fleiss’ kappa [11] for multiple raters.

The main contribution of this work is a direct comparison of three sentiment analysis tools and three LLMs with multiple human raters, using quantitative measures of inter-rater agreement. Our results revealed a cautionary truth: no tool, LLM or otherwise, reliably matches human sentiment judgments, and even human raters only show fair agreement with each other. To address this, we took a majority vote among human raters and then compared it with the tools. Notably, the Twitter-roBERTa-base model showed the best alignment with human rating. We also compared three-class (positive, negative, neutral) and binary (negative vs. non-negative, positive vs. non-positive) sentiment classification. Our findings show higher agreement under the binary sentiment classification than under the three-class classification, with negative vs. non-negative distinctions achieving the strongest agreement. Our study provides practical insights for researchers and practitioners on selecting appropriate methods for analyzing social media sentiment, while highlighting that both bespoke tools and modern LLMs struggle to fully capture the nuances of human judgment in short-form social media texts. The persistence of this challenge, along with the assumption that newer or larger models yield more reliable results, underscores the need for careful evaluation and responsible use of automated methods. Sentiment analysis of short social media posts remains inherently noisy, requiring researchers to approach automated tools, old and new, with appropriate caution.

## 2 Background and Related Work

Sentiment analysis [33, 21] is a branch of NLP that examines people’s expressions of emotions, opinions, or stances on a topic. It considers both polarity (positive, negative, or neutral) and valence (emotional intensity) to classify text [31]. Sentiment analysis is used in many areas including marketing, finance, tourism, politics, and many other fields. Sentiment analysis is also widely applied to social media text, particularly Twitter (now X), where posts are brief, informal, and often filled with slang, hashtags, and sarcasm [32, 1, 27].

For this study, we focused on three representative sentiment analysis tools as well as three LLMs:

1. TextBlob: A Python library for processing textual data that includes a sentiment analysis feature that returns a Sentiment (polarity, subjectivity) tuple, where polarity ranges from [-1.0, 1.0] and subjectivity ranges from [0.0, 1.0]. We only used the polarity score to assign sentiment labels (positive, neutral, and negative) in this study.

2. VADER: A rule-based sentiment analysis model specifically developed for social media text.

3. Twitter-roBERTa-base model: A model trained on 58M tweets and fine-tuned for sentiment analysis with the TweetEval benchmark in 2020 [28, 5].

4. LLMs: Qwen3-32B, GPT-OSS-120B, and Llama-4-Maverick-17B were tested with zero- and one-shot prompting. One-shot prompting performed slightly better or comparable to zero-shot. Since zero-shot results did not afect the overall conclusions, we report only one-shot results for clarity. The results for the zero-shot experiments, along with the prompts used for both the zero-shot and one-shot settings, are available in the project GitHub repository [17].

Several studies have used TextBlob, VADER, and BERT-based models for analyzing public sentiment on social media platforms [2, 24, 6, 4]. Comparative studies show that VADER often outperforms TextBlob for short, casual posts, while transformer-based models excel in capturing nuanced, context-dependent sentiment [8, 18, 4]. Despite these advances, TextBlob and VADER remain widely used due to their simplicity and accessibility [10, 16]. More recently, transformerbased models, like BERT [9] and RoBERTa [22], have achieved state-of-theart performance by capturing complex linguistic and contextual dependencies, particularly when fine-tuned to a specific domain [25, 29, 26, 19]. Our study ofers a comparison of TextBlob and VADER with the fine-tuned Twitter-roBERTabase model and human judgments, highlighting their strengths and limitations. In another recent study, Kwon et al. [20] used LLMs to analyze public sentiment on Twitter. They labeled each tweet using seven libraries, including TextBlob and VADER and applied a majority voting approach to label a dataset of 1.26 million tweets. Their evaluation showed that LLMs such as BERT, GPT-2, and Llama-2 outperformed traditional machine learning classifiers.

In contrast to those previously mentioned works, our study takes a finegrained approach by evaluating 100 tweets with three sentiment analysis tools and six independent human raters, rather than relying on pre-labeled datasets or large-scale automated labeling. This design enabled us to directly compare output from tools to human judgments and also capture the diversity and subjectivity inherent in human judgment, particularly for informal social media content. By including multiple raters, we can quantify inter-rater agreement and evaluate how closely each tool aligns with human perception, ofering insights into tool strengths and limitations in practical social media applications.

## 3 Methodology

Our dataset consists of a subset of 100 tweets [17] containing the keyword “Site C, Khayelitsha” posted between January 2022 to December 2024, collected as part of a study to assess residents’ perceptions of safety and security in Khayelitsha Township, South Africa. We then generated the sentiment labels by running the three sentiment analysis tools and three LLMs and also by gathering data from six human raters. Those human raters were Computer Science graduate students from our lab with substantial exposure to social media content and sentiment analysis tasks. Hereafter, the six human raters are referred to as R1–R6; the three sentiment analysis tools are referred to as TB (TextBlob), VD (VADER), and RB (Twitter-RoBERTa-base); and the three LLMs are referred to as QW (Qwen3-32B), GP (GPT-OSS-120B), and LM (Llama-4-Maverick-17B).

To evaluate the agreement between sentiment analysis tools and human perception, we followed four steps. First, we measured pairwise agreement between human raters and tools using Cohen’s kappa, which provides insight into consistency at the individual rater level. We also tested if simplifying sentiment into binary categories (e.g., negative vs. non-negative and positive vs. non-positive) improves agreement. Next, we extended the analysis to include multiple raters simultaneously by applying Fleiss’ kappa. As we noticed that the human ratings also varied, we used a majority voting label between human raters which served as a label for evaluating tool performance. No ties occurred during the majority voting process; therefore, no tie-breaking procedure was required.

## 3.1 Measuring Inter-Rater Agreement: Cohen’s Kappa

Cohen’s kappa coeficient is a statistic that is used to measure inter-rater agreement between two raters for categorical items. The coeficient ranges from -1 to 1, with 1 indicating perfect agreement, 0 if there is no agreement other than what would be expected by chance, and negative values indicating disagreement (agreement worse than chance). Kappa values can be interpreted in the following ranges: -1.00-0.00 (poor agreement), 0.01-0.20 (slight), 0.21-0.40 (fair), 0.41-0.60 (moderate), 0.61-0.80 (substantial), and 0.81-1.00 (almost perfect).

We hypothesized that classifying sentiment as negative vs. non-negative (or even positive vs. non-positive) might yield higher agreement than the traditional positive/negative/neutral categories. This is because humans are generally more adept at identifying distinct positive or negative sentiments, while neutral statements tend to be more ambiguous and prone to subjective interpretation. To test our theory, we calculated the Cohen’s kappa for negative vs. non-negative (Fig. 1b) and positive vs. non-positive (Fig. 1c).

## 3.2 Multi-Rater Agreement: Fleiss’ Kappa

Fleiss’s kappa is a generalization of Cohen’s kappa for more than two raters. In Table 1, we calculated Fleiss’ kappa to assess inter-rater agreement across multiple raters and tools for various sentiment classification tasks: positive/negative/neutral, negative/non-negative, and positive/non-positive.

![](images/f72cfef3852c2c508d91132686f4d682683e07cb20720f47c6bff720eea47210.jpg)  
(a) Positive, Negative, & Neutral

![](images/4fdaeac102f5dee14578f22ebe9c2412cc419de3fe723d348739630f1035d814.jpg)

(b) Negative vs Non-Negative  
![](images/d20a4367a1e6a12c4e9cf9e8f213f93275b58d9d04907d019bdc9d89ea7f1cc8.jpg)  
(c) Positive vs Non-Positive  
Fig. 1: Cohen’s Kappa (a) Positive, Negative, and Neutral, (b) Negative vs Non-Negative, (c) Positive vs Non-Positive. Darker values indicate better agreement.

## 4 Results

We first calculated the Cohen’s kappa for each pair of raters (Fig. 1a). The kappa values between human raters and tools vary, with some raters showing moderate agreement (e.g., R1 and R6 with a kappa of 0.52) and others showing fair agreement (e.g., R1 and R2 with a kappa of 0.28). The sentiment analysis tools (TB, VD, and RB) generally have slight to fair agreement with the human raters (R1-R6), indicating that the tools are less consistent with human raters. Among those tools, RB has the highest agreement with a human rater (kappa value of 0.72 with R1). In contrast, the LLMs (LM, GP, and QW) have higher agreement with human raters than other sentiment analysis tools, generally achieving fair to moderate agreement with human raters (R1-R6). Among LLMs, GP has the highest agreement with a human rater (kappa value of 0.74 with R6). Additionally, agreement among the LLMs themselves was substantial, with kappa value of 0.69 (LM and QW), 0.74 (GP and QW), and 0.80 (LM and GP).

## 4.1 Higher Inter-Rater Agreement Under Binary Sentiment Classification

The kappa values between raters for negative vs. non-negative show significantly higher agreement compared to the three-class sentiment classification in Fig. 1a. For example, in Fig. 1b, human raters R1 and R6 achieve almost perfect agreement $( \mathrm { k a p p a } = 0 . 8 0 )$ . RB exhibits the highest agreement with the raters (for example, RB with R1 having an almost perfect kappa value of 0.86), which is similar to what we saw in the three-class classification, but with noticeably stronger agreement in the binary negative vs. non-negative classification. Similarly, the LLMs show strong alignment with human judgments in this binary setting: GP achieves substantial agreement with R1 and R6 $\mathrm { \Delta ( k a p p a = 0 . 7 8 ) }$ , LM with R1 $( \mathrm { k a p p a } = 0 . 7 4 )$ , and QW with R1 $( \mathrm { { k a p p a } = 0 . 6 5 ) }$ ). TB and VD still show slight and fair agreement. In particular, VD exhibits poor agreement with R2 $( \mathrm { k a p p a = - 0 . 0 1 } )$ ; however, this low value is consistent with R2’s generally weak agreement across multiple tools and raters.

As shown in Fig. 1c, the kappa values for the positive vs. non-positive classification also show a moderate increase in agreement compared to the three-class classification in Fig. 1a, but the values are generally lower than the negative vs. non-negative classification in Fig. 1b. For example, R1 and R6 have slight agreement with a kappa value of 0.34. However, as shown in Fig. 1b, they have almost perfect agreement (kappa value of 0.80) for the negative vs. non-negative classification. RB continues to show better performance in positive vs. non-positive binary classifications (with kappa values like 0.63 with R5). The LLMs showed higher agreement in positive vs. non-positive classification, with substantial to almost perfect agreement in some pairings. TB and VD show weaker agreement in the positive vs. non-positive classification compared to negative vs. non-negative.

In summary, higher agreement was observed under the binary sentiment classification compared to the three-class classification. For human raters and other non-LLM tools, agreement is generally stronger for negative vs. non-negative, suggesting that negative sentiment is more distinct and easier to identify in our dataset. However, the LLMs tend to agree more with humans on positive sentiment.

## 4.2 Human Consensus via Majority Voting Improves Agreement with Automated Tools

As shown in Table 1, the individual human raters had varying interpretations of sentiment, leading to moderate inter-rater agreement at best among themselves and with the sentiment analysis tools. Agreement among LLMs was substantial to near-perfect, whereas agreement between LLMs and human raters remained at most moderate. To address this inconsistency and improve reliability, we applied a majority voting approach, aggregating the most common sentiment label among human raters (Table 2). This allowed us to establish a more stable consensus, which we then used to compare against sentiment analysis tools and LLMs. We noticed that applying majority voting among human raters significantly improved consistency, making their collective judgment more aligned with automated tools.

Table 1: Inter-Rater Agreement (Fleiss’ Kappa). The highest kappa value for each classification is shown in bold and highlighted in red.
<table><tr><td colspan="4">Pos/Neg/Neutral Neg/Non-Neg Pos/Non-Pos</td></tr><tr><td>All Humans (R1-R6)</td><td>0.39</td><td>0.45</td><td>0.48</td></tr><tr><td>All Tools (TB, VD, RB)</td><td>0.43</td><td>0.49</td><td>0.43</td></tr><tr><td>Humans (R1-R6) with TB</td><td>0.35</td><td>0.41</td><td>0.39</td></tr><tr><td>Humans (R1-R6) with VD</td><td>0.35</td><td>0.42</td><td>0.42</td></tr><tr><td>Humans (R1-R6) with RB</td><td>0.43</td><td>0.51</td><td>0.50</td></tr><tr><td>Humans (R1-R6) with TB, VD, RB</td><td>0.38</td><td>0.45</td><td>0.40</td></tr><tr><td>All LLMs (one-shot)</td><td>0.74</td><td>0.76</td><td>0.86</td></tr><tr><td>Humans with LLaMA (one-shot)</td><td>0.42</td><td>0.49</td><td>0.51</td></tr><tr><td>Humans with Qwen (one-shot)</td><td>0.41</td><td>0.49</td><td>0.51</td></tr><tr><td>Humans with GPT (one-shot)</td><td>0.42</td><td>0.49</td><td>0.52</td></tr><tr><td>Humans with all LLMs (one-shot)</td><td>0.46</td><td>0.54</td><td>0.57</td></tr></table>

Table 2: Human Majority vs. Each Tool / LLM (Cohen’s Kappa). The highest kappa value for each classification is shown in bold and highlighted in red.
<table><tr><td colspan="3">Pos/Neg/Neutral Neg/Non-Neg</td><td rowspan="2">Pos/Non-Pos</td></tr><tr><td>Majority vs. TB</td><td>0.37</td><td>0.41</td></tr><tr><td>Majority vs. VD</td><td>0.40</td><td>0.50</td><td>0.29 0.37</td></tr><tr><td>Majority vs. RB</td><td>0.73</td><td>0.88</td><td>0.61</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Majority vs. LLaMA (one-shot) Majority vs. Qwen (one-shot)</td><td>0.63 0.48</td><td>0.71 0.60</td><td>0.71 0.61</td></tr><tr><td>Majority vs. GPT (one-shot)</td><td>0.57</td><td>0.67</td><td>0.67</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

RB shows almost perfect agreement with the majority vote of human raters, particularly in negative/non-negative (0.88). It also shows substantial agreement in positive/negative/neutral (0.73) and positive/non-positive (0.61). This indicates that RB is highly consistent with the human majority, especially when distinguishing between negative and non-negative sentiments, and also performs substantially in classifying positive/negative/neutral and positive/non-positive sentiment. TB and VD show fair/moderate agreement, with VD performing moderately in negative/non-negative (0.50) but still demonstrating lower kappa values compared to RB. TB performs the worst, especially in positive/nonpositive (0.29), showing fair agreement with the majority vote of human raters. LLM-based models also exhibit strong alignment with the human majority vote.

LM achieves substantial agreement and GP shows consistently moderate to substantial agreement. QW demonstrates moderate agreement overall, with comparatively lower performance in the multi-class positive/negative/neutral task. The agreement with the human majority is strongest for the binary sentiment classification for all LLMs. Despite the strong performance of LLMs, the domainspecific RB model remains the most aligned tool with the human majority overall, achieving the highest agreement except in the positive/non-positive setting where LLaMA attains the highest kappa value.

## 5 Limitations and Future Work

Although this study focused on 100 tweets, the deliberately curated dataset enabled independent annotation by six human raters and a detailed evaluation of inter-rater agreement. Expanding this approach to additional topics, geographic regions, and social media communities will help assess the broader applicability of these findings. While the evaluated sentiment analysis tools produce outputs of varying granularity, including continuous polarity scores (e.g., TextBlob and VADER) and class probabilities (e.g., RoBERTa), we converted all outputs into negative, neutral, and positive labels. This decision was made to ensure methodological consistency across all automated systems and human annotators, as the human reference annotations were collected as discrete sentiment categories rather than continuous ratings. We acknowledge that this simplification discards information about sentiment intensity and prediction uncertainty, which may provide additional insight into cases where models or annotators exhibit low confidence or where sentiment is inherently ambiguous. In addition, this study reports point estimates of agreement statistics without confidence intervals or formal significance testing; therefore, diferences between agreement values should be interpreted descriptively rather than as statistically established. Future research could complement categorical agreement analyses with continuous or probability-based comparisons and report confidence intervals or hypothesis tests using larger datasets to better capture sentiment intensity, model uncertainty, and the statistical reliability of agreement estimates.

## 6 Conclusions

We compared three sentiment analysis tools (TextBlob, VADER, and TwitterroBERTa-base model) and three LLMs (Qwen3-32B, GPT-OSS-120B, and Llama-4-Maverick-17B) on social media text against human judgments.

Higher agreement under binary sentiment classification: Higher agreement was observed under the negative vs. non-negative and positive vs. nonpositive classification than under the three-class classification among human raters and between humans and automated tools. For sentiment tools and humans, negative vs. non-negative classification showed the strongest agreement, suggesting that negative sentiment is more distinct and easier to identify than positive sentiment. Interestingly, LLMs performed better on positive vs. nonpositive classifications, achieving higher alignment with human judgments in detecting positive sentiment.

Twitter-RoBERTa-base emerges as the most reliable tool in matching human sentiment classifications: It consistently showed substantial/almost perfect agreement with the majority vote across all sentiment categories. This shows that carefully trained, domain-specific models can sometimes outperform even recent general-purpose LLMs.

LLMs also show strong alignment with human sentiment: LLMs exhibited substantial agreement among themselves and moderate to substantial agreement with the majority vote by humans. LLMs also performed better in binary sentiment classification, with positive vs. non-positive yielding relatively higher agreement. However, general-purpose LLMs, while powerful, do not automatically outperform classical sentiment analysis tools in social media text as they lack task-specific training on social media sentiment.

TextBlob and VADER show weaker performance: Both TextBlob and VADER exhibit less agreement, particularly in positive/non-positive, indicating they might struggle with identifying positive sentiments as clearly as negative or non-negative ones. TextBlob showed the lowest agreement across all categories.

Human-centered evaluation is important for reliable sentiment analysis: Our results showed how even human raters still disagree, highlighting the dificulty of sentiment analysis of social media text. Multi-rater evaluation is essential for establishing a reliable gold standard. Although our dataset includes only 100 tweets, a well-curated, small-scale dataset allows for detailed analysis and multiple human ratings per tweet. Importantly, automated tools or LLMs should not be used to generate ground truth, as they may not always align well with human perception; human-verified labels remain the benchmark for evaluating sentiment.

Twitter-RoBERTa-base and the LLMs outperform classical lexicon-based tools for sentiment analysis because they apply transformer-based deep learning to understand the context of words. Twitter-RoBERTa-base, as a domainspecific model fine-tuned on TweetEval, is more efective at capturing sentiment than TextBlob and VADER, which rely on fixed rules and lexicons. In contrast, general-purpose LLMs are trained on large corpora but are not specifically adapted to social media text. While LLMs capture nuanced sentiment well, domain-specific fine-tuning could further improve their alignment with human judgments. As these tools continue to advance, our findings are proof positive that the future of sentiment analysis lies in models that truly understand the complexity of human emotions in text and that domain adaptation still remains crucial for reliable sentiment analysis.

Acknowledgments. This project was supported in part through the Minerva Research Initiative, in partnership with the Air Force Ofice of Scientific Research (AFOSR) under grant number FA9550-22-1-0297. We would also like to thank David Calano, Lesley Frew, Kritika Garg, and Tarannum Zaki for their contributions to the evaluation dataset.

## References

1. Albladi, A., Islam, M., Seals, C.: Sentiment analysis of Twitter data using NLP models: a comprehensive review. IEEE Access (2025)

2. Aljedaani, W., Rustam, F., Mkaouer, M.W., Ghallab, A., Rupapara, V., Washington, P.B., Lee, $\mathrm { E . , }$ Ashraf, I.: Sentiment analysis on Twitter data integrating TextBlob and deep learning models: The case of us airline industry. Knowledge-Based Systems 255, 109780 (2022)

3. Barbieri, F., Camacho-Collados, J., Espinosa Anke, $\mathrm { L . , }$ Neves, L.: TweetEval: Unified benchmark and comparative evaluation for tweet classification. In: Findings of the Association for Computational Linguistics: EMNLP 2020. pp. 1644–1650. Association for Computational Linguistics, Online (Nov 2020). https://doi.org/10.18653/v1/2020.findings-emnlp.148

4. Bello, A., Ng, S.C., Leung, M.F.: A BERT framework to sentiment analysis of tweets. Sensors 23(1) (2023). https://doi.org/10.3390/s23010506

5. Camacho-Collados, J., Rezaee, K., Riahi, T., Ushio, A., Loureiro, D., Antypas, D., Boisson, J., Espinosa-Anke, L., Liu, F., Martínez-Cámara, E., et al.: TweetNLP: Cutting-Edge Natural Language Processing for Social Media. In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, Abu Dhabi, U.A.E. (Nov 2022)

6. Çılgın, C., Baş, M., Bilgehan, H., Ünal, C.: Twitter sentiment analysis during COVID-19 outbreak with VADER. AJIT-e: Academic Journal of Information Technology 13(49), 72–89 (2022)

7. Cohen, J.: A coeficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37–46 (1960). https://doi.org/10.1177/001316446002000104

8. Dahal, K.R., Gupta, A., Budhathoki, N.: Comparative analysis of VADER and TextBlob on financial news headlines. Journal of Data Science pp. 1–20 (2025)

9. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). pp. 4171–4186 (2019)

10. Doan, M.L.: Sentiment trend analysis of SpaceX tweets using time-series sentiment classification with TextBlob algorithm. Journal of Digital Society 1(1), 44–67 (2025)

11. Fleiss, J.: Measuring nominal scale agreement among many raters. Psychological Bulletin 76, 378– (11 1971). https://doi.org/10.1037/h0031619

12. Hugging Face: gpt-oss-120b. https://huggingface.co/openai/gpt-oss-120b (2025)

13. Hugging Face: llama-4-maverick-17b-128e-instruct. https://huggingface.co/ meta-llama/Llama-4-Maverick-17B-128E-Instruct (2025)

14. Hugging Face: Qwen3-32b. https://huggingface.co/Qwen/Qwen3-32B (2025)

15. Hutto, C., Gilbert, E.: VADER: A parsimonious rule-based model for sentiment analysis of social media text. In: Proceedings of the international AAAI conference on web and social media. vol. 8, pp. 216–225 (2014)

16. Irfan, M., Sattar, A., Sher, A., Ijaz, M.: Sentiment analysis of public discourse on Pakistan’s political parties: A comparative study using VADER and TextBlob algorithms on Twitter data. Journal of Digital Society 1(2), 152–167 (2025)

17. Jayanetti, Himarsha R.: https://github.com/himarshaj/hard-to-reachenvironments/tree/main/sentiment\_analysis\_tools (2025)

18. Krishna Enduri, M., Sangi, A., Anamalamudi, S., Ramanadham, C., Kallam, Y.R., Yeswanth, P., Sai Reddy, S.K., Gogineni, A.K.: Comparative study on sentimental analysis using machine learning techniques. Mehran University Research Journal of Engineering and Technology 42, 207 (01 2023). https://doi.org/10.22581/muet1982.2301.19

19. Kumar, B., Sadanandam, M.: A fusion architecture of BERT and RoBERTa for enhanced performance of sentiment analysis of social media platforms. International Journal of Computing and Digital Systems 15(1), 51–66 (2024)

20. Kwon, O.H., Vu, K., Bhargava, N., Radaideh, M.I., Cooper, J., Joynt, V., Radaideh, M.I.: Sentiment analysis of the United States public support of nuclear power on social media using large language models. Renewable and Sustainable Energy Reviews 200, 114570 (2024). https://doi.org/10.1016/j.rser.2024.114570

21. Liu, B.: Sentiment analysis and subjectivity. Handbook of Natural Language Processing 2(2010), 627–666 (2010)

22. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V.: RoBERTa: A robustly optimized bert pretraining approach. Tech. Rep. arXiv:1907.11692, arXiv (2019)

23. Loria, S.: TextBlob documentation (2018), {https://textblob.readthedocs.io/ en/dev/}

24. Marrapu, S., Senn, W., Prybutok, V.: Sentiment analysis of Twitter discourse on Omicron vaccination in the USA using VADER and BERT. Journal of Data Science and Intelligent Systems 3(3), 165–175 (2025)

25. Nguyen, D.Q., Vu, T., Nguyen, A.T.: BERTweet: A pre-trained language model for English tweets. Tech. Rep. arXiv:2005.10200, arXiv (2020)

26. Prytula, M.: Fine-tuning BERT, DistilBERT, XLM-RoBERTa and Ukr-RoBERTa models for sentiment analysis of ukrainian language reviews. Machine learning 3(4) (2024)

27. Raisa, J.F., Ulfat, M., Al Mueed, A., Reza, S.S.: A review on Twitter sentiment analysis approaches. In: 2021 international conference on information and communication technology for sustainable development (ICICT4SD). pp. 375–379. IEEE (2021)

28. Rosenthal, S., Farra, N., Nakov, P.: SemEval-2017 task 4: Sentiment analysis in Twitter. In: Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017). pp. 502–518 (2017)

29. Semary, N.A., Ahmed, W., Amin, K., Pławiak, P., Hammad, M.: Improving sentiment classification using a RoBERTa-based hybrid model. Frontiers in human neuroscience 17, 1292010 (2023)

30. Team, Q.: Qwen3 technical report (2025), https://arxiv.org/abs/2505.09388

31. Tian, L., Lai, C., Moore, J.D.: Polarity and Intensity: the two aspects of sentiment analysis. Tech. Rep. arXiv:1807.01466, arXiv (2018)

32. Wang, Y., Guo, J., Yuan, C., Li, B.: Sentiment analysis of Twitter data. Applied Sciences 12(22), 11775 (2022)

33. Yi, J., Nasukawa, T., Bunescu, R., Niblack, W.: Sentiment Analyzer: Extracting sentiments about a given topic using natural language processing techniques. In: Proceedings of the Third IEEE International Conference on Data Mining. pp. 427– 434 (2003). https://doi.org/10.1109/ICDM.2003.1250949