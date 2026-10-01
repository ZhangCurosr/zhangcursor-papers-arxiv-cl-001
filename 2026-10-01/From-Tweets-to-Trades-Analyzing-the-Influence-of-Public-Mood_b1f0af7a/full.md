# From Tweets to Trades: Analyzing the Influence of Public Mood over Stock Market Performance in Turkiye

Ece Elif Adak <sup>a</sup>, Bertaç Şakir Şahin <sup>b</sup> and Şaziye Betül Özateş <sup>a,∗</sup>

<sup>a</sup>Boazici University, 34342 Bebek/Istanbul, Turkiye

<sup>b</sup>Yıldız Technical University, 34220 Esenler/Istanbul, Turkiye

## A R T I C L E I N F O

Keywords:   
Sentiment analysis   
Borsa Istanbul   
BIST100   
BIST30   
Public mood   
Pre-trained language models   
Transformers

## A BS T RA C T

Purpose: This study examines whether domain-specific public mood is associated with stock-market dynamics and whether these relationships vary across communication domains and market conditions. It distinguishes public mood from investor sentiment and investigates whether heterogeneous sources of public communication exhibit diferent relationships with market behaviour.

Design: The study analyses 610,422 posts published by 176 curated X accounts between January 2022 and December 2023, covering Politics and Government, Economy and Finance, and Media and Society. Posts are classified using fine-tuned Turkish transformer models under three domainspecific and one pooled regime. Public mood measures are constructed at daily, weekly, and monthly frequencies and examined alongside BIST100 and BIST30 market measures using correlation, Granger causality, vector autoregression, and impulse response analyses across the full period and selected market conditions.

Findings: Public mood is not associated with the direction of stock-market returns but is associated with the magnitude of price movements, particularly for Media and Society and pooled communication. These relationships become stronger at longer aggregation frequencies. Predictive relationships are concentrated in Economy and Finance communication, while their magnitude and direction vary across market conditions, particularly during the 2023 election period. The pooled measure largely reflects the most active communication domain.

Originality: The study contributes to behavioral-finance research by incorporating communication - domain heterogeneity into the analysis of public mood and market dynamics. It also demonstrates how aggregating heterogeneous sources can obscure domain-specific relationships between public communication and financial markets.

## 1. Introduction

Financial markets face a continuous flow of social-media commentary that rapidly spreads political views, economic forecasts, news and emotional narratives beyond oficial disclosures. Under the eficient market hypothesis, prices should quickly incorporate value-relevant public information, leaving historical linguistic tone with little explanatory power once market data are considered (Fama, 1970), while recent evidence supports semistrong eficiency over longer horizons (Agrrawal and Agarwal, 2025). Behavioural finance, however, suggests that beliefs, attention and nonfundamental optimism or pessimism can influence trading and valuation when uncertainty is high and arbitrage is limited (Baker and Wurgler, 2006).

We therefore distinguish public mood the positive, neutral or negative linguistic valence measured separately in political, economic and social communication from investor sentiment. Since accounts are selected by speaker rather than topic, posts need not originate from investors or concern financial markets; interpreting their tone as investor sentiment would attribute identities and market-specific beliefs that the texts do not establish. Public mood instead captures an observable feature of communication (Dean, 2017), consistent with textual finance research distinguishing public tone from investors’ latent sentiment (Kearney and Liu, 2014; Loughran and McDonald, 2016). This mood may influence markets through attention, information difusion, repeated narratives and interpretations of uncertainty, or reflect beliefs formed elsewhere, making the relationship potentially bidirectional. Evidence from X links online communication to returns, trading volume and volatility (Bollen, Mao and Zeng, 2011; Sul, Dennis and Yuan, 2017).

Turkiye provides a useful setting. Direct participation in Borsa Istanbul rose sharply after 2020. By the end of May 2024 more than 8.2 million investors held Borsa Istanbul equities, with roughly 5.5 million holding BIST100 constituents (of Turkiye, 2024; İstanbul, 2024). A larger investor base increases the importance of the public channels through which economic and political narratives reach market participants. The period from January 2022 to December 2023 combines persistent inflation and exchangerate pressure with several discrete events. The February 2023 earthquakes, the May 2023 presidential and parliamentary elections, and a pronounced wave of initial public oferings in the second half of 2023. These produce economically distinct conditions under which the relationship between public communication and market behaviour may strengthen, weaken or change direction.

Research on textual information and Turkish markets has advanced quickly. These studies establish that usergenerated online text can carry financial information, that the relationship may be bidirectional, and that its intensity varies. What remains open is not whether social-media text relates to market outcomes but which tone of communication is being measured and whether diferent domains of communication behave alike. Current evidence remains disjointed on three interconnected inquiries. First, socialmedia tone is frequently consolidated into a single metric, although political, economic and social discussions may convey distinct information and may trigger various channels of attention and risk awareness. Second, research that accurately gauges investor sentiment often depends on messages created by investors or specific to finance; broader samples from influential public accounts necessitate a framework that does not assume an investor’s identity. Third, temporal variation is usually studied separately from domain composition, leaving limited evidence on whether the direction of information transmission itself changes across normal, crisis and political uncertainty periods. These issues are especially relevant in an emerging market in which political developments, macroeconomic conditions and social events can influence market expectations (Kilimci and Duvar, 2020; Cagli, Can Ergun and Durukan, 2020; Liu, Lee, Huang and Wu, 2023; Liu, Liu and Son, 2026).

To address them, we analyse 610,422 posts published by 176 influential X accounts between January 2022 and December 2023. We group the accounts into Politics and Government, Economy and Finance, and Media and Society. We fine-tune seven transformer-based language models on novel, manually labelled Turkish data for three-class sentiment classification and use the best-performing configuration for each dataset to construct political, economic, social and combined sentiment score calculation. We align these with the BIST100 and BIST30 at daily, weekly and monthly frequencies. We then use Augmented Dickey–Fuller tests, Pearson and Spearman correlations, bidirectional Granger causality tests, vector autoregressions and impulse response functions to examine associations with prices, returns, trading activity and volatility. We repeat the analysis within the 2022 baseline, the 2023 earthquake quarter, the election quarter and the post-election IPO period, and again at the level of individual accounts.

We make four contributions. First, we define the construct under examination clearly. By characterising the textual variable as domain-specific public mood, we separate the observable valence of influential public communication from investor sentiment in the narrower sense used in behavioural finance. Second, we document variation across domains (political, economic and social), moods are not interchangeable. Their strongest associations arise for diferent market outcomes and diferent evaluative criteria. Third, we combine domain variation with bidirectional transmission and changing market conditions, showing that not just the intensity but also the direction the sentiment–market relationship shift when political or disaster-driven uncertainties arise. Fourth, we provide a substantial Turkish-language dataset and a systematic comparison of seven transformer models for three-class sentiment classification, establishing a reproducible framework for further work on public communication and emerging financial markets. We treat the evaluation of language models as a measurement contribution. The financial contribution concerns how diferent types of public tone relate to, and are related to by the market under varying conditions.

## 2. Theoretical Background and Hypotheses

The relevance of public communication to finance can be framed against informational eficiency. In a semi-strong eficient market, publicly available information should be reflected in prices quickly. So historical linguistic tone should add little once market data are taken into account (Agrrawal and Agarwal, 2025; Fama, 1970). Behavioral finance allows a diferent outcome. Where valuation is uncertain and arbitrage is limited; beliefs, attention and swings of optimism or pessimism can afect trading and asset prices even when they do not correspond to fundamentals (Baker and Wurgler, 2006). Noise trader theory gives a specific mechanism. Investors may trade on unclear or poorly grounded signals while rational arbitrageurs act cautiously because mispricing can worsen before it corrects (De Long, Shleifer, Summers and Waldmann, 1990). Public mood can therefore relate to prices and volatility by spreading information and by leading to correlated trading not based on fundamentals.

We use public mood to refer to the positive, neutral or negative sentiment that can be observed in social-media communication from influential sources. Public mood captures positive, neutral or negative tone in influential social media communication, rather than investors’ beliefs. Social media spreads narratives rapidly through networks, although investors difer in access, interpretation and responsiveness. Salience attracts attention (Barber and Odean, 2008), while uneven information absorption (Hong and Stein, 1999) may yield clearer efects over longer windows, without new information. Political mood concerns policy continuity, institutional credibility and regulation; economic mood concerns inflation, rates, currencies, growth and cash flows; social mood may afect trading or volatility more than returns. Elections and policy changes can raise risk premiums, particularly in fragile economies (Pastor and Veronesi, 2013). Political optimism may signal strategic communication or become a contrary indicator when expectations are priced in. Aggregation can therefore obscure distinct or ofsetting efects.

The direction of transmission is also unlikely to run only from communication to the market. Public mood may influence prices by altering attention, expectations and perceived uncertainty. But gains, losses and volatility can reshape subsequent public discussion. Evidence from social media points to a reciprocal relationship between online sentiment and stock prices (Liu et al., 2023), while research on

Borsa Istanbul shows that predictive sequence between sentiment proxies and excess returns emerges only in particular episodes (Cagli et al., 2020). The resulting process is better described as feedback than as a one-way efect. As such, we interpret bidirectional Granger tests as tests of predictive precedence, not as evidence of structural economic causality.

The Adaptive Markets Hypothesis links these arguments by treating market eficiency and investor behaviour as outcomes that evolve with the environment (Lo, 2004). A disaster, an election or a sharp change in market participation can alter the credibility of public messages, the composition of active investors and the decision rules they use. A relationship observed in a stable period may weaken, disappear or reverse when uncertainty dominates. The integrated framework consequently treats the public mood–market relationship as domain-specific, bidirectional and state-dependent, and provides the basis for comparing political, economic, social and combined mood across the baseline, earthquake, election and post-election periods. We test the following hypotheses.

• H1. Domain-specific public mood contains information relevant to Borsa Istanbul market outcomes.

• H2. Political, economic and social public mood exhibit significantly diferent relationships with market returns, trading activity and volatility.

• H3. As textual and market data are aggregated over longer horizons, the relationship between public mood and market outcomes becomes more pronounced.

• H4. Predictive precedence runs in both directions between public mood and market outcomes.

• H5. The magnitude, direction and sign of the relationship vary across normal, disaster-related, politically uncertain and post-election regimes.

## 3. Literature Review

## 3.1. Public mood, social media tone and market outcomes

Sentiment analysis is widely used to examine how public views relate to financial markets, but terminology varies across media tone, textual sentiment, investor opinion, social-media sentiment, and collective mood. These concepts overlap but do not necessarily measure the same underlying construct. Results depend on the information source, classification method, and outcome examined (Kearney and Liu, 2014), while the meaning of positive and negative language is context-dependent and tied to the documents and actors producing it (Loughran and McDonald, 2016). Thus, linguistic valence in public communication should not automatically be interpreted as investor beliefs.

That publicly available language carries market-relevant information is well established. Antweiler and Frank (2004) found by analysing internet stock message boards that online discussion is more informative about volatility than about large return movements. Tetlock (2007) reached a comparable conclusion using financial newspaper content. Media pessimism is followed by downward price pressure and a subsequent reversal, and unusually high or low pessimism is associated with elevated trading volume. These results suggest that textual tone may reflect temporary price pressure, shifts in attention or uncertainty rather than durable changes in fundamental value, and that returns are only one metric through which the market may respond.

The growth of social media extended this work beyond financial journalism and investor forums. Rather than building mood indicators from messages about stocks alone, Bollen et al. (2011) derived them from a broad Twitter stream and found that some dimensions of public mood help predict movements in the Dow Jones Industrial Average while others relate much less. Collective public expression is therefore multidimensional, and not all of its emotional components carry the same financial weight. By contrast, Li, van Dalen and van Rees (2018) examined stock-specific microblogs and found their information content associated with abnormal returns, trading volume and volatility, with these relationships strengthening when messages are weighted by the social influence of the users producing them. Taken together, this evidence indicates that the source and scope of online communication matter. A broad public-mood indicator and opinion expressed within an investment community may both relate to markets without measuring the same phenomenon.

The informational content of social-media language also depends on the wider information environment. Khan, Ghazanfar, Azam, Karami, Alyoubi and Alfakeeh (2022) showed that traditional news coverage and social-media coverage are associated with diferent patterns of turnover and volatility. Ren, Dong, Popovic, Sabnis and Nickerson (2024) showed how social platforms reshape the consumption of traditional media news. Social media may repeat narratives available elsewhere, but repetition can still influence attention and trading. The predictive value of textual tone therefore cannot be assessed without considering who produces the content, whether that producer has a financial focus, and how the information reaches market participants.

Prior research thus provides substantial evidence that public language is associated with returns, trading volume and volatility, and much weaker support for treating every textual indicator as a uniform measure of investor sentiment. Measures constructed from financial forums, firm-specific messages, professional news and broad social-media communication reflect diferent authors and diferent information. Much of this literature also aggregates positive and negative communication into a single market-wide signal even though political, economic and social discourse may difer in credibility, salience and financial relevance. This motivates our stratification of public mood by domain.

## 3.2. Social media analytics and evidence from Borsa Istanbul

Research on social media and Borsa Istanbul has developed along two connected lines: constructing reliable sentiment measures from Turkish-language content, and testing whether those measures inform market outcomes. Early work targeted finance-related messages. Kilimci and Duvar (2020) paired three word-embedding models, Word2Vec, GloVe and FastText, with three deep-learning architectures, convolutional and recurrent neural networks and long shortterm memory networks, to predict the direction of nine actively traded banking stocks using Twitter, financial news sites and public disclosures, and report considerable variation across text sources, indicating that predictive performance depends on the information source and not only on the classification approach. Ateş and Güran (2021) subsequently linked sentiment derived from Turkish financial tweets to daily BIST30 returns using Pearson correlation and Granger causality. Their Granger results are directly relevant here. Over a long sample the predictive sequence runs from returns to sentiment, while some sentiment measures precede returns only within a politically eventful short window covering the June 2018 election. Direction, in other words, was already known not to be fixed on Turkish data.

A second line shows that the financial relevance of social-media content cannot be inferred from its availability alone. Ismayil and Demir (2023), focused on two airlines listed on Borsa Istanbul and separated posts concerning the companies’ shares from posts about their products and services. They found that Twitter activity concerning the shares positively related to stock performance while broader discussion of the companies did not. A relatively small number of financially relevant messages produced correlations comparable to those from the larger set of share-related posts. Which indicates that filtering for market relevance matters more than increasing the volume of text. Cam, Cam, Demirel and Ahmed (2024) concentrated on the classification stage, labelling Turkish posts gathered through Borsa Istanbulrelated hashtags with a multilingual sentiment lexicon. Then compared six supervised classifiers trained on those labels, naive Bayes, logistic regression, support vector machines, �- nearest neighbours, decision trees and a multilayer perceptron, of which the support vector machine and the perceptron performed best. That work advanced the measurement of Turkish financial sentiment, but high classification accuracy does not imply that the resulting indicator will predict returns, volume or volatility. The distinction between successful text classification and successful market prediction is essential when comparing evidence across studies.

Work published in Borsa Istanbul Review has moved beyond positive–negative analysis. Akdogan and Anbar (2024) applied a gated recurrent unit classifier to identify posts concerning Borsa Istanbul, labeled their sentiment with a finetuned Turkish BERT model. They used the resulting social, cognitive and behavioural features in linear and Lasso regressions, random forests and gradient boosting to model the opening level, trading volume and volatility of the BIST100.

Their results showed that social-media information may relate to several dimensions of market activity, while also stressing the need to separate financially relevant content from the broader stream before constructing explanatory variables. Ibrahim, Khan and Kaplan (2025) extended the comparison across information sources by combining local social-media posts with domestic and international news narratives. They found that using sentiment features fed into gradient boosting, XGBoost and random forest models with SHAP values attributing the predictions. International news sources generally played a larger role in forecasting the Turkish stock market than local Twitter content. The informational value of text appears to depend on its source, reach and connection to the economic environment rather than on its origin in social media as such.

Other evidence emphasises volatility and changing market conditions. Bozma and Kul (2020) used sentiment from company-specific tweets within a multivariate GARCH framework and documented relationships between Twitterbased measures and volatility transmission among selected Borsa Istanbul stocks. Sevinç and Coşkun (2026) examined social-media and investor-forum posts concerning stocks identified by the Capital Markets Board in connection with manipulation, using a lexical framework that added a suspicious content category to conventional sentiment classes. The efects of online communication difered across bearish, normal and bullish conditions and across levels of trading activity.

Taken together, the Turkish evidence confirms that social-media information can be associated with market direction, returns, trading activity and volatility, and reveals substantial diferences in textual sources, filtering procedures and evaluation criteria. Most studies rely on finance-specific hashtags, firm-related messages, investor forums or content screened for market relevance. Thus difer conceptually from an indicator constructed from political actors, public institutions, financial commentators, news organisations and other influential accounts. Previous work also generally transforms textual information into an aggregate sentiment measure, even though distinct areas of public communication may carry diferent market-relevant information.

## 3.3. Domain heterogeneity, bidirectional transmission and state dependence

Textual information should not be treated as a single uniform market signal, since the financial importance of language can depend on the topic and on the type of risk or expectation attached to it. Calomiris and Mamaysky (2019) used news from 51 developed and emerging economies to construct topic-specific measures of sentiment, news flow and unusual language. They found that these hold information about future returns, volatility and drawdowns with predictive value difering across countries and horizons. Combining topics into one tone indicator can therefore conceal distinct economic signals.

The relationship between textual indicators and market outcomes may also involve feedback rather than one-way transmission. Liu et al. (2023) reported a generally positive link between social-media investor sentiment and stock prices while noting that this relationship can weaken, vanish or reverse in particular periods. Intraday evidence suggested that sentiment may carry forward-looking information even as price movements remain closely tied to subsequent online sentiment. Liu et al. (2026) extended this to the intraday microstructure. Evidence from Borsa Istanbul points the same way. Cagli et al. (2020) found no overall causality between a composite investor-sentiment index and BIST100 returns using standard full-sample Granger tests. But distinct causality episodes appear once nonlinearities and structural change are assessed with a recursive evolving-window method, with predictive influence running from sentiment to the market, from the market to sentiment, or in both directions depending on the proxy and period. Constantparameter estimates may conceal relationships that exist only during specific episodes.

The predictive role of sentiment also depends on the economic environment. Chung, Hung and Yeh (2012) showed that investor sentiment predicts several stock portfolios during expansions while its predictive ability is generally weak during recessions. This implies that the efects of optimism, misvaluation and limits to arbitrage vary with conditions. The literature thus suggests that textual information may be topic-specific, linked to markets through feedback, and dependent on the prevailing regime. It remains unclear whether political, economic and social public mood exhibit diferent predictive patterns and whether these relationships shift jointly across normal, crisis and politically uncertain periods. We address that gap by examining domain-specific public mood, bidirectional predictive precedence and state dependence within a common empirical framework.

## 4. Methodology

We collect and analyze 600K posts published by 176 curated X accounts between January 2022 and December 2023. The accounts are stratified into three broad domains: Politics and Government, Economy and Finance, and Media and Society. The posts are classified using four fine-tuned Turkish transformer models under three domain-specific training regimes and one pooled regime. To examine the relationship between social media activity and financial market performance, we aggregate the resulting predictions into daily, weekly, and monthly time series and evaluate their relationships with BIST100 and BIST30 performance data using diferent statistical testing approaches. Figure 1 illustrates the overall methodology, and the following subsections describe each step in detail.

## 4.1. Corpus construction

The corpus is selected and constructed by speaker rather than by topic. Turkish studies of social media and Borsa Istanbul have collected text through financial hashtags, cashtags, firm names or disclosure feeds. Under such a filter the sentiment being measured is by construction sentiment about the market. Removing the filter permits a diferent question: whether discourse that is not about the market is still relevant to it. Answering that question requires selection criteria that operate on accounts rather than on posts.

We assigned accounts to categories before collecting any posts. Politics and Government contains the oficial accounts of the Presidency and the ministries, the President of Turkiye, the ministers serving during the study window, the Speaker of the Grand National Assembly, and the leaders of the main opposition parties represented in parliament. Economy and Finance contains the Central Bank of the Republic of Turkiye and its governors during the window, the oficial accounts of Borsa Istanbul and of the Capital Markets Board, the economic ministries, and financial news outlets, economists and market commentators. Media and Society contains general news outlets, journalists, selected public figures, and the oficial accounts of the five largest municipalities. Our assignment is not strictly disjoint, several ministries and ministers whose remit spans economic policy and public services belong to more than one category, so category post counts sum to more than the number of distinct posts.

Two criteria governed our inclusion decisions. Institutional and ofice-holding accounts entered on the basis of their role, irrespective of audience size. All remaining accounts were required to exceed one million followers. The follower condition is a reach condition, a relationship between public discourse and an aggregate index is plausible only for discourse reaching enough market participants to matter. It also has a cost, stated in Section 7.2. It removes the retail conversation, which is where investor sentiment would be found.

The corpus comprises 610,422 distinct posts published by 176 accounts between 1 January 2022 and 31 December 2023. 119,877 posts were published by Politics and Government accounts, 256,135 by Economy and Finance accounts and 297,857 by Media and Society accounts. Posts published on weekends and market holidays are assigned to the next trading day, on the assumption that discourse produced while the exchange is closed can only be acted upon at the next open. This added up to 500 trading days over the two year period.

## 4.2. Annotation

We drew three labelled training sets from the corpus, one per category, each restricted to posts by accounts only in that category, and a fourth pooling all three. Human annotators labelled posts as positive, negative or neutral. We excluded posts too vague in sentiment to assign and removed highly similar posts so that learning achieved by the models were not diminished. Table 1 reports the resulting sizes and class distributions.

We assessed inter-annotator agreement on 150 posts drawn from all three categories and labelled independently by two annotators. Weighted Cohen’s � across the three classes reached 0.817, which falls in the highest band of the conventional interpretation (Landis and Koch, 1977). Two features of the labelled data bear on interpretation. Excluding vague posts removes precisely the cases a classifier finds hardest, so test-set metrics may overstate performance on the full corpus. And the positive class is the majority in all three category sets, especially Politics and Government, where 959 of 1,500 posts are positive and only 81 are neutral.

![](images/cbc4573317193b94868677265464bd63be9f13385938174bf5308ed7537dafff.jpg)  
Figure 1: Methodological outline of the study.

Table 1 Labelled dataset sizes and class distribution.
<table><tr><td>Dataset</td><td>Total</td><td>Train</td><td>Valid.</td><td>Test</td><td>Positive</td><td>Negative</td><td>Neutral</td></tr><tr><td>Politics and Government</td><td>1,500</td><td>1,000</td><td>200</td><td>300</td><td>959</td><td>460</td><td>81</td></tr><tr><td>Economy and Finance</td><td>1,506</td><td>1,006</td><td>200</td><td>300</td><td>705</td><td>652</td><td>149</td></tr><tr><td>Media and Society</td><td>1,502</td><td>1,002</td><td>200</td><td>300</td><td>764</td><td>599</td><td>139</td></tr><tr><td>Combined</td><td>4,508</td><td>3,008</td><td>600</td><td>900</td><td>2,428</td><td>1,711</td><td>369</td></tr></table>

## 4.3. Sentiment classification

We fine-tuned seven pre-trained transformer models on each of the four labelled datasets for three-class sentiment classification: mBERT (Devlin, Chang, Lee and Toutanova, 2019), BERTurk, DistilBERTurk, BERT5urk (Schweter, 2025), and ConvBERTurk, which applies the ConvBERT architecture of Jiang, Yu, Zhou, Chen, Feng and Yan (2020), TurkishBERTweet (Najafi and Varol, 2024) and Gemma-3- 1B (Team, 2025). The set includes multilingual and Turkishspecific pre-training, encoder-only, encoder-decoder and decoder-only architectures, and parameter counts from 68M to 1.42B. TurkishBERTweet is pre-trained specifically on 894 million Turkish X posts. We selected model configurations by using grid search over the number of epochs, the learning rate, the batch size and the maximum sequence length, giving 54 configurations, each run under three random seeds. This is 162 runs for each pairing of a model with a dataset and 4,536 fine-tuning runs in total. We used weighted $F _ { 1 }$ on the held-out test set as the selection criterion in preference to accuracy because of the class imbalance.

Table 2 shows the performance of the best models for each of the four datasets (see Table A.1 in Appendix for a comparison of all models). We observe that the Politics and Government model is the strongest of the four, consistent with the more explicit sentiment expression of political discourse but also with its heavier class imbalance. When measured against a majority-class baseline the ranking in fact reverses, as Economy and Finance gainied the most accuracy over its own baseline. And the Economy and Finance model is the weakest and least stable across random seeds, with a seed-to-seed standard deviation of 2.6 points of weighted $F _ { 1 }$ , against 1.6 for the next least stable dataset and 1.1 and 1.2 for the other two. Classification noise attenuates an observed relationship rather than manufacturing one. This afects the power of the tests reported below rather than their validity.

## 4.4. Constructing the public mood series

We label each post with the model selected for its category, and then with the Combined model, which we apply to the entire corpus. This produces four labelling regimes: three domain-specific and one pooled. We analyse the regimes in parallel rather than ensembling them into a single series. Because whether they agree is itself an empirical question. Post-level labels are mapped to {+1, 0, −1} and the public mood score for regime � on day � is

$$
S _ { j , t } = \frac { N _ { j , t } ^ { p o s } - N _ { j , t } ^ { n e g } } { N _ { j , t } } ,\tag{1}
$$

where $N _ { j , t } ^ { p o s }$ and $\boldsymbol { N } _ { i t } ^ { n e g }$ are the numbers of positive and negative posts assigned to day � under regime � and $N _ { j , t }$ is the total number of posts assigned to that day. Neutral posts do not move the score as its factor is 0 but still enter the denominator, and the score is bounded in [−1, 1]. Equivalently, $S _ { j , t }$ is the mean of the mapped post-level labels for that day. Weekly and monthly series average over Monday-to-Friday windows and calendar months, giving 500 daily, 104 weekly and 24 monthly observations. The score measures polarity and carries no information about post volume.

Table 2 Best single run of the selected classification model for each labelling regime.
<table><tr><td>Labelling regime</td><td>Selected model</td><td>Accuracy</td><td>Weighted  $F _ { 1 }$ </td><td>Macro  $\overline { { F _ { 1 } } }$ </td></tr><tr><td>Politics and Government</td><td>BERTurk</td><td>93.0</td><td>92.9</td><td>86.9</td></tr><tr><td>Economy and Finance</td><td>ConvBERTurk</td><td>86.0</td><td>85.6</td><td>79.7</td></tr><tr><td>Media and Society</td><td>BERTurk</td><td>86.3</td><td>86.6</td><td>80.4</td></tr><tr><td>Combined</td><td>ConvBERTurk</td><td>88.6</td><td>88.1</td><td>81.7</td></tr></table>

Notes: All values are percentages on the held-out test set. Figures are the best of three seeded runs and each sits about one seed standard deviation above the corresponding seed average, which is 91.7, 82.8, 84.9 and 87.2 weighted $F _ { 1 }$ respectively. Macro $F _ { 1 }$ falls 5.9 to 6.4 points below weighted $F _ { 1 }$ in all four cases because the neutral class is the smallest in every dataset. Turkish-specific models outperform multilingual pre-training on every dataset.

## 4.5. Market data and sub-periods

We obtained open, close, high, low and volume series for the BIST100 and BIST30 from Yahoo Finance, and the oficial realised volatility series at 21, 42, 63, 126 and 252- day horizons from Borsa Istanbul. We analyse the two indices in parallel throughout. BIST30 is a subset of BIST100, so agreement between them is a consistency check rather than independent evidence. Disagreement, however, is informative, and reporting both makes the large-capitalisation composition of any result visible.

Derived variables are the daily return, the log return, the intraday change (close relative to open), the overnight gap (open relative to previous close), the intraday range in index points and as a percentage of the open, and rolling standard deviations of the daily return at 5, 10 and 21- day windows. The distinction between the range, which measures the magnitude of a day’s movement, and the return, which measures its direction, is central to the results.

We partition the window into four sub-periods on the basis of events identified from the macroeconomic and political record rather than from inspection of the results. 2022 as baseline (252 trading days) with persistent inflation and lira depreciation but no discrete shock, 2023 Q1 (61 days) containing the 6 February Kahramanmaraş earthquakes and the subsequent suspension of trading, 2023 Q2 (59 days) containing the May 2023 elections and the beginning of the reversal from heterodox to orthodox monetary policy, and 2023 H2 (128 days) combining post-election monetary tightening with a pronounced wave of initial public oferings.

## 4.6. Empirical strategy

We tested all series for a unit root using the Augmented Dickey–Fuller regression

$$
\Delta y _ { t } = \alpha + \gamma y _ { t - 1 } + \sum _ { i = 1 } ^ { k } \delta _ { i } \Delta y _ { t - i } + \epsilon _ { t } ,\tag{2}
$$

with � selected by the Akaike information criterion and the null $H _ { 0 } ~ : ~ \gamma ~ = ~ 0$ of a unit root. Testing precedes estimation and also governs interpretation. We report correlations between two non-stationary series where the analysis produced them, but mark them and do not treat them as evidence. Because two series trending over the same window will correlate whether or not they are related. The BIST100 closed 2021 at 1,857.65 points and 2023 at 7,470.18, so this is not a hypothetical concern.

We compute Pearson � and Spearman $\rho$ between each mood series and each market variable, at all three frequencies, for both indices, over the full period and within each sub-period. We report Spearman alongside Pearson because it is robust to the non-normality of daily financial series. One property of the volatility variables should be recorded here: the rolling standard deviations and the oficial BIST series are computed over overlapping windows, so consecutive daily values of a 21-day series share twenty of their twentyone observations. Where both series entering a correlation are serially dependent, the standard error is understated and the associated �-value is optimistic. We therefore read the volatility correlations as weaker than their nominal levels suggest, and no claim below rests on a volatility correlation alone.

We estimate bivariate vector autoregressions for each mood–market pair,

$$
Y _ { t } = c + \sum _ { i = 1 } ^ { p } A _ { i } Y _ { t - i } + \epsilon _ { t } ,\tag{3}
$$

with $Y _ { t } = \left( S _ { j , t } , m _ { t } \right) ^ { \prime }$ , lag order selected by AIC, and nonstationary components first-diferenced before estimation. We test Granger causality in both directions at lags of 1, 2, 3 and 5 trading days using the standard � test on restricted and unrestricted residual sums of squares. Impulse response functions trace the propagation of a one-standard-deviation innovation. Residual diagnostics are satisfactory throughout. Durbin–Watson statistics of the 56 fitted daily equations run from 1.983 to 2.024, and AIC selects lag orders of five or six for the daily systems.

We interpret these tests narrowly, and deliberately so. A rejection of the null establishes that past values of one series improve the forecast of the other beyond that series’ own history. It does not establish that one series moves the other. Two features of our design make the weaker reading the only defensible one. A substantial share of Economy and Finance posts are institutional announcements of events the market is simultaneously repricing, and a three-to-five day lag is at least as consistent with gradual information difusion as with sentiment transmission. Both are taken up in Section 6.

Table 3 Augmented Dickey–Fuller tests, daily series, full period.
<table><tr><td rowspan="2">Series</td><td colspan="3">BIST100</td><td colspan="3">BIST30</td></tr><tr><td>ADF</td><td>p</td><td>Class</td><td>ADF</td><td>p</td><td>Class</td></tr><tr><td>Open</td><td>-0.635</td><td>0.8629</td><td>NS</td><td>-0.632</td><td>0.8635</td><td>NS</td></tr><tr><td>Close</td><td>-0.719</td><td>0.8417</td><td>NS</td><td>-0.711</td><td>0.8438</td><td>NS</td></tr><tr><td>Volume</td><td>-2.670</td><td>0.0794</td><td>NS</td><td>-3.430</td><td>0.0100</td><td>S</td></tr><tr><td>Return</td><td>-9.755</td><td>0.0000</td><td>S</td><td>-10.000</td><td>0.0000</td><td>S</td></tr><tr><td>Intraday change %</td><td>-10.033</td><td>0.0000</td><td>S</td><td>-10.220</td><td>0.0000</td><td>S</td></tr><tr><td>Range</td><td>-3.098</td><td>0.0267</td><td>S</td><td>-2.510</td><td>0.1130</td><td>NS</td></tr><tr><td>Range %</td><td>-7.932</td><td>0.0000</td><td>S</td><td>-8.130</td><td>0.0000</td><td>S</td></tr><tr><td>Volatility 5d</td><td>-6.979</td><td>0.0000</td><td>S</td><td>-6.938</td><td>0.0000</td><td>S</td></tr><tr><td>Official vol. 21d</td><td>-3.467</td><td>0.0089</td><td>S</td><td>-3.576</td><td>0.0062</td><td>S</td></tr><tr><td>Official vol. 42d</td><td>-2.539</td><td>0.1063</td><td>NS</td><td>-2.526</td><td>0.1092</td><td>NS</td></tr><tr><td>Official vol. 252d</td><td>-0.613</td><td>0.8680</td><td>NS</td><td>-1.213</td><td>0.6679</td><td>NS</td></tr></table>

Notes: S = stationary, NS = non-stationary at the 5% level. The null is a unit root; augmenting lag order selected by AIC. The log return and the 63 and 126-day oficial series behave as their neighbours in the table and are omitted for space.

Therefore we report results as predictive precedence and association throughout, never as efect or impact.

Our design requires a large number of tests. Counting the mood score alone, which is the measure carried into the causality analyses, and pooling the two indices, there are 576 full-period and 1,728 sub-period correlations and 896 full-period and 2,240 sub-period Granger tests, a total of 5,440. At the 0.05 level, 780 are significant against 272 expected under a global null. The tests are not independent, since the four regimes label the same corpus, the two indices share constituents, and lags of the same variable pair are nested. We apply no formal correction. Instead, we give weight to findings that replicate across labelling regimes, across the two indices and across frequencies, and withhold it from isolated results. This is a criterion for reading the evidence rather than a correction to the individual �-values. The counts above should be borne in mind wherever a single significant test is reported.

Finally, we apply the correlation analysis at the level of individual accounts. An account entered this analysis where its posting provided suficient trading-day coverage to support the test, which 169 of the 176 accounts satisfied. We report account-level results in aggregate and do not identify any individual account.

## 5. Empirical Results

## 5.1. Stationarity

Table 3 reports the Augmented Dickey–Fuller tests for both indices. Price levels and the oficial volatility series at horizons of 42 days and longer contain unit roots on both indices. The returns, the intraday change, the range as a percentage of the open, short-horizon calculated volatility and the oficial 21-day series are stationary on both.

Two variables are classified diferently across the two indices, and both matter for what follows. The intraday range is stationary on the BIST100 $( p = 0 . 0 2 6 7 )$ but not on the BIST30 $( p = 0 . 1 1 3 0 )$ . Trading volume is the reverse $( p =$ 0.0794 and $p = 0 . 0 1 0 0 )$ . Both are borderline cases rather than clear rejections, and the range is borderline within the sample as well. It’s non-stationary in the 2022 sub-period and stationary in all three 2023 sub-periods. Therefore we read the range results below as robust on the BIST100 and provisional on the BIST30, and the volume results the other way round.

## 5.2. Contemporaneous association over the full period

Table 4 reports daily correlations between each mood series and the market variables for both indices.

There is no association between public mood and the direction of the daily return. The largest of the eight return coeficients across the two indices is 0.047 and none are significant. The same holds for the intraday change and for the overnight gap. Whatever these accounts are doing, they are not moving with the sign of the daily price change.

The largest coeficients in the table sit on the closing level and on the 252-day oficial volatility. Both series carry unit roots on both indices. A coeficient of 0.347 between Combined mood and the closing level of an index that quadrupled over the sample indicates that two series trended together. We report these rows because the analysis produced them and omitting them would misrepresent the output, but they are not evidence of a relationship.

What survives the stationarity filter is the magnitude of the daily move. Media and Society mood correlates with the intraday range at $r = 0 . 2 5 7$ on the BIST100 and $r =$ 0.289 on the BIST30, and Combined at $r ~ = ~ 0 . 2 1 6$ and $r = 0 . 2 3 9$ , all at $p < 0 . 0 0 1$ . The 21-day calculated volatility, which is stationary on both indices, behaves identically at $r ~ = ~ 0 . 1 7 0$ and $r ~ = ~ 0 . 2 1 4$ for Media and Society. Days on which the index travelled through a wide intraday band were days on which Media and Society and Combined mood ran higher. Rank coeficients exceed the linear ones, so the association is monotonic rather than linear. As noted above, the range series itself is borderline stationary. It fails the test on the BIST30, so the 21-day volatility results carry the more secure version of this finding.

Table 4 Daily correlations between public mood and market variables, full period.
<table><tr><td rowspan="2">Variable</td><td colspan="4">BIST100</td><td colspan="4">BIST30</td></tr><tr><td>Pol.</td><td>Econ.</td><td>Media</td><td> ${ \overline { { \mathsf { C o m b . } } } }$ </td><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td></tr><tr><td>Close</td><td>0.093*</td><td>0.085†</td><td> $\overline { { 0 . 3 2 8 ^ { * * * } } }$ </td><td> $\overline { { 0 . 3 4 7 ^ { * * * } } }$ </td><td>0.098*</td><td>0.076t</td><td> $\overline { { 0 . 3 3 8 ^ { * * * } } }$ </td><td>0.355* ***</td></tr><tr><td>Volume§</td><td>-0.095*</td><td>-0.027</td><td> $0 . 2 0 5 ^ { * * * }$ </td><td>0.079t</td><td>-0.081†</td><td>-0.033</td><td> $0 . 1 3 9 ^ { * * }$ </td><td>0.038</td></tr><tr><td>Return</td><td>-0.007</td><td>0.004</td><td>0.044</td><td>0.047</td><td>-0.014</td><td>-0.005</td><td> $0 . 0 4 0$ </td><td>0.045</td></tr><tr><td>Intraday change %</td><td>0.010</td><td>-0.020</td><td>0.016</td><td>0.020</td><td>0.005</td><td>-0.029</td><td> $0 . 0 1 9$ </td><td>0.023</td></tr><tr><td> $\mathsf { R a n g e } ^ { \ S }$ </td><td>0.029</td><td>-0.010</td><td> $0 . 2 5 7 ^ { * * * }$ </td><td> $0 . 2 1 6 ^ { * * * }$ </td><td>0.039</td><td>-0.024</td><td> $0 . 2 8 9 ^ { * * * }$ </td><td> $0 . 2 3 9 ^ { * * * }$ </td></tr><tr><td> $R a n g e \%$ </td><td>-0.024</td><td>-0.088*</td><td> $0 . 1 2 0 ^ { * * }$ </td><td>0.065</td><td>-0.022</td><td>-0.102*</td><td> $0 . 1 2 8 ^ { * * }$ </td><td>0.064</td></tr><tr><td>Volatility 21d (rolling)</td><td>0.109*</td><td>-0.018</td><td> $0 . 1 7 0 ^ { * * * }$ </td><td> $0 . 1 6 7 ^ { * * * }$ </td><td>0.106*</td><td>-0.045</td><td> $0 . 2 1 4 ^ { * * * }$ </td><td> $0 . 1 9 6 ^ { * * * }$ </td></tr><tr><td>Official vol. 252d‡</td><td>0.164***</td><td>-0.011 0.354***</td><td></td><td>0.354***</td><td> $0 . 1 8 1 ^ { * * * }$ </td><td></td><td>-0.031 0.439***</td><td>0.422***</td></tr></table>

Notes: Pearson � between the daily public mood score of each labelling regime and the market variable named. Pol. = Politics and Government, Econ. = Economy and Finance, Media = Media and Society, Comb. = Combined. $n = 5 0 0$ on the BIST100 and 499 on the BIST30 fo the close, volume, intraday change and range; 499 and 498 for the return and the oficial 252-day series; and 479 and 478 for the 21-day rolling measure. $^ { \dag } p < 0 . 1 0 , ^ { \ast } p < 0 . 0 5 , ^ { \ast \ast } p < 0 . 0 1 , ^ { \ast \ast \ast } p < 0 . 0 0 1 .$ . <sup>‡</sup> marks a series that is non-stationary on both indices; <sup>§</sup> marks one whose classification difers between them.

Table 5 Public mood and market correlations across aggregation frequencies.
<table><tr><td rowspan="2">Variable</td><td colspan="4">BIST100</td><td colspan="4">BIST30</td></tr><tr><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td></tr><tr><td>Range, daily</td><td>0.029</td><td>-0.0100.257</td><td>**</td><td> $\overline { { 0 . 2 1 6 } } ^ { * * * }$ </td><td>0.039</td><td>-0.024 0.289</td><td>***</td><td> $0 . 2 3 9 ^ { * * * }$ </td></tr><tr><td>Range, weekly</td><td>0.087</td><td>0.1280.363</td><td>***</td><td> $0 . 3 7 2 ^ { * * * }$ </td><td>0.098</td><td>0.100</td><td> $0 . 4 0 8 ^ { * * * }$ </td><td> $0 . 4 0 2 ^ { * * * }$ </td></tr><tr><td>Range, monthly</td><td>0.214</td><td>0.214</td><td> $0 . 4 8 2 ^ { * }$ </td><td> $0 . 5 2 3 ^ { * * }$ </td><td>0.233</td><td>0.162</td><td> $0 . 5 3 1 ^ { * * }$ </td><td> $0 . 5 5 8 ^ { * * }$ </td></tr><tr><td>Return, daily</td><td>-0.007</td><td>0.004</td><td>0.044</td><td></td><td>0.047-0.014</td><td>-0.005</td><td>0.040</td><td>0.045</td></tr><tr><td>Return, weekly</td><td>0.160</td><td>-0.046</td><td>0.108</td><td>0.097</td><td>0.173†</td><td>0.002</td><td>0.097</td><td>0.111</td></tr><tr><td>Return, monthly</td><td> $0 . 0 6 0 \ - 0 . 4 6 8 ^ { * }$ </td><td></td><td>0.263</td><td>0.055</td><td></td><td>0.061 -0.402†</td><td>0.240</td><td>0.059</td></tr></table>

Notes: Pearson $r .$ Daily $n = 5 0 0$ (BIST100) and 499 (BIST30), 499 and 498 for the return; weekly $n = 1 0 4 ;$ monthly $n = 2 4$ . Significance and abbreviations as in Table 4. Monthly coeficients rest on 24 observations and most monthly market variables fail the stationarity test; they are reported for completeness only.

The Economy and Finance regime is the exception and runs the other way. Its coeficients against every volatility measure are negative on both indices. Among the variables shown, the only one reaching significance is the range as a percentage of the open, at −0.088 on the BIST100 and −0.102 on the BIST30. The change in trading volume, not tabulated, is the only other Economy and Finance coeficient to reach the 0.05 level, at −0.114 and −0.092. Of the four regimes, the one composed of accounts that talk about the economy is the one whose mood moves least, and most negatively, with same-day market activity.

## 5.3. Frequency aggregation

Table 5 repeats the range and return correlations at weekly and monthly aggregation.

The range relationship strengthens monotonically with aggregation. On the BIST100 the Media and Society coeficient rises from 0.257 daily to 0.363 weekly, and Combined from 0.216 to 0.372. The pattern repeats with BIST30. The return relationship does not strengthen. It remains insignificant at every frequency on both indices, where the largest coeficient is 0.173 at the weekly frequency and significant only at the 0.10 level.

The asymmetry is informative. If the daily null on returns reflected insuficient power, aggregation should improve it yet it does not. If public mood relates to market activity through attention and the gradual difusion of narratives rather than through the pricing of new fundamental information, accumulation over longer windows should strengthen a magnitude relationship while leaving a direction relationship absent. Thus, H3 is supported for magnitude and rejected for direction.

## 5.4. Predictive precedence

Table 6 reports the number of significant Granger tests by direction, regime, index and frequency.

Predictive precedence is concentrated by domain. Economy and Finance accounts for 11 of the 28 daily mood-tomarket tests on the BIST100 and nine of 28 on the BIST30, where chance would place roughly 1.4. These include the daily return at lags of three $( F = 5 . 0 4 , p = 0 . 0 0 1 9 )$ and five $( F = 3 . 0 6 , p = 0 . 0 0 9 8 )$ trading days on the BIST100, replicated on the BIST30 at three $( p \ : = \ : 0 . 0 0 5 4 )$ and five $( p = 0 . 0 1 6 7 )$ , together with the closing level, the intraday change and trading volume. The result survives a change of frequency, seven of 28 weekly tests are significant on the BIST100 and four of 28 on the BIST30. No other regime produces more than two significant mood-to-market tests at either frequency on either index. H4 is supported for this direction, and H2 is supported in the specific sense that the domain determines which market property sentiment relates to.

The market-to-mood direction is thinner over the full period but contains the single strongest test in the analysis. BIST100 daily range predicts Combined mood one day ahead at $F = 1 4 . 1 8 \ : ( p = 0 . 0 0 0 2 )$ with the same relationship appearing in Economy and Finance at $F = 1 0 . 5 3$ and in Media and Society at $F ~ = ~ 4 . 6 7$ . Mood also predicts the range at lag one in those three regimes, which makes the two jointly determined at that horizon rather than one leading the other. This bidirectional lag-one pattern is specific to the BIST100. On the BIST30 the range predicts Economy and Finance mood but mood does not predict the range in any regime.

Table 6 Significant Granger causality tests, full period.
<table><tr><td rowspan="2">Index</td><td rowspan="2">Direction</td><td colspan="4">Daily</td><td colspan="4">Weekly</td></tr><tr><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td></tr><tr><td>BIST100</td><td>Mood → Market</td><td>1/28</td><td>11/28</td><td>1/28</td><td>2/28</td><td>0/28</td><td>7/28</td><td>0/28</td><td>0/28</td></tr><tr><td>BIST100</td><td>Market → Mood</td><td>1/28</td><td>5/28</td><td>2/28</td><td>3/28 0/28</td><td></td><td>0/28</td><td>1/28</td><td>1/28</td></tr><tr><td>BIST30</td><td>Mood → Market 0/28</td><td></td><td>9/28</td><td>0/28</td><td>0/28 1/28</td><td></td><td>4/28</td><td>0/28</td><td>0/28</td></tr><tr><td>BIST30</td><td>Market → Mood 1/28</td><td></td><td>3/28</td><td>0/28</td><td>0/28 0/28</td><td></td><td>0/28</td><td>1/28</td><td>1/28</td></tr></table>

Notes: Counts of tests significant at the 0.05 level out of the tests run, across seven market variables and four lags (1, 2, 3 and 5 periods). Abbreviations as in Table 4.

Table 7 Significant daily Granger causality tests by subperiod and direction.
<table><tr><td>Sub-period</td><td>Tests per direction</td><td>Mood → Market</td><td>Market → Mood</td></tr><tr><td>Panel A: BIST100</td><td></td><td></td><td></td></tr><tr><td>2022 baseline</td><td>112</td><td>6</td><td>0</td></tr><tr><td>2023 Q1 earthquake</td><td>112</td><td>4</td><td>7</td></tr><tr><td>2023 Q2 election</td><td>112</td><td>4</td><td>43</td></tr><tr><td>2023 H2 IPO boom</td><td>112</td><td>3</td><td>10</td></tr><tr><td>Panel B: BIST30</td><td></td><td></td><td></td></tr><tr><td>2022 baseline</td><td>112</td><td>5</td><td>1</td></tr><tr><td>2023 Q1 earthquake</td><td>112</td><td>3</td><td>5</td></tr><tr><td>2023 Q2 election</td><td>112</td><td>7</td><td>36</td></tr><tr><td>2023 H2 IPO boom</td><td>112</td><td>0</td><td>7</td></tr></table>

Notes: Counts of tests significant at the 0.05 level, pooled across the four labelling regimes, seven market variables and four lags.

The lag structure is worth stating plainly. Market-tomood relationships appear at one day lag onward. Moodto-market relationships appear at three and five days lag. Under the straightforward mechanism in which an influential account posts, investors read it, and the price moves at the next open, a lag-one result is the first thing that should appear. It appears at neither frequency and is absent from the contemporaneous correlations. Impulse responses put the magnitudes in perspective. On BIST100 the largest marketto-mood response is 2.6% of a mood standard deviation on the day following a one-standard-deviation market innovation.

## 5.5. State dependence

Table 7 reports the sub-period Granger counts by direction.

In 2022, across 112 daily tests on BIST100, no market variable precedes public mood under any regime. In the election quarter of 2023 the market precedes mood in 43 of 112 tests while mood precedes the market in four. The BIST30 reproduces the reversal at 36 and 7. All four labelling regimes flip together. Politics and Government, which produces a single significant mood-to-market result in the full-period analysis, produces 14 market-to-mood results within that one quarter on the BIST100, covering the opening and closing levels, the return, the intraday change, the range and the volume at lags of two to five days.

Table 8 shows the same instability in the contemporaneous coeficients.

The sign reverses between 2022 and the election quarter on both indices at once. Combined mood correlates with the daily return at � = 0.177 $( p = 0 . 0 0 5 )$ in 2022 on the BIST100 and $r ~ = ~ 0 . 1 6 7$ on the BIST30. In the election quarter the same coeficients are −0.281 and −0.247. Media and Society mood correlates with the range at 0.432 in 2022 and at essentially zero in the election quarter. All four regimes return negative correlations with trading volume in the election quarter, significant in three of four on both indices, with Combined reaching −0.483 and −0.500. All four also return a negative and significant correlation with the oficial 21-day volatility in that quarter on the BIST100, at −0.259 in Economy and Finance to −0.530 in Combined, which is the only period in the entire correlation analysis where all four regimes agree in sign and significance. A higher mood in the election quarter went together with a calmer and thinner market. This supports H5.

The earthquake quarter produces some of the smallest correlation coeficients of the four sub-periods. It is worth stating because the literature on sentiment and crisis would predict the opposite. Three of the twelve BIST100 cells in Table 8 reach significance on 61 observations, and each carries the opposite sign to the full-period coeficient of the same regime on the same variable.

## 5.6. Regime disagreement

The four labelling regimes do not measure the same quantity, and the Combined regime is not an average of the other three. Its mean daily mood score of 0.016 sits against a Politics and Government mean of 0.510, a Media and Society mean of −0.002, an Economy and Finance mean of −0.093. The unweighted three-category mean is 0.138. The same ordering holds within all four sub-periods. In every one, the Combined series falls within 0.023 of the largest contributing category and never closer than 0.094 to the average of the three. Media and Society covers 297,857 of the 610,422 posts, and the pooled series reproduces it.

Table 8 Daily correlations by sub-period.
<table><tr><td rowspan="2">Variable</td><td colspan="4">BIST100</td><td colspan="4">BIST30</td></tr><tr><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td><td>Pol.</td><td>Econ.</td><td>Media</td><td>Comb.</td></tr><tr><td>Daily return</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2022 baseline</td><td>0.054</td><td>0.100</td><td>0.105†</td><td>0.177**</td><td>0.048</td><td>0.098</td><td>0.096</td><td>0.167**</td></tr><tr><td>2023 Q1 earthquake</td><td>0.070</td><td>0.013</td><td>-0.045</td><td>0.067</td><td>0.055</td><td>-0.003</td><td>-0.046</td><td>0.064</td></tr><tr><td>2023 Q2 election</td><td>-0.228t</td><td>-0.164</td><td>-0.095</td><td>-0.281*</td><td>-0.191</td><td>-0.178</td><td>-0.061</td><td>-0.247†</td></tr><tr><td>2023 H2 IPO boom</td><td>-0.019</td><td>-0.035</td><td>0.149†</td><td>0.118</td><td>-0.056</td><td>-0.056</td><td>0.121</td><td>0.088</td></tr><tr><td>Intraday range</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2022 baseline</td><td>0.003</td><td>-0.124*</td><td>0.432***</td><td>0.370***</td><td>0.017</td><td>-0.134*</td><td> $0 . 4 5 0 ^ { * * * }$ </td><td> $0 . 3 8 9 ^ { * * * }$ </td></tr><tr><td>2023 Q1 earthquake -0.259*</td><td></td><td>0.007</td><td>-0.039</td><td>-0.102</td><td>-0.260*</td><td>0.000</td><td>-0.044</td><td>-0.110</td></tr><tr><td>2023 Q2 election</td><td>0.233†</td><td>-0.178</td><td>-0.001</td><td>0.001</td><td>0.216</td><td>-0.178</td><td>0.002</td><td>-0.029</td></tr><tr><td>2023 H2 IPO boom</td><td>-0.080</td><td>-0.026</td><td>-0.000</td><td>-0.156†</td><td>-0.077</td><td>-0.047</td><td>0.049</td><td>-0.137</td></tr><tr><td>Trading volume</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2022 baseline</td><td>0.051</td><td></td><td>0.018 0.425***</td><td> ${ 0 . 4 4 1 } ^ { * * * }$ </td><td>0.079</td><td></td><td> $0 . 0 1 8 0 . 3 0 2 ^ { * * * }$ </td><td> $0 . 3 3 6 ^ { * * * }$ </td></tr><tr><td>2023 Q1 earthquake</td><td>-0.143</td><td> $0 . 3 5 6 ^ { * * } \ - 0 . 2 4 0 ^ { \dag }$ </td><td></td><td>-0.142</td><td>-0.159</td><td></td><td> $0 . 3 5 6 ^ { * * } \ - 0 . 2 2 9 ^ { \dag }$ </td><td>-0.148</td></tr><tr><td>2023 Q2 election</td><td>-0.239t -0.362**</td><td></td><td>-0.285*</td><td>-0.483* ***</td><td>-0.253t -0.335**</td><td></td><td>-0.304*</td><td>-0.500*</td></tr><tr><td>2023 H2 IPO boom</td><td>0.004-0.231</td><td></td><td>0.500 ***</td><td>0.112</td><td>-0.061-0.261</td><td></td><td>0.476 ***</td><td>0.064</td></tr></table>

Notes: Pearson �. Observations: 2022 baseline 252 (251 for the return), earthquake 61, election 59, IPO boom 128. Significance and abbreviations as in Table 4.

Table 9 Accounts correlating significantly with selected market variables, by category.
<table><tr><td rowspan="2">Category</td><td colspan="5"></td><td colspan="3">BIST30</td></tr><tr><td>Accounts</td><td>Return</td><td></td><td>Range %</td><td>Vol 21d</td><td>Return</td><td>Range %</td><td>Vol 21d</td></tr><tr><td>Politics and Government</td><td>70</td><td>1 (1.4)</td><td></td><td>3 (4.3)</td><td>17 (24.3)</td><td>1 (1.4)</td><td>5 (7.1)</td><td>15 (21.4)</td></tr><tr><td>Economy and Finance</td><td>67</td><td>11 (16.4)</td><td></td><td>12 (17.9)</td><td>8 (11.9)</td><td>11 (16.4)</td><td>13 (19.4)</td><td>8 (11.9)</td></tr><tr><td>Media and Society</td><td>64</td><td>3 (4.7)</td><td></td><td>7 (10.9)</td><td>15 (23.4)</td><td>3 (4.7)</td><td>6 (9.4)</td><td>14 (21.9)</td></tr><tr><td>All accounts pooled</td><td>169</td><td>12 (7.1)</td><td></td><td>22 (13.0)</td><td>33 (19.5)</td><td>11 (6.5)</td><td>21 (12.4)</td><td>33 (19.5)</td></tr></table>

Notes: Number of individual accounts whose daily mood series correlates with the named variable at the 0.05 level, with the percentage of the category in parentheses. Chance alone would place roughly 5% in each column. The first three rows use the category model on the accounts of that category; the final row applies the Combined model to all accounts with suficient coverage and is therefore not the sum of the three, which in any case exceeds 169 because the category assignment is not disjoint. No individual account is identified.

The impact is visible throughout the tables above. Combined tracks Media and Society on the range and volatility results and departs from it elsewhere, across the thirteen daily market variables the two agree in sign on twelve. The Economy and Finance predictive results have a much weaker counterpart in the Combined series, which produces two significant daily mood-to-market tests against eleven in the regime it draws from. And the Politics and Government mood score, which is positive on all 500 trading days, has a sign that never varies. So the aggregate series cannot register the shifts in valence that its individual constituents display.

## 5.7. Account-level analysis

Aggregating a category into one daily series assumes that its members behave alike. Table 9 reports the correlations we repeated account by account.

Disaggregation reproduces the aggregate picture on the return and reverses it on volatility. Eleven of the 67 Economy and Finance accounts (16.4%) correlate significantly with the daily return on both indices, against one of 70 in Politics and Government and three of 64 in Media and Society. The first figure is well above chance and the second well below it. The signs of those eleven are split six positive to five negative. So the category result is a matter of individual accounts relating to the index rather than of the category moving with it in a common direction.

On volatility the ordering reverses. Seventeen Politics and Government accounts (24.3%) correlate significantly with the oficial 21-day volatility series, twelve of them positively. It is the largest share of any category and more than twice the 11.9% recorded by Economy and Finance. BIST30 gives 21.4% and 11.9%. The largest single accountlevel correlation with the daily return found anywhere in the analysis (|�| = 0.305) also belongs to a Politics and Government account, although on the trading-day coverage available to that account it stops just short of significance (� = 0.0625).

Individual political accounts do relate to Borsa Istanbul, and more often than the accounts of the other two categories. What the category series loses, it loses in the aggregation. A few dominant accounts drowning out the rest would be the obvious explanation. But Politics and Government is the least concentrated of the three categories, its top decile of accounts holding 28.9% of the category’s posts against 44.0% in Economy and Finance and 54.8% in Media and Society. Institutional register runs evenly through the whole account list, so averaging over it removes the variation that the individual accounts carry.

## 6. Discussion

## 6.1. What each domain measures

The results support H2 in a specific form. The three domains do not difer in the strength of a common relationship. They difer in which property of the market they attach to. Media and Society mood, and the Combined series that inherits its properties, relates to the magnitude of daily price movement. Economy and Finance mood relates to almost nothing contemporaneously and carries the predictive precedence, at lags of three days and longer. Politics and Government mood, in aggregate, relates to little at the category level beyond the volatility series, while its individual members relate to volatility more often than those of any other category.

This ordering follows from the composition of the account sets. Media and Society is the group least constrained by ofice and broadest in subject matter. As such its aggregate tone comes closest to a general public mood. Its full-period mean of −0.002 is the behaviour of a series responding to events in both directions. Economy and Finance sits nearest the flow of financial and policy information, its mean of −0.093 remains negative in all four sub-periods. So accounts covering the Turkish economy described it in negative terms across two years in which the index quadrupled. Politics and Government is constrained by ofice to communicate positively, with a mean of 0.510 and no negative day in 500. Three diferent quantities are being measured under one label, and the domain-specific design is what makes them separable.

## 6.2. Magnitude without direction

Our most consistent result is also the one that constrains interpretation most tightly, and it is what H1 predicted in its semi-strong form. Publicly available tone carries no information about the direction of the next price change, only about how far the index travelled within the day. Two features point away from information pricing, towards attention. The relationship strengthens monotonically with aggregation while the return relationship does not. The daily null on returns is not a power problem. And the rank coeficients exceed the linear ones throughout, so extreme days matter more than a linear measure suggests. This is consistent with the attention and gradual-difusion arguments in Section 2, salient content is more likely to enter the decisions of investors who cannot process all available signals (Barber and Odean, 2008), and information reaching investors at diferent speeds produces efects that accumulate over horizons rather than appearing at once (Hong and Stein, 1999). Under that reading, a wide-range day and elevated public discussion are both expressions of a day on which something happened.

## 6.3. The direction and horizon of transmission

Both directions appear, and the lag structure separating them is the strongest internal evidence against a simple sentiment-transmission story. Three to five trading days is a long interval for a mood to travel but a comfortable one for information to difuse outward from an announcement through a market in which a growing share of participants were new. That direction is not fixed, this was already visible in the Turkish evidence. Ateş and Güran (2021) found returns preceding sentiment over their long sample and sentiment preceding returns only inside a politically eventful window.

The Economy and Finance predictive results are the strongest in the study and also the most exposed. That category contains the Central Bank, the Capital Markets Board, Borsa Istanbul and the economic ministries, so a large share of its posts are announcements rather than opinions. When a rate decision, an inflation print, a disclosure or a regulatory measure enters the mood score as a positively or negatively toned post on the day it appears, the market reprices over the following days. Granger test can separate that sequence from mood carrying information. This is not a hypothetical omitted variable but a documented property of the account list, and it is why we report the finding as predictive precedence rather than as an efect.

One feature cuts the other way. The Economy and Finance labels come from the model with the largest seedto-seed variance of the four. Classification noise attenuates a relationship rather than manufacturing one, so a noisier series producing a stronger result is weaker evidence in favor of the relationship. The finding rests nonetheless on the least stable measurement in the study.

## 6.4. State dependence

H5 is the hypothesis the data support most strongly. It is supported on both indices simultaneously and under all four regimes. The pattern is what the Adaptive Markets Hypothesis anticipates (Lo, 2004), a relationship observed under one set of conditions need not persist when the composition of participants, the credibility of public messages and the decision rules in use change together. It is also what the politicaluncertainty literature would predict. When the resolution of political uncertainty is itself the dominant national news, commentary that follows prices rather than preceding them is close to what should be expected (Pastor and Veronesi, 2013). The corpus registers the same shift internally. Politics and Government mood rises from 0.482 in 2022 to 0.637 in the election quarter, the largest sub-period movement of any regime, while Economy and Finance falls to −0.121. The divergence between political messaging and economic commentary peaks where the macroeconomic record would put it.

The earthquake quarter requires separate comment because it produces the least of the four sub-periods, the opposite of what the crisis-sentiment literature would predict. Three explanations are available and they are not seperate. Trading was suspended for part of the quarter, and the missing days immediately after 6 February are precisely where a leading relationship would have to appear. The posts of these accounts during a humanitarian catastrophe are about the catastrophe, so for several weeks the series may not be measuring economic mood at all. And 61 observations is a thin basis for a lag-five test. The first two explanations are the more consequential, because both imply that the measure stops capturing the same construct during exactly the events where it is expected to matter most.

## 6.5. Aggregation as a measurement problem

The comparison across labelling regimes yields a result that generalises beyond this dataset. A sentiment score computed over all posts in a day does not represent the corpus. It represents whichever category posted most. The Politics and Government series shows the same issue more starkly, averaging 70 accounts constrained by ofice to communicate positively produces a series whose sign never changes in 500 trading days. So whatever valence its members express individually is compressed into a narrow positive band around a mean of 0.510. The account-level results in Table 9 confirm that this is a property of the method rather than of the accounts. The individual members of that category relate to volatility more often than those of any other.

The weakness of the aggregate Politics and Government results is therefore a limitation of the aggregation and not evidence that political speech is unrelated to Borsa Istanbul. The artefact is available to any study that scores an institutionally constrained account set daily and then averages, nothing in the output of a daily mean announces its presence.

## 6.6. Alternative explanations

For attributing market movements to a single driver, Turkiye’s January 2022-December 2023 period is unsuitable: it combines recurring lira depreciation, inflation above 80%, a policy-rate cut to 8.5% followed by post-May 2023 tightening, the February 2023 Kahramanmaraş earthquakes, and an IPO surge that altered the investor base.

Each has a documented impact on the index. Changepoint analysis of Turkish equity volatility relates shifts to central bank decisions, domestic political incidents and global shocks (Altinbas, 2025). Over a longer sample, inflation and inflation uncertainty are both found to raise BIST100 returns, consistent with Turkish equities being held as an inflation hedge (Esen, Yildirim and Akyurt, 2025). Elections are known to produce unusual returns on Borsa Istanbul accumulating from roughly a month before the vote to two weeks after it, with volatility running one and a half to two times its pre-election average (Kayaçetin, 2023). An event study of the 2023 election in particular reports strongly positive cumulative abnormal returns around the second round (Bash and Al-Awadhi, 2023). For the earthquake, a Bayesian structural time series analysis using global indices as controls estimates a substantial negative efect on the BIST100 against its counterfactual (Khan, Cifuentes-Faura and Shahbaz, 2024). Mechanisms unrelated to any account’s posts explain the level path of the index, which is the light in which the non-stationary coeficients of Section 5 should be read.

Three confounds afect the significant results. The magnitude relationship may reflect common causes: major events move markets and prompt related posting, which correlation cannot disentangle. The Economy and Finance predictive results may reflect the announcement channel, while the election-quarter reversal coincides with the election, a policy transition, and the start of rate hikes within only 59 trading days. Finally, the polarity-based measure omits posting volume, which other markets suggest can strongly predict volatility (Hamraoui and Boubaker, 2022).

## 7. Conclusion

We examined whether the tone of communication by influential Turkish public accounts has a measurable relationship with Borsa Istanbul. We analysed 610,422 posts published by 176 curated accounts between January 2022 and December 2023, labelled them with four fine-tuned Turkish transformer models under three domain-specific regimes and one pooled regime, then tested the resulting series against the BIST100 and BIST30 over 500 trading days.

Five findings hold. Public mood is not associated with the direction of returns under any regime, on either index or at any frequency. It is associated instead with the magnitude of daily price movement, and that association strengthens with aggregation in a manner consistent with attention and gradual difusion rather than with the pricing of new information. Predictive precedence runs in both directions and is concentrated by domain, mood-to-market at three and five days and market-to-mood at one. The relationship reverses in sign and direction during the 2023 election quarter under all four regimes and on both indices. And the four labelling regimes do not agree, and the pooled series reproduce whichever category contributes the most posts.

## 7.1. Implications

For regulators and exchange operators, the state dependence result is most relevant. Domain-specific mood indicators appear more useful for identifying episodes when public discourse and market activity become tightly coupled than for predicting market direction. Here, coupling was strongest around a national election, and the market-to-mood direction dominated. This suggests that online commentary in such episodes may reflect price and volatility rather than independently drive instability.

For market participants, the absence of an association with returns applies only to this corpus and construction. Speaker-based sampling and daily averaging discard financial-content selection, posting volume, and within-day timing. The results therefore do not rule out discourse-return relationships; rather, observed associations concern movement magnitude rather than direction, with signs varying across the two-year window.

For researchers, the aggregation result is most transferable. Pooling heterogeneous speaker groups can produce a series whose variance and findings are dominated by the largest group. Category-level and account-level analyses should therefore precede interpretation of pooled measures.

## 7.2. Limitations

Five limitations bound our results. We measure public mood rather than investor sentiment, and the one-millionfollower threshold ensures reach but not investor relevance. The two-year window includes major shocks—a currency crisis, inflation above 80%, a monetary-regime change, a catastrophic earthquake, and a national election—and the results show that relationships were not stable within it, limiting extrapolation. Sub-periods contain only 59-128 trading days, adequate for correlations but limited for five-lag Granger tests; stationarity classifications also difer across periods, making the specifications non-identical. The Economy and Finance model is the weakest and least stable across seeds, yet underlies the strongest predictive result. We conduct many tests without formal multiple-comparison correction, so isolated significant findings warrant caution. Our main robustness criterion is replication across regimes, indices, and frequencies.

## 7.3. Further research

Three extensions follow. Adding posting volume alongside polarity would separate salience from tone and test whether volume better predicts market magnitude. Timevarying or recursive-window causality over a longer sample could assess whether the election-quarter market-to-mood coupling recurs in similar episodes. Finally, replication on Turkish investor forums or the same accounts during a calmer period would help establish which findings are robust beyond this dataset.

## References

Agrrawal, P., Agarwal, R., 2025. A longer-term evaluation of information releases by influential market agents and the semi-strong market eficiency. Journal of Behavioral Finance 26, 20–45.

Akdogan, Y., Anbar, A., 2024. More than just sentiment: Using social, cognitive, and behavioral information of social media to predict stock markets with artificial intelligence and big data. Borsa İstanbul Review 24, 61–82.

Altinbas, H., 2025. Volatility in the turkish stock market: An analysis of influential events. Journal of Asset Management 26, 1–14.

Antweiler, W., Frank, M., 2004. Is all that talk just noise? the information content of internet stock message boards. The Journal of Finance 59, 1259–1294.

Ateş, E., Güran, A., 2021. Pearson correlation and granger causality analysis of twitter sentiments and the daily changes in bist30 index returns. Journal of the Faculty of Engineering and Architecture of Gazi University 36, 1687–1701.

Baker, M., Wurgler, J., 2006. Investor sentiment and the cross-section of stock returns. The Journal of Finance 61, 1645–1680.

Barber, B., Odean, T., 2008. All that glitters: The efect of attention and news on the buying behavior of individual and institutional investors. The Review of Financial Studies 21, 785–818.

Bash, A., Al-Awadhi, A., 2023. Presidential elections and stock market outcomes: An event-study on the efect of turkey’s presidential elections on borsa İstanbul. Cogent Economics & Finance 11, 2265659.

Bollen, J., Mao, H., Zeng, X., 2011. Twitter mood predicts the stock market. Journal of Computational Science 2, 1–8.

Bozma, G., Kul, S., 2020. Twitter ile hisse senetleri oynaklığı tahmin edilebilir mi? Sosyoekonomi 28, 315–326.

Cagli, E., Can Ergun, Z., Durukan, M., 2020. The causal linkages between investor sentiment and excess returns on borsa istanbul. Borsa Istanbul Review 20, 214–223.

Calomiris, C., Mamaysky, H., 2019. How news and its context drive risk and returns around the world. Journal of Financial Economics 133, 299–336.

Cam, H., Cam, A., Demirel, U., Ahmed, S., 2024. Sentiment analysis of financial twitter posts on twitter with the machine learning classifiers. Heliyon 10, e23784.

Chung, S.L., Hung, C.H., Yeh, C.Y., 2012. When does investor sentiment predict stock returns? Journal of Empirical Finance 19, 217–240.

De Long, J., Shleifer, A., Summers, L., Waldmann, R., 1990. Noise trader risk in financial markets. Journal of Political Economy 98, 703–738.

Dean, M., 2017. Political acclamation, social media and the public mood. European Journal of Social Theory 20, 417–434.

Devlin, J., Chang, M.W., Lee, K., Toutanova, K., 2019. Bert: Pretraining of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1, pp. 4171–4186.

Esen, O., Yildirim, D., Akyurt, E., 2025. How inflation and its uncertainty afect stock returns: Insights from borsa istanbul. Trends in Business and Economics 39, 117–128.

Fama, E., 1970. Eficient capital markets: A review of theory and empirical work. The Journal of Finance 25, 383–417.

Hamraoui, I., Boubaker, A., 2022. Impact of twitter sentiment on stock price returns. Social Network Analysis and Mining 12, 28.

Hong, H., Stein, J., 1999. A unified theory of underreaction, momentum trading, and overreaction in asset markets. The Journal of Finance 54, 2143–2184.

Ibrahim, M., Khan, A.u.I., Kaplan, M., 2025. From headlines to stock trends: Natural language processing and explainable artificial intelligence approach to predicting türkiye’s financial pulse. Borsa İstanbul Review 25, 1152–1165.

Ismayil, J., Demir, O., 2023. Relationship between twitter activity and stock performance: Evidence from turkish airline industry. Foresight 25, 701– 715.

Jiang, Z.H., Yu, W., Zhou, D., Chen, Y., Feng, J., Yan, S., 2020. Convbert: Improving bert with span-based dynamic convolution, in: Advances in Neural Information Processing Systems 33, pp. 12837–12848.

Kayaçetin, N., 2023. Elections and stock market returns: Evidence from borsa İstanbul. Journal of Research in Business 8, 20–40.

Kearney, C., Liu, S., 2014. Textual sentiment in finance: A survey of methods and models. International Review of Financial Analysis 33, 171–185.

Khan, K., Cifuentes-Faura, J., Shahbaz, M., 2024. Do earthquakes shake the stock market? causal inferences from turkey’s earthquake. Financial Innovation 10, 145.

Khan, W., Ghazanfar, M., Azam, M., Karami, A., Alyoubi, K., Alfakeeh, A., 2022. Stock market prediction using machine learning classifiers and social media, news. Journal of Ambient Intelligence and Humanized Computing 13, 3433–3456.

Kilimci, Z., Duvar, R., 2020. An eficient word embedding and deep learning based model to forecast the direction of stock exchange market using twitter and financial news sites: A case of istanbul stock exchange (bist 100). IEEE Access 8, 188186–188198.

Landis, J., Koch, G., 1977. The measurement of observer agreement for categorical data. Biometrics 33, 159–174.

Li, T., van Dalen, J., van Rees, P., 2018. More than just noise? examining the information content of stock microblogs on financial markets. Journal of Information Technology 33, 50–69.

Liu, Q., Lee, W.S., Huang, M., Wu, Q., 2023. Synergy between stock prices and investor sentiment in social media. Borsa İstanbul Review 23, 76– 92.

Liu, Q., Liu, Y., Son, H., 2026. From emotion to action: How intraday investor sentiment drives market microstructure. Borsa İstanbul Review 26, 100775.

Lo, A., 2004. The adaptive markets hypothesis: Market eficiency from an evolutionary perspective. The Journal of Portfolio Management 30, 15–29.

Loughran, T., McDonald, B., 2016. Textual analysis in accounting and finance: A survey. Journal of Accounting Research 54, 1187–1230.

Najafi, A., Varol, O., 2024. Turkishbertweet: Fast and reliable large language model for social media analysis. Expert Systems with Applications 255, 124737.

Pastor, L., Veronesi, P., 2013. Political uncertainty and risk premia. Journal of Financial Economics 110, 520–545.

Ren, J., Dong, H., Popovic, A., Sabnis, G., Nickerson, J., 2024. Digital platforms in the news industry: How social media platforms impact traditional media news viewership. European Journal of Information Systems 33, 1–18.

Schweter, S., 2025. Berturk v2: Bert, distilbert, convbert and bert5urk models for turkish. https://doi.org/10.5281/zenodo.14963493.

Sevinç, D., Coşkun, M., 2026. The impact of social media and internet forums posts on the stock market dynamics: A sentiment analysis using a lexical approach in borsa İstanbul. Applied Economics 58, 3490–3502.

Sul, H., Dennis, A., Yuan, L., 2017. Trading on twitter: Using social media sentiment to predict stock returns. Decision Sciences 48, 454–488.

Team, G., 2025. Gemma 3 technical report. arXiv:2503.19786 [cs.CL].

Tetlock, P., 2007. Giving content to investor sentiment: The role of media in the stock market. The Journal of Finance 62, 1139–1168.

of Turkiye, C.M.B., 2024. Monthly statistical bulletin, May 2024. Technical Report. Capital Markets Board of Turkiye. Ankara.

İstanbul, M.K., 2024. Investor statistics. Technical Report. Merkezi Kayıt İstanbul. Istanbul.

Table A.1 Best single run of each of the seven fine-tuned models on the Politics and Government dataset, sorted by weighted F1. All values are percentages.
<table><tr><td>Model</td><td>Accuracy</td><td>Weighted F1</td><td>Precision</td><td>Recall</td></tr><tr><td>BERTurk</td><td>93.0</td><td>92.9</td><td>93.0</td><td>93.0</td></tr><tr><td>ConvBERTurk</td><td>92.3</td><td>92.1</td><td>92.5</td><td>92.3</td></tr><tr><td>DistilBERTurk</td><td>89.0</td><td>88.7</td><td>89.0</td><td>89.0</td></tr><tr><td>TurkishBERTweet</td><td>88.7</td><td>88.3</td><td>88.4</td><td>88.7</td></tr><tr><td>mBERT</td><td>87.0</td><td>86.8</td><td>87.0</td><td>87.0</td></tr><tr><td>Gemma-3-1B</td><td>87.3</td><td>86.6</td><td>87.0</td><td>87.3</td></tr><tr><td>BERT5urk</td><td>83.7</td><td>80.8</td><td>78.1</td><td>83.7</td></tr></table>

## A. Performance of the Seven Fine-tuned Models on the Politics and Government Dataset

Table A.1 demonstrates the performance of the seven diferent fine-tuned models on the Politics and Government dataset.