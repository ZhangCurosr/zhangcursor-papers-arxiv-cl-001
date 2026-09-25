# Agentic Detection of Online Conspiracies

Lior Biton Oren Tsur

Department of Computer and Information Science Ben-Gurion University of the Negev, Israel bitol@post.bgu.ac.il orentsur@bgu.ac.il

## Abstract

Conspiratorial discourse on social media is not always expressed through explicit claims or stable lexical markers. The same surface content may express endorsement, legitimate concerns, criticism, satire, or mockery. The main challenge is therefore not only recognizing conspiracy-related claims, but inferring the speaker’s intent – the utterance’s illocutionary force. We argue that this can be achieved through the use of relevant social contexts and propose an agentic framework, equipped with a set of tools supporting social queries.

We demonstrate the benefits of our approach on a unique dataset of Hebrew tweets, covering 80%–90% of the public Hebrew tweets published over a four-year span (late 2018– early 2023), encompassing several election cycles as well as the COVID pandemic years and related vaccination campaigns. This extensive coverage can be used in recovering different social contexts. Evaluating our framework on a manually-annotated adversarial dataset, we find that contextaware workflows consistently outperform text-only classification and that the agentic framework performs significantly better than other frameworks and settings, including a nonagentic model exposed to the same contexts available to the agent. We further provide an analysis of the results, the errors and efficiency (token economy) tradeoffs.

These findings support viewing the task of conspiracy detection as a socially embedded interpretation task, in which effective classification depends not only on access to contexts, but also on adaptive reasoning in which the agent uses tools on a per-case basis, asking only for evidence relevant to its current reasoning step.

## 1 Introduction

Social media has become a central arena for the circulation of misinformation, rumors, and conspiracy theories (Grinberg et al. 2019; Muhammed and Mathew 2022). Commonly referred to as alternative narratives (Starbird 2017), they are spread and consumed through an ecosystem that combines influencers with various motivations,<sup>1</sup> committed communities, and the participation of ordinary users (Starbird, Arif, and Wilson 2019). These narratives tend to gain traction beyond fringe communities, especially in times of declining trust in institutions: increased political polarization, social unrest, economic or public-health crises, and wars – times in which these ‘bespoke realities’ are ever more dangerous (Douglas and Sutton 2023; Allen, Watts, and Rand 2024; DiResta 2024; Hilberts et al. 2025). Support-for and participation-in political violence were found to be correlated with beliefs in conspiracy theories, especially those held by small homogeneous groups (Enders, Klofstad, and Uscinski 2024). The analysis of the patterns of conspiratorial discourse is of great interest to academic scholars, policymakers, and the online platforms serving as the main vehicle for the phenomena. However, the detection of conspiratorial discourse is theoretically and algorithmically challenging (Mahl, Schafer, and Zeng 2023). Conspiratorial dis-¨ course is not well defined. The same surface-level text may serve different communicative acts, ranging from false narratives to legitimate concerns<sup>2</sup>, and mockery.<sup>3</sup> The challenge is more pronounced in short and informal communication styles prevalent in some online social platforms, and among members of cohesive communities using in-group jargon. Moreover, challenging a dogma with alternative narratives can prove true,<sup>4</sup> exposing bad actors or promoting a just cause (Jackson, Bailey, and Welles 2020).

In this work we follow the working definition provided by Douglas and Sutton (2023): “A conspiracy theory is a belief that two or more actors have coordinated in secret to achieve an outcome and that their conspiracy is of public interest but not public knowledge.” Conspiracy theories are (a) oppositional; (b) describe malevolent acts; (c) ascribe agency to small groups; (d) epistemically risky; and (e) shared in order to form new social realities.

In this work we propose an agentic approach for the detection of conspiratorial discourse. We first demonstrate the shortcomings of text-based approaches, then introduce our agentic framework: given a specific social media post, the agent can call tools to sample more content from the user, inspect the user’s metadata, query the user’s social circle, or utilize the structure of the social network.

The shortcomings of text-based classification are illustrated in Table 1 through five examples: both GPT-5 and Gemini 3 Flash agree on the correct prediction in examples 1&5, agree on an incorrect prediction in example 2, and diverge over examples 3–4. Temporal context may be needed when a tweet indirectly refers to a recent event or news cycle. Conversational context may be needed when a tweet responds to, quotes, or mocks another post. Social context may be needed when interpretation depends on the author’s usual stance, community, or repeated narratives. In such cases, the isolated tweet text may be insufficient to determine whether the author is endorsing a conspiracy narrative, criticizing it, joking about it, or merely reporting on it. In many such cases, the key distinction is therefore not whether conspiracy-related content is present, but whether the speaker is committed to it.

<table><tr><td>ID</td><td>Tweet text, English translation</td><td>Gold GPT-5 Gemini 3 Flash</td><td></td><td></td></tr><tr><td>1</td><td>Remember the crop-duster planes I photographed this afternoon?... Here is the result. Dirty and murky skies. This is what we breathe. This is the air pollution that is killing us. And by the way, they blame us for it. So wake up.</td><td>1</td><td>1</td><td>1</td></tr><tr><td>2</td><td>A teaching assistant from Pardes Hanna tells her students during remote learning on “Zoom&quot; that she does not know who killed former Prime Minister Yitzhak Rabin. “It is not certain that Yigal Amir was the murderer, and there are many conspiracy theories surrounding the issue.&quot; The parents heard the lesson and filed a complaint with the principal, then with the Ministry of Education and the Pardes Hanna local council.</td><td>0</td><td>1</td><td>1</td></tr><tr><td>3</td><td>Like if you think the COVID vaccines contained strawberry-vanilla milkshake ingredients and tiny location-tracking chips, so the government could track us, including facial recognition technology, in order to put us all into a secret G5 database. Share this, because the media channels will not tell you the truth.</td><td>0</td><td>1</td><td>0</td></tr><tr><td>4</td><td>I am suffering from severe pain, but my surgery has been postponed. Hospitals in Jerusalem are striking because they lack funding, while politicians dismiss the doctors’ complaints as empty slogans. This government is killing its people. People around the world are suing their governments to force them to fight global warming.</td><td>0</td><td>0</td><td>1</td></tr><tr><td>5</td><td>It seems to us that we Israelis also have a strong case against one of the most indifferent gov- ernments in the West regarding the greatest problem facing our children and grandchildren.</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 1: Examples illustrating context-dependent Hebrew conspiracy-tweet classification. Tweets are shown in English translation, and original Hebrew texts appear in Appendix B. Gold is the human label, where 1 indicates conspiracy endorsement and 0 indicates non-conspiracy. GPT-5 and Gemini 3 Flash are text-only predictions.

We argue that conspiracy-tweet classification should be treated as a contextual interpretation task rather than a purely textual classification task. Relevant context may come from several levels of the social-media environment: tweet-level metadata, the author’s profile, the author’s previous posts, and the interaction network surrounding the user. We treat these signals as contextual hints rather than as direct proof of user intent. For example, consider the following tweet:

## H1 “Those naive people don’t realize the storm is part of Pfizer’s secret plot to take over the world, setting the New World Order. There’s no hope for you.”<sup>5</sup>

The true label of the tweet is non-conspiracy. The multiple, very explicit conspiracy markers are used sarcastically, as the text suggests that a severe winter storm<sup>6</sup> is part of a plot involving the Pfizer vaccine. Table 2 shows how different workflows interpret the tweet. The text-only model mistakes strong conspiracy markers for endorsement, while contextual workflows recover the sarcastic reading using the storm context, the author’s profile, prior weather-related posts, and ego-network evidence. The agents-debate workflow also reaches the correct label, reasoning about the absurdity and the parodic nature of the text.

However, simply appending all available contexts to every tweet, as part of the input, is neither efficient nor necessarily desirable. Providing excessive contexts may introduce noise and harm performance (Shi et al. 2023). Agentic workflows are, therefore, suitable for this setting as an agent can reason and decide what tool should be used next, or whether the data available is sufficient.

We evaluate our agentic approach on a unique and challenging dataset of Hebrew tweets. The agentic approach achieves the best results compared with both the text-only LLM baseline and the corresponding preloaded-context conditions, improving F1 from 0.535 to 0.730 over the text-only LLM. We further experiment with a “Socratic” two-agent setting, in which one agent questions the reasoning of another agent, encouraging critical reflection. Finally, we provide an analysis of the tool-use behavior.

Specifically, we focus on the Israeli X (formerly Twitter) ecosystem, leveraging a unique dataset estimated to cover 80–90% of public Hebrew tweets from December 2018 to June 2023. Low-resource and morphologically rich languages such as Arabic and Hebrew add yet another layer of complexity to the task of algorithmic detection of conspiratorial discourse. The challenge stems not only from ambiguity introduced by the rich morphology, but also from the limited training over the language of a secluded community of speakers. The dataset spans multiple waves of a global pandemic as well as five general-election cycles reflecting intense political tensions, a fertile ground for contentious narratives, conspiratorial thinking, and increased polarization. Together, these social and linguistic conditions make Hebrew conspiracy-tweet classification a particularly challenging setting for automatic detection.

<table><tr><td>Setting</td><td></td><td>Pred. Retrieved context excerpt</td><td>Main agent reasoning</td></tr><tr><td>Text only</td><td>1</td><td>H1. No external context.</td><td>“The tweet explicitly references classic conspiracy tropes, in- cluding a ‘secret agreement&#x27; by Pfizer to take over the world&#x27; and the establishment of a New World Order&#x27;.&quot;</td></tr><tr><td>context</td><td></td><td>Tweet + author 0 Author profile: the user describes her- “The user&#x27;s profile explicitly states they like to talk nonsense,&#x27; self as liking history, singing, and “talk- further supporting the interpretation of the tweet as a joke or ing nonsense.&quot; Tweet metadata: posted on mockery rather than a genuine conspiracy theory.&quot; Dec. 20, 2021.</td><td></td></tr><tr><td>Previous tweets</td><td>0</td><td>Recent posts include “How is the weather“User history confirms they were discussing the weather and outside?&quot; and “I bought pajamas for personal preparations for the storm, indicating the tweet is a</td><td>Carmel,&quot; shortly before the target tweet. joke rather than a genuine conspiracy belief.&quot;</td></tr><tr><td>Network context</td><td>0</td><td></td><td>Ego-network users mention Storm Carmel “Storm Carmel,&#x27; which occurred on the date of the tweet, was on the date of the tweet. The ego network confirmed by ego-context mentions of the storm by @user1 also includes mainstream news or politi- and @user2. [...] The ego-context includes mainstream jour- cal accounts and ordinary Israeli Twitter nalists and regular users, suggesting general Israeli Twitter dis-</td></tr><tr><td>Two-agent debate</td><td>0</td><td>users. sarcasm.</td><td>course where this brand of political sarcasm is common.&quot; No external retrieval. Analyst 2 argues for “Analyst 2&#x27;s argument regarding sarcasm is convincing. The absurdity of linking a natural storm to a Pfizer secret agree- ment&#x27; for a New World Order&#x27; indicates parody.&quot;</td></tr></table>

Table 2: Changes in agent reasoning given different contexts to the same text (H1). Labels follow the binary annotation scheme: 1 indicates conspiracy endorsement and 0 indicates non-conspiracy. Each setting combines the original tweet with only the context specified in that row.

Contribution Evaluating our context-aware agentic retrieval as well as heterogeneous two-agent “debate” framework, the contribution of this paper is threefold: (i) We demonstrate that the use of various contexts significantly improves the detection of conspiratorial posts, (ii) We explore the patterns of efficient tool-usage, and (iii) We analyze the economical gains facilitated by the agentic framework.

## 2 Related Work

Conspiracy Theories and Online Communities The World Economic Forum’s Global Risks Report 2026 (World Economic Forum 2026) lists misinformation and disinformation among the “most severe global risks”.

False narratives are often emotionally engaging and more appealing than accurate information (Vosoughi, Roy, and Aral 2018), making them an effective tool for garnering attention and promoting political agendas (Starbird 2017; Starbird, Arif, and Wilson 2019). Conspiracy theories are a prominent form of this broader information disorder, explaining complex societal, political, medical, or global events as the result of secret plots by powerful actors. The appeal of conspiratorial claims is associated with the needs for certainty, control, identity, and belonging (Douglas, Sutton, and Cichocka 2017; Douglas et al. 2019). Examples include claims that vaccines cause autism or that climate change is a deliberate fraud (Del Vicario et al. 2016). In Israel, a 2015 survey found that 19% questioned the identity of Prime Minister Yitzhak Rabin’s assassin, while 35% questioned some part of the official account (Caspit 2015).

Online platforms serve as the vehicle through which various narratives are shared and reinforced. Existing network structures are harnessed and new communities are created and maintained by committed individuals. These communities provide a sense of belonging, promoting bespoke interpretations of reality (Del Vicario et al. 2016; Douglas, Sutton, and Cichocka 2017).

This “asymmetry of passion” (DiResta 2024) allows small but highly committed online groups to amplify fringe narratives, creating an illusion of consensus and achieving offline impact (Vosoughi, Roy, and Aral 2018; Starbird, Arif, and Wilson 2019).

Conspiracy Detection The body of computational work explicitly addressing the detection of conspiracy theories is surprisingly sparse. This subsection provides a brief survey of the literature as well as relevant works concerned with related tasks such as stance detection and the classification of ‘fake news’ and misinformation.

Treating conspiracies as narrative structures, rather than true/false statements, Shahsavari et al. (2020) combine semantic role labeling, BERT embeddings, clustering, and community detection to uncover the narratives in emerging COVID-19 conspiracy theories. Similarly, Lei and Huang (2023) enhance a transformer-based classifier with eventrelation graphs to capture narrative structure in long-form news, showing that such structural information improves conspiracy detection and generalization to unseen media sources. Beyond purely textual representations, Steffen (2025) use unsupervised multimodal topic modeling to analyze textual and visual conspiracy discourse in Germanlanguage Telegram channels.

A number of datasets related to various conspiracy theories were also made available. COCO provides tweets covering twelve types of COVID-related conspiracies, together with benchmarks for stance and topic classification (Langguth et al. 2023). YouNICon extends such resources to YouTube, providing a large-scale collection of conspiracyrelated videos with manually labeled subsets for conspiracy detection and topic classification (Liaw et al. 2023). Related work has also evaluated transformer-based models on topic-diverse conspiracy data. Phillips, Ng, and Carley (2022) introduce a Twitter dataset that spans climate change, COVID-19 origins and vaccines, and the Epstein-Maxwell trial, and evaluate models of the BERT-family for conspiracy, stance, and topic classification. Based on this data set, George et al. (2024) augment a RoBERTa-based conspiracy classifier with text-derived psycho-linguistic features that capture emotion and moral framing, showing that these additional signals improve classification over text alone. Extending this line of work beyond topic-specific classification, Fort et al. (2023) examine whether models trained on one conspiracy topic can generalize to unseen topics. They find that BART outperforms SVM baselines, but cross-topic performance remains variable and can decline substantially.

Diab, Nefriana, and Lin (2024) further show that discussing a conspiracy does not necessarily imply endorsing it, using human-labeled Reddit posts to train BERT-family classifiers and evaluate zero- and few-shot GPT prompting for distinguishing conspiracy endorsement from criticism, skepticism, and debunking.

A few works use LLMs for conspiracy-related tasks. LLMs were found to perform similarly to fine-tuned transformers in detecting conspiracies in German Telegram data (Pustet, Steffen, and Mihaljevic 2024). Liu et al.´ (2024) instruction-tune an emotion-oriented LLM for conspiracy detection, topic classification, and stance-related tasks, showing improvements over general-purpose LLMs and ChatGPT on most evaluated tasks. Complementarily, Corso, Pierri, and De Francisci Morales (2025) evaluate open-weights LLMs in a zero-shot setting for detecting conspiracy theories from TikTok video transcripts, finding that some prompting configurations achieve high precision, outperforming a fine-tuned RoBERTa baseline, although performance varies substantially across models and prompts.

Contextual Detection of Misinformation Conspiracy detection is closely related to the broader literature on misinformation, rumors, and fake-news detection, where additional contextual and social signals have been incorporated to facilitate accurate classification.

At the content level, approaches have combined lexical, topic, hashtag, and URL features with information from the source message (Castillo, Mendoza, and Poblete 2011; Qazvinian et al. 2011). User- and source-level signals have also been incorporated, including user credibility and stance (Li, Zhang, and Si 2019), as well as source-user behavior jointly modeled with article content (Ruchansky, Seo, and Liu 2017). Temporal context has been captured from sequences of related posts and user responses, modeling how rumor discussions evolve over time (Ma et al. 2016), while propagation and network structure have been represented through diffusion features (Castillo, Mendoza, and Poblete 2011) and graph-based approaches such as Bi-GCN, which models both top-down propagation from the source post and bottom-up dispersion through the discussion graph (Bian et al. 2020).

Together, these works show that user’s stance, and engagement patterns, along with propagation structure, provide important signals that could be harnessed to achieve improved classification.

Agentic Frameworks for Fact Verification Agentic approaches have been applied to evidence gathering and verification in misinformation detection. FactAgent decomposes fake-news verification into a structured expert workflow that combines the LLM’s internal analysis with external search and source-credibility tools (Li, Zhang, and Malthouse 2024). MARO assigns subtasks such as linguistic analysis, external fact consistency, and user-comment analysis to specialized agents, and automatically optimizes the decision rules used to combine their analysis for crossdomain misinformation detection (Li et al. 2025). An agentic LLM approach for conspiracy-endorsement detection was designed to mitigate the “Reporter Trap,” in which objective reporting is incorrectly classified as endorsement, on the PsyCoMark benchmark (Spanakis et al. 2026).

We note that while the tasks of fact verification and the detection of misinformation bear similarities to the task at hand, they also differ significantly. We are not interested in the ‘truth’ value of a statement but in its pragmatic function.<sup>7</sup>

While our work is inspired by these works, it leverages a distinct set of tools and evidence sources appropriate for this interpretive setting. Rather than retrieving factual evidence to verify a claim, or applying a fixed set of analytical dimensions, our agent can selectively retrieve tweet metadata, the author’s profile, prior discourse, and local ego-network context according to the needs of each case. This also distinguishes our approach from recent agentic conspiracydetection work, which reasons primarily over the target text, retrieved examples, and limited contextual signals, without incorporating user history or broader discourse and network context. Our approach is also unsupervised, requiring no task-specific fine-tuning.

To the best of our knowledge, this is the first work to apply an agentic framework to conspiracy detection using tools that selectively retrieve user information, prior user discourse, and network context.

## 3 Data

## 3.1 Data Curation

Raw Data We used a large archive of Hebrew-language tweets collected between December 2018 and March 2023. The archive, containing 270 million tweets posted by 3.3 million unique users, is estimated to cover 80–90% of public

Hebrew Twitter during this period.<sup>8</sup>

Conspiracy-related Tweets We followed previous work on the creation of conspiracy datasets and used a comprehensive set of keywords in order to extract conspiracyrelated content. This stage is designed to achieve “high recall”, resulting in an adversarial dataset, retaining tweets that either contain conspiracy narratives or discuss related topics (e.g., questions and legitimate concerns regarding vaccines as well as anti-vaccination content). The retrieval rules combine topic-specific keywords and general conspiracy cues. Topics include flat-earth claims, the Rabin assassination, the Yemenite children affair, COVID, medical and pharmaceutical control, global elites and the Illuminati, QAnon, chemtrails, 9/11, climate change, the moon-landing hoax, UFOs and aliens, and the “deep state”. Brief descriptions of these topics, together with the full retrieval lexicon, are provided in Appendix D.

A tweet should match more than one filter in order to be considered. For example the following tweet, retrieved by the chemtrails-related rules.

T1 “Remember the photo of the crop-duster planes I took this afternoon? Here is the result. Dirty and murky skies. This is what we breathe. This is the air pollution that’s killing us. And by the way, they blame it on us. So wake up.”<sup>9</sup>

The tweet matches several topic-related terms and phrases, including “air pollution”, “plane”, “spraying”<sup>10</sup>, and “sky”, together with broader conspiracy cues such as “killing us” and “wake up”. The example illustrates the purpose of the retrieval stage: the rules identify tweets containing topicrelated and potentially conspiratorial signals, but do not determine the label. The full list of keywords, and retrieval rules and thresholds can be found in Appendix D. Removing duplications, this process produced 33,121 unique tweets, each likely to relate to one of the topics listed above.

Adversarial Subset We first used Gemini 3 Flash and GPT-5 to independently classify all 33,121 candidates based on text alone, using the preliminary sampling prompt provided in Appendix A.1. We used disagreement between Gemini 3 Flash and GPT-5 as a signal for cases in which text-only predictions differed across models. The two models disagreed on 3,915 of the 33,121 candidates (11.8%).

A subsample of 504 tweets was annotated by two annotators, see details in Section 3.2 below. The annotated dataset is used to evaluate our classification pipeline. Once validated, the pipeline can be applied to the larger adversarial set. The raw dataset is available to the agents to query, using the various tools.

<sup>8</sup>We used Twitter’s streaming API, tracking a list of Hebrew stopwords. The 80%–90% coverage is estimated by extrapolating the coverage achieved by the tracked terms over Hebrew tweets collected using the Decahose, which provided a random sample of 10% of the public stream.

<sup>9</sup>Original Hebrew is available in Appendix B, Example T1.

<sup>10</sup>In Hebrew, the term used for a ‘crop-duster plane’ is a ‘spraying plane’, after its aerial application of fertilizers and pesticides.

<table><tr><td>Gemini 3 Flash</td><td>GPT-5</td><td>Annotated Count (%)</td><td>Conspiracy-related Count (%)</td></tr><tr><td>0</td><td>0</td><td>26 (5.2%)</td><td>19,283 (58.2%)</td></tr><tr><td>0</td><td>1</td><td>234 (46.4%)</td><td>2,835 (8.6%)</td></tr><tr><td>1</td><td>0</td><td>212 (42.1%)</td><td>1,080 (3.3%)</td></tr><tr><td>1</td><td>1</td><td>32 (6.3%)</td><td>9,923 (30.0%)</td></tr><tr><td>Total</td><td></td><td>504 (100%)</td><td>33,121 (100%)</td></tr></table>

Table 3: Composition of the evaluation set and the full candidate pool according to text-only predictions from Gemini 3 Flash and GPT-5. 0(1) indicates a prediction of not (conspiracy). Note the balanced-oversampling of ‘0 1’ and ‘1 0’.

## 3.2 Data Annotation

We sampled a set of 504 tweets for manual annotation. Cases in which Gemini 3 Flash and GPT-5 produced conflicting predictions were over-sampled. The distribution of tweets by models agreement groups is available in Table 3.

Manual annotation of conspiracy-related tweets is challenging and time consuming. The annotator is required to have a deep understanding of the different topics and related conspiracies. These involve some bizarre claims and in-group lingo, internet culture, and niche use of memes. Labels should be assigned based on the communicative function of the tweet rather than its surface wording alone, and the working definition provided by Douglas and Sutton (2023). The adversarial construction of the dataset (see above) makes the annotation even more challenging. Informed decision is often made only after careful examination of the different contexts – a time consuming and cognitively draining undertaking.

Due to the complexity of the annotation process we took the following approach: All 504 tweets were manually labeled by two domain experts. The first annotated each tweet from scratch: model predictions were hidden and all relevant contexts were examined in order to assign the gold label. The second expert reviewed the labels assigned by the first annotator, exploring the contexts only in cases the original label seemed lacking without further evidence. In the final stage the annotators had a consolidation session, reaching agreement on the gold labels, resulting in 331 non-conspiracy tweets (65.7%) and 173 conspiracy tweets (34.3%).

## 4 Methods

## 4.1 The Agentic Pipeline

Agentic frameworks extend language models beyond direct generation by enabling them to interact with external tools and environments during inference. ReAct (Yao et al. 2023) exemplifies this by interleaving reasoning with actions and observations in an iterative loop with an external environment, while Toolformer (Schick et al. 2023) takes a more static approach, learning when and how to invoke external APIs and incorporate their outputs into generation without an ongoing interactive loop.

In this work we adopt the former approach. The agent is prompted with the task definition and specific guidelines introducing the available tools, regulating the tool use, and defining the input-output formats used through the reasoning and acting loop. This agentic workflow is illustrated in Figure 1 and the full prompt is available in Appendix A.4.

![](images/71a0c47f7eed6c86c457f532d3b786cb684e9c7d0275edf968c52c1880664d51.jpg)  
Figure 1: Full agentic workflow.

## 4.2 Contexts and Tools

Prior work, mostly on fake-news and misinformation, has demonstrated the utility of social context for classification tasks. Fortunately, our unique raw dataset provides a good approximation of the full knowledge of users’ public history and social interactions. A set of four tools was developed, allowing the agent to query and retrieve different types of context, as the need arises in the reasoning loop. We note that tools only serve data that was timestamped before the target tweet was posted, masking any “futuristic” signals. The available tools are described in the remainder of this Subsection.

Tweet Information (metadata) The tweet-information tool retrieves metadata about the target tweet, including its timestamp, text length, hashtags, URLs, and source. These signals can situate the tweet in a temporal or topical context and indicate links to external material or unusual posting behavior.

Author Information (metadata) The author-information tool retrieves profile-level information about the tweet author, including username, display name, bio, account creation date, verification status, follower and following counts, total tweet count, language, and location when available. Such information can provide cues about the author’s identity, self-presentation, or broader account characteristics that may help interpret intent.

User History This tool samples up to 20 tweets posted by the same user in the 30 days preceding the target tweet. This context can reveal prior stance, recurring narratives, repeated references, or ongoing themes that may clarify whether the target tweet endorses, mocks, or merely references a conspiracy narrative.

Network-Aware Ego Context The network tool recovers and returns the ego-network of the user, based on retweet interactions occurring in the 14 days preceding the target tweet. Nodes are users whose content the target author retweeted during this period. This signal provides the immediate community of the user and a glimpse into the content she recently amplified. Nodes are ranked by interaction frequency, and the five most frequently retweeted accounts are retained. For each of these nodes, the tool returns basic profile information and up to five original tweets posted during the 30 days preceding the target tweet.

The restrictions on the number of tweets returned (20 or 5, depending on the tool), the timespan from which we sample (14 or 30 days) or the number of neighbors considered, are applied for efficiency purposes, curbing down runtime. Our purpose here is to demonstrate the advantage of the agentic framework using the specified tools, thus optimizing these hyperparameters is beyond the scope of this work.

## 4.3 Baseline Models

The following settings are used as baselines:

Tweet-only LLM In the most basic setting, an LLM (Gemini-3-flash) is prompted with the target tweet and the task definition. No contexts are provided, and only the tweet is considered.

Degenerate Agent A Gemini 3-flash model stripped of its tools sets the baseline in our experiments. In this setting, the context(s) are preloaded and appended to the target tweet and the classification instructions in the original prompt. The degenerate agent does reason about the provided contexts, but it has no agency as it cannot actively decide on tool use through the reasoning loop.

![](images/547b5e4526e411b0faaae22bdc1ab4eab03c94baf74ab0f3338a2fb6efac2fdf.jpg)  
Figure 2: Two-agent debate workflow. Analyst 2 critiques Analyst 1’s initial decision before the final prediction.

Agents-debate Baseline Multi-agent debates provide a complementary mechanism for exposing alternative interpretations through critique and reconsideration (Liang et al. 2024; Han, Zheng, and Tang 2025; Liu et al. 2025). We use a simple debate setting in which one model (Analyst 1) classifies a target tweet, then invites another model (Analyst 2) to debate and criticize its reasoning and classification decisions. Analyst 2 is instructed to pay particular attention to possible non-literal interpretations. Given the critical response provided by Analyst 2, Analyst 1 re-weighs her position and produces the final decision. Analyst a is only exposed to the target tweet and to the reasoning and decisions made by the other analyst. This workflow is illustrated in Figure 2 and the respective prompts and protocol are available in Appendix A.6. Note that the analysts differ both in the assigned persona and in the foundation model used – Gemini-3-flash and GPT-5 as Analysts 1 and 2, respectively.

## 4.4 Experimental Setting

We formulate the task as binary tweet-level classification. Given a target Hebrew tweet, the system predicts whether it should be labeled as conspiracy (1) or non-conspiracy (0). In all settings, the model (agent) returns a structured output containing the predicted label along with a short rationale (see Table 2). We report accuracy, precision, recall and F-Score over the annotated dataset, establishing the basic performance of each model and setting.

Our experimental settings, described below, aim to separate three factors that are often conflated in context-aware classification: (i) the availability of contextual evidence; (ii) the type of evidence provided; and (iii) the mechanism through which that evidence is accessed.

We thus have two main experimental settings: the Preloaded mode, using degenerate agents, and the Agentic mode. In the Preloaded mode the agent receive the context by default, whereas in the Agentic mode agents determine whether to access a context and when.

Within each of these settings, we assess the contribution of each context/tool separately and in a full setting in which all contexts/tools are available to the agent. The various settings are further explained in Table 4.

Execution The agentic workflows were implemented using LangChain (Chase 2022) and LangGraph (LangChain, Inc. 2024). Gemini-3-Flash was used for the text-only baseline and all single-agent LLM conditions. GPT-5 was used as the second agent in the debate setting. Temperature was set to 0.2 in all settings. Full prompts are provided in Appendix A.

## 5 Results and Analysis

All settings (Table 4) were first evaluated on the annotated dataset. We report overall accuracy and precision, recall and F1-score with respect to the positive class. Classification results complement with error analysis.

We follow the classification results with analysis of the tool usage: exploring the utility, usage frequency, and temporal patterns in which tools are used.

Finally, based on all reasoning steps, we explore the input/output tradeoff (in “token economy”) between the agentic and the degenerate frameworks.

## 5.1 Overall Performance

All classification results are presented in Table 5. The baseline results are established by the Text-only LLM with accuracy of 0.6 and F1-score of 0.535. Best over-all performance (accuracy of 0.790 and F-score of 0.73) are achieved by the full agentic framework (all tools available). In comparison, the degenerate agent using all contexts achieves an accuracy of 0.707 and F-score of 0.67. Allowing only one tool, the best results are achieved with the User History context (Accuracy: 0.748; F: 0.698), outperforming the Degenerate Agent with full context. This last result demonstrates the benefits of using the agentic framework, compared to the use of elaborated contexts as input.

The Two-agent Debate framework outperforms the single LLM baseline but falls short compared to all other settings.

These results confirm, yet again, the challenge in conspiracy detection and the need for contexts, especially over our “adversarial” dataset.

We compare the text-only LLM with the best-performing setting Agentic: All Tools model, referred to below as the full-context agent.

Tables 6(a) and (b) show that contextual retrieval improves classification across both classes. In absolute terms, the full-context agent reduces false positives from 142 to 76 and false negatives from 58 to 30. Context therefore helps both identify conspiratorial discourse and reject nonconspiratorial tweets that contain similar surface-level language.

## 5.2 Error Analysis

In order to gain insight into the performance of the agent, we explore cases in which both the agent and the baseline erred as well as instances in which the baseline provided the correct classification while the agent misclassified it.

Errors introduced by context Although achieving a significant overall improvement (F1 0.53 → 0.73), in some cases, the agent (all tools available) misclassified tweets that were correctly classified by the baseline, 78% of these are false positives. This result reminds us that sometimes words should be taken at face value and incomplete contexts, often depending on hyperparameter settings, may distort the interpretation.

Shared errors We qualitatively explored the cases over which both the baseline and the agent erred. We find that these cases fall under two main categories: (i) Failure to distinguish conspiracy claims from strong political or constitutional criticism and accusations of corruption. (ii) Tweets that explicitly mock a conspiracy theory or discuss it without endorsing it. The following tweets, respectively, illustrate the two categories: (a) “I propose a different compromise... that the Netanyahu trial (the spectacle) take place in my backyard... Half the nation is certain that cases were fabricated here to carry out a coup. Half the nation feels that democracy has been destroyed. Half the nation thinks the judicial system is corrupt to the core. All of them are patriots and believe that we have reached the abyss, on the verge of an explosion. We must reach broad consensus”; and (b) “checked with the Ministry of Pandemics. The new virus will arrive redacted – highly classified.” The original tweets with English translations are provided as Examples E1 and E2, respectively, in Appendix C.

We note, however, that the agent does classify many such cases correctly.

<table><tr><td>Setting</td><td>Context / Tools</td><td> Description</td></tr><tr><td>LLM</td><td></td><td>Gemini 3 Flash classifies the tweet without added context.</td></tr><tr><td rowspan="4">Degenerate</td><td>Tweet+User</td><td>Tweet and author information are inserted into the prompt.</td></tr><tr><td>History</td><td>User history is inserted into the prompt.</td></tr><tr><td>Network</td><td>Ego-network context is inserted into the prompt.</td></tr><tr><td>ALL</td><td>All context sources are inserted into the prompt.</td></tr><tr><td>Debate</td><td></td><td>Gemini 3 Flash and GPT-5 classify through the fixed debate workflow.</td></tr><tr><td rowspan="4">Agentic</td><td>Tweet+User</td><td>Agent may retrieve tweet and author information.</td></tr><tr><td>History</td><td>Agent may retrieve previous tweets by the same author.</td></tr><tr><td>Network</td><td>Agent may retrieve ego-network context.</td></tr><tr><td>ALL</td><td>Agent may use all contextual tools.</td></tr></table>

Table 4: Experimental conditions evaluated in the study.

<table><tr><td>Setting</td><td>Acc.</td><td>Prec.</td><td>Recall</td><td>F1</td></tr><tr><td>Text-only LLM</td><td>0.603</td><td>0.447</td><td>0.665</td><td>0.535</td></tr><tr><td>Two-agent Debate</td><td>0.683</td><td>0.534</td><td>0.584</td><td>0.558</td></tr><tr><td>Degenerate: Tweet+User</td><td>0.691</td><td>0.540</td><td>0.669</td><td>0.597</td></tr><tr><td>Degenerate: Network</td><td>0.667</td><td>0.509</td><td>0.803</td><td>0.623</td></tr><tr><td>Degenerate: History</td><td>0.681</td><td>0.521</td><td>0.855</td><td>0.648</td></tr><tr><td>Agentic: Tweet+User</td><td>0.762</td><td>0.657</td><td>0.642</td><td>0.649</td></tr><tr><td>Agentic: Network</td><td>0.726</td><td>0.578</td><td>0.751</td><td>0.653</td></tr><tr><td>Degenerate: Full Context</td><td>0.707</td><td>0.546</td><td>0.866</td><td>0.670</td></tr><tr><td>Agentic: History</td><td>0.748</td><td>0.593</td><td>0.850</td><td>0.698</td></tr><tr><td>Agentic: All tools</td><td>0.790</td><td>0.653</td><td>0.827</td><td>0.730</td></tr></table>

Table 5: Overall classification performance across the evaluated conditions.

(a) Text-only LLM
<table><tr><td colspan="3">Pred. C Pred. NC</td></tr><tr><td>Gold C</td><td>115</td><td>58</td></tr><tr><td>Gold NC</td><td>142</td><td>189</td></tr></table>

(b) Agentic: All Tools
<table><tr><td colspan="3">Pred. C Pred. NC</td></tr><tr><td>Gold C</td><td>143</td><td>30</td></tr><tr><td>Gold NC</td><td>76</td><td>255</td></tr></table>

Table 6: Confusion matrices for the text-only LLM and Agentic: All Tools. C denotes conspiracy and NC denotes non-conspiracy.

## 5.3 Tool-Use Patterns

The trace logs show that an agent with tool access does not simply use all tools at her disposal, retrieving all contexts for every tweet. As evident from Table 7 no tool was used by the agent in 10.9% of the cases.

Considering all runs in which the agent did not use any of the tools, both the agent and the text-only LLM achieved 94.5% accuracy and produced identical predictions. In contrast, among the runs in which the agent requested at least one context to be retrieved, text-only accuracy was 56.1%, compared with 77.1% for the agent, suggesting that contextual retrieval improves performance over the harder cases where text-only classification struggles.

Table 7 also presents the patterns of consecutive tool usage. In 36.3% of the cases the agent requested both the User Info and User History in the first step, making the final judgment immediately after. Three tools were used in about 25% of the cases, with User Info + User History → Network being the most common pattern (12.3% of the cases). The userate of all tools (regardless of call order) is presented in Ta-

<table><tr><td>Pattern</td><td>Count</td><td>Share</td></tr><tr><td>User Info + User History</td><td></td><td>183 36.3%</td></tr><tr><td>User Info + User History → Network</td><td></td><td>6212.3%</td></tr><tr><td>No Tool Use</td><td></td><td>5510.9%</td></tr><tr><td>User Info → User History</td><td></td><td>47 9.3%</td></tr><tr><td>Tweet Info + User Info → User History</td><td>31</td><td>6.2%</td></tr><tr><td>User Info → User History → Network</td><td></td><td>28 5.6%</td></tr><tr><td>Other Patterns</td><td></td><td>98~19.4%</td></tr></table>

Table 7: Tool-use patterns in the On-demand Full Context condition. “+” marks tools called in the same step. “→” marks a later tool call after observing a previous result. Author Info, User History, Tweet Info, and Network refer to the corresponding context sources described in Section 4.2.

ble 8. User Info (metadata) proved the most important tool, used in 89.1% of cases. User History was called 81% of the cases, while Tweet Info (metadata) and Network contexts were used in 21-25% of the cases.
<table><tr><td>Tool in On-demand Full Context</td><td>Run-use rate</td></tr><tr><td>Tweet information</td><td>21.8%</td></tr><tr><td>Network context</td><td>24.4%</td></tr><tr><td>User history</td><td>81.0%</td></tr><tr><td>User information</td><td>89.1%</td></tr></table>

Table 8: Tool-use rates within the On-demand Full Context condition.

## 5.4 Analysis of ‘Token Economy’

Token Economy is a colloquial term often used to describe, estimate, and manage the costs of LLM usage. Generally speaking, tokens are used in three distinct stages: input tokens in the prompt, generated tokens at the output, and reasoning tokens. Reasoning tokens may not be visible to the user and are often considered as “hidden” generated output. However, some models explicitly output the intermediate reasoning steps. API providers often charge per token, usually charging different rates for input and output tokens. Agentic frameworks introduce a tradeoff between input, reasoning and output tokens. While initial prompts are typically shorter, the reasoning loops and the retrieved information may result in a significant increase in the use of reasoning and (intermediate) output tokens (Bai et al. 2026). These frameworks and billing policies directly translate tokens into dollar cost, creating real economic incentives to minimize token usage (Bergemann, Bonatti, and Smolin 2025).

While we do not have direct access to token usage,<sup>11</sup> we can provide a rough estimate through the number of initial and intermediate input and output character count.

The best performing agent (all tools available) used an average of 11,434 input characters per instance, across all tool calls, compared with an average of only 8,251 characters in the degenerate agent setting with full context. The numbers of output characters produced (final and intermediate) by the models are 1048 (agent) and 614 (degenerate agent). On average, our agent consumes ∼ ×1.5 input characters and ∼ ×2, a noneligible tradeoff that should be considered in light of the a improvement in accuracy (0.707 → 0.79) and F-score (0.67 → 0.73) and frequently changing billing policies. A more thorough analysis of this tradeoff is beyond the scope of this work.

## 6 Discussion

Contexts The error patterns suggest that the central challenge in conspiracy-tweet classification is not merely recognizing conspiratorial content, but inferring the underlying function of the utterance, the illocutionary act, in terms of speech acts theory Austin (1975). These two examples (original language in Appendix B) illustrate the challenge and the benefits of the agentic framework:

D1: “My brother texted me about how lemons kill cancer cells and can treat several types of cancer, and how pharmaceutical companies are hiding it from us. And I’m considered the black sheep of the family.”

D2: “‘The Zombie Guide: The term ‘conspiracy theory’ is already passe. From now on, when you´ encounter a truth you cannot or do not want to investigate, instead of saying ‘conspiracy theory,’ say: the Earth is flat. That way, you can avoid thinking about and investigating thefrightening truth”

The first tweet quotes a conspiratorial message verbatim, but uses the concluding words to mock it. Text-only LLMs tend to misinterpret this as ‘sheep’ metaphor is common in the conspiratorial discourse. The second example seems to mock conspiracy theories while in fact it promotes them by mocking its critiques. With contexts, however, the agents correctly classify the utterances. Context appears to help by calibrating the textual interpretation rather than merely adding more text. User history can reveal recurring stance, rhetorical style, and narrative participation, while author and ego-network information provide additional cues about the social and discursive setting of a post.

Multi-Agent Debate Interestingly, the two analysts (agents) in the debate settings initially disagreed on 17.9% of the target tweets. Following the critique offered by Analyst 2, Analyst 1 reversed her decision in 60% of the cases. Of these changes, (66.7%) corrected an initial error, while a correct decision was reversed in 33.3% of the cases. While the debate’s overall performance was unimpressive, this framework shows clear potential. This direction will be explored in future work.

Limitations This work comes with some limitations. First, hyper parameters were not optimized. This work serves as a convincing proof of concept for the utility of the agentic framework on the specific task and over a challenging adversarial dataset. As such we did not optimize hyperparameters or explored other possible settings, e.g., using other LLMs in the different settings. Consequently, the analysis of the results and the errors apply to this dataset and the specific models used. Other or future models may perform differently in terms of reasoning or token efficiency.

An Ethical Note The use of user histories and network/social context should be attuned to privacy considerations, particularly when user-level information is retained, analyzed, or released. Moreover, incorrectly labeling a user may incur social costs. The system is therefore better suited to research purposes rather than as an automated moderation tool without human review.

## 7 Conclusion

This paper examined Hebrew conspiracy-tweet classification as a context-dependent interpretation task. Using a large local archive of Hebrew tweets and a targeted evaluation set of manually labeled tweets, we find that agentic approach performs significantly better than other approaches, even when the exact same contextual information is available. Our findings suggest treating conspiracy detection on social media as a socially embedded interpretation problem rather than purely a text-classification problem.

## References

Allen, J.; Watts, D. J.; and Rand, D. G. 2024. Quantifying the impact of misinformation and vaccine-skeptical content on Facebook. Science, 384(6699): eadk3451.

Austin, J. L. 1975. How to do things with words, volume 5. Harvard university press.

Bai, L.; Huang, Z.; Wang, X.; Sun, J.; Mihalcea, R.; Brynjolfsson, E.; Pentland, A.; and Pei, J. 2026. How do ai agents spend your money? analyzing and predicting token consumption in agentic coding tasks. arXiv preprint arXiv:2604.22750.

Bergemann, D.; Bonatti, A.; and Smolin, A. 2025. The Economics of Large Language Models: Token Allocation, Fine-Tuning, and Optimal Pricing. In Proceedings of the 26th ACM Conference on Economics and Computation.

Bian, T.; Xiao, X.; Xu, T.; Zhao, P.; Huang, W.; Rong, Y.; and Huang, J. 2020. Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 549–556.

Caspit, B. 2015. [Survey on Conspiracy Beliefs Concerning the Assassination of Yitzhak Rabin]. Maariv. Hebrew-language news article.

Castillo, C.; Mendoza, M.; and Poblete, B. 2011. Information Credibility on Twitter. In Proceedings of the 20th International Conference on World Wide Web, 675–684. New York, NY, USA: Association for Computing Machinery.

Chase, H. 2022. LangChain. Software.

Corso, F.; Pierri, F.; and De Francisci Morales, G. 2025. Conspiracy Theories and Where to Find Them on TikTok. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8346–8362. Vienna, Austria: Association for Computational Linguistics.

Del Vicario, M.; Bessi, A.; Zollo, F.; Petroni, F.; Scala, A.; Caldarelli, G.; Stanley, H. E.; and Quattrociocchi, W. 2016. The Spreading of Misinformation Online. Proceedings of the National Academy ofSciences, 113(3): 554–559.

Diab, A.; Nefriana, R.; and Lin, Y.-R. 2024. Classifying Conspiratorial Narratives at Scale: False Alarms and Erroneous Connections. Proceedings of the International AAAI Conference on Web and Social Media, 18(1): 340–353.

DiResta, R. 2024. Invisible Rulers: The People Who Turn Lies into Reality. New York: PublicAffairs.

Douglas, K. M.; and Sutton, R. M. 2023. What are conspiracy theories? A definitional approach to their correlates, consequences, and communication. Annual review ofpsychology, 74(1): 271–298.

Douglas, K. M.; Sutton, R. M.; and Cichocka, A. 2017. The Psychology of Conspiracy Theories. Current Directions in Psychological Science, 26(6): 538–542.

Douglas, K. M.; Uscinski, J. E.; Sutton, R. M.; Cichocka, A.; Nefes, T.; Ang, C. S.; and Deravi, F. 2019. Understanding Conspiracy Theories. Political Psychology, 40(S1): 3–35.

Enders, A. M.; Klofstad, C. A.; and Uscinski, J. E. 2024. The Relationship Between Conspiracy Theory Beliefs and Political Violence. Harvard Kennedy School Misinformation Review, 5(6).

Fort, M.; Tian, Z.; Gabel, E.; Georgiades, N.; Sauer, N.; Dakota, D.; and Kubler, S. 2023. Bigfoot in Big Tech: Detecting Out of¨ Domain Conspiracy Theories. In Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing, 353–363. Varna, Bulgaria: INCOMA Ltd., Shoumen, Bulgaria.

George, A. R.; Ahrens, M.; Pierrehumbert, J. B.; and McMahon, M. 2024. Conspiracy Detection Beyond Text: Exploring the Feasibility of Adding Psycho-Linguistic Features to Enhance Conspiracy Detection Models. In Disinformation in Open Online Media, volume 15175 of Lecture Notes in Computer Science, 32–45. Cham: Springer.

Grinberg, N.; Joseph, K.; Friedland, L.; Swire-Thompson, B.; and Lazer, D. 2019. Fake News on Twitter During the 2016 U.S. Presidential Election. Science, 363(6425): 374–378.

Han, C.; Zheng, W.; and Tang, X. 2025. Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 15114– 15129. Suzhou, China: Association for Computational Linguistics.

Hilberts, S.; Govers, M.; Petelos, E.; and Evers, S. 2025. The impact of misinformation on social media in the context of natural disasters: narrative review. JMIR infodemiology, 5: e70413.

Jackson, S. J.; Bailey, M.; and Welles, B. F. 2020. # HashtagActivism: Networks ofrace and genderjustice. Mit Press.

LangChain, Inc. 2024. LangGraph. Software.

Langguth, J.; Schroeder, D. T.; Filkukova, P.; Brenner, S.; Phillips,´ J.; and Pogorelov, K. 2023. COCO: An Annotated Twitter Dataset of COVID-19 Conspiracy Theories. Journal ofComputational Social Science, 6(2): 443–484.

Lei, Y.; and Huang, R. 2023. Identifying Conspiracy Theories News based on Event Relation Graph. In Findings of the Association for Computational Linguistics: EMNLP 2023, 9811–9822. Singapore: Association for Computational Linguistics.

Li, H.; Wang, A.; Li, K.; Wang, Z.; Zhang, L.; Qiu, D.; Liu, Q.; and Su, J. 2025. A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, 5709–5725. Suzhou, China: Association for Computational Linguistics.

Li, Q.; Zhang, Q.; and Si, L. 2019. Rumor Detection by Exploiting User Credibility Information, Attention and Multi-task Learning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 1173–1179. Florence, Italy: Association for Computational Linguistics.

Li, X.; Zhang, Y.; and Malthouse, E. C. 2024. Large Language Model Agent for Fake News Detection. arXiv preprint arXiv:2405.01593.

Liang, T.; He, Z.; Jiao, W.; Wang, X.; Wang, Y.; Wang, R.; Yang, Y.; Shi, S.; and Tu, Z. 2024. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 17889–17904. Miami, Florida, USA: Association for Computational Linguistics.

Liaw, S. Y.; Huang, F.; Benevenuto, F.; Kwak, H.; and An, J. 2023. YouNICon: YouTube’s CommuNIty of Conspiracy Videos. Proceedings ofthe International AAAI Conference on Web and Social Media, 17(1): 1102–1111.

Liu, Y.; Liu, Y.; Zhang, X.; Chen, X.; and Yan, R. 2025. The Truth Becomes Clearer Through Debate! Multi-Agent Systems with Large Language Models Unmask Fake News. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, 504–514. New York, NY, USA: Association for Computing Machinery.

Liu, Z.; Liu, B.; Thompson, P.; Yang, K.; and Ananiadou, S. 2024. ConspEmoLLM: Conspiracy Theory Detection Using an Emotion-Based Large Language Model. arXiv preprint arXiv:2403.06765.

Ma, J.; Gao, W.; Mitra, P.; Kwon, S.; Jansen, B. J.; Wong, K.-F.; and Cha, M. 2016. Detecting Rumors from Microblogs with Recurrent Neural Networks. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 3818–3824. AAAI Press.

Mahl, D.; Schafer, M. S.; and Zeng, J. 2023. Conspiracy theories¨ in online environments: An interdisciplinary literature review and agenda for future research. New media & society, 25(7): 1781– 1801.

Muhammed, S. T.; and Mathew, S. K. 2022. The disaster of misinformation: a review of research in social media. International Journal ofData Science and Analytics, 13(4): 271–285.

Phillips, S. C.; Ng, L. H. X.; and Carley, K. M. 2022. Hoaxes and Hidden Agendas: A Twitter Conspiracy Theory Dataset: Data Paper. In Companion Proceedings ofthe Web Conference 2022, 876– 880. New York, NY, USA: Association for Computing Machinery.

Pustet, M.; Steffen, E.; and Mihaljevic, H. 2024. Detection of´ Conspiracy Theories Beyond Keyword Bias in German-Language Telegram Using Large Language Models. In Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024), 13–27. Mexico City, Mexico: Association for Computational Linguistics.

Qazvinian, V.; Rosengren, E.; Radev, D. R.; and Mei, Q. 2011. Rumor Has It: Identifying Misinformation in Microblogs. In Proceedings ofthe 2011 Conference on Empirical Methods in Natural Language Processing, 1589–1599. Edinburgh, Scotland, UK: Association for Computational Linguistics.

Ruchansky, N.; Seo, S.; and Liu, Y. 2017. CSI: A Hybrid Deep Model for Fake News Detection. In Proceedings ofthe 2017 ACM

In addition, characterize the text according to the following criteria

1. Label: whether the text is conspiratorial   
("conspiracy" / "non\_conspiracy")

- Keep the response short and clear.

on Conference on Information and Knowledge Management, 797– 806. New York, NY, USA: Association for Computing Machinery.

Schick, T.; Dwivedi-Yu, J.; Dess\`ı, R.; Raileanu, R.; Lomeli, M.; Hambro, E.; Zettlemoyer, L.; Cancedda, N.; and Scialom, T. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Advances in Neural Information Processing Systems, volume 36, 68539–68551. Curran Associates, Inc.

Shahsavari, S.; Holur, P.; Wang, T.; Tangherlini, T. R.; and Roychowdhury, V. 2020. Conspiracy in the Time of Corona: Automatic Detection of Emerging COVID-19 Conspiracy Theories in Social Media and the News. Journal of Computational Social Science, 3(2): 279–317.

Shi, F.; Chen, X.; Misra, K.; Scales, N.; Dohan, D.; Chi, E. H.; Scharli, N.; and Zhou, D. 2023. Large language models can be¨ easily distracted by irrelevant context. In Proceedings of the 40th International Conference on Machine Learning, 31210–31227. PMLR.

Spanakis, P.; Lymperaiou, M.; Filandrianos, G.; Voulodimos, A.; and Stamou, G. 2026. AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection. In Proceedings of the 20th International Workshop on Semantic Evaluation (2026), 1175–1204. San Diego, California, USA: Association for Computational Linguistics.

Starbird, K. 2017. Examining the Alternative Media Ecosystem Through the Production of Alternative Narratives of Mass Shooting Events on Twitter. In Proceedings of the International AAAI Conference on Web and Social Media, volume 11, 230–239.

Starbird, K.; Arif, A.; and Wilson, T. 2019. Disinformation as Collaborative Work: Surfacing the Participatory Nature of Strategic Information Operations. Proceedings of the ACM on Human-Computer Interaction, 3(CSCW): 1–26.

Steffen, E. 2025. More than Memes: A Multimodal Topic Modeling Approach to Conspiracy Theories on Telegram. Proceedings of the International AAAI Conference on Web and Social Media, 19(1): 1831–1844.

Vosoughi, S.; Roy, D.; and Aral, S. 2018. The Spread of True and False News Online. Science, 359(6380): 1146–1151.

World Economic Forum. 2026. The Global Risks Report 2026. Technical report, World Economic Forum, Geneva, Switzerland.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

## A Prompt Templates and Agent Instructions

The following prompt templates and agent instructions document the LLM workflows used in this study. Anglebracketed fields below denote values instantiated for each tweet. Experimental prompts are reproduced verbatim from the implementation, while the preliminary sampling prompt is provided in English translation.

## A.1 Preliminary Sampling Prompt

The following prompt was used only to obtain preliminary text-only predictions for disagreement-based sampling of the evaluation set. Its outputs were not used as gold labels or as reported experimental results. The original prompt was written in Hebrew. An English translation is provided below.

You classify Hebrew tweets according to   
their degree of conspiratorial content.

2. Score: degree of conspiratorial content   
between 0 and 1

1. A claim involving a secret connection or   
hidden plot

2. Contradiction of an accepted official   
explanation or position

3. Use of unreliable sources or vague   
language   
("they say...", "everyone knows...")

4. Encouragement of distrust toward major   
institutions   
(government, science, media, etc.)

5. An "us versus them" framing

You will receive a list of tweets. Each   
tweet has a row\_id and text.

Required output:   
Return only valid JSON as a list of objects,   
in the same order and with   
the same number of items as the input.

For each item, return:   
{   
"row\_id": "the received identifier",   
"label": "conspiracy" or "non\_conspiracy",   
"score": number between 0 and 1,   
"signals": [list of numbers from   
{1,2,3,4,5}],   
"rationale": "two short sentences   
explaining the classification"   
}

- Make sure to distinguish conspiratorial   
claims from irony or ridicule.

## A.2 Common Tweet Input

The following user-message template was used for singleagent classification:   
Classify this tweet.   
tweet id: <tweet\_id>   
day: <day>   
tweet text: <tweet\_text>

## A.3 Text-Only Classification

You classify whether a Hebrew tweet is   
conspiracy or non\_conspiracy.   
Classify from the tweet text alone.   
Return only valid JSON:   
{   
"label": 0 or 1,   
"score": number between 0 and 1,   
"rationale": short explanation grounded in   
the tweet text   
}   
Rules:   
- score - conspiracy confidence (0 = not   
conspiracy, 1 = conspiracy)

## A.4 On-Demand Retrieval

The on-demand conditions used the same prompt structure, with only the descriptions of the tools available in a given condition included. The On-demand Full Context system prompt is reproduced below.

You classify whether a Hebrew tweet is   
conspiracy or non\_conspiracy.   
First, decide whether the tweet can be   
confidently classified from the text   
alone.   
If yes, do not use any tools.

Only if the tweet is ambiguous (e.g., sarcasm, mockery, exaggeration, or unclear intent) and additional context could change the decision, you may use tools.

When using tools, choose them based on the missing information and use as few as necessary.

Stop using tools once you can confidently classify the tweet.

```prolog
Tool usage rules:
- tweet_info -> returns metadata (tweet date
, text length, hashtags, URLs, posting
source).
Can be used to interpret hashtags and
links, understand the timing of the
tweet, and identify how it was posted
(e.g., automated or unusual sources).
```

- user\_info -> returns profile (bio, account   
age, followers, verification, language)   
Can be used when the author’s identity can   
help understand intent. Can help   
identify ideological alignment,   
suspicious/bot-like users, or unusual   
account characteristics.   
- previous\_tweets -> returns earlier tweets   
by the same user with dates and text.   
Can be used to discover repeated   
narratives, consistent stance, or   
ongoing themes that indicate genuine   
belief rather than a one-off joke.   
- ego\_context -> returns top interacting   
users, their profile, and recent tweets.   
Can be used to understand the typical   
language and tone of the community (   
what is normal vs unusual). Neighbors’   
recent tweets can indicate what   
topics, claims, and narratives are   
currently discussed in the user’s   
surrounding community, helping   
interpret an ambiguous tweet.   
Return only valid JSON:   
{   
"label": 0 or 1,   
"score": number between 0 and 1,   
"rationale": short explanation grounded in   
the tool outputs   
}   
Rules:   
- score - conspiracy confidence (0 = not   
conspiracy, 1 = conspiracy)   
- If you used tools, ground the rationale in   
those tool outputs and explicitly   
attribute evidence to tweet metadata,   
user profile, user history, or ego  
context.

## A.5 Preloaded Context

The preloaded conditions used the following system prompt:   
You classify whether a Hebrew tweet is   
conspiracy or non\_conspiracy.   
Classify using the tweet text and any   
additional context related to the tweet   
or user that is already provided in the   
user message.   
Return only valid JSON:   
{   
"label": 0 or 1,   
"score": number between 0 and 1,   
"rationale": short explanation grounded in   
the provided text and context   
}   
Rules:   
- score - conspiracy confidence (0 = not   
conspiracy, 1 = conspiracy)

independently and challenge Analyst 1   
only when there is meaningful evidence   
that the classification may be incorrect   
Pay close attention to:   
- sarcasm   
- irony   
- jokes or exaggeration   
- mocking or non-literal tone   
Only infer sarcasm or irony when there is   
clear evidence in the tweet. Do not   
treat extreme or unusual claims as   
sarcasm by default.   
Distinguish between real conspiracy belief   
and language that imitates or mocks it.   
Challenge the "conspiracy" label only when   
an alternative interpretation is clearly   
supported by the text.   
Also challenge a non-conspiracy   
classification when the tweet appears to   
explain a significant social or   
political event through a secret plot by   
powerful actors.   
Think independently and respond concisely in   
the required format.   
Do not return JSON.   
Moderator: respond to Analyst 1.   
Format:   
- First line exactly: agreement: agree   
or: agreement: disagree   
Second line exactly: verdict: conspiracy   
or: verdict: non\_conspiracy   
- Then 1-3 short sentences explaining your   
reasoning.   
Do not return JSON.   
Moderator: read Analyst 2’s response and   
make your final decision.   
Decide whether to keep your original opinion   
or revise it based on Analyst 2’s   
argument.   
Only revise your decision if Analyst 2   
provides a convincing reason.   
Return EXACTLY one valid JSON object and   
nothing else:   
{   
"label": 0 or 1,   
"score": number between 0 and 1,   
"rationale": "short explanation grounded   
in the discussion, reflecting whether   
Analyst 2 influenced your decision"   
}

<table><tr><td>Condition</td><td>Available context</td></tr><tr><td>Tweet+User</td><td>tweet_info,user_info</td></tr><tr><td>History</td><td>previous_tweets</td></tr><tr><td>Network</td><td>ego_context</td></tr><tr><td>Full Context</td><td>all four sources</td></tr></table>

Table 9: Context sources available in the matched ondemand and preloaded conditions.

- If the provided context affects the   
decision, ground the rationale in that   
context and explicitly attribute the   
evidence to the relevant part of the   
provided context.

For preloaded conditions, the context was appended to the tweet text using the following template before the common user-message template above was applied:

Additional information that may help with   
classification:   
<preloaded\_context>

The available context was matched across the on-demand and preloaded conditions:

## A.6 Two-Agent Debate

The debate condition used the same tweet input fields as the common input template above and no external contextual tools.

You are Analyst 1 in a tweet-classification   
debate.   
Follow the moderator’s instruction exactly.   
Task:   
Classify tweets as:   
- conspiracy   
- non\_conspiracy   
For final decisions:   
- Return valid JSON only   
- label 1 = conspiracy   
- label 0 = non\_conspiracy   
Moderator: give your initial opinion.   
Format:   
- First line exactly: verdict: conspiracy   
or: verdict: non\_conspiracy   
- Then 1-3 short sentences explaining your   
reasoning.   
Do not return JSON.   
You are Analyst 2, a critical and skeptical   
reviewer in a tweet-classification   
debate.   
Do not agree with Analyst 1 automatically,   
but do not disagree merely because the   
tweet is ambiguous. Evaluate the tweet

Rules:   
- label 1 = conspiracy   
- label 0 = non\_conspiracy   
- score = conspiracy confidence (0 = not   
conspiracy, 1 = conspiracy)   
- The rationale should reflect your stance   
after considering Analyst 2   
- No text before or after the JSON

## B Original Hebrew Examples and Translations

This appendix provides the original Hebrew texts, English translations, and posting dates for tweet examples quoted in the paper.

## B.1 Standalone Examples

Example H1: Pfizer storm tweet. Date: December 20, 2021. Original Hebrew:

,    7 n7 DN'N D'O IN

## English translation:

“Those naive people do not realize that the storm is part of Pfizer’s secret agreement to take over the world and create a New World Order. There is no hope for you.”

## B.2 Discussion Examples

Example D1: Lemon and pharmaceutical concealment. Date: March 5, 2019.   
Original Hebrew:

'  ' .n'nn ni'non

## English translation:

“My brother sent me a WhatsApp message about how lemon kills cancer cells and can treat several types of cancer, and how pharmaceutical companies are hiding this from us. And I’m the black sheep of the family.”

Example D2: The Zombie Guide. Date: August 8, 2021.   
Original Hebrew:

"' ""   
,   
" n" D ,7

## English translation:

“The Zombie Guide: The term ‘conspiracy theory’ is already passe. From now on, when you encounter a ´ truth you cannot or do not want to investigate, instead of saying ‘conspiracy theory,’ say: the Earth is flat. That way, you can avoid thinking about and investigating the frightening truth.”

## B.3 Examples from Table 1

Example T1: Chemtrails / crop-duster planes. Date: February 3, 2019.   
Original Hebrew:

.?    ' .n . nix D'nn

## English translation:

“Remember the crop-duster planes I photographed this afternoon?... Here is the result. Dirty and murky skies. This is what we breathe. This is the air pollution that is killing us. And by the way, they blame us for it. So wake up.”

Example T2: Rabin assassination classroom report.   
Date: October 23, 2020.

z  n    i "" ' nn ,    " "  i ,

## English translation:

“A teaching assistant from Pardes Hanna tells her students during remote learning on “Zoom” that she does not know who killed former Prime Minister Yitzhak Rabin. “It is not certain that Yigal Amir was the murderer, and there are many conspiracy theories surrounding the issue.” The parents heard the lesson and filed a complaint with the principal, then with the Ministry of Education and the Pardes Hanna local council.”

Example T3: COVID vaccine satire. Date: January 4, 2023.   
Original Hebrew:

7   '    " .G5-   D n

## English translation:

“Like if you think the COVID vaccines contained strawberry-vanilla milkshake ingredients and tiny location-tracking chips, so the government could track us, including facial recognition technology, in order to put us all into a secret G5 database. Share this, because the media channels will not tell you the truth.”

Example T4: Hospital strike criticism.

Date: August 31, 2021.

Original Hebrew:

. an  nin D' D'an naio ax   
.      n  ' n  "' n n'y 'n' nni   
'   ' . ' ''

## English translation:

“I am suffering from severe pain, and my surgery has been postponed. In Jerusalem, every hospital is on strike, and rightly so, because they have no money. And what about me, Idit Silman? And Member of Knesset Marin from Yisrael Beiteinu, who mocks the doctors by saying that they are merely making empty declarations. A government that is murdering its people.”

Example T5: Climate policy lawsuit.

Date: February 2, 2019.

Original Hebrew:

, '    " ',' ' '

## English translation:

“People around the world are suing their governments to force them to fight global warming. It seems to us that we Israelis also have a strong case against one of the most indifferent governments in the West regarding the greatest problem facing our children and grandchildren.”

## C Error Analysis Examples

Example E1: Strong institutional criticism. Date: November 8, 2019.   
Original Hebrew:

n D     D ,     ? n' 7       ' ,'n  ; . U

## English translation:

“And who is responsible for the severe erosion of public trust in the justice system in recent years? Mainly the system’s leaders: Roni Alsheikh, who as police commissioner led dirty and inflammatory investigations against Netanyahu and his wife; State Attorney

Shai Nitzan, who lies constantly and pursues selective enforcement; and Attorney General Mandelblit, who conducts himself disgracefully as a weak man susceptible to blackmail.”

Example E2: Satirical conspiracy language. Date: February 19, 2023.   
Original Hebrew:

n ' -  '  ' -    ' nnnn ....  75 -7 ini in'7y

## English translation:

“I checked with the Ministry of Pandemics. The new virus will arrive redacted – highly classified.

Whoever contracts the new epidemic will be made to disappear for 75 years ... hahaha.”

## D Conspiracy Topics and Retrieval Lexicon

This appendix lists the lexical resources used for rule-based candidate retrieval. The retrieval process uses general conspiracy cues and topic-specific rules. A matched tweet is not automatically labeled as conspiratorial.

## D.1 Conspiracy Topic Guide

The candidate-retrieval process covers the following con spiracy topics and related narratives:

• Flat-earth claims: The belief that Earth is a flat disc rather than a sphere, and that space agencies, governments, and scientists worldwide are colluding to fake evidence (like satellite images) of a round Earth.

• Rabin assassination: Beyond the established fact that Yigal Amir shot Israeli PM Yitzhak Rabin in 1995, various theories claim the Israeli security service (Shin Bet) orchestrated or deliberately allowed the assassination, sometimes alleging a broader plot involving multiple shooters or government complicity.

• Yemenite children affair: Refers to the disappearance of hundreds of infants, mostly from Yemenite Jewish immigrant families, in Israel during the 1950s. While official inquiries attributed most cases to poor record-keeping and high infant mortality, conspiracy theories allege a systematic, state-orchestrated program to kidnap the children and give them to Ashkenazi families or sell them abroad.

• COVID conspiracy theories: A range of claims including that the virus was deliberately engineered and released (sometimes tied to bio weapons labs), that the pandemic was exaggerated or fabricated to control populations, or that vaccines were designed to cause harm, implant tracking devices, or serve population-control agendas.

• Medical and pharmaceutical control: The belief that pharmaceutical companies, medical associations, and regulators suppress cheap or natural cures (for cancer, etc.) to protect profits from ongoing treatments, and/or that vaccines and medications are pushed for financial or population-control motives rather than genuine health benefits.

• Global elites and the Illuminati: The theory that a secretive group of powerful individuals or families (sometimes named after the historical 18th-century Bavarian Illuminati) secretly controls world governments, economies, and media to advance a hidden agenda, often toward a “New World Order.”

• QAnon: A wide-ranging theory originating in 2017 alleging that a cabal of Satan-worshipping, childtrafficking elites in government, media, and business secretly runs the world, and that a hidden figure (“Q”) was leaking insider information about an anticipated reckoning against them.

• Chemtrails: The claim that visible trails left by aircraft are not ordinary condensation (contrails) but chemical or biological agents deliberately sprayed for purposes like weather control, population control, or geoengineering, hidden from the public.

• 9/11 conspiracy theories: Various claims disputing the official account of the September 11, 2001 attacks, including that the U.S. government had foreknowledge and allowed them to happen (“LIHOP”), directly orchestrated them (“MIHOP”), or that the World Trade Center towers were brought down by controlled demolition rather than the plane impacts and fires.

• Climate change denial/conspiracy: Claims that anthropogenic climate change is a hoax or exaggeration fabricated by scientists, governments, or organizations for financial gain, political control, or to justify regulatory overreach, despite the strong scientific consensus supporting human-caused warming.

• Moon-landing hoax: The claim that NASA’s Apollo moon landings (1969–1972) were staged, often in a film studio, to win the Cold War space race against the Soviet Union, and that supposed anomalies in photos and footage prove fabrication.

• UFOs and aliens: Theories ranging from claims that governments (especially the U.S.) have made contact with or recovered extraterrestrial spacecraft/beings and covered it up, to beliefs that aliens have influenced human history or secretly live among us.

• The “deep state”: The belief that a hidden, unelected network of career bureaucrats, intelligence officials, and other entrenched interests secretly wields real power behind the scenes, undermining or controlling elected officials regardless of who holds office.

## D.2 Retrieval Rules

## General Conspiracy Cues

General conspiracy cues are expressions that may indicate conspiratorial framing across multiple topics.
<table><tr><td>Cue category</td><td>Examples</td></tr><tr><td>Hidden truth and exposure</td><td>,  ,  , ,, ,      </td></tr><tr><td>Concealment and silence</td><td>,&#x27;,&quot;,,&#x27;n, , ,,n,</td></tr><tr><td>Manipulation and deception</td><td>,  ,&quot;,,, ,, , , &#x27;7 D&#x27;TI</td></tr><tr><td>Secret actors and hidden organization</td><td> , , ,n,, ,&#x27; ,&#x27; n , nn ,n&#x27; n</td></tr><tr><td>Institutional distrust</td><td>,&#x27;&#x27;,, , ,,&#x27;,n  ,</td></tr><tr><td>Awakening and warning</td><td>   ,,n,,</td></tr><tr><td>Fraud and staged events</td><td> , ,,7 , ,,,</td></tr><tr><td>Criminal or harmful framing</td><td>,,,,  </td></tr><tr><td>Other conspiratorial cues</td><td>,,,,,,,, ,,</td></tr><tr><td>Political nicknames and coded references</td><td> </td></tr></table>

Table 10: Examples of general conspiracy cues used in candidate retrieval.

## Topic-Specific Retrieval Rules

Each topic contains three components: phrases, keywords, and threshold values. For each topic, a tweet is retrieved when it contains either enough topic-specific phrases or enough topic-specific keywords, and also contains enough general conspiracy cues.

<table><tr><td>Component Values</td><td></td></tr><tr><td>Phrases</td><td>,  , , ,  , , , , , ,  , &#x27;,  ,  ,  ,</td></tr><tr><td>Keywords</td><td>D,, , ,,&#x27; , ,, ,,, ,, ,, &#x27; , , ,&#x27;,&#x27;,,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 11: Retrieval rule for the Flat Earth topic.

Yemenite Children Affair
<table><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>D&quot;0],n &#x27;&#x27; n,7&#x27;n 7,n7 n7,7&#x27;n 1n,D&#x27;&#x27;9n,D&#x27;n D&#x27;&#x27;,&#x27;n &#x27;&#x27; , , ,&quot; ,&quot;</td></tr><tr><td>Keywords</td><td>,&#x27; ,,&#x27; ,,&#x27;,,,&#x27;  ,,&#x27;,&#x27; , , ,n , ,n ,, , , ,,  , , ,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 12: Retrieval rule for the Yemenite Children topic.

Rabin Assassination
<table><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>,  ,  , , ,   ,, 17 0,70 70</td></tr><tr><td>Keywords</td><td>,,,,,,,,,,,, ,,  , ,&#x27;, ,</td></tr><tr><td>Thresholds</td><td>min phrase hits = 1, min keyword hits = 2, min cue hits = 2</td></tr></table>

Table 13: Retrieval rule for the Rabin assassination topic.

Health and COVID Conspiracy
<table><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>big pharma, 5g kills,&#x27;  &#x27; , &#x27;n,&#x27; &#x27;n ,&#x27;n &#x27; , n , n , , q, &#x27;n , &#x27;n  an , ,&quot; &quot;  , &#x27;&#x27;  , &#x27;n, &#x27;, , , , 7&#x27;,5 a ,n</td></tr><tr><td>Keywords</td><td>covid, pfizer, pcr, moderna, plandemic, 5g, vaccine, Soka Gakkai, HIV, chip, mrna, ,7y ,nın &#x27; , , , , , , ,&#x27; ,&#x27; , ,, ,,&#x27; , &quot; ,&#x27; ,&#x27; , ,, 7 ,, &quot;, , n , 5, ,&#x27;,&#x27; &#x27;,&#x27; ,, &#x27; , , , ,, , ,&#x27; 5 ,&#x27; 5</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 14: Retrieval rule for the Health and COVID conspiracy topic.

Illuminati and Global Elites
<table><tr><td colspan="2"></td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>,  ,&#x27;&#x27; &#x27;,  , &#x27; ,  ,&#x27;  ,&#x27;  ,&#x27; &#x27;  ,&#x27; , ,&#x27; , ,&quot; , ,</td></tr><tr><td>Keywords</td><td>&#x27;  illuminati, freemasn, 2030, &#x27;&#x27;n&#x27;, ,,n,,&quot; ,&quot;,7,&#x27; 79 ,1 &#x27;7,00&quot; ,0 ,00 ,00&#x27;17,00&#x27;77 ,&#x27;07,7, ,&#x27;77 ,77,n , ,,,  ,  ,7 ,,, ,&#x27;</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 15: Retrieval rule for the Illuminati and global elites topic.

<table><tr><td colspan="2">QAnon</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>save the children, qanon, WWG1WGA, Q, 4chan, 8chan, q anon, yo ,'7əIT 7 n⊃ ,ı7 " , , , ,, "</td></tr><tr><td>Keywords</td><td>pizzagate, hollywood, CIA, 7', n ,'77 ,n, ,ı' , ,|' ,'n ," ' ,'7' ' , , , ,, ,,, </td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr><tr><td colspan="2">Chemtrails</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>chemtrail, 7"'n' ,07"n"7 ,'n "n ,0 'n ,0' 'n, nn7 ,n '0 "  , , "," ,' ,</td></tr><tr><td>Keywords</td><td>gngineerng, , , ,,", ,  , ,,,,, ,,,, , ,  ,,' ,',,", , , d', Di',,n , ,,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 16: Retrieval rule for the QAnon topic.

Table 17: Retrieval rule for the Chemtrails topic.

<table><tr><td colspan="2">9/11</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>inside b, Lse Change, , &#x27; , 11 ,&#x27;  ,&#x27; &#x27;&#x27; , &#x27; &#x27; ,n 11-, 11,</td></tr><tr><td>Keywords</td><td>9/11, , fbi, ,γ, ,,&quot; ,&quot;,γ,, ,,,&#x27;,&#x27; , ,&quot; , , ,,  , ,,,,, &#x27;&#x27;</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 18: Retrieval rule for the 9/11 topic.

<table><tr><td colspan="2">Climate Hoax</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>t ,  ,&#x27;&#x27;  ,&#x27;  ,     &quot; , , , , ,&#x27;</td></tr><tr><td>Keywords</td><td>2,  ,&#x27; , ,,&#x27; ,&#x27; , ,,&#x27; ,, , ,   ,&#x27; ,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 19: Retrieval rule for the Climate Hoax topic.

<table><tr><td colspan="2">Moon Landing Hoax</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>mon hx, Moned, n n', ,"' ," a' ,'" 7' ,n 7','  n' ,  </td></tr><tr><td>Keywords</td><td>ClA, 7 , ,q,ıi,ı,γ  ,n'77n ,77n ,n'n,',,7i, ,an ,, , , ,,, , ,,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>ufo,  ,n ," , ,  ,51  , ,D'"n n ,'' , n ,n</td></tr><tr><td>Keywords</td><td>rswel, alien, area51, 51, n'n, ,D'υ7" ,',' ,D"n ,D'n ,  ,,'n,77n ' ,n ,n ,υ ,υ" ,D'</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 2, min_cue_hits = 2</td></tr></table>

Table 20: Retrieval rule for the Moon Landing Hoax topic.

Table 21: Retrieval rule for the UFOs and aliens topic.

<table><tr><td colspan="2"></td></tr><tr><td>Component</td><td>Values</td></tr><tr><td>Phrases</td><td>De Sate, &quot; &#x27;, ,&#x27;  &#x27;, , , &#x27;, &#x27; 700 &#x27;,0&quot;00&#x27;</td></tr><tr><td>Keywords</td><td>,,,&#x27;&#x27; ,,,,,,,, , , ,,&#x27; ,,,&#x27; ,,,, ,,,,,,&#x27;,&#x27; ,, ,,, ,, , ,, , ,</td></tr><tr><td>Thresholds</td><td>min_phrase_hits = 1, min_keyword_hits = 3, min_cue_hits = 2</td></tr></table>

Table 22: Retrieval rule for the Deep State topic.