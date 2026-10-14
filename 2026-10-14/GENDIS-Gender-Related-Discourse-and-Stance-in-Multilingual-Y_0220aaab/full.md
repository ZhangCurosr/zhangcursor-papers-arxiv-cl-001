# GENDIS: Gender-Related Discourse and Stance in Multilingual YouTube Comments from India, Kenya, and Nigeria

Luise Koch<sup>1,2</sup>, Manuel Tonneau<sup>1,3,4</sup>, Niyati Malhotra<sup>1</sup>, Sunny Rai<sup>5</sup>, Ana Maria Munoz Boudet<sup>1</sup>, Victor Orozco-Olvera<sup>1</sup>, Samuel P. Fraiberger<sup>1,4</sup>, Sharath Chandra Guntuku<sup>5</sup>

<sup>1</sup>World Bank, <sup>2</sup>Technical University of Munich, <sup>3</sup>University of Oxford, <sup>4</sup>New York University, <sup>5</sup>University of Pennsylvania Correspondence: luise.koch@tum.de

## Abstract

Research on gender-related online content is often operationalized through harm-centered tasks that collapse non-harmful gender-related and unrelated content, while existing datasets remain concentrated in high-resource languages and Global North settings. To bridge these gaps, we introduce GENDIS, a dataset of 9,600 multilingual, human-annotated YouTube comments from India, Kenya, and Nigeria (3,200 per country), annotated using a twolevel framework that first identifies whether gender is referenced and then classifies genderrelated content as negative, positive, or neutral. This separation provides a denominator for gender-related discourse independent of harm, enabling both prevalence estimation and characterization of its evaluative range. In the uniform random stratum, 23.8–35.0% of comments are gender-related, and among those with a resolved stance, 82% are neutral or positive. Across models, gender-reference detection is substantially easier than stance classification, especially for positive and neutral speech. Lightly supervised classifiers match frontier zero-shot large language models at substantially lower cost and outperform generic toxicity detection, while cross-country transfer remains limited. GENDIS provides a locally grounded resource for measuring when gender is invoked online and how that reference is framed.

Content warning: This paper contains examples of gender-related online speech that some readers may find ofensive or distressing, including sexism, misogyny, transphobia, and references to genderbased violence.

## 1 Introduction

Gender is one of the most common lenses through which people talk about one another online, and the way they invoke it runs a wide range: from misogynistic abuse and policing of how women and men should behave, through plain description, to solidarity, praise, and calls for equality. Each end has real stakes. Hostile gendered attention pushes women and gender minorities out of public conversation (Matias, 2019; Koch et al., 2025b), while supportive responses can challenge discriminatory stereotypes, express solidarity, and provide counterspeech (Guest et al., 2021; Mun et al., 2023).

![](images/3d56b0a40f6b8e6ff2a602dd098f895053e9dfe67eee9893ea07fdfe904a490c.jpg)  
Figure 1: GENDIS consists of three annotated datasets (N = 3,200 each) sampled from country-associated channel corpora in India, Kenya, and Nigeria. The dataset is strongly multilingual and code-mixed, with English alongside languages and varieties such as Hindi, Swahili, Sheng, Nigerian Pidgin, Yoruba, Hausa, and Igbo.

Measuring this whole range is therefore necessary not only to identify harmful gender-related speech, but also to characterize what it is being distinguished from: descriptive, supportive, and otherwise non-harmful uses of gender. Similar gendered language can appear in hostile content, neutral description, or counterspeech and solidarity, gender reference, and evaluative stance must be treated as distinct dimensions. Despite this importance, NLP research on gender-related speech faces two limitations. First, gender-related online content has often been operationalized through harm-centered tasks such as sexism and misogyny detection, or within broader hate speech and toxicity frameworks (Abercrombie et al., 2023; Guest et al., 2021; Kirk et al., 2023). In such tasks, nonharmful content is often treated as a broad background class, obscuring the distinction between comments unrelated to gender and comments that invoke gender neutrally or positively. Separating gender reference from evaluative stance therefore makes it possible to measure how often gender is invoked independently of harm and, conditional on that reference, how it is framed. Second, existing resources remain concentrated in a limited number of high-resource languages and Global North settings (Tonneau et al., 2024a), despite emerging work in settings such as Brazil, India, and Nigeria (Arora et al., 2024; Tonneau et al., 2024b; Koch et al., 2025a). This concentration matters because gendered meanings are expressed through locally specific linguistic varieties, slang, codeswitching, and gender norms, and models developed in one national setting often transfer poorly to another (Udupa and Koch, 2024; Maronikolakis et al., 2022).

To fill these gaps, we introduce GENDIS, a twolevel framework and human-annotated evaluation set for gender-related speech in comments from YouTube channels from India, Kenya, and Nigeria (Figure 1). Level 1 of the framework identifies whether a comment is gender-related, and Level 2 classifies its evaluative stance as negative, positive, neutral, or unsure. Using locally reviewed multilingual keyword lists and a random stratum, we sample 9,600 YouTube comments, have them annotated, and evaluate a toxicity baseline, zeroshot large language models, and lightly supervised classifiers on this annotated set at both levels of the framework. We find that detecting genderrelated speech is substantially easier than classifying its evaluative stance, where positive and neutral speech are the hardest categories to detect accurately and remain an open challenge. A lightly supervised classifier trained on country-specific annotated data matches frontier zero-shot large language models at both levels at a fraction of the cost and beats the smaller and open models at Level 2, while a generic toxicity classifier performs poorly. We make three contributions:

1. A two-level annotation framework that first identifies whether gender is referenced and then classifies its evaluative stance, allowing harmful gender-related speech to be distinguished from both gender-unrelated and nonharmful gender-related content.

2. GENDIS, comprising locally reviewed multilingual keyword lists and a human-annotated evaluation set of 9,600 comments from country-associated YouTube channel corpora covering India, Kenya, and Nigeria, together with disaggregated annotations and analyses of annotator disagreement.<sup>1</sup>

3. A benchmark of a toxicity baseline, zero-shot large language models, and lightly supervised classifiers, showing that generic toxicity detection is poorly suited to the task and small amounts of country-specific supervision are highly valuable.

## 2 Related Work

Annotation of gender-related discourse Computational research has largely operationalized gender-related online content through sexism, misogyny, or gender-based violence (Abercrombie et al., 2023; Guest et al., 2021; Kirk et al., 2023). Within this harm-centered literature, annotation schemes distinguish increasingly finegrained forms of hostile and benevolent sexism, pejoratives, misogyny, gendered attacks, and threats (Jha and Mamidi, 2017; Zeinert et al., 2021; Koch et al., 2025a). Some schemes further distinguish direct sexism from reported sexism or responses to it (Chiril et al., 2020; Plaza et al., 2023), while others separately annotate gender bias, or optimistic, pessimistic, and neutral attitudes (Bhattacharya et al., 2020; Kumar et al., 2024; Singh et al., 2025). However, these distinctions remain situated within harm-centered schemes or corpora selected around a predetermined gender-related topic. We are not aware of existing resources that combine a firststage decision about whether a comment invokes gender, regardless of harm, with a subsequent classification of the gender-related content as negative, positive, or neutral.

Geographic skew, transfer, and localization Resources for analyzing socially and culturally situated online discourse remain concentrated in a limited set of languages and national settings (Tonneau et al., 2024a). In response to this imbalance, a growing body of locally grounded harmfullanguage research has incorporated sociocultural knowledge and experience-led annotation in settings including Brazil, India, Kenya, Nigeria, and other African contexts (Maronikolakis et al., 2022; Koch et al., 2025a; Arora et al., 2024; Muhammad et al., 2025). This work demonstrates that linguistic forms, annotation judgments, and model performance are sensitive to local cultural and linguistic conditions. However, most of these resources remain focused on hate, abuse, or genderbased harm, and gender-specific resources are usually limited to a single country or linguistic setting. A shared framework for studying broader genderrelated discourse across multiple Global South contexts, therefore, remains lacking.

<table><tr><td>Dataset</td><td>Platform</td><td>Region</td><td>#Lang</td><td>Gender ref. beyond harm</td><td>Size</td></tr><tr><td>EDOS (Kirk et al., 2023)</td><td>Gab/Reddit</td><td>Unspecified</td><td>1</td><td>No</td><td>20,000</td></tr><tr><td>XtremeSpeech (Maronikolakis et al., 2022)</td><td>Multiple</td><td>BR/DE/IN/KE</td><td>5</td><td>No</td><td>20,297</td></tr><tr><td>NaijaHate (Tonneau et al., 2024b)</td><td>Twitter</td><td>NG</td><td>5</td><td>No</td><td>35,976</td></tr><tr><td>Uli (Arora et al., 2024)</td><td>Twitter</td><td>IN</td><td>3</td><td>No</td><td>23,266</td></tr><tr><td>AfriHate (Muhammad et al., 2025)</td><td>Twitter</td><td>Africa</td><td>15</td><td>No</td><td>90,437</td></tr><tr><td>EXIST 2023 (Plaza et al., 2023)</td><td>Twitter</td><td>Unspecified</td><td>2</td><td>No</td><td>10,034</td></tr><tr><td>Jha and Mamidi (2017)</td><td>Twitter</td><td>Unspecified</td><td>1</td><td>No</td><td>10,095</td></tr><tr><td>Guest et al. (2021)</td><td>Reddit</td><td>Unspecified</td><td>1</td><td>No</td><td>6,567</td></tr><tr><td>Singh et al. (2025)</td><td>YouTube</td><td>IN</td><td>2</td><td>Partial</td><td>12,698</td></tr><tr><td>Bhattacharya et al. (2020)</td><td>YouTube</td><td>IN</td><td>3</td><td>Partial</td><td>12,073</td></tr><tr><td>ComMA (Kumar et al., 2024)</td><td>Multiple</td><td>IN</td><td>4</td><td>Partial</td><td>57,363</td></tr><tr><td>GENDIS (ours)</td><td>YouTube</td><td>IN/KE/NG</td><td>18</td><td>Yes</td><td>9,600</td></tr></table>

Table 1: Prior (gendered) harm datasets and GENDIS compared across platform, region, number of languages, whether gender is referenced beyond harm, and size. GENDIS uniquely treats gender reference beyond harm as a first-class annotation target, capturing negative, positive, and neutral evaluative stance across multiple Global South contexts. Further datasets are discussed in §2.

Gendered harm in YouTube comments YouTube comments are an increasingly common source for studying gendered harm, from misogyny and aggression in Indian languages (Bhattacharya et al., 2020) and optimistic, pessimistic, and neutral attitudes in Hindi–English (Singh et al., 2025) to two-level sexism schemes for Turkish (Mut Altin and Saggion, 2024) and Italian femicide-related comments, where implicit misogyny is hardest to classify (Ferrando et al., 2024).

These resources focus on sexism or misogyny within a single linguistic or national setting. We extend them with a shared two-level framework across India, Kenya, and Nigeria that treats negative, positive, and neutral gender-related speech as substantive categories rather than defining the task around harm. Table 1 contrasts GENDIS with these prior resources.

## 3 The GENDIS Dataset

## 3.1 Large-scale YouTube corpus

We used a YouTube corpus constructed from lists obtained through a paid Social Blade subscription<sup>2</sup>. For each country of interest, namely India, Kenya, and Nigeria, we collected the 5,000 YouTube channels with the highest subscriber counts according to Social Blade at the time of extraction. We subsequently collected publicly available video metadata and comments associated with these channels through YouTube’s Data API v3 during the time period ranging from May 2025 to February 2026. The resulting corpus comprises approximately 11 million videos and 1.7 billion comments. It is heavily skewed toward India, which contributes approximately 1.6 billion comments, compared with 85 million for Kenya and 80 million for Nigeria.

## 3.2 Keyword list construction

Seed lists To identify comments likely to contain gender-related speech, we constructed countryspecific multilingual keyword lists. We reviewed gender-specific datasets and annotation frameworks covering hostile and benevolent sexism (Jha and Mamidi, 2017), misogynistic language and counterspeech (Guest et al., 2021; Zeinert et al., 2021), fine-grained sexism (Kirk et al., 2023), gendered aggression and gender bias (Bhattacharya et al., 2020; Arora et al., 2024; Kumar et al., 2024), and optimistic, pessimistic, and neutral attitudes toward women (Singh et al., 2025). These resources informed seed categories covering gender-group terms and pronouns, femininity and masculinity, sexuality and bodies, gender roles and behavioral expectations, gendered slurs and abuse, benevolent sexism, equality and supportive language, counterspeech, hashtags, abbreviations, and other genderrelevant expressions.

Expansion and expert review The seed lists were expanded in November 2025 using Grok 4 (see full prompt in Appendix B.1). One author first back-checked the Grok-generated candidates for duplicates, irrelevant terms, and variants that did not refer to gender. Three country-context reviewers per country then reviewed the lists following written guidelines (Appendix B.2). Reviewers assessed each term for local relevance and contextual ambiguity, checked translations and language or dialect labels, and identified formal, slang, dialectal, and code-switched usage. They also proposed missing terms and flagged underrepresented forms of gender-related discourse, including vocabulary specific to social groups or online platforms.

Final lists The final keyword lists comprise 1,530 entries across India (778), Nigeria (386), and Kenya (366). They cover English, relevant local languages and linguistic varieties, code-mixed expressions, and YouTube-specific forms such as hashtags, abbreviations, and emojis. Appendix Table 4 summarizes the principal languages and varieties represented in each list. Of the 1,530 list entries, 776 were observed in the collected YouTube comments (498 in India, 137 in Kenya, and 141 in Nigeria).

## 3.3 Sampling

For each country, we constructed a sample of 3,200 unique comments in three strata: 1,500 drawn through the country-specific keyword list, 1,500 drawn uniformly at random from the full corpus with no keyword condition, and 200 drawn from comments matching none of the terms in the keyword list.

Keyword stratum The first stratum was designed to maximize the number of comments likely to contain gender-related speech. For each keyword, we sampled up to 20 comments, retaining all available comments when fewer than 20 were present. We then pooled the keyword-specific samples and deduplicated them by comment ID, so that a comment matching several terms appeared only once. All matched terms were retained in the sampling metadata.

Random stratum We drew 1,500 comments per country uniformly at random from the full corpus, applying no keyword condition and no other filtering. This keyword-independent stratum avoids keyword-selection bias (Wiegand et al., 2019; Ousidhoum et al., 2020) and supports an unbiased estimate of prevalence and of real-world performance, testing whether models recognize gender-related speech rather than reproducing the retrieval terms.

Keyword-free stratum A third stratum of 200 comments per country was sampled from comments matching no term in the keyword list. It gives annotated coverage of the portion of the corpus the keyword lists do not reach, and tests models on comments that share no lexical overlap with the retrieval terms. Because it describes the keywordfree subpopulation rather than the corpus, we retain it in the dataset and use it in all model evaluation, but exclude it from prevalence estimation, which rests on the uniform random stratum alone.

## 3.4 Annotation

Annotation team We recruited nine annotators, three per country, through professional networks in India, Kenya, and Nigeria, forming panels with relevant country-specific linguistic, cultural, and social-media knowledge. Annotator demographics, compensation, and linguistic backgrounds are reported in Appendix D.

Annotation task The task follows a two-level structure (Figure 2). At Level 1, annotators determine whether gender is referenced, explicitly or implicitly. Relevant cues include gendered pronouns and categories, appearance and body references, romantic or family roles, stereotyped femininity or masculinity, gendered slurs, and culturally specific or implied gender meanings. A general insult is not gender-related unless gender is invoked, and if no gender reference is identified, annotation stops at Level 1. At Level 2, for gender-related comments, annotators judge the evaluative stance, the way a comment treats gender rather than its surface tone (see Figure 2 and Appendix A.1 for examples). Comments are labeled negative (harmful, stereotyping, objectifying, or violent uses, including benevolent sexism and restrictive gender norms), positive (afirming or empowering a gender group or promoting equality), neutral (descriptive, non-evaluative references, such as identifying “the woman in blue”), or unsure (gender is present but its stance cannot be determined confidently). Because stance is not tone, benevolent sexism counts as negative despite its positive tone, and where positive and negative content co-occur, the negative label takes precedence.

![](images/8225c07a2cb8e45efb5a50ab1f4ae51d62f293220f16014f2b787f4669dcc552.jpg)  
Figure 2: The two-level annotation scheme, with illustrative example comments in the shaded boxes. Level 1 detects whether a comment is gender-related, sending comments with no gender reference (such as general insults) to a leaf and gender-related ones to Level 2, which labels how gender is used. Further examples per category and country are in Appendix A.1.

Procedure Each of the 9,600 comments was labeled independently by three annotators in the relevant country panel. Annotators were shown only the text of the sampled comment; video titles, descriptions, surrounding comments, and other contextual metadata were not provided. The guidelines and annotation interface were piloted on a preliminary batch and revised before annotation began, and the task carried a content warning because comments could contain sexism, misogyny, transphobia, sexualized language, and gender-based violence. Annotation was primarily prescriptive: annotators applied the shared definitions, examples, and decision rules of the guidelines (Appendix A) rather than recording purely personal reactions (Röttger et al., 2022). Because annotator identities, beliefs, and social positions can nonetheless afect judgments of gender norms and misogyny (Sap et al., 2022; Mostafazadeh Davani et al., 2022; Jiang et al., 2024), we retain the individual labels and analyze disagreement rather than treating it solely as error.

Label aggregation At Level 1, the aggregated label is the category selected by at least two of the three annotators. Level 2 ratings are applicable only for annotators who classified the comment as gender-related at Level 1. For a comment with a gender-related Level 1 majority, a resolved Level 2 label requires at least two applicable ratings agreeing on gender-negative, gender-positive, or genderneutral. Among comments with a gender-related Level 1 majority, 17 had at least two applicable annotators select unsure at Level 2 and are reported separately. For the remaining 362 comments, no two applicable annotators agreed on a Level 2 category; we retain these as no Level 2 majority (Appendix Table 7). Neither group serves as a reference class in the benchmark.

Inter-annotator agreement We calculate interannotator agreement from the individual annotations before majority aggregation, over all 3,200 comments per country. Agreement ranges from moderate to substantial and varies by country and annotation level. At Level 1, Krippendorf’s α is .80 in India, .79 in Kenya, and .54 in Nigeria. At Level 2, where annotators classify the nature of gendered speech, α is .83 in India, .72 in Kenya, and .49 in Nigeria. For the combined final label, α is .77 in India, .72 in Kenya, and .47 in Nigeria. Fleiss’ κ gives similar conclusions; full figures are reported in Appendix Table 8. Level 2 agreement is lower because the task requires annotators not only to identify gender reference, but also to annotate whether the gendered content is negative, positive or neutral.

Disagreement patterns Disagreement is concentrated at Level 2 rather than in the basic identification of gender-related speech. Across the annotation, 362 comments received a gender-related Level 1 majority, but no Level 2 majority, meaning that at least two annotators agreed that the comment was gender-related, but no two applicable Level 2 judgements converged on the same negative, positive, or neutral label. This pattern is highly uneven across countries: Nigeria accounts for 306 of these cases, or 18.9% of its 1,623 genderrelated comments, compared with 14 cases in India (1.0% of 1,470) and 42 in Kenya (3.1% of 1,350). These splits highlight the interpretive dificulty of Level 2. In many cases, annotators agree that a gendered cue is present but difer over whether the reference is merely descriptive or whether it carries an evaluative, supportive, stereotyping, or derogatory stance. Examples include short, code-mixed, or culturally specific comments built around gendered praise, teasing, or proverbial expressions, such as nwanyi oma (Igbo, “good woman”), “big boy”, or “man must hustle”. Such comments can read as a description to some annotators and as an afirmation, stereotype, or normative judgement to others. We therefore treat these cases not simply as annotation noise but as evidence of the normative ambiguity involved in classifying the nature of gendered speech. Following our conservative aggregation rule, comments without a Level 2 majority are assigned to the final unsure category, while a separate provenance flag distinguishes them from comments for which annotators explicitly reached a majority unsure label.

## 3.5 Descriptive statistics

Label distribution Across the evaluation set, 4,443 of 9,600 comments (46%) receive a genderrelated Level 1 majority, ranging from 42% in Kenya to 51% in Nigeria (Appendix Figure 3). Among the resolved Level 2 categories, gendernegative is the largest, with 1,868 comments, followed by gender-neutral with 1,621 and genderpositive with 575. Gender-neutral comments are particularly common in Kenya, where they clearly outnumber gender-negative comments (680 versus 503). Nigeria contains both the largest number of gender-negative comments (734) and the largest number of comments with a gender-related Level 1 majority but no Level 2 majority (306). Because the set is keyword-enriched by construction, these shares characterize the annotated sample rather than the prevalence of gender-related speech in the corpus, which we estimate from the random stratum below.

Prevalence We use the uniform random stratum to estimate the prevalence of gender-related content in comments to top YouTubers per country. About three in ten comments are gender-related, from 23.8% in Kenya (95% CI 21.7–26.0) through 30.3% in India (28.1–32.7) to 35.0% in Nigeria (32.6–37.4). By comparison, the keywordenriched stratum is gender-related at 66%, more than twice as high, confirming that the lists greatly increase collection eficiency. Among the random comments with a resolved Level 2 majority (1,226 of 4,500), 67% are neutral, 18% negative, and 16% positive, so most gender-related speech in this keyword-independent sample is not abusive, the very pattern a harm-only lens misses.

Language composition To document the linguistic makeup of the annotation sample, we labeled the language of every comment with a large language model (gpt-5-nano, audited in Appendix C), recording which languages from a country-specific list appear and whether two or more are mixed in the same comment (Table 5). All three sets are strongly multilingual. Codemixing is pervasive, afecting 32% of comments in India, 40% in Kenya, and 30% in Nigeria. English is widespread, appearing in 58% of Indian, 67% of Kenyan, and 82% of Nigerian comments, alongside major local languages such as Hindi (46% of Indian comments), Swahili (51% of Kenyan comments), and Nigerian Pidgin (18% of Nigerian comments). A further 6 to 12% of comments are in languages outside each country’s list, reflecting the international audiences of large channels. This diversity confirms that English-centric resources would miss much of the gender-related speech in these settings and motivates the locally grounded lists described above.

Gender-positive discourse To characterize what the positive class contains beyond its aggregate frequency, we conducted an exploratory qualitative analysis of all 575 majority-labeled gender-positive comments. We grouped recurring forms and linguistic patterns descriptively rather than introducing a second annotation scheme; the themes were not independently coded and we therefore do not report inter-annotator agreement or category frequencies. Recurring forms include support for gender equality, safety, and anti-GBV eforts; praise of women’s strength, competence, achievement, and leadership; positive constructions of masculinity emphasizing respect, care, and responsibility; and everyday gendered afirmation. Illustrative country-specific patterns include empowerment and rights discourse in Indian comments, with expressions such as nari shakti (‘women’s power’); Kenyan comments combining expressions such as mwanamke bomba (‘an impressive woman’) and mama wa maendeleo (‘woman of development’) with anti-femicide advocacy and positive constructions of masculinity; and Nigerian comments using vernacular praise forms such as woman wey sabi (‘a capable woman’), nwanyi oma (‘good woman’), and correct guy (‘a good/solid guy’). Positive comments can also refer to harms such as violence, femicide, or abuse while condemning them, showing why positive stance cannot be reduced to positive surface sentiment.

## 4 Experiments

We benchmark a range of models against the human annotations: each reads a comment and predicts whether it is gender-related (Level 1) and, for gender-related comments, its evaluative stance (Level 2). All models are scored on the majority labels over the 9,600 comments from India, Kenya, and Nigeria, letting us compare modeling strategies with very diferent data and compute requirements.

## 4.1 Models

We benchmark seven models: a training-free toxicity baseline, two supervised classifiers, and four large language models in a zero-shot fashion.

Toxicity baseline A toxicity baseline maps the score of an of-theshelf multilingual toxicity classifier (unitary/multilingual-toxic-xlm-rob onto our labels, treating high-toxicity comments as gender-related and gender-negative, which tests whether gender-related speech reduces to toxicity.

Supervised We train two supervised classifiers with a single stratified five-fold partition by country and label, reporting out-of-fold predictions so that every comment is scored by a model that never saw it. We fine-tune XLM-R (Conneau et al., 2020), the standard multilingual classification baseline, and train SetFit (Tunstall et al., 2022), a sampleeficient few-shot method that contrastively finetunes a sentence-transformer on same- versus diferent-label pairs and fits a lightweight classifier on the resulting embeddings, which suits our small data and rare positive and neutral classes. We evaluate three training regimes: within-country (train and test within each country’s folds), pooled crossvalidation (CV, across all three countries with country-stratified folds), and leave-one-countryout (LOCO, train on two countries and test on the held-out third) to measure cross-cultural transfer.

Zero-shot LLMs We prompt four large language models, two open and two closed, with the label definitions and a fixed output schema. The open models are the instruction-tuned decoders Llama-3.1-8B and Gemma-2-9B, widely used open-weight reference models at a size that runs on a single GPU and can be deployed locally.

The closed models are gpt-5-nano, a small lowcost model of the kind most likely to be deployed at scale, and gpt-5.4, a current frontier model included as a state-of-the-art reference. The prompt was fixed in advance with no prompt search, so the zero-shot scores reflect out-of-the-box behavior.

## 4.2 Evaluation

Metrics All models are scored on the same instances. Macro-F1 is the headline metric given the class imbalance, reported overall and per country. We evaluate Level 1 (gender-related or not, over all comments), Level 2 (negative, positive, or neutral, conditioned on gold gender-related comments so that Level 1 errors do not leak in), and an endto-end four-way score (E2E) over the not genderrelated, negative, positive, and neutral classes.

## 5 Results

We report the results of our experiments in Table 2 and Appendix G. All scores are macro-F1 unless stated otherwise.

Detection easier than stance Across every model, macro-F1 is far higher on Level 1 detection than on Level 2 stance classification. SetFit, the strongest model at Level 2, reaches 0.83 on Level 1 but 0.67 on Level 2. Within Level 2, the relative dificulty of the three classes is the same for every model. Every model handles the gendernegative class best and diverges on the other two, with SetFit the only model whose per-class F1 clears 0.55 on all three, and in the end-to-end setting, gender-neutral is the hardest class (per-class F1 0.50 for SetFit CV). This pattern follows the data. Detecting a gender reference can lean on surface cues such as gender-related terms and pronouns, whereas stance must be inferred from context. Neutral, descriptive references sit close to both the not gender-related and gender-negative classes and are most often confused with them (Appendix Figure 4), while gender-positive comments lack a lexical signature of their own and share the women’s-rights and safety vocabulary of the negative pole, so supportive comments are most often misread as negative rather than recognized as affirming.

Supervision rivals frontier models The model families separate cleanly, with supervised classifiers matching or beating the zero-shot large language models, which in turn beat the toxicity baseline. SetFit achieves the best macro-F1 at Level 2, within-country or pooled. It ties the zero-shot models on Level 1, and on Level 2 improves over gpt-5-nano by eleven points (0.67 F1 versus 0.56) and over the open decoders by more than twentyfive. Only the frontier gpt-5.4 comes close on this Level 2 gap, reaching 0.64, but it is a large proprietary model billed per call, whereas SetFit trains on country-specific labeled data and runs locally, so a lightly supervised contrastive model matches or beats a frontier zero-shot model at a small fraction of the cost. XLM-R is competitive on Level 1 (0.82) but falls seven points on Level 2, consistent with SetFit being the stronger choice for the rare positive and neutral classes. Among the open decoders, Gemma-2-9B roughly matches gpt-5-nano on Level 1 (0.79 versus 0.81), and Llama-3.1-8B trails only slightly (0.76), so a 9B open model is competitive at detecting gender-related speech, but both fall well behind on Level 2 (0.41 versus 0.56 for gpt-5-nano) and collapse specifically on the neutral class (per-class F1 0.17 and 0.28), plausibly because harm-oriented instruction tuning biases them toward an evaluative reading. The toxicity baseline is weakest of all, lowest on Level 1 (0.46) and never predicting the positive class. Built to fire on abuse, it recovers part of the negative pole but misses neutral and positive gender-related speech, direct evidence that gender-related speech is not reducible to toxicity.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Baseline</td><td colspan="2">Supervised (CV)</td><td colspan="4">Zero-shot LLMs</td></tr><tr><td>Toxicity XLM-R</td><td>SetFit</td><td>Llama-3.1-8B</td><td>Gemma-2-9B</td><td>gpt-5-nano</td><td>gpt-5.4</td></tr><tr><td colspan="2">Level 1 (detection)</td><td>.46</td><td>.82</td><td>.83</td><td>.76</td><td>.79</td><td>.81</td><td>.84</td></tr><tr><td rowspan="4">Level 2</td><td>negative</td><td>.35</td><td>.71</td><td>.76</td><td>.70</td><td>.71</td><td>.71</td><td>.77</td></tr><tr><td>neutral</td><td>.59</td><td>.55</td><td>.66</td><td>.28</td><td>.17</td><td>.45</td><td>.65</td></tr><tr><td>positive</td><td>.00</td><td>.56</td><td>.60</td><td>.24</td><td>.34</td><td>.53</td><td>.50</td></tr><tr><td>macro</td><td>.31</td><td>.61</td><td>.67</td><td>.41</td><td>.41</td><td>.56</td><td>.64</td></tr><tr><td>End-to-end (4-way)</td><td></td><td>.26</td><td>.58</td><td>.63</td><td>.49</td><td>.47</td><td>.61</td><td>.64</td></tr></table>

Table 2: F1 on the same N = 9,600 comments for every system, using out-of-fold predictions for the supervised models. Level 1 and End-to-end report macro-F1. The Level 2 rows report per-class F1 and their macro-average, conditioned on gold gender-related comments. Supervised models use pooled cross-validation (CV), with other regimes in Table 10. Best per row in bold.

Country and language disparities Model performance is uneven across countries and languages, and the two axes reinforce each other. Per-country scores (Appendix Table 9) do not track annotation agreement: SetFit leads gpt-5-nano on Level 2 in every country, by the widest margin in Nigeria (0.68 versus 0.57), and Nigeria, despite the lowest inter-annotator agreement, has the highest SetFit scores, which we read cautiously because its macro is dominated by the more separable not genderrelated and negative comments. Cross-country transfer is the main bottleneck. Within-country training matches pooling all three countries (Set-Fit within two points on every country), but leaveone-country-out is far worse (0.75 on Level 1, 0.47 on Level 2) and steepest when India is held out (Level 1 0.69). Since a within-country model sees less data than a leave-one-country-out model yet outperforms it, the degradation reflects countryspecific distribution shift rather than training-set size (Appendix Table 10). The same gap appears across languages (Appendix I): every model scores highest on English and lower on local-language, code-mixed, and out-of-list comments, widening from Level 1 to Level 2 (for gpt-5-nano, 0.87 on English falling to 0.72–0.76 elsewhere at Level 1, and 0.62 to 0.36 at Level 2). This is not a country artefact, as English outscores non-English within every country. Here too, local supervision helps most, as SetFit is the most robust on out-of-list content (Level 2 0.61 versus 0.36 for gpt-5-nano). Because the language labels are automatic and the smallest group is small, we read these breakdowns as descriptive.

Real-world performance Because the headline scores are dominated by the keyword-enriched stratum, we re-score every model on the random stratum alone, drawn uniformly from the corpus regardless of keyword match, which better approximates real-world deployment (Appendix H, Table 11). Level 1 macro-F1 stays within a point of the full evaluation set for the learned models (SetFit 0.84, gpt-5.4 0.84, gpt-5-nano 0.80), so detecting gender-related speech is not an artefact of the retrieval terms. The toxicity baseline instead collapses to 0.47 and recovers only 7% of gender-related comments, sharpening the point that gender-related speech is not reducible to toxicity. With round-2 data, Level 2 is reportable on this sample for the first time (n = 1,226, up from 94): SetFit keeps its Level 1 lead and widens it at Level 2 (0.64 versus 0.53 for gpt-5.4), so its stanceclassification advantage over the frontier zero-shot model is, if anything, larger on this more representative sample.

## 6 Discussion

Gender-related speech is common and often non-negative Sampling independently of our keyword lists, we find that in the uniform random stratum about three in ten comments are genderrelated, from 23.8% in Kenya to 35.0% in Nigeria, and that most of it is non-negative: among gender-related comments with a resolved stance, 67% are neutral and 16% positive. This distribution is not characterized by a harm-centered operationalization (Abercrombie et al., 2023; Guest et al., 2021; Kirk et al., 2023), and our toxicity baseline bears it out, recovering part of the negative pole but none of the neutral or positive content that dominates. Still, the negative pole is far from marginal: gender-negative speech runs from 2.5% of all comments in India to 9.2% in Nigeria in this keyword-independent sample. This exceeds the hate speech prevalence that Tonneau et al. (2025) find on a representative day of Twitter, 0.7% on average and only 0.2% and 0.3% in Kenya and Nigeria (1.7% in India), and most sharply so in Nigeria. Although the platforms difer, the gap suggests that comment sections beneath popular channels concentrate gendered hostility well beyond a platformwide sample.

Light local supervision rivals frontier zeroshot models A contrastive classifier trained on country-specific labeled data matches the strongest zero-shot large language model at both detecting gender-related speech and judging its stance, and beats the open models by more than twenty points at Level 2, while running locally at a small fraction of the cost. This echoes evidence that lightweight supervised models remain competitive with, or better than, zero-shot large language models for hate speech and sexism detection (Tonneau et al., 2025; Plaza-del arco et al., 2023), and the advantage is largest on the rare positive and neutral classes, where general-purpose instruction tuning appears to bias models toward a harm-oriented reading. The qualitative analysis helps explain this dificulty: positive stance ranges from locally specific forms of gendered praise to equality advocacy and anti-GBV solidarity, and the latter often contains the same harm-related vocabulary as negative comments. For a task whose labels are few and locally specific, task-specific supervision is not a fallback but the more accurate and cheaper option wherever per-country data can be collected. Even so, stance classification remains harder than detection for every model, with positive and neutral speech the hardest categories, marking where GENDIS leaves the most room for future work.

Gendered meaning is local, and transfer is the bottleneck The cost of supervision is that it does not travel. Leave-one-country-out training is far worse than within-country or pooled training even though it uses more data, and every model is strongest on English and weaker on local-language and code-mixed comments, with the gap widening from detection to stance. Because a within-country model outperforms a larger leave-one-country-out one, the degradation reflects country-specific distribution shift rather than data volume. This is consistent with evidence that gendered and harmful language is carried by locally specific varieties, slang, and norms that transfer poorly across settings (Tonneau et al., 2024a; Maronikolakis et al., 2022; Koch et al., 2025a), and it is why we ground GENDIS in three Global South contexts with locally reviewed lists and locally informed annotation rather than porting resources built elsewhere. Measuring gender-related speech across its full range and across settings will require locally grounded data in each new context, not a single model assumed to generalize.

## 7 Conclusion

We introduced GENDIS, a two-level framework and human-annotated evaluation set of 9,600 comments from India, Kenya, and Nigeria that separates whether a comment is gender-related from whether the gender-related content is negative, positive, or neutral, treating these as distinct categories rather than collapsing gender into abuse. Agreement varies substantially across the three contexts, and in our benchmark, a contrastive model trained on country-specific labeled data matches frontier zero-shot models at both detecting gender-related speech and classifying it, at a fraction of the cost, while generic toxicity detection fails on both. Much gender-related speech is not abusive, underscoring the need to measure its full evaluative range rather than only its harmful pole. We release GENDIS as a foundation for measuring gender-related speech beyond explicit abuse across linguistically and culturally diverse settings.

## Limitations

The keyword-enriched portion of the evaluation set is built to concentrate gender-related discourse and is not a prevalence estimate: its label distribution describes the annotated sample, not genderrelated discourse in the wider corpus. Only the uniform random stratum supports prevalence estimation, and these estimates remain bounded by the Social Blade sampling frame: country assignments do not verify creator nationality, rankings are time-specific, and highly subscribed channels may include organizations, group channels, and international audiences. They therefore characterize the sampled country-associated channel corpora rather than national populations or YouTube users more broadly. The 200 comments per country drawn from keyword-free comments describe that subpopulation rather than the corpus, so we exclude them from prevalence and report them separately.

Each comment was labeled by three annotators from the relevant country panel. Because these small panels were recruited through professional networks, they may be subject to selection bias and cannot be assumed to represent their national contexts; their size also prevents us from attributing disagreement to particular demographic characteristics. Moreover, the majority labels represent the aggregated judgments of these particular panels under our operational guidelines, not a universal or culturally definitive account of whether genderrelated content is positive, negative, or neutral. Other appropriately informed panels could reasonably produce diferent label distributions, especially for implicit or normatively contested cases.

The study covers three countries and a single platform, so transfer to other regions, languages, or platforms is untested, and our leave-one-countryout results indicate that transfer is already limited across the three countries studied. The country panels did not cover every language and variety present in the sample; although annotators could consult translation tools and flag uncertain cases, linguistic nuance may therefore be lost, particularly in slang and code-mixed content.

The computational benchmark scores models against aggregated majority labels and excludes comments without a resolved Level 2 label, so it under-represents the most contested cases, which are precisely where measurement is hardest. Finally, we evaluate a single prompt per zero-shot model and one configuration per supervised model, so we do not characterize the prompt and hyperparameter sensitivity that prior work has shown can be substantial.

## Ethics Statement

The study did not undergo formal institutional ethics review. We nevertheless adopted dataminimization, annotator-welfare, and reporting safeguards appropriate to the potentially harmful and identifiable nature of the material. The study uses comments posted on publicly accessible YouTube pages, but public visibility is not treated as equivalent to informed consent or ethical exemption (Franzke et al., 2020). We reproduce a limited number of short verbatim comments solely to illustrate the annotation categories. Because such comments may remain searchable, we omit usernames, channel and video identifiers, direct links, comment IDs, and other identifying metadata, and avoid reproducing exceptionally sensitive, threatening, or personally identifying material. No such identifiers are included in the illustrative tables.

Annotators were informed that the task could involve sexism, misogyny, transphobia, sexualized language, and references to gender-based violence. They received a content warning, could take breaks, use the unsure category, and contact the research team about dificult or distressing cases. These safeguards recognize that repeated exposure to harmful content can entail emotional labor and psychological burden for annotators and researchers (AlEmadi and Zaghouani, 2024). Annotators were not required to disclose personal experiences of gender-based harm.

Gender-related speech is culturally and linguistically situated. We recruited annotators with relevant country and language knowledge and collected optional background information to document the perspectives represented in the annotation process. Because the country panels are small, sensitive characteristics are reported only in aggregate and are not used to evaluate individual annotators.

Disagreement is retained and analyzed rather than treated solely as error, but we do not infer that particular demographic characteristics caused particular labels (Sap et al., 2022; Mostafazadeh Davani et al., 2022). The authors’ own positionality with respect to the settings studied is described in Appendix E.

The annotated data and keyword lists contain harmful terms and potentially identifying usergenerated content. The release of keyword lists will include documentation and a content warning. Raw usernames, direct links, and other identifiers will not be released. The resource is intended for research, measurement, and evaluation. It should not be interpreted as a ready-made system for fully automated moderation, surveillance, or censorship.

Finally, we recognize that research on gendered online abuse can reproduce harm if it extracts from afected communities without accountability. Our analysis therefore focuses on structural patterns, annotation challenges, and measurement limitations rather than sensational examples. The broader aim is to support more accurate and context-sensitive understanding of gendered online harm while acknowledging the limits of computational classification and the responsibilities involved in studying harmful digital content.

## Acknowledgments

We thank our annotators in India, Kenya, and Nigeria, whose careful work made this dataset possible, and the anonymous reviewers for their constructive feedback.

The study was supported by funding from the Gates Foundation (INV057844), the Penn Global Research Engagement Fund, and Penn MEDI-ATED. The findings, interpretations, and conclusions expressed in this article are those of the authors alone and shall not be attributed to the Gates Foundation. They also do not necessarily represent the views of the International Bank for Reconstruction and Development/World Bank and its affiliated organizations or those of the Executive Directors of the World Bank or the governments they represent.

## References

Gavin Abercrombie, Aiqi Jiang, Poppy Gerrard-abbott, Ioannis Konstas, and Verena Rieser. 2023. Resources for automated identification of online gender-

based violence: A systematic review. In The 7th Workshop on Online Abuse and Harms (WOAH), pages 170–186, Toronto, Canada. Association for Computational Linguistics.

Maryam M. AlEmadi and Wajdi Zaghouani. 2024. Emotional toll and coping strategies: Navigating the efects of annotating hate speech data. In Proceedings ofthe Workshop on Legal and Ethical Issues in Human Language Technologies @ LREC-COLING 2024.

Arnav Arora, Maha Jinadoss, Cheshta Arora, Denny George, Brindaalakshmi, Haseena Khan, Kirti Rawat, Div, Ritash, and Seema Mathur. 2024. The Uli dataset: An exercise in experience led annotation of oGBV. In Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024), pages 212– 222, Mexico City, Mexico. Association for Computational Linguistics.

Shiladitya Bhattacharya, Siddharth Singh, Ritesh Kumar, Akanksha Bansal, Akash Bhagat, Yogesh Dawer, Bornini Lahiri, and Atul Kr. Ojha. 2020. Developing a multilingual annotated corpus of misogyny and aggression. In Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying, pages 158–168, Marseille, France. European Language Resources Association (ELRA).

Patricia Chiril, Véronique Moriceau, Farah Benamara, Alda Mari, Gloria Origgi, and Marlène Coulomb-Gully. 2020. An annotated corpus for sexism detection in French tweets. In Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020). European Language Resources Association.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Chiara Ferrando, Marco Madeddu, Viviana Patti, Mirko Lai, Sveva Pasini, Giulia Telari, and Beatrice Antola. 2024. Exploring YouTube comments reacting to femicide news in Italian. In Proceedings of the Tenth Italian Conference on Computational Linguistics (CLiC-it 2024), pages 356–365, Pisa, Italy. CEUR Workshop Proceedings.

Aline Shakti Franzke, Anja Bechmann, Michael Zimmer, and Charles M. Ess. 2020. Internet research: Ethical guidelines 3.0. Technical report, Association of Internet Researchers.

Ella Guest, Bertie Vidgen, Alexandros Mittos, Nishanth Sastry, Gareth Tyson, and Helen Margetts. 2021. An expert annotated dataset for the detection of online misogyny. In Proceedings of the 16th Conference

ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 1336– 1350, Online. Association for Computational Linguistics.

Akshita Jha and Radhika Mamidi. 2017. When does a compliment become sexist? analysis and classification of ambivalent sexism using twitter data. In Proceedings of the Second Workshop on NLP and Computational Social Science, pages 7–16, Vancouver, Canada. Association for Computational Linguistics.

Aiqi Jiang, Nikolas Vitsakis, Tanvi Dinkar, Gavin Abercrombie, and Ioannis Konstas. 2024. Re-examining sexism and misogyny classification with annotator attitudes. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15103– 15125, Miami, Florida, USA. Association for Computational Linguistics.

Hannah Kirk, Wenjie Yin, Bertie Vidgen, and Paul Röttger. 2023. SemEval-2023 task 10: Explainable detection of online sexism. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2193–2210, Toronto, Canada. Association for Computational Linguistics.

Luise Koch, Raji Ghawi, Jürgen Pfefer, and Janina Isabel Steinert. 2025a. Online misogyny against female candidates in the 2022 brazilian elections: A threat to women’s political representation? Information, Communication & Society, 29(7):2104–2124.

Luise Koch, Maria Paula Russo Riva, and Janina Isabel Steinert. 2025b. Technology-facilitated genderbased violence against politically active women: A systematic review of psychological and political consequences and women’s coping behaviors. Trauma, Violence, & Abuse. Advance online publication.

Ritesh Kumar, Shyam Ratan, Siddharth Singh, Enakshi Nandi, Laishram Niranjana Devi, Akash Bhagat, Yogesh Dawer, Bornini Lahiri, and Akanksha Bansal. 2024. A multilingual, multimodal dataset of aggression and bias: the ComMA dataset. Language Resources and Evaluation, 58(2):757–837.

Antonis Maronikolakis, Axel Wisiorek, Leah Nann, Haris Jabbar, Sahana Udupa, and Hinrich Schuetze. 2022. Listening to afected communities to define extreme speech: Dataset and experiments. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1089–1104, Dublin, Ireland. Association for Computational Linguistics.

J. Nathan Matias. 2019. Preventing harassment and increasing group participation through social norms in 2,190 online science discussions. Proceedings ofthe National Academy ofSciences, 116(20):9785–9789.

Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. 2022. Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions ofthe Associationfor Computational Linguistics, 10:92–110.

Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Abinew Ali Ayele, David Ifeoluwa Adelani, and 1 others. 2025. AfriHate: A multilingual collection of hate speech and abusive language datasets for African languages. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1854–1871, Albuquerque, New Mexico. Association for Computational Linguistics.

Jimin Mun, Emily Allaway, Akhila Yerukola, Laura Vianna, Sarah-Jane Leslie, and Maarten Sap. 2023. Beyond denouncing hate: Strategies for countering implied biases and stereotypes in language. In Findings of the Association for Computational Linguistics: EMNLP 2023. Association for Computational Linguistics.

Lutfiye Seda Mut Altin and Horacio Saggion. 2024. A novel corpus for automated sexism identification on social media. In Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages @ LREC-COLING 2024, pages 10–15, Torino, Italia. ELRA and ICCL.

Nedjma Ousidhoum, Yangqiu Song, and Dit-Yan Yeung. 2020. Comparative evaluation of label-agnostic selection bias in multilingual hate speech datasets. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2532–2542, Online. Association for Computational Linguistics.

Laura Plaza, Jorge Carrillo-de Albornoz, Roser Morante, Enrique Amigó, Julio Gonzalo, Damiano Spina, and Paolo Rosso. 2023. Overview of EXIST 2023 – learning with disagreement for sexism identification and characterization. In Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2023), volume 14163 of Lecture Notes in Computer Science, pages 316–342. Springer.

Flor Miriam Plaza-del arco, Debora Nozza, and Dirk Hovy. 2023. Respectful or toxic? using zero-shot learning with language models to detect hate speech. In The 7th Workshop on Online Abuse and Harms (WOAH), pages 60–68, Toronto, Canada. Association for Computational Linguistics.

Paul Röttger, Bertie Vidgen, Dirk Hovy, and Janet Pierrehumbert. 2022. Two contrasting data annotation paradigms for subjective NLP tasks. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 175– 190, Seattle, United States. Association for Computational Linguistics.

Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, and Noah A. Smith. 2022. Annotators with attitudes: How annotator beliefs and identities bias toxic language detection. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational

Linguistics: Human Language Technologies, pages 5884–5906, Seattle, United States. Association for Computational Linguistics.

Boaz Shmueli, Jan Fell, Soumya Ray, and Lun-Wei Ku. 2021. Beyond fair pay: Ethical implications of NLP crowdsourcing. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3758–3769, Online. Association for Computational Linguistics.

M. Six Silberman, Bill Tomlinson, Rochelle LaPlante, Joel Ross, Lilly Irani, and Andrew Zaldivar. 2018. Responsible research with crowds: Pay crowdworkers at least minimum wage. Communications of the ACM, 61(3):39–41.

Aakash Singh, Deepawali Sharma, and Vivek Kumar Singh. 2025. Misogynistic attitude detection in YouTube comments and replies: A high-quality dataset and algorithmic models. Computer Speech & Language, 89:101682.

Manuel Tonneau, Diyi Liu, Samuel Fraiberger, Ralph Schroeder, Scott A. Hale, and Paul Röttger. 2024a. From languages to geographies: Towards evaluating cultural bias in hate speech datasets. In Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024), pages 283–311, Mexico City, Mexico. Association for Computational Linguistics.

Manuel Tonneau, Diyi Liu, Niyati Malhotra, Scott A. Hale, Samuel Fraiberger, Victor Orozco-Olvera, and Paul Röttger. 2025. HateDay: Insights from a global hate speech dataset representative of a day on Twitter. In Proceedings ofthe 63rdAnnual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 2297–2321, Vienna, Austria. Association for Computational Linguistics.

Manuel Tonneau, Pedro Quinta De Castro, Karim Lasri, Ibrahim Farouq, Lakshmi Subramanian, Victor Orozco-Olvera, and Samuel Fraiberger. 2024b. NaijaHate: Evaluating hate speech detection on Nigerian Twitter using representative data. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 9020–9040, Bangkok, Thailand. Association for Computational Linguistics.

Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, and Oren Pereg. 2022. Eficient few-shot learning without prompts. In Second Workshop on Eficient Natural Language and Speech Processing (ENLSP-II), NeurIPS 2022.

Sahana Udupa and Luise Koch. 2024. Tackling online misogyny in political campaigns: Promise and limitations of artificial intelligence. Feminist Media Studies, 24(6):1428–1434.

Michael Wiegand, Josef Ruppenhofer, and Thomas Kleinbauer. 2019. Detection of abusive language: the problem of biased datasets. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 602–608, Minneapolis, Minnesota. Association for Computational Linguistics.

Philine Zeinert, Nanna Inie, and Leon Derczynski. 2021. Annotating online misogyny. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3181–3197, Online. Association for Computational Linguistics.

## A Detailed Annotation Guidelines

This appendix reproduces the guidelines provided to annotators, lightly edited for formatting and to remove identifying details.

Content warning This document and the related task may include examples of gender-based violence, sexism, transphobia, and other potentially distressing content. Annotators were told to take breaks as needed and to contact the research team if they needed support.

Introduction Annotators labeled social-media posts (for example, YouTube comments) according to whether they contain gender-related speech and, if so, whether that speech is negative, positive, or neutral.

Labels Annotators assign at least one primary label per post, in a two-step process. The first step is to identify whether the comment contains genderrelated speech. A comment is gender-related if it does any of the following:

• contains the words “he”, “his”, “him”, “she”, “her”, or “hers” (pay special attention to these words);

• comments on a person’s physical appearance, including wardrobe or pictures;

• references a person’s romantic or sexual relationships, or marital or parental status, for example calling them a mother or father;

• makes judgments about competence based on a person’s gender;

• references stereotyped masculinity or femininity;

• mentions a person’s gender explicitly, such as stating they are a man, woman, congressman, or congresswoman;

• encourages the election of a specific gender;

• compares a person to an individual of the opposite gender;

• uses gendered slurs or slang such as “bitch”, “dick”, “asshole”, “dude”, “yes man”, or “girl boss”;

The second step is to identify the nature of the gender-related comment, as one of four categories:

• Gender positive: posts that afirm, empower, support, or promote equality for any gender group. This may include gender-equality or rights-based messaging, positive masculinities (emotional openness, non-violence), supportive feminist or pro-equality hashtags (for example #MeToo, #HeForShe), and afirmation of trans, nonbinary, or gender-diverse identities.

• Gender negative: posts that use gender in a harmful, derogatory, stereotyping, exclusionary, sexualizing, objectifying, or violent way. This may include hostile misogyny or misandry, benevolent sexism (paternalistic praise reinforcing hierarchy), restrictive or harmful masculinities, intersectional targeting (gender combined with caste, class, religion, or ethnicity), moral policing or behavioral enforcement, transphobia or gender-policing of LGBTQ+ individuals, and the normalization or trivialization of genderbased violence.

• Neutral: posts that contain a gender reference that is descriptive, factual, or non-evaluative.

• Flag unsure: for comments identified as genderrelated at Level 1, use this label when the evaluative stance cannot be determined confidently.

Decision rules First, is gender referenced or implied (Level 1)? “0” no, not gender-related, stop; “1” yes, gender-related, continue. Second, how is gender used (Level 2)? “0” descriptive or nonevaluative, neutral; “1” harmful or stereotyping, negative; “2” afirming or equality-promoting, positive; “3” unclear, unsure.

## Clarifications

• Distinguish clearly between posts that are not gender-related (no reference to gender at all) and those that are gender-related but neutral (gender is mentioned in a purely descriptive, nonevaluative way). If gender is mentioned in any form, treat the post as gender-related and then classify it.

• Benevolent sexism always counts as gender negative (1).

• If a post contains both positive and negative gender-related content, classify it as gender negative (1).

• General insults without gender content are neutral (0).

• Sexual content counts as gender-related only when it explicitly targets or refers to a specific gender group.

• Sarcasm reinforcing harmful stereotypes is gender negative (1).

• Hashtags and emojis count as content and must be labeled accordingly. Ignore URLs.

• If gender meaning is culturally implied in the local context, treat it as gender-related.

## Examples

• Gender negative (1): “Women shouldn’t speak in public.” “Real men don’t cry.” “Trans women are fake.” “She should know her place.” “Men are trash.”

• Gender positive (2): “Girls deserve equal opportunities.” “Men should be able to express emotions.” “Support #MeToo.” “Trans women deserve rights.” “Proud to support #HeForShe.”

• Neutral (0): “This video is great.” “The woman in blue is the host.” “Thanks bro” (nonevaluative use).

## Edge cases

• “Women are strong, but emotional” is negative (1), a harmful stereotype.

• “Slay queen!” is positive (2), a positive genderrelated expression.

• “King behavior!” is positive (2), positive masculinity.

• “Be a real man and stop crying” is negative (1), restrictive masculinity.

• “He’s the man of the house” is negative (1), gender-role enforcement.

## Annotator tips

• If gender is present but its evaluative stance remains unclear, use flag unsure (3) and add a comment.

• Context matters: rely on cultural interpretation when needed.

• If a comment contains language you cannot confidently understand, use a translation tool where possible. If the translation remains unclear, particularly for slang, code-mixing, or culturally specific expressions, avoid inferring meaning beyond what can be established from the text; if gender is identifiable but its stance remains unclear, use flag unsure.

• If gender is explicitly mentioned, the post cannot be neutral unless the reference is purely descriptive.

• Always prioritize the efect of the statement, not the intention.

## A.1 Additional Illustrative Comments

Table 3 provides two additional majority-labeled examples of each Level 2 category for each country, complementing the verbatim examples shown in the scheme (Figure 2). Original spelling, punctuation, grammatical forms, and emojis are preserved. A dash in the translation column indicates that the original comment is already in English. The examples are illustrative and are not intended to represent all genderrelated speech within a country.

Table 3: Additional verbatim examples of gender-positive, gender-neutral, and gender-negative comments from Kenya, Nigeria, and India.
<table><tr><td>Country</td><td>Label</td><td>Original comment</td><td>English translation</td></tr><tr><td>Kenya</td><td>Positive</td><td>hongera sana dada umeongeza heshima kwa wanawake, kuwa wanaweza</td><td>Congratulations, sister. You have increased respect for women by showing that women are capable.</td></tr><tr><td></td><td>Positive</td><td>#endfemicideke...a perfect song for the movement..stop killing women</td><td></td></tr><tr><td></td><td>Neutral</td><td>like her teachings pale TikTok</td><td>I didn&#x27;t know she&#x27;s the singer of this song , I also I didn&#x27;t know she was the singer of this song. I also like her teachings on TikTok.</td></tr><tr><td></td><td>Neutral</td><td>cute babies,wanakaa nicholas but rengi ni ya mama yao</td><td>Cute babies, they look like Nicholas, but their complexion comes from their mother.</td></tr><tr><td></td><td>Negative</td><td>kuna baadh ya wanawake wana makalio makubwa ila brain hakuna kitu</td><td>Some women have big buttocks, but no brains.</td></tr><tr><td></td><td>Negative</td><td>wanawake wenye sauti kama hiyo are very rudy</td><td>Women with voices like that are very rude.</td></tr><tr><td>Nigeria</td><td>Positive</td><td>we should all be feminists - see the ted talk by chimamanda ngozi adichie.</td><td></td></tr><tr><td></td><td>Positive</td><td>You&#x27;re doing a great job Abigail! I&#x27;m a fellow</td><td></td></tr><tr><td></td><td>Neutral</td><td>Nigerian go girl! wai mumuni matan shi na waneh da allah</td><td>For God&#x27;s sake how many wives does the holyman have?</td></tr><tr><td></td><td>Neutral</td><td>from a nigerian to you; &#x27;aristo&#x27; is a slang for sugar daddy - a rich man, usually more mature in age who is probably married and has a side chick(a young woman/girl) and spends money</td><td></td></tr><tr><td></td><td>Negative</td><td>on them. nice movie. i almost cried but hard man no dey cry. i give it 10/10</td><td>Nice movie. I almost cried, but a tough man does not cry. I give it 10/10.</td></tr><tr><td></td><td>Negative</td><td>this girl is insane. need to repent her behavior is bad. 21 years with fallen breasts and moving naked in the house. a woman must respect her</td><td></td></tr><tr><td>India</td><td>Positive</td><td>body period mouli doesnt have to pretend... she is a strong women</td><td></td></tr><tr><td></td><td>Positive</td><td>) ②. </td><td>Giving women an equal share of political power is a women&#x27;s right. Naam Tamilar Katchi, Seeman.</td></tr><tr><td></td><td></td><td>SLFLTT Harsh bhaiya ko bulawo</td><td>Call Harsh bhaiya [older brother]</td></tr><tr><td></td><td>Neutral Neutral</td><td>teri bhabhi</td><td>Your sister-in-law.</td></tr><tr><td></td><td>Negative</td><td></td><td>This shows that girls&#x27; brains are in their knees,</td></tr><tr><td></td><td>Negative</td><td>ise feminism ki bimari h aur kuch nhi. famous</td><td>meaning that girls are unintelligent. She has the disease of feminism, nothing else. She also wants to become famous by speaking</td></tr></table>

## B Keyword Resource Review and Construction

Country-context reviewers reviewed the countryspecific lists for local relevance, translation accuracy, language and dialect labels, and formal, slang, hybrid, or code-switched usage. They were also asked to identify ambiguous or rarely used terms and propose missing expressions, particularly coded language and locally specific slang.

## B.1 Keyword expansion prompt

The following shared prompt template was used to expand the seed lists. The country, relevant languages and linguistic varieties, and attached seed list were changed for each context. Follow-up prompts were issued separately for individual languages and varieties.

I am reaching out to ask for your   
, support in reviewing and refining   
the attached list of keywords related to   
, gender-based violence and   
gendered hate speech.   
Focus on keywords used in [COUNTRY].   
, Generate terms both in English   
transliteration and in the relevant   
, native script. Use information   
from multiple social-media platforms and   
, the wider internet.   
Assess the relevance, accuracy, and   
, cultural appropriateness of the   
terms, with particular attention to how   
, gender-based violence and   
gendered language appear in YouTube   
, content.   
Please review the attached instructions   
, and guideline document.   
Do not impose a strict limitation on the   
, range of search terms or   
categories. Be expansive and provide a   
, list of 100 candidate terms   
across diferent categories.

For Kenya, the following country-specific follow-up instruction was used:

Kenyan keywords are attached in a text   
, file. Expand the list to cover   
Swahili, Sheng, English, and other   
, relevant Kenyan languages or   
dialects. Be as exhaustive as possible   
, and provide a full list.

Equivalent follow-up prompts were issued for India and Nigeria, with the relevant country, language, and linguistic-variety names substituted. Additional prompts requested the expansion separately for each language or variety.

## B.2 Keyword-list review guidelines

Reviewers completed three tasks. First, they assessed whether each candidate term was actively used in their local context, flagged terms that were rare or ambiguous without context, and identified missing terms. Second, they checked translations and language or dialect labels, including whether terms were formal, slang, hybrid, or code-switched. Third, they proposed additional locally used keywords, supplying a translation or explanation and a language or dialect label where possible. Reviewers were also asked whether particular forms of gender-related speech (including positive genderrelated speech, benevolent sexism, transphobia, and economic control) or vocabulary associated with particular social groups and platforms were underrepresented.

<table><tr><td></td><td>Country Principal languages and varieties represented</td><td>Entries</td></tr><tr><td>India</td><td>English, Hindi, Bengali, Marathi, Tamil, Telugu, Hinglish, Sylheti, Urdu-linked expressions, and other code-mixed forms</td><td>778</td></tr><tr><td>Nigeria</td><td>English, Nigerian Pidgin, Hausa, Igbo, Yoruba, and code-mixed forms</td><td>386</td></tr><tr><td>Kenya</td><td>English, Swahili, Sheng, Luo, Kikuyu, Meru, Embu, Kalenjin, and code-mixed forms</td><td>366</td></tr><tr><td>Total</td><td></td><td>1,530</td></tr></table>

Table 4: Principal language and variety coverage of the final country-specific keyword lists. The lists also include hashtags, abbreviations, emojis, transliterations, slang, and other platform-specific or hybrid forms.

## C Language Composition

To characterize the linguistic makeup of the annotation sample, we labeled the language of each of the 9,600 comments with a large language model (gpt-5-nano), prompted separately for each country with a list of the locally relevant languages plus an “Other” option for any language outside that list. For each comment, the model returned the set of languages present and a flag indicating whether two or more of them are mixed. We take the codemixing rate from this flag and report each language as a presence rate, the share of comments in which it appears, including within code-mixed comments. Because a single comment can contribute to more than one language, presence rates sum to more than 100%. Table 5 summarizes the result.

These labels are automatic and approximate. Spot-checking indicates reliable identification of English, the dominant local languages, and outof-list languages such as Russian and Arabic, but a fuzzier boundary between Swahili, Sheng, and Somali in Kenya, and occasional romanized locallanguage comments misrouted to “Other”.

<table><tr><td>Language / category</td><td>% of comments</td></tr><tr><td>India</td><td></td></tr><tr><td>English</td><td>58</td></tr><tr><td>Hindi</td><td>46</td></tr><tr><td>Tamil</td><td>11</td></tr><tr><td>Bengali</td><td>5</td></tr><tr><td>Telugu</td><td>3</td></tr><tr><td>Marathi</td><td>2</td></tr><tr><td>Code-mixed</td><td>32</td></tr><tr><td>Other (off-list)</td><td>6</td></tr><tr><td>Kenya</td><td></td></tr><tr><td>English</td><td>67</td></tr><tr><td>Swahili</td><td>51</td></tr><tr><td>Sheng</td><td>9</td></tr><tr><td>Somali</td><td>5</td></tr><tr><td>Code-mixed</td><td>40</td></tr><tr><td>Other (off-list)</td><td>12</td></tr><tr><td>Nigeria</td><td></td></tr><tr><td>English</td><td>82</td></tr><tr><td>Nigerian Pidgin</td><td>18</td></tr><tr><td>Yoruba</td><td>12</td></tr><tr><td>Hausa</td><td>7</td></tr><tr><td>Igbo</td><td>7</td></tr><tr><td>Code-mixed</td><td>30</td></tr><tr><td>Other (off-list)</td><td>10</td></tr></table>

Table 5: Language composition of the annotation sample, as a share of each country’s 1,000 comments. Values are presence rates (a comment can contribute to more than one language, so they sum to over 100%). Code-mixed is the share of comments mixing two or more languages, and Other the share in languages outside the country’s language list. Languages present in under 1.5% of comments are omitted. Labels are automatic and approximate (Appendix C).

## D Annotator Background

Demographics The nine annotators (three per country) were each a national of the country whose comments they annotated. Across the three teams, six were women and three were men, with one man and two women in each country. The Indian team was aged 20 to 29: two members had completed undergraduate degrees and were pursuing postgraduate study, and one was an undergraduate student. The Kenyan team had one member aged 20 to 29 and two aged 30 to 39, all educated to at least undergraduate level, with degrees in fields such as engineering and law. The Nigerian team had two members aged 30 to 39 and one aged 40 to 49, with backgrounds spanning computer science, business and marketing, and medicine and global health.

Compensation Annotators were compensated according to institutional short-term consultant rates set by country and consultant level. Because compensation is a central dimension of ethical annotation work, and dataset documentation should make the annotation labor and pay conditions transparent, we report these rates rather than only stating that annotators were paid (Silberman et al., 2018; Shmueli et al., 2021). Daily rates were INR 400 in India, KES 1,360 to 1,808 in Kenya, and NGN 11,775 in Nigeria, corresponding to approximately USD 4.40, USD 10.47 to 13.92, and USD 8.60 per day, respectively. The annotation task took approximately 10-12 hours per annotator.

Reporting Background information was provided voluntarily and was available for eight of the nine annotators, with the ninth annotator’s characteristics reported from the research team’s knowledge. We report characteristics at the level of each country team rather than for individual annotators, because the small panels could otherwise make annotators identifiable. Although annotator backgrounds may shape the interpretation of culturally specific and normatively ambiguous comments, the small panels do not allow us to attribute disagreement to particular demographic characteristics.

<table><tr><td>Context</td><td>Responses</td><td>Reported languages and varieties</td><td>Relevant contextual familiarity</td></tr><tr><td>India</td><td>3/3</td><td>English, Hindi, and Bengali, with addi- Respondents reported familiarity with Indian dian languages reported.</td><td>tional partial familiarity with other In- social-media discourse, gendered cultural references, and relevant stereotypes.</td></tr><tr><td>Nigeria</td><td>3/3</td><td>English, Hausa, and Yoruba.</td><td>Respondents reported familiarity with Nigerian social-media discourse and locally used gendered expressions.</td></tr><tr><td>Kenya</td><td>2/3</td><td>English, Swahili/Kiswahili, Sheng, Kikuyu, Kamba, and Luo.</td><td>Respondents reported familiarity with Kenyan social-media discourse, code-switching, and locally used gendered expressions.</td></tr></table>

Table 6: Aggregated linguistic and contextual expertise reported by the eight annotators who completed the optional questionnaire. Sensitive demographic and professional characteristics are omitted because the country-level panels are small.

## E Author Positionality

The author team spans academic institutions in Europe and the United States and an international development organization, with disciplinary backgrounds in gender and technology-facilitated violence, computational social science, NLP, and development research. Several authors have personal, academic, or professional ties to India, and authors familiar with the Indian context contributed to reviewing and refining the corresponding keyword resources. We do not assume that the research team itself represents the cultural and linguistic diversity of the three country contexts studied. We therefore relied on country-context reviewers and annotators in India, Kenya, and Nigeria for local linguistic and cultural interpretation, while recognizing that these small panels likewise cannot represent their national contexts in full.

## F Label Distribution and Inter-Annotator Agreement

Table 7 reports the distribution of majority Level 1 and Level 2 labels for each country, and Figure 3 visualizes the aggregated label shares within each 3,200-comment sample. Table 8 reports interannotator agreement as Fleiss’ κ and Krippendorf’s α at both levels and for the combined label. Agreement varies across the three countries and is consistently lower at Level 2 than Level 1, reflecting that judging whether the gender-related content is negative, positive, or neutral is harder than judging whether gender is present.

![](images/0855e04073a183fc1b95806871eaecde352d3ac9deaf61cf9c7d15172a8a56d7.jpg)  
Figure 3: Distribution of aggregated labels within each country’s 3,200-comment annotated sample. Each sample comprises 1,500 keyword-enriched comments, 1,500 comments drawn uniformly at random from the full corpus, and 200 keyword-free comments. Because the full sample is partly keyword-enriched, these percentages do not estimate prevalence in the broader corpus; prevalence estimates use the uniform random stratum only. Exact counts are reported in Table 7.

<table><tr><td>Country</td><td>Not gender-related</td><td>GENNEG</td><td>GENPOS</td><td>Neutral</td><td>Unsure</td><td>No L2 majority</td></tr><tr><td>India</td><td>1,730</td><td>631</td><td>113</td><td>704</td><td>8</td><td>14</td></tr><tr><td>Kenya</td><td>1,850</td><td>503</td><td>125</td><td>680</td><td>0</td><td>42</td></tr><tr><td>Nigeria</td><td>1,577</td><td>734</td><td>337</td><td>237</td><td>9</td><td>306</td></tr><tr><td>All</td><td>5,157</td><td>1,868</td><td>575</td><td>1,621</td><td>17</td><td>362</td></tr></table>

Table 7: Distribution of majority labels by country (N = 3,200 each). GENNEG denotes gender-negative and GENPOS gender-positive comments. A Level 2 label is assigned when at least two of the three annotators agree on it. Unsure counts comments whose agreed Level 2 label is unsure, and No L2 majority counts gender-related comments on which no two annotators agree on any Level 2 label. The two columns are reported separately and under the same rule for all three countries.
<table><tr><td>Country</td><td>Level 1</td><td>Level 2</td><td>Combined</td></tr><tr><td>India</td><td>.80 / .80</td><td>.82 / .83</td><td>.77 /.77</td></tr><tr><td>Kenya</td><td>.79 / .79</td><td>.71 / .72</td><td>.72 / .72</td></tr><tr><td>Nigeria</td><td>.55 / .54</td><td>.49 / .49</td><td>.47 / .47</td></tr><tr><td>All</td><td>.71 /.71</td><td>.70 / .68</td><td>.66 / .65</td></tr></table>

Table 8: Inter-annotator agreement by country, reported as Fleiss’ κ / Krippendorf’s α and calculated from the individual annotations before majority aggregation, over all 3,200 comments per country and all 9,600 comments for the All row. Level 1 distinguishes gendered from not-gendered comments. Level 2 distinguishes negative, positive, neutral, and unsure gendered speech. The combined label treats not-gendered comments as one category and combines gendered comments with their Level 2 label.

## G Per-Country Model Results

Table 9 reports SetFit macro-F1 per country under the three training regimes, with zero-shot gpt-5-nano for comparison, and Table 10 reports overall macro-F1 for both supervised models under each regime. They back the cross-cultural-transfer analysis in the main text: within-country and pooled CV are close and both far exceed leaveone-country-out, so the transfer drop reflects crosscultural diference rather than training-set size.

<table><tr><td rowspan="2">Country</td><td colspan="3">SetFit</td></tr><tr><td>Within L1/L2</td><td>CV LOCO L1/L2 L1/L2</td><td>gpt-5-nano L1/L2</td></tr><tr><td>Kenya</td><td>.86 / .64</td><td>.84 / .64 .75 / .48</td><td>.84 / .56</td></tr><tr><td>Nigeria</td><td>.86 / .69</td><td>.86 / .68 .81 / .41</td><td>.80 / .57</td></tr><tr><td>India</td><td>.79 / .64</td><td>.79 / .64 .69 / .48</td><td>.78 / .55</td></tr></table>

Table 9: Per-country macro-F1 for SetFit under the three training regimes, with zero-shot gpt-5-nano for comparison. Within-country and pooled CV are within two points of each other in every country and both far exceed leave-one-country-out, whose drop is steepest for India, so the degradation reflects cross-cultural diference rather than training-set size.

<table><tr><td>System</td><td>Regime</td><td>L1</td><td>L2</td><td>E2E</td></tr><tr><td>SetFit</td><td>Within</td><td>.84</td><td>.70</td><td>.65</td></tr><tr><td></td><td>CV</td><td>.83</td><td>.67</td><td>.63</td></tr><tr><td></td><td>LOCO</td><td>.75</td><td>.47</td><td>.46</td></tr><tr><td>XLM-R</td><td>Within</td><td>.82</td><td>.58</td><td>.55</td></tr><tr><td></td><td>CV</td><td>.82</td><td>.61</td><td>.58</td></tr><tr><td></td><td>LOCO</td><td>.77</td><td>.49</td><td>.48</td></tr></table>

Table 10: Supervised models by training regime: within-country, pooled cross-validation (CV), and leave-one-country-out (LOCO), reported as overall macro-F1. Best regime per model and metric in bold: within-country and CV trade places and are always close, while LOCO is never best, the signature of a transfer gap. Per-country SetFit results are in Table 9.

<table><tr><td>System</td><td>L1</td><td>L2</td></tr><tr><td>SetFit (CV)</td><td>.84</td><td>.64</td></tr><tr><td>gpt-5.4</td><td>.84</td><td>.53</td></tr><tr><td>XLM-R (CV)</td><td>.83</td><td>.58</td></tr><tr><td>gpt-5-nano</td><td>.80</td><td>.51</td></tr><tr><td>Gemma-2-9B</td><td>.76</td><td>.31</td></tr><tr><td>Llama-3.1-8B</td><td>.73</td><td>.35</td></tr><tr><td>Toxicity</td><td>.47</td><td>.38</td></tr></table>

Table 11: Macro-F1 on the random stratum only (n = 4,500, of which 1,226 gender-related with a resolved Level-2 label), a keyword-independent sample. Level 1 scores stay within a few points of the full evaluation set (Table 2), so keyword enrichment does not overstate detection, while the toxicity baseline collapses. With round-2 data, Level 2 is reportable here for the first time (previously $n = 9 4 ) \mathrm { : }$ : SetFit keeps its lead, while gpt-5.4’s Level 2 score drops well below its keywordenriched score, widening the gap with SetFit on this more representative sample.

## H Detection on the Random Stratum

The main evaluation set is dominated by the keyword-enriched stratum, which by construction concentrates gender-related speech. To gauge detection on a sample closer to unrestricted comments, we re-score every model on the random stratum alone (comments drawn uniformly from the corpus regardless of keyword match, $n = 4 { , } 5 0 0 .$ of which 1,337 are gender-related and 1,226 carry a resolved Level-2 label). Table 11 gives the results. Every learned model stays within a point of its full-set Level 1 score, so detection is not inflated by the retrieval terms, whereas the toxicity baseline falls to 0.47 and recovers only 7% of gender-related comments. Level 2 is reportable on this sample for the first time: SetFit keeps its Level 1 lead and extends it at Level 2 (0.64 versus 0.53 for gpt-5.4).

## I Performance by Language

Using the automatic language labels (Appendix C), we group the comments into English, locallanguage (non-English monolingual), code-mixed, and out-of-list (Other) sets, and recompute macro-F1 within each group (Table 12). Every model is strongest on English and weaker elsewhere, the gap widens from Level 1 to Level 2, and the locally supervised SetFit is the most robust on non-English and code-mixed content. As these labels are automatic and approximate (Appendix C), and the Other group is small, the breakdown is descriptive.

<table><tr><td></td><td colspan="4">Level 1 (detection)</td><td colspan="4">Level 2 (stance)</td></tr><tr><td>System</td><td>Eng</td><td>Local</td><td>Mixed</td><td>Other</td><td>Eng</td><td>Local</td><td>Mixed</td><td>Other</td></tr><tr><td>SetFit (CV)</td><td>.90</td><td>.73</td><td>.79</td><td>.71</td><td>.71</td><td>.62</td><td>.65</td><td>.61</td></tr><tr><td>gpt-5-nano</td><td>.87</td><td>.76</td><td>.75</td><td>.72</td><td>.62</td><td>.51</td><td>.55</td><td>.36</td></tr><tr><td>Llama-3.1-8B</td><td>.83</td><td>.69</td><td>.71</td><td>.70</td><td>.47</td><td>.34</td><td>.40</td><td>.26</td></tr><tr><td>Gemma-2-9B</td><td>.85</td><td>.73</td><td>.73</td><td>.73</td><td>.44</td><td>.40</td><td>.38</td><td>.29</td></tr><tr><td>XLM-R (CV)</td><td>.89</td><td>.71</td><td>.77</td><td>.68</td><td>.62</td><td>.56</td><td>.58</td><td>.58</td></tr><tr><td>Toxicity</td><td>.50</td><td>.40</td><td>.41</td><td>.51</td><td>.36</td><td>.24</td><td>.32</td><td>.35</td></tr></table>

Table 12: Macro-F1 by language group for Level 1 (gender-related or not) and Level 2 (negative, positive, or neutral, conditioned on gold gender-related comments). Groups are derived from the automatic per-comment language label: English, Local (non-English monolingual), Mixed (code-mixed), and Other (languages outside the country’s language list). Group sizes: English 3,509, Local 2,078, Code-mixed 3,287, Other 726. Every system peaks on English and the gap widens at Level 2, while SetFit holds up best on the Other group.

## J Model Error Analysis

Figure 4 shows the end-to-end four-way confusion matrix for SetFit (CV), row-normalized so that each row is the recall for a gold class. Neutral gender-related speech is the hardest class, most often misread as gender-negative or not genderrelated, as descriptive gender references sit close to both the abusive and the not gender-related classes.

![](images/34357de4c67ea900a3a487dd93b9539ddf2882828441cd91a09723e22704a42b.jpg)  
Figure 4: End-to-end four-way confusion for SetFit (CV), row-normalized (%, so each row is recall for a gold class). Neutral gender-related speech is the hardest class, most often misread as negative or not genderrelated.