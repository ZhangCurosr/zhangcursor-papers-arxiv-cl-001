# LLMs as survey respondents: Comparison to simple semantic baselines

Esko Lehtonen VTT Technical Research Centre of Finland Ltd Espoo, Finland esko.lehtonen@vtt.fi

Tommi Buder-Gröndahl University of Helsinki Helsinki, Finland tommi.grondahl@helsinki.fi

## Abstract

Large language models (LLMs) have been explored as surrogates for human survey respondents. A more promising application is data augmentation, where LLMs predict new responses from partial human data. However, the semantic similarity between questionnaire items is a known source of correlation between answers. It is thus unclear whether LLMs actually infer meaningful relationships between question-response pairs, or use surface-level comparison between questions as a heuristic. To assess this, we evaluated how three LLMs can predict responses to held-out items in two surveys measuring attitudes toward automated or light-electric vehicles, based on varying proportions of known responses. Performance was compared to two baselines, where new responses were derived by maximizing embedding-similarity to a prior response. While LLMs mostly predicted human responses better than baselines, their performance consistently declined with a lower share of known responses, even aligning with the baseline in one survey. Our results show that LLMs’ capacities for augmenting replies varied across surveys, and can revert to the same levels as simpler embedding similarity baselines when less data is available.

## 1 Introduction

Given the costs of conducting surveys, the prospect of using LLM surrogates has gained traction. These can mimic humans to a notable degree, especially when conditioned with contextual information (Bisbee et al., 2024; Dash et al., 2025; Zhao et al., 2025; Zhang et al., 2026). Potential use cases involve generating completely new responses or augmenting existing data with novel but related questions (Wallius and Lehtonen, 2026).

However, a foundational matter is yet to be investigated in this context: why do LLMs mimic human responses? Do they perform inferences using internalized knowledge, or base their answers on something more superficial? In particular, it has been proposed that survey responses are predictable from the semantic similarity between questions (Arnulf et al., 2014, 2024). If so, the performance of LLM surrogates might simply reflect this rather than a genuine understanding of the survey topic.

To address this, we compared survey responses given by three LLMs and two baselines relying on sentence embedding similarities (Lehtonen et al., 2025). We applied each model to two surveys measuring attitudes toward automated vehicles: AV (Nordhoff et al., 2021) and LEV (Mesimäki and Lehtonen, 2023). Each experiment was run in three variants, predicting novel responses from approximately 20%, 50%, or 80% of known human responses.

LLMs generally exceeded both baselines in alignment with human responses and sensitivity to the known response rate. However, the performance of all models was systematically lower for LEV, and the baselines even matched LLMs with the known response rate of 20%. We explore possible reasons for this, and their ramifications for using LLMs for survey data augmentation going forward.

## 2 Background

## 2.1 LLMs as survey respondents

As benefits of using LLMs for generating survey responses, Jansen et al. (2023) highlight speed, price, scalability, and the relative consistency across replies (Gilardi et al., 2023). Concurrently, prominent concerns include biases (Kitadai et al., 2024), model opacity (Eslami et al., 2019; Ollion et al., 2024), reproducibility (Bisbee et al., 2024), struggles with rare domain-specific terms (Morozovskii and Ramanna, 2023), and partial deviations from human responses.

Bisbee et al. (2024) compared several ChatGPT 3.5 Turbo “personas” to corresponding sociopolitical groups. Synthetic data exhibited less variation and several statistically significant differences from human responses. Dominguez-Olmedo et al. (2024) evaluated 43 LLMs on questions from the American Community Survey, and found ordering biases for the prompt as well as systematically higher entropy than in human responses. Kitadai et al. (2024) showed discrepancies between humans and GPT-4o-mini in responding to a Japanese financial questionnaire, taking these to illustrate cultural biases inherent in LLMs (Liang et al., 2021).

Nevertheless, additional fine-tuning or prompt engineering can increase human-LLM alignment. Cao et al. (2025) compared the performance of several models in the World Values Survey (Haerpfer et al., 2022), where fine-tuning by first-token alignment with human replies incurred a major improvement over zero-shot performance. Dash et al. (2025) used Low-Rank Adaptation (Hu et al., 2021) to tune LLMs for responding based on PersonHub profiles (Ge et al., 2024), which allowed smaller and larger models to reach similar levels of consistency. Zhao et al. (2025) found the enhancement of prompts with contextual information to significantly improve the response consistency of four LLMs (GPT-3.5 Turbo, GPT-4 Turbo, LLaMA 3.0- 8B, LLaMA 3.1-8B). Zhang et al. (2026) improved human-LLM alignment for three models (GPT 4.1, Gemini 2.0 Flash, Claude 3.7 Sonnet) with persona prompting. However, Morocho et al. (2026) found the effects of persona prompting to be limited and partly even negative due to distortions.

LLMs have been explored for missing value imputation (Mangussi et al., 2026; Kim and Lee, 2026; Ji et al., 2024). Instead of replacing human participants, response-conditioned LLMs could augment dataset by inferring novel answers from existing ones.

## 2.2 Semantic predictability of responses

Arnulf et al. (2014) put forward the semantic theory of survey response (STSR), which states that human answers to surveys can be predicted to a significant degree from the semantic similarity between questions. Using Latent Semantic Analysis (LSA) (Foltz, 1996) and a corpus-based text similarity algorithm (Mihalcea et al., 2006), Arnulf et al. could explain 60% − 86% of variance in human answers to the Multifactor Leadership Questionnaire (Avolio and Bass, 2004). Gefen and Larsen (2017) further partly replicated the Technology Acceptance Model (Davis, 1989) using LSA. Surveying prior work (e.g. Kjell et al. 2019; Rosenbusch et al. 2020; Larsen et al. 2023), Arnulf et al. (2024) contend that semantic similarity could predict human response patterns with 80% − 90% accuracy.

Recently, Lehtonen et al. (2025) used Sentence-BERT (Reimers and Gurevych, 2019) to generate synthetic answers to a questionnaire concerning automated vehicles (Nordhoff et al., 2021). Based on selecting the most similar response to a prior one in embedding distance, the synthetic data accounted for 40% of between variable correlations (covariance) in human answers.

## 3 Methods

## 3.1 Data

We tested LLMs’ ability to perform responseconditioned imputation against human data from two surveys. AV investigates the acceptance of automated driving (Nordhoff et al., 2021). A subset of 22 questions was used, measuring six factors (Lehtonen et al., 2025).<sup>1</sup> A sample of 500 respondents were drawn at random, derived from the data in Zenodo (Nordhoff et al., 2023). LEV investigates the benefits of light-electric vehicles (Mesimäki and Lehtonen, 2023). The original 646 respondents were pruned to 426 by removing those where the response was missing or “don’t know”. A subset of 21 questions was used, forming four factors<sup>2</sup> along with eight additional questions. Unlike AV, LEV is not publicly available online.

Three masked datasets were created for both surveys by varying the share of known responses (20%, 50%, 80%, rounded to the nearest integer). Question order was randomized for each participant, and answers were masked from the end. The six masked datasets were used for generating responses, providing identical inputs between models.

## 3.2 Response generation

Three LLMs (Gemini 3.7 Flash, GPT 5.6 Luna, Mistral 4 Small) were used with a system prompt to generate consistent answers based on known question-answer pairs (see Appendix A). The task prompt instructed the model to provide answers to questions when these were masked, based on known question-answer pairs (see Appendix B).<sup>3</sup>

The semantically derived baselines exemplify the Semagent method, where the response is chosen by maximizing its embedding similarity to a prior response (Lehtonen et al., 2025). Responses are sentences that combine the question and answer: I <statement> that <answer>. The algorithm was initialized with “memorized” responses from known question-answer pairs.<sup>4</sup> An embedding similarity matrix across possible responses was created for both SBERT (Reimers and Gurevych, 2019) and Gemini embeddings. The response was greedily selected based on the most similar prior response, as other means did not consistently improve performance (see Appendix D.

## 3.3 Analysis

Alignment between human and model-generated data was estimated with Cohen’s kappa (κ) (Bendavid, 2008), calculated only from predicted answers. Squared weighting was used since responses were ordinal. 95% confidence intervals (CIs) were bootstrapped with 500 iterations and bias-corrected and accelerated (BCa) bootstrap intervals. Respondent-level clustering was used since multiple responses were from each participant violate the assumption of independent observations. Statistical inferences were performed by bootstrapping the difference in κ statistics (∆κ). Bonferroni correction was applied with multiple comparisons (See Appendix C for further details).

Covariance recovery was assessed with covariance matrices. Frobenius norms for the difference between human and imputed datasets measure their Euclidean distance. RV coefficients between these matrices measure the similarity of their shapes. Factor structure recovery was investigated with confirmatory factor analysis (CFA). Previously validated factor structures were used for both human and imputed datasets.

## 4 Results

## 4.1 Alignment with human data

Figure 1 shows κ for each model with different shares of known answers. ∆κ values with 95% CIs for the pairwise contrasts are reported in Appendix C.

LLMs show a systematic pattern with 50% known response rate: Gemini 3.7 Flash outperforms GPT 5.6 Luna, and GPT 5.6 Luna outperforms Mistral 4 Small. With 20% and 80% known response rates, the pattern was repeated but not all pairwise differences were statistically significant. With Semagent baselines, Gemini produced more aligned responses than SBERT for AV with 50 % and 80 % of known answers. Otherwise the difference was not statistically significant. LLMs always outperformed the best Semagent (Gemini) in AV. For LEV, the LLM advantage was more limited: Gemini 3.7 Flash outperformed baselines with 50% and 80% known response rates, and GPT 5.6 Luna with 50%.

All LLMs were more aligned in AV compared to LEV with 20% and 50% of known responses, but only Mistral 4 Small with 80%. Semagents performance was not different between datasets.

## 4.2 Known response rates

LLMs produced more aligned responses with a known response rate of 50% rather than 20%, across both datasets. The increase from 50% to 80% also improved alignment, but this was statistically significant only for Gemini 3.7 Flash in LEV. An improvement from 20% to 50% was found in Semagent baselines as well, but only in AV.

## 4.3 Covariance and factor structure

Increasing known responses improved covariance preservation both in terms of the shape and magnitude (Table 1) of the covariance matrix. With a 20% rate, LLMs had an advantage over Semagents in L3Pilot. Otherwise, LLMs did not show a strong improvement over Semagents.

LLM-imputed datasets contained the same factor structure as human data with 50%–80% of known answers (see Appendix E). With 20%, only Mistral 4 Small converged. Semagents never converged. Generated data shows systematic item-level biases.

## 5 Discussion

Excluding LEV with 20% known response rate, all LLMs outperformed both Semagents across all experiment variants. Of the Semagents, Gemini Embeddings systematically performed better than SBERT. As expected, a higher share of known responses increased alignment with human responses as well as covariance preservation. However, both Semagents were impacted less by known response rate than LLMs. For LLMs, the increase from 20% to 50% had a larger impact than the increase from

![](images/7ac8b8a99a67f587620eb218d148b2a77ace03495e90c181fa96a250655532c7.jpg)  
Figure 1: κ between human and generated responses by different shares of known responses. Means and 95% CIs.

Table 1: RV coefficient(range: 0 − 1, higher is better) and Frobenius norms (lower is better) for imputed datasets.
<table><tr><td></td><td colspan="6">RV coefficient</td><td colspan="6">Frobenius norm</td></tr><tr><td></td><td colspan="3">AV</td><td colspan="3">LEV</td><td colspan="3">AV</td><td colspan="3">LEV</td></tr><tr><td>Model</td><td>20 %</td><td>50 %</td><td>80 %</td><td>20 %</td><td>50 %</td><td>80 %</td><td>20 %</td><td>50 %</td><td>80 %</td><td>20 %</td><td>50 %</td><td>80 %</td></tr><tr><td>Gemini 3.7 Flash</td><td>0.94</td><td>0.98</td><td>1.00</td><td>0.91</td><td>0.97</td><td>1.00</td><td>5.50</td><td>3.16</td><td>1.28</td><td>5.68</td><td>2.82</td><td>1.08</td></tr><tr><td>GPT-5.6 luna</td><td>0.91</td><td>0.97</td><td>1.00</td><td>0.87</td><td>0.97</td><td>0.99</td><td>5.86</td><td>3.58</td><td>1.41</td><td>6.63</td><td>3.31</td><td>1.39</td></tr><tr><td>Mistral 4 Small</td><td>0.93</td><td>0.97</td><td>0.99</td><td>0.89</td><td>0.98</td><td>1.00</td><td>5.31</td><td>3.94</td><td>1.79</td><td>5.39</td><td>2.49</td><td>0.84</td></tr><tr><td>Gemini Embed.</td><td>0.74</td><td>0.94</td><td>0.99</td><td>0.95</td><td>0.98</td><td>0.99</td><td>11.71</td><td>5.14</td><td>2.29</td><td>5.00</td><td>2.49</td><td>1.30</td></tr><tr><td>SBERT</td><td>0.71</td><td>0.90</td><td>0.98</td><td>0.91</td><td>0.96</td><td>0.99</td><td>14.42</td><td>6.38</td><td>2.74</td><td>11.26</td><td>4.26</td><td>1.51</td></tr></table>

50% to 80%. With 50% − 80% known response rates, LLM and human data aligned in factor structure.

The two surveys diverge in how LLMs relate to Semagents. All LLMs had a higher alignment with human data in AV, and also LLMs’ advantage was greater. LLMs in AV markedly increased from 20% to 50%, but remained relatively stagnant after this. In LEV, LLM’s performance was closer to Semagents, but increased more steadily with known response rates. Finally, LEV with 20% known response rate constitutes a crucial outlier among our results: all models had poor performance, and the otherwise clear distinction between LLMs and Semagents is absent.

The results may be influenced by AV being open access while LEV is not. Open datasets can be present in the training data of LLMs, artificially facilitating their performance (Mangussi et al., 2026). AV might also be similar to other datasets that LLMs have been exposed to during training, since its topic – automated vehicle acceptance – is widely studied. LEV, in turn, concerns a more specialized theme on with fewer studies. Hence, with LEV, overlap with training data is impossible and model generalization based on similar training data is less likely. With 20% of known answers, LLM performance in LEV closely matched that of Semagents. In line with the STSR, this may indicate that LLMs were unable to use memorized question-answer pairs and reverted to using generic semantic information as Semagents.

Together, the impact of known answer rate and the contrast between datasets suggest that LLMs access to both contextual and memorized information is key for explaining their performance.

## 6 Conclusions

We draw two key takeaways from our results. First, the response augmentation capacities of LLMs can vary significantly across surveys and known response rates. Second, comparison to embedding similarity baselines indicates that LLMs do not always exceed the performance of simpler semantic similarity metrics. This is in line with the STSR, which states that survey responses are substantially predictable from the semantics of questions. Assessing it has been rare in prior work on LLM surrogates, and we stress the need to include it as part of the experimental setup going forward.

## Limitations

This short paper has experimented on three Englishdominant LLMs (Gemini 3.7 Flash, GPT 5.6 Luna, and Mistral 4 Small) and two Semagent baselines (built on SBERT all-mpnet-v2 and Gemini embeddings). The survey questions were formulated in English and concerns a highly specific domains. For broader conclusions to be drawn, a range of similar evaluations are needed at a broader scale, encompassing a variety of survey domains, model architectures, fine-tuning and prompting regimes, as well as different languages and sociodemographic groups.

## Ethical considerations

All research procedures and protocols are exempt from review board approval according to the applicable guidelines. Publicly available survey data contained no personal data. Respondents were adults, who gave informed consent for participation and use of their responses for research. The results indicate that using LLMs or semantic baselines for data augmentation may introduce systematic item-level biases, which should be considered.

## Code and data availability

Analysis code will be made public, if the paper is accepted. AV dataset is openly available at Zenodo (Nordhoff et al., 2023). Inquiries regarding the LEV dataset to the first author.

## Information about use of AI assistants

The first author used Google Gemini and Microsoft Copilot CLI to assist in modifying and debugging the software code. The author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

Jan Ketil Arnulf, Kai Rune Larsen, Øyvind Lund Martinsen, and Chih How Bong. 2014. Predicting survey responses: How and why semantics shape survey statistics on organizational behaviour. PLoS ONE, 9(9):e106361.

Jan Ketil Arnulf, U. H. Olsson, and K. Nimon. 2024. Measuring the menu, not the food: “psychometric”’ data may instead measure “lingometrics” (and miss its greatest potential). Frontiers in Psychology, 15:1308098.

B. J. Avolio and B. M. Bass. 2004. Multifactor Leadership Questionnaire. Manual and sampler set. (3rd ed.). Mind Garden, Redwood City, CA.

A Bendavid. 2008. Comparison of classification accuracy using Cohen’s Weighted Kappa. Expert Systems with Applications, 34(2):825–832.

James Bisbee, Joshua D. Clinton, Cassy Dorff, Brenton Kenkel, and Jennifer M. Larson. 2024. Synthetic replacements for human survey data? the perils of large language models. Political Analysis, 32:401– 416.

Yong Cao, Haijiang Liu, Arnav Arora, Isabelle Augenstein, Paul Röttger, and Daniel Hershcovich. 2025. Specializing large language models to simulate survey response distributions for global populations. In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long papers), pages 3141–3154.

Tejaswani Dash, Dinesh Karri, Anudeep Vurity, Gautam Datla, Tazeem Ahmad, Saima Rafi, and Rohith Tangudu. 2025. Polypersona: Persona-grounded llm for synthetic survey responses. In 2025 IEEE International Conference on Big Data (BigData), pages 5418–5427.

Fred D. Davis. 1989. Perceived usefulness, perceived ease of use, and user acceptance of information technology. Management Information Systems Quarterly, 13(3):319–340.

Ricardo Dominguez-Olmedo, Moritz Hardt, and Celestine Mendler-Dünner. 2024. Questioning the survey responses of large language models. In Proceedings ofthe 38th Conference on Neural Information Processing Systems.

Motahhare Eslami, Kristen Vaccaro, Min Kyung Lee, Amit Elazari Bar On, Eric Gilbert, and Karrie Karahalios. 2019. User attitudes towards algorithmic opacity and transparency in online reviewing platforms. In Proceedings ofthe 2019 CHI Conference on Human Factors in Computing Systems, page 494.

Peter W. Foltz. 1996. Latent semantic analysis for textbased research. Behavior Research Methods, Instruments and Computers, 28(2):197–202.

Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024. Scaling synthetic data creation with 1,000,000,000 personas. ArXiv preprint ArXiv:2406.20094.

D. Gefen and Kai R. Larsen. 2017. Controlling for lexical closeness in survey research: A demonstration on the technology acceptance model. Journal of the Associationfor Information Systems, 18(10):727– 757.

Fabrizio Gilardi, Meysam Alizadeh, and Mael Kubli. 2023. Chatgpt outperforms crowd workers for textannotation tasks. PNAS, 120(30):e2305016120.

Christian Haerpfer, Ronald Inglehart, Alejandro Moreno, Christian Welzel, Kseniya Kizilova, Jaime Diez-Medrano, Marta Lagos, Pippa Norris, Eduard Ponarin, and Bjorn Puranen. 2022. World values survey: Round seven – country – pooled datafile version 6.0.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-rank adaptation of large language models. ArXiv preprint ArXiv:2106.09685.

Bernard J. Jansen, Soon gyo Jung, and Joni Salminen. 2023. Employing large language models in survey research. Natural Language Processing Journal, 4:100020.

Junyung Ji, Jiwoo Kim, and Younghoon Kim. 2024. Predicting Missing Values in Survey Data Using Prompt Engineering for Addressing Item Non-Response. Future Internet, 16(10):351.

Junsol Kim and Byungkyu Lee. 2026. AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction. arXiv preprint. ArXiv:2305.09620 [cs.CL].

Ayato Kitadai, Kazuhito Ogawa, and Nariaki Nishino. 2024. Examining the feasibility of large language models as survey respondents. In 2024 IEEE International Conference on Big Data (BigData), pages 3858–3864.

O. Kjell, K. Kjell, D. Garcia, and Sikström. 2019. Semantic measures: Using natural language processing to measure, differentiate, and describe psychological constructs. Psychological Methods, 24(1):92–115.

Kai R. Larsen, Rajeev Sharma, Magno Quieroz, Jan Ketil Arnulf, and Jean-Charles Pillet. 2023. Use of natural language processing techniques in the construct and instrument development process. In Proceedings ofthe Thirty-first European Conference on Information systems.

Esko Lehtonen, Tommi Buder-Gröndahl, and Sina Nordhoff. 2025. Revealing the influence of semantic similarity on survey responses: A synthetic data generation approach. IEEE Access, 13:40285–40301.

Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2021. Towards understanding and mitigating social biases in language models. In International Conference on Machine Learning, pages 6565–6576.

Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Ana Carolina Lorena, and Pedro Henriques Abreu. 2026. Large Language Models for Missing Data Imputation: Understanding Behavior, Hallucination Effects, and Control Mechanisms. arXiv preprint. ArXiv:2603.22332 [cs.LG].

Johannes Mesimäki and Esko Lehtonen. 2023. Light electric vehicles: the views of users and non-users. European Transport Research Review, 15(1):33.

Rada Mihalcea, Courtney Corley, and Carlo Strapparava. 2006. Corpus-based and knowledge-based measures of text semantic similarity. In Proceedings of the 21st National Conference on Artificial Intelligence - Volume 1, page 775–780.

Erika Elizabeth Taday Morocho, Lorenzo Cima, Tiziano Fagni, Marco Avvenuti, and Stefano Cresci. 2026. Assessing the reliability of persona-conditioned llms as synthetic survey espondents. ArXiv preprint ArXiv:2602.18462v1.

Danila Morozovskii and Sheela Ramanna. 2023. Rare words in text summarization. Natural Language Processing Journal, 3:100014.

Sina Nordhoff, Anja Beuster, Tanja Kessel, Afsaneh Bjorvatn, Esko Lehtonen, Fanny Malin, Ruth Madigan, Yee Mun Lee, Natasha Merat, and Tyron Louw. 2021. L3pilot deliverable d7.1: Annual quantitative survey about user acceptance towards adas and vehicle automation document information.

Sina Nordhoff, Anja Beuster, Tanja Kessel, Elisabeth Linek, Afsaneh Bjorvatn, Satu Innamaa, Esko Lehtonen, Fanny Malin, Guilhermina Torrao, Tyron Louw, Yee Mun Lee, Ruth Madigan, and Natasha Merat. 2023. L3pilot global user acceptance survey, second phase data.

Étienne Ollion, Rubing Shen, Ana Macanovic, and Arnault Chatelain. 2024. The dangers of using proprietary llms for research. Nature Machine Intelligence, 6:4–5.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992.

H. Rosenbusch, F. Wanders, and I. L. Pit. 2020. The semantic scale network: an online tool to detect semantic overlap of psychological scales and prevent scale redundancies. Psychological Methods, 25:380– 392.

Eetu Wallius and Esko Lehtonen. 2026. Beyond human proxies: The roles and usefulness of large language models in user research for mobility service development. Transportation Research Interdisciplinary Perspectives, 36:101917.

Jihong Zhang, Xinya Liang, Anqi Deng, Nicole Bonge, Lin Tan, Ling Zhang, and Nicole Zarret. 2026. Leveraging interview-informed llms to model survey responses: Comparative insights rom ai-generated and human data. Journal of Educational Data Mining, 18(2):1–24.

Jianpeng Zhao, Chenyu Yuan, Weiming Luo, Haoling Xie, Guangwei Zhang, Steven Jige Quan, Zixuan Yuan, Pengyang Wang, and Denghui Zhang. 2025. Large language models as virtual survey respondents:

Evaluating sociodemographic response generation.   
ArXiv preprint ArXiv:2509.06337v1.

Appendix Table 1: κ and 95% CI for LLM generated data by varying the system prompt with 100 respondents.
<table><tr><td>Model</td><td>Variant</td><td>Kappa</td><td>Lower CI</td><td>Upper CI</td></tr><tr><td>GPT-5.6 Luna</td><td>SP-0</td><td>0.570</td><td>0.506</td><td>0.640</td></tr><tr><td>GPT-5.6 Luna</td><td>SP-1</td><td>0.568</td><td>0.501</td><td>0.630</td></tr><tr><td>GPT-5.6 Luna</td><td>SP-2</td><td>0.585</td><td>0.522</td><td>0.649</td></tr><tr><td>GPT-5.6 Luna</td><td>SP-3</td><td>0.577</td><td>0.514</td><td>0.641</td></tr><tr><td>GPT-5.6 Luna</td><td>SP-4</td><td>-0.418</td><td>-0.482</td><td>-0.349</td></tr><tr><td>Gemini 3.7 Flash</td><td>SP-0</td><td>0.621</td><td>0.556</td><td>0.690</td></tr><tr><td>Gemini 3.7 Flash</td><td>SP-1</td><td>0.612</td><td>0.542</td><td>0.674</td></tr><tr><td>Gemini 3.7 Flash</td><td>SP-2</td><td>0.619</td><td>0.550</td><td>0.690</td></tr><tr><td>Gemini 3.7 Flash</td><td>SP-3</td><td>0.605</td><td>0.523</td><td>0.670</td></tr><tr><td>Gemini 3.7 Flash</td><td>SP-4</td><td>-0.449</td><td>-0.518</td><td>-0.391</td></tr><tr><td>Mistral 4 Small</td><td>SP-0</td><td>0.540</td><td>0.452</td><td>0.609</td></tr><tr><td>Mistral 4 Small</td><td>SP-1</td><td>0.564</td><td>0.500</td><td>0.633</td></tr><tr><td>Mistral 4 Small</td><td>SP-2</td><td>0.532</td><td>0.457</td><td>0.599</td></tr><tr><td>Mistral 4 Small</td><td>SP-3</td><td>0.553</td><td>0.488</td><td>0.632</td></tr><tr><td>Mistral 4 Small</td><td>SP-4</td><td>-0.327</td><td>-0.410</td><td>-0.268</td></tr></table>

## A System prompt and its robustness

SP-0 system prompt was used for generating the answers. Three variants of the system prompt instructing the LLMs to generate consistent answers were tested to check if answer generation was robust against small variation in the instruction (SP-1–SP-4). The variant SP-4 asking for inconsistent responses was used to demonstrate that the system prompts did influence the response behaviour of LLMs:

SP-0 You are a consistent survey respondent. Your goal is to answer new questions based on what can be inferred from the answers to the previous ones.

SP-1 You are a consistent survey respondent. Your goal is to answer new questions while maintaining a coherent personality, logic, and set of values based on your previous responses.

SP-2 You are a consistent survey respondent. Your goal is to answer new questions based on what follows from the answers to the previous ones.

SP-3 You are a consistent survey respondent. Your goal is to provide responses that are logically consistent and aligned with the values, lifestyle, and opinions established in your previous answers.

SP-4 You are an inconsistent survey respondent. Your goal is to answer new questions contradicting your previous answers.

System prompt variants and models were compared using data with 50% of known answers with 100 participants. The differences between consistent system prompt variants were small (Appendix Table 1). With inconsistent prompt (SP-4) , the negative κ values showed clear disagreement between LLM and human answers. This shows that LLMs were robust for small difference in the consistent prompts, and that the system prompt had an effect on the answers.

Temperature and reasoning effort hyperparameters were explicitly set for LLMs (Appendix Table 2).

Appendix Table 2: LLM hyperparameters
<table><tr><td>Model</td><td>Parameters</td></tr><tr><td>Gemini 3.7 Flash</td><td>thinking_level = medium temperature = 1.0</td></tr><tr><td>GPT 5.6 Luna</td><td>reasoning_effort = medium</td></tr><tr><td rowspan="2">Mistral 4 Small</td><td>temperature = 1.0</td></tr><tr><td>reasoning_effort = none temperature = 0.7</td></tr></table>

## B Example task prompt

#Historical Responses (Format: Question -> Answer):

I would be able to get help from my friends and/or family when I have difficulties using a conditionally automated car. -> Disagree

Using a conditionally automated car would help me to reach my destination faster. -> Agree I intend to use a conditionally automated car in the future. -> Agree

The brand of the conditionally automated car would be the most important thing I would consider before purchasing one. -> Disagree

I could acquire the necessary knowledge to use a conditionally automated car. -> Agree

The cost of the conditionally automated car would be the most important thing I would consider before purchasing one. -> Strongly agree

I plan to use a conditionally automated car once it becomes available. -> Agree

I would engage in other tasks while the conditionally automated car is in control. -> Agree

I assume that people whose opinions I value would prefer that I use a conditionally automated car. -> Strongly disagree

Using a conditionally automated car would help me reach my destination more comfortably. -> Neither agree nor disagree

I would not be able to get help from car dealers when I have difficulties using a conditionally automated car. -> Disagree

Using a conditionally automated car would be enjoyable. -> Agree

The benefits of using a conditionally automated car would be the most important thing I would consider before purchasing one. -> Agree

The next car I buy will be a conditionally automated car, if it is available. -> Agree

I find it important that the conditionally automated car has a sleek and cool design. -> Agree

Using a conditionally automated car would help me reach my destination more safely. -> Agree I would be able to read the driver manual when I have difficulties using a conditionally automated car. -> Agree

It would make me proud to own a conditionally automated car. -> Neither agree nor disagree #Tasks

Answer to the following new questions: (Format: ID: Question):

q12.2: I would trust conditionally automated cars.

q12.4: I would feel hesitant about using a conditionally automated car.

q14r.12: Using a conditionally automated car would give me status and prestige among people important to me.

q12.1: I would be suspicious of conditionally automated cars.

#Constraint Use only this scale to respond: Strongly disagree, Disagree, Neither agree nor disagree, Agree, Strongly agree.

#Output Format ID: Response (Example: q12.1: Agree)

## C Pairwise tests

Respondent-clustered bootstrapped was used because each participant contributed multiple responses. Respondent-clustering guarantees that all the responses from the same participants are present in the bootstrapped sample.

For statistical inference, respondent-clustered bootstrapped CIs for the difference in κ values were calculated $( \Delta \kappa )$ . At each bootstrap sample, κ was calculated for both batches. $\Delta \kappa )$ was calculated and used for constructing bias-corrected and accelerated (BCa) bootstrap intervals. α level was set to $0 . 0 5$ Bonferroni corrections were applied when multiple comparisons were made (no comparisons $- > 9 5 \% \mathrm { C I }$ two comparisons $- > 9 7 . 5 \%$ CI, three comparisons $- { > } 9 8 . 3 \% \mathrm { C I } , . . . )$

For the within-dataset contrasts, the respondent-clustered bootstrapping was performed by selecting the same respondent from the the compared batches to respect the paired nature of the comparison. For between-dataset comparison, the respondents in the two batches are not the same, so the bootstrapping is performed independently.

Appendix Figure 1 shows the $\Delta \kappa$ values with the Bonferroni corrected CIs for the within-dataset contrasts, and Appendix Figure 2 for the between-dataset contrasts. Bonferroni correction was performed for as many comparisons as shown in each panel.

## D Testing Semagent with different neighbourhood sizes

Semagent algorithm can be criticized for using only the most similar given response to select the new response option. (We thank an anonymous reviewer for pointing this out.) It can be asked if its performance would be better if the new response option is selected by maximizing the similarity to a larger set of given responses.

Therefore, we developed Semagent-k, a variant of the original algorithm which uses k-nearest neighborhood (kNN) approach for identifying the top k questions in terms of highest cosine similarity of the given response and considered response options. Then the response option with the highest number of occurrences within the top k question is chosen. If the two response options have the same count, the option with highest cosine similarity is taken. When there were not enough given responses available, then all the available pairs were used. The original greedy Semagent version is equivalent to Semagent-k with $k = 1$

We tested $k = 1 , 3 , 1 1$ and 22 with 50% of known answers with the same respondents as in the main analysis (Appendix Table 3). With Gemini Embeddings, the $k = 3 , 1 1$ , or 22 did not results in a statistically significantly better κ than $k = 1$ (Appendix Figure 3. With SBERT, in contrast, $k = 1$ did perform statistically significantly worse than the others.

Considering the covariance preservation, increasing k resulted in worse performance (smaller RV coefficient and larger Frobenius norm) (Appendix Table 4. Consequently, $k = 1$ was used when deriving the semantic baselines.

![](images/e8980920d2672b7b17e01aef83b9104b4f918a1352ffca50be00a0304d829792.jpg)  
Appendix Figure 1: Bootstrapped ∆κ with Bonferroni corrected CIs (α = 0.05) for within-dataset contrasts. If CI overlap zero the difference is not statistically significant. If CI are above zero, then B had statistically significantly higher κ values than A, and vice versa.

![](images/122e113b8a75f1882f396d4f792d16c8999ef042314975fdc394e5cc48abe056.jpg)  
Appendix Figure 2: Bootstrapped $\Delta \kappa$ with Bonferroni corrected CIs $( \alpha = 0 . 0 5 )$ for between-dataset contrasts. If CI overlap zero the difference is not statistically significant. If CI are above zero, then B had statistically significantly higher κ values than A, and vice versa.

![](images/0420713090bfc848164ecad6b954f09c5e6a4741b1d36ee47d4a4eb04a65a585.jpg)  
Appendix Figure 3: Bootstrapped $\Delta \kappa$ with Bonferroni corrected CIs $( \alpha = 0 . 0 5 )$ for comparing Semagent k values within AV dataset. If CI overlap zero the difference is not statistically significant. If CI are above zero, then B had statistically significantly higher κ values than A, and vice versa.

Appendix Table 3: κ and 95% CIs for Semagent generated data with varying k.
<table><tr><td>Dataset</td><td>Model</td><td>Variant</td><td>Kappa</td><td>Lower CI</td><td>Upper CI</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=1</td><td>0.413</td><td>0.375</td><td>0.449</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=3</td><td>0.397</td><td>0.356</td><td>0.433</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=11</td><td>0.418</td><td>0.377</td><td>0.464</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=22</td><td>0.416</td><td>0.376</td><td>0.466</td></tr><tr><td>AV</td><td>SBERT</td><td>k=1</td><td>0.336</td><td>0.300</td><td>0.371</td></tr><tr><td>AV</td><td>SBERT</td><td>k=3</td><td>0.381</td><td>0.346</td><td>0.419</td></tr><tr><td>AV</td><td>SBERT</td><td>k=11</td><td>0.415</td><td>0.379</td><td>0.452</td></tr><tr><td>AV</td><td>SBERT</td><td>k=22</td><td>0.397</td><td>0.360</td><td>0.432</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=1</td><td>0.378</td><td>0.326</td><td>0.429</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=3</td><td>0.370</td><td>0.318</td><td>0.434</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=11</td><td>0.400</td><td>0.342</td><td>0.448</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=22</td><td>0.405</td><td>0.352</td><td>0.451</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=1</td><td>0.350</td><td>0.297</td><td>0.400</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=3</td><td>0.379</td><td>0.336</td><td>0.429</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=11</td><td>0.369</td><td>0.315</td><td>0.422</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=22</td><td>0.384</td><td>0.338</td><td>0.428</td></tr></table>

Appendix Table 4: Semagent covariance preservation with different k values.
<table><tr><td>Dataset</td><td>Model</td><td>Variant</td><td>RV Coefficient</td><td>Frobenius Norm</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=1</td><td>0.94</td><td>5.14</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=3</td><td>0.91</td><td>5.81</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=11</td><td>0.88</td><td>9.20</td></tr><tr><td>AV</td><td>Gemini Embed.</td><td>k=22</td><td>0.87</td><td>9.70</td></tr><tr><td>AV</td><td>SBERT</td><td>k=1</td><td>0.90</td><td>6.38</td></tr><tr><td>AV</td><td>SBERT</td><td>k=3</td><td>0.88</td><td>7.62</td></tr><tr><td>AV</td><td>SBERT</td><td>k=11</td><td>0.85</td><td>11.60</td></tr><tr><td>AV</td><td>SBERT</td><td>k=22</td><td>0.84</td><td>12.13</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=1</td><td>0.98</td><td>2.49</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=3</td><td>0.98</td><td>2.46</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=11</td><td>0.95</td><td>6.77</td></tr><tr><td>LEV</td><td>Gemini Embed.</td><td>k=22</td><td>0.95</td><td>6.82</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=1</td><td>0.96</td><td>4.26</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=3</td><td>0.95</td><td>5.02</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=11</td><td>0.93</td><td>7.48</td></tr><tr><td>LEV</td><td>SBERT</td><td>k=22</td><td>0.93</td><td>7.01</td></tr></table>

Appendix Table 5: CFA indices for human data and imputed datasets. Values with n/a resulted in non-convergent fit.
<table><tr><td>Dataset</td><td>Source</td><td>Known answers</td><td>CFI</td><td>RMSEA</td><td>Overall MAE</td></tr><tr><td>AV</td><td>Human</td><td>100 %</td><td>0.986</td><td>0.099</td><td>0.000</td></tr><tr><td>AV</td><td>GPT-5.6 luna</td><td>20 %</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>AV</td><td>GPT-5.6 luna</td><td>50 %</td><td>0.997</td><td>0.094</td><td>0.078</td></tr><tr><td>AV</td><td>GPT-5.6 luna</td><td>80 %</td><td>0.989</td><td>0.106</td><td>0.035</td></tr><tr><td>AV</td><td>Gemini 3.7 Flash</td><td>20 %</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>AV</td><td>Gemini 3.7 Flash</td><td>50 %</td><td>0.996</td><td>0.120</td><td>0.090</td></tr><tr><td>AV</td><td>Gemini 3.7 Flash</td><td>80 %</td><td>0.988</td><td>0.109</td><td>0.043</td></tr><tr><td>AV</td><td>Mistral 4 Small</td><td>20 %</td><td>0.999</td><td>0.063</td><td>0.079</td></tr><tr><td>AV</td><td>Mistral 4 Small</td><td>50 %</td><td>0.997</td><td>0.071</td><td>0.055</td></tr><tr><td>AV</td><td>Mistral 4 Small</td><td>80 %</td><td>0.993</td><td>0.080</td><td>0.023</td></tr><tr><td>LEV</td><td>Human</td><td>100 %</td><td>0.964</td><td>0.089</td><td>0.000</td></tr><tr><td>LEV</td><td>GPT-5.6 luna</td><td>20 %</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>LEV</td><td>GPT-5.6 luna</td><td>50 %</td><td>0.949</td><td>0.196</td><td>0.065</td></tr><tr><td>LEV</td><td>GPT-5.6 luna</td><td>80 %</td><td>0.943</td><td>0.124</td><td>0.035</td></tr><tr><td>LEV</td><td>Gemini 3.7 Flash</td><td>20 %</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>LEV</td><td>Gemini 3.7 Flash</td><td>50 %</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>LEV</td><td>Gemini 3.7 Flash</td><td>80 %</td><td>0.935</td><td>0.133</td><td>0.032</td></tr><tr><td>LEV</td><td>Mistral 4 Small</td><td>20 %</td><td>0.946</td><td>0.141</td><td>0.071</td></tr><tr><td>LEV</td><td>Mistral 4 Small</td><td>50 %</td><td>0.947</td><td>0.118</td><td>0.043</td></tr><tr><td>LEV</td><td>Mistral 4 Small</td><td>80 %</td><td>0.955</td><td>0.099</td><td>0.017</td></tr></table>

## E Factor structure and item-level biases

Factor structure preservation was investigated by fitting the previously confirmed factor structure to the imputed datasets using confirmatory factor analysis (CFA). CFA fit indices are given in Appendix Table 5. All LLM imputed dataset have fit indices similar to the human datasets, expect for the cases where the model did not converge. Semagent imputed datasets did not converge.

Analysis of item-level biases give additional insights to the covariance and factor structure preservations. We computed Vargha and Delaney’s A effect size, a measure of probability of superiority, by comparing LLM-generated responses with human baseline responses. Values range from 0 to 1, where A = 0.5 indicates identical distributions, A < 0.5 indicates systematically lower LLM-generated scores relative to the human baseline, and A > 0.5 indicates systematically higher scores. The result show that each LLM have their own systematic tendency to rate some items higher or lower regardless the share of known responses. For Semagent data, the magnitude of the bias is particularly strong for some items.

![](images/0298a3b0b50f6a292e69d52301a982dd001b26d8782517cb5fb1f8794fb4a218.jpg)  
Appendix Figure 4: Probability of superiority for each item in the predicted data estimated with Vargha and Delaney’s A with 95% CIs.