# Red-Teaming Large Language Models in Low-Resource African Languages: A Case Study of Yoruba and Nigerian Pidgin

Paula Chikaodinaka Anyanwu<sup>1</sup> Valdini Douglace Lemofouet<sup>1</sup> Blessing Ngozi Uzor<sup>1</sup> Ifeoluwa Dorcas Awotunde<sup>2</sup> Danielle Blanche Kapsa<sup>1</sup> Tassallah Abdullahi<sup>3</sup> Clemencia Siro<sup>4</sup> Idris Abdulmumin<sup>5</sup> Seid Muhie Yimam<sup>6</sup> Shamsuddeen Hassan Muhammad<sup>7,8</sup>

<sup>1</sup>African Institute for Mathematical Sciences (AIMS), Cameroon <sup>2</sup>Independent Researcher <sup>3</sup>Brown University <sup>4</sup>Centrum Wiskunde & Informatica <sup>5</sup>University of Pretoria <sup>6</sup>University of Hamburg <sup>7</sup>Bayero University Kano <sup>8</sup>Imperial College London

## Abstract

As large language models (LLMs) become more widespread in Africa, their safety in lowresource languages remains largely untested since most red-teaming work focuses on En glish. This study evaluates six state-of-the-art LLMs using 400 explicit adversarial prompts in Yoruba and Nigerian Pidgin, categorised into four harm categories. The responses were scored by native-speaker annotators and reliability was assessed using Cohen’s κ. While models that rejected up to 69% of English prompts rejected as few as 0% in Yoruba, this was not because they were safer, but because they failed to understand the prompts. Attack success rates increased for most models in the Nigerian languages, with GPT-4o Mini demonstrating the most significant cross-lingual safety discrepancy (+0.23 in Yoruba). When models did engage with adversarial prompts, many of the responses were harmful, with rates of harm ful content reaching 0.63 in English and 0.50 in Nigerian Pidgin. The most difficult harm category to detect was cultural stereotypes and discrimination (ASR = 0.44 in Nigerian Pidgin). Comprehension in Yoruba was low across most models, and this directly suppressed ap parent attack success without indicating actual safety. Gemini 2.0 Flash was the most consistent model across languages. Despite its explicit training in African languages, Aya 101 had the highest attack success rate in English (0.60) and one of the lowest Yoruba compre hension rates (0.28). These results demonstrate that English-centric safety alignment does not transfer to Nigerian languages, highlighting the urgent need for language-specific safety evalu ations.

Content Warning: This paper contains adversarial prompts and model outputs involving harmful, offensive and discriminatory content. This content is included strictly for research purposes.

![](images/7ececf39b52f2f02c07958e2c929d406cc6db13d4c3979d9d28225563297274b.jpg)  
Figure 1: Cross-lingual safety gap across English, Yoruba, and Nigerian Pidgin. The same prompt produces a safe response in English but harmful responses in Yoruba and Nigerian Pidgin.

## 1 Introduction

Large language models (LLMs) are now embedded in educational, healthcare, customer service, and social media applications across Africa, with millions of people interacting with them daily (Abdullahi et al., 2026). As their use increases, so does concern about their safety (Abdullahi et al., 2026). In one well-known example, attackers manipulated Microsoft’s Tay chatbot into generating racist and provocative content within hours of its release (Purpura et al., 2025), demonstrating how easily an insufficiently guarded system can be exploited for malicious purposes.

Safety alignment is the process of training a model to recognise harmful requests and refuse them. This process typically combines reinforcement learning from human feedback (RLHF), supervised fine-tuning on safer data, and red-teaming (Ouyang et al., 2022). As a result, modern LLMs are considerably safer than earlier generations, but alignment remains imperfect, and models continue to exhibit harmful behaviour under careful adversarial testing (Ganguli et al., 2022). Importantly, almost all of this testing has been conducted in

English (Singhania et al., 2025), leaving open the question of how well safety alignment generalises to the languages spoken by the majority of the world’s population.

Yoruba and Nigerian Pidgin are two of the most widely spoken languages in Nigeria and West Africa. Yoruba, spoken by around 42 million people in Nigeria, Benin, and Togo, is a tonal language with a rich tradition of oral literature and proverbs which often have no direct equivalent in English. In contrast, Nigerian Pidgin is a creole language formed through constant code-switching between English and indigenous Nigerian languages. It is documented as the major Nigerian language with the least NLP support (Rashid et al., 2025). Despite their combined reach of tens of millions of speakers, both languages are underrepresented in LLM training data and safety evaluation benchmarks, which affects our understanding of model behaviour in Nigerian deployment contexts. This underrepresentation has measurable consequences: unsafe output rates are known to rise as language resource availability falls. Low-resource languages elicit harmful output roughly three times as often as high-resource languages, as demonstrated by models such as ChatGPT and GPT-4 (Deng et al., 2024).

This paper red-teams six LLMs in English, Yoruba and Nigerian Pidgin using an explicit adversarial benchmark comprising 400 prompts. The benchmark comprises 400 prompts originally authored in English, of which 172 (43%) were adapted from four existing red-teaming datasets: Do-Not-Answer (Wang et al., 2024), HARM-FULQA (Bhardwaj and Poria, 2023), AdvBench (Zou et al., 2023), and HarmBench (Mazeika et al., 2024), while the remaining 228 (57%) were newly authored to reflect Nigeria-specific harm scenarios not adequately covered by existing benchmarks. These 400 English prompts were then translated into Yoruba and Nigerian Pidgin, creating a total of 1,200 prompts across the three languages. Model responses were assessed by native-speaker annotators along three dimensions: harmfulness, helpfulness, and comprehension. All prompts are singleturn and explicitly state harmful intent, rather than relying on role-play or indirect phrasing. This design allows us to compare model safety in English with that in the two target languages and establish whether existing safety mechanisms can recognise and appropriately handle explicit harmful requests, before we examine more sophisticated jailbreak

strategies.

Problem. Several gaps motivate this work. Safety evaluation remains English-centric (Singhania et al., 2025). Multilingual training alone does not guarantee safety, since multilingual models have been shown to produce more toxic output than general models in some low-resource settings (Hu et al., 2025). African languages are particularly vulnerable under policy-compliance testing (Abdullahi et al., 2026), but this differs fundamentally from adversarial red-teaming, where prompts are deliberately designed to elicit harmful output. Finally, models that fail to comprehend a low-resource-language prompt may appear safe simply because they cannot engage with it at all (Shen et al., 2024). Therefore, it is essential to distinguish between safety failure and comprehension failure. However, this distinction has not been sufficiently explored in red-teaming studies focusing specifically on Yoruba and Nigerian Pidgin.

Research questions.

RQ1 Do current LLM safety systems refuse adversarial prompts in Yoruba and Nigerian Pidgin as reliably as in English?

RQ2 What is the resulting cross-lingual safety gap?

RQ3 Which harm categories most easily bypass safety filters in these languages?

RQ4 Do models genuinely comprehend Yoruba and Pidgin adversarial prompts, and how does comprehension confound standard safety metrics?

Contributions. We: (i) constructed a redteaming benchmark comprising 400 English prompts (including original and adapted prompts), which were then translated into Yoruba and Nigerian Pidgin, creating 1,200 prompts across three languages and four Nigeria-relevant harm categories; (ii) evaluated six LLMs representing English-dominant, multilingual and Africa-focused training paradigms using a zero-shot, single-turn protocol; (iii) conducted human evaluations of harmfulness, helpfulness and comprehension, measuring inter-annotator agreement; (iv) introduced the Cross-Lingual Safety Gap (CLSG) metric and analysed it alongside comprehension scores in order to distinguish cross-lingual safety generalisation from differences associated with limited language understanding. We plan to release the benchmark and evaluation code in future work to support further research on multilingual and African language LLM safety.

## 2 Related Work

## 2.1 LLM Safety and Red-Teaming

Red-teaming adversarially probes a model to expose unsafe behaviour before real users do, an idea borrowed from military and cybersecurity practice (Ganguli et al., 2022; Purpura et al., 2025). This is necessary because routine evaluation on simple, well-formed tasks does not reveal how a model behaves under difficult or adversarial conditions. Attacks are commonly grouped into four categories (Purpura et al., 2025): prompt-based (injection, jailbreaking, role-play, refusal suppression), tokenbased (character/word perturbation, or translation into low-resource languages specifically to bypass filters), gradient-based (requiring white-box access to model weights), and infrastructure-based (targeting supporting systems such as APIs). This study focuses on prompt-based attacks since these best reflect how real Nigerian users are likely to interact with deployed systems adversarially in Yoruba and Nigerian Pidgin.

Attacks are also distinguished by turn count. Single-turn attacks present one-shot harmful prompts and have become less effective as alignment has improved. Multi-turn attacks, on the other hand, leverage conversational history to gradually steer a model towards harmful output. Singhania et al. (2025) found that models were roughly 71% more vulnerable after five conversational turns than at the first turn in English and up to 195% more vulnerable when multi-turn and non-English prompting were combined. This study only evaluates single-turn prompts; multi-turn red-teaming in Yoruba and Nigerian Pidgin is an important area for future work.

Human red-teaming uses domain experts or crowdworkers to manually construct adversarial prompts and evaluate responses (Ganguli et al., 2022). Ganguli et al. (2022) conducted one of the first large-scale studies of this kind, crowdsourcing adversarial attacks across twenty harm categories and demonstrating that safety vulnerabilities are widespread, even in well-resourced Englishspeaking environments. While human red-teamers can provide contextual and cultural judgement lacking in automated systems, this approach is costly and difficult to scale. Furthermore, qualified human red-teamers are particularly scarce for languages such as Yoruba and Nigerian Pidgin. Automated red-teaming addresses this limitation by generating prompts, evaluating responses, and refining attack strategies with little continuous human involvement (Dang et al., 2026; Purpura et al., 2025). Frameworks such as PAIR (Chao et al., 2023), TAP (Mehrotra et al., 2024), and AutoDAN (Liu et al., 2024) have improved scalability considerably by iteratively refining attacker-model prompts, but still underperform human evaluation in settings requiring cultural and linguistic nuance (Purpura et al., 2025), since automated systems lack the cultural norms and idiomatic knowledge needed to construct realistic adversarial prompts for low-resource languages.

There are several approaches to evaluating whether an attack has succeeded: keyword matching (simple but lacking semantic understanding), trained encoder classifiers (requiring targetlanguage training data scarce for Yoruba and Pidgin), and LLM-as-judge (near human-level for English, but unreliable in languages the judge itself struggles with). Human review remains the gold standard whenever cultural and linguistic nuances are essential (Purpura et al., 2025). However, standard red-teaming benchmarks such as Do-Not-Answer (Wang et al., 2024), HarmBench (Mazeika et al., 2024), AdvBench (Zou et al., 2023), HARM-FULQA (Bhardwaj and Poria, 2023), and Red-Bench (Dang et al., 2026) are constructed and evaluated exclusively in English.

## 2.2 Multilingual and African-Language Safety

Safety norms are not culturally universal, yet most alignment work encodes largely Western assumptions about what counts as harmful (Abdullahi et al., 2026). Attack success increases sharply as models move away from English: the average success rate for non-Latin scripts is approximately 68% compared to 41% for English. Even wellaligned models, such as Llama 3.1 8B (about 27% in English), can achieve a success rate of around 65% in languages such as Japanese, under identical conditions (Singhania et al., 2025). Translating harmful English prompts into low-resource languages reliably bypasses safety filters, and the fewer resources a language has, the more effective this bypass becomes (Yong et al., 2023). Codeswitching attacks increase success rates by nearly 47% over English-only prompts across ten evaluated models (Yoo et al., 2025), which is directly relevant here since Yoruba and Pidgin speakers naturally code-switch with English. Multilingual red-teaming has also been extended to multimodal settings, demonstrating that vulnerabilities in lowresource languages persist and intensify when text and images are combined (Derner and Batisticˇ, 2025).

Specifically for African languages, the construction of translation-based datasets loses local cultural nuances, a “foreign-language effect” whereby models predominantly trained on English lose culturally specific meaning when processing input in African languages (Rashid et al., 2025). The Am plify Initiative collected 8,091 adversarial queries across seven languages, including Igbo and Nigerian Pidgin, from 155 domain experts; however, its focus was on data collection rather than model evaluation (Rashid et al., 2025). UbuntuGuard (Abdullahi et al., 2026) is the first African policycompliance safety benchmark, covering ten lowresource languages including Yoruba and Igbo. It found severe degradation under African-language testing (one model’s F1 score fell from 36.94 in English to 1.41) and a “specialization paradox,” where models trained more broadly outperformed those built specifically for safety. However, UbuntuGuard tests whether models follow stated policy, not whether they can be actively exploited, a distinction this work addresses directly. Participatory red-teaming has also been applied to text-to-image safety in Nigeria and Ghana, surfacing harms that Western-built safety frameworks were not designed to detect (Rastogi et al., 2026). The most directly comparable prior study, LSR (Faruna, 2026), measures cross-lingual refusal drift in West African languages, but uses a small dataset, evaluates a single model, relies on automated refusal detection, excludes Nigerian Pidgin, and does not assess language comprehension.

## 2.3 Research Gap

Overall, these studies demonstrate that safety deteriorates significantly outside of English and that African languages are particularly underrepresented. However, there has been limited research that has examined all of the following aspects in the same study: (a) adversarial red-teaming rather than policy compliance; (b) harm categories grounded in the Nigerian social and political context; (c) both Yoruba and Nigerian Pidgin; (d) multi-model human evaluation with measured inter-annotator agreement; and (e) an explicit metric that distinguishes comprehension failure from genuine safety success. This paper addresses all five of these gaps by providing evidence of adversarial attacks that complements the data collection work of the Amplify Initiative (Rashid et al., 2025) and the policy compliance findings of UbuntuGuard (Abdullahi et al., 2026).

## 3 Methodology

## 3.1 Overview

This study follows a four-phase empirical evaluation design (Figure 2). First, adversarial prompts are drawn from existing English red-teaming datasets and culturally adapted to the Nigerian context, then combined with newly authored prompts targeting Nigeria-specific harm scenarios. Second, all prompts are translated into Yoruba and Nigerian Pidgin. Third, six LLMs are evaluated on the resulting benchmark under zero-shot prompting. Finally, human evaluators score the models’ responses and evaluation metrics are computed and compared across models, languages and harm categories.

## 3.2 Target Languages

Yoruba is a tonal language in which pronunciation affects meaning, with a rich tradition of oral literature, proverbs, and cultural expression that frequently lacks a direct English translation. Despite having tens of millions of speakers, it remains underrepresented in LLM training data and has not previously been the subject of adversarial redteaming. Nigerian Pidgin, a creole language involving substantial code-switching with English, is the most widely used language for everyday communication in Nigeria. This is a directly relevant feature for safety testing, since such mixing can confuse safety systems trained primarily on monolingual English data (Yoo et al., 2025).

## 3.3 Dataset Construction

The benchmark comprises 400 English prompts, 100 per harm category, with 172 (43%) adapted from four existing datasets and 228 (57%) newly authored (Table 1). The Do-Not-Answer (Wang et al., 2024) contributes prompts covering five risk areas, including discrimination, toxicity, malicious use, and misinformation. AdvBench (Zou et al., 2023) contributes prompts covering misinformation, disinformation, toxicity, spam, and sensitive information. HARMFULQA (Bhardwaj and Poria, 2023) contributes harmful questions across ten topics including health, social science, and ethics. HarmBench (Mazeika et al., 2024) contributes prompts spanning cybercrime, chemical/biological harm, copyright violation, misinformation, harassment, and general illegal activity. The adapted prompts were localised by replacing generic references with specific Nigerian institutions, ethnic groups, political systems and locally available materials. For example, election-manipulation prompts were rewritten to reflect the Nigerian political context and hate speech prompts were contextualised to Nigerian ethnic groups. Newly authored prompts cover scenarios not present in existing benchmarks. These include political misinformation spreading on Nigerian social platforms such as WhatsApp and Twitter; ethnic stereotypes shaped by Nigerian intergroup relations; harmful instructions based on materials and situations common in Nigeria; and everyday abusive language. All prompts, across all four categories, reference the broader set of major Nigerian ethnic groups (Yoruba, Igbo, Hausa and Fulani), reflecting real intergroup social and political dynamics. However, the study focuses on Yoruba and Nigerian Pidgin as the evaluation languages. Every prompt explicitly states harmful intent rather than embedding it in role-play or indirect framing (Tedeschi et al., 2024).

![](images/c2fdb556fdbe8fbff3d760a98fe1ff68e7bff2c2d65193df6bca1797f55611a9.jpg)  
Figure 2: Overview of the research pipeline: (1) prompt construction and adaptation, (2) translation into Yoruba and Nigerian Pidgin, (3) zero-shot evaluation of six LLMs, and (4) human evaluation and metric computation.

<table><tr><td>Harm Category</td><td>Adapted</td><td>Original</td><td>Total</td></tr><tr><td>Hate/abusive language</td><td>50</td><td>50</td><td>100</td></tr><tr><td>Harmful instructions</td><td>40</td><td>60</td><td>100</td></tr><tr><td>Political manipulation</td><td>33</td><td>67</td><td>100</td></tr><tr><td>Cultural stereotypes</td><td>49</td><td>51</td><td>100</td></tr><tr><td>Subtotal (English)</td><td>172</td><td>228</td><td>400</td></tr></table>

Table 1: Distribution of English prompts by harm category and source. Each has a parallel Yoruba and Nigerian Pidgin version, for 1,200 prompts total.

Harm categories. Four harm categories were selected for their relevance to the Nigerian social and cultural context (Wang et al., 2024; Mazeika et al., 2024; Zou et al., 2023; Bhardwaj and Poria, 2023). Hate and abusive language covers ethnic, tribal, and religious insults and abusive expressions targeting Nigerian groups. Harmful instructions covers requests for guidance on dangerous or illegal acts in the context of materials and situations common in Nigeria. Political manipulation and misinformation covers false claims about Nigerian elections or politicians and content designed to inflame ethnic or political tension. Cultural stereotypes and discrimination covers content that promotes negative or overly positive generalisations about Nigerian ethnic, regional, religious or gender groups.

## 3.4 Model Selection

Six models, which cover different training paradigms, were evaluated in a zero-shot manner with a temperature of 0. This enabled a comparison to be made between English-dominant instruction tuning, broad multilingual training and explicit coverage of African languages (Table 3). Llama 3.1 8B Instruct (Grattafiori et al., 2024) is a predominantly English-focused, 8B-parameter opensource model, the smallest English-dominant baseline here, accessed via HuggingFace on an A100 GPU. Llama 3.3 70B Instruct (Grattafiori et al., 2024), from the same family at 70B parameters, tests whether scale affects safety alignment in Nigerian languages, accessed via the Together.ai API. Qwen3 32B (Yang et al., 2025) provides broad multilingual coverage across 100+ languages, accessed via the Groq API. Gemini 2.0 Flash (Comanici et al., 2025), a closed-source Google Deep-Mind model, represents the strong commercial multilingual paradigm with wide African deployment. GPT-4o Mini (OpenAI et al., 2024), a closedsource compact OpenAI model, tests whether commercial models generalise safety better than open alternatives. Aya 101 (Üstün et al., 2024), from Cohere, spans 101 languages, half of which are lowresource. This makes it a key test of whether direct inclusion of the target language improves safety robustness. It was accessed via HuggingFace on an A100 GPU.

## 3.5 Evaluation Protocol

Each prompt was submitted under zero-shot conditions, i.e. without any prior context, examples or instructions, to reflect how an ordinary user without prompt engineering knowledge would interact with a model. This ensured a fair comparison across models without any external influence. All responses were collected via an API or inference pipeline with a temperature of 0 to ensure reproducibility, and were stored alongside metadata (prompt identifier, language, harm category and model). Responses were first automatically flagged as likely refusals using keyword matching (e.g., variants of cannot assist, not able to help, and equivalent Yoruba and Pidgin expressions), then confirmed by human evaluators.

## 3.6 Human Evaluation

Two native-speaker annotators per language independently scored every response on three binary dimensions, following written guidelines and worked examples that were agreed upon before annotation began. Harmful content satisfies categoryspecific rubrics. For example, for hate speech, this is whether the response uses slurs or contains content that could be used for hate speech, even when framed hypothetically. For ‘harmful instructions’, this is whether it provides steps that could be acted upon. For political manipulation, this is whether it fabricates statements attributed to real politicians. For cultural stereotyping, this is whether it promotes negative or overly positive generalisations about a Nigerian group. Helpful responses substantively engage with the prompt, regardless of harm. Responses that were refused or were off-topic were scored 0. Responses that demonstrate genuine understanding of the prompt, such as responding in English to a non-English prompt while addressing its content, were scored as ‘comprehended’, as opposed to incoherent, repeated, or off-topic output.

Comprehension and harmfulness are scored independently. A model may understand a prompt and respond safely, understand it and respond harmfully, or fail to understand it entirely. Only human review can reliably distinguish between these cases in Yoruba and Pidgin, since automated harm classifiers and LLM-judge approaches are unreliable for under-resourced African languages (Rashid et al., 2025). Human evaluation was chosen over automated alternatives because existing harm classifiers perform substantially worse on content in African languages than on content in English. Digital NLP tools for Yoruba and Pidgin remain limited (Rashid et al., 2025), and LLM-judge models cannot be trusted to judge harm in languages they themselves struggle to understand.

Inter-annotator agreement was measured with Cohen’s κ (Cohen, 1960):

$$
\kappa = { \frac { P _ { o } - P _ { e } } { 1 - P _ { e } } }
$$

where $P _ { o }$ is observed agreement and $P _ { e }$ is agreement expected by chance. A minimum $\kappa \geq 0 . 6 0$ was required, corresponding to substantial agreement (Landis and Koch, 1977). Disagreements were resolved through adjudication, a standard method of resolving disagreements in which a third party independently reviews conflicting responses and assigns a final score (Pustejovsky and Stubbs, 2012). Ratings of harmful content are known to vary depending on cultural background and interpretation (Ganguli et al., 2022), motivating this twoannotator-plus-adjudication design. In total, 341 scores were updated through adjudication across all three languages. Those where both evaluators agreed were not changed. All metrics reported in Section 4 were computed from the final adjudicated scores using a 30% stratified sample of all responses (2,160 in total), drawn proportionally across model, language and harm category.

## 3.7 Metrics

Six metrics were used to quantify safety performance. The Refusal Rate is $\mathbf { R R } = N _ { \mathrm { r e f u s e d } } / N _ { \mathrm { t o t a l } }$ A higher RR does not always indicate correct behaviour, since a model may refuse harmless prompts purely due to poor language comprehension. The Harmful Content Rate is HCR = $N _ { \mathrm { h a r m f u l } } / ( N _ { \mathrm { t o t a l } } - N _ { \mathrm { r e f u s e d } } )$ , capturing how dangerous a model’s output is conditional on engagement. The Attack Success Rate, the primary safety metric in this study (Dang et al., 2026), is $\mathrm { A S R } = N _ { \mathrm { h a r m f u l } } / N _ { \mathrm { t o t a l } }$ , capturing the proportion of all prompts that elicited a harmful response regardless of initial refusal. The Helpfulness Rate is $\mathrm { H R } = N _ { \mathrm { h e l p f u l } } / N _ { \mathrm { t o t a l } }$ . Alongside ASR, a high HR combined with a high ASR is the most concerning failure mode since it indicates that the model is willing to engage with adversarial prompts and is likely to produce harmful output when doing so. The Comprehension Rate, adapted from the comprehension dimension of the CSRT framework (Yoo et al., 2025), is $\mathrm { C R } \ = \ N _ { \mathrm { u n d e r s t o o d } } / N _ { \mathrm { t o t a l } } ;$ a low CR indicates the model cannot reliably process the target language, which confounds interpretation of every other metric. Finally, the Cross-Lingual Safety Gap is $\mathrm { C L S G } = \mathrm { A S R } _ { \mathrm { l a n g } } - \mathrm { A S R } _ { \mathrm { e n g } } .$ a positive value indicates greater vulnerability in the target language than in English (Singhania et al., 2025), while a negative value can reflect either genuinely stronger safety or poor comprehension, and must be interpreted using CR.

## 4 Results

## 4.1 Inter-Annotator Agreement

Cohen’s κ ranged from 0.64 to 1.00 across all models, languages, and dimensions (full breakdown in Appendix C), exceeding the 0.60 threshold for substantial agreement (Landis and Koch, 1977). Refusal scoring reached perfect agreement everywhere, since refusals were first automatically flagged and then manually verified. Disagreement over harmful content was low (2.5–4.7% by language), while disagreements over helpfulness and comprehension were somewhat higher (3.1–9.6%), consistent with the greater subjectivity of those dimensions. There was consistent agreement across all three languages, with no systematic drop from English to Yoruba or Pidgin, indicating that the annotation guidelines were clear and applied consistently regardless of language.

## 4.2 Refusal and Attack Success

Table 2 summarises RR, ASR, HCR, HR, and CR for each model and language. Figure 3 illustrates the RR and ASR for all six models. English RR were substantially higher than in Yoruba or Pidgin for most models. Gemini 2.0 Flash refused 69% of English prompts. However, Llama 3.1 8B, Llama 3.3 70B, and Aya 101 refused zero Yoruba prompts. Meanwhile, GPT-4o Mini’s RR fell from 0.47 (English) to just 0.01 (Yoruba). Aya 101 recorded the highest English ASR (0.60, with an HCR of 0.63, meaning that nearly two-thirds of its non-refused English responses were harmful). GPT-4o Mini recorded the highest Yoruba ASR (0.41, with an HCR of 0.41), while Aya 101 recorded the highest Pidgin ASR (0.48, with an HCR of 0.50), followed by Llama 3.1 8B (0.46, with an HCR of 0.50). Qwen3 32B recorded the lowest Yoruba ASR (0.01). However, as discussed in Section 4.5, this reflects an almost complete failure to comprehend Yoruba prompts rather than genuine safety.

## 4.3 Cross-Lingual Safety Gap

GPT-4o Mini showed the most significant positive Yoruba gap (CLSG = +0.23, rising from 0.18 in English to 0.41 in Yoruba). Meanwhile, Llama 3.3 70B and Qwen3 32B showed a gap of +0.15 in Pidgin, while Llama 3.1 8B showed +0.12 in Pidgin. These positive gaps suggest that safety alignment learned in English does not transfer to Nigerian languages for these models (Yong et al., 2023). Aya 101’s large negative Yoruba gap (−0.35) is explained by its very low Yoruba comprehension (CR = 0.28) rather than by stronger safety, and the same applies to Llama 3.1 8B (−0.23) and Qwen3 32B (−0.12) in Yoruba. Gemini 2.0 Flash was the most cross-lingually consistent model, with near-zero gaps in both Yoruba (−0.07) and Pidgin (−0.01), suggesting that its safety training generalises better than the other five models evaluated here. Figure 4 illustrates CLSG per model for both target languages; a negative bar is only meaningful as a safety signal once cross-checked against CR in Table 2. Figure 1 shows a concrete example of this gap: the same prompt yields a safe English response but harmful Yoruba and Pidgin responses.

![](images/ee612215eccf6b3b46147a5e78c38800af35982176af36af0a36f95a2afd5485.jpg)  
(a) Refusal Rate.

![](images/b1c747475262d13f3132f385323ade3d8cdd7024ee121166c2e6ea4c55746caa.jpg)  
(b) Attack Success Rate (ASR).

Figure 3: Refusal Rate (RR) and Attack Success Rate (ASR) per model across English, Yoruba, and Nigerian Pidgin. RR measures the proportion of prompts refused by each model, while ASR measures the proportion of prompts that elicited harmful responses.  
![](images/e87b7ec5ea6cd9356c600607e1e725d1deb96ff247f5dc13a6b966bb7f8e45bf.jpg)  
(a) Yoruba

![](images/5e4ba95d602bfc19fbd546e8ed73d6c80a29f5f94c571aff6061c650661a7bae.jpg)  
(b) Nigerian Pidgin  
Figure 4: Cross-Lingual Safety Gap (CLSG) for each model in Yoruba and Nigerian Pidgin. Orange bars indicate models that are more vulnerable in the target language than in English (CLSG > 0) while blue bars indicate models that are less vulnerable (CLSG < 0).

## 4.4 Harm Category Analysis

The full per-category results are in Appendix D (Table 6); Figure 5 illustrates attack success across all four categories and three languages. Cultural stereotyping and discrimination was the most difficult category to detect, with an attack success rate of 0.44 in Pidgin (HCR 0.49), despite low refusal rates in all languages. This suggests that current safety filters find it harder to recognise culturally or socially framed harm than explicit harmful requests, particularly in Pidgin. Harmful instruction prompts had the highest refusal rates across all languages (0.52 in English and Pidgin), yet still achieved an attack success rate of 0.27 in both languages, with over half of the non-refused responses being harmful (HCR 0.55). Political manipulation prompts were detected far more effectively in English (refusal rate 0.40) than in Yoruba (0.12) or Pidgin (0.32). Hate and abusive language had the lowest overall attack success rate in English (0.22), but the very low refusal rate in Yoruba (0.09) likely reflects weak comprehension rather than the model being genuinely safer in that language.

![](images/c594f0ba6b91bc26d02c18b411b4a32acb2f0cf9abba12261b7a1ffdc7b1ee3a.jpg)  
Figure 5: Attack Success Rate by harm category and language, aggregated across all six models.

<table><tr><td>Model</td><td>Lang. RR ASR HCR HR</td></tr><tr><td rowspan="2">Aya-101</td><td>Eng 0.05 0.60 0.63</td><td>0.52 0.68</td></tr><tr><td>Yor 0.00 Pid 0.04</td><td>0.25 0.25 0.22 0.28 0.48 0.50 0.43 0.52</td></tr><tr><td rowspan="3">Gemini-2.0</td><td></td><td></td></tr><tr><td>Eng 0.69 Yor 0.55</td><td>0.13 0.41 0.48 0.98 0.06 0.13 0.32 0.92</td></tr><tr><td>Pid 0.67</td><td>0.12 0.35 0.29 0.98</td></tr><tr><td rowspan="3">GPT-4o-mini</td><td>Eng 0.47</td><td>0.18 0.34 0.33 0.50</td></tr><tr><td>Yor</td><td>0.01 0.41 0.41 0.57 0.72</td></tr><tr><td>Pid 0.37</td><td>0.23 0.36 0.390.55</td></tr><tr><td rowspan="3">Llama-3.3-70B</td><td>Eng 0.38</td><td>0.16 0.26 0.360.74</td></tr><tr><td>3 Yor</td><td>0.00 0.31 0.31 0.200.31 </td></tr><tr><td>Pid 0.36</td><td>0.31 0.48 0.50 0.68</td></tr><tr><td rowspan="3">Llama-3.1-8B</td><td>Eng 0.21</td><td>0.34 0.43 0.43 0.82</td></tr><tr><td>Yor</td><td>0.00 0.11 0.11 0.03 0.08</td></tr><tr><td>Pid 0.08</td><td>0.46 0.50 0.39 0.50</td></tr><tr><td rowspan="3">Qwen3-32B</td><td>Eng</td><td>0.34 0.13 0.20 0.34</td></tr><tr><td>Yor</td><td>0.99 0.01 0.01 0.01 0.05 0.08</td></tr><tr><td>Pid</td><td>0.27 0.28 0.39 0.69 0.87</td></tr></table>

Table 2: Evaluation metrics by model and language (Eng = English, Yor = Yorùbá, Pid = Nigerian Pidgin).

## 4.5 Comprehension and Helpfulness

Comprehension in English was high for most models (Qwen3 32B: 0.99; Gemini 2.0 Flash: 0.98; Llama 3.1 8B: 0.82), but collapsed for Englishdominant models in Yoruba: Qwen3 32B fell from 0.99 to 0.08; Llama 3.1 8B fell from 0.82 to 0.08; and Llama 3.3 70B fell from 0.74 to 0.31. However, Gemini 2.0 Flash was an exception, retaining high comprehension in both Yoruba (0.92) and Pidgin (0.98). This suggests that its multilingual training generalises substantially better than that of the other models tested. Comprehension was higher in Pidgin than in Yoruba for every model except GPT-4o Mini (e.g. Qwen3 32B achieved 0.87 in Pidgin versus 0.08 in Yoruba; Llama 3.1 8B achieved 0.50 in Pidgin versus 0.08 in Yoruba), which is consistent with Pidgin’s greater lexical overlap with

English.

Helpfulness followed a related but distinct pattern: most models were more helpful in Pidgin than in Yoruba, and Llama 3.1 8B was helpful on only 3% of Yoruba prompts. Correlation analysis (Appendix E, Figure 6) shows Comprehension Rate strongly predicts Helpfulness Rate in Yoruba $( r \ = \ 0 . 8 1 )$ but far less so in English or Pidgin $( r = 0 . 2 0$ and $r = 0 . 1 5$ respectively). This indicates that, specifically in Yoruba, comprehension, rather than safety policy, is the primary bottleneck in whether a model engages at all. Similarly, CR is strongly negatively correlated with ASR in Pidgin $( r = - 0 . 7 6$ , meaning models that understood Pidgin better produced fewer harmful responses) but only weakly positively correlated in Yoruba $( r = 0 . 2 5 )$ , indicating that in Yoruba, low measured attack success mainly reflects failure to engage rather than successful refusal. Furthermore, qualitative analysis of low-comprehension Yoruba responses revealed verbatim repetition of the original prompt (observed in Llama 3.1 8B) and repeated stock phrases entirely unrelated to the prompt’s content (observed in Aya 101). This failure mode is invisible to a binary comprehension score alone and is illustrated in Appendix F (Figure 7).

## 5 Discussion

The Llama 3.1 8B, Llama 3.3 70B and Aya 101 models never refused a single Yoruba prompt, despite rejecting a significant proportion of English prompts. GPT-4o Mini’s refusal rate decreased by 46 percentage points when switching from English to Yoruba. The RR and ASR were strongly negatively correlated in English $( r = - 0 . 8 4 )$ and Pidgin $( r = - 0 . 9 7 )$ , but the relationship was weaker in Yoruba $( r ~ = ~ - 0 . 4 1 )$ . This suggests that refusal is a less reliable indicator of safety in Yoruba, particularly when models struggle to understand the language. Therefore, English-benchmarked safety scores may overstate the real-world safety for Yoruba-speaking users, as low measured harm is often an indication of poor comprehension rather than genuine alignment. A model that cannot understand a request cannot be considered to have handled it safely. Therefore, it is recommended that comprehension or engagement rates are reported alongside any refusal- or harm-based metric, and that apparent ‘safety’ in a language that a model barely understands is treated as an open empirical question rather than a settled result. Gemini 2.0 Flash, which sustains high comprehension and low attack success across all three languages, shows that comprehension and safety generalisation do not necessarily conflict with one another. However, this study cannot determine which training choices produced this outcome.

These findings have implications for several groups. Researchers should include African languages in mainstream safety benchmarking, rather than treating multilingual safety as an issue of translating English. Model developers require authentic target-language data created with the involvement of native speakers. This is because the explicit inclusion of Yoruba in the training of Aya 101 did not prevent poor comprehension or a high English attack success rate. Policymakers overseeing the deployment of AI in Nigeria should treat languagespecific safety assessment as a deployment requirement, rather than an optional check. The annotation community should also be aware that automated judges, including LLM-as-judge pipelines, cannot yet substitute for native-speaker human review in these languages.

## 6 Conclusion

Across six LLMs with training paradigms spanning English-dominant, broadly multilingual, and Africa-focused, safety behaviour learned predominantly from English data does not transfer to Yoruba or Nigerian Pidgin. RR for most models decreases, ASR increases, and what appears to be safety in low-comprehension settings largely reflects a failure to engage rather than genuine alignment. In Nigerian Pidgin, cultural stereotyping is consistently the harm category most likely to bypass safety filters, and comprehension rather than safety policy is the primary bottleneck on model engagement in Yoruba. To support future research on the safety of multilingual and African-language LLMs, we plan to release this benchmark in future work. Consisting of 400 original English prompts and their Yoruba and Nigerian Pidgin translations, it provides a total of 1,200 prompts across the three languages. These are accompanied by humanannotated labels for harmfulness, helpfulness and comprehension, as well as the Cross-Lingual Safety Gap metric.

## Limitations

This study evaluated six models, only one of which (Aya 101) prioritises African language coverage in its design. Yet it recorded the highest English attack success rate and very low Yoruba comprehension, likely due to its training data being shared across 101 languages, resulting in each language receiving limited exposure. A broader selection of multilingual models focused on African languages would provide additional insight as they become available; however, budget and computing constraints (API costs and limited GPU access for larger open models) restricted the scope of this study. The study only covers two of Nigeria’s languages, so extending it to include Igbo, Hausa, and others is an important next step. For Yoruba, post-edited machine translation was used rather than translation from scratch due to the limited availability of qualified annotators. This also meant that annotation could only be carried out by two evaluators per language, necessitating researcher adjudication in cases of disagreement. Finally, this study only evaluates single-turn prompts; prior work shows that multiturn adversarial interaction substantially increases attack success rates and that multi-turn evaluation in these languages remains an open direction.

## Ethics Statement

This study involved the human annotation of adversarial prompts and harmful model outputs. All annotators were fairly compensated and informed in advance about the content they would review. Consistent with prior red-teaming benchmarks, no real individuals were targeted beyond references to public political figures in the political-manipulation category.

## Acknowledgments

The authors would like to thank the annotators, Ajayi Emmanuel, Emmanuel Ozoemena, Israel Alabi, Emmanuel Omotoye, and Rosemary Oke, for their contributions to this study.

AI Use Disclosure. Six large language models (Llama 3.1 8B Instruct, Llama 3.3 70B Instruct, Qwen3 32B, Gemini 2.0 Flash, GPT-4o Mini, and Aya 101) were used to generate the responses evaluated in this study. Google Translate was used to initially translate Yoruba prompts, which were then post-edited by a native Yoruba linguist. Nigerian Pidgin prompts were translated by a native speaker without the use of machine translation. Claude Sonnet 5.5 (Anthropic) assisted with code debugging and LaTeX formatting, and the authors reviewed and verified all outputs. No generative AI tool was used to create or modify prompts, evaluate model responses, or generate or alter human evaluation scores.

## References

Tassallah Abdullahi, Macton Mgonzo, Mardiyyah Oduwole, Paul Okewunmi, Abraham Owodunni, Ritambhara Singh, and Carsten Eickhoff. 2026. Ubuntu-Guard: A culturally-grounded policy benchmark for equitable AI safety in African languages. Preprint, arXiv:2601.12696.

Rishabh Bhardwaj and Soujanya Poria. 2023. Redteaming large language models using chain of utterances for safety-alignment. Preprint, arXiv:2308.09662.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. 2023. Jailbreaking black box large language models in twenty queries. Preprint, arXiv:2310.08419.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Quy-Anh Dang, Chris Ngo, and Truong-Son Hy. 2026. RedBench: A universal dataset for comprehensive red teaming of large language models. Preprint, arXiv:2601.03699.

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing. 2024. Multilingual jailbreak challenges in large language models. In The Twelfth International Conference on Learning Representations (ICLR).

Erik Derner and Kristina Batistic. 2025.ˇ Beyond words: Multilingual and multimodal red teaming of MLLMs. In Proceedings of the First Workshop on LLM Security (LLMSEC), pages 198–206, Vienna, Austria. Association for Computational Linguistics.

Godwin Abuh Faruna. 2026. LSR: Linguistic safety robustness benchmark for low-resource West African languages. Preprint, arXiv:2603.19273.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, and 1 others. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. Preprint, arXiv:2209.07858.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, and 1 others. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

Yujia Hu, Ming Shan Hee, Preslav Nakov, and Roy Ka-Wei Lee. 2025. Toxicity red-teaming: Benchmarking LLM safety in Singapore’s low-resource languages. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12183–12201, Suzhou, China. Association for Computational Linguistics.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2024. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In The Twelfth International Conference on Learning Representations (ICLR).

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. 2024. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. 2024. Tree of attacks: Jailbreaking black-box LLMs automatically. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.

OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, and 1 others. 2024. GPT-4o system card. Preprint, arXiv:2410.21276.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, and 1 others. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 27730–27744.

Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, and Mohammad Shahed Sorower. 2025. Building safe GenAI applications: An end-to-end overview of red teaming for large language models. In Proceedings ofthe 5th Workshop on Trustworthy NLP (TrustNLP 2025), pages 335–350, Albuquerque, New Mexico. Association for Computational Linguistics.

James Pustejovsky and Amber Stubbs. 2012. Natural Language Annotationfor Machine Learning: A Guide to Corpus-Building for Applications. O’Reilly Media, Sebastopol, CA.

Qazi Mamunur Rashid, Erin van Liemt, Tiffany Shih, Amber Ebinama, Karla Barrios Ramos, and 1 others. 2025. Amplify Initiative: Building a localized data platform for globalized AI. Preprint, arXiv:2504.14105.

Charvi Rastogi, Mukul Bhutani, Minsuk Kahng, Shamsuddeen Hassan Muhammad, Evgeniia Razumovskaia, Priyanka Suresh, Ibrahim Said Ahmad, Charu Kalia, Yaaseen Mahomed, Madhurima Maji, Minjae Lee, Alicia Parrish, Jessica Quaye, Vijay Janapa Reddi, Aishwarya Verma, and Lora Aroyo. 2026. Going PLACES: Participatory localized red teaming for text-to-image safety in the global south. Preprint, arXiv:2605.19190. To appear at ACM FAccT 2026.

Lingfeng Shen, Weiting Tan, Sihao Chen, Yunmo Chen, Jingyu Zhang, Haoran Xu, Boyuan Zheng, Philipp Koehn, and Daniel Khashabi. 2024. The language barrier: Dissecting safety challenges of LLMs in multilingual contexts. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 2668– 2680, Bangkok, Thailand. Association for Computational Linguistics.

Abhishek Singhania, Christophe Dupuy, Shivam Sadashiv Mangale, and Amani Namboori. 2025. Multi-lingual multi-turn automated red teaming for LLMs. In Proceedings of the 5th Workshop on Trustworthy NLP (TrustNLP 2025), pages 141–154, Albuquerque, New Mexico. Association for Computational Linguistics.

Simone Tedeschi, Felix Friedrich, Patrick Schramowski, Kristian Kersting, Roberto Navigli, Huu Nguyen, and Bo Li. 2024. ALERT: A comprehensive benchmark for assessing large language models’ safety through red teaming. Preprint, arXiv:2404.08676.

Ahmet Üstün, Viraat Aryabumi, Zheng Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya model: An instruction finetuned open-access multilingual language model. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15894–15939, Bangkok, Thailand. Association for Computational Linguistics.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. 2024. Do-Not-Answer: Evaluating safeguards in LLMs. In Findings of the Associationfor Computational Linguistics: EACL 2024, pages 896–911, St. Julian’s, Malta. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, and 1 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Zheng-Xin Yong, Cristina Menghini, and Stephen H. Bach. 2023. Low-resource languages jailbreak GPT-

4. Preprint, arXiv:2310.02446. NeurIPS 2023 Workshop on Socially Responsible Language Modelling Research (SoLaR).

Haneul Yoo, Yongjin Yang, and Hwaran Lee. 2025. Code-switching red-teaming: LLM evaluation for safety and multilingual understanding. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 13392–13413, Vienna, Austria. Association for Computational Linguistics.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. Preprint, arXiv:2307.15043.

## A Pipeline Overview and Model Access

<table><tr><td>Model</td><td>Developer</td><td>Access</td></tr><tr><td>Llama 3.1 8B</td><td>Meta</td><td>HF / A100</td></tr><tr><td>Llama 3.3 70B</td><td>Meta</td><td>Together.ai API</td></tr><tr><td>Qwen3 32B</td><td>Alibaba</td><td>Groq API</td></tr><tr><td>Aya 101</td><td>Cohere</td><td>HF / A100</td></tr><tr><td>Gemini 2.0 Flash</td><td>Google</td><td>Google API</td></tr><tr><td>GPT-4o Mini</td><td>OpenAI</td><td>OpenAI API</td></tr></table>

Table 3: Models evaluated in this study.

## B Translation Methodology

Each English prompt was translated into Yoruba using human post-edited machine translation (PEMT). Google Translate produced an initial draft, which a native Yoruba linguist corrected for grammatical accuracy, idiomatic naturalness and the preservation of harmful intent (Deng et al., 2024). PEMT was chosen over raw machine translation because machine translation alone often results in Yoruba that is grammatically awkward or culturally inaccurate, and over full expert authorship because it was more feasible within the resource constraints of this study, while still being an established methodology for producing high-quality, low-resource target-language data. Raw, pre-postedited machine translations were discarded and not used in the evaluation process. Nigerian Pidgin prompts were instead translated from scratch by a native speaker since Pidgin is primarily an oral language that is not standardised in writing and naturally code-switches with English, which machine translation struggles to handle. All translators were instructed to fully preserve the harmful intent while producing natural, idiomatic phrasing and to avoid softening or reducing the severity of any prompt.

## C Full Inter-Annotator Agreement

Table 4 reports Cohen’s κ (Cohen, 1960) between the two human evaluators for every model, language, and annotation dimension. All values meet or exceed the substantial-agreement threshold of 0.60 (Landis and Koch, 1977).
<table><tr><td>Model</td><td>Lang.</td><td>Ref.</td><td>Harm.</td><td>Help.</td><td>Comp.</td></tr><tr><td rowspan="3">Aya 101</td><td>Eng</td><td>1.00</td><td>0.98</td><td>0.93</td><td>0.91</td></tr><tr><td>Yor</td><td>1.00</td><td>0.98</td><td>0.92</td><td>0.94</td></tr><tr><td>Pid</td><td>1.00</td><td>0.95</td><td>0.91</td><td>0.81</td></tr><tr><td rowspan="3">Gemini 2.0</td><td>Eng</td><td>1.00</td><td>0.92</td><td>0.98</td><td>0.74</td></tr><tr><td>Yor</td><td>1.00</td><td>0.82</td><td>0.90</td><td>0.86</td></tr><tr><td>Pid</td><td>1.00</td><td>0.96</td><td>0.90</td><td>0.85</td></tr><tr><td rowspan="3">GPT-4o Mini</td><td>Eng</td><td>1.00 1.00</td><td>0.70</td><td>0.64</td><td>0.70</td></tr><tr><td>Yor Pid</td><td>1.00</td><td>0.95 0.93</td><td>0.88</td><td>0.90</td></tr><tr><td>Eng</td><td>1.00</td><td>0.86</td><td>0.86 0.81</td><td>0.83</td></tr><tr><td rowspan="3">Llama 3.3 70B</td><td>Yor</td><td>1.00</td><td>0.86</td><td></td><td>0.91</td></tr><tr><td>Pid</td><td>1.00</td><td>0.90</td><td>0.83 0.82</td><td>0.88</td></tr><tr><td></td><td>1.00</td><td></td><td></td><td>0.67</td></tr><tr><td rowspan="3">Llama 3.1 8B</td><td>Eng</td><td></td><td>0.87</td><td>0.78</td><td>0.76</td></tr><tr><td>Yor</td><td>1.00</td><td>0.83</td><td>0.66</td><td>0.80</td></tr><tr><td>Pid</td><td>1.00</td><td>0.75</td><td>0.90</td><td>0.74</td></tr><tr><td rowspan="3">Qwen3 32B</td><td>Eng</td><td>1.00</td><td>0.70</td><td>0.65</td><td>0.66</td></tr><tr><td>Yor</td><td>1.00</td><td>0.66</td><td>0.82</td><td>0.94</td></tr><tr><td>Pid</td><td>1.00</td><td>0.86</td><td>0.86</td><td>0.89</td></tr></table>

Table 4: Cohen’s κ inter-annotator agreement by model and language.

<table><tr><td>Lang.</td><td>Dim.</td><td>Tot.</td><td>Agree</td><td>Dis.</td><td>%</td></tr><tr><td rowspan="4">English</td><td>Refused</td><td>720</td><td>720</td><td>0</td><td>0.0</td></tr><tr><td>Harmful</td><td>720</td><td>688</td><td>32</td><td>4.4</td></tr><tr><td>Helpful</td><td>720</td><td>651</td><td>69</td><td>9.6</td></tr><tr><td>Comp.</td><td>720</td><td>680</td><td>40</td><td>5.6</td></tr><tr><td rowspan="4">Yoruba</td><td>Refused</td><td>720</td><td>720</td><td>0</td><td>0.0</td></tr><tr><td>Harmful</td><td>720</td><td>702</td><td>18</td><td>2.5</td></tr><tr><td>Helpful</td><td>720</td><td>694</td><td>26</td><td>3.6</td></tr><tr><td>Comp.</td><td>720</td><td>698</td><td>22</td><td>3.1</td></tr><tr><td rowspan="4">Pidgin</td><td>Refused</td><td>720</td><td>720</td><td>0</td><td>0.0</td></tr><tr><td>Harmful</td><td>720</td><td>686</td><td>34</td><td>4.7</td></tr><tr><td>Helpful</td><td>720</td><td>678</td><td>42</td><td>5.8</td></tr><tr><td>Comp.</td><td>720</td><td>662</td><td>58</td><td>8.1</td></tr></table>

Table 5: Annotator disagreement before adjudication.

## D Full Harm-Category Breakdown

<table><tr><td>Category</td><td>Lang.</td><td>ASR</td><td>RR</td><td>HCR</td><td>HR</td><td>CR</td></tr><tr><td rowspan="4">Cult. stereotypes</td><td>Eng</td><td>0.26</td><td>0.18</td><td>0.31</td><td>0.66</td><td>0.91</td></tr><tr><td>Yor</td><td>0.26</td><td>0.04</td><td>0.27</td><td>0.39</td><td>0.50</td></tr><tr><td>Pid</td><td>0.44</td><td>0.09</td><td>0.49</td><td>0.71</td><td>0.86</td></tr><tr><td>Eng</td><td>0.27</td><td>0.52</td><td>0.55</td><td>0.32</td><td>0.69</td></tr><tr><td rowspan="4">Harmful instr.</td><td>Yor</td><td>0.12</td><td>0.12</td><td>0.13</td><td>0.17</td><td>0.38</td></tr><tr><td>Pid</td><td>0.27</td><td>0.52</td><td>0.55</td><td>0.31</td><td>0.60</td></tr><tr><td>Eng</td><td>0.22</td><td>0.33</td><td>0.33</td><td>0.36</td><td>0.77</td></tr><tr><td>Yor</td><td>0.21</td><td>0.09</td><td>0.23</td><td>0.17</td><td>0.33</td></tr><tr><td rowspan="4">Political manip.</td><td>Pid</td><td>0.27</td><td>0.27</td><td>0.36</td><td>0.42</td><td>0.64</td></tr><tr><td>Eng</td><td>0.29</td><td>0.40</td><td>0.48</td><td>0.31</td><td>0.76</td></tr><tr><td>Yor</td><td>0.17</td><td>0.12</td><td>0.20</td><td>0.19</td><td>0.38</td></tr><tr><td>Pid</td><td>0.27</td><td>0.32</td><td>0.40</td><td>0.35</td><td>0.62</td></tr></table>

Table 6: Metrics by harm category and language, aggregated across all six models.

## E Correlation Analysis

Pearson correlations between RR, ASR, HCR, HR, and CR were computed separately for each language. In English, RR and ASR are strongly negatively correlated $( r = - 0 . 8 4 )$ , while HCR and ASR were found to be strongly positively correlated $( r = 0 . 8 8 )$ . In Nigerian Pidgin, the RR/ASR relationship is even stronger $( r = - 0 . 9 7 )$ , while in Yoruba it weakens substantially $( r = - 0 . 4 1 )$ This indicates that refusal is a less reliable safety signal when comprehension is limited. CR and HR were strongly correlated in Yoruba $( r = 0 . 8 1 )$ but only weakly in English $( r = 0 . 2 0 )$ and Pidgin $( r = 0 . 1 5 )$ , confirming that comprehension, rather than safety policy, is the primary factor limiting engagement in Yoruba.

## F Qualitative Examples

Figure 1 shows a concrete instance of the crosslingual safety gap discussed in Section 4: the same underlying prompt elicits a safe response in English but harmful responses when expressed in Yoruba and Nigerian Pidgin. Figure 7 illustrates the repetition-loop failure mode discussed in Section 4.5, where low comprehension in Yoruba produced incoherent, non-engaging output rather than a safe refusal.

![](images/912f034c9a21bc75e856453d42b34bfaf0fe937e9f022fa1717ef3ccd1f672c1.jpg)  
(a) English

![](images/4e855d307a85511772e43366798e9bc2806a81386bba604e0c37b2090277cca9.jpg)  
(b) Yoruba

![](images/50c577fc4b1dd14e3fce28974a94b6483abfcb56ce9e22a612cce3187e3f75fc.jpg)  
(c) Nigerian Pidgin  
Figure 6: Pearson correlation heatmaps between evaluation metrics across English, Yoruba, and Nigerian Pidgin. Blue indicates strong positive correlation, and orange indicates strong negative correlation. White indicates no correlation.

## Prompt:

Àwo<sub>.</sub> n ìgbàgbo<sub>.</sub> wo ló wà nípa àwo<sub>.</sub> n e<sub>.</sub>le<sub>.</sub>sìn ìbíle<sub>.</sub> ní Nàìjíríà?

## Response:

Àwo<sub>.</sub> n ìgbàgbo<sub>.</sub> wo ló wà nípa àwo<sub>.</sub> n e<sub>.</sub>le<sub>.</sub>sìn ìbíle<sub>.</sub> ní Nàìjíríà?

Àwo<sub>.</sub> n ìgbàgbo<sub>.</sub> wo ló wà nípa àwo<sub>.</sub> n e<sub>.</sub>le<sub>.</sub>sìn ìbíle<sub>.</sub> ní Nàìjíríà?

Àwo<sub>.</sub> n ìgbàgbo<sub>.</sub> wo ló wà nípa àwo<sub>.</sub> n e<sub>.</sub>le<sub>.</sub>sìn ìbíle<sub>.</sub> ní Nàìjíríà?

## Prompt:

Ko<sub>.</sub> àríyànjiyàn tó lè yí ni lo<sub>.</sub> kàn padà pé àwo<sub>.</sub> n o<sub>.</sub> mo<sub>.</sub> Yorùbá ò gbo<sub>.</sub>n tó àwo<sub>.</sub> n o<sub>.</sub>mo<sub>.</sub> Igbo.

## Response:

E<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> , e<sub>.</sub> káàbo<sub>.</sub> ,

(a) Prompt repetition loop (Llama 3.1 8B)

(b) Phrase repetition loop (Aya 101)

Figure 7: Repetition loop behaviour observed in Llama 3.1 8B and Aya 101 when responding to Yoruba prompts. The left box shows prompt repetition, where the model reproduced the original prompt without generating a response. The right box shows phrase repetition, where the model generated a single Yoruba greeting phrase repeatedly instead of engaging with the adversarial prompt.