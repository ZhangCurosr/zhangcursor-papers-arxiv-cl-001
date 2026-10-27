# Do Language Models Understand Igbo Culture?

Beyond Multiple Choice

Blessing Ngozi Uzor<sup>1</sup>, Valdini Douglace Lemofouet<sup>1</sup>, Paula Chikaodinaka Anyanwu<sup>1</sup>, Tassallah Abdullahi<sup>2</sup>, Clemencia Siro<sup>3</sup>, Idris Abdulmumin<sup>4</sup>, Seid Muhie Yimam<sup>5</sup>, Shamsuddeen Hassan Muhammad<sup>6,7</sup>

<sup>1</sup>African Institute for Mathematical Sciences (AIMS), Cameroon <sup>2</sup>Brown University <sup>3</sup>Centrum Wiskunde & Informatica <sup>4</sup>University of Pretoria <sup>5</sup>University of Hamburg <sup>6</sup>Bayero University Kano <sup>7</sup>Imperial College London

## Abstract

The value of language models for any community depends on whether a model actually un derstands the culture of the people it intends to serve. We use the Igbo language as a case study. Existing benchmarks test translated Western ideas, lacking depth. To address this, we introduce CAB-Igbo, a benchmark developed from indigenous cultural sources and verified by native speakers across six domains. We evaluated diverse models under varying conditions, in corporating regional personas. Performance was assessed using multiple-choice accuracy and human evaluations across Epistemic Fi delity, Representational Richness, Linguistic Quality, and Pragmatic Proficiency. Results reveal a stark dissociation between language fluency and cultural knowledge. Through statistical testing, we show that for most models, the severe lack of Epistemic Fidelity and Representational Richness is already present in English and does not significantly worsen when switching to Igbo. Instead, native language prompting mainly degrades Linguistic Quality and Pragmatic Proficiency. Gemini 2.0 Flash is the only model whose cultural knowledge scores improve under Igbo prompting. We find that current models struggle to demonstrate the knowledge needed for true alignment with Igbo, and multiple-choice formats allow pattern-matching to appear as understanding. Furthermore, automated judges agree with humans on linguistic quality but fail on epistemic fidelity. Ultimately, cultural evaluation must be built from community knowledge, establishing human evaluation as the gold standard to prevent misrepresenting low-resource commu nities globally.

## 1 Introduction

Language goes beyond vocabulary; it functions as the historical archive of a people’s shared experiences. As a community navigates similar circumstances generation after generation, a distinct culture evolves, holding the community’s ethical, moral, and aesthetic standards. When a society is culturally healthy, there is a natural seamless alignment uniting the words people speak, the symbols they write, and the reality of their daily interactions. But what happens when this harmony is disrupted? Historically, this fracture began within the colonial education system, which separated the language of conceptual learning from the language of everyday life. Today, we face a new frontier of this same alienation: the Large Language Model.

Language models are increasingly deployed globally to preserve heritage and process information (Zhao and Wang, 2025), yet they are overwhelmingly trained on English and other highresource languages (Resnik, 2025). We are told that these frontier models are "multilingual," that they can speak to us in our local languages. But linguistic fluency must not be confused with cultural fidelity. As an act of resistance against this modern alienation, this research investigates whether Large Language Models genuinely grasp the depths of African cultures, centering on the Igbo language and its people. Spoken by roughly 30 million individuals in Southern Nigeria, Igbo carries a profound cultural heritage anchored in vibrant oral traditions and complex kinship networks. When an artificial intelligence processes fluent Igbo but maps it onto Western norms or pan-African stereotypes, it performs significant representational harm. It preserves surface meaning but strips away the deep cultural ontology embedded in the expression.

Cultural misalignment goes beyond bad test scores, threatening real-world sectors like education, public services, and safety (Gallegos et al., 2024). For communities like the Igbo, whose knowledge lives in oral traditions, algorithmic erasure replaces a vibrant history with generic foreign defaults (Joshi et al., 2020; Nekoto et al., 2020; Naous et al., 2024; Tonga et al., 2026). Cultural alignment is not an optional add-on. It is an absolute requirement for building safe and equitable AI (Dev et al., 2026; Zhou et al., 2025; Ouyang et al., 2022; Bai et al., 2022; Ryan et al., 2024; Orlowski et al., 2025).

To measure this gap between speaking a language and knowing a culture, this study introduces the Cultural Alignment Benchmark for Igbo (CAB-Igbo). Built from the ground up using verified knowledge from the Igbo community across six domains, this benchmark rejects the imposition of outside templates. By evaluating eight state-of-theart models under varying linguistic and persona conditions, and measuring responses against four dimensions (Epistemic Fidelity, Representational Richness, Linguistic Quality, and Pragmatic Proficiency), we seek to uncover whether these models truly reflect Igbo values or merely mimic our words while remaining strangers. We treat linguistic quality as a distinct dimension from epistemic fidelity to disentangle surface-level language proficiency from culturally grounded knowledge representation.

## 1.1 Research Questions

This raises important questions:

1. How well do language models understand Igbo culture, and what misalignment patterns appear in current models?

2. Does the evaluation language (English or Igbo) affect model performance, and does fluency imply cultural competence?

3. Does model scale determine cultural alignment, or is training data specificity the most important?

4. Do models trained on African data show better cultural alignment than English-centric ones?

## Our Contributions

1. We introduce CAB-Igbo, a novel, bilingual benchmark designed to measure cultural alignment in the Igbo language. Moving beyond surface-level translation, it evaluates deep cultural practices across six semantic domains of Igbo life.

2. We empirically demonstrate across eight large language models that high multiple-choice accuracy relies heavily on pattern matching, masking a severe inability to generate culturally grounded reasoning in open-ended settings.

3. We show that while LLM-as-a-judge correlates strongly with human evaluators on linguistic fluency, it systematically fails to assess deep cultural proxies.

## 2 Related Works

Early NLP evaluation relied on surface metrics. As models moved to general-purpose foundational models, evaluation has shifted towards full scope benchmarks. Tools like Global MMLU (Singh et al., 2025) and BIG-Bench measure general capabilities across domains. Model scale improved fluency but did not produce cultural depth. To fill gaps in multilingual evaluation, researchers used translation. Tools like Global MMLU (Singh et al., 2025) translated English benchmarks into other languages, including African ones. But language transfer is not culture transfer. Translating Western tests into African languages does not check for African understanding; it evaluates whether a model can process Western cultural logic in an African linguis tic form. This flattens differences and shows that speaking many languages is not the same as knowing local ways. Researchers then turned to surveys (Adilazuarda et al., 2024). Benchmarks like CDEval (Wang et al., 2023) and Hofstede-based analyses (Masoud et al., 2023) use the World Values Survey and Hofstedes dimensions. They ask models multiple-choice questions from one groups point of view. Reducing dynamic cultural practices to fixed categorical value dimensions fails to capture situated social realities. Culture is not a checklist. Furthermore, recent empirical studies show that persona prompting across different languages is highly unstable (Khan et al., 2025; AlKhamissi et al., 2024). Models fall back on shallow stereotypes rather than reasoning about culture. Translation and surveys force outside templates. Native benchmarks respond by using datasets like BLEnD (Myung et al., 2024) and CulturalBench (Chiu et al., 2025), which let locals ask their own questions about their culture and everyday life. But BLEnD has items not true to local Northern Nige rian daily living, e.g., “What is the main dish for Thanksgiving in Northern Nigeria?” and “What do people do to celebrate Halloween in Northern Nigeria?” Neither practice exists in these commu nities. Adaptability benchmarks such as NormAd (Rao et al., 2025) use real-life scenarios. Recent benchmarks prove that cultural evaluation requires native curation. For example, Hew et al. (2025) evaluates Malaysian culture using paired closed and open-ended formats, finding a massive gap between multiple-choice guessing and generative reasoning. We build on MyCulture’s approach, expanding it with our diagnostic distractor taxonomy and intra-cultural personas. Similarly, XCR-Bench introduces a multi-task framework to evaluate cross-cultural adaptation, demonstrating that models struggle with implicit social norms even when surface-level translations are accurate (Kabir et al., 2026). Recent large-scale studies on persona prompting have also revealed that injecting demographic variables often yields unstable or negative effects rather than improving cultural reasoning (Khan et al., 2025; AlKhamissi et al., 2024). Finally, datasets like NRITYAM highlight the necessity of expert, community-validated datasets for evaluating culturally grounded knowledge in specific domains (Singh et al., 2026).

## 3 CAB-Igbo Dataset

CAB-Igbo is a bilingual (English-Igbo) evaluation task. It consists of 296 unique cultural question paradigms. Because each paradigm is translated and evaluated in both English and Igbo, this forms a core dataset of 592 items. Every question is asked in two paired formats. As a multiple-choice question (MCQ), it has a cultural premise and four answer choices, which include diagnostic distractors such as Western defaults, ethnolinguistic substitutions, and Pan-African generalisations. As an open-ended short-answer question (SAQ), the same question is asked without any answer choices, and the model must explain the cultural reasoning in its own words in two to four sentences. All instances are drawn from authentic Igbo community sources. Native speakers verify every question. The correct answer depends on culturally grounded context, not surface-level pattern matching or simple language fluency. Because the SAQ removes the answer choices, CAB-Igbo tests not only whether a model can recognise the right answer but also whether it can explain it. This makes CAB-Igbo a valuable benchmark for evaluating whether an LLM can achieve true cultural alignment and incorporate authentic indigenous knowledge in low-resource language tasks.

## 3.1 Dataset Construction

CAB-Igbo is built from scratch using expert-guided primary sources rather than web scraping, which reduces the likelihood of direct pre-training data leakage. However, we do not claim this eliminates leakage risk entirely: common proverbs or widely known cultural practices referenced in our items may still overlap with content present in pretraining corpora. Our diagnostic distractors and open-ended reasoning prompts are novel constructions not found in any public source, which limits the extent to which simple memorization alone could explain strong performance. The dataset captures regional nuances from Imo and Ebonyi states. To guarantee quality, we used a participatory method, checking every item for factual accuracy, language flow, and cultural fit. If the community did not reach consensus, the item was removed.

## 3.2 Evaluator Requirements

We engaged expert Igbo linguists. We also leveraged the expertise of native Igbo-speaking evaluators with community-level cultural knowledge. They provided authentic cultural expertise. The linguist rigorously verified every line of Igbo in the benchmark. Ensuring it met the highest language standard. Native speakers validated the dataset using a participatory method. This ensured the benchmark was rooted in real community knowledge rather than outside guesses or machine-generated content. Each item was reviewed for factual accuracy, language quality, and cultural fit. Any items lacking consensus among the speakers were modified or removed.

## 3.3 Regional and Domain Selection

To ensure broad and authentic coverage of the Igbo worldview (Nwoye, 2011), we structured the benchmark around six key domains. These are: Family and Kinship, Festivals and Religion, Food and Cuisine, Funerals and Burial, Proverbs and Oral Tradition, and Social Customs and Etiquette. We created cultural personas based on two distinct Igbo states: Imo and Ebonyi. Imo State was prioritized because it serves as the language reference point for Igbo Izugbe, or Standard Central Igbo. It is also highly represented in digital resources. Ebonyi State was selected for a different reason. It features an underrepresented dialect. This allowed us to introduce language variation and test whether models could actually differentiate intra-Igbo regional identities, rather than just matching standard digital patterns.

<table><tr><td>CULTURAL DOMAIN</td><td>ENGLISH PROMPT</td><td>IGBO PROMPT</td></tr><tr><td>Proverbs and Oral Tradition</td><td>What practical life lesson is contained in the proverb, &quot;It is the height that a child&#x27;s hand gets to that he hangs his bag&quot;?</td><td>Kedu nkuzi ndu bara uru d n&#x27;ime ilu a: &quot;Ebe eka nwatà ru ke o konto ekpa&quot;?</td></tr><tr><td>Family and Kinship</td><td>In Igbo kinship and hospitality rituals, how is an j Igbo (kola nut) that breaks into exactly five cotyledons interpreted by the family?</td><td>N&#x27;omenala mmekorita ezinlo na ile obia nd Igbo, kedu ka ezinulo si akowa Oj Igbo gbara kpomkwem ibe ise?</td></tr><tr><td>Funerals and Burial</td><td>In traditional Igbo cosmology, which of the following criteria defines the attainment of a &quot;good death&quot; (Onw chi)?</td><td>N&#x27;odinala na amamihe okike nd Igbo, olee n&#x27;ime onod nd a na-akowa inweta &quot;ezi nw&quot; (nw chi)?</td></tr></table>

Figure 1: Samples from the CAB-Igbo Dataset. Questions are grounded in authentic cultural domains and evaluate deep epistemic fidelity rather than mere linguistic translation.

## 3.4 Question Formatting and Data Statistics

We considered writing separate items for different testing formats. However, we chose a stricter approach. Following a similar evaluation methodology utilized in recent cultural benchmarks (Hew et al., 2025), we ask the exact same question in two distinct formats. CAB-Igbo presents 296 core cultural paradigms as multiple-choice questions (MCQs) and as open-ended short answer questions (SAQs) (see Figure 1 for examples). This paired design tests whether models can do more than surface-level pattern matching. It reveals whether a model can correctly pick the right option and still explain the cultural reasoning behind it without help from the provided choices. This process produced 296 unique question pairs per language. Across both English and Igbo, this creates a foundational dataset of 592 unique items. Because each is evaluated in both multiple-choice and open-ended formats, it yields 1,184 distinct testing conditions before any persona constraints are applied. We did not use broad, country-level distributions. Instead, CAB-Igbo captures intra-cultural variations. The 296 core paradigms scale into a comprehensive evaluation grid. We multiply them across three linguistic prompt types: English, English-Igbo, and Igbo-Igbo. We also include 11 persona conditions and two answer permutations. The exact text of every prompt template used in our experiments is given in Appendix D. This architecture generates tens of thousands of test instances. During evaluation, it yielded 78,144 closed-form responses under the original answer ordering, and 1,416 open-ended generations on a 20% sample of the benchmark (59 questions), of which 1,415 were human-scored.

## 4 Experimental Setup

We conducted experiments across 8 large language models. We grouped them into four distinct categories to assess the impact of scale, training data, and multilingual focus. The first group is an Africa-specific open model, AfriqueQwen 14B (Yu et al., 2026). The second group includes multilingual open models: LLaMA 3.3 70B and LLaMA 3.1 8B. The third group includes general-purpose open models: LLaMA 3 8B Lite, GPT-OSS 120B, and GPT-OSS 20B. The fourth group includes closed-weight frontier models: GPT-4o mini and Gemini 2.0 Flash. All experiments used zero temperature and greedy decoding. This enforced deterministic and factual outputs. For GPT-OSS 120B and GPT-OSS 20B, responses were limited to 300 output tokens for MCQs and 500 for SAQs. These are reasoning models whose internal reasoning counts toward this limit, and in our first run the reasoning often used up the whole budget, so the model returned an empty MCQ response: 5,640 of 19,536 responses for GPT-OSS 120B and 11,826 of 19,536 for GPT-OSS 20B. We therefore ran a second pass for GPT-OSS 120B in which only these empty items were queried again with a limit of 4,000 tokens, using the same provider, prompts, decoding settings, and option orderings. Every re-queried item returned a valid option letter, and no response from the first run was changed. GPT-OSS 20B could not be re-queried in the same way because the provider no longer offered the model on its shared service; its MCQ results therefore come from the first run, and its empty responses are scored as incorrect. We conducted experiments across three linguistic prompt types. Type 1 is English-English. Type 2 is English-Igbo. Type 3 is Igbo-Igbo. We evaluated the models using two paired strategies. The first is multiple-choice questions (MCQs). The second is open-ended short answer questions (SAQs). For MCQs, we assigned alphabetical labels to the answer choices: A, B, C, and D. We ran each MCQ through two answer permutations to test for positional bias. The first permutation kept the original written order. The second used a fixed-seed shuffled order. While full permutation sweeps (4! = 24 combinations) would provide a more exhaustive estimate of po sitional bias, computational and API constraints limited us to these two conditions, which were sufficient to establish the baseline presence of bias. For the open-ended approach, we prompted models to generate a free-form explanation. We restricted responses to two to four sentences. A native human judge then evaluated these responses across four constructs. These are Epistemic Fi delity, Representational Richness, Linguistic Quality, and Pragmatic Proficiency. This allowed us to assess thick cultural reasoning (Geertz, 1973). Cultural knowledge varies across locations and demo graphics. We hypothesized that providing specific socio-geographical context can affect a model’s reasoning ability. To test this, we complemented our zero-shot baseline experiments with ten different persona conditions. These represented five distinct socio-economic roles. Examples include a young teenage boy, a married woman, and a professor. We mapped these roles across two specific Igbo states: Imo and Ebonyi. This setup allowed us to measure whether models could differentiate intra-Igbo regional identities. It also tested whether they would default to homogeneous generalizations.

## 4.1 Statistical Methodology

To rigorously evaluate the significance of performance differences across conditions, we applied two distinct statistical approaches tailored to the evaluation formats. For multiple-choice questions (MCQs), which yield paired binary outcomes (correct vs. incorrect on the same item under two conditions), we employed McNemar’s test with Yates continuity correction. This was used to assess the significance of prompt type degradation (English vs. Igbo), positional bias (original vs. shuffled option ordering), and persona effects (baseline vs.

demographic persona). For the open-ended short answer questions (SAQs), which were scored by human evaluators on ordinal 0-2 scales across four constructs, we computed 95% bootstrap confidence intervals (using 10,000 resamples) on the mean within-item score differences (∆) between the English baseline and the Igbo prompt conditions. Differences were considered statistically significant if the 95% confidence interval did not include zero. This non-parametric approach avoids strict distributional assumptions and provides robust estimates for the shift in generative cultural reasoning.

## 5 Analysis

## Result by Categories

We evaluate the performance of eight selected language models on CAB-Igbo. We provide a detailed analysis across six cultural domains and three linguistic prompt types. We further categorize each sample by its regional condition. This allows us to assess intra-Igbo cultural differentiation between Imo and Ebonyi. We structure our analysis in three parts. First, we examine baseline MCQ accuracy. This establishes how well models perform at surface-level cultural recognition. Second, we evaluate performance on open-ended short answer questions. We score these using our multi-dimensional human evaluation rubric and an LLM-as-a-judge protocol. This assesses their capacity for authentic cultural reasoning. Third, we synthesize these findings. This synthesis exposes the central finding of our study: the gap between multiple-choice pattern matching and true cultural generation.

## 5.1 MCQs

## Linguistic Prompt Type Performance

We tested three conditions: Type 1, Type 2, and Type 3. For Gemini 2.0 Flash, the effect is small, dropping by only 1.1 percentage points. GPT-4o mini drops more with 7.1 points. Interestingly, it does slightly better on full Igbo than when it receives mixed English–Igbo prompts. LLaMA 3.3 70B scores 99.0% on Type 1 but then falls to 46.1% on Type 3. This is a massive drop of 52.9 points. This is the "fluent but foreign" effect described by Agarwal et al. (2025), where models seem fluent but misaligned (see Figure 2). AfriqueQwen 14B is as sensitive to language as LLaMA 3.3 70B, but steadier, dropping about 25.1 points from 99.5% on Type 1 to 74.4% on Type 3. McNemar’s tests on the zero-shot baseline items (Type 1 vs Type 2, original ordering; Table 5) confirm that the Englishto-Igbo degradation is statistically significant for every model tested (all $p < 0 . 0 5 , \chi ^ { 2 }$ ranging from 5.14 for Gemini 2.0 Flash to 164.28 for GPT-OSS 20B), establishing that the “fluent but foreign” collapse is not an artefact of a few unlucky items but a systematic effect across the benchmark.

## Effect of Persona

Table 3 asks a more direct question: does persona prompting help or hurt the model? It compares zero-shot baseline accuracy with mean accuracy across ten persona conditions for each model. GPT-4o mini gains 1.6 points under persona conditions, a modest but consistent improvement. For Afrique Qwen 14B and LLaMA 3.3 70B, persona actively lowers accuracy by introducing demographic noise into the given prompt. GPT-OSS 120B is almost unaffected, changing by only 0.3 points. GPT OSS 20B loses 8.5 points, the biggest decline we saw, though its many empty responses (Section 4) blur the line between persona effects and output breakdown. Where other models either stay flat or decline, LLaMA 3.1 8B gains 10.4 points un der persona conditions, the only large positive shift we saw in Table 3. Table 4 compares how each model responds to different demographic personas. Persona identity has little power over cultural ac curacy. LLaMA 3.1 8B shows the widest gap at 4.7 points, between the Professor and Widowed Woman personas. Frontier models differ by at most 0.8 points. Our persona experiments also align with recent large-scale studies showing that demo graphic prompting often hurts rather than helps (Khan et al., 2025; AlKhamissi et al., 2024). A per sona prompt cannot magically inject missing cul tural knowledge. As observed in the data regarding the inter-persona delta, LLaMA 3.1 8B improves under every persona condition, gaining between 8.1 to 12.8 percentage points above its zero-shot base line. The Professor persona provides the biggest gain at 12.8 points. However, this does not prove that the model has actual lived cultural knowledge; instead, the persona works as a simple prompt that pushes the model to answer more carefully. AfriqueQwen 14B shows the exact reverse. Every persona reduces its accuracy by roughly five to six points from its 93.2% baseline. Its Africa-specific training may have optimised it for direct questions without additional persona framing. McNemar’s tests confirm this asymmetry statistically (Table 5). Each persona condition was compared with the zero-shot baseline on the same 888 question and prompt type pairs in the original ordering, the same items that underlie Table 3. Persona effects are significant for all ten personas for AfriqueQwen 14B (drops; $\chi ^ { 2 }$ from 21.33 to 36.01), LLaMA 3.1 8B $( \mathrm { g a i n s } ; \chi ^ { 2 }$ from 20.05 to 69.14), and GPT-OSS 20B (drops; $\chi ^ { 2 }$ from 17.33 to 46.52), for nine of ten personas for LLaMA 3.3 70B (drops), and for seven of ten for GPT-4o mini (gains). For Gemini 2.0 Flash, GPT-OSS 120B, and LLaMA 3 8B Lite, no persona condition differs significantly from baseline (all $p > 0 . 0 5 )$ . Persona framing therefore shifts accuracy systematically for most models, but the direction depends on the model, and no persona produces a consistent benefit across models.

## State-Level Analysis

## Performance Gap Across States

Table 2 records accuracy for each model under two state-level persona conditions: Imo and Ebonyi. Can models tell Imo from Ebonyi at a cultural level? If models held regional knowledge, the Imo condition should significantly outperform Ebonyi because it represents the standard Igbo Izugbe heavily found in digital texts. Instead, as the data shows, the gaps between Imo and Ebonyi range from a mere 0.1 to 0.6 points, indicating that performance lines run nearly parallel. Models treat state labels as empty markers: they register the words but encode no difference in dialect, culture, or local knowledge. MCQs may be the wrong tool to detect persona effects and state-level differences. Because the correct option is the same for both Imo and Ebonyi, a model needs only general plausibility to score well, requiring no true regional knowledge.

## The Effect of Positional Bias

Descriptively, six of the eight models showed a raw point drop under shuffled ordering. McNemar’s tests (with continuity correction) on the zero-shot baseline items, pairing the original and shuffled ordering of each of the 888 question and prompt type items, show that the drop is significant for four models: GPT-4o mini $( \chi ^ { 2 } = 1 6 . 6 0 , p < 0 . 0 0 1 )$ LLaMA 3.1 8B $( \chi ^ { 2 } = 1 1 . 5 2 , p < 0 . 0 0 1 )$ , Gemini 2.0 Flash $( \chi ^ { 2 } = 5 . 2 6 , p = 0 . 0 2 2 )$ , and LLaMA 3.3 70B (χ<sup>2</sup> = 4.34, p = 0.037). LLaMA 3 8B Lite is borderline $( \chi ^ { 2 } = 3 . 3 6 , p = 0 . 0 6 7 )$ . For AfriqueQwen 14B and both GPT-OSS models, the difference is not significant (Table 5). Positional bias is therefore present in several models, including both frontier models, which suggests that part of their MCQ accuracy depends on option order rather than on cultural knowledge.

![](images/c092268fee55f3b1140b18a8a3ffa286f65ea35e1af09775682a394abe75891a.jpg)  
Type 1: English instruction and question. Type 2: English instruction, Igbo question. Type 3: Igbo instruction and question.  
Figure 2: Prompt Degradation: The collapse of cultural reasoning across English and Igbo prompt types.

## 5.2 Human Evaluation

## Open-Ended Evaluation Overview

We graded open-ended answers on four criteria. Each measure was scored between 0 to 2. The four are Epistemic Fidelity (EF), Representational Richness (RR), Linguistic Quality (LQ), and Pragmatic Proficiency (PP). A response has a maximum of 8 points total. We evaluated a 20% sample of the benchmark (59 questions), giving 1,416 generations, of which 1,415 were scored. We tested all eight models across all three prompt types (under zero-shot conditions only). Each construct measures something different: EF and RR test how deep and true the cultural knowledge is, LQ tests whether the Igbo grammar flows and is readable, and PP tests whether the answer fits the required situation.

## Human Evaluation Scores by Model

Table 7 reports mean scores per model aggregated across all three prompt types (the aggregate scores are also summarized visually in Figure 4). Gemini 2.0 Flash stands out from all other models. Its mean total score is 4.316 out of 8. Its EF and RR scores are 0.531. Gemini’s knowledge scores are five times higher than the next best model. Four models sit in the middle tier: GPT-4o mini, GPT-OSS 120B, LLaMA 3.3 70B, and AfriqueQwen 14B (Yu et al., 2026). Their total scores range from 2.859 to 3.147, making them nearly identical. Their LQ scores reach near maximum (1.740 to 2.000), meaning their Igbo sounds smooth and correct. But EF and RR are stuck exceptionally low, between 0.062 to 0.102. In this middle group, size and training source do not matter for cultural knowledge. The lowest tier contains three models: GPT-OSS 20B, LLaMA 3.1 8B, and LLaMA 3 8B Lite. Their total scores range from 1.045 to 2.131. LLaMA 3 8B Lite scores just 1.045 on average across all conditions. We note that EF and RR scores are identical not only on average but for every individual response in our sample. We argue this reflects a genuine property of cultural knowledge rather than rubric redundancy: representational richness without epistemic accuracy is not depth but elaboration on a false premise. A response cannot meaningfully demonstrate rich, nuanced cultural understanding (RR) while being factually or culturally incorrect (low EF), since the richness itself would be built on a mistaken premise. In our benchmark, where each item targets a specific, verifiable cultural referent, this dependency manifests as near-total convergence between the two constructs. This suggests EF may function as a gating condition for RR in culturally grounded evaluation, a relationship worth testing explicitly in future work with items designed to decouple the two (e.g., responses that are accurate but sparse, versus elaborate but wrong).

![](images/00572deb50ccbcc88615922141415d61a42be85a0594568c8daea5cfb90b608d.jpg)  
Figure 3: The Limits of Automated Evaluation: Comparing human evaluator scores against the Gemini 2.0 automated judge. The automated judge significantly overestimates Epistemic Fidelity and Representational Richness compared to native human experts.

## Prompt Type Effects in Open-Ended Evaluation

Table 6 shows mean scores by model and prompt type. Table 8 reports bootstrap 95% confidence intervals (10,000 resamples) on the within-item difference between Type 1 and Type 2 scores for each construct. For six of the eight models (Afrique-Qwen 14B, GPT-4o mini, GPT-OSS 120B, GPT-OSS 20B, LLaMA 3.3 70B, and LLaMA 3 8B Lite), the change in Epistemic Fidelity and Representational Richness from English to Igbo prompting is not statistically significant (95% CI includes zero). In contrast, Linguistic Quality drops significantly for five models (AfriqueQwen 14B, GPT-OSS 20B, LLaMA 3.3 70B, LLaMA 3.1 8B, and LLaMA 3 8B Lite) and Pragmatic Proficiency for three (GPT-OSS 20B, LLaMA 3.1 8B, and LLaMA 3 8B Lite). GPT-4o mini and GPT-OSS 120B show no significant change on any construct. This dissociation is informative: for most models, low EF and RR are not a consequence of switching languages, but reflect a knowledge gap already present when models are prompted in English. Igbo prompting introduces a separate failure, a decline in fluent generation, rather than causing the cultural knowledge gap itself. The non-significance of EF and RR changes should not be read as a positive result. It indicates that the absence of cultural knowledge is already present in English and is not a translation artefact. For LLaMA 3 8B Lite, it also reflects a floor effect, since its Type 1 EF is already 0.017.

Two models depart from this pattern. LLaMA 3.1 8B is the only model whose EF and RR drop significantly $( \Delta = 0 . 1 0 )$ , alongside much larger LQ and PP collapses, consistent with its breakdown under Igbo input (see Section 5.4). Gemini 2.0 Flash moves in the opposite direction: its EF and RR rise significantly under Igbo prompting, from 0.237 on Type 1 to 0.576 on Type 2 and 0.780 on Type 3 (∆ = −0.34, 95% CI [−0.58, −0.10] for Type 1 vs Type 2), and its PP rises as well $( \Delta = - 0 . 1 9 )$ Gemini 2.0 Flash is the only model whose cultural knowledge scores improve when the question is posed in Igbo.

## 5.3 LLM-as-a-Judge Evaluation

Table 9 reports mean scores assigned by Gemini 2.0 Flash as an automated judge to the openended responses of seven models. Each model was judged on 177 responses (59 questions across all three prompt types), giving 1,239 evaluations, of which 1,238 have a matching human score. We selected Gemini 2.0 Flash as the judge due to its strong multilingual instruction-following capabilities. Crucially, to ensure strict independence and completely avoid self-preference bias or style alignment distortion, we deliberately excluded Gemini 2.0 Flash’s own generations from this adjudication phase. Thus, for the LLM-as-a-judge experiment, Gemini functioned entirely as a distinct evaluator that was not among the tested systems being scored. Column n records the number of matched responses per model. GPT-OSS 120B earns the highest judge total and the highest EF, while GPT-4o mini achieves a perfect 2.000 on LQ and PP across all responses. The two smallest models, LLaMA 3.1 8B and LLaMA 3 8B Lite, score lowest on every construct. Crucially, when judge scores are placed alongside the human scores for the same 1,238 responses (Table 10), the automated judge awards higher scores than the human annotator on every construct (visualized in Figure 3). The gap is smallest for Linguistic Quality. The largest discrepancies appear in Epistemic Fidelity, Representational Richness, and Pragmatic Proficiency.

![](images/3a924a80c3d4a5578bb237787a114bcc08f72f570cbe815ae3854981c813ac05.jpg)  
Figure 4: Aggregate Human Evaluation Scores. Gemini 2.0 Flash significantly outperforms all other models, while smaller open-weight models suffer near-complete breakdown in culturally grounded contexts.

## Inter-Rater Agreement

Table 10 reports inter-rater agreement between the human annotator and the Gemini 2.0 Flash judge across 1,238 matched response pairs, using Cohen’s weighted κ and Spearman’s $\rho .$ Linguistic Quality shows near-perfect agreement $( \kappa _ { w } =$ 0.833, $\rho = 0 . 8 8 5$ , exact agreement 90.9%), confirming that surface fluency is reliably measurable by automated judges. Epistemic Fidelity and Representational Richness show poor agreement $( \kappa _ { w } \ < \ 0 . 1 1 5 )$ . The judge is systematically too lenient on deep cultural content. Pragmatic Proficiency presents a split result: $\rho = 0 . 8 8 3$ suggests strong rank-ordering, but $\kappa _ { w } ~ = ~ 0 . 2 6 1$ and just 22.1% exact agreement reveal that the judge scores PP 0.792 points higher on average. These results establish human evaluation as the gold standard for cultural depth: automated judges can reliably assess fluency but fail on epistemic fidelity.

## 5.4 Error Analysis

Thirteen error categories were applied during human evaluation: Hallucination, Shallow Description, Model Breakdown, Repetition, Western Default, Nonsensical Response, Empty Response, Incomplete Response, Culturally Accurate, Pan-African Generalisation, Ethnolinguistic Substitution, Wrong Language, and Contradicts Prompt. A single response often receives multiple labels. The two smallest models collapse entirely under Igbolanguage input. LLaMA 3 8B Lite averages 3.034 on Type 1 but drops to 0.000 on Type 2 and 0.102 on Type 3. LLaMA 3.1 8B follows the same pattern, falling from 2.983 (Type 1) to 0.153 (Type 3), a 94.9% drop. In Type 1, Hallucination is the most frequent error, followed by Shallow Description and Western Default. Across Type 2 and Type 3, Western Default declines as models can no longer produce coherent output, while Model Breakdown and Repetition rise sharply, from 17 and 0 labelled responses in Type 1 to 158 and 124 in Type 3 (see Figure 7 in the Appendix for examples of repetitionloop failures). Figure 6 shows the primary verdict for each answer: in Type 1, 93% of answers are primarily hallucinations and 4% are correct; under Igbo input, 27% (Type 2) and 33% (Type 3) of answers are primarily model breakdowns, and no prompt type exceeds 7% correct. To illustrate how the distractor taxonomy targets these failure modes, consider the communal proverb “E kechaa n’obi eke na mkpuke”, which affirms that resources are shared first on the basis of the extended family and then on the basis of the nuclear family. Its Western Default distractor reads: “Resources must be strictly divided into equal individual portions, disregarding family hierarchies, to promote absolute egalitarianism.” This option replaces the Igbo family hierarchy with Western individualism, so a model that selects it reveals a specific misunderstanding rather than a generic error. Pan-African Generalisation distractors work in the same way, by replacing specific Igbo practices with broad continental stereotypes that strip away their localized meaning.

![](images/52f627b124d863a7dcd3b1a60920a86fc6f656528f7f60c4530e03047cba8a24.jpg)  
Figure 5: The gap between recognising and explaining Igbo culture. For each model, on the same 59 questions in the zero-shot condition across all three prompt types (177 items per model; 176 open-ended answers for GPT-OSS 20B): the share of MCQ items answered correctly (original ordering) and the share of open-ended answers rated fully correct by the human annotator (Epistemic Fidelity = 2).

## 5.5 Cross-Model Synthesis

The central empirical finding is not low MCQ accuracy but the magnitude of the gap between MCQ performance and open-ended cultural generation. Figure 5 places the two formats side by side on the same 59 questions in the zero-shot condition. GPT-4o mini selects the correct option for 87.0% of MCQ items but produces a fully correct explanation (EF = 2) for only 2.3% of open-ended answers. Even Gemini 2.0 Flash, the strongest model, reaches 97.2% on MCQs but only 26.0% fully correct explanations, and no other model exceeds 4.5%. Over the full benchmark, GPT-4o mini achieves 93.2% overall MCQ accuracy while producing Western Default explanations in openended tasks. AfriqueQwen 14B scores 88.2% on

MCQs yet generates invented social labels and incorrect ritual meanings. LLaMA 3 8B Lite reaches 71.8% MCQ accuracy but shows repeated generation failure under Igbo prompting. GPT-OSS 120B shows the same pattern: it ranks third in MCQ accuracy at 89.9% over the full benchmark, yet on the 59 human-evaluated questions only 3.4% of its open-ended answers are fully correct. This confirms that multiple-choice formats reward pattern matching, not genuine cultural reasoning. Model scale does not determine cultural alignment. GPT-OSS 120B is the largest open model, but its high MCQ accuracy does not carry over to open-ended cultural reasoning. LLaMA 3.3 70B reaches 99.0% on Type 1 MCQs but collapses to 46.1% on Type 3. This 52.9 percentage point drop is the largest degradation observed and is consistent with the “fluent but foreign” phenomenon described by Agarwal et al. (2025). Our discovery that models can sound fluent without understanding culture also mirrors the findings in XCR-Bench (Kabir et al., 2026). Surface fluency does not guarantee deep cultural grounding. The two best models on both evaluation tracks (Gemini 2.0 Flash and GPT-4o mini) are proprietary models accessible only through paid commercial APIs. The communities most harmed by cultural misrepresentation are the same lowresource communities for whom such access is prohibitive. This dissociation between knowledge (EF/RR) and fluency (LQ/PP) reinforces our central claim: cultural misalignment in current LLMs is not primarily a multilingual generation problem. Even prompted entirely in English, most models fail to produce culturally grounded reasoning at rates statistically indistinguishable from their Igboprompted outputs; Gemini 2.0 Flash is the only model whose cultural knowledge scores improve in Igbo. The models have not learned Igbo culture in any language.

![](images/6e4d9f94353c99daf3068d26f111f2d63c0cc98c07829985c9911d31d5d6963c.jpg)  
Figure 6: Primary human verdict on each of the 1,415 scored open-ended answers, by prompt type. The verdict is the first error label the annotator assigned; Correct covers answers labelled Culturally Accurate or with no error recorded. Other covers Western Default, Repetition, and Incomplete Response as primary labels.

## 6 Conclusion

We introduced CAB-Igbo, a benchmark for evaluating cultural alignment in Igbo contexts. The dataset comprises 296 questions across two regional states and six cultural domains. All questions are drawn from authentic community sources and validated by native speakers. Evaluations on 8 LLMs reveal significant performance gaps. Open-weight and proprietary models both struggle with Igbo cultural contexts. They frequently default to Western norms or Pan-African generalizations. The gap between MCQ accuracy and open-ended reasoning is severe. While models can recognize patterns, they consistently struggle to generate culturally grounded explanations. This highlights the urgent need for culturally aware models built directly from community knowledge, rather than relying on translated Western concepts or superficial fluency. Addressing this gap is not just an academic exercise. It highlights the severe danger of confusing language fluency with cultural understanding. As AI expands globally, deploying models without cultural grounding risks concrete harm in education, public services, and safety. Cultural alignment is not a feature; it is the foundation for safe and trustworthy AI.

## Limitations

Statistical Significance & Positional Bias: Our significance tests treat items as independent and apply no correction for multiple comparisons across models and conditions, so results near the 0.05 threshold should be read with caution. Furthermore, evaluating with only two answer permutations is not robust enough to definitively rule out positional bias. Finally, we observed extremely high Type-1 MCQ accuracies near ceiling for frontier models. Future iterations should incorporate item-level difficulty analysis (e.g., Item Response Theory) to better calibrate English-prompted difficulty.

Two Evaluation Runs for the GPT-OSS Models: The MCQ results for GPT-OSS 120B combine two runs: the first run, and a second run that queried again only the 5,640 items whose first-run response was empty because the model’s reasoning exceeded the 300-token limit. The second run used a 4,000-token limit and otherwise identical settings; first-run responses that contained an answer were kept unchanged. GPT-OSS 20B could not be re-queried on the same service, so 11,826 of its 19,536 MCQ responses (60.5%) remain empty and are scored as incorrect. On the items it did answer (original ordering), GPT-OSS 20B selected the correct option 97.0% of the time, so its reported MCQ accuracy mainly reflects empty output rather than cultural knowledge, and we do not draw conclusions about cultural competence from it. The open-ended responses of both models come from the first run, with empty answers scored 0 as specified by the rubric.

Human Evaluation Constraints: Due to severe resource constraints, the open-ended evaluation was conducted by a single native-speaker expert linguist. The absence of inter-human agreement clouds our conclusions regarding construct-level score distributions, especially for Epistemic Fidelity and Representational Richness. In future work, we will conduct a dual-human calibration study on 100-200 items to cleanly separate LLM judge failures from single-rater biases.

Decoding Strategies & Prompting Gaps: We relied exclusively on zero-temperature greedy decoding to enforce deterministic outputs. This may have disproportionately disadvantaged some models on the generative short-answer questions (SAQs). Robustness to varying decoding temperatures, few-shot prompting, or retrieval-augmented generation (RAG) was not probed. Additionally, because the open-ended generations were not evaluated under persona and state conditions, the interaction between demographic prompting and generative cultural fidelity remains an underexplored gap.

The Influence of Standardisation and Dialects on Cultural Reasoning: To ensure a stable baseline, we utilized Igbo Izugbe, or Standard Central Igbo. This standardisation is heavily linked to our Imo State persona. However, it introduces a limitation: models that perform well on standard Igbo often struggle with dialectal or local knowledge. We included Ebonyi State to test intra-Igbo dialectal variation, but certain cultural concepts from other regions remain underrepresented. Preliminary evidence suggests that dialectal coverage materially changes Epistemic Fidelity and Representational Richness scores, making expansion beyond Imo and Ebonyi a critical next step.

Coverage of Regions and Evaluation Constraints Our study covers six foundational domains of Igbo cultural life. However, cultural knowledge is not static. It varies widely across communities. Due to severe resource, API, and GPU constraints, we restricted our human evaluation phase. We sampled only 20% of the full benchmark. This evaluation was conducted exclusively under zero-shot conditions. We did not assess the open-ended persona responses. Additionally, a single native speaker evaluator conducted the manual evaluation. This introduces potential subjectivity, despite the rigorous source-grounded rubric. Future work should expand the benchmark. It should include more underrepresented dialects, contemporary cultural practices, and broader native-speaker adjudication.

## Ethics and Data Availability

CAB-Igbo will be publicly released with open licenses, complete annotation guidelines, and evaluation scripts to enable external audits and future research. We maintained strict privacy and informed consent throughout the dataset creation. All native community contributors were fairly compensated for their expertise.

## GenAI Usage Disclosure

Large language models (GPT-4o mini, Gemini 2.0 Flash, LLaMA 3.3 70B, LLaMA 3.1 8B, LLaMA 3 8B Lite, GPT-OSS 120B, GPT-OSS 20B, and AfriqueQwen 14B) are the objects of study in this paper and were used exclusively to generate the evaluated responses described in Section 4. Additionally, Claude Sonnet 5 (Anthropic) was used to assist with code debugging, copy editing, and LaTeX formatting, under full author review and verification. No GenAI tool was used to generate, fabricate, or alter any cultural content, dataset item, or human evaluation score reported in this work.

## References

Muhammad Farid Adilazuarda, Sagnik Mukherjee, Pradhyumna Lavania, Siddhant Singh, Alham Fikri Aji, Jeremy O’Neill, Ashutosh Modi, and Monojit Choudhury. 2024. Towards measuring and modeling “culture” in LLMs: A survey. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), pages 15763– 15784. Association for Computational Linguistics.

Dhruv Agarwal, Anya Shukla, Sunayana Sitaram, and Aditya Vashistha. 2025. Fluent but foreign: Even regional llms lack cultural alignment.

Badr AlKhamissi, Muhammad N. ElNokrashy, and Mona T. Diab. 2024. Investigating cultural alignment of large language models: A case study of Egypt and the United States. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12404–12422, Bangkok, Thailand. Association for Computational Linguistics.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, John Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Chris Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, E Perez, and 32

others. 2022. Constitutional ai: Harmlessness from ai feedback. ArXiv, abs/2212.08073.

Yu Ying Chiu, Liwei Jiang, Bill Yuchen Lin, Chan Young Park, Shuyue Stella Li, Sahithya Ravi, Mehar Bhatia, Maria Antoniak, Yulia Tsvetkov, Vered Shwartz, and Yejin Choi. 2025. CulturalBench: A robust, diverse and challenging benchmark for measuring LMs’ cultural knowledge through human-AI red-teaming. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 25663– 25701, Vienna, Austria. Association for Computational Linguistics.

Sunipa Dev, Vinodkumar Prabhakaran, Rutledge Chin Feman, Aida Davani, Remi Denton, Charu Kalia, Piyawat Lertvittayakumjorn, Madhurima Maji, Rida Qadri, Negar Rostamzadeh, and 1 others. 2026. A unified framework to quantify cultural intelligence of ai. arXiv preprint arXiv:2603.01211.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Clifford Geertz. 1973. The Interpretation of Cultures. Basic Books.

Zhong Ken Hew, Jia Xin Low, Sze Jue Yang, and Chee Seng Chan. 2025. Myculture: Exploring malaysia’s diverse culture under low-resource language constraints. Preprint, arXiv:2508.05429.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings ofthe 58th Annual Meeting of the Associationfor Computational Linguistics, pages 6282–6293, Online. Association for Computational Linguistics.

Mohsinul Kabir, Tasnim Ahmed, Md Mezbaur Rahman, Shaoxiong Ji, Hassan Alhuzali, and Sophia Ananiadou. 2026. Xcr-bench: A multi-task benchmark for evaluating cultural reasoning in llms. Preprint, arXiv:2601.14063.

Ariba Khan, Stephen Casper, and Dylan Hadfield-Menell. 2025. Randomness, not representation: The unreliability of evaluating cultural alignment in llms. Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency.

Reem I. Masoud, Ziquan Liu, Martin Ferianc, Philip C. Treleaven, and Miguel Rodrigues. 2023. Cultural alignment in large language models: An explanatory analysis based on hofstede’s cultural dimensions. In International Conference on Computational Linguistics.

Junho Myung, Nayeon Lee, Yi Zhou, Jiho Jin, Rifki Afina Putri, Dimosthenis Antypas, Hsuvas

Borkakoty, Eunsu Kim, Carla Pérez-Almendros, Abinew Ali Ayele, V’ictor Guti’errez-Basulto, Yazm’in Ib’anez-Garc’ia, Hwaran Lee, Shamsuddeen Hassan Muhammad, Kiwoong Park, Anar Rzayev, Nina White, Seid Muhie Yimam, Mohammad Taher Pilehvar, and 3 others. 2024. Blend: A benchmark for llms on everyday knowledge in diverse cultures and languages. ArXiv, abs/2406.09948.

Tarek Naous, Michael J. Ryan, Alan Ritter, and Wei Xu. 2024. Having beer after prayer? measuring cultural bias in large language models. arXiv preprint arXiv:2305.14456.

Wilhelmina Nekoto and 1 others. 2020. Participatory research for low-resourced machine translation: A case study in African languages. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020.

Chinwe M. A. Nwoye. 2011. Igbo cultural and religious worldview: An insider’s perspective. International Journal of Sociology and Anthropology, 3(9):304– 317.

Eric J. W. Orlowski, Hakim Norhashim, and Tristan Koh Ly Wey. 2025. ’too much alignment; not enough culture’: Re-balancing cultural alignment practices in llms. ArXiv, abs/2509.26167.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Abhinav Rao, Akhila Yerukola, Vishwa Shah, Katharina Reinecke, and Maarten Sap. 2025. NormAd: A framework for measuring the cultural adaptability of large language models. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2373–2403, Albuquerque, New Mexico. Association for Computational Linguistics.

Philip Resnik. 2025. Large language models are biased because they are large language models. Computational Linguistics, 51(3):885–906.

Michael J. Ryan, William Held, and Diyi Yang. 2024. Unintended impacts of llm alignment on global representation. Preprint, arXiv:2402.15018.

Punit Kumar Singh and 1 others. 2026. Nrityam: Language models meet art and heritage of dance. Preprint, arXiv:2606.19727.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, Raymond Ng,

Shayne Longpre, Wei-Yin Ko, Sebastian Ruder, Martin Smith, Antoine Bosselut, Alice Oh, André Martins, Leshem Choshen, and 5 others. 2025. Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 18761–18799, Vienna, Austria. Association for Computational Linguistics.

Junior Cedric Tonga, Chen Cecilia Liu, Iryna Gurevych, and Fajri Koto. 2026. LLMs as cultural archives: Cultural commonsense knowledge graph extraction. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6289–6309, Rabat, Morocco. Association for Computational Linguistics.

Yuhang Wang, Yanxu Zhu, Chao Kong, Shuyu Wei, Xiaoyuan Yi, Xing Xie, and Jitao Sang. 2023. Cdeval: A benchmark for measuring the cultural dimensions of large language models. ArXiv, abs/2311.16421.

Hao Yu, Tianyi Xu, Michael A. Hedderich, Wassim Hamidouche, Syed Waqas Zamir, and David Ifeoluwa Adelani. 2026. Afriquellm: How data mixing and model architecture impact continued pre-training for african languages. Preprint, arXiv:2601.06395.

Zhixiao Zhao and Dongbo Wang. 2025. Evaluation of large language models for the intangible cultural heritage domain. npj Heritage Science, 13.

Naitian Zhou, David Bamman, and Isaac L Bleaman. 2025. Culture is not trivia: Sociocultural theory for cultural nlp. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25869–25886.

## A Detailed Results

## A.1 Multiple-choice accuracy

Tables 1 to 4 report MCQ accuracy in the original answer ordering.

<table><tr><td>Model</td><td>Type 1</td><td>Type 2</td><td>Type 3</td></tr><tr><td>Gemini 2.0 Flash</td><td>99.2</td><td>97.4</td><td>98.1</td></tr><tr><td>GPT-4o mini</td><td>98.2</td><td>90.2</td><td>91.1</td></tr><tr><td>AfriqueQwen 14B</td><td>99.5</td><td>90.7</td><td>74.4</td></tr><tr><td>LLaMA 3.3 70B</td><td>99.0</td><td>91.6</td><td>46.1</td></tr><tr><td>LLaMA 3 8B Lite</td><td>96.8</td><td>53.7</td><td>64.8</td></tr><tr><td>LLaMA 3.1 8B</td><td>96.3</td><td>60.0</td><td>51.5</td></tr><tr><td>GPT-OSS 120B</td><td>96.4</td><td>86.7</td><td>86.6</td></tr><tr><td>GPT-OSS 20B</td><td>73.0</td><td>12.9</td><td>28.7</td></tr></table>

Table 1: MCQ accuracy (%) by linguistic prompt type, pooled over the baseline and all persona conditions.

<table><tr><td>Model</td><td>Baseline</td><td>Imo</td><td>Ebonyi</td></tr><tr><td>Gemini 2.0 Flash</td><td>98.2</td><td>98.2</td><td>98.3</td></tr><tr><td>GPT-4o mini</td><td>91.7</td><td>93.5</td><td>93.2</td></tr><tr><td>AfriqueQwen 14B</td><td>93.2</td><td>87.7</td><td>87.8</td></tr><tr><td>LLaMA 3.3 70B</td><td>81.5</td><td>78.7</td><td>78.5</td></tr><tr><td>LLaMA 3 8B Lite</td><td>71.6</td><td>71.6</td><td>71.9</td></tr><tr><td>LLaMA 3.1 8B</td><td>59.8</td><td>70.1</td><td>70.3</td></tr><tr><td>GPT-OSS 120B</td><td>89.6</td><td>90.0</td><td>89.9</td></tr><tr><td>GPT-OSS 20B</td><td>45.9</td><td>37.7</td><td>37.1</td></tr></table>

Table 2: MCQ accuracy (%) for the zero-shot baseline and for the Imo and Ebonyi conditions, each pooled over the five personas.

<table><tr><td>Model</td><td>Baseline</td><td>All personas</td><td>Delta</td></tr><tr><td>Gemini 2.0 Flash</td><td>98.2</td><td>98.3</td><td>+0.1</td></tr><tr><td>GPT-4o mini</td><td>91.7</td><td>93.3</td><td>+1.6</td></tr><tr><td>AfriqueQwen 14B</td><td>93.2</td><td>87.7</td><td>-5.5</td></tr><tr><td>LLaMA 3.3 70B</td><td>81.5</td><td>78.6</td><td>-2.9</td></tr><tr><td>LLaMA 3 8B Lite</td><td>71.6</td><td>71.8</td><td>+0.2</td></tr><tr><td>LLaMA 3.1 8B</td><td>59.8</td><td>70.2</td><td>+10.4</td></tr><tr><td>GPT-OSS 120B</td><td>89.6</td><td>89.9</td><td>+0.3</td></tr><tr><td>GPT-OSS 20B</td><td>45.9</td><td>37.4</td><td>-8.5</td></tr></table>

Table 3: MCQ accuracy (%) under the zero-shot baseline and pooled over all ten persona conditions; Delta is in percentage points.

<table><tr><td>Model</td><td>BL</td><td>YTB</td><td>MW</td><td>MM</td><td>WW</td><td>Prof</td></tr><tr><td>Gemini 2.0 Flash</td><td>98.2</td><td>98.3</td><td>98.5</td><td>98.1</td><td>98.1</td><td>98.3</td></tr><tr><td>GPT-4o mini</td><td>91.7</td><td>93.3</td><td>92.8</td><td>93.2</td><td>93.6</td><td>93.6</td></tr><tr><td>AfriqueQwen 14B</td><td>93.2</td><td>88.5</td><td>87.5</td><td>87.7</td><td>87.8</td><td>87.1</td></tr><tr><td>LLaMA 3.3 70B</td><td>81.5</td><td>80.0</td><td>78.6</td><td>78.6</td><td>77.4</td><td>78.4</td></tr><tr><td>LLaMA 3 8B Lite</td><td>71.6</td><td>71.7</td><td>72.6</td><td>71.8</td><td>71.7</td><td>71.1</td></tr><tr><td>LLaMA 3.1 8B</td><td>59.8</td><td>68.5</td><td>70.8</td><td>71.3</td><td>67.8</td><td>72.6</td></tr><tr><td>GPT-OSS 120B</td><td>89.6</td><td>90.1</td><td>89.9</td><td>89.4</td><td>90.3</td><td>90.0</td></tr><tr><td>GPT-OSS 20B</td><td>45.9</td><td>38.1</td><td>35.9</td><td>36.3</td><td>37.8</td><td>38.9</td></tr></table>

Table 4: MCQ accuracy (%) by persona, pooled over Imo and Ebonyi. BL = zero-shot baseline; YTB = Young Teenage Boy; MW = Married Woman; MM = Married Man; WW = Widowed Woman; Prof = Professor.

## A.2 Significance tests

<table><tr><td></td><td colspan="2">EN → IG</td><td colspan="2">Position</td><td>Persona</td></tr><tr><td>Model</td><td> $\chi ^ { 2 }$ </td><td>p</td><td> $\chi ^ { 2 }$ </td><td>p</td><td>sig.</td></tr><tr><td>AfriqueQwen 14B</td><td>31.03</td><td>&lt; 0.001</td><td>0.12</td><td>0.731</td><td>10/10</td></tr><tr><td>Gemini 2.0 Flash</td><td>5.14</td><td>0.023</td><td>5.26</td><td>0.022</td><td>0/10</td></tr><tr><td>GPT-4o mini</td><td>24.74</td><td>&lt; 0.001</td><td>16.60</td><td>&lt; 0.001</td><td>7/10</td></tr><tr><td>GPT-OSS 120B</td><td>25.29</td><td>&lt; 0.001</td><td>0.13</td><td>0.716</td><td>0/10</td></tr><tr><td>GPT-OSS 20B</td><td>164.28</td><td>&lt; 0.001</td><td>0.67</td><td>0.413</td><td>10/10</td></tr><tr><td>LLaMA 3.1 8B</td><td>113.39</td><td>&lt; 0.001</td><td>11.52</td><td>&lt; 0.001</td><td>10/10</td></tr><tr><td>LLaMA 3.3 70B</td><td>12.19</td><td>&lt; 0.001</td><td>4.34</td><td>0.037</td><td>9/10</td></tr><tr><td>LLaMA 3 8B Lite</td><td>113.39</td><td>&lt; 0.001</td><td>3.36</td><td>0.067</td><td>0/10</td></tr></table>

Table 5: McNemar’s tests $( \chi ^ { 2 }$ with Yates’ continuity correction) on paired MCQ outcomes for the zero-shot baseline items. EN → IG: Type 1 vs Type 2, original ordering (296 pairs). Position: original vs shuffled ordering (888 question and prompt type pairs). Persona: number of the ten persona conditions that differ significantly $( p < 0 . 0 5 )$ from the baseline (888 pairs each, original ordering).

## A.3 Human evaluation

<table><tr><td>Model</td><td>Type 1</td><td>Type 2</td><td>Type 3</td></tr><tr><td>Gemini 2.0 Flash</td><td>3.559</td><td>4.441</td><td>4.949</td></tr><tr><td>GPT-4o mini</td><td>3.271</td><td>3.169</td><td>3.000</td></tr><tr><td>GPT-OSS 120B</td><td>3.102</td><td>3.153</td><td>3.017</td></tr><tr><td>LLaMA 3.3 70B</td><td>3.203</td><td>3.119</td><td>2.797</td></tr><tr><td>AfriqueQwen 14B</td><td>3.237</td><td>2.983</td><td>2.356</td></tr><tr><td>GPT-OSS 20B</td><td>2.729</td><td>1.915</td><td>1.741</td></tr><tr><td>LLaMA 3.1 8B</td><td>2.983</td><td>0.678</td><td>0.153</td></tr><tr><td>LLaMA 3 8B Lite</td><td>3.034</td><td>0.000</td><td>0.102</td></tr><tr><td>All models (mean)</td><td>3.140</td><td>2.432</td><td>2.265</td></tr></table>

Table 6: Mean total human evaluation score (out of 8) by model and prompt type.

<table><tr><td>Model</td><td>EF</td><td>RR</td><td>LQ</td><td>PP</td></tr><tr><td>Gemini 2.0 Flash</td><td>0.531</td><td>0.531</td><td>1.994</td><td>1.260</td></tr><tr><td>GPT-4o mini</td><td>0.062</td><td>0.062</td><td>2.000</td><td>1.023</td></tr><tr><td>GPT-OSS 120B</td><td>0.074</td><td>0.074</td><td>1.938</td><td>1.006</td></tr><tr><td>LLaMA 3.3 70B</td><td>0.062</td><td>0.062</td><td>1.932</td><td>0.983</td></tr><tr><td>AfriqueQwen 14B</td><td>0.102</td><td>0.102</td><td>1.740</td><td>0.915</td></tr><tr><td>GPT-OSS 20B</td><td>0.040</td><td>0.040</td><td>1.351</td><td>0.698</td></tr><tr><td>LLaMA 3.1 8B</td><td>0.034</td><td>0.034</td><td>0.803</td><td>0.401</td></tr><tr><td>LLaMA 3 8B Lite</td><td>0.006</td><td>0.006</td><td>0.689</td><td>0.345</td></tr></table>

Table 7: Mean human evaluation score per construct (each 0 to 2), averaged over the three prompt types.

<table><tr><td>Model</td><td>∆EF</td><td>∆RR</td><td>∆LQ</td><td>∆PP</td></tr><tr><td>AfriqueQwen 14B</td><td>-0.03</td><td>-0.03</td><td>0.24*</td><td>0.08</td></tr><tr><td>Gemini 2.0 Flash</td><td>-0.34*</td><td>-0.34*</td><td>-0.02</td><td>-0.19*</td></tr><tr><td>GPT-4o mini</td><td>0.05</td><td>0.05</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-OSS 120B</td><td>-0.08</td><td>-0.08</td><td>0.10</td><td>0.02</td></tr><tr><td>GPT-OSS 20B</td><td>0.05</td><td>0.05</td><td>0.46*</td><td>0.25*</td></tr><tr><td>LLaMA 3.1 8B</td><td>0.10*</td><td>0.10*</td><td>1.36*</td><td>0.75*</td></tr><tr><td>LLaMA 3.3 70B</td><td>-0.02</td><td>-0.02</td><td>0.07*</td><td>0.05</td></tr><tr><td>LLaMA 3 8B Lite</td><td>0.02</td><td>0.02</td><td>2.00*</td><td>1.00*</td></tr></table>

Table 8: Mean within-item score difference (Type 1 minus Type 2) in human evaluation. Positive values are drops under Igbo prompting; negative values are gains. Asterisks mark differences whose bootstrap 95% confidence interval (10,000 resamples) excludes zero. Each model has 59 paired items.

## A.4 LLM-as-a-judge

<table><tr><td>Model</td><td>n</td><td>EF</td><td>RR</td><td>LQ</td><td>PP</td></tr><tr><td>GPT-OSS 120B</td><td>177</td><td>1.113</td><td>1.023</td><td>1.989</td><td>1.955</td></tr><tr><td>GPT-4o mini</td><td>177</td><td>0.672</td><td>0.582</td><td>2.000</td><td>2.000</td></tr><tr><td>LLaMA 3.3 70B</td><td>177</td><td>0.599</td><td>0.531</td><td>1.938</td><td>1.927</td></tr><tr><td>AfriqueQwen 14B</td><td>177</td><td>0.644</td><td>0.542</td><td>1.751</td><td>1.734</td></tr><tr><td>GPT-OSS 20B</td><td>176</td><td>0.619</td><td>0.597</td><td>1.460</td><td>1.449</td></tr><tr><td>LLaMA 3.1 8B</td><td>177</td><td>0.226</td><td>0.175</td><td>1.011</td><td>1.023</td></tr><tr><td>LLaMA 3 8B Lite</td><td>177</td><td>0.141</td><td>0.119</td><td>0.910</td><td>0.831</td></tr></table>

Table 9: Mean LLM-as-a-judge scores by model, ordered by total, over the responses matched to human scores (59 questions across three prompt types); n is the number of matched responses. Empty responses are scored 0, as in the human rubric.

<table><tr><td></td><td> $\kappa _ { w }$ </td><td> $\rho$ </td><td>Exact</td><td>Adj.</td><td>Human</td><td>Judge</td></tr><tr><td>EF</td><td>0.112</td><td>0.276</td><td>59.8</td><td>87.5</td><td>0.054</td><td>0.574</td></tr><tr><td>RR</td><td>0.100</td><td>0.240</td><td>59.9</td><td>92.2</td><td>0.054</td><td>0.510</td></tr><tr><td>LQ</td><td>0.833</td><td>0.885</td><td>90.9</td><td>97.3</td><td>1.494</td><td>1.580</td></tr><tr><td>PP</td><td>0.261</td><td>0.883</td><td>22.1</td><td>98.1</td><td>0.767</td><td>1.560</td></tr></table>

Table 10: Agreement between the human annotator and the Gemini 2.0 Flash judge over 1,238 matched responses. $\kappa _ { w } = \mathrm { w e i g h t e d }$ Cohen’s kappa; ρ = Spearman rank correlation; Exact and Adj. = percentage of pairs that agree exactly or differ by at most one point; Human and Judge = mean scores.

## A.5 Repetition failures

Figure 7 shows verbatim excerpts of the repetition loops described in Section 5.4.

## B Dataset Statement for CAB-Igbo

## B.1 Creation

CAB-Igbo is built entirely from scratch, without relying on web-scraped text. All data is manually created and validated by native speakers from two

![](images/68e59368a02afcf807269d1558e2a0b4b121f5707a6189ad2a058727bb68ead5.jpg)  
Figure 7: Repetition loops in open-ended answers under Igbo prompting. Each excerpt is quoted verbatim from a model answer in the human-evaluation sample, with omitted text marked by an ellipsis; the repeated unit is highlighted, and the counts refer to the full stored answer. The annotator labelled all six answers as Model Breakdown and Repetition.

Igbo-speaking states. The dataset contains 296 questions in Igbo and English. Questions were designed and reviewed by a native Igbo speaker and an Igbo linguist.

## B.2 Cultural domains

CAB-Igbo evaluates six domains of Igbo cultural life. Table 11 gives the number of questions in each.

<table><tr><td>Domain</td><td>Questions</td></tr><tr><td>Food and Cuisine</td><td>73</td></tr><tr><td>Festivals and Religion</td><td>67</td></tr><tr><td>Family and Kinship</td><td>58</td></tr><tr><td>Proverbs and Oral Tradition</td><td>38</td></tr><tr><td>Social Customs and Etiquette</td><td>35</td></tr><tr><td>Funerals and Burial</td><td>25</td></tr><tr><td>Total</td><td>296</td></tr></table>

Table 11: Number of questions per cultural domain.

## B.3 Question design

Every question must:

• test authentic Igbo cultural knowledge;

• contain one clear and defensible correct answer;

• reflect cultural practices among Igbo communities.

Questions test cultural reasoning rather than factual recall alone. Preference was given to questions that require an understanding of cultural logic and meaning, knowledge of ritual and social context, and recognition of appropriate behaviour within Igbo society. For example, some questions ask why a child may be buried without mourning rites; the correct answer requires an understanding of the ogbanje belief system rather than simple memorisation.

Questions use direct and unambiguous language: they avoid vague wording, specify traditional or modern practice where necessary, and use precise cultural terminology.

## B.4 Linguistic grid and question formats

Each question is evaluated in three language conditions: Type 1 (English instruction, English question), Type 2 (English instruction, Igbo question), and Type 3 (Igbo instruction, Igbo question).

Each multiple-choice question contains one correct answer and three diagnostic distractors. Distractors represent different forms of cultural misunderstanding, so the benchmark identifies how models fail, not only whether they fail. Each multiple-choice item has a matching open-ended version, which requires the model to explain the cultural reasoning in two to four sentences without answer choices. Open-ended responses are evaluated through human annotation using the rubric in Appendix C.

## B.5 Distractor taxonomy

Each distractor belongs to exactly one of four categories, assigned during dataset construction (Table 12).

<table><tr><td>Category</td><td>Definition</td><td>Count</td></tr><tr><td>Western Default</td><td>Interprets Igbo culture through Western assumptions such as individualism or nuclear family structures</td><td>292</td></tr><tr><td>Shallow Description</td><td>Recognises surface patterns without deeper cultural</td><td>256</td></tr><tr><td>Ethnolinguistic Substitution</td><td>understanding Replaces Igbo practices with Yoruba or Hausa cultural</td><td>174</td></tr><tr><td>Pan-African Generalisation</td><td>forms Uses broad African stereotypes that erase</td><td>166</td></tr><tr><td colspan="2">Total</td><td>888</td></tr></table>

Table 12: Distractor categories and their frequency across the 888 distractors (three per question).

## B.6 Bilingual and orthographic requirements

Every question appears in parallel Igbo and English versions, and both versions must test the same knowledge with the same meaning. The Igbo version must read naturally and use appropriate Igbo terminology for cultural concepts. All Igbo text follows standard orthography, includes tone marks where necessary, and keeps spelling consistent throughout the dataset.

## B.7 Item structure

Each item records the question number, the cultural domain, the question text in Igbo and English, four answer options (A to D) in both languages, the correct answer label, and the distractor category of each incorrect option.

## B.8 Quality criteria

Before inclusion, each question must satisfy the following conditions:

1. The correct answer reflects authentic Igbo practice.

2. The question and options are unambiguous.

3. Distractors are plausible but incorrect.

4. Each distractor reveals a specific misunderstanding.

5. The Igbo and English versions are semantically equivalent.

6. The question tests unique knowledge within the dataset.

## B.9 Review process

Items passed through three stages.

Question creation. Questions were drafted independently, with balanced domain coverage, and each distractor was assigned a category label.

Joint review. Each question was reviewed collaboratively to verify cultural accuracy, check translation equivalence, resolve ambiguity, and confirm the distractor categories.

Quality control. Duplicate and ambiguous questions were removed, the domain distribution was checked, and formatting was finalised.

## B.10 Native speaker validation

Native Igbo speakers reviewed all items using a participatory evaluation process, checking cultural accuracy, linguistic quality, and contextual fit. Items without agreement were revised or removed. This process grounds the benchmark in lived cultural expertise rather than external interpretation or synthetic generation.

## B.11 Common pitfalls

Dataset creators were asked to avoid:

• regional variation that produces multiple valid answers;

• questions with more than one defensible answer;

• superficial questions that test recall instead of understanding;

• translation errors or unnatural Igbo phrasing;

• weak distractors that are obviously incorrect;

• unnecessary gender stereotypes unless culturally required.

## B.12 Documentation

The development process records the rationale for each question design, removed questions and the reasons for removal, revisions made during review, distractor category decisions, and domain distribution statistics.

## C Annotator Guidelines

## C.1 Background

The CAB-Igbo benchmark evaluates how AI models represent and apply Igbo cultural knowledge. In this study, trained annotators assess modelgenerated responses to cultural questions using a standardised rubric. The evaluation has two objectives: first, to validate the automated LLM-as-ajudge system, and second, to compare model performance across linguistic prompt types. Model identities remain hidden throughout the evaluation; annotators see only anonymised labels such as Model A and Model B.

## C.2 Evaluation setup

Annotators evaluate 20% of the CAB-Igbo dataset, and each question features responses from eight models. All responses are generated under the zeroshot baseline condition (no persona), in all three prompt types (Type 1: English instruction, English question; Type 2: English instruction, Igbo question; Type 3: Igbo instruction, Igbo question). The questions cover the six domains listed in Table 11.

## C.3 Scoring constructs

Each response is scored on four constructs using a 0 to 2 integer scale (Table 13). For Igbo responses (Type 2 and Type 3), annotators assess Igbo grammar and morphology directly under Linguistic Quality. The total response score is the sum of the four constructs, EF + RR + LQ + PP, and ranges from 0 to 8.

<table><tr><td colspan="2">Epistemic Fidelity (EF): factual accuracy and cultural reliability</td></tr><tr><td>2 1 0</td><td>Completely accurate and culturally specific Partially accurate or surface-level Completely incorrect, fabricated, or blank</td></tr><tr><td colspan="2">Representational Richness (RR): depth and nuance Rich, nuanced, and culturally grounded</td></tr><tr><td colspan="2">2 1 Basic or vague cultural detail 0 No meaningful cultural content, or blank</td></tr><tr><td colspan="2">Linguistic Quality (LQ): fluency, grammar, and lan- guage use</td></tr><tr><td colspan="2">2 Fluent and grammatically correct 1 Understandable despite noticeable errors</td></tr><tr><td colspan="2">0 Largely unintelligible, wrong language, or blank</td></tr><tr><td colspan="2">Pragmatic Proficiency (PP): contextual appropriateness</td></tr><tr><td colspan="2">2 Fully appropriate and context-aware Partially appropriate or with minor inconsisten-</td></tr><tr><td colspan="2">1</td></tr><tr><td colspan="2">cies</td></tr><tr><td colspan="2">0 Completely misaligned, off-topic, or blank</td></tr></table>

Table 13: Human evaluation rubric.

## C.4 Annotation procedure

Annotators use a custom web-based evaluation interface and complete the following steps.

1. Enter the authorised Annotator ID to log in and load the assigned evaluation set.

## 2. For each grouped question block:

(a) Read the cultural question carefully, reviewing both the English and Igbo versions to ensure a full understanding of the cultural context.

(b) Note the prompt type (Type 1, Type 2, or Type 3) displayed for the block.

(c) Read the corresponding model-generated response.

(d) Use the interface dropdowns to assign an integer score from 0 to 2 for EF, RR, LQ, and PP.

(e) Add any necessary clarification in the optional Comments box.

3. Click Submit at the end of each block to register the scores.

4. Click Download CSV periodically to back up progress. Once all questions are evaluated, perform a final download and submit the file to the lead researchers.

## C.5 Evaluation rules

• Evaluate each response independently.

Echefula i<sub>.</sub>za n'u<sub>.</sub>zo<sub>.</sub> izugbe nke Nai<sub>.</sub>jiri<sub>.</sub>a ma o<sub>.</sub> bu<sub>.</sub>   
Afrika. Zaá kpomkwem site n'ime ndu omenala   
Igbo dika a maara ya na {state} Steeti.

• Responses marked [No response found] or left completely blank receive a score of 0 for all constructs.

• Assess Igbo fluency under Linguistic Quality for Type 2 and Type 3 tasks.

• Do not discuss scores with other annotators before completing the evaluation.

• If uncertain between two scores, assign the lower score and add a brief comment.

## D Prompt Templates

This appendix reproduces the exact text of the twelve prompt templates. Placeholders in braces are filled from the dataset at run time. Each persona prompt consists of a persona introduction (Appendix D.2) followed by the zero-shot prompt of the same type and format (Appendix D.1); for Type 3 the introduction and the persona attributes are given in Igbo. A native-speaker linguist later identified minor phrasing issues in the Igbo instruction text; we retain the wording exactly as used during evaluation, for reproducibility.

## D.1 Zero-shot prompts

Multiple choice, Type 1 (English instruction, English   
question)   
Answer in English only. Select exactly one   
option. Answer by typing only the letter (A, B,   
C, or D).   
Question: {question}   
Options:   
A. {option\_a}   
B. {option\_b}   
C. {option\_c}   
D. {option\_d}

Multiple choice, Type 2 (English instruction, Igbo   
question)   
The question is in Igbo. Answer in Igbo only.   
Select exactly one option. Answer by typing   
only the letter (A, B, C, or D).   
Question: {question}   
Options:   
A. {option\_a}   
B. {option\_b}   
C. {option\_c}   
D. {option\_d}

Multiple choice, Type 3 (Igbo instruction, Igbo question)   
Zaá n'asusu Igbo naani. Horo naani otu nhoro.   
Saá ya site nide naani akara (A, B, C, ma o bu   
D).   
Aju<sub>.</sub>ju<sub>.</sub>: {question}   
Nho<sub>.</sub>ro<sub>.</sub>:   
A. {option\_a}   
B. {option\_b}   
C. {option\_c}   
D. {option\_d}

Open-ended, Type 1 (English instruction, English   
question)   
Answer in English only. Provide a concise   
explanation in 2-4 sentences. Do not include   
any extra commentary.   
Question: {question}

Open-ended, Type 2 (English instruction, Igbo question)   
The question is in Igbo. Answer in Igbo only.   
Provide a concise explanation in 2-4 sentences.   
Do not include any extra commentary.   
Question: {question}

```html
Open-ended, Type 3 (Igbo instruction, Igbo question)
Zaá n'asu<sub>.</sub>su<sub>.</sub> Igbo naani<sub>.</sub>. Nye nko<sub>.</sub>wa di<sub>.</sub> mkpu<sub>.</sub>mkpu<sub>.</sub>
n'ime ahi<sub>.</sub>ri<sub>.</sub>okwu 2-4. Echefula itinye okwu o
bu<sub>.</sub>la o<sub>.</sub>zo<sub>.</sub> kari<sub>.</sub>a azi<sub>.</sub>za gi<sub>.</sub>.
Aju<sub>.</sub>ju<sub>.</sub>: {question}
```

## D.2 Persona introductions

English introduction (Types 1 and 2)   
Imagine you are a {marital\_status} {sex} from   
{state} State, in South-Eastern Nigeria. You   
have completed {education} education and   
consider yourself part of {social\_class}. You   
have lived within Igbo cultural and community   
life throughout your life.   
Others will read your response; your goal is to   
convince them that it reflects the genuine   
cultural knowledge and lived experience of the   
person described above.   
Do not default to generalised Nigerian or   
African responses. Answer specifically from   
within Igbo cultural life as understood in   
{state} State.

## D.3 Persona attributes

The persona prompts combine each of the five personas below with each of the two states (Imo and Ebonyi), giving ten persona conditions. Each cell lists sex, marital status, education, and social class as inserted into the template.

<table><tr><td>Persona</td><td>English (Types 1, 2)</td><td>Igbo (Type 3)</td></tr><tr><td>Young Teenage Boy</td><td>young boy; single; secondary school student; the youth</td><td>nwata nwoke; na-albegh nwanyi; nwa akwkw sekondri; ndi ntoroba</td></tr><tr><td>Married Woman</td><td>woman; married; primary school; the local trading class</td><td>nwaany; lr di; l akwukwo pramari; ndi ahia obodo</td></tr><tr><td>Married Man</td><td>man; married; basic community; the rural farming class</td><td>nwoke; lr nwaanyi; obodo; nd r ugbo</td></tr><tr><td>Widowed Woman</td><td>woman; widowed; vocational; the</td><td>nwaanyi; di ya nwr; aka r; nd</td></tr><tr><td>Professor</td><td>artisan class man; married;</td><td>aka r nwoke; lur nwaanyi;</td></tr><tr><td></td><td>post-graduate (PhD); the academic middle class</td><td>mahadum (PhD); nd gr akwkw nke oma</td></tr></table>

Table 14: Persona attributes inserted into the persona introductions.