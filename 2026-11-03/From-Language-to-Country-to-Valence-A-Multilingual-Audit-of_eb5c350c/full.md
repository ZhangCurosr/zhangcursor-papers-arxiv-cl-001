# From Language to Country to Valence: A Multilingual Audit of Stereotype Bias Across LLMs

Zihao Sun   
University of Technology Sydney Sydney, Australia   
sun.zihao@student.uts.edu.au

## Abstract

Large language models (LLMs) are increasingly considered for education, recruitment, workplace decision support, and other consequential applications, making pre-deployment social-bias detection essential. Yet major benchmarks are English-centred or translated, focus mainly on negative stereotypes, and may miss linguistic and cultural variation. We investigate gender, racial, and religious bias using 12,000 locally written Reddit posts in English, Spanish, French, and German. Only search terms are translated, preserving local context. The dataset is balanced by language and category; English gender is additionally balanced by stereotype valence. Three openweight and three proprietary API-based models are evaluated with a two-blank design that reduces grammatical-gender cues. Religion has the highest absolute choice imbalance in every model–language condition. Spanish is high est for gender, whereas German is highest for race and religion. The evaluated API models score below the open-weight models throughout. Negative English-gender items exceed positive items; Australian and Mexican contexts exceed their Canadian and Spanish counterparts. These descriptive differences measure choice concentration, not stereotype endorsement or causal country effects. The crossed patterns show how aggregate scores can conceal category- and context-specific behaviour, motivating closer inspection before deployment.

## 1 Introduction

LLMs increasingly mediate information and decisions in education, recruitment, and workplaces. Their integration into collaborative workflows makes pre-deployment auditing relevant not only to model developers but also to teams selecting and localising decision-support systems (Gallegos et al., 2024; Weidinger et al., 2022; Mitchell et al., 2019; Raji et al., 2020). A system used across languages and communities may exhibit different patterns for different social groups; a global score can obscure the contexts requiring closer scrutiny. Nevertheless, what a bias score establishes depends on its operationalisation (Blodgett et al., 2020; Lum et al., 2025).

Existing evaluations probe associations (Caliskan et al., 2017; May et al., 2019), contrastive sentences and questions (Nangia et al., 2020; Nadeem et al., 2021; Parrish et al., 2022), or generated text (Sheng et al., 2019; Dhamala et al., 2021; Nozza et al., 2021; Smith et al., 2022; Sap et al., 2020). English-dominant resources and translated examples, however, may not preserve local social situations, linguistic forms, or stereotype content (Joshi et al., 2020; Artetxe et al., 2020; Talat et al., 2022; Hershcovich et al., 2022). Cross-cultural and multilingual evaluations demonstrate the importance of language, locale, and variety (Gamboa et al., 2025; Nie et al., 2024; Naous et al., 2024; Fleisig et al., 2024).

We translate retrieval terms rather than complete posts, building on Reddit-based bias evaluation (Barikeri et al., 2021). Locally written contexts preserve native wording and community references, but are deliberately non-parallel: the resulting comparisons describe context-specific model behaviour, not isolated causal language effects. Agreement-aware linked blanks reduce grammatical clues (Zhao et al., 2018; Stanovsky et al., 2019; Zhao et al., 2020). We also separate positive and negative stereotypes: favourable generalisations can still essentialise a group (Fiske et al., 2002; Kay et al., 2013; Siy and Cheryan, 2016).

Our contributions are:

• We construct a 12,000-item multilingual dataset from locally written Reddit posts, balanced across English, Spanish, French, and German and the gender, race, and religion categories. Translating only search terms, followed by multilingual filtering and languageaware human annotation, preserves locally written context.

• We evaluate three open-weight and three proprietary API-based models with an agreementaware two-blank format that prevents grammatical gender from revealing the expected answer.

• We add two focused context analyses: positive versus negative gender stereotypes, and within-language country comparisons for English and Spanish.

## 2 Related Work

CrowS-Pairs and StereoSet evaluate preferences among stereotypical and counter-stereotypical alternatives (Nangia et al., 2020; Nadeem et al., 2021); BBQ examines bias in question answering (Parrish et al., 2022). RedditBias brings naturally occurring conversation into evaluation (Barikeri et al., 2021), while counterfactual probes compare outcomes after changing sensitive attributes (Huang et al., 2020). Such instruments measure distinct behaviours, not interchangeable indicators of downstream harm. Constrained tests need not predict behaviour in longer responses (Lum et al., 2025).

Multilingual audits additionally face translation and cultural-validity problems (Talat et al., 2022; Nie et al., 2024). Region-specific Spanish resources and evaluations distinguish varieties rather than treating a shared language as a single cultural setting (España-Bonet and Barrón-Cedeño, 2024; Grandury et al., 2025). Cross-cultural value probes and cultural-adaptability frameworks likewise motivate context-sensitive evaluation (Arora et al., 2023; Rao et al., 2025). Reddit corpora offer community-grounded material for such work (Baumgartner et al., 2020; Barikeri et al., 2021). Our design complements controlled translation studies by preserving local contexts and reporting crossed category–language profiles. Its nonparallel corpora limit attribution of a difference to language alone.

## 3 Dataset and Evaluation Design

## 3.1 Evaluation Questions

RQ1: Does aggregate constrained stereotypechoice imbalance remain stable across languages and models? RQ2: How do gender, race, and religion differ across those languages and models? RQ3: How does imbalance differ by Englishgender stereotype valence and between countrylinked contexts within a shared language?

## 3.2 Local-Language Collection and Allocation

Following RedditBias’s distinction between targets and attributes (Barikeri et al., 2021), queries combine identity terms with traits, roles, or behaviours. The project-defined comparisons are women with men, White with Black, and Christians with Muslims; these are tested label pairs, not exhaustive identity taxonomies. Gender attributes include caring, capable, emotional, domineering; race attributes include hardworking, resilient, aggressive, untrustworthy; religion attributes include charitable, compassionate, dogmatic, intolerant. Attributes retrieve candidates, rather than predetermine which group is associated with a trait. Contextual annotation determines stereotype presence and valence. Equal category sizes do not match lexical severity or topic, and alternative-attribute robustness is not tested.

GPT-4o, DeepL, and Google Translate independently produce Spanish, French, and German search-term candidates. Source and candidate terms are encoded with sentence-transformers /paraphrase-multilingual-mpnet-base-v2, a multilingual knowledge-distillation checkpoint (Reimers and Gurevych, 2020). Candidates with cross-lingual cosine similarity above .8 are retained; disagreements and morphologically ambiguous terms receive human review. Only the retained terms are translated: retrieved posts remain in their original language and local context.

Each language contributes 1,000 items per category, hence 3,000 per language and 12,000 overall. English and Spanish are each divided equally between two country-linked source strata: Canada/Australia and Mexico/Spain, respectively. Each country contributes 500 items per category. The English-gender cell additionally contains 500 positive and 500 negative items, split 250/250 within each country; other cells have no valence quota. French and German contribute 1,000 items per category without a country split. Country provenance indicates a community context, not an author’s nationality. The English frame includes r/australia, r/sydney, r/melbourne, r/canada, r/toronto, and r/vancouver; these communities are not assumed nationally representative.

<table><tr><td>Language</td><td>R</td><td>D</td><td>L</td><td>C</td><td>U</td><td>F</td></tr><tr><td>English</td><td>14,808</td><td>12,868</td><td>11,353</td><td>6,733</td><td>3,615</td><td>3,000</td></tr><tr><td>Spanish</td><td>14,193</td><td>12,346</td><td>11,028</td><td>7,527</td><td>3,574</td><td>3,000</td></tr><tr><td>French</td><td>12,642</td><td>11,931</td><td>10,554</td><td>5,348</td><td>4,521</td><td>3,000</td></tr><tr><td>German</td><td>13,976</td><td>11,187</td><td>10,763</td><td>6,412</td><td>5,548</td><td>3,000</td></tr><tr><td>Total</td><td>55,619</td><td>48,332</td><td>43,698</td><td>26,020</td><td>17,258</td><td>12,000</td></tr></table>

Table 1: Multilingual collection flow. R/D/L/C/U/F denote retrieval, deduplication, language pass, context pass, unanimous annotation, and final allocation.

## 3.3 Filtering and Human Annotation

We remove deleted content, very short posts, duplicates, quotations, counter-speech, and ambiguous cases; verify language and community provenance; and require a meaningful target–attribute relation. Table 1 reports each filtering stage. Annotators are recruited through Amazon Mechanical Turk (MTurk) and must meet the task’s qualification criteria. Three native-language annotators judge each remaining candidate. Country-conditioned items use annotators resident in, and familiar with, the corresponding country; country eligibility is checked during qualification. Before annotation, workers receive the participant information and consent statement reproduced in Appendix A.

One multiple-choice selection combines stereotype presence, category, and valence: no stereotype, or positive/negative gender, race, or religion. Retention requires unanimous agreement on the same non-zero label. Positive labels are available in every language, but only English gender is quotabalanced for the valence study. Retained counterfactuals are checked for grammaticality, semantic equivalence apart from identity, and predefined stereotype direction.

## 3.4 Agreement-Aware Prompting and Models

For race and religion, counterfactuals change the identity while retaining the surrounding sentence. Gender agreement requires linked changes: “Ella es \_\_\_” can reveal ingeniera through grammar rather than association. We instead blank both subject and occupation, for example “\_\_\_ trabaja como en una empresa de tecnología”, with alternatives Él/ingeniero and Ella/ingeniera (he/engineer and she/engineer). Both alternatives are grammatical. The pair is swapped as a unit and A/B option order is counterbalanced.

The six systems are LLaMA 3 8B, Mistral 7B, and Gemma 7B (open-weight), and GPT-4o,

Claude 3 Opus, and Gemini 1.5 Pro (proprietary APIs). Decoding uses temperature .7, top-p .9, short outputs, randomised option order, and ten valid categorical responses per prompt. The protocol records raw responses, parsed choices, refusals, invalid answers, and retry indices separately. The score conditions on valid answers; refusal rates are not included in it. Model access is not a controlled factor: scale, training data, alignment, and serving procedures also differ.

## 4 Measurement and Descriptive Scope

For item i, let $X _ { i }$ count selections of the contextannotated stereotypical alternative among k valid responses. We report

$$
B _ { i , k } = \frac { 2 | X _ { i } - k / 2 | } { k } , \qquad \overline { { B } } _ { k } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } B _ { i , k } .\tag{1}
$$

$\mathbf { A } \mathbf { t } k = 1 0$ , item scores are 0, .2, .4, .6, .8, 1. Zero is an observed five–five split; one is unanimity. Ten stereotypical and ten counter-stereotypical choices both score one. Thus B measures choice concentration, not endorsement direction or social harm.

Even independent balanced responses yield nonzero absolute scores. For $X \sim$ Binomial $( k , . 5 )$ ,

$$
\mathbb { E } _ { 0 } [ B _ { k } ] = { \frac { 2 } { k 2 ^ { k } } } \sum _ { x = 0 } ^ { k } { \Big | } x - { \frac { k } { 2 } } { \Big | } { \binom { k } { x } } .\tag{2}
$$

For ten repetitions, this equals 63/256 = .24609375, and $P _ { 0 } ( B _ { 1 0 } > 0 ) = . 7 5 3 9 0 6 2 5$ . The reference depends on repetition count and is diagnostic, not a calibrated test of dependent model responses. Being above or below it establishes neither preference direction nor statistical significance.

All rankings and contrasts below are descriptive. Sample SDs indicate dispersion across the specified models or model-paired differences, not item-level sampling uncertainty. The six systems are a fixed evaluation set; common rankings across them do not establish generalisation beyond the sampled items or model set. Summaries are calculated from tabulated condition values before display rounding; last-digit differences should not be overinterpreted.

<table><tr><td>Model</td><td>English</td><td>Spanish</td><td>French</td><td>German</td></tr><tr><td>LLaMA 3</td><td>.308/.31/.33</td><td>.39/.39/.41</td><td>.32/.36/.40</td><td>.36/.41/.43</td></tr><tr><td>Mistral 7B</td><td>.285/.28/.29</td><td>.36/.36/.38</td><td>.29/.33/.36</td><td>.33/.37/.40</td></tr><tr><td>Gemma 7B</td><td>.345/.34/.35</td><td>.42/.42/.44</td><td>.35/.39/.42</td><td>.39/.44/.46</td></tr><tr><td>GPT-40</td><td>.1795/.18/.19</td><td>.23/.22/.25</td><td>.18/.21/.24</td><td>.21/.24/.27</td></tr><tr><td>Claude 3</td><td>.172/.17/.18</td><td>.22/.21/.24</td><td>.17/.20/.23</td><td>.20/.23/.26</td></tr><tr><td>Gemini 1.5</td><td>.1915/.19/.20</td><td>.24/.23/.26</td><td>.19/.22/.25</td><td>.22/.25/.28</td></tr><tr><td>Mean</td><td>.247/.245/.257</td><td>.310/.305/.330</td><td>.250/.285/.317</td><td>.285/.323/.350</td></tr><tr><td>SD</td><td>.075/.074/.076</td><td>.090/.095/.090</td><td>.079/.085/.086</td><td>.085/.094/.090</td></tr></table>

Table 2: Core absolute-imbalance results. Each cell lists gender/race/religion. The mean and sample SD rows summarise six models per language–category cell; SD measures between-model dispersion, not item-level uncertainty. English gender is pooled from the equal valence splits in Table 3.

## 5 Results and Interpretation

## 5.1 Category, Language, and Model Profiles

The core crosses six models, four languages, and three categories. Table 2 retains all 72 condition scores. Category-averaged language means are .250 for English, .284 for French, .315 for Spanish, and .319 for German (RQ1). English is lowest for every model; German is highest or tied highest at the tabulated precision. Across the 24 model–language cells, religion averages .313 $( S D = . 0 8 7 )$ , race .290 $( S D = . 0 8 7 )$ , and gender .273 (SD = .081). Religion is highest in every cell; every model retains this category order after averaging languages (RQ2). These are subcorpusdependent concentration rankings, not a hierarchy of social harm.

## 5.1.1 Within-language category contrasts

English follows religion > gender > race (.257/.247/.245), whereas Spanish has religion > gender > race (.330/.310/.305). Its gender–race separation is only .005. French and German return to religion > race > gender, with religion– gender gaps of .067 and .065, compared with .010 for English. English has the narrowest category range (.012), compared with .025 for Spanish. A language-wide mean therefore mixes distinct category profiles rather than describing equally elevated scores for every group.

The French profile also persists within access families. Open-weight gender/race/religion means are .320/.360/.393; API means are .180/.210/.240.

French gender averages .320 (SD = .030) among the three open-weight models and .180 (SD = .010) among the APIs. The same category ordering survives a substantial difference in model baselines. Between-family separation partly explains the larger six-model SDs and should not be mistaken for uncertainty over items.

## 5.1.2 Within-category language contrasts

For gender, the six-model means order English < French < German < Spanish. English and French are nearly tied (.247 versus .250): Claude and Gemini have slightly higher English than French gender scores at the tabulated precision. For race and religion, every model orders English < French < Spanish < German. Spanish exceeds German by .025 for gender, while German exceeds Spanish by .018 for race and .020 for religion. Their .004 aggregate gap thus hides a crossover: neither language is uniformly higher. Reading categories separately changes which local contexts appear most concentrated, despite the common aggregate language ordering.

Every evaluated open-weight score exceeds every API score in its matched cell, but family gaps range from .130 for English race to .170 for Spanish race. The separation is therefore neither constant across languages nor shared equally by categories. These are observed access-family patterns, not effects of being “free” or “paid”.

## 5.1.3 Interpreting the finite-sample reference

From the tabulated scores, all 12 open-weight model–language means exceed .2461; 11 of the 12 API means are below it. The exception is Gemini 1.5 Pro in German (.250). This comparison locates the point estimates, not which cells differ statistically from balanced choice. Below-reference scores do not imply counter-stereotypical preference and warrant checking repetition dependence; above-reference scores likewise do not establish stereotype endorsement.

<table><tr><td>Model Core Positive Negative</td></tr><tr><td> $\Delta _ { \mathrm { v a l } }$  LLaMA 3 8B .368 .247  $. 3 6 9 \ + . 1 2 2$ </td></tr><tr><td>Mistral 7B .336 .233  $. 3 3 7 \ + . 1 0 4 $ </td></tr><tr><td>Gemma 7B .397 .274  $. 4 1 6 ~ + . 1 4 2$ </td></tr><tr><td>Open-weight mean .367 .251  $. 3 7 4 \ + . 1 2 3$ </td></tr><tr><td>GPT-40 .217 .147  $. 2 1 2 \ + . 0 6 5$ </td></tr><tr><td>Claude 3 Opus .207 .143  $. 2 0 1 ~ + . 0 5 8$ </td></tr><tr><td>Gemini 1.5 Pro .227 .158  $. 2 2 5 \ + . 0 6 7$ </td></tr><tr><td>API mean .217 .149  $. 2 1 3 \ + . 0 6 3$ </td></tr></table>

Table 3: Model-level absolute imbalance and Englishgender valence results. Core averages the tabulated language–category scores; $\Delta _ { \mathrm { v a l } } = B _ { \mathrm { n e g a t i v e } } -$ $B _ { \mathrm { p o s i t i v e } } .$

## 5.2 Study 1: English-Gender Stereotype Valence

For RQ3’s valence comparison, positive items average .200 $( S D = . 0 5 8 )$ and negative items .293 $( S D = . 0 9 2 )$ . Table 3 gives the six model results. Model-paired negative-minus-positive differences average .093 $( S D \ = \ . 0 3 5 )$ , range from .058 to .142, and are positive for every model. The openweight mean contrast (.123) exceeds the API contrast (.063).

Higher negative-item scores describe stronger concentration, not greater endorsement of negative stereotypes. Positive and negative examples are different posts, balanced in count within each country, rather than an intervention reversing the valence of otherwise identical sentences. The common sign across models does not establish item-level precision. The result is restricted to English gender; the potential harm of favourable generalisations is a distinct concern (Fiske et al., 2002; Siy and Cheryan, 2016).

## 5.3 Study 2: Country Context Within a Shared Language

Country comparisons retain the language, models, category allocation, decoding procedure, and score definition while changing the country-linked Reddit source. Table 4 reports the category-specific scores. Categories are first averaged within each model for locale-level summaries. Canada averages .220 (SD = .069), versus .279 (SD = .081) for Australia; their model-paired difference is .059 $( S D = . 0 1 3 )$ . Spain averages .301 $( S D = . 0 8 8 )$ versus .329 $( S D = . 0 9 5 )$ for Mexico; the paired difference is .028 (SD = .007).

The Australia–Canada gender/race/religion contrasts are .094/.036/.048, with paired SDs .018/.009/.011. Mexico–Spain contrasts are .018/.028/.038, with SDs .005/.007/.009. Every model’s contrast is positive in every category, but the largest gap is gender for Australia–Canada and religion for Mexico–Spain. Canada follows religion > race > gender, whereas Australia follows gender > religion > race (.294/.281/.263). Spain places gender above race, whereas Mexico’s gender and race means are both .319. Thus a shared direction at country level can still combine different category profiles within the same language.

Open-weight mean country gaps exceed API gaps: .070 versus .049 for Australia–Canada and .034 versus .022 for Mexico–Spain. Larger SDs for country scores than for paired differences reflect differing model baselines, not precise estimation over Reddit items. Pairing is between model summaries; posts from different countries are not matched sentence pairs. Topic, community, and wording vary alongside country context, so these findings neither identify causal country effects nor rank national populations by prejudice.

## 6 Discussion and Deployment Implications

The Spanish–German crossover is the central example of aggregation hiding context: a small languagewide difference combines opposing category orders. The French access-family comparison and the country-specific category profiles similarly show why a single model or language summary is incomplete. For teams designing multilingual decisionsupport workflows, these descriptive profiles identify context–category cells that merit closer inspection, rather than establish a universal model ranking.

A practical implication is to document the intended languages, communities, and social categories when comparing candidate models, and to examine representative responses and refusals alongside concentration scores. This links local evaluation to model reporting and internal audit processes (Mitchell et al., 2019; Raji et al., 2020). It is a deployment implication, not a tested intervention: this study does not evaluate a deployed collaborative system or show improved human decision outcomes. Lower absolute scores do not certify fairness; high scores can reflect consistently stereotypical or counter-stereotypical choices.

<table><tr><td>Locale</td><td>Category</td><td>LLaMA</td><td>Mistral</td><td>Gemma</td><td>GPT-40</td><td>Claude</td><td>Gemini</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td colspan="9">English: Canada-Australia</td></tr><tr><td>Canada</td><td>Gender</td><td>.255</td><td>.237</td><td>.283</td><td>.141</td><td>.132</td><td>.150</td><td> $. 2 0 0 \pm . 0 6 6$ </td></tr><tr><td></td><td>Race</td><td>.288</td><td>.260</td><td>.316</td><td>.166</td><td>.157</td><td>.175</td><td> $. 2 2 7 \pm . 0 6 9$ </td></tr><tr><td></td><td>Religion</td><td>.301</td><td>.263</td><td>.319</td><td>.171</td><td>.162</td><td>.180</td><td> $. 2 3 3 \pm . 0 7 0$ </td></tr><tr><td>Australia</td><td>Gender</td><td>.361</td><td>.333</td><td>.407</td><td>.218</td><td>.212</td><td>.233</td><td> $. 2 9 4 \pm . 0 8 4$ </td></tr><tr><td></td><td>Race</td><td>.332</td><td>.300</td><td>.364</td><td>.194</td><td>.183</td><td>.205</td><td> $. 2 6 3 \pm . 0 7 9$ </td></tr><tr><td></td><td>Religion</td><td>.359</td><td>.317</td><td>.381</td><td>.209</td><td>.198</td><td>.220</td><td> $. 2 8 1 \pm . 0 8 1$ </td></tr><tr><td colspan="9">Spanish: Spain-Mexico</td></tr><tr><td>Spain</td><td>Gender</td><td>.379</td><td>.350</td><td>.408</td><td>.223</td><td>.214</td><td>.232</td><td> $. 3 0 1 \pm . 0 8 8$ </td></tr><tr><td></td><td>Race</td><td>.373</td><td>.344</td><td>.402</td><td>.209</td><td>.200</td><td>.218</td><td> $. 2 9 1 \pm . 0 9 2$ </td></tr><tr><td></td><td>Religion</td><td>.387</td><td>.358</td><td>.416</td><td>.235</td><td>.226</td><td>.244</td><td> $. 3 1 1 \pm . 0 8 5$ </td></tr><tr><td>Mexico</td><td>Gender</td><td>.401</td><td>.370</td><td>.432</td><td>.237</td><td>.226</td><td>.248</td><td> $. 3 1 9 \pm . 0 9 2$ </td></tr><tr><td></td><td>Race</td><td>.407</td><td>.376</td><td>.438</td><td>.231</td><td>.220</td><td>.242</td><td> $. 3 1 9 \pm . 0 9 9$ </td></tr><tr><td></td><td>Religion</td><td>.433</td><td>.402</td><td>.464</td><td>.265</td><td>.254</td><td>.276</td><td> $. 3 4 9 \pm . 0 9 4$ </td></tr></table>

Table 4: Country-context absolute-imbalance results. Final column: mean and sample SD across six models. Each country-pair midpoint recovers the pooled category cell in Table 2. Australian gender is obtained by equal-country balance from pooled English gender and the Canadian row.

## 7 Conclusion

We presented a multilingual, context-aware audit across six systems, four languages, and gender, race, and religion. Native posts retain local context and linked blanks reduce grammatical cues. Religion has the highest pooled score in every model– language cell, but Spanish leads for gender whereas German leads for race and religion. API scores are lower than open-weight scores; negative Englishgender items and Australian/Mexican contexts have higher observed concentration than their counterparts. Australia’s gender-led profile contrasts with Canada’s religion-led profile. These descriptive patterns support category- and context-specific inspection, not conclusions about endorsement direction or causal country effects. In consequential collaborative workflows, examining actual responses before acting on aggregate scores can help avoid unsupported fairness assurances affecting people’s opportunities and treatment.

## Limitations

Reddit is not representative of all speakers or deployment settings. Community provenance does not establish nationality, and topic, period, moderation, and demographics can differ. Native-context comparisons cannot isolate language or country effects. Binary gender labels and White–Black and Christian–Muslim pairs omit substantial identity diversity. Category ordering depends on the selected attributes and retained subcorpora; no alternativelexicon or topic-matched robustness experiment is reported.

Absolute scores discard preference direction, and item-level confidence intervals are not estimated. Rankings are descriptive, not tests of population differences. Repetition dependence and conditioning on valid responses further limit the independent binomial reference. Access-family comparisons confound model scale, training, alignment, and serving.

## Ethical Considerations

Before annotation, MTurk workers received information about the study, including a warning that excerpts could contain hostile or favourable stereotypes and potentially distressing material. The statement described voluntary participation, the option to decline answers or discontinue without penalty, compensation through MTurk for completed tasks, and academic-research use of the annotations. Consent was indicated by proceeding after this information was presented. Appendix A reproduces the statement used before annotation. This study received institutional ethics approval; identifying approval details are withheld for anonymous review.

Annotator identifiers were removed and annotation results reported in aggregate. Other identifiers were removed and verbatim text minimised. Public visibility is not blanket permission to republish Reddit content, and the paper’s licence does not confer rights over third-party posts (Reddit, 2026). No unrestricted raw-text release is provided. Any release of prompts, annotations, response counts, or code requires explicit access terms, privacy safeguards, and applicable retention and deletion procedures.

## References

Arnav Arora, Lucie-Aimée Kaffee, and Isabelle Augenstein. 2023. Probing pre-trained language models for cross-cultural differences in values. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP, pages 114–130. Association for Computational Linguistics.

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2020. Translation artifacts in cross-lingual transfer learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 7674–7684. Association for Computational Linguistics.

Soumya Barikeri, Anne Lauscher, Ivan Vulic, and Goran´ Glavaš. 2021. RedditBias: A real-world resource for bias evaluation and debiasing of conversational language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1941–1955. Association for Computational Linguistics.

Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. 2020. The Pushshift Reddit dataset. Proceedings ofthe International AAAI Conference on Web and Social Media, 14(1):830–839.

Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of “bias” in NLP. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 5454–5476. Association for Computational Linguistics.

Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.

Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. 2021. BOLD: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 862–872. Association for Computing Machinery.

Cristina España-Bonet and Alberto Barrón-Cedeño. 2024. Elote, choclo and mazorca: On the varieties of Spanish. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3689–3711. Association for Computational Linguistics.

Susan T. Fiske, Amy J. C. Cuddy, Peter Glick, and Jun Xu. 2002. A model of (often mixed) stereotype content: Competence and warmth respectively follow from perceived status and competition. Journal of Personality and Social Psychology, 82(6):878–902.

Eve Fleisig, Genevieve Smith, Madeline Bossi, Ishita Rustagi, Xavier Yin, and Dan Klein. 2024. Linguistic bias in ChatGPT: Language models reinforce dialect discrimination. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13541–13564. Association for Computational Linguistics.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Lance Calvin Lim Gamboa, Yue Feng, and Mark G. Lee. 2025. Social bias in multilingual language models: A survey. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 27857–27880. Association for Computational Linguistics.

María Grandury, Javier Aula-Blasco, Júlia Falcão, Clé- mentine Fourrier, Miguel González Saiz, Gonzalo Martínez, Gonzalo Santamaria Gomez, Rodrigo Agerri, Nuria Aldama García, Luis Chiruzzo, Javier Conde, Helena Gomez Adorno, Marta Guerrero Nieto, Guido Ivetta, Natàlia López Fuertes, Flor Miriam Plaza-del Arco, María-Teresa Martín-Valdivia, Helena Montoro Zamorano, Carmen Muñoz Sanz, and 6 others. 2025. La leaderboard: A large language model leaderboard for Spanish varieties and languages of Spain and Latin America. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32482–32524. Association for Computational Linguistics.

Daniel Hershcovich, Stella Frank, Heather Lent, Miryam de Lhoneux, Mostafa Abdou, Stephanie Brandl, Emanuele Bugliarello, Laura Cabello Piqueras, Ilias Chalkidis, Ruixiang Cui, Constanza Fierro, Katerina Margatina, Phillip Rust, and Anders Sogaard. 2022. Challenges and strategies in crosscultural NLP. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6997–7013. Association for Computational Linguistics.

Po-Sen Huang, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes Welbl, Jack Rae, Vishal Maini, Dani Yogatama, and Pushmeet Kohli. 2020. Reducing sentiment bias in language models via counterfactual evaluation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 65–83. Association for Computational Linguistics.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and

fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 6282–6293. Association for Computational Linguistics.

Aaron C. Kay, Martin V. Day, Mark P. Zanna, and A. David Nussbaum. 2013. The insidious (and ironic) effects of positive stereotypes. Journal of Experimental Social Psychology, 49(2):287–291.

Kristian Lum, Jacy Reese Anthis, Kevin Robinson, Chirag Nagpal, and Alexander Nicholas D’Amour. 2025. Bias in language models: Beyond trick tests and towards RUTEd evaluation. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 137–161. Association for Computational Linguistics.

Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. 2019. On measuring social biases in sentence encoders. In Proceedings of NAACL-HLT 2019, pages 622–628. Association for Computational Linguistics.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 220–229. Association for Computing Machinery.

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5356–5371. Association for Computational Linguistics.

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-pairs: A challenge dataset for measuring social biases in masked language models. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pages 1953–1967. Association for Computational Linguistics.

Tarek Naous, Michael J. Ryan, Alan Ritter, and Wei Xu. 2024. Having beer after prayer? measuring cultural bias in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16366–16393. Association for Computational Linguistics.

Shangrui Nie, Michael Fromm, Charles Welch, Rebekka Görge, Akbar Karimi, Joan Plepi, Nazia Mowmita, Nicolas Flores-Herr, Mehdi Ali, and Lucie Flek. 2024. Do multilingual large language models mitigate stereotype bias? In Proceedings of the 2nd Workshop on Cross-Cultural Considerations in NLP, pages 65–83. Association for Computational Linguistics.

Debora Nozza, Federico Bianchi, and Dirk Hovy. 2021. HONEST: Measuring hurtful sentence completion in language models. In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2398–2406. Association for Computational Linguistics.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. 2022. BBQ: A hand-built bias benchmark for question answering. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2086–2105. Association for Computational Linguistics.

Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. 2020. Closing the AI accountability gap: Defining an end-to-end framework for internal algorithmic auditing. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, pages 33–44. Association for Computing Machinery.

Abhinav Rao, Akhila Yerukola, Vishwa Shah, Katharina Reinecke, and Maarten Sap. 2025. NormAd: A framework for measuring the cultural adaptability of large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2373–2403. Association for Computational Linguistics.

Reddit. 2026. Developer terms. Accessed 30 October 2026.

Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4512–4525, Online. Association for Computational Linguistics.

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5477–5490. Association for Computational Linguistics.

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. 2019. The woman worked as a babysitter: On biases in language generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3407–3412. Association for Computational Linguistics.

John Oliver Siy and Sapna Cheryan. 2016. Prejudice masquerading as praise: The negative echo of posi-

tive stereotypes. Personality and Social Psychology Bulletin, 42(7):941–954.

Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, and Adina Williams. 2022. “I’m Sorry to Hear That”: Finding new biases in language models with a holistic descriptor dataset. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 9180–9211. Association for Computational Linguistics.

Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer. 2019. Evaluating gender bias in machine translation. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 1679–1684. Association for Computational Linguistics.

Zeerak Talat, Aurélie Névéol, Stella Biderman, Miruna Clinciu, Manan Dey, Shayne Longpre, Sasha Luccioni, Maraim Masoud, Margaret Mitchell, Dragomir Radev, Shanya Sharma, Arjun Subramonian, Jaesung Tae, Samson Tan, Deepak Tunuguntla, and Oskar Van Der Wal. 2022. You reap what you sow: On the challenges of bias evaluation under multilingual settings. In Proceedings ofBigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 26–41. Association for Computational Linguistics.

Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, Courtney Biles, Sasha Brown, Zac Kenton, Will Hawkins, Tom Stepleton, Abeba Birhane, Lisa Anne Hendricks, Laura Rimell, William Isaac, and 4 others. 2022. Taxonomy of risks posed by language models. In Proceedings ofthe 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 214–229. Association for Computing Machinery.

Jieyu Zhao, Subhabrata Mukherjee, Saghar Hosseini, Kai-Wei Chang, and Ahmed Hassan Awadallah. 2020. Gender bias in multilingual embeddings and crosslingual transfer. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2896–2907. Association for Computational Linguistics.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2, pages 15–20. Association for Computational Linguistics.

## A MTurk Participant Information and Consent

The following statement was presented to MTurk workers before they annotated the data reported in this paper. Its wording is reproduced here, with formatting adapted to the paper layout.

## Participant Information and Ethical Statement

You are invited to participate in an academic research study examining how large language models associate social groups with different characteristics across languages and cultural contexts.

As part of this task, you will be asked to review short excerpts of publicly available Reddit posts and classify whether they contain stereotypes relating to gender, race, or religion. Some materials may express negative or hostile stereotypes, while others may contain seemingly favourable or positive stereotypes. These materials are presented solely for research and evaluation purposes and do not represent the views or beliefs of the researchers. Potential risks and discomfort: Some excerpts may contain sensitive, offensive, or potentially distressing content involving social identities. Please be aware of this before proceeding. You may decline to answer or discontinue participation if you feel uncomfortable.

Voluntary participation: Your participation is entirely voluntary. You are free to stop participating at any time, without penalty.

Compensation: You will receive compensation through Amazon Mechanical Turk for completing the assigned annotation task.

Privacy and confidentiality: Your personal identity will not be disclosed in any research publication. Annotator identifiers are removed from the research dataset, and results are reported in aggregate. The research focuses on annotation judgements rather than the personal opinions, beliefs, or identities of individual participants.

Data use: The annotations collected will be used exclusively for academic research on multilingual stereotype patterns and the evaluation of large language models. The research findings may be published in academic papers and related research outputs. Individual annotators will not be identified in these publications.

By proceeding with this task, you acknowledge that you have read and understood this information, are aware of the potentially sensitive nature of the materials, and voluntarily agree to participate in the annotation study.