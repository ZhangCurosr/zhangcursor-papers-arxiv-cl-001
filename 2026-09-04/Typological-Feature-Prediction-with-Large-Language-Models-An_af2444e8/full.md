# Typological Feature Prediction with Large Language Models: An In-Context Learning Approach

Qianwen Wang<sup>r∗</sup> York Hay Ng<sup>r∗</sup> Aditya Khan<sup>r</sup> En-Shiun Annie Lee<sup>r♣</sup> <sup>r</sup>University of Toronto, Canada <sup>♣</sup>Ontario Tech University, Canada

vivianqianwen.wang@mail.utoronto.ca, york.ng@mila.quebec, aditya.khan@columbia.edu

## Abstract

Typological features are widely used in multilingual NLP, and the prediction of such features holds downstream utility. However, existing methods to predict missing values lack interpretable justifications for predictions, while their performance across resource levels and feature types remains underexplored. Given LLMs’ abilities in meta-linguistic reasoning and in providing rationales, we investigate LLMs’ performance in typological feature prediction via an in-context learning approach with linguistic data from URIEL+ and Glottolog. We find that zero-shot prompting is insuficient, but when given phylogenetic and geographic neighbour evidence, LLMs substantially outperform all baselines without disadvantaging low-resource languages. We further find that most LLM rationales are consistent with the provided evidence, ofering a step toward explainable typological feature prediction.

## 1 Introduction

Typological features describe the structural properties of languages. They are widely used across multilingual NLP for tasks including transfer language selection (Lin et al., 2019), cross-lingual dependency parsing (Naseem et al., 2012; Täckström et al., 2013; de Lhoneux et al., 2018; Üstün et al., 2020), and performance prediction of multilingual models (Hirak et al., 2026; Anugraha et al., 2025; Xia et al., 2020). Surveys have confirmed that typological features benefit downstream performance (Ponti et al., 2019; Bjerva, 2024).

The databases that house these features, however, are usually incomplete. URIEL+ (Khan et al., 2025), the most recent typological knowledge base, aggregates features from databases such as WALS (Dryer and Haspelmath, 2013), Grambank (Skirgård et al., 2023) and PHOIBLE (Moran and McCloy, 2019), yet 87% of its typological feature matrix remains empty (Ng et al., 2025). Moreover, sparsity is worst precisely for the lowresource languages (LRLs) that stand to benefit most from methods using those typological features (Shipton et al., 2026).

Typological feature prediction is a task aimed at addressing this gap. Existing approaches use knearest-neighbour imputation (Littell et al., 2017), low-rank matrix completion (Khan et al., 2025; Ng et al., 2026), and random forests trained on external features (Amirzadeh et al., 2025). However, inequalities in prediction performance across resource levels and feature types remain underexplored, while output predictions are usually not accompanied with interpretable justifications.

Recent work has shown that LLMs exhibit metalinguistic reasoning skills (Yang et al., 2025). Coupled with their ability to output rationales, they are a natural candidate for this task. In this work, we ask: how well do pretrained LLMs reason over linguistic evidence, and provide rationales,for typological feature prediction?

We propose an in-context learning approach, constructing prompts with data from URIEL+ (Khan et al., 2025) and Glottolog (Hammarström et al., 2025), a catalogue of the world’s languages and families. Our exploration suggests that:

1. Zero-shot prompting, where only language metadata is provided, is insuficient for this task, but LLMs excel at reasoning over linguistic evidence when evidence is provided.

2. LLMs, once given suficient contextual information, exhibit consistent performance across language resource levels and typological properties, consistently exceeding non-LLM baselines on typological feature prediction.

3. Human annotators confirm that our prompt design supplies suficient evidence, and find most LLM generated rationales to be consistent with the evidence.

## 2 Related Work

Typological Feature Prediction. The SIGTYP 2020 Shared Task (Bjerva et al., 2020) formalized the task of typological feature prediction over WALS. The winning system combined conditional probabilities with language embeddings (Vastl et al., 2020). Methods that operate on typological data alone include k-nearest-neighbour imputation and SoftImpute, a low-rank matrix factorization used in URIEL+ (Khan et al., 2025) that remains a strong baseline.

A separate line of work incorporates external data. Amirzadeh et al. (2025) proposed per-feature random forests trained with POS-tag distributions and Wikipedia statistics, while prior probing-based approaches predict features from multilingual representations trained on parallel text (Malaviya et al., 2017; Östling and Kurfalı, 2023).

Bjerva (2024) critiques these methods as fundamentally correlation-based. Easy features are trivially predicted while performance in atypical cases, despite being the most typologically interesting, remains poor. In a survey of linguists, explainability was rated as the most important property of prediction methods, which all existing methods lack. Moreover, methods that rely on external data sources or parallel text are inapplicable to the lowresource languages with more missing features.

LLMs and Linguistic Reasoning. A growing body of work demonstrates that LLMs can perform substantive linguistic analysis. Tanzer et al. (2024) showed that prompting an LLM with a single grammar book enables translation of unseen low-resource languages, although Aycock et al. (2025) found that performance mainly hinges on the book’s parallel examples. Meanwhile, Yang et al. (2025) found that linguistic cues (e.g. grammatical descriptions) consistently improve LLMs’ meta-linguistic reasoning across typologically diverse languages. In typological feature prediction, Hus and Anastasopoulos (2026) applied retrievalaugmented generation with grammar books to predict Grambank features, demonstrating that retrieval yielded substantial improvement.

Our work continues to investigate LLMs’ ability to reason over linguistic evidence. In contrast to prior approaches, our method does not rely on external resources where availability is limited for low-resource languages, while aiding interpretability by providing rationales of predictions.

## 3 Method

## 3.1 Task Formulation

We frame typological feature prediction as an imputation task over an existing knowledge base against held-out observed values, following the formulation identified in Bjerva (2024). We apply this task on the URIEL+ typological matrix (Khan et al., 2025), given its comprehensive coverage of typological features and languages. Let $\mathbf { M } \in \{ 0 , 1 , \bot \} ^ { N \times F }$ denote URIEL+, with $N { = } 4 { , } 5 5 5$ languages and $F { = } 8 0 0$ binary typological features, where ⊥ indicates a missing entry. Given a query $( \ell , f )$ over target language ℓ and typological feature $f ,$ , where $\mathbf { M } [ \ell , f ] = \bot$ , the task is to predict $\hat { \nu } \in \{ 0 , 1 \}$ . We use Glottolog (Hammarström et al., 2025) as the sole source of metadata for ℓ, as it provides data for all languages covered by URIEL+.

Data splits. We aim to preserve equal representation between typological feature types and language resource levels in our data splits. URIEL+ documents four types of typological features: syntactic, morphological, phonological and inventory. Moreover, we categorise the resource level of languages by the number of features with an observed value. The bottom 33% of languages form the low-resource group (LRL), the middle 34% the medium-resource group (MRL), and the top 33% the high-resource group (HRL).

We then sample 100 $( \ell , f )$ observed pairs for each resource level and feature type where $\mathbf { M } [ \ell , f ] \neq \bot$ , yielding 1 200 pairs as our test set <sup>1</sup>. A separate validation set of equal size is held out for hyperparameter tuning for baseline methods. The remaining 888,741 observed entries form the training matrix $\mathbf { M } _ { \mathrm { t r a i n } }$

## 3.2 Prompt Construction

For each query, we construct a prompt from four content blocks drawn from URIEL+ and Glottolog. The full prompt template is shown in Appendix A.

(1) Target language metadata. The language’s name, Glottocode, ISO 639-3 code, language family, and macroarea.

(2) Anchor features. For each target feature $f ,$ we pre-compute the top 10 features most correlated with f across ${ \bf M } _ { \mathrm { t r a i n } }$ , termed anchor features. Given a query language ℓ, we then provide the top

<table><tr><td>Method</td><td>Inputs</td><td>LRL</td><td>MRL</td><td>HRL</td><td>S</td><td>P</td><td>INV</td><td>M</td><td>Overall</td></tr><tr><td>Random</td><td>一</td><td>.500</td><td>.613</td><td>.584</td><td>.446</td><td>.689</td><td>.696</td><td>.402</td><td>.568</td></tr><tr><td>Majority</td><td>1</td><td>.637</td><td>.660</td><td>.659</td><td>.565</td><td>.777</td><td>.730</td><td>.427</td><td>.653</td></tr><tr><td>kNN (cosine) (k = 3)</td><td>(A)</td><td>.754</td><td>.741</td><td>.817</td><td>.690</td><td>.815</td><td>.927</td><td>.662</td><td>.775</td></tr><tr><td>SoftImpute</td><td>(A)</td><td>.800</td><td>.817</td><td>.841</td><td>.721</td><td>.907</td><td>.908</td><td>.715</td><td>.821</td></tr><tr><td>kNN (phylogenetic) (k = 6)</td><td>(A) + (B)</td><td>.732</td><td>.718</td><td>.695</td><td>.659</td><td>.794</td><td>.753</td><td>.600</td><td>.714</td></tr><tr><td>kNN (geographic) (k = 6)</td><td>(A) + (C)</td><td>.720</td><td>.773</td><td>.710</td><td>.689</td><td>.795</td><td>.774</td><td>.646</td><td>.733</td></tr><tr><td>Random Forest</td><td>(A) + (B) + (C)</td><td>.739</td><td>.719</td><td>.734</td><td>.663</td><td>.852</td><td>.743</td><td>.595</td><td>.731</td></tr><tr><td>Llama-3.1-70B in-context prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base, zero-shot (metadata only)</td><td></td><td>.433</td><td>.424</td><td>.531</td><td>.463</td><td>.521</td><td>.418</td><td>.442</td><td>.465</td></tr><tr><td>+ anchor features</td><td>(A)</td><td>.459</td><td>.482</td><td>.587</td><td>.483</td><td>.592</td><td>.547</td><td>.411</td><td>.512</td></tr><tr><td>+ anchor + phylogenetic neighb.</td><td>(A) + (B)</td><td>.759</td><td>.795</td><td>.789</td><td>.785</td><td>.845</td><td>.787</td><td>.690</td><td>.782</td></tr><tr><td>+ anchor + geographic neighb.</td><td>(A) + (C)</td><td>.736</td><td>.778</td><td>.833</td><td>.793</td><td>.780</td><td>.867</td><td>.720</td><td>.788</td></tr><tr><td>+ all</td><td> $\mathbf { \eta } ( \mathbf { A } ) + \mathbf { \eta } ( \mathbf { B } ) + \mathbf { \eta } ( \mathbf { C } )$ </td><td>.919</td><td>.895</td><td>.914</td><td>.914</td><td>.917</td><td>.933</td><td>.868</td><td>.909</td></tr><tr><td>Gemma-4-31B in-context prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base, zero-shot (metadata only)</td><td>一</td><td>.472</td><td>.547</td><td>.539</td><td>.507</td><td>.669</td><td>.336</td><td>.450</td><td>.521</td></tr><tr><td>+ anchor features</td><td>(A)</td><td>.580</td><td>.654</td><td>.629</td><td>.508</td><td>.777</td><td>.677</td><td>.476</td><td>.622</td></tr><tr><td>+ anchor + phylogenetic neighb.</td><td>(A) + (B)</td><td>.828</td><td>.847</td><td>.831</td><td>.786</td><td>.899</td><td>.872</td><td>.762</td><td>.835</td></tr><tr><td>+ anchor + geographic neighb.</td><td>(A) + (C)</td><td>.858</td><td>.878</td><td>.929</td><td>.871</td><td>.914</td><td>.944</td><td>.841</td><td>.893</td></tr><tr><td>+ all</td><td> $\mathbf { \eta } ( \mathbf { A } ) + \mathbf { \eta } ( \mathbf { B } ) + \mathbf { \eta } ( \mathbf { C } )$ </td><td>.834</td><td>.840</td><td>.907</td><td>.841</td><td>.900</td><td>.909</td><td>.802</td><td>.865</td></tr><tr><td>GPT-5.5 in-context prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base, zero-shot (metadata only)</td><td></td><td>.622</td><td>.657</td><td>.693</td><td>.607</td><td>.767</td><td>.653</td><td>.602</td><td>.661</td></tr><tr><td>+ anchor features</td><td>(A)</td><td>.687</td><td>.707</td><td>.754</td><td>.600</td><td>.851</td><td>.816</td><td>.586</td><td>.719</td></tr><tr><td>+ anchor + phylogenetic neighb.</td><td>(A) + (B)</td><td>.847</td><td>.872</td><td>.837</td><td>.811</td><td>.908</td><td>.921</td><td>.742</td><td>.851</td></tr><tr><td>+ anchor + geographic neighb.</td><td>(A) + (C)</td><td>.880</td><td>.899</td><td>.915</td><td>.923</td><td>.928</td><td>.937</td><td>.800</td><td>.900</td></tr><tr><td>+ all</td><td> $\mathbf { \eta } ( \mathbf { A } ) + \mathbf { \eta } ( \mathbf { B } ) + \mathbf { \eta } ( \mathbf { C } )$ </td><td>.869</td><td>.905</td><td>.881</td><td>.872</td><td>.917</td><td>.930</td><td>.807</td><td>.885</td></tr></table>

Table 1: Macro F1 of feature prediction methods on URIEL+, grouped by language resource level (LRL/MRL/HRL) and by feature type (S=syntax, P=phonology, INV=phonetic inventory, M=morphology). Inputs: (A)=typological matrix, (B)=phylogenetic neighbours, (C)=geographic neighbours. Cell shading is column-wise (darker = higher F1); per-column best is in bold.

5 anchor feature values that are observed for ℓ . Additionally, for each observed anchor feature a with value $\nu _ { a } ,$ we provide the global prevalence of the feature $P ( f { = } 1 \mid a { = } \nu _ { a } )$ over ${ \bf M } _ { \mathrm { t r a i n } }$

(3) Phylogenetic neighbours. We build a pool of candidates by breadth-first search from ℓ in the Glottolog family tree, selecting 5 languages greedily to maximise coverage of the target and anchor features. For each neighbour we include its value for f (if observed), up to 3 observed anchor values, and its proximity rank. We further aggregate yes/no/missing vote counts for f across the set of neighbours, and provide details of the closest phylogenetic neighbour supporting both values of f as contrastive evidence.

(4) Geographic neighbours. We construct this pool identically as (3), with the candidate pool ranked by Haversine distance over Glottolog coordinates. We take the same per-neighbour data, vote counts, and contrastive evidence as above.

## 3.3 Baselines

We compare against seven imputation methods, organized by the information they consume:

Trivial baselines. Random draws each prediction from a Bernoulli distribution with the per-feature positive rate over $\mathbf { M } _ { \mathrm { t r a i n } }$ . Majority predicts the more frequent value per feature.

Matrix-only methods. kNN-cosine (Littell et al., 2017) performs k-nearest-neighbour imputation over ${ \bf M } _ { \mathrm { t r a i n } }$ using cosine similarity. SoftImpute (Mazumder et al., 2010) completes the matrix via iterative low-rank completion with soft-thresholded SVD, and it was the strongest baseline in URIEL+ (Khan et al., 2025).

Glottolog-based kNN. To test the utility of phylogenetic and geographic information, we evaluate two variants of kNN-cosine. kNN-phylogenetic ranks neighbours by path distance in the Glottolog tree. kNN-geographic ranks them by Haversine distance over geographic coordinates.

Random Forest. For a direct non-LLM comparison, we mirror Amirzadeh et al. (2025)’s architecture but restrict the model to the data available to the LLM. For each feature, we train a random forest classifier<sup>2</sup> over target language metadata (family, macroarea, coordinates), the values and global prevalence of anchor features, as well as vote count for phylogenetic and geographic neighbours. Since this baseline is provided the same information from Glottolog as the LLM, we aim to isolate the impact of LLM reasoning abilities.

## 4 Results

We evaluate on two open-source models, Llama-3.1-70B (Grattafiori et al., 2024) and Gemma-4- 31B (DeepMind, 2026), on their instruction-tuned variants, alongside one proprietary model, GPT-5.5 (OpenAI, 2026). Table 1 reports macro F1 on the test set across resource levels and feature types.

Zero-shot prompting underperforms baselines. All three LLMs underperform the strongest baselines when given only language metadata (Llama: F1 0.465, Gemma: F1 0.521, GPT-5.5: F1 0.661), where they efectively perform zero-shot prediction. Adding anchor features provides a moderate improvement for all three models, reaching F1 0.512, 0.622, and 0.719 respectively. Nonetheless, these configurations remain below kNN-cosine (F1 0.775) and SoftImpute (F1 0.821). This highlights that metadata-only prompting, where LLMs have little evidence to reason over, is insuficient, therefore motivating the inclusion of typological and neighbour evidence.

LLMs excel at reasoning over evidence. When neighbour evidence is introduced, performance improves substantially. The strongest neighbourbased configuration of every LLM outperforms SoftImpute, the best performing baseline (F1 0.821). Llama with all inputs achieves the highest overall performance (F1 0.909). On the other hand, Gemma and GPT-5.5 instead perform best when anchor features and geographic neighbour evidence are included, reaching F1 0.893 and 0.900 respectively. This shows that providing linguistic evidence improves LLMs’ prediction performance.

Random Forest receives the same Glottologderived inputs as the LLMs, yet it scores below even baselines such as SoftImpute and kNN-cosine, which see only the typological matrix. This suggests that reasoning ability is a key source of the LLMs’ advantage. Appendix D further demonstrates that LLMs’ reasoning exceeds simply following the plurality, and that performance is robust even when evidence disagrees.

With suficient evidence, performance gaps narrow across resource levels and feature types. With metadata alone, all three LLMs perform noticeably worse on LRLs than HRLs. Conversely, when given all inputs, Llama exhibits marginally better performance on LRLs (F1 0.919) than HRLs (F1 0.914). While the inequality persists in Gemma, the performance gap between HRLs and LRLs remains similar, from 0.067 (base) to 0.073 (all). For GPT-5.5, this gap narrows substantially from 0.071 (base) to 0.012 (all).

Across feature types, most baseline methods exhibit varying performance, generally favouring phonological and phonetic inventory features above syntactic and morphological features. By comparison, the LLMs are more consistent across types: when given all inputs, Llama and Gemma’s mean performance across feature types ranges only 0.065 and 0.107 respectively. For GPT-5.5, the corresponding range is 0.123.

## 5 Annotation Study

To study whether the prompt provides suficient evidence, and whether LLM generated rationales are consistent with that evidence, we conducted a human annotation study over a subset of 180 (ℓ f) pairs, collecting 15 pairs per resource level and feature type. Five annotators, given the same prompt as the LLM, independently recorded their prediction and confidence. Separately, they annotated whether the LLM’s rationale contradicts the evidence provided. The full protocol is described in Appendix F.

Suficiency of evidence in prompts. We observed high inter-annotator agreement on predictions, with Fleiss’ κ = 0 888. Aggregated by majority vote, human predictions achieved F1 0.987, with individual annotators averaging an F1 of 0.963. This confirms that the prompt provides suficient evidence to perform feature prediction.

LLM-human comparison. On the same subset, our best performing LLM, Llama-3.1-70B, achieves F1 0.915. While performance is below that of humans, there is strong agreement between Llama and aggregated human predictions, exhibiting Cohen’s κ = 0 831.

<table><tr><td rowspan="2">Confidence</td><td colspan="2">Human</td><td colspan="2">LLM</td></tr><tr><td>Acc.</td><td>%</td><td>Acc.</td><td>%</td></tr><tr><td>Low</td><td>0.794</td><td>0.134</td><td>0.643</td><td>0.156</td></tr><tr><td>Medium</td><td>0.963</td><td>0.256</td><td>0.962</td><td>0.439</td></tr><tr><td>High</td><td>0.997</td><td>0.610</td><td>1.000</td><td>0.406</td></tr></table>

Table 2: Accuracy by self-reported confidence tier. % indicates the proportion of predictions in each tier.

Confidence calibration. Table 2 reports the accuracy for humans and Llama, grouped by confidence. We observe that for both humans and Llama, the accuracy of predictions consistently increases with confidence, with Llama achieving perfect accuracy when high confidence is reported. While Llama tends to report lower confidence compared to humans overall, this result highlights the validity of Llama’s reported confidences.

Rationale quality. On average, annotators identified contradictions in 20.9% of Llama’s rationales, though inter-annotator agreement on contradiction judgments was moderate (Fleiss’ κ = 0 415), which reflects the subjective nature of evaluating rationales. While the fact that the majority of rationales are consistent with the provided evidence does not inherently imply the typological validity of the rationales, it nevertheless demonstrates that Llama is moderately successful at generating consistent rationales <sup>3</sup>. This aids in the explainability of predictions.

## 6 Conclusion

We investigated whether LLMs can perform typological feature prediction through reasoning over linguistic evidence. Our results suggest that while metadata-only prompting underperforms baselines, LLMs excel at integrating phylogenetic, geographic and typological evidence in ways that classical methods cannot. Simultaneously, providing more evidence helps LLMs perform more consistently across language resource levels and typological feature types. We further conducted a human annotation study, which supports our prompt design and finds that most LLM rationales are consistent with the evidence given, ofering a step toward the explainable prediction systems called for by Bjerva (2024).

## Limitations

Task assumptions. Our evaluation tests predictions only on entries that were already observed. However, prediction of the 87% of entries that are truly missing from URIEL+ might represent a harder task, and may involve more atypical (language, feature) pairs. Moreover, while many typological features lie on a gradient rather than being categorical or even binary features (Levshina et al., 2023), our task inherits the binary assumption of knowledge bases such as URIEL and URIEL+. We frame this work as an exploratory study of LLMs as a prediction method, rather than a comprehensive evaluation of their typological knowledge.

Furthermore, our proposed method relies on evidence being available from neighbouring languages, and is thus tied to the data availability of the knowledge base used. We acknowledge that where no such evidence exists, our method is inapplicable; hence it should not be taken as a substitute for field-linguistic documentation.

Evaluation scale. Due to the computational costs of language model inference, we perform prediction only on a limited subset of 1,200 examples. While this may not be wholly representative of performance on the whole dataset, and may be subject to noise, we mitigate this concern by ensuring equal representation of language resource levels and typological feature types in both test and validation splits. We however acknowledge that the aforementioned costs pose a substantial barrier to large-scale typological feature prediction or even database completion.

Explainability boundaries. While our method generates rationales alongside predictions, we acknowledge that this may not be a truthful representation of the model’s internal reasoning process (Turpin et al., 2023). Although the majority of rationales were found to be consistent, our failure analysis (Appendix G) confirms that correct predictions can occasionally accompany flawed rationales. Therefore, rationales are better understood as a plausible justification of the prediction only.

Model scope. Our experiments cover two opensource model families alongside GPT-5.5, a proprietary frontier model. Hence, our evaluation remains limited to three model families. Nonetheless, the substantial improvements from linguistic evidence are observed across all three, suggesting that our findings are not necessarily tied to a single architecture or model family. Generally, our observed trends are stable across model families, suggesting that the central conclusions are not tied to a single architecture.

Data contamination. URIEL+ does not contribute any novel typological data on its own, but rather aggregates multiple typological knowledge bases, which were released before the knowledge cut-of for the models used in our study. Thus, there is a minor risk that those models are relying only on pretrained knowledge. However, the poor performance under the base configuration across models, and the fact that URIEL+ additionally preprocesses the data in its constituent knowledge bases, suggests that data contamination poses only a minimal risk.

## Ethical Considerations

The task of typological feature prediction involves linguistic data from typological databases, which is free from any personally identifiable data. We source our data from URIEL+, which is publicly available.

We collect only numeric annotations of feature value predictions, confidences, and contradictions from human annotators in a structured file format. The data collected is thus free from personally identifiable information. All annotations are reported in aggregate, without any personal nor sensitive information.

## Acknowledgments

We are thankful to Richard Tzong-Han Tsai and his lab for helping run API-credentialed GPT-5.5 experiments. In addition, we are incredibly grateful to the following annotators for volunteering their time: Jiaqi Ji, Jinxi Li, Mason Shipton, and Qifan Yang. Finally, we thank Madelaine Jarcew for her feedback on the annotator study, and for providing us with linguistic resources to help annotators complete their task.

## References

Hamidreza Amirzadeh, Sadegh Jafari, Anika Harju, and Rob van der Goot. 2025. data2lang2vec: Data driven typological features completion. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6520–6529, Abu Dhabi, UAE. Association for Computational Linguistics.

David Anugraha, Genta Indra Winata, Chenyue Li, Patrick Amadeus Irawan, and En-Shiun Annie Lee. 2025. ProxyLM: Predicting language model performance on multilingual tasks via proxy models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1981–2011, Albuquerque, New Mexico. Association for Computational Linguistics.

Seth Aycock, David Stap, Di Wu, Christof Monz, and Khalil Simaan. 2025. Can LLMs really learn to translate a low-resource language from one grammar book? In International Conference on Learning Representations, volume 2025, pages 12334–12357.

Johannes Bjerva. 2024. The role of typological feature prediction in NLP and linguistics. Computational Linguistics, 50(2):781–794.

Johannes Bjerva, Elizabeth Salesky, Sabrina J. Mielke, Aditi Chaudhary, Giuseppe G. A. Celano, Edoardo Maria Ponti, Ekaterina Vylomova, Ryan Cotterell, and Isabelle Augenstein. 2020. SIGTYP 2020 shared task: Prediction of typological features. In Proceedings ofthe Second Workshop on Computational Research in Linguistic Typology, pages 1–11, Online. Association for Computational Linguistics.

Miryam de Lhoneux, Johannes Bjerva, Isabelle Augenstein, and Anders Søgaard. 2018. Parameter sharing between dependency parsers for related languages. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4992–4997, Brussels, Belgium. Association for Computational Linguistics.

Google DeepMind. 2026. Gemma 4 model card.

Matthew S. Dryer and Martin Haspelmath, editors. 2013. WALS Online (v2020.4). Zenodo.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, et al. 2024. The llama 3 herd of models.

Harald Hammarström, Robert Forkel, Martin Haspelmath, and Sebastian Bank. 2025. Glottolog 5.2. Accessed: 2025-09-16.

Vitalii Hirak, Jaap Jumelet, and Arianna Bisazza. 2026. Assessing the impact of typological features on multilingual machine translation in the age of large language models. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2416–2434, Rabat, Morocco. Association for Computational Linguistics.

Jonathan Hus and Antonios Anastasopoulos. 2026. A RAG approach for typological database completion. In Proceedings ofthe 8th Workshop on Research in Computational Linguistic Typology and Multilingual NLP, pages 39–49, Rabat, Morocco. Association for Computational Linguistics.

Aditya Khan, Mason Shipton, David Anugraha, Kaiyao Duan, Phuong H. Hoang, Eric Khiu, A. Seza Dogruöz, and En-Shiun Annie Lee. 2025.˘ URIEL+: Enhancing linguistic inclusion and usability in a typological and multilingual knowledge base. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6937–6952, Abu Dhabi, UAE. Association for Computational Linguistics.

Natalia Levshina, Savithry Namboodiripad, Marc Allassonnière-Tang, Mathew Kramer, Luigi Talamo, Annemarie Verkerk, Sasha Wilmoth, Gabriela Garrido Rodriguez, Timothy Michael Gupton, Evan Kidd, Zoey Liu, Chiara Naccarato, Rachel Nordlinger, Anastasia Panova, and Natalia Stoynova. 2023. Why we need a gradient approach to word order. Linguistics, 61(4):825–883.

Yu-Hsiang Lin, Chian-Yu Chen, Jean Lee, Zirui Li, Yuyan Zhang, Mengzhou Xia, Shruti Rijhwani, Junxian He, Zhisong Zhang, Xuezhe Ma, Antonios Anastasopoulos, Patrick Littell, and Graham Neubig. 2019. Choosing transfer languages for cross-lingual learning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3125–3135, Florence, Italy. Association for Computational Linguistics.

Patrick Littell, David R. Mortensen, Ke Lin, Katherine Kairis, Carlisle Turner, and Lori Levin. 2017. URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 8–14, Valencia, Spain. Association for Computational Linguistics.

Chaitanya Malaviya, Graham Neubig, and Patrick Littell. 2017. Learning language representations for typology prediction. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2529–2535, Copenhagen, Denmark. Association for Computational Linguistics.

Rahul Mazumder, Trevor Hastie, and Robert Tibshirani. 2010. Spectral regularization algorithms for learning large incomplete matrices. J. Mach. Learn. Res., 11:2287–2322.

Steven Moran and Daniel McCloy. 2019. PHOIBLE 2.0.

Tahira Naseem, Regina Barzilay, and Amir Globerson. 2012. Selective sharing for multilingual dependency parsing. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 629–637, Jeju Island, Korea. Association for Computational Linguistics.

York Hay Ng, Phuong Hanh Hoang, and En-Shiun Annie Lee. 2025. Less is more: The efectiveness of compact typological language representations. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25805–25816, Suzhou, China. Association for Computational Linguistics.

York Hay Ng, Aditya Khan, Xiang Lu, Matteo Salloum, Michael Zhou, Phuong Hanh Hoang, A. Seza Dogruöz, and En-Shiun Annie Lee. 2026.˘ Modality matching matters: Calibrating language distances for cross-lingual transfer in URIEL+. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 4: Student Research Workshop), pages 110–130, Rabat, Morocco. Association for Computational Linguistics.

OpenAI. 2026. Introducing GPT-5.5.

Robert Östling and Murathan Kurfalı. 2023. Language embeddings sometimes contain typological generalizations. Computational Linguistics, 49(4):1003– 1051.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Edoardo Maria Ponti, Helen O’Horan, Yevgeni Berzak, Ivan Vulic, Roi Reichart, Thierry Poibeau, Ekate-´ rina Shutova, and Anna Korhonen. 2019. Modeling language variation and universals: A survey on typological linguistics for natural language processing. Computational Linguistics, 45(3):559–601.

Mason Shipton, York Hay Ng, Aditya Khan, Phuong H. Hoang, Xiang Lu, A. Seza Dogruöz, and Annie En-˘ Shiun Lee. 2026. Simple additions, substantial gains: Expanding scripts, languages, and lineage coverage in URIEL+. In Proceedings of the Fifteenth Language Resources and Evaluation Conference, pages 11045–11059, Palma de Mallorca, Spain. ELRA Language Resource Association.

Hedvig Skirgård, Hannah J. Haynie, Damián E. Blasi, Harald Hammarström, Jeremy Collins, Jay J. Latarche, Jakob Lesage, Tobias Weber, Alena Witzlack-Makarevich, Sam Passmore, Angela Chira, Luke Maurits, Russell Dinnage, Michael Dunn, Ger Reesink, Ruth Singer, Claire Bowern, Patience Epps, Jane Hill, Outi Vesakoski, Martine Robbeets, Noor Karolin Abbas, Daniel Auer, Nancy A. Bakker, Giulia Barbos, Robert D. Borges, Swintha Danielsen, Luise Dorenbusch, Ella Dorn, John Elliott, Giada Falcone, Jana Fischer, Yustinus Ghanggo Ate, Hannah Gibson, Hans-Philipp Göbel, Jemima A. Goodall, Victoria Gruner, Andrew Harvey, Rebekah Hayes, Leonard Heer, Roberto E. Herrera Miranda, Nataliia Hübler, Biu Huntington-Rainey, Jessica K. Ivani, Marilen Johns, Erika Just, Eri Kashima, Carolina Kipf, Janina V. Klingenberg, Nikita König, Aikaterina Koti, Richard G. A. Kowalik, Olga Krasnoukhova, Nora L. M. Lindvall, Mandy Lorenzen, Hannah Lutzenberger, Tânia R. A. Martins, Celia Mata German, Suzanne van der Meer, Jaime Montoya Samamé, Michael Müller, Saliha Muradoglu, Kelsey Neely, Johanna Nickel, Miina

Norvik, Cheryl Akinyi Oluoch, Jesse Peacock, India O. C. Pearey, Naomi Peck, Stephanie Petit, Sören Pieper, Mariana Poblete, Daniel Prestipino, Linda Raabe, Amna Raja, Janis Reimringer, Sydney C. Rey, Julia Rizaew, Eloisa Ruppert, Kim K. Salmon, Jill Sammet, Rhiannon Schembri, Lars Schlabbach, Frederick W. P. Schmidt, Amalia Skilton, Wikaliler Daniel Smith, Hilário de Sousa, Kristin Sverredal, Daniel Valle, Javier Vera, Judith Voß, Tim Witte, Henry Wu, Stephanie Yam, Jingting Ye, Maisie Yong, Tessa Yuditha, Roberto Zariquiey, Robert Forkel, Nicholas Evans, Stephen C. Levinson, Martin Haspelmath, Simon J. Greenhill, Quentin D. Atkinson, and Russell D. Gray. 2023. Grambank reveals the importance of genealogical constraints on linguistic diversity and highlights the impact of language loss. Science Advances, 9(16):eadg6175.

Oscar Täckström, Ryan McDonald, and Joakim Nivre. 2013. Target language adaptation of discriminative transfer parsers. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1061–1071, Atlanta, Georgia. Association for Computational Linguistics.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. 2024. A benchmark for learning to translate a new language from one grammar book. In International Conference on Learning Representations, volume 2024, pages 18955–18985.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: unfaithful explanations in chain-of-thought prompting. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Ahmet Üstün, Arianna Bisazza, Gosse Bouma, and Gertjan van Noord. 2020. UDapter: Language adaptation for truly Universal Dependency parsing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2302–2315, Online. Association for Computational Linguistics.

Martin Vastl, Daniel Zeman, and Rudolf Rosa. 2020. Predicting typological features in WALS using language embeddings and conditional probabilities: ÚFAL submission to the SIGTYP 2020 shared task. In Proceedings of the Second Workshop on Computational Research in Linguistic Typology, pages 29–35, Online. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical

Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Mengzhou Xia, Antonios Anastasopoulos, Ruochen Xu, Yiming Yang, and Graham Neubig. 2020. Predicting performance for natural language processing tasks. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 8625– 8646, Online. Association for Computational Linguistics.

Changbing Yang, Franklin Ma, Freda Shi, and Jian Zhu. 2025. LingGym: How far are LLMs from thinking like field linguists? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1314–1340, Suzhou, China. Association for Computational Linguistics.

## A ICL Prompt Template

This appendix provides the prompt template used for in-context learning based on linguistic evidence.

## A.1 System Message

You are a linguist working with data from a linguistic knowledge base.

Your task is to predict a target language’s typological features given facts about the target language, other observed typological features, and evidence from phylogenetic and geographic neighbors.

## A.2 User Prompt

The user prompt is assembled from multiple evidence blocks, including metadata, anchor features, phylogenetic and geographic neighbours, vote summaries, contrastive nearest-neighbour evidence, global anchor-feature clues, and final output instructions.

Metadata. We include the language’s name, Glottocode, ISO 639-3 code, family lineage (written as family > parent when both levels are distinct, as a single level otherwise, or as Isolate), macro-area(s) from Glottolog, and the name of the typological feature f to predict.

Anchor features. To identify anchor features relevant to a target typological feature, we precompute a pairwise feature-correlation matrix from $\mathbf { M } _ { \mathrm { t r a i n } }$ using the Phi coeficient, following Ng et al. (2025):

$$
\phi ( f _ { i } , f _ { j } ) = \frac { n _ { 1 1 } n _ { 0 0 } - n _ { 1 0 } n _ { 0 1 } } { \sqrt { ( n _ { 1 } . n _ { 0 } . n . _ { 1 } n . _ { 0 } ) } } ,
$$

where $n _ { a b }$ denotes the number of languages with feature values a for $f _ { i }$ and b for $f _ { j }$

For a query language ℓ, we include up to 5 observed anchor feature values. If fewer than 3 of the top-10 anchors are observed for ℓ, the candidate pool is expanded to the top-15 correlated features before selecting up to 5 observed values.

We additionally compute, for each observed anchor feature a with value $\nu _ { a } .$ , the conditional support counts from $\mathbf { M } _ { \mathrm { t r a i n } }$ , defined as the number of languages with f=1 and with f=0 given a=v<sub>a</sub> respectively. An aggregate tally then reports how many anchor features generally favour value 1, value 0, or are tied.

Phylogenetic neighbours. We retrieve 5 phylogenetic neighbours (i.e. languages with the exact same, or similar, family lineage) for ℓ. We first build a candidate pool by breadth-first search over the Glottolog family tree starting from ℓ, up to a limit of 400 languages. For language isolates, this pool is padded with arbitrary languages in a fixed order up to 400.

From this pool, neighbours are selected by a greedy procedure maximising coverage over the anchor and target features: at each step the candidate is chosen that covers the most anchor or target features not yet observed among selected neighbours, breaking ties by whether f is itself observed, typological similarity to ℓ, and proximity (i.e. shortest-path distance to ℓ) rank. The nearest candidate in the full pool supporting f = 1 is included by default beforehand, ensuring that contrastive evidence is always present.

For each selected neighbour, the prompt lists its observed value for f, followed by up to 3 observed anchor-feature values, together with its proximity rank in the full candidate pool. The block also includes:

• A vote summary: counts of selected neighbours with f=1, f=0, and f unobserved.

• The closest language in the full candidate pool (by genealogical proximity rank) supporting f = 1, and the closest supporting f =0.

Geographic neighbours. Geographic neighbours are constructed identically, first constructing a candidate pool by ranking languages on Haversine distance from ℓ’s Glottolog coordinates. Neighbour headers in the prompt report Haversine distance in km (rather than a proximity rank), as do the entries for the closest-supporting neighbours. The same greedy selection applies to geographic neighbours.

Task and output. The prompt asks the model to predict f ∈ {0 1} and appends four reasoning guidance instructions: compare support for each value; weigh evidence holistically; note that a smaller number of closer neighbours may outweigh a larger but weaker group; and note that a minority value may be predicted if better supported. Output is constrained to one minified JSON object with keys rationale (at most 2 sentences), value ("0" or "1"), and confidence (low|medium|high). Three example outputs are appended in-prompt.

## A.3 Full Prompt Template

Target language:   
Name: {lang\_name}   
Glottocode: {glottocode}   
- ISO639-3: {iso}   
- Family lineage: {lineage}   
- Macro-area: {macroarea}   
Typological feature to predict:   
{feature}   
[Optional: anchor features]   
The following are {lang\_name}'s observed   
typological features that correlate with   
the target feature:   
{anchor\_feature\_1}: {0|1}   
{anchor\_feature\_2}: {0|1}   
[Optional: phylogenetic evidence]   
The following languages are {lang\_name}'s   
phylogenetic neighbours (observations for   
anchor features and the target feature,   
when available, are listed):   
1) {neighbor\_name} (proximity rank={rank}):   
- {feature\_or\_anchor}: {0|1}   
2)   
[Optional: geographic evidence]   
The following languages are {lang\_name}'s   
geographic neighbours (observations for   
anchor features and the targetfeature,   
when available, are listed):   
1) {neighbor\_name} (km={distance\_km}):   
- {feature\_or\_anchor}: {0|1}   
2) ...   
[Optional: vote summaries]   
Phylogenetic neighbour votes:   
- yes: {n\_yes}   
- no: {n\_no}   
- missing: {n\_missing}   
Geographic neighbour votes:   
- yes: {n\_yes}   
- no: {n\_no}   
- missing: {n\_missing}   
[Optional: contrastive evidence]   
The closest neighbours (by phylogeny and

```jsonl
geography) supporting each possible value:
- Closest phylogenetic neighbour supporting
value 1:
{label or none observed}
- Closest phylogenetic neighbour supporting
value 0:
{label or none observed}
- Closest geographic neighbour supporting
value 1:
{label or none observed}
- Closest geographic neighbour supporting
value 0:
{label or none observed}
[Optional: global anchor-feature clue
summary]
Across the knowledge base, when languages
exhibit the same anchor feature values:
1) {clue_feature}={0|1} ->
{yes_count} languages support {feature}=1
/{no_count} languages support {feature}=0
...
Overall:
{n_support_1} features support value 1 /
{n_support_0} features support value 0 /
{n_tied} features are tied
Task:
Predict the missing value for
feature {feature} (allowed values: 0 or 1).
Reasoning guidance:
Compare support for value 0 versus value 1
- Weigh all evidence holistically.
- A smaller number of closer or more
relevant neighbours may outweigh a larger
but weaker group.
- A minority value may be predicted if
better supported by the overall evidence.
Output:
Return exactly one valid minified JSON
object on one line with keys:
- rationale
- value
- confidence
Constraints:
- rationale: at most 2 sentences
- value: "0" or "1"
- confidence: "low", "medium", or "high"
Example outputs:
{"rationale":"While the majority of
neighbours support 0, the closest
phylogenetic neighbours support 1.",
"value":"1","confidence":"medium"}
{"rationale":"Observed anchor features
and phylogenetic evidence strongly
support value 1.","value":"1",
"confidence":"high"}
{"rationale":"Evidence is mixed, but anchor
statistics slightly favor value 0.",
"value":"0","confidence":"low"}
```

## B Test Set Composition

Table 3 provides additional statistics on the linguistic composition of the 1,200 test pairs. In addition to the balanced sampling across resource levels and feature types described in Section 3, the test set covers 171 distinct language families and a broad range of geographic macroareas.

<table><tr><td>Category</td><td>Coverage</td></tr><tr><td>Resource levels</td><td>400 pairs each for LRL / MRL / HRL</td></tr><tr><td>Feature types</td><td>300 pairs each for S / P / INV / M 171</td></tr><tr><td>Distinct families Largest families</td><td>Austronesian 15.6%, Atlantic-Congo</td></tr><tr><td>Macroareas</td><td>15.0%, Indo-European 9.1% Africa 26.1%, Eurasia 23.0%, Papune-</td></tr><tr><td></td><td>sia 22.4%, North America 10.6%, South America 10.2%, Australia 4.0%</td></tr></table>

Table 3: Composition of the 1,200-pair test set. Family and macroarea percentages denote the proportion of test pairs belonging to each group.

## C Evaluation Metrics for Imbalanced Data

Typological features can be highly imbalanced, with some features predominantly taking value 0 or 1. Our main experiments therefore report macro-F1 rather than accuracy and include a majority baseline. The majority baseline achieves a macro-F1 of 0.653, compared with 0.909 for the best LLM configuration, suggesting that the observed gains cannot be explained by majority-label prediction alone.

To further evaluate performance under label imbalance, we additionally report sensitivity, specificity, and balanced accuracy. Sensitivity measures recall on positive labels, specificity measures recall on negative labels, and balanced accuracy averages the two:

$$
{ \mathrm { B a l A c c } } = { \frac { { \mathrm { S e n s i t i v i t y } } + { \mathrm { S p e c i f i c i t y } } } { 2 } } .
$$

The imbalance-aware metrics preserve the main trend observed with macro-F1. Metadata-only and anchor-only prompting show substantial asymmetry between positive- and negative-label performance, particularly for Llama-3.1-70B. Performance becomes considerably more balanced once neighbour evidence is introduced.

Among the non-LLM baselines, Random Forest and SoftImpute achieve balanced accuracies of 0.876 and 0.873, respectively. In comparison,

<table><tr><td>Method</td><td>Inputs</td><td>Sens. Spec. Bal.</td><td></td><td>Acc.</td></tr><tr><td>Random</td><td></td><td>.600</td><td>.785</td><td>.692</td></tr><tr><td>Majority</td><td></td><td>.617</td><td>.885</td><td>.751</td></tr><tr><td>kNN (cosine) (k = 3)</td><td>(A)</td><td>.786</td><td>.898</td><td>.842</td></tr><tr><td>SoftImpute</td><td>(A)</td><td>.823</td><td>.924</td><td>.873</td></tr><tr><td>kNN (phylogenetic)</td><td> $( k \ = \ ( \mathbf { A } ) \substack { + } ( \mathbf { B } )$ </td><td>.775</td><td>.882</td><td>.828</td></tr><tr><td>6) kNN (geographic) (k = 6)</td><td> $( \mathrm { A } ) { + } ( \mathrm { C } )$ </td><td>.746</td><td>.876</td><td>.811</td></tr><tr><td>Random Forest</td><td> $\mathbf { \left( A \right) + } \mathbf { \left( B \right) + } \mathbf { \left( C \right) }$ </td><td>.792</td><td>.961</td><td>.876</td></tr><tr><td colspan="5">Llama-3.1-70B in-context prompting Base, zero-shot</td></tr><tr><td>(metadata only)</td><td></td><td>.792</td><td>.322</td><td>.557</td></tr><tr><td>+ anchor features</td><td>(A)</td><td>.907</td><td>.311</td><td>.609</td></tr><tr><td>+ anchor + phylogenetic</td><td>(A)+(B)</td><td>.955</td><td>.795</td><td>.875</td></tr><tr><td>+ anchor + geographic</td><td> $( \mathrm { A } ) { + } ( \mathrm { C } )$ </td><td>.839</td><td>.878</td><td>.859</td></tr><tr><td>+ all</td><td> $\mathbf { \eta } ( \mathbf { A } ) \mathbf { + } ( \mathbf { B } ) \mathbf { + } ( \mathbf { C } )$ </td><td>.961</td><td>.936</td><td>.948</td></tr><tr><td colspan="5">Gemma-4-31B in-context prompting Base, zero-shot</td></tr><tr><td>(metadata only)</td><td></td><td>.521</td><td>.799</td><td>.660</td></tr><tr><td>+ anchor features</td><td>(A)</td><td>.589</td><td>.872</td><td>.730</td></tr><tr><td>+ anchor + phylogenetic (A)+(B)</td><td></td><td>.921</td><td>.880</td><td>.901</td></tr><tr><td>+ anchor + geographic</td><td> $( \mathrm { A } ) { + } ( \mathrm { C } )$ </td><td>.986</td><td>.907</td><td>.946</td></tr><tr><td>+ all</td><td> $\mathbf { \left( A \right) + } \mathbf { \left( B \right) + } \mathbf { \left( C \right) }$ </td><td>.980</td><td>.879</td><td>.930</td></tr></table>

Table 4: Sensitivity, specificity, and balanced accuracy for the feature-prediction methods. Inputs follow Table 1: (A)=typological matrix, (B)=phylogenetic neighbours, and (C)=geographic neighbours. Cell shading is column-wise (darker = higher performance); percolumn best is in bold.

Llama-3.1-70B with all evidence achieves the highest balanced accuracy of 0.948, with sensitivity 0.961 and specificity 0.936. Gemma-4-31B with anchor and geographic evidence performs similarly, reaching balanced accuracy 0.946 with sensitivity 0.986 and specificity 0.907. These results indicate that the strongest evidence-grounded LLM configurations perform well on both labels rather than obtaining their gains primarily from dominant feature values.

## D Comparison Against a Plurality Voting Heuristic

A reasonable concern about the performance of LLMs is whether they are applying any reasoning beyond the heuristic of simply following the plurality vote from neighbours or anchor features. The random forest baseline partly addresses this, as it is trained over the same evidence the LLM sees, and would capture such a rule if one suficed for the task; yet the random forest model reaches only 0.731 macro F1, falling far short of Llama-3.1-70B’s 0.909.

To more rigorously test how far the support counts alone determine the gold value, we evaluate plurality voting over each evidence block on its own, predicting the value supported by the majority of observed phylogenetic neighbours, geographic neighbours, or anchor features respectively. Where there is a tie, a value of 1 is predicted. We additionally test these methods on a non-trivial subset of test examples, comprising 10.2% of the test set, where phylogenetic and geographic plurality votes disagree.

<table><tr><td>Method</td><td>Full</td><td>Non-trivial</td></tr><tr><td>Llama-3.1-70B</td><td>.909</td><td>.893</td></tr><tr><td>Phylogenetic neighbours, plurality</td><td>.770</td><td>.538</td></tr><tr><td>Geographic neighbours, plurality</td><td>.759</td><td>.459</td></tr><tr><td>Anchor features, plurality</td><td>.735</td><td>.518</td></tr></table>

Table 5: Macro F1 of Llama-3.1-70B under the allinputs configuration against plurality voting over individual evidence blocks, both on the full test set and on the non-trivial subset.

Table 5 shows the performance of plurality voting compared to Llama-3.1-70B on both the full test set and the non-trivial subset. We observe that under all evidence blocks, simply predicting based on the plurality vote yields a performance substantially lower than the LLM. This gap is particularly stark on the non-trivial subset, where plurality voting underperforms even further, whereas LLM prediction retains a respectable macro F1 of 0.893. This therefore shows that the LLMs’ aggregation of evidence is not reducible to simply counting votes, and in fact LLMs hold an advantage where the evidence conflicts.

## E Within-Family Model Size Comparison

<table><tr><td>Config</td><td>Llama 8B</td><td>Llama 70B</td></tr><tr><td>Base (metadata only)</td><td>.147</td><td>.465</td></tr><tr><td>+ anchor features</td><td>.425</td><td>.512</td></tr><tr><td>+ anchor + phylogenetic</td><td>.480</td><td>.782</td></tr><tr><td>+ anchor + geographic</td><td>.661</td><td>.788</td></tr><tr><td>+ all</td><td>.535</td><td>.909</td></tr></table>

Table 6: Overall macro F1 for Llama-3.1 at two model sizes. Cell shading is column-wise (darker = higher F1).

To study the efect of model size on imputation performance, Table 6 shows the macro F1 scores for Llama-3.1 at 8B and 70B sizes. Across configurations, we find that performance increases as model size increases, with a particularly substantial increase at the base configuration (F1 0.147 vs 0.465). This is consistent with our expectation that larger models have more intrinsic knowledge (limited as it is) and better reasoning abilities. Interestingly, the 8B model exhibits a similar trend to Gemma-4-31B, with performance greater under geographic evidence than under the configuration containing all inputs. GPT-5.5 exhibits the same pattern in Table 1. This suggests that the efect is not explained by simply the model size. It is the case that heterogeneous evidence can sometimes reduce performance.

## F Annotation Details and Protocol

Five annotators were recruited for the annotation task described in Section 5. Annotators were undergraduate students with backgrounds in computer science or linguistics, from the authors’ institution. They were informed that their annotation would be used for a research context (along with contextual information about what the task would entail), and that they would participate voluntarily without monetary compensation, to which they provided their explicit consent.

For each of the 180 (ℓ f) cases collected according to the rule stated in Section 5, the annotation task proceeded in two stages.

1. Stage 1: Given the same typological facts given to the LLM (namely, all four context blocks described in Section 3.2), the annotator was asked to decide whether the target language has the typological feature stated or not (1/0). In addition, they were asked to state their confidence level on the same scale as the LLM (High/Medium/Low). They were also told to mark their absolute confidence in their prediction, rather than their relative confidence between cases.

2. Stage 2: Given the exact same set of cases as in Stage 1, the annotator is now presented with what Llama-3.1-70B predicted, as well as their rationale. Their job was to determine whether the rationale of given by the LLM is consistent with the information provided to them, and the prediction they made (Yes/No). Note that Llama-3.1-70b was chosen, due to its superior performance in Table 1.

Each annotator completed the task independently of each other. These cases were presented to annotators in a well-designed web app for ease of annotation. Within a particular stage, annotators completed each case sequentially, with the option to go back to a previous case if desired. However, annotators were instructed to only start Stage 2 once they completed Stage 1. This was to avoid their decision making in Stage 1 from being influenced by the LLM’s reasoning process (which Stage 2 exposes).

To aid in their decision-making, we provided two aid sheets: 1) an IPA chart<sup>4</sup>, and 2) a two-page reference sheet on linguistics concepts containing information on phonetics, phonology, morphology, and syntax, and how to read the IPA chart. Annotators were strictly forbidden from using LLMs or AI tools for this task, or on any material related to this task.

Annotators were allowed as much time as desired to complete this task, but we informed them that we expect each case per stage, to take roughly three minutes. Annotators shared their results with the investigators independently of each other.

## G Qualitative Analysis of LLM Rationales

In this section, we provide a qualitative analysis of Llama-3.1-70B rationales to understand what evidence the model appears to use, when this evidence use leads to correct predictions, and when it produces incomplete or misleading explanations. In particular, from the 180 cases used in the annotation study, we construct two subsets of rationales to analyse.

The first subset consists of all cases where a majority of annotators (i.e. three) agreed that the LLM provided a flawed rationale. There were 30 such majority-contradiction cases. On this restricted subset, the LLM still achieved accuracy 0.867 (F1 0.464), while the human plurality achieved accuracy 0.967 (F1 0.492). The low F1 values mainly reflect the strong class imbalance in this subset, where almost all gold labels are 0. Only four cases combined a majority-marked inconsistent rationale with an incorrect LLM prediction. We call this the contradiction subset.

To match this subset, we also randomly sampled another 30 cases where no annotator marked it as having a reasoning error. On this subset, the LLM predicted 29 of 30 cases correctly. We call this the non-contradiction subset.

Of note, in none of the cases studied, do the rationales mention the target language’s metadata, or any external fact about the language not provided by the prompt. Instead, the rationales overwhelmingly cite evidence explicitly present in the prompt (or no evidence at all). Thus, the model’s successful explanations are best understood as promptgrounded evidence summaries, not as demonstrations of innate typological knowledge.

Pattern 1: Cross-source agreement produces the clearest successes. The clearest successful rationales occur when phylogenetic neighbours, geographic neighbours, and anchor-feature statistics all suggest the same label. For example, dalo1238 (Daloa Bété), a medium-resource language with target feature P\_BILABIALS, had unanimous neighbour support for value 1: 5/5 phylogenetic neighbours and 5/5 geographic neighbours supported 1, and both available anchor clues also supported 1. The LLM predicted the gold value 1 with high confidence and gave the rationale: “The observed anchorfeatures are known to support 1 when they are 0. Furthermore, both phylogenetic and geographic neighbours align strongly with value 1, with all observed neighbours supporting 1.” This is a faithful summary of the prompt evidence.

With that said, the contradiction subset shows that even when all the evidence in the prompt is aligned, the model can still fail to report it. For shan1277 (Shan), a low-resource language with target feature P\_LATERAL\_OBSTRUENTS, both phylogenetic and geographic neighbours gave majority support for value 0, and the anchor clue also supported value 0. The model nevertheless predicted 1 and gave only the rationale “Insuficient direct evidence”. Even if the evidence was given to the model, it failed to interpret it, and provided an incorrect prediction.

Pattern 2: Neighbour evidence is usually primary, while anchor evidence is usually secondary. Another observation is that the LLM usually follows neighbour evidence, especially when phylogenetic and geographic neighbours agree. A successful example from the non-contradiction set is awad1243 (Awadhi), a high-resource language with target feature M\_PRESENT\_MARK. The single anchor clue supported value 0, but neighbour evidence strongly supported value 1: four of five phylogenetic neighbours and all five geographic neighbours supported 1. The LLM predicted the gold value 1 with high confidence, stating that “the majority of phylogenetic and geographic neighbours support value 1” and that the closest neighbours also supported 1, “outweighing the general trend” from languages with the same anchor value. Hence it prioritised local neighbour evidence over a global anchor trend, rather than a prediction based on independent knowledge about Awadhi’s family or typological profile.

Even if the model appears to favour neighbour evidence, it does not always make it clear how it weighs difering neighbour information. Indeed, from the contradiction subset, we may consider the case of dura1244 (Dura). This is a medium resource language with target feature S\_POLARQ\_MARK\_SECOND. The majority of neighbours supported 0, but the closest phylogenetic and geographic neighbours both supported 1. The model predicted correctly, but its rationale only began to describe the conflict: “Although the majority of both phylogenetic and geographic neighbours support 0, the closest neighbour in both categories, Bujhyal, supports 1”. It did not give an explanation of how it weighed the neighbour evidence given.

A similar, but distinct observation is that for low-resource cases, the LLM had a reliance on using neighbour evidence, according to its rationale. This is generally because these cases often provided fewer target-language anchor features, which when combined with the fact that the LLM naturally seems to disfavour anchor information, resulted in this reliance.

Pattern 3: Proximity and contrastive evidence are useful, but can be incorrectly interpreted. Recall that the prompt asks the model to consider whether a smaller number of closer neighbours may outweigh a larger but weaker group. From the noncontradiction subset, we see that the LLM can use this instruction sensibly. For example, chak1270 (Chak), a medium-resource language with target feature M\_COME\_VERB\_SUPPLETION, had four of five phylogenetic neighbours and four of five geographic neighbours supporting 0. The closest geographic neighbour supporting 0 was Anu-Hkongso at 48.0 km, while the closest geographic neighbour supporting 1 was Zeme Naga at 450.7 km. The LLM predicted the gold value 0 with medium confidence and explained: “The majority of both phylogenetic and geographic neighbours support value 0, and the closest geographic neighbour supporting value 0 is significantly closer than the closest geographic neighbour supporting value 1.”

The contradiction cases show that proximity evidence can be misinterpreted or even omitted from reasoning though. In anut1237 (Anuta), a low-resource language with target feature M\_UNPREDICTABLE\_GENDER\_CLASS, the prompt gave no target-language anchor features, so the model had to rely on neighbour evidence. The rationale stated that “the majority of phylogenetic and geographic neighbours support value 0” and noted that the closest phylogenetic neighbour supporting 0 was more proximal than the closest phylogenetic neighbour supporting 1. This correctly described part of the evidence, and the model predicted the gold value. However, the rationale omitted the countervailing geographic information: the closest geographic neighbour supporting 1 was closer than the closest geographic neighbour supporting 0. In nata1254 (Northern Amis), a medium-resource language with target feature M\_PROD\_PLURAL\_MARK, the model similarly cited closest-neighbour support for value 0, saying that “the closestphylogenetic neighbour Sirayaic and geographic neighbour Atayal support value 0”, but then added that “the observed anchorfeatures do notprovide strong evidencefor value”, even though the prompt did not provide target-language anchor features. These are not necessarily prediction failures, but they are rationale-faithfulness failures: the model gives a simplified proximity account that omits or misstates part of the prompt evidence.

Proximity can also be overweighted. The one incorrect prediction in the no-contradiction subset was garo1247 (Garo), a high-resource language with target feature P\_IMPLOSIVES. The gold label and human majority were 0. Both neighbour blocks supported 0 by a 4-to-1 majority, and both anchor clues supported 0. The LLM nevertheless predicted 1 with medium confidence. Its rationale began: “Although the majority of phylogenetic and geographic neighbours support 0, the closest phylogenetic and geographic neighbours supporting value 1 are relatively closer than those supporting value 0.” This rationale overstates the importance of proximity. Indeed, the positive phylogenetic supporter was only slightly closer than the nearest negative supporter, while the nearest geographic supporter of 0, Assamese at 121.2 km, was much closer than the nearest geographic supporter of 1, Nepali Kurux at 473.0 km. Thus, proximity is useful when it reinforces broader evidence, but it can become misleading if the evidence is not given appropriate weight.

Pattern 4: Generic rationales often mark uncertainty. One fallback phrase that the LLM provided in both non-contradiction and contradiction subsets was: “Insuficient direct evidence”. In the no-contradiction subset, four of the 30 rationales used exactly this phrase; all four predictions were correct and low-confidence. Hence, it seems to be an indicator for a lack of confidence on the model’s part.

In the contradiction subset, the phrase occurred in 8 of the 30 majority-contradiction cases, and all four cases that combined a majority-marked inconsistent rationale with an incorrect LLM prediction used this fallback rationale. One can recall a case with Shan, where it ignored aligned support throughout the prompt for 0, accompanied by rationale. Another case is ewee1241 (Ewe), a high-resource language with target feature M\_PROD\_AUGMENTATIVE\_NOUN, the same rationale was given in a conflicted case: phylogenetic neighbours and anchor clues supported 0, while geographic neighbours strongly supported 1. The gold label was 1, but both the LLM and the human majority predicted 0. Here the problem was that the rationale failed to describe the conflict that made the case dificult.

Pattern 5: Phonological and inventory cases are hard with conflicting evidence. The model can handle phonological and inventory features when evidence is clear. An example is dalo1238 (Daloa Bété) for P\_BILABIALS. However, the majoritycontradiction subset suggests that fine-grained segmental features are a common source of actual errors when evidence conflicts. Of the four cases where a majority-marked inconsistent rationale coincided with an incorrect prediction, three involved phonological or inventory features. In nort2921 (North Wahgi), a low-resource language with target feature P\_FRICATIVES, the neighbour majority supported value 0, but the anchor clue supported 1; the model gave the generic rationale “Insuficient direct evidence” and predicted 1, while both the gold label and human majority were 0. The aforementioned Shan and Garo cases are similar. These cases suggest that the model’s evidence weighting for these kinds of cases is brittle, when the evidence given can be dificult to interpret.

The takeaway. While the rationales given by the LLM is a step towards explainable typological prediction, they are best understood as a method to interpret evidence given to it. From the rationales analysed, we do not have any reason to say that the LLM uses any innate typological knowledge, or knowledge about the language outside of the prompt (despite given metadata) to reason in this task. Furthermore, the LLM appears to have innate biases in how it prioritises certain information over others. Nonetheless, even if the rationale the LLM provides sometimes does not provide a full account of its behaviour, it does ofer (often strong) clues towards its prediction.

## H Baseline Architectures

## H.1 k-Nearest Neighbours

We chose $k = 3$ for kNN-cosine and k = 6 for kNN-phylogenetic and geographic. This was determined following a search over k, evaluating each model on macro F1 on the validation split. We then choose the value of k where performance begins to plateau or decrease. Where a tie occurs, a value of 1 is predicted.

## H.2 Random Forest Classifier

Following Amirzadeh et al. (2025), we train a random forest classifier for each typological feature $f .$ To best match the inputs visible to the LLM, we train the classifier on the following features given some language ℓ (totalling 42 features):

Metadata

• P(f = 1| top-level family)

$P ( f = 1 | \mathrm { m a c r o a r e a } )$

Anchor features For each of the top 5 correlated anchor features a which are observed for ℓ:

$\ell { \boldsymbol { \mathrm { s } } }$ value of feature $a , \nu _ { a } = \mathbf { M } _ { \mathrm { t r a i n } } [ \ell , a ]$

• Global anchor feature clue, $P ( f = 1 | a = \nu _ { a } )$

Phylogenetic neighbours For each of the 5 nearest phylogenetic neighbours $\ell _ { p } \mathbf { . }$

$\ell _ { p } \mathrm { { ' s } }$ value of $f , \mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { p } , f ]$

• Whether f is missing in $\ell _ { p }$

Supplemented by the following aggregate features:

• Number of phylogenetic neighbours voting yes (i.e. where $\mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { p } , f ] = 1 \ \wedge$ $\mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { p } , f ] \neq \bot )$

• Number of phylogenetic neighbours voting no (i.e. where $\mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { p } , f ] = 0 \land \mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { p } , f ] \neq$ ⊥)

• Number of phylogenetic neighbours with missing f

• The rank of the closest phylogenetic neighbour voting yes

• The rank of the closest phylogenetic neighbour voting no

Geographic neighbours For each of the 5 nearest geographic neighbours $\ell _ { g } \colon$

$\ell _ { g } \mathrm { ^ { * } s }$ value of $f , \mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { g } , f ]$

• Whether f is missing in $\ell _ { g }$

Supplemented by the following aggregate features:

• Number of geographic neighbours voting yes (i.e. where $\mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { g } , f ] = 1 \wedge \mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { g } , f ] \neq$ ⊥)

• Number of geographic neighbours voting no (i.e. where $\mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { g } , f ] = 0 \land \mathbf { M } _ { \mathrm { t r a i n } } [ \ell _ { g } , f ] \neq$ ⊥)

• Number of geographic neighbours with missing f

• The rank of the closest geographic neighbour voting yes

• The rank of the closest geographic neighbour voting no

Following a grid search over the number of estimators n\_estimators and maximum tree depth max\_depth, evaluating macro F1 performance on the validation split, we choose n\_estimators= 500 and max\_depth= 8, with the remaining hyperparameters set to scikit-learn (Pedregosa et al., 2011) defaults.

## I Computing Infrastructure

Computational experiments on all three opensource models (Llama-3.1-8B, Llama-3.1-70B, Gemma-4-31B) across all five configurations were run on a single Nvidia H100 GPU, totalling 25 compute hours. GPT-5.5 experiments were conducted through inference on Azure OpenAI. All other experiments were run on CPU.

<table><tr><td>Artifact</td><td>License</td></tr><tr><td>Packages</td><td></td></tr><tr><td>URIEL+ (Khan et al., 2025)</td><td>CC BY-SA 4.0</td></tr><tr><td>scikit-learn (v1.8) (Pedregosa et al., 2011)</td><td>BSD-3-Clause</td></tr><tr><td>Transformers (v5.1) (Wolf et al., 2020)</td><td>Apache 2.0</td></tr><tr><td>Datasets</td><td></td></tr><tr><td>Glottolog (v5.2) (Hammarström et al., 2025)</td><td>CC BY 4.0</td></tr><tr><td>Models</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct (Grattafiori et al., 2024) Llama-3.1-70B-Instruct (Grattafiori et al., 2024)</td><td>Llama 3.1 Community License Agreement</td></tr><tr><td>Gemma-4-31B-IT (DeepMind, 2026)</td><td>Llama 3.1 Community License Agreement</td></tr><tr><td>GPT-5.5 (OpenAI, 2026)</td><td>Apache 2.0 Azure OpenAI (Microsoft Product Terms)</td></tr></table>

Table 7: Artifacts used in this study, and their licenses.

## J Use of Generative AI

We used generative AI only in a limited capacity. Namely, generating auto-code completions (which were verified) and checking the grammar and structure of our text.

## K Licenses for Artifacts

We present the licenses for this study in Table 7. Our usage of these artifacts is in line with the licenses these artifacts were released under.