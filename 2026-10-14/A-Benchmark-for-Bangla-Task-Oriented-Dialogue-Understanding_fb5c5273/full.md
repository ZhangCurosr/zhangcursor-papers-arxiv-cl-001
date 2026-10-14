# A Benchmark for Bangla Task-Oriented Dialogue Understanding

Afrina Alam Lamia East West University Dhaka, Bangladesh

Sumiaya Sharif<sup>\*</sup> East West University Dhaka, Bangladesh

Prottoy Modak<sup>\*</sup> East West University Dhaka, Bangladesh

Ohiduzzaman Rayhan East West University Dhaka, Bangladesh

Nafis Sadeq East West University Dhaka, Bangladesh

## Abstract

Task-oriented dialogue understanding remains underexplored for Bangla, particularly in multilingual settings involving English and romanized Bangla (Banglish). We introduce BanglaTOD, a parallel tri-lingual benchmark comprising 209 multi-turn dialogues across nine Bangladesh-specific service domains and 64 intents. Each conversational turn is aligned across Bangla, English, and Banglish, yielding 4,689 language-specific turn records annotated with intents, token-level slots, state updates, and accumulated dialogue states. We establish context-aware baselines for joint intent classification and slot filling using mBERT, BERT-base-uncased, and BanglaBERT. Crosslanguage analysis reveals substantial performance differences across models and linguistic representations, while removing dialogue history reduces intent accuracy by 19.0–25.7 percentage points. BanglaTOD provides a foundation for multilingual dialogue understanding and future dialogue state tracking research in Bangladesh-specific conversational applications. The dataset is publicly available at Hugging Face.

## 1 Introduction

Task-oriented dialogue systems rely on identifying user intents, extracting relevant slot values, and maintaining conversational context across turns (Budzianowski et al., 2018; Chen et al., 2019; Zhu et al., 2020; Rastogi et al., 2020). Despite advances in multilingual natural language understanding (Schuster et al., 2019; Ding et al., 2022), resources for Bangla task-oriented dialogue remain limited (Khan et al., 2021; Karim and Uzuner, 2025; Sadeq et al., 2020). This limitation is particularly relevant in Bangladesh, where digital communication frequently involves Bangla, English, and Banglish (Alam et al., 2025; Hossain et al., 2026; Bari et al., 2026). Variations in language, script, and spelling introduce additional challenges for intent recognition, slot extraction, and dialogue state tracking.

Existing multilingual resources, including MultiATIS++ (Xu et al., 2020), MTOP (Li et al., 2021), and GlobalWoZ (Ding et al., 2022), have advanced cross-lingual task-oriented NLU but provide limited coverage of Bangladesh-specific conversational settings. Bangla NLP resources have primarily addressed language modeling, general language understanding, and generation (Bhattacharjee et al., 2022, 2023). More recently, Bangla intent classification and slot filling have been investigated using translated task-oriented datasets (Karim and Uzuner, 2025). However, resources containing multi-turn conversations with parallel Bangla, English, and Banglish utterances and structured dialogue-state annotations remain scarce. Translated corpora also do not contain realistic ontology for Bangladesh.

To address this gap, we introduce BanglaTOD, a parallel tri-lingual benchmark for task-oriented dialogue understanding in Bangladesh. The dataset comprises 209 manually constructed multiturn dialogues spanning nine service domains and 64 intents. Each conversational turn is aligned across Bangla, English, and Banglish, producing 4,689 language-specific turn records. The annotations include intents, token-level slots, turn-level state updates, and accumulated dialogue states, supporting both utterance-level understanding and future dialogue state tracking research. We additionally establish context-aware baselines using three pretrained encoders and examine their performance across linguistic representations.

Our main contributions are:

• A turn-aligned multilingual dialogue resource: We introduce BanglaTOD, comprising 209 dialogues across nine Bangladeshi service domains and 64 intents, with parallel

Bangla, English, and Banglish utterances annotated for intent, slots, and dialogue states.

• Context-aware multilingual benchmarks: We evaluate mBERT, BERT-base-uncased, and BanglaBERT for joint intent classification and slot filling across all three linguistic representations.

• Analysis of language and context: We examine cross-language performance differences, the contribution of dialogue history, and recurring intent and slot prediction errors, highlighting challenges for multilingual taskoriented dialogue understanding.

## 2 Dataset and Methodology

## 2.1 Dataset Coverage and Construction

BanglaTOD comprises 209 multi-turn dialogues across nine Bangladesh-specific service domains and 64 intents. Domains were selected for their relevance to everyday service interactions, informed by existing task-oriented dialogue resources. Four annotators constructed the dialogues through pairwise role-playing (user and system), to simulate realistic service interactions. Each intent represents a distinct user goal within its domain. All dialogues and annotations were manually reviewed to identify and correct errors. Table 1 summarizes the dataset coverage.

Dialogues were constructed as user–system interactions spanning 2–10 turns. They include follow-up requests that refer to information introduced in earlier turns, capturing contextual dependencies beyond isolated utterances.

<table><tr><td>Domain</td><td>Dialogues</td><td>Intents</td></tr><tr><td>Education</td><td>50</td><td>9</td></tr><tr><td>Government Service</td><td>32</td><td>19</td></tr><tr><td>Healthcare</td><td>25</td><td>5</td></tr><tr><td>Mobile Financial Serv.</td><td>25</td><td>5</td></tr><tr><td>E-commerce</td><td>25</td><td>10</td></tr><tr><td>Restaurant</td><td>16</td><td>5</td></tr><tr><td>Tourist</td><td>14</td><td>5</td></tr><tr><td>Travel</td><td>12</td><td>4</td></tr><tr><td>Hotel</td><td>10</td><td>2</td></tr><tr><td>Total</td><td>209</td><td>64</td></tr></table>

Table 1: Domain and intent coverage of the dataset.

## 2.2 Parallel Representations and Annotations

Each dialogue is represented in Bangla, English, and Banglish, with corresponding conversational turns aligned across all three representations. The shared dialogue structure enables controlled comparisons of equivalent conversational content across languages and scripts. Each dialogue contains intent labels and structured slot annotations. A slot annotation specifies its type, textual expression, canonical value, and token boundaries. To capture information accumulated across turns, state update records newly introduced or modified values, while dialogue state stores the accumulated state after each turn.

<table><tr><td>Characteristic</td><td>Value</td></tr><tr><td>Total dialogues</td><td>209</td></tr><tr><td>Languages</td><td>3</td></tr><tr><td>Domains</td><td>9</td></tr><tr><td>Intents</td><td>64</td></tr><tr><td>Turns per language Language-specific turn records</td><td>1,563</td></tr><tr><td>Average turns per dialogue</td><td>4,689 7.48</td></tr><tr><td>Minimum turns per dialogue</td><td></td></tr><tr><td>Maximum turns per dialogue</td><td>2</td></tr><tr><td>Unique slot types</td><td>10 348</td></tr></table>

Table 2: Overview of BanglaTOD.

<table><tr><td>Language</td><td>Tokens</td><td>Vocabulary</td><td>Avg. Tokens/Turn</td></tr><tr><td>Bangla</td><td>15,726</td><td>2,448</td><td>10.06</td></tr><tr><td>English</td><td>17,858</td><td>2,192</td><td>11.43</td></tr><tr><td>Banglish</td><td>15,412</td><td>2,661</td><td>9.86</td></tr></table>

Table 3: Lexical statistics for the language variants.

The dataset is distributed in JSON format. Each dialogue contains a unique identifier, domain and intent information, and an ordered sequence of turns. Each turn records its speaker and the three aligned annotated utterance representations. This structure supports intent classification, slot filling, and dialogue state tracking.

## 2.3 Dataset Statistics

BanglaTOD contains 1,563 conversational turns, represented as 4,689 language-specific turn records. Dialogues average 7.48 turns, and the dataset contains 348 unique slot types. Table 3 summarizes lexical differences across representations. English utterances have the highest average token count, while Banglish exhibits the largest vocabulary.

## 3 Experimental Setup

Data and Split. We partition the 209 dialogues into training (137), validation (36), and test (36) sets using dialogue-level split with intent-based stratification. All three parallel representations of each dialogue remain in the same partition to prevent cross-language leakage.

Models and Training. We fine-tune three encoders: bert-base-multilingual-cased (mBERT) (Devlin et al., 2019), BERTbase-uncased (BERT-base) (Devlin et al., 2019), and sagorsarker/bangla-bert-base (BanglaBERT) (Sarker, 2020). Each encoder is trained on pooled Bangla, English, and Banglish data and evaluated separately on each language. Following Chen et al. (2019), the joint architecture uses a two-layer intent classifier (768 256 64) and a linear BIO slot classifier. Domain predictions are derived from the predicted intents. Models are trained using AdamW with a weighted joint loss $( \mathcal { L } _ { i n t } + 1 . 5 \mathcal { L } _ { s l o t } )$ across three seeds (42, 43, 44). Additional training details are provided in Appendix A. Each target utterance is encoded with up to three preceding turns from the same dialogue and language, separated by [SEP] tokens (Padmakumar et al., 2023).

Evaluation. We report intent accuracy (equivalent to micro-F1 for single-label classification), macro-F1, and weighted-F1. Slot filling is evaluated using entity-level precision, recall, and F1 based on exact span and type matches. We additionally report domain accuracy and joint exact match (EM), which requires both a correct intent and an exact match of the complete slot-label sequence. Results are averaged across three seeds.

## 4 Results and Discussion

## 4.1 Cross-Language Analysis

Table 4 shows substantial differences across the three language representations, despite their parallel conversational content. mBERT achieves the strongest overall performance and remains relatively consistent across languages, with intent accuracy ranging from 0.573 to 0.625. In contrast, BERT-base performs substantially better on English and Banglish than on Bangla, while BanglaBERT achieves its highest intent accuracy on Bangla (0.659) but transfers poorly to the other representations.

These patterns are consistent with differences in pretraining language and script coverage (Rust et al., 2021; Wu and Dredze, 2020). The parallel structure of BanglaTOD enables such comparisons while controlling for conversational content, revealing the challenges of supporting nativescript, English, and romanized language representations within a shared dialogue system.

## 4.2 Fine-Grained Intent and Slot Challenges

Across all backbones, domain classification is considerably more accurate than intent classification. For example, mBERT achieves 0.859 domain accuracy but only 0.601 intent accuracy. Intent macro-F1 is also consistently lower than weighted-F1 (0.408 versus 0.623 for mBERT), suggesting difficulties with infrequent intent categories.

Error inspection further reveals confusion between semantically related intents, including cash\_out\_information and transaction\_support, and between tour\_cancellation and modify\_booking. Government-service intents, which constitute the largest intent inventory in the dataset, are particularly challenging. These observations highlight the difficulty of distinguishing fine-grained user goals in a limited-data setting.

Slot filling presents an additional challenge. Even the strongest model achieves only 0.330 entity-level F1. Errors frequently involve omitted slots and incomplete multi-word spans, whereas short, regularly formatted values are recovered more reliably. Together, these results suggest that fine-grained intent recognition and structured slot extraction remain important challenges for lowresource task-oriented dialogue understanding.

## 4.3 Effect of Dialogue Context

To assess the importance of conversational history, we retrain each backbone without preceding turns. Table 5 summarizes the ablation. Removing context reduces intent accuracy by 19.0–25.7 percentage points across the three backbones. The corresponding reductions in token-level slot macro-F1 are smaller. This pattern suggests that preceding turns are particularly useful for resolving context-dependent user intentions, such as followup questions that omit previously established information. The consistent improvements across backbones demonstrate the importance of multiturn context and motivate evaluation beyond isolated utterances.

## 4.4 Intent Error Analysis

Table 6 illustrates variation in per-intent F1. While hotel\_booking\_cancellation is recognized reliably across backbones, several governmentservice intents perform poorly. The latter belong to the domain with the largest intent ining\_id and date\_of\_birth are identified correctly, although their support is small. Other labels, such as application\_status, are omitted entirely. Generic labels including query and service show low precision, indicating difficulty distinguishing broad slot categories. Multi-word expressions present an additional challenge. The continuation label I-requested\_information achieves only 0.277 recall for mBERT. Similar omissions occur for the other backbones: BanglaBERT and BERT-base predict O for 47 and 30 of 65 requested\_information tokens, respectively. These errors help explain the low entity-level F1, where recovering an incomplete span does not count as a correct entity.

<table><tr><td></td><td colspan="4">Intent / Domain</td><td colspan="3">Slot (entity-level)</td><td>Joint</td></tr><tr><td>Model / Language</td><td>Dom. Acc</td><td>Micro-F1</td><td>Macro-F1</td><td>Wtd-F1</td><td>Micro-F1</td><td>Macro-F1</td><td>Wtd-F1</td><td>EM</td></tr><tr><td>mBERT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Overall</td><td> $0 . 8 5 9 _ { \pm . 0 0 7 }$ </td><td> $0 . 6 0 1 { \scriptstyle \pm . 0 1 2 }$ </td><td> $0 . 4 0 8 { \scriptstyle \pm . 0 1 5 }$ </td><td> $0 . 6 2 3 { \scriptstyle \pm . 0 1 2 }$ </td><td> $0 . 3 3 0 { \scriptstyle \pm . 0 0 3 }$ </td><td> $0 . 1 9 4 _ { \pm . 0 0 3 }$ </td><td> $0 . 3 0 2 { \scriptstyle \pm . 0 0 6 }$ </td><td> $0 . 2 1 9 _ { \pm . 0 1 0 }$ </td></tr><tr><td>Bangla</td><td> $0 . 8 6 9 _ { \pm . 0 1 8 }$ </td><td> $0 . 6 0 5 { \scriptstyle \pm . 0 1 9 }$ </td><td> $0 . 4 5 3 _ { \pm . 0 0 7 }$ </td><td> $0 . 6 1 3 { \scriptstyle \pm . 0 1 9 }$ </td><td> $0 . 3 2 3 _ { \pm . 0 1 1 }$ </td><td> $0 . 2 0 3 _ { \pm . 0 0 3 }$ </td><td> $0 . 2 8 5 _ { \pm . 0 1 5 }$ </td><td> $0 . 2 2 5 _ { \pm . 0 0 9 }$ </td></tr><tr><td>English</td><td> $0 . 8 5 9 _ { \pm . 0 0 6 }$ </td><td> $0 . 6 2 5 _ { \pm . 0 0 7 }$ </td><td> $0 . 4 8 2 _ { \pm . 0 0 9 }$ </td><td> $0 . 6 3 6 _ { \pm . 0 1 0 }$ </td><td> $0 . 3 4 5 _ { \pm . 0 0 6 }$ </td><td> $0 . 2 3 1 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 3 1 8 _ { \pm . 0 0 4 }$ </td><td> $0 . 2 3 7 _ { \pm . 0 0 6 }$ </td></tr><tr><td>Banglish</td><td> $0 . 8 4 9 _ { \pm . 0 0 7 }$ </td><td> $0 . 5 7 3 { \scriptstyle \pm . 0 1 3 }$ </td><td> $0 . 4 2 6 _ { \pm . 0 1 4 }$ </td><td> $0 . 5 9 2 _ { \pm . 0 0 8 }$ </td><td> $0 . 3 2 1 { \scriptstyle \pm . 0 0 5 }$ </td><td> $0 . 2 1 2 _ { \pm . 0 0 4 }$ </td><td> $0 . 3 0 1 { \scriptstyle \pm . 0 0 4 }$ </td><td> $0 . 1 9 5 _ { \pm . 0 1 8 }$ </td></tr><tr><td>BERT-base</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Overall</td><td> $0 . 8 2 2 _ { \pm . 0 1 1 }$ </td><td> $0 . 5 5 1 { \scriptstyle \pm . 0 3 4 }$ </td><td> $0 . 3 4 8 _ { \pm . 0 3 1 }$ </td><td> $0 . 5 6 5 _ { \pm . 0 3 7 }$ </td><td> $0 . 2 8 5 _ { \pm . 0 0 7 }$ </td><td> $0 . 1 8 2 _ { \pm . 0 0 3 }$ </td><td> $0 . 2 5 5 _ { \pm . 0 0 6 }$ </td><td> $0 . 1 8 9 _ { \pm . 0 0 1 }$ </td></tr><tr><td>Bangla</td><td> $0 . 7 5 8 { \scriptstyle \pm . 0 1 8 }$ </td><td> $0 . 4 3 5 _ { \pm . 0 4 1 }$ </td><td> $0 . 3 1 7 { \scriptstyle \pm . 0 3 8 }$ </td><td> $0 . 4 4 0 _ { \pm . 0 4 5 }$ </td><td> $0 . 2 1 5 _ { \pm . 0 1 4 }$ </td><td> $0 . 1 2 4 { \scriptstyle \pm . 0 1 3 }$ </td><td> $0 . 1 7 4 _ { \pm . 0 1 2 }$ </td><td> $0 . 1 4 8 _ { \pm . 0 1 2 }$ </td></tr><tr><td>English</td><td> $0 . 8 7 4 _ { \pm . 0 1 6 }$ </td><td> $0 . 6 2 0 { \scriptstyle \pm . 0 3 5 }$ </td><td> $0 . 4 5 8 _ { \pm . 0 2 9 }$ </td><td> $0 . 6 2 3 { \scriptstyle \pm . 0 4 3 }$ </td><td> $0 . 3 0 8 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 2 2 2 _ { \pm . 0 0 7 }$ </td><td> $0 . 2 8 3 _ { \pm . 0 0 7 }$ </td><td> $0 . 2 0 7 _ { \pm . 0 1 2 }$ </td></tr><tr><td>Banglish</td><td> $0 . 8 3 5 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 6 0 0 { \scriptstyle \pm . 0 2 6 }$ </td><td> $0 . 4 2 9 _ { \pm . 0 1 4 }$ </td><td> $0 . 6 0 5 _ { \pm . 0 2 9 }$ </td><td> $0 . 3 2 9 _ { \pm . 0 1 1 }$ </td><td> $0 . 2 3 1 { \scriptstyle \pm . 0 0 3 }$ </td><td> $0 . 2 9 5 _ { \pm . 0 0 8 }$ </td><td> $0 . 2 1 0 { \scriptstyle \pm . 0 0 4 }$ </td></tr><tr><td>BanglaBERT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Overall</td><td> $0 . 6 7 5 _ { \pm . 0 1 1 }$ </td><td> $0 . 3 8 0 { \scriptstyle \pm . 0 1 7 }$ </td><td> $0 . 2 3 0 { \scriptstyle \pm . 0 1 6 }$ </td><td> $0 . 3 9 3 _ { \pm . 0 2 3 }$ </td><td> $0 . 1 9 2 _ { \pm . 0 0 7 }$ </td><td> $0 . 0 9 9 { \scriptstyle \pm . 0 0 6 }$ </td><td> $0 . 1 6 5 _ { \pm . 0 0 5 }$ </td><td> $0 . 1 2 3 _ { \pm . 0 0 7 }$ </td></tr><tr><td>Bangla</td><td> $0 . 8 3 2 _ { \pm . 0 2 9 }$ </td><td> $0 . 6 5 9 { \scriptstyle \pm . 0 4 2 }$ </td><td> $0 . 4 7 8 _ { \pm . 0 4 1 }$ </td><td> $0 . 6 6 7 _ { \pm . 0 4 9 }$ </td><td> $0 . 2 8 8 _ { \pm . 0 0 5 }$ </td><td> $0 . 1 8 0 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 2 5 2 _ { \pm . 0 0 5 }$ </td><td> $0 . 2 0 3 _ { \pm . 0 2 4 }$ </td></tr><tr><td>English</td><td> $0 . 5 8 3 _ { \pm . 0 0 4 }$ </td><td> $0 . 2 3 2 _ { \pm . 0 1 9 }$ </td><td> $0 . 1 4 9 _ { \pm . 0 2 1 }$ </td><td> $0 . 2 2 0 { \scriptstyle \pm . 0 3 1 }$ </td><td> $0 . 1 4 4 _ { \pm . 0 1 3 }$ </td><td> $0 . 0 6 4 _ { \pm . 0 1 1 }$ </td><td> $0 . 1 0 8 { \scriptstyle \pm . 0 1 1 }$ </td><td> $0 . 0 7 7 { \scriptstyle \pm . 0 0 7 }$ </td></tr><tr><td>Banglish</td><td> $0 . 6 1 0 { \scriptstyle \pm . 0 1 5 }$ </td><td> $0 . 2 4 9 _ { \pm . 0 0 4 }$ </td><td> $0 . 1 5 5 { \scriptstyle \pm . 0 0 6 }$ </td><td> $0 . 2 2 7 { \scriptstyle \pm . 0 1 1 }$ </td><td> $0 . 1 2 9 _ { \pm . 0 1 7 }$ </td><td> $0 . 0 5 9 { \scriptstyle \pm . 0 1 1 }$ </td><td> $0 . 0 9 6 _ { \pm . 0 1 4 }$ </td><td> $0 . 0 8 9 _ { \pm . 0 0 6 }$ </td></tr></table>

Table 4: Test results as $\mathrm { \ m e a n _ { \pm s t d } }$ over seeds 42, 43, 44 (context-aware input). Domain accuracy is derived from the predicted intent. Dom. = domain, Wtd = weighted, EM = exact match, Std = standard deviation.

<table><tr><td>Model</td><td>Metric</td><td>Ctx</td><td>No Ctx</td><td>∆</td></tr><tr><td>mBERT</td><td>Intent acc.</td><td>0.605</td><td>0.356</td><td>+0.249</td></tr><tr><td></td><td>Token macro-F1</td><td>0.250</td><td>0.215</td><td>+0.035</td></tr><tr><td>BERT-base.</td><td>Intent acc.</td><td>0.578</td><td>0.321</td><td>+0.257</td></tr><tr><td></td><td>Token macro-F1</td><td>0.216</td><td>0.179</td><td>+0.037</td></tr><tr><td>BanglaBERT</td><td>Intent acc.</td><td>0.385</td><td>0.195</td><td>+0.190</td></tr><tr><td></td><td>Token macro-F1</td><td>0.128</td><td>0.106</td><td>+0.021</td></tr></table>

Table 5: Effect of dialogue context. Ctx = up to three previous turns; $\Delta = \mathrm { C t x } - \mathrm { N o }$ ctx.
<table><tr><td>Intent</td><td>mBERT</td><td>BERT</td><td>BanglaBERT</td></tr><tr><td>hotel_book_cancel</td><td>1.000</td><td>1.000</td><td>0.882</td></tr><tr><td>nid_card_service</td><td>0.250</td><td>0.154</td><td>0.000</td></tr><tr><td>police_services</td><td>0.000</td><td>0.154</td><td>0.267</td></tr><tr><td>govt_fin_assist</td><td>0.333</td><td>0.231</td><td>0.100</td></tr><tr><td>land_records</td><td>0.588</td><td>0.286</td><td>0.190</td></tr><tr><td> $\mathtt { t a x \_ s e r v i c e s }$ </td><td>0.000</td><td>0.476</td><td>0.286</td></tr></table>

Table 6: Selected per-intent F1 scores on the test set at the seed-42 checkpoint.

ventory, including semantically overlapping categories. Recurring confusions involve related user goals. For mBERT, cash\_out\_information is predicted as transaction\_support in three of four instances in each language representation. Similarly, BERT-base confuses tour\_cancellation with modify\_booking across all three representations. These errors suggest that semantic overlap between intent categories contributes to classification difficulty.

Detailed results on slot errors appear in Appendix B. Short, structured values such as book-

## 5 Conclusion

We introduce BanglaTOD, a parallel Bangla, English, and Banglish benchmark for multi-turn taskoriented dialogue understanding in Bangladeshspecific service domains. Baseline evaluations reveal substantial variation across language representations, persistent challenges in fine-grained intent classification and slot extraction, and clear benefits from incorporating dialogue history. These findings establish BanglaTOD as a resource for studying multilingual and contextdependent dialogue understanding in low-resource settings. Future work includes hierarchical intent modelling, Banglish normalization, and endto-end dialogue state tracking.

## Limitations

BanglaTOD is a relatively small, manually constructed dataset with uneven coverage across domains, intents, and slot types. Role-played conversations may not fully capture the linguistic diversity, informal expressions, and spelling variations encountered in real-world interactions, particularly in Banglish. Although the parallel representations enable controlled comparisons, they do not fully reflect naturally occurring code-switching or language-specific conversational patterns.

## References

Sadia Alam, Md Farhan Ishmam, Navid Hasin Alvee, Md Shahnewaz Siddique, Md Azam Hossain, and Abu Raihan Mostofa Kamal. 2025. BnSentMix: A diverse Bengali-English code-mixed dataset for sentiment analysis. In Proceedings of the First Workshop on Language Models for Low-Resource Languages, pages 68–77, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

KM Fahim A Bari, Muhammad Abdullah Adnan, and Nafis Sadeq. 2026. Leveraging code-mixed product metadata and user feedback for personalized recommendation on daraz bangladesh. Preprint, arXiv:2606.16387.

Abhik Bhattacharjee, Tahmid Hasan, Wasi Ahmad, Kazi Samin Mubasshir, Md Saiful Islam, Anindya Iqbal, M. Sohel Rahman, and Rifat Shahriyar. 2022. Banglabert: Language model pretraining and benchmarks for low-resource language understanding evaluation in bangla. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1318–1327. Association for Computational Linguistics.

Abhik Bhattacharjee, Tahmid Hasan, Wasi Uddin Ahmad, and Rifat Shahriyar. 2023. BanglaNLG and BanglaT5: Benchmarks and resources for evaluating low-resource natural language generation in Bangla. In Findings of the Association for Computational Linguistics: EACL 2023, pages 726–735, Dubrovnik, Croatia. Association for Computational Linguistics.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018.´ MultiWOZ - a large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 5016–5026, Brussels, Belgium. Association for Computational Linguistics.

Qian Chen, Zhu Zhuo, and Wen Wang. 2019. Bert for joint intent classification and slot filling. arXiv preprint arXiv:1902.10909.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186.

Bosheng Ding, Junjie Hu, Lidong Bing, Mahani Aljunied, Shafiq Joty, Luo Si, and Chunyan Miao. 2022. GlobalWoZ: Globalizing MultiWoZ to develop multilingual task-oriented dialogue systems. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1639–1657, Dublin, Ireland. Association for Computational Linguistics.

Md Arman Hossain, Mubashir Jawad, Fariha Khandaker Moon, Sonia Binte Siraj, Masfiqur Rahaman, Raihan ul Islam, Ahmed Wasif Reza, and Nafis Sadeq. 2026. Constraint-aware conversational job recommendation in code-mixed low-resource settings. Preprint, arXiv:2610.05787.

A H M Rezaul Karim and Özlem Uzuner. 2025. Leveraging machine-generated data for joint intent detection and slot filling in Bangla: A resource-efficient approach. In Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025), pages 208–216, Abu Dhabi, UAE. International Committee on Computational Linguistics.

Fahim Shahriar Khan, Mueeze Al Mushabbir, Mohammad Sabik Irbaz, and Md Abdullah Al Nasim. 2021. End-to-end natural language understanding pipeline for bangla conversational agents. In 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA), pages 205–210.

Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2021. MTOP: A comprehensive multilingual task-oriented semantic parsing benchmark. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2950–2962, Online. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Vishakh Padmakumar, Behnam Hedayatnia, Di Jin, Patrick Lange, Seokhwan Kim, Nanyun Peng, Yang Liu, and Dilek Hakkani-Tur. 2023. Investigating the representation of open domain dialogue context for transformer models. In Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 538–547, Prague, Czechia. Association for Computational Linguistics.

Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta, and Pranav Khaitan. 2020. Towards

scalable multi-domain conversational agents: The schema-guided dialogue dataset. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8689–8696.

Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian´ Ruder, and Iryna Gurevych. 2021. How good is your tokenizer? on the monolingual performance of multilingual language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3118–3135, Online. Association for Computational Linguistics.

Nafis Sadeq, Shafayat Ahmed, Sudipta Saha Shubha, Md. Nahidul Islam, and Muhammad Abdullah Adnan. 2020. Bangla voice command recognition in end-to-end system using topic modeling based contextual rescoring. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7894–7898.

Sagor Sarker. 2020. Banglabert: Bengali mask language model for bengali language understanding.

Sebastian Schuster, Sonal Gupta, Rushin Shah, and Mike Lewis. 2019. Cross-lingual transfer learning for multilingual task oriented dialog. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3795–3805.

Shijie Wu and Mark Dredze. 2020. Are all languages created equal in multilingual BERT? In Proceedings ofthe 5th Workshop on Representation Learningfor NLP, pages 120–130, Online. Association for Computational Linguistics.

Weijia Xu, Batool Haider, and Saab Mansour. 2020. End-to-end slot alignment and recognition for crosslingual NLU. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5052–5063, Online. Association for Computational Linguistics.

Qi Zhu, Kaili Huang, Zheng Zhang, Xiaoyan Zhu, and Minlie Huang. 2020. Crosswoz: A large-scale chinese cross-domain task-oriented dialogue dataset. Transactions of the Association for Computational Linguistics, 8:281–295.

## A Additional Training Details

All three backbones use identical optimization settings, summarized in Table 7.

For each target utterance $u _ { t }$ , the input concatenates up to three preceding turns c<sub>1</sub>, c<sub>2</sub>, c<sub>3</sub>:

$$
[ \mathrm { C L S } ] ~ u _ { t } [ \mathrm { S E P } ] ~ c _ { 1 } ~ c _ { 2 } ~ c _ { 3 } ~ [ \mathrm { S E P } ] .
$$

The maximum input length is 256 subtokens, with no truncation observed in the training or validation examples. Slot supervision is restricted to

<table><tr><td>Parameter</td><td>Value</td><td></td></tr><tr><td>Optimizer</td><td>AdamW 2019)</td><td>(Loshchilov and Hutter,</td></tr><tr><td>Learning rate</td><td> $1 . 5 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>LLRD</td><td>0.95</td><td></td></tr><tr><td>Weight decay</td><td>0.01</td><td></td></tr><tr><td>Scheduler</td><td>Cosine, 10% warm-up</td><td></td></tr><tr><td>Batch size</td><td>16</td><td></td></tr><tr><td>Epochs</td><td>≤ 100, patience 8</td><td></td></tr><tr><td>Dropout</td><td>0.15</td><td></td></tr><tr><td>Gradient clipping</td><td>1.0</td><td></td></tr><tr><td>Loss</td><td> $\mathcal { L } _ { i n t } + 1 . 5 \mathcal { L } _ { s l o t }$ </td><td></td></tr><tr><td>Intent loss</td><td>Class-weighted CE, label smoothing 0.1</td><td></td></tr><tr><td>Model selection</td><td> $0 . 5 \mathrm { A c c } _ { i n t } + 0 . 5 \mathrm { F } 1 _ { s l o t } ^ { t o k }$ </td><td>en,macro</td></tr><tr><td>Seeds</td><td>42, 43, 44</td><td></td></tr></table>

Table 7: Shared training configuration.

the first subtoken of each word in the target utterance; context, special tokens, and continuation subtokens are masked from the slot loss.

## B Slot Error Analysis

Table 8 summarizes representative token-level slot predictions for mBERT.

<table><tr><td>Error type</td><td>Slot label</td><td>P</td><td>R</td><td>Support</td></tr><tr><td>Reliable</td><td>B-booking_id</td><td>1.000</td><td>1.000</td><td>9</td></tr><tr><td></td><td>B-date_of_birth</td><td>1.000</td><td>1.000</td><td>3</td></tr><tr><td>Omitted</td><td>B-action</td><td>1.0000.500</td><td></td><td>6</td></tr><tr><td></td><td>B-hotel_name</td><td>1.0000.333</td><td></td><td>3</td></tr><tr><td></td><td>B-appl_status</td><td>0.000 0.000</td><td></td><td>3</td></tr><tr><td>Low precision B-query</td><td></td><td>0.2080.333</td><td></td><td>15</td></tr><tr><td></td><td>B-service</td><td>0.276 0.667</td><td></td><td>12</td></tr><tr><td>Multi-word</td><td>I-requ_info</td><td>0.4480.277</td><td></td><td>47</td></tr><tr><td></td><td>I-query</td><td>0.857 0.286</td><td></td><td>21</td></tr></table>

Table 8: Representative token-level slot predictions for mBERT at the seed-42 checkpoint. P = precision; R = recall; support = number of tokens.