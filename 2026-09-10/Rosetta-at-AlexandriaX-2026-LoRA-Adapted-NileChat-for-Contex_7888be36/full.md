# Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation

Nada Esmaeil<sup>1</sup>, Fathima Rena<sup>2</sup>, Sibi Subhash<sup>2</sup>, Osama Elgendy<sup>3</sup>, Mina Naguib<sup>1</sup>, Salma Omar<sup>4</sup>, Muhammad Arif<sup>5</sup>

<sup>1</sup>Tanta University, Egypt <sup>2</sup>Alliance University, India <sup>3</sup>Robusta Studio, Egypt <sup>4</sup>Alexandria University, Egypt <sup>5</sup>Yale University, USA

## Abstract

This paper describes the Rosetta system for Subtask 1 (Context-Aware English-to-Dialectal Arabic Dialogue Translation) of the AlexandriaX shared task, participating in both constrained and unconstrained tracks. The approach fine-tunes a LoRA adapter on NileChat-3B using structured system/user prompts that condition generation on dialect and dialogue context. For the unconstrained track, the adapter is additionally pretrained on MADAR and PADIC. Rosetta ranked 4th in the constrained track (spBLEU 26.10) and 5th in the unconstrained track (spBLEU 25.09). The experimental results demonstrate that external pretraining helps only two of thirteen dialects while slightly hurting overall performance, suggesting negative transfer.

## 1 Introduction

Arabic exhibits strong diglossia (Ferguson, 1959): Modern Standard Arabic (MSA) is the formal written variety, while regional dialects dominate everyday and online communication. Dialects diverge from MSA and from one another lexically, morphologically and pragmatically, and most lack standardized orthography or large parallel resources (Zaidan and Callison-Burch, 2014). Consequently, systems trained mainly on MSA generalize poorly to dialectal text.

AlexandriaX Subtask 1 (El Mekki et al., 2026) addresses this gap by requiring context-aware translation of English dialogue turns into a specified Arabic dialect. Systems receive the source turn together with preceding dialogue history, target country/dialect label, domain, speaker/addressee gender and (when available) persona information, and must produce only the translation of the current turn.

Participation spans both competition tracks under the oficial task constraints:

• Constrained: only the provided Alexandria data; models  5B parameters.

• Unconstrained: external data and larger models allowed.

The proposed approach fine-tunes a LoRA adapter (Hu et al., 2022) on NileChat-3B (El Mekki et al., 2025), an Arabic-pretrained instruction-tuned model, using structured prompts that encode dialect and conversational context. For the unconstrained track, the adapter is first pretrained on MADAR (Bouamor et al., 2018) and PADIC (Meftouh et al., 2015). The system placed 4th (spBLEU (Goyal et al., 2022) 26.10) and 5th (spBLEU 25.09) respectively. Performance is weakest on Mauritanian, Libyan, Sudanese and Moroccan varieties; external pretraining improves only Libyan and Moroccan while slightly degrading the overall average, and the pattern is not fully explained by dialect-family coverage alone. All code, fine-tuning scripts, and adapter weights are publicly available.<sup>1</sup>

## 2 Background

## 2.1 Task setup

The input to Subtask 1 of the AlexandriaX shared task (El Mekki et al., 2026) is an English dialogue turn together with its associated metadata (dialect/country label, domain, gender, dialogue history, and persona where available); the output is the translation of that turn only, in the target dialect, with prior turns supplied as context rather than retranslated.

<table><tr><td>Dialect</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Egyptian (EG)</td><td>3,108</td><td>1,113</td><td>1,113</td></tr><tr><td>Jordanian (JO)</td><td>5,501</td><td>1,113</td><td>1,109</td></tr><tr><td>Lebanese (LB)</td><td>8,906</td><td>1,118</td><td>1,110</td></tr><tr><td>Libyan (LY)</td><td>0</td><td>0</td><td>1,309</td></tr><tr><td>Moroccan (MA)</td><td>2,573</td><td>1,110</td><td>1,111</td></tr><tr><td>Mauritanian (MR)</td><td>5,515</td><td>1,114</td><td>1,119</td></tr><tr><td>Omani (OM)</td><td>6,280</td><td>1,109</td><td>1,107</td></tr><tr><td>Palestinian (PS)</td><td>14,933</td><td>1,110</td><td>1,111</td></tr><tr><td>Saudi (SA)</td><td>8,470</td><td>1,110</td><td>1,114</td></tr><tr><td>Sudanese (SD)</td><td>0</td><td>0</td><td>915</td></tr><tr><td>Syrian (SY)</td><td>6,071</td><td>1,119</td><td>1,114</td></tr><tr><td>Tunisian (TN)</td><td>2,034</td><td>1,116</td><td>1,114</td></tr><tr><td>Yemeni (YE)</td><td>3,089</td><td>1,118</td><td>1,113</td></tr><tr><td>Total</td><td>66,480</td><td>12,250</td><td>14,459</td></tr></table>

Table 1: Number of turns in the Alexandria train, development, and test splits by dialect.

## 2.2 Dataset

The organizers provide the Alexandria dataset (EL Mekki et al., 2026) of dialectal Arabic dialogues spanning multiple countries/dialects. Table 1 summarizes the number of turns per dialect across the train, development, and test splits.

## 2.3 Related work

Arabic’s diglossic nature — a formal MSA standard alongside diverse spoken dialects — has long challenged machine translation, compounded by persistent data scarcity across dialects (Zbib et al., 2012; Sajjad et al., 2020; Kadaoui et al., 2023). Early parallel resources remain limited in scale: PADIC (Meftouh et al., 2015) covers a handful of Maghrebi and Levantine dialects, while MADAR (Bouamor et al., 2018) ofers translations into 25 city-level dialects within the travel domain. Such resources are typically narrow in domain and lack dialogue-level context (Malaysha et al., 2024; Taguchi et al., 2025). The Alexandria dataset underlying this shared task addresses these gaps with a larger, conversation-based corpus spanning 13 Arab countries. In this work, we use MADAR and PADIC not as evaluation benchmarks but as external pretraining resources, aiming to leverage their broader dialectal coverage.

Adapting general-purpose LLMs to specific languages and cultures typically involves prompt engineering, culturally grounded fine-tuning, or continued pretraining on target-specific data (Bang et al., 2023; AlKhamissi et al., 2024; Naous et al., 2024). Arabic LLMs follow either path, trained from scratch (Sengupta et al., 2023) or adapted from existing multilingual models (Bari et al., 2025; Fanar Team et al., 2025). NileChat (El Mekki et al.,

2025) follows the latter approach, continuing pretraining of Qwen2.5-3B on synthetic and retrievalbased Arabic dialectal data, and outperforms other Arabic-aware models of comparable size while explicitly targeting specific dialectal communities (Egyptian and Moroccan Arabic) rather than treating Arabic as monolithic.

## 3 System Overview

The Rosetta system freezes UBC-NLP/NileChat-3B (El Mekki et al., 2025) and attaches a LoRA adapter $( r = 3 2 , \alpha = 3 2 , \mathrm { d r o p o u t { = } 0 ) }$ (Hu et al., 2022) targeting the attention and MLP projections $( \mathrm { q / k / v / o \_ p r o j , g a t e / u p / d o w n \_ p r o j } )$ . Submissions were made to both the constrained and unconstrained tracks; the two configurations share the same architecture and difer only in the pretraining data used prior to task fine-tuning.

## 3.1 Prompting Strategy

Each example is presented as a system–user pair. Following the prompting guidelines established in the Alexandria dataset release (EL Mekki et al., 2026), the system prompt instructs the model to return only the translation while respecting meaning, tone and gender direction. The user prompt supplies the target dialect, metadata (country, domain, participants, speaker, gender direction) and the conversation history (speaker + source + translation for preceding turns). At inference, the model’s own previous outputs are fed back as history. A concrete example of this multi-turn conversational prompt is detailed in Appendix A.2.

Pretraining Prompt. Unlike the Alexandria prompts, the external pretraining prompts do not contain dialogue history or detailed conversational metadata. The system prompt instructs the model to act as a translator and to return only the translation, while the user prompt specifies the target dialect, the corresponding country, and the English sentence. An illustrative example of a MADAR instance prompt formatted under this scheme is provided in Appendix A.1.

## 3.2 Tracks

In the constrained track the adapter is fine-tuned solely on the Alexandria training set (EL Mekki et al., 2026). In the unconstrained track it is first pretrained on MADAR (Bouamor et al., 2018) and PADIC (Meftouh et al., 2015) (mapped to the same country-level dialect labels and converted to

<table><tr><td colspan="2"></td><td rowspan="2">MADAR</td><td colspan="2">PADIC</td></tr><tr><td>Dialect</td><td>Train</td><td>Dev</td><td>Train Dev</td></tr><tr><td>Egyptian (EG)</td><td>13,800</td><td>1,600</td><td>0</td><td>0</td></tr><tr><td>Jordanian (JO)</td><td>3,200</td><td>400</td><td>0</td><td>0</td></tr><tr><td>Lebanese (LB)</td><td>10,600</td><td>1,200</td><td>0</td><td>0</td></tr><tr><td>Libyan (LY)</td><td>3,200</td><td>400</td><td>0</td><td>0</td></tr><tr><td>Moroccan (MA)</td><td>12,200</td><td>1,400</td><td>5,767</td><td>641</td></tr><tr><td>Mauritanian (MR)</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Omani (OM)</td><td>1,600</td><td>200</td><td>0</td><td>0</td></tr><tr><td>Palestinian (PS)</td><td>1,600</td><td>200</td><td>5,767</td><td>641</td></tr><tr><td>Saudi (SA)</td><td>3,200</td><td>400</td><td>0</td><td>0</td></tr><tr><td>Sudanese (SD)</td><td>1,600</td><td>200</td><td>0</td><td>0</td></tr><tr><td>Syrian (SY)</td><td>3,200</td><td>400</td><td>5,767</td><td>641</td></tr><tr><td>Tunisian (TN)</td><td>12,200</td><td>1,400</td><td>0</td><td>0</td></tr><tr><td>Yemeni (YE)</td><td>1,600</td><td>200</td><td>0</td><td>0</td></tr><tr><td>Total</td><td>68,000</td><td>8,000</td><td>17,301</td><td>1,923</td></tr></table>

Table 2: Number of training and development examples by dialect in the MADAR and PADIC datasets.

English–dialect pairs via NLLB-200 (NLLB Team et al., 2022)), then fine-tuned on Alexandria data.

## 4 Experimental Setup

## 4.1 Data Splits

We use the oficial Alexandria splits (EL Mekki et al., 2026) (Table 1). For the unconstrained track, MADAR (Bouamor et al., 2018) and PADIC (Meftouh et al., 2015) are additionally used during an adapter pre-training stage before fine-tuning on the Alexandria training data. The number of training and development examples used from MADAR and PADIC is summarized in Table 2.

## 4.2 Preprocessing

For the unconstrained track, English–dialect parallel data is constructed from MADAR and PADIC by translating their MSA side to English with facebook/nllb-200-3.3B (NLLB Team et al., 2022) (beam size 4, max length 256).

MADAR city-level labels are mapped to Alexandria country-level categories (e.g., Cairo/Alexandria/Aswan  Egyptian, Rabat/Fes  Moroccan). From PADIC, three dialects were incorporated, filtering out rows unaligned with MSA. The MADAR development splits and a 10% held-out portion of PADIC were used to monitor the pretraining stage.

## 4.3 Training Configuration

Maximum sequence length is 1,024. Training is conducted for one epoch on a single Tesla T4 (batch size 8, gradient accumulation 4, learning rate $2 ~ \times ~ 1 0 ^ { - 4 }$ , cosine schedule, warmup 0.03, AdamW 8-bit (Loshchilov and Hutter, 2019; Dettmers et al., 2022)). Loss is computed only on the assistant completion. Training was implemented with Unsloth and the PEFT library under 4-bit NF4 quantization (bitsandbytes). To mitigate exposure bias (Ranzato et al., 2016), history noising is applied: with probability $p _ { \mathrm { t r u n c } } = 0 . 1 5$ the history is truncated to a random prefix, and with probability $p _ { \mathrm { n o i s e } } = 0 . 2 5$ individual previous translations undergo word-level drop/swap/duplication. The clean reference remains the training target. History noising was applied only during task fine-tuning, not during the MADAR/PADIC pretraining stage.

## 4.4 Decoding and Context Handling

At inference time, deterministic beam-search decoding is employed with five beams and a length penalty of 0.7 (Wu et al., 2016). Sampling is disabled and early stopping is enabled. The same decoding configuration is used for both the constrained and unconstrained systems.

For dialogue translation, previously generated translations are incorporated into the prompt as conversation history when translating subsequent turns within the same dialogue.

## 5 Results

Oficial results. Table 3 reports per-dialect sp-BLEU (Goyal et al., 2022) (SentencePiece with the flores200 tokenizer) and chrF++ (Popović, 2017) scores for our submitted systems on the oficial test set, for both the constrained and unconstrained tracks. The overall average spBLEU / chrF++ is 26.10 / 41.79 for the constrained track (4th place) and 25.09 / 41.02 for the unconstrained track (5th place). All scores above are oficial test-set evaluation results from the submitted systems.

Analysis by dialect. Performance varies substantially across dialects in both tracks. Syrian, Jordanian, Palestinian, Saudi, Lebanese, and Egyptian consistently achieve the highest scores (all above 29 spBLEU in the constrained track), while Mauritanian is the clear outlier at the low end (12.13 constrained, 10.83 unconstrained) — roughly a third of the score of the best-performing dialects. Libyan, Moroccan, and Sudanese also trail the mid-to-high performing group, consistent with our expectation that Maghrebi and near-Maghrebi dialects, along with Mauritanian, are comparatively low-resource (Younes et al., 2020) and linguistically more distant from the dialects that dominate available pretraining and task data (Zaidan and Callison-Burch, 2014).

<table><tr><td rowspan="2">Dialect</td><td colspan="2">Constrained</td><td colspan="2">Unconstrained</td></tr><tr><td>spBLEU</td><td>chrF++</td><td>spBLEU</td><td>chrF++</td></tr><tr><td>Egyptian (EG)</td><td>30.85</td><td>45.43</td><td>30.41</td><td>44.89</td></tr><tr><td>Jordanian (JO)</td><td>33.39</td><td>48.45</td><td>32.26</td><td>47.57</td></tr><tr><td>Lebanese (LB)</td><td>29.70</td><td>44.53</td><td>28.09</td><td>43.22</td></tr><tr><td>Libyan (LY)</td><td>19.82</td><td>37.74</td><td>20.14</td><td>37.71</td></tr><tr><td>Moroccan (MA)</td><td>21.42</td><td>36.66</td><td>21.84</td><td>37.52</td></tr><tr><td>Mauritanian (MR)</td><td>12.13</td><td>29.23</td><td>10.83</td><td>28.16</td></tr><tr><td>Omani (OM)</td><td>26.07</td><td>42.06</td><td>24.78</td><td>41.59</td></tr><tr><td>Palestinian (PS)</td><td>30.55</td><td>45.58</td><td>29.56</td><td>44.60</td></tr><tr><td>Saudi (SA)</td><td>30.56</td><td>46.20</td><td>29.58</td><td>45.36</td></tr><tr><td>Sudanese (SD)</td><td>21.59</td><td>37.40</td><td>20.49</td><td>36.77</td></tr><tr><td>Syrian (SY)</td><td>34.73</td><td>50.35</td><td>33.64</td><td>49.25</td></tr><tr><td>Tunisian (TN)</td><td>26.12</td><td>40.55</td><td>24.50</td><td>38.98</td></tr><tr><td>Yemeni (YE)</td><td>22.36</td><td>39.05</td><td>20.11</td><td>37.64</td></tr><tr><td>Average</td><td>26.10</td><td>41.79</td><td>25.09</td><td>41.02</td></tr></table>

Table 3: Oficial test-set performance (spBLEU and chrF++) across dialects for the constrained and unconstrained tracks.

Constrained vs. unconstrained comparison. Counterintuitively, the unconstrained track where the adapter is additionally pretrained on MADAR and PADIC before task fine-tuning — scores lower on average than the constrained track (25.09 vs. 26.10), and underperforms on 11 of the 13 dialects. The two exceptions are Libyan (+0.32) and Moroccan (+0.42), both Maghrebi varieties for which MADAR and PADIC provide comparatively strong coverage (Table 2). This is consistent with pretraining helping specifically where it adds the most relevant, dialect-matched data, while diluting or interfering with the adapter’s fit elsewhere.

However, this explanation is not fully suficient: Sudanese receives a comparable injection of previously-absent training signal (1,600 MADAR examples, versus zero in-task training examples in the constrained setting) but degrades under the unconstrained setup (-1.10), the opposite direction from Libyan despite an analogous zero-resource starting point. Sudanese is not a Maghrebi variety, so data availability alone does not explain the divergence. A plausible contributing factor is variation in back-translation quality across dialect families — our MADAR/PADIC pairs are constructed via NLLB-200 MSA English translation (§4.2), and translation noise on the MSA source side may be unevenly distributed across dialects, disproportionately harming some target varieties. We treat the overall pattern as a genuine, only partially understood instance of negative transfer rather than a clean Maghrebi-coverage story, and leave a controlled study of pretraining-data quality per dialect to future work.

Qualitative error analysis. To complement the aggregate metrics, we manually inspected the 20 lowest-scoring Egyptian Arabic (EG) sentencelevel spBLEU outputs on the dev set. Two recurring patterns emerge. First, the model tends to translate every source word even when the natural dialectal rendering is more concise, producing fluent but overly literal output. For example, given the source:

“Not at all, please, go ahead. I hope everything is okay.”, the model produces:

$$
X _ { ( u ) } d = \lim \limits _ { b  \infty } a _ { i } \sin b \sin b \sin b \sin b \sin c
$$

where the reference instead condenses the turn to: $\int \limits _ { a } ^ { b } d x - \int \limits _ { a } ^ { b } d x = 1$

reflecting how Egyptian speakers compress routine social exchanges rather than translating each clause.

Second, the model under-generates the English code-switching that is pervasive in naturalistic Egyptian dialogue (e.g., technical or borrowed terms like scan or notification), instead rendering them in Arabic script even when the reference preserves the Latin form: for “It’s all scanned and confirmed in the system. You should get a notification now.”, the model outputs:

$$
\begin{array}{c} v ^ { s } ( x ^ { s } ) s ^ { s } ( 1 , s ^ { s } ) = 0 , v ^ { s } ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( 2 , 0 ) = 1 , v ( , 0 ) = 1 , v ( , 0 ) = 1 , v ( , 0 ) = 1 , v = , 0 = 1 , v ( , 0 ) = 1 , v , 0 = 1 , v = , 0 0 , v = \in 1 , v \in \in \end{array}
$$

$i s = i \cos 1 \sin 5 \sin 3 \leq i \sin 3 \leq i \sin 3 \cos 4 1 \sin 5 \sin 9 2$ دިܳڢมฆ. notification

To quantify this, we flagged all dev examples where the reference contains Latin-script tokens but the model’s prediction does not. This under-generation pattern is widespread and dialectdependent, afecting 319 Moroccan, 148 Lebanese, 123 Tunisian, and 70 Egyptian examples, among others. The reverse case — Latin script appearing in the prediction but absent from the reference — is comparatively rare except for Tunisian (212 cases), suggesting the model has a general bias toward suppressing code-switching that it must actively overcome, with dialect-specific variation in how often this succeeds.

## 6 Conclusion

This work presented Rosetta, a LoRA-adapted NileChat system for context-aware English-todialectal-Arabic dialogue translation across thirteen regional varieties. Structured prompting that explicitly incorporates dialogue history and speaker metadata, together with history noising, enables competitive performance under tight parameter and data constraints. Under oficial evaluation, the constrained system ranked 4th (sp-BLEU 26.10), while the unconstrained configuration placed 5th (spBLEU 25.09). External sentence-level dialectal pretraining did not yield overall downstream gains and induced mild negative transfer across most dialects, with improvements restricted solely to Libyan and Moroccan and no single explanatory factor fully accounting for the pattern.

## References

Badr AlKhamissi, Muhammad ElNokrashy, Mai Alkhamissi, and Mona Diab. 2024. Investigating cultural alignment of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12404–12422, Bangkok, Thailand. Association for Computational Linguistics.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 675–718, Nusa Dua, Bali. Association for Computational Linguistics.

M Saiful Bari, Yazeed Alnumay, Norah A. Alzahrani, Nouf M. Alotaibi, Hisham Abdullah Alyahya, Sultan AlRashed, Faisal Abdulrahman Mirza, Shaykhah Z. Alsubaie, Hassan A. Alahmed, Ghadah Alabduljabbar, Raghad Alkhathran, Yousef Almushayqih, Raneem Alnajim, Salman Alsubaihi, Maryam Al Mansour, Saad Amin Hassan, Dr. Majed Alrubaian, Ali Alammari, Zaki Alawami, and 7 others. 2025. ALLam: Large language models for arabic and english. In The Thirteenth International Conference on Learning Representations.

Houda Bouamor, Nizar Habash, Mohammad

Salameh, Wajdi Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa, Fadhl Eryani, Alexander Erdmann, and Kemal Oflazer. 2018. The MADAR Arabic dialect corpus and lexicon. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 2022. 8-bit optimizers via block-wise quantization. Preprint, arXiv:2110.02861.

Abdellah El Mekki, Houdaifa Atou, Omer Nacar, Shady Shehata, and Muhammad Abdul-Mageed. 2025. NileChat: Towards linguistically diverse and culturally aware LLMs for local communities. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10967–10991, Suzhou, China. Association for Computational Linguistics.

Abdellah El Mekki, AbdelRahim A. Elmadany, Samar M. Magdy, Saad Ezzini, Mo El-Haj, Mustafa Jarrar, Zaid Alyafeai, Bernard Ghanem, and Muhammad Abdul-Mageed. 2026. AlexandriaX 2026: The First Shared Task on Dialectal Arabic Machine Translation. In Proceedings of The Fourth Arabic Natural Language Processing Conference: Shared Tasks, Budapest, Hungary. Association for Computational Linguistics.

Abdellah EL Mekki, Samar M. Magdy, Houdaifa Atou, Ruwa AbuHweidi, Baraah Qawasmeh, Omer Nacar, Thikra Al-hibiri, Razan Saadie, Hamzah A. Alsayadi, Nadia Ghezaiel Hammouda, Alshima Mohammed Alkhazimi, Aya Hamod, Al-Yas Yaqoob Al-Ghafri, Wesam El-Sayed, Asila Ismail al Sharji, Mohamad Ballout, Anas Belfathi, Karim Ghaddar, Serry Sibaee, and 28 others. 2026. Alexandria: A multidomain dialectal Arabic machine translation dataset for culturally inclusive and linguistically diverse LLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32567–32592, San Diego, California, United States. Association for Computational Linguistics.

Fanar Team, Ummar Abbas, Mohammad Shahmeer Ahmad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari, Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Masoomali Fatehkia, Anastasios Fragkopoulos, Maram Hasanain, and 23 others. 2025. Fanar: An arabic-centric multimodal generative ai platform. Preprint, arXiv:2501.13944.

Charles A. Ferguson. 1959. Diglossia. WORD, 15(2):325–340.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for lowresource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Karima Kadaoui, Samar M. Magdy, Abdul Waheed, Md Tawkat Islam Khondaker, Ahmed Oumar El-Shangiti, El Moatez Billah Nagoudi, and Muhammad Abdul-Mageed. 2023. TARJAMAT: Evaluation of bard and ChatGPT on machine translation of ten Arabic varieties. In Proceedings of ArabicNLP 2023, pages 52–75, Singapore (Hybrid). Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Sanad Malaysha, Mo El-Haj, Saad Ezzini, Mohammed Khalilia, Mustafa Jarrar, Sultan Almujaiwel, Ismail Berrada, and Houda Bouamor. 2024. AraFinNLP 2024: The first Arabic financial NLP shared task. In Proceedings ofthe Second Arabic Natural Language Processing Conference, pages 393–402, Bangkok, Thailand. Association for Computational Linguistics.

Karima Meftouh, Salima Harrat, Salma Jamoussi, Mourad Abbas, and Kamel Smaili. 2015. Machine translation experiments on PADIC: A parallel Arabic DIalect corpus. In Proceedings of the 29th Pacific Asia Conference on Language, Information and Computation, pages 26– 34, Shanghai, China.

Tarek Naous, Michael J Ryan, Alan Ritter, and Wei Xu. 2024. Having beer after prayer? measuring cultural bias in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16366–16393, Bangkok, Thailand. Association for Computational Linguistics.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Hefernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, and 20 others. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Maja Popović. 2017. chrF++: words helping character n-grams. In Proceedings of the Second Conference on Machine Translation, pages 612–618, Copenhagen, Denmark. Association for Computational Linguistics.

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence level training with recurrent neural networks. Preprint, arXiv:1511.06732.

Hassan Sajjad, Ahmed Abdelali, Nadir Durrani, and Fahim Dalvi. 2020. AraBench: Benchmarking dialectal Arabic-English machine translation. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5094–5107, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, Lalit Pradhan, Zain Muhammad Mujahid, Massa

Baali, Xudong Han, Sondos Mahmoud Bsharat, and 13 others. 2023. Jais and jais-chat: Arabiccentric foundation and instruction-tuned open generative large language models. Preprint, arXiv:2308.16149.

Chihiro Taguchi, Seng Mai, Keita Kurabe, Yusuke Sakai, Georgina Agyei, Soudabeh Eslami, and David Chiang. 2025. Languages still left behind: Toward a better multilingual machine translation benchmark. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20131–20143, Suzhou, China. Association for Computational Linguistics.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jef Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, and 12 others. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. Preprint, arXiv:1609.08144.

Jihene Younes, Emna Souissi, Achour Hadhémi, and Ahmed Ferchichi. 2020. Language resources for maghrebi arabic dialects’ nlp: a survey. Language Resources and Evaluation, 54.

Omar F. Zaidan and Chris Callison-Burch. 2014. Arabic dialect identification. Computational Linguistics, 40(1):171–202.

Rabih Zbib, Erika Malchiodi, Jacob Devlin, David Stallard, Spyros Matsoukas, Richard Schwartz, John Makhoul, Omar F. Zaidan, and Chris Callison-Burch. 2012. Machine translation of Arabic dialects. In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 49–59, Montréal, Canada. Association for Computational Linguistics.

༡؇༥۰੼ ොٺ؇ج ިܳ . دިܳڢมฆ ނި ل۬ ༡؊ۏ٭ٴ጑ጥ أَ؇ . اܳگ؇༟۰ اරඝ ሒᇭ ި۱   
،۰ਃ಻؇ّ .ሒᇿިڢ

## A Prompt Templates

This section details the prompt templates used across our experiments. Section A.1 shows the single-turn prompt template used for MADAR and PADIC adapter pretraining, and Section A.2 illustrates the full multi-turn conversational prompt used for the Alexandria dataset.

## A.1 Pretraining Prompt Template (MADAR / PADIC)

The pretraining prompt contains only sentencelevel translation instructions along with the target country metadata:

## System Prompt:

You are an expert translator.

\- Return only the translated text.

\- Do not add any code, explanations, comments,

or any other extra text.

\- Consider the country in your translation.

## User Prompt:

Translate the English sentence into Egyptian Arabic.

Metadata:

\- Country: EG

## Sentence to Translate:

It’s at the end of the hall. I’ll get you some right now. If you need anything else, just let me know.

Assistant Response:

## A.2 Alexandria Conversational Prompt Template

The task fine-tuning and inference prompt includes conversational history, participants, domain, and gender direction:

## System Prompt:

You are an expert translator.

\- Return only the translated text.

\- Do not add any code, explanations, comments, or any other extra text.

\- Keep the meaning and tone and respect the gender direction.

\- Consider the country, the domain, the participants, and the speaker in your translation.

## User Prompt:

Translate the English sentence into Egyptian Arabic (Cairene) Dialect.

## Metadata:

\- Country: EG

\- Domain: Agriculture and farming

\- Participants: Wholesale Buyer, Wholesale Seller

\- Speaker: Wholesale Seller

\- Speaker Direction: female  male

## Conversation History:

Wholesale Buyer: Good morning. I’m looking to source 10 tons of premium artichokes for export. People say the best quality in Obour comes from your section, is that right?

## Translation:

## Sentence to Translate:

Good morning to you. You heard correctly. My artichokes are the best you’ll find. They are top-grade, perfect for export. Let me show you a sample.

## Assistant Response:

؜٭ۇٴ.۰ أور ل۹ ༠ܹ٭มฃ ይዧٺݱڎߌߵ، ݁أި݄ل ا৑৙ިَاع، أۋފ݆ ݆݁ اܳފިق، ሒᇭ