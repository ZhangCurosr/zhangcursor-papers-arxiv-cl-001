# REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic

Ahmed M. Fetouh<sup>1</sup> and Mariam Labib<sup>2</sup> and Nsrin Ashraf<sup>1,2</sup> and Omer Dawood<sup>3</sup>

and Hamada Nayel<sup>1,3</sup>

<sup>1</sup>Faculty of Computers and Artificial Intelligence, Benha University, Egypt

<sup>2</sup>Computer Engineering, Elsewedy University of Technology, Cairo, Egypt

<sup>3</sup>College of Engineering, Wadi Ad Dwaser,

Prince Sattam Bin Abdulaziz University, KSA

## Abstract

Detecting harmful content in Arabic is challenging because of dialectal variations and the semantic overlap between different harm categories. This paper presents a twostage approach for classifying Arabic user prompts into seven categories of harmful content. First, GLiNER is used to extract harm-related entities from each prompt. The extracted entities are then added to the original text and passed to a fine-tuned MARBERT classifier for final classification. The approach combines explicit entity information with the contextual understanding of a pretrained Arabic language model. The system was evaluated on the development set of the ArGuard 2026 shared task. The results achieved a macro-F1 score of 0.886 and an accuracy of 0.897, demonstrating promising performance in finegrained harmful content detection. The system reported an accuracy of 0.793 and Macro-F score of 0.735 on a blind test set for subtask 1. For subtask 2, it achieved an accuracy of 0.770 and Macro-F score of 0.754. The analysis also showed that distinguishing between related categories remains challenging. These findings suggest that integrating NER with Arabic language models is a promising approach to harmful content detection.

## 1 Introduction

The rapid adoption of large language models (LLMs) and conversational artificial intelligence has increased the importance of detecting harmful content in user prompts. Although these systems provide valuable capabilities for information retrieval, content generation, and interactive assistance, they may also encounter prompts involving violence, self-harm, harassment, bullying, hate speech, fraud, and adult content. Reliable harmful content detection is therefore essential for improving the safety and responsible use of language technologies. However, identifying harmful content remains challenging, particularly in Arabic, where dialectal variation, informal expressions, and context-dependent meanings can complicate automated classification (Ashraf et al., 2025, 2022; Nayel et al., 2021).

Fine-grained harmful content detection presents additional challenges beyond distinguishing between safe and unsafe text. Systems must identify the specific category of harmful content expressed in a prompt, even when different categories share similar vocabulary or semantic characteristics. For example, bullying, harassment, and hate speech may all involve offensive language or threats, while self-harm and harm to others may share references to violence or physical injury. Furthermore, a prompt may contain potentially harmful entities without explicitly expressing harmful intent, making it necessary to consider both the words used and their surrounding context. Class imbalance can further complicate this task, as models may perform unevenly across categories with different numbers of training examples (Nayel, 2020).

Transformer-based language models have demonstrated their usefulness in text classification by learning contextual representations from large-scale pretraining. MARBERT is relevant to Arabic text because its pretraining includes Modern Standard Arabic and dialectal Arabic data (Abdul-Mageed et al., 2021). Nevertheless, contextual representations may not always capture sufficiently explicit entity-level information needed to distinguish among harmful content categories. Named entity recognition (NER) offers a complementary approach by identifying relevant entities and expressions, such as weapons, selfinjury references, sensitive financial information, and offensive expressions (Eldouby et al., 2026). Integrating these extracted entities into a classification pipeline may provide additional structured cues to support fine-grained decisions.

This paper presents our system for Subtask B2 of ArGuard 2026 (Alam et al., 2026), which combines GLiNER-based entity extraction with a fine-tuned MARBERT classifier. Harm-related entities are first extracted and transformed into a structured representation that is appended to the original prompt. The enriched input is then classified by MARBERT into one of seven harmful-content categories. By integrating explicit entity information with contextual language representations, the approach improves category discrimination. While achieving strong results on categories such as adult content and hate speech, error analysis reveals persistent confusion among semantically similar classes, particularly harm to others, bullying, and harassment, highlighting the ongoing challenge of fine-grained harmful content detection in Arabic.

The main contributions of this work are as follows:-

• It presents a two-stage framework that integrates generalist NER with Arabic transformer-based classification for finegrained harmful content detection,

• it introduces a structured entity-augmented representation that makes harm-related entity cues explicit within the classifier’s input,

• it provides an empirical evaluation and error analysis on the ArGuard 2026 development set, identifying category-specific strengths and the principal sources of classification errors.

The remainder of this paper is organized as follows. Section 2 describes the shared-task objectives and dataset. Section 3 presents the overall system architecture, followed by the entity extraction and input representation methods. The MARBERT classification approach and experimental configuration are then described. Finally, the paper reports the evaluation results, discusses classification errors and ablation experiments, and concludes with directions for future work.

## 2 Task and Data

Subtask B2 of ArGuard 2026 (Alam et al., 2026), asks systems to assign each Arabic user prompt to one of seven granular harm categories: adult\_content, harm-to-others, self-harm, harassment, FraudandDeception, Bully, and hate\_speech. The training set contains 16,835 prompts (Kmainasi et al., 2026). Each sample consists of a prompt, a binary safety label (safe/unsafe), a granular label, and a dialect tag (e.g., Modern Standard Arabic and Levantine). The class distribution is imbalanced: adult\_content is the largest class (4,592 samples, 27.28%), while hate\_speech is the smallest (943 samples, 5.60%), an imbalance ratio of roughly 4.9:1. The task is difficult because several categories overlap semantically (for example, harassment, bullying, and hate speech), because harmful intent is often expressed through dialectal or idiomatic wording, and because surface-benign prompts can still belong to a sensitive category.

<table><tr><td>Category</td><td>Samples</td><td>%</td><td></td></tr><tr><td>adult_content</td><td>4,592</td><td>27.28%</td><td rowspan="4"></td></tr><tr><td>harm-to-others</td><td>3,040</td><td>18.06%</td></tr><tr><td>self-harm</td><td>2,723</td><td>16.17%</td></tr><tr><td>harassment</td><td>2,255</td><td>13.39%</td></tr><tr><td>FraudandDeception</td><td>1,764</td><td>10.48%</td></tr><tr><td>Bully</td><td>1,518</td><td>9.02%</td></tr><tr><td>hate_speech</td><td>943</td><td>5.60%</td></tr><tr><td>Total</td><td>16,835</td><td>100%</td></tr></table>

Figure 1: Granular category distribution in the training set.

Before modeling, we inspected missing values, duplicate prompts and identifiers, and prompt length statistics (in characters and words). Rows with missing values were removed from the training set. The labelled development set was used for model selection and evaluation, and the unlabeled test set was used only for final prediction.

## 3 System Overview

Our system is a two-stage pipeline that couples NER with a pretrained Arabic transformer. In the first stage, GLiNER (Zaratiana et al., 2024) extracts harm-related entities from the prompt. In the second stage, the extracted entities are serialized as text and appended to the original prompt, and the enriched text is classified by a fine-tuned MARBERT model. The intuition is that explicit entity cues (weapons, self-injury terms, financial credentials, protected groups, insults) give the classifier a compact, category-oriented summary of the prompt that complements the contextual representation learned by the encoder. The same pipeline is applied unchanged to the training, development, and test sets.

![](images/338246086fb6db170d1cb39a2fd4f9b1f1ee1d86bfd12b3a83686df323bbc521.jpg)  
Figure 2: The pipeline of the proposed system.

## 3.1 Entity Extraction with GLiNER

We use GLiNER, a generalist NER model that accepts an arbitrary label set at inference time, with the multilingual checkpoint<sup>1</sup>. Because GLiNER is zero-shot with respect to the label set, no NERspecific training data is required. We define 34 entity labels organized into seven harm domains aligned with the task categories:

• Violence and physical harm: weapon, firearm, explosive, bomb, knife, poison, violent action, physical harm.

• Self-harm: suicide, suicide method, self harm, self injury.

• Fraud and sensitive information: credit card, bank account, financial information, password, credential, identity document, personal information.

• Adult content: sexual activity, pornographic content, nudity, explicit content.

• Hate: ethnic group, religious group, nationality, protected group.

• Harassment: insult, threat, bullying, offensive expression.

• Contextual entities: person, organization, location.

Entities are predicted with a confidence threshold of 0.5. For every detected entity we store its surface text, confidence score, and character offsets. Duplicate entities within the same label are collapsed by surface form, yielding a unique set of entities per label for each prompt. The results are saved in JSONL format together with the prompt identifier and, where available, the gold labels.

## 3.2 NER-Augmented Text Representation

The structured entity output is converted into a plain-text block that follows the original prompt. For every one of the 34 labels, in a fixed order, we write one line of the form label: entity\_1, entity\_2, ..., or label: none if nothing was detected. The final model input is:

<original prompt>   
[ENTITIES]   
weapon: <entities or none>   
firearm: <entities or none>   
location: <entities or none>

The fixed ordering and explicit none values keep the input format identical across samples, so that the presence or absence of a cue in a given domain is directly visible to the classifier. Granular labels are mapped to integer identifiers with a label encoder fitted on the training set; the same encoder is reused for the development set and to decode test predictions.

## 3.3 MARBERT Classifier

We fine-tune MARBERT (UBC-NLP/MARBERT), a BERT-based model pretrained on large-scale Modern Standard Arabic and dialectal Arabic text, which suits the dialectal variation present in the data. We attach a sequence-classification head with seven output units and train with the standard cross-entropy objective, without class weighting or resampling. Inputs are tokenized with the MARBERT tokenizer, truncated and padded to a maximum length of 256 tokens. Only the granular label is predicted; the binary safe/unsafe label is not used as an input or auxiliary target.

Table 1: Training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup</td><td>10%</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Train / eval batch size</td><td>32 / 16</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Model selection</td><td>Macro-F1</td></tr></table>

## 3.4 Training Configuration

Training uses the Hugging Face Trainer<sup>2</sup> with the hyperparameters in Table 1. After each epoch the model is evaluated on the development set, and the checkpoint with the highest macro-F1 is restored at the end of training.

## 4 Evaluation and Inference

The official primary metric is macro-F1, which weights all seven categories equally and is therefore appropriate under class imbalance. We additionally report accuracy, macro-averaged precision, and macro-averaged recall, together with per-class precision, recall, and F1 and raw and rownormalized confusion matrices to analyze interclass confusion.

At test time, each prompt passes through the identical pipeline: GLiNER entity extraction with the same label set and threshold, construction of the [ENTITIES]-augmented text, tokenization with the MARBERT tokenizer, and prediction with the selected checkpoint. The predicted class identifiers are mapped back to granular label names with the training label encoder.

## 5 Results

Table 2 summarizes the development-set performance of the GLiNER + MARBERT system on the 2,405 development prompts. The system reaches a macro-F1 of 0.886 and an accuracy of 0.897. Macro-averaged precision and recall are close (0.885 and 0.888), indicating that the model is not systematically biased toward over- or under-predicting any category.

Table 3 reports per-class results. adult\_content is almost perfectly recognized $( \mathrm { F 1 } = 0 . 9 8 7 )$ , and hate\_speech also performs strongly (F1 = 0.923) despite being the smallest training class, which suggests that the entity cues for protected groups and offensive expressions are informative. The weakest class is harm-to-others (F1 = 0.815), followed by Bully (0.834) and harassment (0.848).

Table 2: Overall development-set results.
<table><tr><td>Metric Score</td></tr><tr><td>Macro-F1 0.886</td></tr><tr><td>Accuracy 0.897</td></tr><tr><td>Macro-Precision 0.885</td></tr><tr><td>Macro-Recall 0.888</td></tr></table>

## 5.1 Error Analysis

As shown in Figure 3, errors concentrate among semantically adjacent categories rather than being spread uniformly. The largest confusion is between harm-to-others and self-harm: 28 harm-to-others prompts are predicted as self-harm and 27 self-harm prompts as harm-to-others, which is plausible since both involve physical harm, weapons, and violent actions, and differ mainly in who the target is. harm-to-others is also confused with Bully, FraudandDeception, and harassment (18 prompts each), so it acts as a hub class that absorbs errors from several neighbors. A second cluster involves Bully and harassment, with 12 and 14 prompts misclassified in the two directions respectively; these categories share insult- and threat-related entities, so the entity block provides little discriminative signal between them. Bully is also occasionally predicted as hate\_speech (7 prompts), consistent with bullying that targets a protected group. In contrast, adult\_content is rarely confused with any other class.

## 6 Ablation Study

We conduct ablation experiments to isolate the contribution of the NER-augmented input and to test the design choices of the entity extraction and input construction stages. Unless stated otherwise, every variant is trained with the configuration in Table 1 and evaluated on the same development set, with macro-F1 as the primary metric. Because differences between variants can be small relative to fine-tuning variance, each variant is trained with three random seeds and we report the mean and standard deviation.

Table 3: Per-class development-set classification report of the GLiNER + MARBERT system.
<table><tr><td>Category</td><td>Precision</td><td>Recall</td><td>F1</td><td>Support</td></tr><tr><td>Bully</td><td>0.8341</td><td>0.8341</td><td>0.8341</td><td>217</td></tr><tr><td>FraudandDeception</td><td>0.8842</td><td>0.9087</td><td>0.8963</td><td>252</td></tr><tr><td>adult_content</td><td>0.9863</td><td>0.9878</td><td>0.9871</td><td>657</td></tr><tr><td>harassment</td><td>0.8457</td><td>0.8509</td><td>0.8483</td><td>322</td></tr><tr><td>harm-to-others</td><td>0.8231</td><td>0.8060</td><td>0.8145</td><td>433</td></tr><tr><td>hate_speech</td><td>0.9191</td><td>0.9259</td><td>0.9225</td><td>135</td></tr><tr><td>self-harm</td><td>0.9044</td><td>0.8997</td><td>0.9021</td><td>389</td></tr><tr><td>Accuracy</td><td></td><td>0.8969</td><td></td><td>2405</td></tr><tr><td>Macro average</td><td>0.8853</td><td>0.8876</td><td>0.8864</td><td>2405</td></tr><tr><td>Weighted average</td><td>0.8966</td><td>0.8969</td><td>0.8967</td><td>2405</td></tr></table>

![](images/077c2edf2bc07a24d1bd9b9e26c19af734b78de5af49bef4330b71757e3e1c17.jpg)  
Figure 3: Development-set confusion matrix of the GLiNER + MARBERT system.

## 6.1 Effect of NER Augmentation

We first compare the full system against two reduced inputs: (i) Prompt only, where MARBERT is fine-tuned on the raw prompt without the [ENTITIES] block, and (ii) Entities only, where the classifier sees only the serialized entity block. The first isolates the benefit of the entity cues over the contextual encoder alone; the second measures how much category information the extracted entities carry by themselves.

## 6.2 Effect of Entity Label Domains

To measure the contribution of each domain, we remove one domain at a time from the GLiNER label set (and from the serialized block), re-extract entities, and retrain. A large drop when a domain is removed indicates that its entity cues are important for the corresponding category, while a negligible change suggests redundancy with the encoder’s contextual representation.

## 6.3 Design Choices in Entity Extraction and Serialization

We further vary three design choices of the pipeline, changing one factor at a time from the full system:

• Serialization format. Our default lists all 34 labels in a fixed order with explicit none values. We compare it with a compact format that lists only labels with at least one detected entity. The compact format shortens the input, which matters under the 256-token limit, but removes the explicit absence signal.

• GLiNER confidence threshold. We vary the threshold in {0.3, 0.5, 0.7} to study the trade-off between recall of entity cues (low threshold) and noise from spurious entities.

• Maximum sequence length. The maximum length ∈ {128, 256, 512} to test whether truncation of the entity block affects performance.

## 6.4 Discussion

The experimental results demonstrate that the proposed pipeline achieves promising performance in fine-grained harmful content detection for Arabic user prompts. On the development set of 2,405 prompts, the system obtains a macro-F1 score of 0.886 and an accuracy of 0.897. The macro-averaged precision and recall are 0.885 and 0.888, respectively, indicating a relatively balanced classification performance across the seven categories. Since macro-F1 gives equal importance to each category, it provides a useful assessment in the presence of the class imbalance observed in the training data. Nevertheless, the overall scores conceal meaningful differences in performance between individual categories.

## 6.5 Category-Specific Performance

The system achieves its strongest performance on adult\_content, with an F1-score of 0.987. This result suggests that the linguistic cues associated with this category are comparatively distinguishable within the evaluated dataset. The hate\_speech category also achieves a high F1- score of 0.923, despite having the smallest support in the development set, with 135 instances. Similarly, self-harm and FraudandDeception obtain F1-scores of 0.902 and 0.896, respectively. These results indicate that the system can recognize several distinct types of harmful content with relatively high consistency.

In contrast, harm-to-others is the most challenging category, with an F1-score of 0.815, followed by Bully at 0.834 and harassment at 0.848. These differences suggest that classification difficulty is not determined solely by the number of available examples. Instead, the semantic relationship between categories and the contextual interpretation required to distinguish them appear to be important factors. In particular, categories involving threats, insults, violence, and physical injury may share similar lexical and entity-level cues, making their boundaries difficult to learn from surface information alone.

## 6.6 Analysis of Classification Errors

The confusion matrix provides further insight into the limitations of the proposed approach. The most notable confusion occurs between harm-to-others and self-harm. Specifically, 28 instances of harm-to-others are classified as self-harm, while 27 instances of self-harm are classified as harm-to-others. Both categories may contain references to physical injury, weapons, violent actions, or other forms of harm. However, distinguishing between them often requires identifying the intended target of the harmful action and interpreting the relationship between the entities and the surrounding context. Entity extraction can identify relevant expressions, but the presence of a particular entity does not necessarily establish whether the intended harm concerns the user or another person.

Another source of error is the confusion between Bully and harassment. The confusion matrix records 12 bullying instances classified as harassment and 14 harassment instances classified as bullying. This overlap is understandable because both categories can involve insults, threats, and offensive expressions. Although the entityaugmented representation explicitly identifies these cues, such information may not be sufficient to distinguish the communicative purpose, target, or situational context of an utterance. The model must therefore rely on contextual evidence beyond the presence of shared entities.

The harm-to-others category also receives incorrect predictions from several other categories, including Bully, FraudandDeception, and harassment, with 18 instances from each of these categories classified as harm-to-others. This pattern suggests that the category is associated with a broad range of potentially harmful expressions and may overlap with other forms of harmful behavior. In addition, seven bullying instances are classified as hate\_speech, which may reflect cases in which offensive behavior targets a protected group. These findings emphasize the importance of modeling semantic relationships and contextual intent when distinguishing closely related harmful-content categories.

By comparison, adult\_content exhibits relatively few classification errors, with only eight of its 657 development instances assigned to other categories. This comparatively clear separation suggests that some categories have more distinctive cues than others. Overall, the error analysis indicates that future improvements should focus particularly on categories whose definitions overlap semantically rather than treating all classification errors as equivalent.

## 6.7 Implications of Integrating NER with MARBERT

The proposed framework combines two complementary components: GLiNER identifies harm-related entities, while MARBERT performs contextual classification using the original prompt enriched with a structured entity representation. The entity extraction stage makes selected cues explicit, including references to violence, selfinjury, financial credentials, protected groups, and offensive expressions. The classification stage then processes these cues alongside the original text rather than relying exclusively on the extracted entities.

This design provides a systematic way to incorporate entity-level information into Arabic harmful-content classification. Retaining the original prompt is particularly important because isolated entities may not reveal the intent of a message. For example, mentioning a weapon or a harmful action does not necessarily indicate a request to cause harm; the surrounding context determines how the reference should be interpreted. Combining the original prompt with the extracted entity information allows the classifier to consider both the broader linguistic context and explicit entity cues.

However, the observed results do not, by themselves, establish that entity augmentation outperforms a MARBERT-only baseline. Such a conclusion requires a direct comparison with the prompt-only configuration under the same experimental conditions. The ablation study is intended to investigate this question, as well as the effects of entity-label domains, confidence thresholds, serialization formats, and maximum sequence length. The corresponding comparisons should be reported with complete numerical results before drawing firm conclusions about the individual contribution of each design choice.

## 7 Limitations and Future Directions

Several limitations should be considered when interpreting these findings. First, the evaluation discussed here is based on the ArGuard development set. Although it provides evidence of the system’s performance on the available evaluation data, further testing on the official test set and additional Arabic harmful-content datasets would be necessary to assess generalization across different sources and linguistic conditions.

Second, the performance differences between categories indicate that the system remains sensitive to semantic overlap. Future work could investigate methods that explicitly model the target and intent of harmful actions, incorporate relationships between extracted entities, or use category-specific training strategies to distinguish closely related labels. Such approaches may be particularly useful for separating harm-to-others from self-harm and distinguishing Bully from harassment.

Third, the quality of the enriched input depends on the entities detected by GLiNER. Missed entities may remove potentially useful evidence, while incorrectly extracted entities may introduce irrelevant information into the classifier’s input. The fixed serialization of 34 entity labels also increases the input length, making the representation sensitive to the selected sequencelength limit. Future experiments should therefore examine entity extraction errors, the impact of alternative serialization strategies, and the balance between additional entity information and input efficiency.

Finally, because the system addresses safetyrelated content, classification performance should not be interpreted as a complete measure of practical safety. Errors may have different consequences depending on the category and application. Future evaluations could therefore examine category-specific error costs, calibration, robustness to dialectal variation, and performance on prompts whose harmful intent is expressed indirectly or ambiguously.

In summary, the results support the feasibility of combining generalist entity extraction with an Arabic pretrained transformer for finegrained harmful-content classification. The system achieves strong overall development-set performance, while the error analysis identifies important challenges involving target identification, contextual intent, and overlapping category definitions. Addressing these challenges and completing the planned ablation comparisons are important steps toward establishing the effectiveness and broader applicability of the proposed framework.

## References

Muhammad Abdul-Mageed, AbdelRahim Elmadany, and El Moatez Billah Nagoudi. 2021. ARBERT & MARBERT: Deep bidirectional transformers for Arabic. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7088–7105, Online. Association for Computational Linguistics.

Firoj Alam, Md. Rafiul Biswas, Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Hamdy Mubarak, George Mikros, Abul Hasnat, and Wajdi Zaghouani. 2026. ArGuard shared task: Harmful content detection in Arabic memes and LLM prompts. In Proceedings of The Fourth Arabic Natural Language Processing Conference: Shared Tasks, pages 313–325, Budapest, Hungary. Association for Computational Linguistics.

Nsrin Ashraf, Mariam Labib, Tarek Elshishtawy, and Hamada Nayel. 2025. REGLAT at MAHED shared task: A hybrid ensemble-based system for Arabic hate speech detection. In Proceedings of The Third Arabic Natural Language Processing

Conference: Shared Tasks, pages 645–650, Suzhou, China. Association for Computational Linguistics.

Nsrin Ashraf, Hamada Nayel, and Mohamed Taha. 2022. A comparative study of machine learning approaches for rumors detection in covid-19 tweets. In 2022 2nd International Mobile, Intelligent, and Ubiquitous Computing Conference (MIUCC), pages 384–387.

Lamiaa Eldouby, Islam A. Moneim, and Hamada Nayel. 2026. A rule-based model with fuzzy matching for disease named entity recognition in arabic biomedical texts. Procedia Computer Science, 275:618–627. 7th International Conference on AI in Computational Linguistics.

Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Abul Hasnat, Md. Rafiul Biswas, Wajdi Zaghouani, and Firoj Alam. 2026. AHA-Memes: A fine-grained multimodal benchmark for understanding hate in arabic memes. arXiv preprint arXiv:2607.27393.

Hamada Nayel. 2020. NAYEL at SemEval-2020 task 12: TF/IDF-based approach for automatic offensive language detection in Arabic tweets. In Proceedings ofthe Fourteenth Workshop on Semantic Evaluation, pages 2086–2089, Barcelona (online). International Committee for Computational Linguistics.

Hamada Nayel, Eslam Amer, Aya Allam, and Hanya Abdallah. 2021. Machine learning-based model for sentiment and sarcasm detection. In Proceedings of the Sixth Arabic Natural Language Processing Workshop, pages 386–389, Kyiv, Ukraine (Virtual). Association for Computational Linguistics.

Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. 2024. GLiNER: Generalist model for named entity recognition using bidirectional transformer. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5364–5376, Mexico City, Mexico. Association for Computational Linguistics.