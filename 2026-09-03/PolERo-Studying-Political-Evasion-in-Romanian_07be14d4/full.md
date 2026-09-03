# PolERo: Studying Political Evasion in Romanian

Gabriel Stefan and Sergiu Nisioi   
Laboratory for Cybernetic Research and Good Life   
Human Language Technologies Research Center Faculty of Mathematics and Computer Science University of Bucharest gabrielstefan04@gmail.com sergiu.nisioi@unibuc.ro

## Abstract

Political evasion refers to responses that engage with a question while withholding the requested information. Recent NLP work frames political evasion as a classification task using a two-level taxonomy of response clarity and fine-grained evasion strategies. Existing work on response clarity and evasion classification is limited to English, leaving open whether the taxonomy and model behavior transfer across languages and political contexts. We introduce PolERo, a dataset of 3,574 human-annotated question-answer pairs extracted from official transcripts of five Romanian presidents. We evaluate multiple classification approaches on both datasets under matched conditions, including TF-IDF baselines, fine-tuned encoder models, a proposed sliding-window encoder, and zero/few-shot LLM prompting. We study crosslingual transfer through joint bilingual training and machine-translation-based data augmentation. Our results indicate that fine-tuned encoders are competitive, cross-lingual transfer is asymmetric, and ambivalent evasion categories involving pragmatic cues remain the main challenge across all model families.

## 1 Introduction

Political interviews often contain responses that are relevant to the interviewer’s questions while at the same time remaining non-responsive. Such responses may involve implicit answers, partial replies, topic shifts (Msagalla and Visser, 2021), deflections, or strategic ambiguity (Bach et al., 2025), making it difficult to determine whether the requested information was actually provided. In political science and discourse analysis, these phenomena are commonly studied under the notions of equivocation and evasion (Bavelas et al., 1988; Bull, 1994; Clayman, 2001; Bull, 2003; Rasiah, 2010; Romaniuk, 2013; Bull and Strawson, 2020).

Beyond political communication, response clarity is also relevant to broader NLP problems including dialogue understanding, question answering, and implicitly for creating better user-facing technologies. Despite recent progress in political discourse processing (Glavaš et al., 2019; Huguet Cabot et al., 2020; Afli et al., 2024), response clarity remains largely unexplored in NLP. Existing work has focused primarily on question answerability (Rajpurkar et al., 2018; Min et al., 2020), responder intent (Ferracane et al., 2021), or broader discourse phenomena in political speech (Subramanian et al., 2019; Reinig et al., 2024; Verma et al., 2026), while leaving open the problem of determining whether a response adequately addresses a question. Unlike traditional QA settings (Rajpurkar et al., 2018; Min et al., 2020; Clark et al., 2020a), response clarity requires reasoning about pragmatic alignment, modeling information sufficiency (Jain and Garimella, 2026), and discourse structure (Mann, 1984), extending beyond standard answerability formulations.

Recently, Thomas et al. (2024) introduced the first NLP dataset and taxonomy of 3,400 questionanswer (QA) pairs from the U.S. presidential interviews. The dataset served as a reference for SemEval-2026 Task 6: CLARITY (Thomas et al., 2026) a task that aims to detect and classify response ambiguity in political discourse. Evidence for response clarity classification currently remains restricted to U.S. English political interviews. Furthermore, response clarity is not a purely linguistic phenomenon; it is shaped by institutional context, interaction format, and political communication practices. Currently, no resource targets response clarity or evasion classification in Romanian political speech. To the best of our knowledge, such resources are also absent for non-English languages more broadly.

To address this, we introduce PolERo (Political Evasion in Romanian), a dataset of 3,574 humanannotated question-answer pairs extracted from official transcripts covering five Romanian presidents. PolERo adopts the exact annotation protocol and two-level taxonomy proposed by Thomas et al. (2024), enabling controlled comparison between English and Romanian political interviews.

Romanian is an Eastern Romance language with historically limited NLP resources. Recent efforts have significantly expanded this landscape, introducing general evaluation benchmarks and foundational corpora (Dumitrescu and Avram, 2020; Dumitrescu et al., 2021; Rogoz et al., 2026), alongside Romanian-adapted language models (Masala et al., 2020; Dumitrescu et al., 2020; Dima et al., 2024; Masala et al., 2024). Furthermore, Romanian political communication has attracted sustained interest in discourse studies and political linguistics with a focus on political opinion mining (Gîfu and Cristea, 2013; Vasilescu et al., 2024) and hate-speech discourse (Gheorghiu and Praisler, 2022; Dinu et al., 2025).

Nevertheless, despite this progress, no prior datasets or analyses target response clarity or evasion classification in any type of speech. This makes PolERo, to the best of our knowledge, the first such resource for a non-English language. This paper makes several contributions:

• The release of PolERo,<sup>1</sup> a human-annotated dataset for response clarity and evasion classification in a Romanian political corpus.

• Multiple classification results for both PolERo and the English dataset of Thomas et al. (2024) under matched conditions, covering encoder models, a multi-head encoder for long text classification, and different LLM prompting strategies.

• A cross-lingual transfer analysis examining the consistency of evasion patterns and model behavior across the two corpora, considering low-data scenarios and machine-translationbased data augmentation.

## 2 Dataset and Annotation

## 2.1 Data Collection and Preprocessing

We collect official transcripts from the Romanian presidential websites,<sup>2</sup> covering official press conferences, interviews, and declarations from five Romanian presidents: Ion Iliescu, Traian Basescu,˘ Klaus Iohannis, Ilie Bolojan, and Nicusor Dan. The corpus spans 25 years between 2001–2026.

In such contexts, it is common for a single interviewer question to contain multiple subquestions. The initial pool of extracted data contained 6,234 raw question-response candidates, comprising 8,233 sub-questions in total. For the current task, we enforce a one-to-one questionresponse mapping, and we use GPT-5.4 to flag the set of instances where more than one question is asked in a single instance. As such, we remove 2,053 multi-part entries and 607 entries containing no extractable question. During the annotation process, evaluators were instructed to flag any remaining double-barreled questions in the labeling platform (see the appendix B.1). No such instances were reported, indicating that no remaining multipart questions were present in the retained corpus. The retained corpus consists of 3,574 single-part QA pairs. We divide the dataset into a training set of 3,278 and a test set of 296 examples.

![](images/3ead264380365352385285b60f327575c3bfd982ec52f85ca28423eb6317acb6.jpg)  
Figure 1: Clarity-evasion two-level taxonomy mapping.

## 2.2 Taxonomy

PolERo adopts the two-level response clarity taxonomy of Thomas et al. (2024), focusing on the distinction between clarity and ambiguity in political responses (see Figure 1). The first level assigns each response one of three clarity labels: Clear Reply — the information is given in the requested form, Ambivalent — a valid answer is given but admits multiple interpretations, or Clear Non-Reply — the speaker openly refuses to share the information. The second level divides these classes into nine fine-grained evasion categories. Clear Reply contains only Explicit information stated in the requested form.

The Ambivalent category includes cases where:

1. Implicit information is conveyed without being explicitly stated; 2. General information is provided, lacking the requested specificity; 3. Partial half-answers are provided; 4. Dodging or ignoring the question altogether; and 5. Deflection of the topic by shifting focus to a different point. The final category, named Clear Non-Reply, includes Declining to Answer in an explicit manner while acknowledging the question, Ignorance when respondents admit they do not know the answer, and Clarification when respondents ask for more details.

An expert in linguistics, working professionally as an authorized translator, adapted the initial English taxonomy for Romanian, fixing category definitions and boundary cases. Appendix A reproduces the Romanian annotation guidelines.

## 2.3 Annotation Protocol

We build a screening quiz of 15 QA pairs spanning ambiguous evasion categories, where each item is labeled by an expert to serve as a benchmark reference. The quiz is used as a screening step to ensure that the three annotators understand the taxonomy and can consistently handle borderline and difficult cases. All annotators are native Romanian women aged 22–24 with undergraduate training in psychology.

Annotators complete a training phase that covers all nine categories of taxonomy (subsection 2.2) using worked examples. Each instance requires a single annotation decision: assigning one of the nine evasion labels to the response. The three-way clarity label is then inferred by traversing the taxonomy upward. The annotation process spanned five weeks. During this period, we held three group review sessions with the expert to resolve disagreements and clarify boundary cases. Mean annotation time was 69 seconds per instance, totaling approximately 27 hours per annotator. All annotators were compensated for their work (see section 9).

A single annotator labels each training instance, with the workload evenly distributed between the three. All three independently label the test split to support inter-annotator agreement measurement and evaluation using multiple annotations. For 2.4% of test instances where all three annotators assigned different evasion labels, the expert assigned the final label.

## 2.4 Worked Example

Î: “Domnule Pres<sub>,</sub> edinte, p˘adurea este o chestiune de sigurant<sub>,</sub> ˘a nat<sub>,</sub>ional˘a?”

R: “Eu cred c˘a p˘adurile sunt o chestiune de suflet nat<sub>,</sub>ional s<sub>,</sub> i de aceea sunt foarte încrez˘ator c˘a vom reusi s˘a schimb˘am lucrurile în bine.”

English:

Q: “Mr. President, areforests a matter ofnational security?”

A: “I believeforests are a matter ofnational spirit, and for that reason I am very confident that we will succeed in changing things for the better.”

The response receives clarity label Ambivalent and evasion label Deflection. The answer engages with the topic introduced in the question (forests as a national concern), but does not address whether forests constitute a national security issue. It reframes the discussion and shifts the attention away from the original premise without explicitly rejecting it.

![](images/0b5edd3ce0cec469fc2421ea16af2b095c88606a3ff103a8fc20a118d08cf9bf.jpg)  
Figure 2: Overall clarity distribution for PolERo compared to the English CLARITY dataset.

## 2.5 Dataset Statistics

Label distributions are similar across splits. In the training set, Clear Reply accounts for 54.9% of instances, Ambivalent for 37.7%, and Clear Non-Reply for 7.4%. The test set shows a comparable distribution (56.8%, 36.1%, 7.1%).

Figure 2 compares the overall clarity label distribution of PolERo with the English CLARITY dataset (Thomas et al., 2024). Clear Reply is the majority class in the Romanian dataset (55.09%), while the English corpus is primarily composed of Ambivalent responses (59.80%). Another point of departure between the two datasets is the way in which multi-part questions have been treated: in English, the interviews are decomposed into separate sub-questions using ChatGPT, while in Romanian, we discard such instances as we consider AI models insufficient for creating gold-standard data. An exploratory analysis is provided in the Appendix C. The analysis reveals variation in response clarity across speakers, communicative settings, and topics. Romanian presidential discourse is dominated by Explicit answers (55.09%), although this proportion decreases in more formal contexts such as declarations and press conferences. Evasion strategies also vary across presidents and topics: earlier speakers rely more heavily on Dodging, whereas post-2014 presidents exhibit more balanced distributions with higher rates of General, Deflection, and Declining to answer. Finally, the length of the answer aligns with established theories of verbose evasion (Bavelas et al., 1988; Rasiah, 2010; Bull and Strawson, 2020), with indirect strategies such as General, Deflection, and Partial responses tending to be substantially longer than direct responses.

## 2.6 Annotation Quality

To assess the performance of the three annotators, the expert independently labeled a benchmark set of 85 instances. Against this gold standard, the annotators achieved accuracies of 95.3%, 95.3%, and 94.1% on the clarity task, and 89.4%, 88.2%, and 84.7% on the evasion task (Appendix B.3).

Inter-annotator agreement on the test set, measured using Fleiss’s κ (Fleiss, 1971), is 0.843 for clarity and 0.678 for evasion, corresponding to near-perfect and substantial agreement on the scale of Landis and Koch (1977). At the category level, agreement is highest for Clarification (κ = 1.000), Claims Ignorance (κ = 0.881), and Explicit (κ = 0.858), and lowest for Deflection (κ = 0.364) and General (κ = 0.464). A detailed analysis of the annotation is provided in Appendix B.

## 3 Experimental Setup

## 3.1 Task Formulation

We report macro-averaged F1 on the test split of each dataset as the primary metric across all experiments. We evaluate the models on two classification tasks derived from the two-level taxonomy (subsection 2.2): (1) clarity classification, a three-class task over the coarse-grained labels; and (2) evasion classification, a nine-class task over the fine-grained labels. For the three-class clarity task, we also compute a macro-F1 score by deterministically (3) mapping each predicted evasion label to its parent clarity class. Thus, we can directly compare models trained specifically for the three-class task against those whose coarse-grained labels are inferred through the fine-grained objective.

![](images/b7e8ebc435a8665bc3ebd9e38e65b309e8fbae00b9aed6a93b46cbf3947eead5.jpg)  
Figure 3: Multi-head architecture. The tokenized concatenated input (Q ⊕ A) is split into overlapping chunks, each chunk is encoded by a shared encoder, chunk representations are aggregated via element-wise max-pooling, and two task-specific heads predict clarity (3 classes) and evasion (9 classes).

## 3.2 Models

We establish baselines across four model families on both the English CLARITY dataset and PolERo, training and evaluating each language independently. Input formats, hyperparameters, and prompt templates are reported in Appendix D.2. A TF-IDF logistic regression classifier serves as a non-neural lower bound. The model comparison covers single-head (SH) encoder fine-tuning, a custom multi-head (MH) encoder rendered in Figure 3, and LLMs under zero-shot and few-shot prompting without fine-tuning: GPT-5.4, Llama-3.3-70B-Instruct, Qwen3.6-35B-A3B, Gemma-4- 31B-it, DeepSeek-V4-Pro, and gpt-oss-120b.

Standard encoder configurations are finetuned independently for each task. In English, we evaluate RoBERTa-large (Liu et al., 2019), ModernBERTlarge (Warner et al., 2025), ELECTRA-large (Clark et al., 2020b), BERT-large-cased (Devlin et al., 2019), and XLM-RoBERTa-large (Conneau et al., 2020). The same five architectures are fine-tuned on PolERo, with the addition of RoBERT-large (Masala et al., 2020),<sup>3</sup> and Romanian BERT (Dumitrescu et al., 2020), both pretrained on Romanian corpora, and multilingual mBERT-base (Devlin et al., 2019).

The Multi-head (MH) encoder architecture contains a shared encoder and two linear heads that predict clarity and evasion jointly (Figure 3). To process long question-answer pairs without truncation, sequences are segmented into overlapping chunks and encoded independently. The chunk representations are aggregated via element-wise max-pooling into a single response vector, which is passed to a 3-class clarity head and a 9-class evasion head. The training objective is the unweighted sum of the two cross-entropy losses. Models are trained with 7-fold stratified cross-validation, and fold predictions are ensembled by probability averaging at inference time. Matching the single-head configurations, we train RoBERTa-large, XLM-RoBERTalarge (added to support the cross-lingual experiments below), and RoBERT-large, a Romanianonly encoder model. Full architectural details and mathematical formulations together with component ablations isolating the contribution of chunked pooling, joint supervision, and ensembling are reported in Appendix E.

## 4 Results and Discussion

The results in this work are reported on the English dataset using the development set from the SemEval 2026 CLARITY Shared Task, since the official test of the task was not publicly available at the time of writing. The multi-head RoBERTalarge achieves one of the strongest scores (0.8) on the SemEval 2026 CLARITY task (Stefan and Nisioi, 2026; Thomas et al., 2026) among encoderonly models, especially given the size of under 400 million parameters.<sup>4</sup> However, on the development set, the same model obtains a smaller .713 score for the 3-way clarity task prediction.

Table 1 reports the macro-F1 scores for the strongest models from each approach. A complete set of results for all models, cross-lingual setups, and prompting strategies is provided in Appendix F.

Traditional classifiers. TF-IDF + logistic regression reaches a macro-F1 .422/0.453 on English/Romanian 3-class clarity detection, confirming that surface-level lexical features alone are insufficient. Additional per-class error patterns are reported in Appendix D.

Monolingual encoder fine-tuning. RoBERTalarge is the strongest single-head English encoder model for the 3-class clarity (C) identification task (.580) and the 9-class fine-grained evasion (E) detection task (.481). The mapped (M) fine-grained task significantly outperforms (.656) the simple clarity identification task, showing that it is better to train on a more fine-grained taxonomy and merge than to train directly on fewer un-balanced classes.

Unsurprisingly, for Romanian, this model performs significantly worse than encoder models pre-trained on Romanian, such as RoBERT-large (Masala et al., 2020). The latter is the strongest single-head model on PolERo (.504 evasion, .701 mapped). More detailed encoder comparisons are available in Appendix F.

<table><tr><td rowspan="2">Model</td><td colspan="3">EN (CLARITY)</td><td colspan="3">RO (PolERo)</td></tr><tr><td>C</td><td>E</td><td>M</td><td>C</td><td>E</td><td>M</td></tr><tr><td>TF-IDF + LR</td><td>.422</td><td>.301</td><td>.441</td><td>.453</td><td>.312</td><td>.472</td></tr><tr><td>Best Encoders (Fine-Tuned) RoBERTa (SH)</td><td>.580</td><td>.481</td><td>.656</td><td>.646</td><td>.428</td><td>.613</td></tr><tr><td>RoBERTa (MH) XLM-R. (SH)</td><td>.713 .579</td><td>.568 .419</td><td>.731 .564</td><td>.696 .680</td><td>.501 .365</td><td>.652 .686</td></tr><tr><td>XLM-R. (MH)</td><td>.665</td><td>.475</td><td>.644</td><td>.694</td><td>.508</td><td>.706</td></tr><tr><td>RoBERT (SH)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RoBERT (MH)</td><td></td><td>一</td><td>一</td><td>.688</td><td>.504</td><td>.701</td></tr><tr><td>Best LLMs (Few-Shot Prompting)</td><td>1</td><td>一</td><td>一</td><td>.729</td><td>.545</td><td>.763</td></tr><tr><td>Llama-3.3-70B (few-shot)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-OSS-120B (few-shot)</td><td>.535</td><td>.382</td><td>.564</td><td>.555</td><td>.312</td><td>.411</td></tr><tr><td></td><td>.542</td><td>.554</td><td>.729</td><td>.553</td><td>.580</td><td>.729</td></tr><tr><td>DeepSeek-V4-Pro (few-shot)</td><td>.601</td><td>.545</td><td>.670</td><td>.627</td><td>.594</td><td>.751</td></tr><tr><td>Gemma-4-31B-it</td><td>.721</td><td>.583</td><td>.662</td><td>.672</td><td>.619</td><td>.738</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>.669</td><td>.544</td><td>.747</td><td>.746</td><td>.682</td><td>.760</td></tr><tr><td>GPT-5.4</td><td>.703</td><td>.626</td><td>.761</td><td>.763</td><td>.666</td><td>.763</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Summary of test-set macro-F1 for the highestperforming models across all tested approaches. C: Clarity (direct 3-class prediction). E: Evasion (direct 9- class prediction). M: Mapped (3-class clarity prediction obtained by mapping the predicted 9-class evasion label upward through the taxonomy). SH: Single-head, MH: Multi-head. Best scores are highlighted in bold. LLMs obtain the strongest results for the fine-grained Evasion prediction, while smaller-sized multi-head encoders remain competitive for predicting response clarity.

Multi-head fine-tuning. The custom multi-head model architecture improves performance across all metrics compared to the corresponding singlehead model on both datasets. Multi-head

RoBERTa-large is the strongest English encoder configuration, outperforming its single-head counterpart by 13 macro-F1 points on direct clarity and 9 on evasion, and surpassing GPT-5.4 on direct clarity prediction. The multi-head RoBERT-large improves over its single-head Romanian-specific counterpart by 4 points on clarity, 4 on evasion, and 6 on mapped clarity. These gains are consistent with the dual clarity+evasion tasks acting as a regularizer: gradients from the coarser clarity supervision help stabilize predictions for rare evasion categories, and vice-versa.

Multi-head XLM-RoBERTa-large obtains smaller gains in both languages than the languagespecialized backbones, indicating that multilingual pretraining sacrifices some within-language performance for broader cross-language coverage. As expected, models pretrained specifically for a given language consistently outperform multilingual models on that language: RoBERT-large achieves higher scores than XLM-RoBERTa-large on all Romanian evaluation metrics, while RoBERTalarge similarly outperforms XLM-RoBERTa-large on the English benchmarks.

<table><tr><td>Step</td><td>Component</td><td>Clarity</td><td>Evasion</td></tr><tr><td colspan="4">English (CLARITY), RoBERTa-large</td></tr><tr><td>SO</td><td>Single-head</td><td>.597±.006</td><td>.474±.024</td></tr><tr><td>S1</td><td>Multi-head</td><td>.638±.007</td><td>.478±.013</td></tr><tr><td>S2</td><td>Sliding-window</td><td>.648±.024</td><td>.481±.040</td></tr><tr><td>S3</td><td>Cross-validation</td><td>.662±.017</td><td>.513±.016</td></tr><tr><td>S4</td><td>Ensembling</td><td>.694±.009</td><td>.523±.013</td></tr><tr><td colspan="4">Gain over SO +.097</td></tr><tr><td>Romanian (PolERo), RoBERT-large</td><td></td><td></td><td></td></tr><tr><td>SO</td><td>Single-head</td><td>.678±.022</td><td>.511±.015</td></tr><tr><td>S1</td><td>Multi-head</td><td>.701±.022</td><td>.531±.030</td></tr><tr><td>S2</td><td>Sliding-window</td><td>.705±.017</td><td>.515±.019</td></tr><tr><td>S3</td><td>Cross-validation</td><td>.707±.015</td><td>.516±.046</td></tr><tr><td>S4</td><td>Ensembling</td><td>.721±.011</td><td>.537±.021</td></tr><tr><td colspan="4">Gain over SO +.043</td></tr></table>

Table 2: Cumulative ablation of the multi-head architecture.

Table 2 adds each component cumulatively to a single-head baseline, repeating the full encoder pipeline for both languages, with RoBERTa-large as the English backbone and RoBERT-large as the Romanian one. Each configuration is run over three random seeds (7, 42, 123); we report mean and standard deviation of test-set macro-F1 together with the increment over the preceding step. Relative to the single-head baseline, the complete architecture gains +0.097 macro-F1 on English clarity, +0.049 on English evasion, +0.043 on Romanian clarity and +0.026 on Romanian evasion. The joint multihead objective provides the largest single gain on clarity (+0.041 English, +0.023 Romanian), while cross-validation and ensembling account for most of the improvement on the evasion task (+0.032, +0.010 on English).

Predicting mapped vs. direct clarity. The twostep strategy of predicting the nine fine-grained evasion categories and then projecting upward through the taxonomy outperforms the direct clarity head for RoBERTa-large on English, for RoBERT-large on Romanian, and for nearly every LLM configuration on both datasets. This is consistent with the results reported at the Shared Task (Thomas et al., 2026), where the winning reverse derivation strategy proved to be the single most effective approach across the entire competition.

Four encoders do not benefit from this approach: mBERT-base, BERT-large-cased, ELECTRA-large, and ModernBERT-large. In each case, the evasion head assigns most predictions to Explicit and General, causing the projection step to overpredict Clear Reply at the expense of Ambivalent and Clear Non-Reply. ELECTRA-large on the Romanian split is the clearest example: the model predicts Explicit for every instance, yielding a macro-F1 of .084, and the mapped clarity score drops to .24 because Ambivalent and Clear Non-Reply are never predicted. ModernBERT-large shows the same pattern, though less severely, with two categories still receiving zero recall on Romanian (Claims ignorance and Partial/half-answer). The two-step strategy therefore depends on the backbone being able to assign a meaningful probability mass to rare categories. Otherwise, the projection step amplifies the model’s bias toward the majority class rather than correcting it.

LLM prompting. LLMs achieve some of the best results across all metrics on both datasets, though their ranking varies by task, and there are many cases when models perform significantly weaker than encoders (Table 1). Here we report only the strongest candidates; for a full set of results, see Appendix F.3 and Table 11. Gemma-4- 31B-it obtains the best few-shot scores on direct clarity prediction for English, Qwen3.6-35B-A3B achieves the best score on few-shot evasion for Romanian, and GPT-5.4 attains strong mapped clarity scores on both, although it is on-par with the multi-

head RoBERT model.

DeepSeek-V4-Pro and gpt-oss-120B fall between these models. Few-shot prompting improves the mapped metric for most models because the examples help clarify the boundaries between similar categories, especially Dodging-Deflection and Implicit-General. Under zero-shot prompting, these are the category pairs that models commonly confuse, and even one example per category substantially improves alignment. Llama-3.3-70B is the exception. Its mapped clarity score on English drops from .624 to .564 under few-shot prompting. An analysis of the outputs reveals that the model becomes heavily biased toward Deflection, predicting it for 79.5% of all English test instances, regardless of the gold label. The same holds for Romanian, where Deflection accounts for 44% of all predictions. In our experiments, the 70B Llama model obtained relatively weak results, which was unexpected given that it is one of the strongest baselines at the SemEval Task 6 CLARITY (Thomas et al., 2026).

Data contamination. The performance differences between LLMs and fine-tuned encoders likely stem from multiple factors. Firstly, LLMs benefit from broad parametric world knowledge stored in large billion-parameter-sized networks, which help to resolve entity-dependent evasion. Secondly, all evaluated LLMs (except Llama-3.3- 70B) are prompted with reasoning enabled, allowing for potentially better deliberation over pragmatic intent. Notably, Llama-3.3-70B, which lacks reasoning, is also the weakest LLM in both languages. Thirdly, we cannot rule out that these models might have been exposed to the data (which is publicly available) during their pre-training phase. To assess whether the advantage of LLMs is influenced by data contamination, we explore the CoDeC framework, Contamination Detection via Completion (Zawalski et al., 2026), on three opensource LLMs: Qwen3.6-35B-A3B, DeepSeek-V4- Pro, and Gemma-4-31B-it. CoDeC scores are extracted solely from the text, yielding contamination percentages on the English test set of 37%, 38%, and 45%; while on the Romanian test set, the corresponding scores are 43%, 41%, and 39%. All six values fall below the 80% threshold that would indicate a clear case of data contamination, according to the Zawalski et al. (2026). Even if the models could have been exposed to the same texts during their pre-training (this includes the encoders), the associated labels were not openly available; therefore, we could not find sufficient evidence for contamination.

## 5 Cross-Lingual Transfer

We assess cross-lingual generalization using the multi-head (MH) architecture and XLM-RoBERTalarge and RoBERTa-large, although the latter is reserved for Appendix, Table 11 since it is not designed to be a multilingual encoder.

Each model is trained in multiple cross-lingual configurations:

• EN → RO: trained on English, evaluated on Romanian.

• RO → EN: trained on Romanian, evaluated on English.

• EN + RO → RO and → EN: joint training on the concatenation of both datasets, evaluated only on Romanian or English.

The first two configurations explore the ability to generalize across the two languages and political spheres (U.S. vs. Romania), while the remaining configurations explore the possibility of increasing the predictive power using multilingual data augmentation. Results are available in Table 3.

Training on English data and predicting on Romanian yields a significant 6 to 7 point drop (.694 → .635) using both XLM- and RoBERTalarge models. Training in the opposite direction, in Romanian → English, results in a worse performance degradation of 11 to 14 points. In both scenarios, the performance degradation makes the MH models similar to other single-head models such as mBERT-base, BERT-large-cased, ELECTRAlarge, or ModernBERT (see Table 11 containing complete experimental results in the Appendix).

The model trained on Romanian strongly overpredicts Clear Reply, misclassifying 54% of English Ambivalent answers as Clear Reply, and similarly maps most English Implicit and Deflection responses to Explicit. These results indicate that cross-lingual predictions do not lead to catastrophic losses and that the task may benefit from additional multilingual data.

Multilingual augmentation. As shown in Table 3, combining the two datasets and predicting on each test set leads to improvements over individual training of the same model. One of the best configurations for Romanian is joint EN+RO training with XLM-RoBERTa-large, which achieves the highest Romanian evasion score among fine-tuned models (0.570). This may be the case because joint training regularizes rare categories: Romanian examples such as Clarification or Claims ignorance are reinforced by similar English instances, reducing the data scarcity that causes failures. For example, Declining to answer reaches perfect recall on the Romanian test split under joint training. Furthermore, joint training encourages the shared encoder to model evasion in a cross-lingual way rather than relying on language-specific lexical cues. The English dataset contains more Ambivalent evasive answers, where evasion is expressed through topic shifts and intent rather than specific words. When trained jointly, the model may transfer this ability to Romanian, improving its ability to detect evasion without clear lexical signals. As a result, Romanian evasion scores increase by 0.062 over the inlanguage XLM-RoBERTa-large baseline, whereas the English scores show an insignificant improvement (+0.008).

RoBERTa-large does not benefit in the same way (Table 11). Its tokenizer and pretraining are optimized for English, so adding Romanian data introduces more noise than useful signal, causing English evasion performance to drop from 0.568 to 0.503. Overall, the English performance drops consistently across all criteria when doing joint training.

## 5.1 Machine Translation

As a secondary analysis, we extend the crosslingual setup with machine-translated training and test data in order to reduce the biases of the multilingual encoders. All translations are generated with NLLB-200-distilled-600M (NLLB Team et al., 2022). Appendix F.5 reports a semi-automatic analysis of the quality estimation of the translations using XCOMET-XL (Guerreiro et al., 2024). The model indicates higher MT quality for RO→EN than EN→RO, although manual inspection shows that many EN→RO translations flagged as errors are acceptable, with pragmatic categories such as Deflection remaining the most error-prone in both translation directions.

Translating the English data into Romanian partially recovers cross-lingual clarity performance but provides limited gains for evasion detection, according to the results presented in Table 3. Categories such as Deflection and Dodging suffer the largest degradation under MT augmentation and also receive the lowest translation quality scores, whereas more formulaic categories such as Clarification and Declining to answer translate reliably. A transfer asymmetry between English and Romanian is visible — machine translating texts from Romanian to English degrades the results more than the other way around. We believe that differences in class distributions and different discourse structures, shaped by historical practices, geopolitical contexts, and topics of interest, make the Romanian data less useful for detecting evasion in U.S. English. MT-based training set augmentation provides inconclusive results compared to individual or joint multilingual training.

<table><tr><td rowspan="2">Model</td><td colspan="2">EN (CLARITY)</td><td rowspan="2"></td><td colspan="3">RO (PolERo)</td></tr><tr><td>C</td><td>E M</td><td>C</td><td>E</td><td>M</td></tr><tr><td colspan="7">Best encoders fine-tuned on the training set RoBERT-large (MH) 1 1</td></tr><tr><td>RoBERTa (MH)</td><td>.713</td><td>.568</td><td>.731</td><td>.729 .i</td><td>.545 .-</td><td>.763 .-</td></tr><tr><td colspan="7">Cross-lingual XLM-R multi-head models:</td></tr><tr><td>Individual Training Joint Training (EN+RO)</td><td>.665 .679</td><td>.475 .483</td><td>.644 .642</td><td>.694 .723</td><td>.508 .570</td><td>.706 .717</td></tr><tr><td colspan="7">Low-data scenario; only test set available</td></tr><tr><td>EN → RO  $\mathrm { E N } 2 \mathrm { R O }  \mathrm { R O }$ </td><td>一</td><td>一 一</td><td></td><td>.635 .665</td><td>.399 .412</td><td>.655 .706</td></tr><tr><td> $\mathbf { R O } \to \mathbf { E N }$ </td><td>.551</td><td>.398</td><td>.527</td><td>一</td><td>一</td><td>一</td></tr><tr><td> $\mathtt { R O 2 E N } \to \mathtt { E N }$ </td><td>.477</td><td>.358</td><td>.439</td><td>一</td><td>一</td><td>一</td></tr><tr><td colspan="7">MT augmentation</td></tr><tr><td> $\mathrm { R O } + \mathrm { E N } 2 \mathrm { R O }  \mathrm { R O }$   $\mathrm { E N } + \mathrm { R O } 2 \mathrm { E N }  \mathrm { E N }$ </td><td>.694</td><td>.477</td><td>.602</td><td>.718 一</td><td>.489 一</td><td>.684 一</td></tr><tr><td colspan="7">Best LLMs (Few-Shot Prompting)</td></tr><tr><td>Gemma-4-31B-it</td><td>.721</td><td>.583</td><td>.662</td><td>.672</td><td>.619</td><td>.738</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>.669</td><td>.544</td><td>.747</td><td>.746</td><td>.682</td><td>.760</td></tr><tr><td>GPT-5.4</td><td>.703</td><td>.626</td><td>.761</td><td>.763</td><td>.666</td><td>.763</td></tr></table>

Table 3: The first two and the last three rows contain the best models for each language and task. The arrow → shows the training and testing direction of the XLM-R, multi-head models. The labels EN2RO and RO2EN indicate machine translated training sets. In low-data scenarios it is best to use few shot prompting and LLMs. When data is available, joint multi-lingual training achieves better results than monolingual training using machine translated data.

## 6 Remarks

Based on our Romanian case study, we propose the following takeaways to guide future assessments of political evasion in new languages:

1. In low-resource settings, when creating a dataset for a new language such as Romanian, it is advisable to first benchmark the dataset using few-shot prompting with available LLMs, preferably open-weight models.

2. If LLMs perform poorly for a particular language, translating existing datasets with NLLB-200 or similar tools and training crosslingual encoders may be a viable alternative, although factors related to data representativeness (Dogruöz et al.˘ , 2023), geopolitical factors, and local histories must be carefully taken into account.

3. If a sufficiently large training set can be constructed for the target language (i.e., at least 3,000 samples) and existing monolingual pretrained encoders are available, then a strong strategy is to use the multi-head approach described in Section 3; otherwise, if monolingual encoders are not available, then joint multilingual training may be more beneficial than augmenting the data with machine-translated examples.

4. The strong performance of LLMs must be weighed against cost and efficiency; our proposed multi-head encoder proved competitive with LLMs on the clarity classification task while using fewer than 500M parameters; at the same time, the encoders have limitations in terms of fine-grained evasion detection.

5. The distribution of ambiguous responses can have a significant effect on the generalizability of cross-lingual models.

## 7 Conclusions

In this work, we introduce PolERo, a novel Romanian dataset of 3,574 question-answer pairs from presidential transcripts annotated using a two-level taxonomy of evasion. We compare multiple models on an existing English dataset and on our proposed Romanian dataset under matched conditions, covering traditional classifiers, standard encoder finetuning, a custom multi-head architecture for joint clarity and evasion prediction, and large language models under zero-shot and few-shot prompting.

Our results only partially corroborate previous findings (Thomas et al., 2026) that prompting modern LLMs strongly outperforms fine-tuned encoders. We show that highly competitive results can be obtained on a language such as Romanian using encoder models only. We propose a slidingwindow and max-pooling approach with two classification heads that significantly increases the predictive power of encoder models, making them competitive with LLMs for the three-class clarity classification task while using only a fraction of the memory footprint. However, for the nine-class fine-grained evasion task, larger models achieve better predictive performance.

Finally, we investigate cross-lingual transfer from the perspective of multilingual training versus monolingual training with machine-translated texts. Our results reveal an asymmetric relation: Englishtrained models generalize better to Romanian than the other way around. Joint multilingual training yields good cross-lingual performance for Romanian, while ambivalent categories remain the main challenge across all model families, as they depend on pragmatic intent rather than surface form.

## 8 Limitations

PolERo covers a single non-English language. Whether the taxonomy, annotation difficulty, and model behavior observed here generalize to other languages and political contexts remains an open question.

The annotation protocol assigns a single annotator per training instance, with triple annotation reserved for the test split. Evasion classification requires interpreting communicative intent, which can vary across annotators. Our annotators have undergraduate training in psychology rather than political science or linguistics, which may affect interpretation of borderline cases. Full triple annotation with domain experts would likely reduce this uncertainty, but at substantially higher cost.

## 9 Ethical Considerations

Source data consists of official transcripts published on the public archives of the Romanian presidency. The speakers are public officials whose statements were issued in their official capacity; no private communications, personal data, or nonpublic records are included.

The three annotators were each paid the equivalent of 80 EUR for approximately 27 hours of labeling work distributed over five weeks. The linguistic expert contributed voluntarily. Before participation, annotators were informed of the task purpose, the political nature of the content, and the option to withdraw at any point without penalty.

Clarity and evasion classification of political speech can be used to selectively frame or attack individual speakers. We release PolERo for research on response clarity in political discourse and discourage its deployment as a standalone instrument of public accountability. Model predictions are not ground truth: as the inter-annotator agreement statistics show, even trained human annotators disagree on a fraction of instances. Automated labels should not be treated as definitive characterizations of any individual speaker’s behavior.

Finally, the computational cost associated with the experimental pipeline presented in this work, covering fine-tuning and inference experiments, was conducted on a single NVIDIA H100 80GB GPU, with a total estimated budget of approximately 180 GPU hours. AI was used to check grammar and to generate boilerplate code.

## Acknowledgments

We would like to thank the reviewers for their helpful comments, and in particular, reviewer NvEV for their creative suggestions on how this work can be used as a political tool.

This research is partially supported by the project “Romanian Hub for Artificial Intelligence - HRIA”, Smart Growth, Digitization and Financial Instruments Program, 2021-2027, MySMIS no. 351416, and by InstRead: Research Instruments for Text Complexity, Simplification, and Readability Assessment CNCS - UEFISCDI project number PN-IV-P2-2.1-TE-2023-2007.

## References

Haithem Afli, Houda Bouamor, Cristina Blasi Casagran, and Sahar Ghannay, editors. 2024. Proceedings of the Second Workshop on Natural Language Processing for Political Sciences @ LREC-COLING 2024. ELRA and ICCL, Torino, Italia.

Parker Bach, Carolyn E Schmitt, and Shannon C Mc-Gregor. 2025. Let me be perfectly unclear: strategic ambiguity in political communication. Communication Theory, 35(2):96–106.

Janet Beavin Bavelas, Alex Black, Lisa Bryson, and Jennifer Mullett. 1988. Political equivocation: A situational explanation. Journal of Language and Social psychology, 7(2):137–145.

Peter Bull. 1994. On identifying questions, replies, and non-replies in political interviews. Journal of language and social psychology, 13(2):115–131.

Peter Bull. 2003. The Microanalysis ofPolitical Communication: Claptrap and Ambiguity. Routledge, London and New York.

Peter Bull and Will Strawson. 2020. Can’t answer? Won’t answer? An analysis of equivocal responses by Theresa May in Prime Minister’s Questions. Parliamentary Affairs, 73(2):429–449.

Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020a. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. Transactions ofthe Association for Computational Linguistics, 8:454– 470.

Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020b. Electra: Pretraining text encoders as discriminators rather than generators. Preprint, arXiv:2003.10555.

Steven E Clayman. 2001. Answers and evasions. Language in society, 30(3):403–442.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

George-Andrei Dima, Andrei-Marius Avram, Cristian-George Craciun, and Dumitru-Clementin Cercel. 2024. RoQLlama: A lightweight Romanian adapted language model. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4531–4541, Miami, Florida, USA. Association for Computational Linguistics.

Anca Dinu, Andreea C. Moldovan, and Adina Marincea. 2025. AntiSemRO: Studying the Romanian expression of antisemitism. In Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing - Natural Language Processing in the Generative AI Era, pages 291–298, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.

A. Seza Dogruöz, Sunayana Sitaram, and Zheng Xin˘ Yong. 2023. Representativeness as a forgotten lesson for multilingual and code-switched data collection and preparation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5751–5767, Singapore. Association for Computational Linguistics.

Stefan Dumitrescu, Andrei-Marius Avram, and Sampo Pyysalo. 2020. The birth of Romanian BERT. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4324–4328, Online. Association for Computational Linguistics.

Stefan Daniel Dumitrescu and Andrei-Marius Avram. 2020. Introducing RONEC - the Romanian named entity corpus. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4436–4443, Marseille, France. European Language Resources Association.

Stefan Daniel Dumitrescu, Petru Rebeja, Beata Lorincz, Mihaela Gaman, Andrei Avram, Mihai Ilie, Andrei Pruteanu, Adriana Stan, Lorena Rosia, Cristina Iacobescu, Luciana Morogan, George Dima, Gabriel Marchidan, Traian Rebedea, Madalina Chitez, Dani Yogatama, Sebastian Ruder, Radu Tudor Ionescu, Razvan Pascanu, and Viorica Patraucean. 2021. Liro: Benchmark and leaderboard for romanian language tasks. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

Elisa Ferracane, Greg Durrett, Junyi Jessy Li, and Katrin Erk. 2021. Did they answer? subjective acts and intents in conversational discourse. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1626–1644, Online. Association for Computational Linguistics.

Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological bulletin, 76(5):378–382.

Oana Celia Gheorghiu and Alexandru Praisler. 2022. Hate speech revisited in Romanian political discourse: from the Legion of the Archangel Michael (1927–1941) to AUR (2020–present day). Humanities and social sciences communications, 9(1):235.

Daniela Gîfu and Dan Cristea. 2013. Towards an automated semiotic analysis of the romanian political discourse. Computer Science Journal of Moldova, 61(1):36–64.

Goran Glavaš, Federico Nanni, and Simone Paolo Ponzetto. 2019. Computational analysis of political texts: Bridging research efforts across communities. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics: Tutorial Abstracts, pages 18–23, Florence, Italy. Association for Computational Linguistics.

Nuno M. Guerreiro, Ricardo Rei, Daan van Stigt, Luisa Coheur, Pierre Colombo, and André F. T. Martins. 2024. xCOMET: Transparent machine translation evaluation through fine-grained error detection. Transactions of the Association for Computational Linguistics, 12:979–995.

Pere-Lluís Huguet Cabot, Verna Dankers, David Abadi, Agneta Fischer, and Ekaterina Shutova. 2020. The Pragmatics behind Politics: Modelling Metaphor, Framing and Emotion in Political Discourse. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 4479–4488, Online. Association for Computational Linguistics.

Akriti Jain and Aparna Garimella. 2026. Knowing what’s missing: Assessing information sufficiency in question answering. In Findings of the Association for Computational Linguistics: EACL 2026, pages 4163–4174, Rabat, Morocco. Association for Computational Linguistics.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020. Focal loss for dense object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2):318–327.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. Preprint, arXiv:1907.11692.

William C. Mann. 1984. Discourse structures for text generation. In 10th International Conference on Computational Linguistics and 22nd Annual Meeting ofthe Associationfor Computational Linguistics, pages 367–375, Stanford, California, USA. Association for Computational Linguistics.

Mihai Masala, Denis C. Ilie-Ablachim, Dragos Corlatescu, Miruna Zavelca, Marius Leordeanu, Horia Velicu, Marius Popescu, Mihai Dascalu, and Traian Rebedea. 2024. Openllm-ro – technical report on open-source romanian llms. Preprint, arXiv:2405.07703.

Mihai Masala, Stefan Ruseti, and Mihai Dascalu. 2020. RoBERT – a Romanian BERT model. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6626–6637, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020. AmbigQA: Answering ambiguous open-domain questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5783– 5797, Online. Association for Computational Linguistics.

Brighton Msagalla and Marianna Visser. 2021. Agendasetting through topic shift in Tanzanian parliamentary debate: The derailment of strategic manoeuvring. Southern African Linguistics and Applied Language Studies, 39(4):327–337.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, and 20 others. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia. Association for Computational Linguistics.

Parameswary Rasiah. 2010. A framework for the systematic analysis of evasion in parliamentary discourse. Journal ofPragmatics, 42(3):664–680.

Ines Reinig, Ines Rehbein, and Simone Paolo Ponzetto. 2024. How to do politics with words: Investigating speech acts in parliamentary debates. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 8287– 8300, Torino, Italia. ELRA and ICCL.

Ana-Cristina Rogoz, Radu Tudor Ionescu, Alexandra-Valentina Anghel, Ionut-Lucian Antone-Iordache, Simona Coniac, and Andreea Iuliana Ionescu. 2026. A large-scale benchmark for evaluating large language models on medical question answering in romanian. Preprint, arXiv:2508.16390.

Tanya Romaniuk. 2013. Pursuing answers to questions in broadcast journalism. Research on Language & Social Interaction, 46(2):144–164.

Gabriel Stefan and Sergiu Nisioi. 2026. SG-UniBuc-NLP at SemEval-2026 task 6: Multi-head RoBERTa with chunking for long-context evasion detection. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 964–972, San Diego, California, USA. Association for Computational Linguistics.

Shivashankar Subramanian, Trevor Cohn, and Timothy Baldwin. 2019. Target based speech act classification in political campaign text. In Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (\*SEM 2019), pages 273–282, Minneapolis, Minnesota. Association for Computational Linguistics.

Konstantinos Thomas, Giorgos Filandrianos, Maria Lymperaiou, Chrysoula Zerva, and Giorgos Stamou. 2024. “I Never Said That”: A dataset, taxonomy and baselines on response clarity classification. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5204–5233, Miami, Florida, USA. Association for Computational Linguistics.

Konstantinos Thomas, Giorgos Filandrianos, Maria Lymperaiou, Chrysoula Zerva, and Giorgos Stamou. 2026. SemEval-2026 task 6: CLARITY – unmasking political question evasions. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 3704–3715, San Diego, California, USA. Association for Computational Linguistics.

Andra Vasilescu, Mihaela-Viorica Constantinescu, Ari adna Stefanescu, and˘ Serban Hartular. 2024. Insights

into Romanian political discourse. Cambridge Scholars Publishing.

Bhuvanesh Verma, Mounika Marreddy, and Alexander Mehler. 2026. Predicting convincingness in political speech: How emotional tone shapes persuasive strength. In The Proceedings for the 15th Workshop on Computational Approaches to Subjectivity, Sentiment Social Media Analysis (WASSA 2026), pages 37–51, Rabat, Morocco. Association for Computational Linguistics.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Griffin Thomas Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2526–2547, Vienna, Austria. Association for Computational Linguistics.

Michal Zawalski, Meriem Boubdir, Klaudia Bał azy, Besmira Nushi, and Pablo Ribalta. 2026. Detecting data contamination in llms via in-context learning. In International Conference on Learning Representations, volume 2026, pages 1337–1371.

## A Taxonomy and Annotation Guidelines

## A.1 Romanian Annotation Guidelines

Label names. We retained the nine original English label names (Explicit, Implicit, General, Partial/half-answer, Dodging, Deflection, Declining to answer, Claims ignorance, Clarification) to ensure our dataset remains directly comparable with the English corpus.

Descriptions and examples. We translated all category descriptions into Romanian and replaced the English examples of Thomas et al. (2024) with Romanian examples drawn from political discourse data in our dataset. Each example pairs an answer (R) with a question (Î) and a short rationale identifying the features that trigger the label.

A Romanian linguistic expert assisted with example selection to ensure that each category is illustrated by a clear and representative instance. Table 10 reproduces the guidelines, with English descriptions and bilingual examples (Romanian as supplied to annotators; English translation in italics for readability).

## A.2 Summaries and Quality Control

Annotators were required to classify the evasion strategy based on the original text. In addition, the annotators received GPT-5.4-generated summaries and were asked to verify the validity of the summaries. To ensure that the annotators were reading the source text, we inserted 60 intentionally incorrect summaries. The annotators correctly flagged 58 of these (96.67%), producing only two false negatives. None of the other summaries were flagged as inconsistent.

## B Annotation Protocol

## B.1 Labeling Platform

We used Labelbox<sup>5</sup> to manage the annotation workflow. Each item displayed the original interview question and answer verbatim, with the GPT-5.4- generated question and answer summaries attached as side panels for reference (Figure 4). The annotator completed two tasks per item: (i) assign one of the nine evasion labels from the Level-2 taxonomy, and (ii) verify whether the answer summary correctly reflects the original.

The platform provided the complete annotation guidelines, which included the taxonomy table (Table 10) and an extended set of examples for bordercase examples covering categories with closely related semantics (e.g. Dodging vs. Deflection, General vs. Partial/half-answer). Annotators could consult the guidelines at any point during a labeling session.

Annotators could additionally tag any instance with one of five issue categories:

• Parsing Error — the QA pair was malformed during extraction.

• Not a Question — the journalist’s statement does not solicit any information.

• Multiple Questions — a single journalist’s question contains several sub-questions.

• Insufficient Context — the response cannot be accurately classified without additional background information.

• Borderline / Hard to Decide — the response is difficult to classify and may fall between two evasion categories.

Annotators could also create custom issue categories to report any additional problems not covered by the predefined taxonomy.

Items tagged with the first three issues were filtered from the released corpus. Items tagged Insufficient

Context or Borderline / Hard to Decide were escalated to the expert, who either adjudicated a final label or removed the instance.

![](images/882a187b8ad275a99317641b569cbe62214151c9267b94d9295a9bf0559760df.jpg)  
Figure 4: Labelbox annotation interface. Left panel: evasion classification and summary validity check. Center: original question and answer. Right panel: GPT-5.4-generated question and answer summaries as attachments.

## B.2 Inter-Annotator Agreement

Raw annotator agreement on the test set is high, especially for clarity. All three annotators agree on 87.2% of clarity items, while the remaining 12.8% receive a two-thirds majority, leaving no items without consensus. Agreement on evasion is slightly lower but still substantial: 69.6% of items receive full three-way agreement, 28.0% receive a twothirds majority, and only 2.4% result in complete disagreement between all three annotators.

Figure 5 shows pairwise Fleiss’ κ across the three clarity labels. All pairwise values are at least 0.85, with the lowest agreement at the Clear Reply-Ambivalent boundary (κ = 0.85). Disagreement is mainly found between Clear Reply and Ambivalent answers rather than between the two non-ambiguous labels (Clear Reply vs. Clear Non-Reply: κ = 0.98).

![](images/ea18753efe8a4bb000c85a413a58bf51d0a773a316f155502dd9d4c0925e7225.jpg)  
Figure 5: Pairwise Fleiss’ κ between clarity categories.

Figure 6 shows the same matrix for the nine evasion categories. The lowest pairwise κ values occur for Deflection, in particular Deflection-Dodging $( \kappa = 0 . 2 6 )$ and Deflection-General $( \kappa = 0 . 3 8 )$ . These confusions reflect how close the categories are in the taxonomy. Dodging and Deflection both involve leaving the question unanswered. They differ in whether the answer acknowledges the question before shifting away. General and Deflection differ in whether the answer remains on topic at all, which depends on how broadly the topic is interpreted.

![](images/06b17187315446d43412e4f1faa046017f93534ccaf2b9d155fb8b6c5b2e4d8e.jpg)  
Figure 6: Pairwise Fleiss’ κ between evasion categories.

Figure 7 shows the per-category one-vs-rest Fleiss’ κ. Six categories cross the substantialagreement threshold of $\kappa ~ = ~ 0 . 6$ (Clarification, Claims ignorance, Explicit, Declining to answer, Partial/half-answer, Dodging); two fall between fair and substantial (Implicit, General); one falls in the fair range (Deflection).

![](images/64427e5fd85f2cd75b820bf47b4cb0be7b0786381e7bfdfd4b0b8f45f1385a03.jpg)  
Figure 7: One-vs-rest Fleiss’ κ per evasion category.

## B.3 Expert Benchmark Evaluation

The linguistic expert independently labeled a benchmark of 85 instances drawn from the test split.

Figure 8 reports per-annotator accuracy against the expert: 95.3%, 95.3%, 94.1% on clarity and 89.4%, 88.2%, 84.7% on evasion. Clarity accuracy is 5–10 percentage points higher than evasion accuracy for each annotator, which is consistent with the greater difficulty of the nine-class task compared to the three-class task.

![](images/fd6526188c44567c7c077e077be2ff8bf60d4edd41993029ad7259f6d1229223.jpg)  
Figure 8: Per-annotator clarity and evasion accuracy against expert benchmark.

## C Exploratory Data Analysis

This appendix reports descriptive statistics over the 3,574 question-answer pairs in the dataset (3,278 train; 296 test). The statistics are computed over the full dataset and use the majority-vote labels on the test set for both clarity and evasion categories.

## C.1 Label Distribution

Figures 9 and 10 show the label distributions for clarity and evasion. Explicit answers make up the majority of the dataset (55.09%). The other eight evasion strategies are much less common, with Clarification occurring in only 1.09% of the question-answer pairs.

![](images/b9fa7c494f0ad4d5f15f9d89843794c91589ef115e5a2600cbabc1fe442e60a5.jpg)  
Figure 9: Overall label distribution for Level 1 (Clarity).

![](images/0ce59ae703cd4bd2ec66254adc985d73fa265318123be129fcfa16e929ecb2ca.jpg)  
Figure 10: Overall label distribution for Level 2 (Evasion).

CLARITY splits multi-part interviewer turns into one labeled row per sub-question using Chat-GPT (Thomas et al., 2024), whereas PolERo removes them entirely. In CLARITY, 67.4% of instances are such fragments, and they are labeled Ambivalent far more often than single-question instances (63.7% vs. 51.7%). When limiting CLAR-ITY to examples with a single question (N = 1,225): Ambivalent falls from 59.8% to 51.7% and Clear Reply rises from 30.1% to 36.1% (Figure 11). The difference between the two distributions shrinks minimally.

![](images/8facd515b12896727cb414389cef61dc2bc99b70a47b4209bf924a058431f94f.jpg)  
Figure 11: Clarity distribution of the English CLARITY corpus before and after restricting it to single-question instances, compared with PolERo.

## C.2 Distribution by Speaker

The corpus covers five Romanian presidents (Figure 12). Coverage reflects data availability: Bas-˘ escu (1,214) and Iohannis (1,100) account for 64.8% of the corpus; Bolojan (56) is much less frequent, as he has been an interim president for 103 days.

Figure 13 shows the clarity rates for each speaker. Iliescu and Basescu give the most direct answers,˘ with Clear Reply rates of 62.06% and 62.60%. In contrast, Bolojan gives the most Ambivalent answers (53.57%), and Iohannis has the highest rate of Clear Non-Replies (11.55%).

![](images/c4fcdf805377c6a169e3fc3c0ca71993aee369ca18c3b4f76c4ed687921825da.jpg)  
Figure 12: Total number of question-answer pairs per speaker.

![](images/2bc9b0857e3a80a6055cd598ed1bc6949f8cbc4540d5409c1129e72a9aa4a122.jpg)  
Figure 13: Clarity label distribution per speaker.

Evasion distributions per speaker are shown in Figure 14. Dodging is most frequent for Iliescu (11.87%) and Basescu (9.06%).˘ Deflection is most common for Bolojan (12.50%) and Dan (10.08%). Declining to answer is highest for Iohannis (9.09%) and Bolojan (8.93%). The post-2014 speakers (Iohannis, Bolojan, and Dan) show a more balanced distribution of evasion types, with higher proportions of General, Deflection, and Declining to answer responses. In contrast, Iliescu and Basescu˘ rely more heavily on Dodging, while still remaining largely dominated by Explicit answers.

## C.3 Distribution by Communicative Context

We assigned a context label to each source transcript from event titles using keywords associated with each category. Interviews make up the majority of the dataset (1,904 pairs), followed by declarations (1,213 pairs) and press conferences (457 pairs).

Figure 15 reports clarity rates across the three resulting contexts. Interviews have the highest rate of Clear Replies (63.03%) and the lowest rate of

![](images/4643e4a4eefac1dcb768516890b59a9e51ce51ce8c66225c4e034aaf1be6a455.jpg)  
Figure 14: Evasion label distribution per speaker.

Clear Non-Replies (3.62%). Declarations and press conferences show very similar clarity distributions.

![](images/86a3958982f2fa52a17a9fdecf38251d42d8f82ff80714cd7954c682065d5344.jpg)  
Figure 15: Clarity distribution per communicative context.

Part of the higher clarity rate in interviews is linked to differences in topic distribution. Personal & Others makes up 38.39% of interview questions, compared to only 12.94% in declarations and 8.75% in press conferences (see Section C.4). This category also has the highest Clear Reply rate of any topic (65.62%). Press conferences contain more Justice & Anti-corruption and Domestic Policy questions, which have the lowest Clear Reply rates.

Evasion distributions per context are shown in Figure 16. Interviews show a higher percentage of Dodging (10.45%), but low levels of the other evasion types. Declarations and press conferences have roughly four to six times the rate of Declining to answer (8.41% and 6.13%) and higher Deflection (7.50% and 10.72%) compared to interviews (1.42% and 3.31%). Press conferences have the highest Deflection rate overall. If we exclude interviews and consider only the more formal settings of declarations and press conferences (n=1,670), the overall proportion of direct answers drops significantly: the Explicit rate falls to 46.05%, while evasive categories such as Deflection (8.38%) and Declining to answer (7.79%) account for a larger share of responses.

![](images/1bcfab633a59919156f103e483b358abbddbb9916c2e602ad2a36c5ef636c34f.jpg)  
Figure 16: Evasion label distribution per communicative context.

Figure 17 crosses speaker with context. Iliescu and Basescu interviews (˘ n=754, n=1,150) explain the high interview Explicit rate (62.47%, 63.39%) and also account for most Dodging (12.47%, 9.13%) in interviews. Iohannis declarations (n=852) account for the bulk of Declining to answer in the declaration row (9.74%). Bas-˘ escu declarations show the highest General share (23.80%) and Bolojan press conferences (n=36) the highest combined non-Explicit share (72.22%).

![](images/dc921a0427302f9e650689bfc02107b3e873131106e4e6ca765a859981d4c152.jpg)  
Figure 17: Evasion distribution per speaker × context cell.

## C.4 Distribution by Topic Category

Categories were defined by first inspecting the question content and identifying nine main topics. GPT-5.4 then assigned a single category label to each question. Figure 18 shows the overall distribution of these topics.

Figure 19 shows the clarity rates for each topic.

![](images/0015f30ae5d314524b7c334c8ecc4576f54dd0f2378a566727711290f7dbc4ff.jpg)  
Figure 18: Total number of question-answer pairs per topic category.

Even though Justice & Anti-corruption questions make up only 7.81% of the data, they trigger the highest rate of Clear Non-Replies (15.05%).

![](images/0ce48740a207ce2260f5bd4c7f35fd3982015d1b7b1261c46664d32a34383fad.jpg)  
Figure 19: Clarity label distribution per topic category.

Per-category evasion distributions are shown in Figure 20. Personal & Others has the highest Dodging share (11.21%), although most responses in this category are still Explicit. Justice & Anticorruption has the highest rates of Declining to answer (9.32%) and Claims ignorance (5.02%). Education & Research shows the highest share of Implicit (11.76%), followed by Society & Social Issues (11.40%). General responses are most frequent for Society & Social Issues (18.42%) and Public Health (18.10%).

## C.5 Temporal Coverage

QA pairs span 2001–2026 in two main periods: 2001–2007 (Iliescu and Basescu,˘ 1,935 pairs) and 2015–2026 (Iohannis, Bolojan, and Dan, 1,543 pairs). The intervening 2008–2014 window contains 96 pairs, due to limited archival availability. Per-year volumes and per-speaker timelines are shown in Figures 21 and 22.

Year-level clarity composition is shown in Figure 23. Clear Non-Reply rates remain below 6% in 2001–2007, and increase to around 10–20% in 2016–2026, with Ambivalent responses increasing in parallel.

![](images/be32595d25bfac02e857a60355e592d4b75b759c54b70120290dd839b026005a.jpg)  
Figure 20: Evasion label distribution per topic category.

![](images/ea70aca6a47614c80b78efe3c8010ed58c5f899a44cb97b57e62ee7ab3d58c63.jpg)  
Figure 21: Yearly volume of QA pairs, 2001–2026.

## C.6 Length and Verbosity

Mean question length is 30.7 words, while the median is 21 words; mean answer length is 76.9 words, while the median is 45 words. Overall length distributions are shown in Figure 24.

Answer length differs substantially by label (Figures 25 and 26). Clear Replies average 88.4 words, Ambivalent answers 67.4, and Clear Non-Replies 39.7.

Looking at the specific evasion strategies, Clarification answers are the shortest (mean 6.4, median 4) followed by Dodging (mean 27.6, median 13). In contrast, General, Deflection, and Partial/halfanswer responses are the longest non-Explicit categories (averaging 89.7, 88.6, and 87.3 words). This aligns with verbose evasion tactics, where politicians surround the question with extra information instead of answering it directly.

Per-speaker mean answer length ranges from 52.9 words (Dan) to 110.6 (Bolojan, n=56), with Iohannis at 73.1, Basescu at˘ 69.3 and Iliescu at 102.6. The full per-speaker answer-length distribution is shown in Figure 27. Speaker verbosity does not align with evasion: Iliescu is among the most verbose speakers and also one of the most Explicit, while Dan is the most concise and is also predominantly Explicit.

![](images/d68a59004f81ac3b25a0a769c74923fe18f23043382bf0daa248b52d968d8653.jpg)  
Figure 22: Yearly QA volume per speaker.

![](images/dffd0a6b1c0a8497d287f83e93473c96e583861e5589401065002144dc9c8409.jpg)  
Figure 23: Clarity label composition per year, 2001– 2026.

## D Experiment Details

This appendix expands the experimental setup and analysis summarised in Sections 3 and 4.

## D.1 Evaluation

Across all experiments, we evaluate model performance using macro-averaged F1 as the primary metric. The macro-F1 is calculated as the unweighted mean of the class-specific F1 scores:

$$
F 1 _ { \mathrm { { m a c r o } } } = { \frac { 1 } { | C | } } \sum _ { i = 1 } ^ { | C | } F 1 _ { i }\tag{1}
$$

where $| C |$ is the total number of classes (3 for clarity, 9 for evasion), and $F 1 _ { i }$ is the harmonic mean of precision $( P _ { i } )$ and recall $( R _ { i } )$ for class i:

$$
F 1 _ { i } = 2 \cdot { \frac { P _ { i } \cdot R _ { i } } { P _ { i } + R _ { i } } }\tag{2}
$$

To compute precision and recall, model predictions are compared against human annotations.

For the 3-class clarity task, predictions are evaluated strictly against a single ground-truth label obtained via majority vote. A prediction is considered correct only if it exactly matches this label.

![](images/156a7cbdafd9589f606dadff69561250397c0216f262b2a5581894a38b206894.jpg)

![](images/29158b103784d509b5343be4fdd0821701affbc7da8f4b098a4b258743487c19.jpg)  
Figure 24: Question and answer word-count distributions.

![](images/57b86e535068e8c8e4468ca6d5bc0dae4a989ecb21abaf33c1245736829fbb54.jpg)  
Figure 25: Answer word-count distribution per clarity label.

For the 9-class evasion task, a prediction is considered correct if it matches at least one annotator’s label, which reflects valid differences in human interpretation. If it matches none, we use the majority-vote label as the ground truth for an incorrect prediction.

## D.2 Methodology

TF-IDF baseline. We extract word unigram TF-IDF features and tune min $\underline { { d f } } ~ \in ~ \{ 1 , 2 , 3 \}$ and max \_ $\begin{array} { c c l } { \displaystyle { d f } } & { \in } & { \{ 0 . 9 , 0 . 9 5 , 1 . 0 \} } \end{array}$ We use an L2- regularized logistic regression classifier with classbalanced weights. We perform a grid search over $C \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 3 , 5 , 1 0 \}$ and select the best model based on development set macro-F1.

For CLARITY, the best configuration is min $\_ d f = 2 ,$ , max $\scriptstyle - d f = 0 . 9 5 , \ C = 0 . 5$ for clarity, and min \_df=2, max $\scriptstyle - d f = 0 . 9 , C = 2$ for evasion. For PolERo, the best parameters are min $\scriptstyle - d f = 3 ,$ max \_df=0.9, C=5 for clarity and min $\_ d f = 1$ max \_df=0.9, C=2 for evasion.

## Encoder Fine-Tuning

Input formatting. Encoder models use the input format “Question: {q}\nAnswer: $\{ \mathsf { a } \} ^ { \flat }$ On CLARITY, q is the GPT-decomposed atomic question field paired with the interview\_answer. On PolERo, q is the interview\_question field paired with the same interview\_answer, since the Romanian dataset already contains single-question inputs.

![](images/5aff1e0f16b0870052be963566776ce2a32ff245174c775745fc1ecff16d1d38.jpg)  
Figure 26: Answer word-count distribution per evasion label.

![](images/8213ac455ce4c99b3b55301bc7b3b115f0cf1ee7ec48a7cc79dabade2ec8d0b4.jpg)  
Figure 27: Answer word-count distribution per speaker.

Single-head encoders. Each of the eight singlehead encoder configurations is fine-tuned separately for each task for up to 20 epochs using AdamW, with a batch size of 64. The maximum sequence length is set to 512 for all models except ModernBERT-large, which uses a maximum sequence length of 8192. The best checkpoint is selected on an 85/15 train/validation split of the official training set and evaluated once on the test set. Learning rates vary by architecture: $5 \times 1 0 ^ { - 6 }$ for ModernBERT-large and ELECTRAlarge; $1 \times 1 0 ^ { - 5 }$ for RoBERTa-large and XLM-RoBERTa-large; and $2 \times 1 0 ^ { - 5 }$ for BERT-largecased, mBERT-base, RoBERT-large, and Romanian BERT.

Multi-head architecture. We apply this architecture to three encoder configurations: RoBERTalarge, XLM-RoBERTa-large, and RoBERT-large. All multi-head models use 7-fold cross-validation stratified by the clarity label. Models are trained for 20 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a batch size of 8. We use early stopping based on the combined validation macro-F1 of both heads, with a patience of 5. At inference time, we ensemble all 7 folds by averaging their predicted class probabilities and selecting the most probable class for each classification head.

## Large Language Models

Open-source LLMs were run locally, while proprietary models were accessed via the OpenAI API platform<sup>6</sup>. All models are queried with default settings and reasoning configurations, using a restricted output schema. Zero-shot prompts include the task description, the full set of category names, and one-line definitions. Few-shot prompts add one labeled example per category. Prompts are run separately for the clarity and evasion tasks and use the dataset’s language.

## E Multi-Head Architecture Details

This appendix expands the multi-head architecture introduced in Section 3 and isolates the contribution of each component on the English dataset. All ablation numbers are 7-fold cross-validation means and standard deviations of validation macro-F1, computed on the training split with folds stratified by the clarity label. Test-set scores for this configuration are reported in Table 11.

## E.1 Hierarchical Input Processing

Responses in both datasets are often longer than the 512-token limit of standard pretrained encoders. Because evasion cues can appear anywhere in a response, simple truncation may remove the relevant span. To avoid this, we segment each tokenized question-answer sequence $T$ into overlapping windows of length L=512 with stride S=256:

$$
C _ { k } = T [ k S : k S + L ] \quad { \mathrm { f o r } } \ k = 0 , \dots , M - 1 ,\tag{3}
$$

where $M = \lceil \operatorname* { m a x } ( | T | - L , 0 ) / S \rceil + 1$ . The final chunk extends to $| T |$ | and is zero-padded if shorter. The window size $L$ is fixed by the encoder’s positional embedding capacity. The 50% overlap ensures that every interior token is included in at least two consecutive windows, reducing the chance that important evasion cues are separated across chunks.

Each chunk is encoded independently by the shared encoder. We extract the hidden state at position 0 of each chunk, $h _ { k } = H _ { k } [ 0 , : ] \in \mathbb { R } ^ { d }$ , and aggregate the M chunk vectors via element-wise max-pooling:

$$
v _ { j } = \operatorname* { m a x } _ { k = 0 } ^ { M - 1 } h _ { k , j } \quad { \mathrm { f o r ~ } } j = 1 , \dots , d .\tag{4}
$$

The position-0 token corresponds to the encoder’s start-of-sequence token only for the first chunk; for subsequent chunks, it is the first content token of the window. We use this position uniformly because the encoder distributes contextual information across all tokens.

The pooled vector v is passed through dropout $( p { = } 0 . 1 )$ and fed into two linear classification heads:

$$
\hat { y } _ { c } = \mathrm { s o f t m a x } ( W _ { c } \cdot \mathrm { D r o p o u t } ( v ) + b _ { c } ) ,\tag{5}
$$

$$
\hat { y } _ { e } = \mathrm { s o f t m a x } ( W _ { e } \cdot \mathrm { D r o p o u t } ( v ) + b _ { e } ) ,\tag{6}
$$

with $W _ { c } \in \mathbb { R } ^ { 3 \times d }$ and $W _ { e } \in \mathbb { R } ^ { 9 \times d }$ . The training objective is the unweighted sum of two cross-entropy losses, $\mathcal { L } = \mathcal { L } _ { c } + \mathcal { L } _ { e }$ . At inference time, the 7 models are ensembled by averaging predicted class probabilities and taking the argmax per head.

## E.2 Component Ablations

The ablations below isolate the impact of each architectural choice over single-head encoder finetuning.

Pooling strategy. Table 4 compares elementwise max-pooling with mean-pooling and a baseline that keeps only the first chunk representation. Max-pooling performs best. Mean-pooling performs worse than max-pooling, indicating that simple averaging is less effective for aggregating chunk representations. The first-chunk baseline misses any signal beyond the first 512 tokens, which affects 28.8% of English responses.

<table><tr><td>Pooling</td><td>Clarity</td><td>Evasion</td></tr><tr><td>First chunk only</td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td> $0 . 4 2 \pm 0 . 0 1$ </td></tr><tr><td>Mean-pooling</td><td> $0 . 6 8 \pm 0 . 0 2$ </td><td> $0 . 4 3 \pm 0 . 0 2$ </td></tr><tr><td>Max-pooling</td><td> ${ \bf 0 . 7 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 2 }$ </td></tr></table>

Table 4: Chunk aggregation strategy. 7-fold CV validation macro-F1 $( \mathrm { m e a n } \pm \mathrm { s t d } )$ .

Multi-task versus single-task. Table 5 shows that joint training matches single-task performance on clarity and improves single-task evasion by +0.03 macro-F1. The coarser clarity supervision acts as a regularizer for rare evasion categories: each evasion label maps deterministically to a clarity label, so gradients from $\mathcal { L } _ { c }$ help stabilize representations of the parent class, even when the evasion head provides limited signal.

<table><tr><td>Objective</td><td>Clarity</td><td>Evasion</td></tr><tr><td>Single-task (clarity)</td><td> ${ \bf 0 . 7 0 \pm 0 . 0 2 }$ </td><td>一</td></tr><tr><td>Single-task (evasion)</td><td></td><td> $0 . 4 2 \pm 0 . 0 1$ </td></tr><tr><td>Multi-task</td><td> $0 . 7 0 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 2 }$ </td></tr></table>

Table 5: Effect of joint training. 7-fold CV validation macro-F1 (mean ± std).

Ensemble size. Table 6 reports performance for $k \in \{ 3 , 5 , 7 \}$ folds. Increasing k reduces variance and increases the amount of training data per fold model. Improvements are consistent across both subtasks and larger for the more difficult evasion task.

<table><tr><td>Folds</td><td>Clarity</td><td>Evasion</td></tr><tr><td>3</td><td> $0 . 6 6 \pm 0 . 0 1$ </td><td> $0 . 4 2 \pm 0 . 0 2$ </td></tr><tr><td>5</td><td> $0 . 6 8 \pm 0 . 0 2$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td></tr><tr><td>7</td><td> ${ \bf 0 . 7 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 2 }$ </td></tr></table>

Table 6: Effect of ensemble size. k-fold CV validation macro-F1 (mean ± std).

Loss function. Table 7 compares unweighted cross-entropy with inverse-frequency classweighted cross-entropy and focal loss $( \gamma { = } 2 )$ (Lin et al., 2020). Neither alternative improves macro-F1. Class weighting increases minorityclass recall (Clear Non-Reply from 0.62 to 0.67, Partial/half-answer from 0.00 to 0.10), but this is offset by lower precision on majority classes. Focal loss yields intermediate performance between the two. Most evasion errors occur between pragmatically similar classes (Implicit, Deflection, General, Dodging) that share surface features. Reweighting increases the training signal for rare classes but does not reduce overlap between similar evasion categories.

<table><tr><td>Loss</td><td>Clarity</td><td>Evasion</td></tr><tr><td>Cross-entropy (unweighted)</td><td> ${ \bf 0 . 7 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 2 }$ </td></tr><tr><td>Class-weighted CE</td><td> $0 . 6 6 \pm 0 . 0 2$ </td><td> $0 . 4 1 \pm 0 . 0 2$ </td></tr><tr><td>Focal (γ=2)</td><td> $0 . 6 9 \pm 0 . 0 2$ </td><td> $0 . 4 4 \pm 0 . 0 2$ </td></tr></table>

Table 7: Effect of loss function. 7-fold CV validation macro-F1 (mean ± std).

## E.3 Seed Variation

We repeat the full multi-head pipeline for both monolingual configurations under three random seeds (7, 42, 123). The seed controls the crossvalidation split, the per-fold training seed, and weight initialization. Table 8 reports mean and standard deviation of macro-F1.

Cross-validation scores are stable: combined macro-F1 varies $\mathsf { b y } \pm . 0 0 2$ on English $\mathrm { a n d } \pm . 0 0 8$ on Romanian, as each figure averages over seven folds. Test-set scores vary more, as they come from a single evaluation: clarity is . $. 6 7 6 { \scriptstyle \pm . 0 1 9 }$ on English and . $7 4 3 { \scriptstyle \pm . 0 2 3 }$ on Romanian, evasion . $. 5 3 5 { \scriptstyle \pm . 0 2 4 }$ and $. 5 5 8 { \scriptstyle \pm . 0 3 7 } .$ . This variation is smaller than the differences we report between model families, so the effects cannot be attributed to seed choice.

<table><tr><td>Metric</td><td>English</td><td>Romanian</td></tr><tr><td colspan="3">7-fold cross-validation (validation folds)</td></tr><tr><td>Clarity</td><td> $. 7 0 0 { \scriptstyle \pm . 0 0 1 }$ </td><td> $. 6 7 6 _ { \pm . 0 0 6 }$ </td></tr><tr><td>Evasion</td><td> $. 4 5 0 { \scriptstyle \pm . 0 0 4 }$ </td><td> $. 3 8 8 { \scriptstyle \pm . 0 1 1 }$ </td></tr><tr><td>Combined</td><td> $. 5 7 5 { \scriptstyle \pm . 0 0 2 }$ </td><td>.532±.008</td></tr><tr><td colspan="3">Test set (7-fold ensemble)</td></tr><tr><td>Clarity</td><td> $. 6 7 6 _ { \pm . 0 1 9 }$ </td><td> $. 7 4 3 { \scriptstyle \pm . 0 2 3 }$ </td></tr><tr><td>Evasion (any-of)</td><td> $. 5 3 5 { \scriptstyle \pm . 0 2 4 }$ </td><td> $. 5 5 8 { \scriptstyle \pm . 0 3 7 }$ </td></tr><tr><td>Mapped-clarity</td><td> $. 6 8 2 { \scriptstyle \pm . 0 2 1 }$ </td><td>.739±.027</td></tr></table>

Table 8: Seed variation of the multi-head architecture: macro-F1 mean ± standard deviation over three seeds (7, 42, 123), with RoBERTa-large as the English backbone and RoBERT-large as the Romanian one.

## F Extended Model Comparison andError Analysis

Table 11 presents the complete test-set macro-F1 results for all baselines, architectural variants, crosslingual transfer setups, and LLM prompting strategies discussed in this work.

## F.1 Traditional classifier

The TF-IDF logistic regression baseline achieves similar macro-F1 on both languages (0.42 on English, 0.45 on Romanian), but its error patterns show which parts of the task rely on lexical cues. The model correctly identifies most Clear Reply instances in Romanian (64% recall), but performs much worse in English (33% recall).

In contrast, the Ambivalent class is often confused with the other two labels. In Romanian, this misclassification goes both ways: 48% of true Ambivalent instances are predicted as Clear Reply, while 33% of true Clear Reply instances are predicted as Ambivalent. This suggests that any answer that reuses question vocabulary is often treated as a reply by the baseline, even when it fails to provide a concrete answer.

Clear Non-Reply performs worst, with an F1 of 0.25 on English and 0.26 on Romanian (and recall as low as 24%). As reported in Appendix C.6, these are typically very short non-replies (e.g., Clarification, Declining to answer). Because this classifier relies on term frequency, these brief answers generate weak feature signals, causing the model to underpredict these minority classes.

The slightly higher performance on Romanian is consistent with the higher share of Explicit replies in its training distribution (Section C.1). More cases can be solved through lexical overlap.

## F.2 Encoder Models

Best single-head models. Among single-head encoders, RoBERTa-large is the strongest English model on evasion.

RoBERT-large is the strongest model on PolERo. It is pretrained on Romanian text and uses a tokenizer trained on Romanian data, unlike the shared multilingual vocabulary used by XLM-RoBERTa. Although XLM-RoBERTa-large is the strongest multilingual model among those evaluated and is competitive with monolingual encoders on Romanian clarity, it underperforms on the fine-grained evasion task.

Failure modes. ELECTRA-large and ModernBERT-large are the two main failure cases. ELECTRA uses a replaced-token-detection (RTD) objective, which trains the model at the token level to distinguish real from replaced tokens. On small and imbalanced datasets like those tested, we hypothesize that RTD may make the model more susceptible to overfitting on frequent lexical cues rather than learning broader evasion features. As a result, on the Romanian dataset, the model assigns all inputs to the Explicit class.

ModernBERT-large shows the same issue, though less severely. Its local-global attention mechanism improves efficiency for longer input sequences compared to standard BERT encoders, which may offer limited benefit for short questionanswer inputs. In our experiments, three rare categories (Claims ignorance, Partial/half-answer, Deflection) receive zero recall on Romanian, and the mapped clarity predictions inherit the same majority-class bias.

Multi-head architecture. Multi-head models consistently outperform single-head models across all reported metrics. The strongest performance is observed for RoBERTa-large (English) and RoBERT-large (Romanian), both of which are pretrained on in-language corpora and appear to benefit most from the joint signal in our experiments.

XLM-RoBERTa-large shows larger relative gains, but still underperforms monolingual encoders, suggesting that multilingual representations may not fully capture fine-grained signals within a given language.

![](images/d00e37581357133a652bb4795be5acbbec27a320a6350abac641095bce8ba5cb.jpg)  
Figure 28: Out-of-fold (OOF) confusion matrix for the 9-class evasion task on CLARITY (EN) using multihead RoBERTa-large.

![](images/0a1b61b391c4c79c3f07aaff83b78da3f0fc9532939aa489b559efb77e3978ca.jpg)  
Figure 29: Out-of-fold (OOF) confusion matrix for the 9-class evasion task on PolERo (RO) using multi-head RoBERT-large.

Error Analysis. While the boundary between Clear Reply and Ambivalent remains the largest source of errors, the out-of-fold confusion matrices for our strongest models (Figure 28, Figure 29) show that the fine-grained errors between evasion strategies concentrate along two main semantic axes.

The most prominent axis of confusion is Implicit– Explicit. Both represent valid, on-topic answers. The difference depends on whether the requested information is stated directly or must be inferred from context. In English, this axis accounts for 285 errors, and it remains a major source of error in Romanian with 215 misclassifications.

The second major axis is General-Explicit. Both classes provide relevant information in response to the requested information. The distinction depends on whether the response directly addresses the specific details of the question or remains more general. This axis accounts for 146 errors in English, but increases substantially in Romanian, becoming the largest source of error with 315 misclassifications.

Another distinct source of errors is Dodging-Deflection. Both involve topic shifts. The difference depends on whether the model first attempts to answer the question before shifting topic or shifts topic without addressing the question. In English, this axis accounts for 157 errors, although decreasing to 22 errors in Romanian.

Partial/half-answer is the most difficult category for all encoders, with an F1 of 0.066 on English and 0.024 on Romanian. Correctly identifying it requires determining whether only part of a multipart question is answered while another part is ignored, a relational judgment that standard encoder representations do not capture well.

Error structure. We inspect where the errors of the strongest fine-tuned encoders happen relative to annotator agreement on the test split. For multihead RoBERTa-large on the English evasion task, 43.7% of errors occur on items with full 3/3 annotator agreement and 50.4% on items with 2/3 agreement, while only 5.9% involve perfect disagreement. The pattern is more evident on multihead RoBERT-large for Romanian, where 54.7% of evasion errors and 56.6% of clarity errors fall on items with unanimous labels. Errors occur on items with the cleanest annotation signal rather than on the genuinely ambiguous ones, largely because unanimous and high-agreement items make up the vast majority of the dataset. The higher apparent accuracy on the perfect-disagreement subset (.758 on English evasion vs. .528 on 3/3 items) is an expected result of the evasion evaluation rule (Section D.1), which marks a prediction as correct if it matches any of the three distinct annotator labels.

Calibration. The difference between top-1 probability on correct and incorrect predictions is narrow across both tasks. The English RoBERTa-large multi-head averages .605 on correct evasion predictions and .483 on incorrect ones. The clarity head averages .835 vs. .726. The Romanian RoBERTlarge multi-head is more confident in both cases (.923 vs. .822 for clarity; .859 vs. .637 for evasion), consistent with the more imbalanced Romanian label distribution. Confident errors occur in both languages: the Romanian clarity head assigns probability 1.00 to Clear Reply on single-word responses unanimously labeled Ambivalent. The English evasion head assigns probability above .85 to Explicit on 3/3-agreement Implicit and Deflection responses. Top-1 probability is therefore not a reliable rejection signal at deployment.

<table><tr><td></td><td colspan="3">Single-head</td><td colspan="3">Multi-head</td><td colspan="3">∆ (MH – SH)</td></tr><tr><td>Configuration</td><td>C</td><td>E</td><td>M</td><td>C</td><td>E</td><td>M</td><td>C</td><td>E</td><td>M</td></tr><tr><td> $\mathrm { E N } \to \mathrm { R O }$ </td><td>.506</td><td>.282</td><td>.512</td><td>.635</td><td>.399</td><td>.655</td><td>+.129</td><td>+.117</td><td>+.143</td></tr><tr><td> $\mathrm { E N } 2 \mathrm { R O }  \mathrm { R O }$ </td><td>.565</td><td>.314</td><td>.616</td><td>.665</td><td>.412</td><td>.706</td><td>+.100</td><td>+.098</td><td>+.090</td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { R O }$ </td><td>.695</td><td>.403</td><td>.686</td><td>.723</td><td>.570</td><td>.717</td><td>+.028</td><td>+.167</td><td>+.031</td></tr><tr><td> $\mathrm { E N }  \mathrm { E N } 2 \mathrm { R O }$ </td><td>.535</td><td>.334</td><td>.574</td><td>.616</td><td>.448</td><td>.612</td><td>+.081</td><td>+.114</td><td>+.038</td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { E N }$ </td><td>.628</td><td>.459</td><td>.617</td><td>.679</td><td>.483</td><td>.642</td><td>+.051</td><td>+.024</td><td>+.025</td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { E N } + \mathrm { R O }$ </td><td>.676</td><td>.508</td><td>.688</td><td>.737</td><td>.529</td><td>.705</td><td>+.061</td><td>+.021</td><td>+.017</td></tr><tr><td>Mean gain</td><td></td><td></td><td></td><td></td><td></td><td></td><td>+.075</td><td> $\mathbf { + . 0 9 0 }$ </td><td>+.057</td></tr></table>

Table 9: Single-head vs. multi-head XLM-RoBERTa-large across cross-lingual configurations. C: clarity, E: evasion, M: mapped clarity; test-set macro-F1.

Single-head vs. multi-head. Table 9 isolates the architectural contribution from the training-data configuration by running each cross-lingual setting with both a standard single-head encoder and the multi-head architecture, using XLM-RoBERTalarge as a common backbone with identical data, input format and evaluation protocol. The multi-head architecture outperforms the single-head baseline in every configuration and on every metric, by a mean of +.075 macro-F1 on clarity, +.090 on evasion and +.057 on mapped clarity. The gains are therefore attributable to the architecture rather than to the training-data configuration. The advantage is largest on the fine-grained evasion task, consistent with the joint objective stabilising rare categories. It is also largest where training data is scarcest: zero-shot transfer (EN → RO) gains +.129 on clarity, whereas the joint bilingual settings gain +.028 and +.051.

## F.3 LLM Prompting Analysis

The complete LLM results in Table 11 show that GPT-5.4 achieves the best mapped scores on both languages and the strongest English evasion performance. Qwen3.6-35B-A3B performs best on Romanian evasion under few-shot prompting, while Gemma-4-31B-it is best on English direct clarity.

For both languages, the performance difference between the strongest LLMs and encoder ensembles is larger on the evasion task than on the clarity task. The clarity task mainly requires checking whether the answer matches the question, which strong encoders handle well. In contrast, the nineclass evasion task requires distinguishing between closely related rhetorical strategies and benefits from broader discourse and pragmatic knowledge.

Multi-head fine-tuning reduces this difference, but does not eliminate it. Performance on the ambivalent evasion categories remains the main challenge for both model families.

## F.4 Machine Translation

Translate-Train recovers part of the mapped clarity gap but yields only small improvements on the evasion task. Machine translation preserves enough propositional content to recover the coarse threeclass distinction, but distorts pragmatic cues that separate ambivalent categories.

Deflection and Dodging are most affected, as both rely more on discourse structure than lexical cues. As a result, both show substantial recall drops under translation compared to in-language singlehead models. On the Romanian test set, recall for Deflection decreases from 41% to 10%, while Dodging drops from 50% to 30%.

## F.5 Machine Translation Quality

We evaluate the quality of the NLLB-200-distilled-600M translations used in the cross-lingual experiments with XCOMET-XL, a model that estimates translation quality without reference translations. Results are reported for both translation directions on the training and test splits.

For RO→EN, translating the Romanian data resulted in mean QE scores of 0.650 (train, n=3,278) and 0.655 (test, n=296), with clear variation across categories. Formulaic categories translate well, including Clarification (0.781 train, 0.941 test) and Declining to answer (0.763 train, 0.763 test), which obtain the highest scores. Categories driven by pragmatics score lower. Deflection has the lowest scores in both splits (0.582 train, 0.473 test). This drop matches the recall degradation for Deflection and Dodging under MT augmentation.

For EN→RO, translating the English data resulted in mean QE scores of 0.218 (train, n=3,448) and 0.171 (test, n=308). Although these scores are notably low, a manual inspection of the translations shows that many spans flagged as errors are acceptable translations. Despite this noise, the relative ranking across categories is consistent with the RO→EN direction: Deflection and General score lowest, while Clarification and Claims ignorance score highest.

<table><tr><td>Clarity (L1)</td><td>Evasion (L2)</td><td>Description</td><td>Example (Romanian; English in italics)</td></tr><tr><td>Clear Reply</td><td>Explicit</td><td>The requested information is given directly, in the form re- quested, without ambiguity or required inference.</td><td>Î: Când va fi gata autostrada? R: Lucrările vor fi finalizate în luna martie 2025. Q: When will the highway be ready? A: Works will be completed in March 2025.</td></tr><tr><td rowspan="4">Ambivalent</td><td>Implicit</td><td>The requested information is conveyed but not stated in the expected form; the listener must infer the conclusion.</td><td>Î: Vă veti da demisia din funcția de ministru? R: Nu mai pot rămâne o secundă în acest guvern. Q: Will you resign as minister? A: I cannot stay in this government a second longer. Î: În ce lună va fi gata autostrada?</td></tr><tr><td>General</td><td>The response stays on topic but remains too broad, abstract, or vague to deliver the requested specifics.</td><td>R: Dezvoltarea infrastructurii e o prioritate zero pen- tru noi și muncim zi de zi pe șantiere. Q: In which month will the highway be ready? A: Infrastructure development is our top priority and we work every day on site.</td></tr><tr><td>Partial/half- answer</td><td>Covers concretely only one com- ponent of the requested informa- tion and omits the rest.</td><td>Î: Ce companii de stat vor fi privatizate anul acesta? R: Vă pot spune cu certitudine că Pota Română se află pe această listă. Q: Which state-owned companies will be privatised this year? A: I can tell you with certainty that the</td></tr><tr><td>Dodging</td><td>Does not engage with the sub- stance of the question; the topic is neither acknowledged nor touched.</td><td>Romanian Postal Service is on the list. Î: De ce ati pierdut alegerile locale? R: Presa ar trebui să se ocupe de problemele reale ale cetățenilor. Q: Why did you lose the local elections? A: The press</td></tr><tr><td></td><td>Deflection</td><td>Acknowledges the topic of the question but shifts the argument and focus to something other than what is asked.</td><td>should focus on citizens&#x27;real problems. Î: Câti bani ati alocat pentru acest program? R: Programul este un proiect la care inem foarte mult, însă marea realizare de anul acesta este că am reusit să modernizăm peste 50 de şcoli. Q: How much money did you allocate to this pro- gramme? A: The programme is a project we care deeply about, but this year&#x27;s big achievement is that</td></tr><tr><td rowspan="3">Clear Non-Reply</td><td>Declining to an- swer</td><td>A verbal, assumed and explicit refusal to provide the requested information.</td><td>Î: Când veti publica raportul financiar? R: Nu voi face niciun comentariu pe acest subiect până la finalizarea auditului. Q: When will you publish the financial report? A: I will not comment on this matter until the audit is complete.</td></tr><tr><td>Claims igno- rance</td><td>The speaker explicitly admits not holding the requested infor- mation at the moment.</td><td>Î: La ce dată exactă a ordonat guvernul reamenajarea navei HMAS Kanimbla? R: Nu stiu acea dată. Voi afla si voi informa Camera. Q: On what exact date did the government order the refit of HMAS Kanimbla? A: I do not know that date.</td></tr><tr><td>Clarification</td><td>Does not provide the requested information and instead asks for clarification of what the ques- tion refers to.</td><td>I will find out and inform the Chamber. Î: A fost decizia dumneavoastră să eliberati fondul? R: Vă referiți la fondul public sau cel privat? Q: Was it your decision to release the fund? A: Do you mean the public fund or the private one?</td></tr></table>

Table 10: PolERo taxonomy. Descriptions and examples are taken from the Romanian annotation guidelines used by annotators; English translations are provided in italics for readability.

<table><tr><td rowspan="2">Model</td><td colspan="3">CLARITY (EN, n=308)</td><td colspan="3">PolERo (RO, n=296)</td></tr><tr><td>Clarity</td><td>Evasion</td><td>Mapped</td><td>Clarity</td><td>Evasion</td><td>Mapped</td></tr><tr><td>TF-IDF + Logistic Regression</td><td>.422</td><td>.301</td><td>.441</td><td>.453</td><td>.312</td><td>.472</td></tr><tr><td>Single-head encoder fine-tuning</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mBERT-base</td><td>.529</td><td>.347</td><td>.517</td><td>.601</td><td>.328</td><td>.502</td></tr><tr><td>BERT-large-cased</td><td>.587</td><td>.361</td><td>.581</td><td>.475</td><td>.282</td><td>.501</td></tr><tr><td>ELECTRA-large</td><td>.607</td><td>.231</td><td>.524</td><td>.500</td><td>.084</td><td>.241</td></tr><tr><td>ModernBERT-large</td><td>.575</td><td>.360</td><td>.524</td><td>.440</td><td>.246</td><td>.443</td></tr><tr><td>RoBERTa-large</td><td>.580</td><td>.481</td><td>.656</td><td>.646</td><td>.428</td><td>.613</td></tr><tr><td>XLM-RoBERTa-large</td><td>.579</td><td>.419</td><td>.564</td><td>.680</td><td>.365</td><td>.686</td></tr><tr><td>Romanian BERT (RO)</td><td></td><td></td><td></td><td>.684</td><td>.470</td><td>.634</td></tr><tr><td>RoBERT-large (RO)</td><td></td><td></td><td></td><td>.688</td><td>.504</td><td>.701</td></tr><tr><td>Multi-head encoder fine-tuning</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RoBERTa-large</td><td>.713</td><td>.568</td><td>.731</td><td>.696</td><td>.501</td><td>.652</td></tr><tr><td>XLM-RoBERTa-large</td><td>.665</td><td>.475</td><td>.644</td><td>.694</td><td>.508</td><td>.706</td></tr><tr><td>RoBERT-large (RO)</td><td></td><td></td><td></td><td>.729</td><td>.545</td><td>.763</td></tr><tr><td colspan="3">Cross-lingual transfer (XLM-RoBERTa-large, multi-head)</td><td></td><td></td><td></td><td></td></tr><tr><td>EN → RO</td><td></td><td></td><td></td><td>.635</td><td>.399</td><td>.655</td></tr><tr><td>EN2RO → RO</td><td></td><td></td><td></td><td>.665</td><td>.412</td><td>.706</td></tr><tr><td>RO → RO2EN</td><td></td><td></td><td></td><td>.681</td><td>.494</td><td>.702</td></tr><tr><td> $\mathrm { R O } + \mathrm { R O } 2 \mathrm { E N } \to \mathrm { R O }$ </td><td></td><td></td><td></td><td>.708</td><td>.492</td><td>.681</td></tr><tr><td> $\mathrm { R O } + \mathrm { E N } 2 \mathrm { R O } \to \mathrm { R O }$ </td><td></td><td></td><td></td><td>.718</td><td>.489</td><td>.684</td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { R O }$ </td><td></td><td></td><td></td><td>.723</td><td>.570</td><td>.717</td></tr><tr><td>RO → EN</td><td>.551</td><td>.398</td><td>.527</td><td></td><td></td><td></td></tr><tr><td>RO2EN → EN</td><td>.477</td><td>.358</td><td>.439</td><td></td><td></td><td></td></tr><tr><td>EN → EN2RO</td><td>.616</td><td>.448</td><td>.612</td><td></td><td></td><td></td></tr><tr><td>EN + EN2RO → EN</td><td>.621</td><td>.460</td><td>.647</td><td></td><td></td><td></td></tr><tr><td>EN + RO2EN → EN</td><td>.694</td><td>.477</td><td>.602</td><td></td><td></td><td></td></tr><tr><td>EN + RO → EN</td><td>.679</td><td>.483</td><td>.642</td><td></td><td></td><td></td></tr><tr><td colspan="3"> $\mathrm { E N } + \mathrm { R O }  \mathrm { E N } + \mathrm { R O } ^ { \dag }$  Clarity .737</td><td>Evasion .529</td><td></td><td>Mapped .705</td><td></td></tr><tr><td>Cross-lingual transfer (RoBERTa-large, multi-head)</td><td></td><td></td><td></td><td>.624</td><td>.409</td><td></td></tr><tr><td>EN → RO EN2RO → RO</td><td></td><td></td><td></td><td>.594</td><td>.421</td><td>.639 .584</td></tr><tr><td>RO → RO2EN</td><td></td><td></td><td></td><td>.625</td><td>.427</td><td>.635</td></tr><tr><td> $\mathrm { R O } + \mathrm { R O } 2 \mathrm { E N } \to \mathrm { R O }$ </td><td></td><td></td><td></td><td>.664</td><td>.512</td><td>.652</td></tr><tr><td>RO + EN2RO → RO</td><td></td><td></td><td></td><td>.677</td><td>.491</td><td>.674</td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { R O }$ </td><td></td><td></td><td></td><td>.712</td><td>.463</td><td>.681</td></tr><tr><td>RO → EN</td><td>.572</td><td>.337</td><td>.499</td><td></td><td></td><td></td></tr><tr><td>RO2EN → EN</td><td>.590</td><td>.409</td><td>.543</td><td></td><td></td><td></td></tr><tr><td>EN → EN2RO</td><td>.479</td><td>.357</td><td>.502</td><td></td><td></td><td></td></tr><tr><td>EN + EN2RO → EN</td><td>.672</td><td>.513</td><td>.654</td><td></td><td></td><td></td></tr><tr><td>EN + RO2EN → EN</td><td>.675</td><td>.505</td><td>.641</td><td></td><td></td><td></td></tr><tr><td>EN + RO → EN</td><td>.671</td><td>.503</td><td>.683</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { E N } + \mathrm { R O }  \mathrm { E N } + \mathrm { R O } ^ { \dag }$ </td><td></td><td>Clarity .715</td><td></td><td>Evasion .528</td><td>Mapped .718</td><td></td></tr><tr><td colspan="3">LLM prompting (no fine-tuning)</td><td></td><td></td><td></td><td></td></tr><tr><td>Llama-3.3-70B (zero-shot)</td><td>.474</td><td>.377</td><td>.624</td><td>.479</td><td>.354</td><td>.446</td></tr><tr><td>Llama-3.3-70B (few-shot)</td><td>.535</td><td>.382</td><td>.564</td><td>.555</td><td>.312</td><td>.411</td></tr><tr><td>Gemma-4-31B-it (zero-shot)</td><td>.655</td><td>.546</td><td>.619</td><td>.657</td><td>.560</td><td>.713</td></tr><tr><td>Gemma-4-31B-it (few-shot)</td><td>.721</td><td>.583</td><td>.662</td><td>.672</td><td>.619</td><td>.738</td></tr><tr><td>GPT-OSS-120B (zero-shot)</td><td>.548</td><td>.559</td><td>.683</td><td>.575</td><td>.544</td><td>.740</td></tr><tr><td>GPT-OSS-120B (few-shot)</td><td>.542</td><td>.554</td><td>.729</td><td>.553</td><td>.580</td><td>.729</td></tr><tr><td>DeepSeek-V4-Pro (zero-shot)</td><td>.599</td><td>.512</td><td>.693</td><td>.604</td><td>.556</td><td>.696</td></tr><tr><td>DeepSeek-V4-Pro (few-shot)</td><td>.601</td><td>.545</td><td>.670</td><td>.627</td><td>.594</td><td>.751</td></tr><tr><td>Qwen3.6-35B-A3B (zero-shot)</td><td>.673</td><td>.536</td><td>.719</td><td>.684</td><td>.631</td><td>.738</td></tr><tr><td>Qwen3.6-35B-A3B (few-shot)</td><td>.669</td><td>.544</td><td>.747</td><td>.746</td><td>.682</td><td>.760</td></tr><tr><td>GPT-5.4 (zero-shot)</td><td>.687 .703</td><td>.613 .626</td><td>.729 .761</td><td>.687 .763</td><td>.649</td><td>.744 .763</td></table>

Table 11: Full test-set macro-F1 across all models. Clarity: direct 3-class prediction. Evasion: direct 9-class prediction. Mapped: 3-class clarity prediction obtained by mapping the predicted 9-class evasion label upward through the taxonomy. For cross-lingual setups, EN2RO denotes English data machine-translated into Romanian, and RO2EN denotes Romanian data machine-translated into English. Best score per column is bolded; second-best is underlined.  
<sup>†</sup> Evaluated on the combined EN+RO test set (n=604).