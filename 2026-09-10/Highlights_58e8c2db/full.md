## Highlights

Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection? The Impact of Emotion-Aware AI on Proactive Online Safety

Hamed Jelodar, Amir Firouzi, Yen-Wu Lo, Maryam Tanha, Sajjad Dadkhah

• Introduces an AI-based system for early cyberbullying detection.

• Uses emotion-aware filtering to reduce unnecessary processing.

• Enables early intervention to support mental health.

• Improves proactive monitoring of harmful online content.

# Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection? The Impact of Emotion-Aware AI on Proactive Online Safety

Hamed Jelodar<sup>a,∗</sup>, Amir Firouzi<sup>a</sup>, Yen-Wu Lo<sup>a</sup>, Maryam Tanha<sup>b</sup>, Sajjad Dadkhah<sup>a</sup>

<sup>a</sup>Faculty ofComputer Science, University ofNew Brunswick, Fredericton, NB, Canada <sup>b</sup>Khoury College of Computer Sciences, Northeastern University, Vancouver, BC, Canada

## Abstract

Healthcare systems, mental health, and public well-being are increasingly affected by cyberbullying and harmful online interactions. This paper presents CareGuard, an early-warning framework designed to support healthcare-driven mental health protection and proactive online safety through the detection of cyberbullyingrelated content using advanced natural language processing techniques. Care-Guard integrates zero-shot semantic labeling with fine-tuned transformer-based models, including BERT, DistilBERT, and RoBERTa, to enable robust and contextaware classification across sensitive cyberbullying categories. To improve efficiency and reduce unnecessary computation in healthcare-oriented monitoring settings, the framework incorporates an emotion-aware filtering mechanism alongside cosine similarity–based semantic screening, allowing the system to focus on semantically relevant and emotionally salient content. Experimental results on benchmark datasets demonstrate that CareGuard effectively balances detection accuracy and computational efficiency, highlighting its potential for scalable deployment in healthcare systems, mental health monitoring, and online safety applications.

## Keywords:

Artificial Intelligence, Healthcare Informatics, Cyberbullying, Society, Online

## 1. Introduction

According to a report by NBC News, a child in Texas died by suicide during an online game, allegedly due to cyberbullying. A 16-year-old boy from Michigan was identified as the suspect and pleaded guilty to aiding suicide and misdemeanor harassment [33]. Mental health challenges associated with social media and cyberbullying are a significant issue that affects individuals of all ages [1, 2]. Cyberbullying includes harmful behaviors such as sending abusive messages, posting negative comments, and excluding individuals from online communities. These actions can lead to serious psychological effects, including anxiety, depression, and social isolation [3, 4, 5]. Compared to traditional bullying, cyberbullying is more difficult to detect and control due to its anonymous and pervasive nature. Children and adolescents are especially vulnerable, as they are highly active on digital platforms [6, 7]. To address these challenges, we propose CareGuard, an AI-driven early warning system that leverages large language models (LLMs) and advanced natural language processing (NLP) techniques. The system is designed to detect and categorize cyberbullying and online violence in real time by analyzing digital communications across multiple platforms.

Developing such a system presents several challenges. Ensuring high detection accuracy while minimizing false positives is essential to maintain user trust. In addition, protecting user privacy and adhering to ethical standards are critical considerations. CareGuard addresses these challenges by integrating advanced algorithms that balance detection performance with privacy preservation. The system is designed to be both effective and ethically responsible, making it a valuable tool for enhancing online safety.

## 1.1. Research Motivation

The motivation behind this research stems from the increasing prevalence and severity of cyberbullying and online violence, which pose significant risks to mental health and well-being. The large volume, diversity, and dynamic nature of online communications make timely detection and intervention challenging. Existing methods often focus on static classification and lack the ability to provide early warnings or support proactive intervention. Therefore, there is a need for more robust and scalable approaches that can effectively detect harmful content and facilitate early response.

## 1.2. Research Contributions

The main contribution of this research is the development of CareGuard, an AI-driven early warning framework for detecting cyberbullying and online violence. The key contributions are as follows:

• We propose a novel NLP-based framework that integrates zero-shot semantic labeling with fine-tuned transformer models for cyberbullying detection.

• We introduce an emotion-aware filtering mechanism based on emotion annotation and cosine similarity to reduce irrelevant data processing and improve computational efficiency.

• We developd a multi-task post-modeling analysis module using NLP techniques and prompt engineering to support interpretability and decision-making.

• We design CareGuard as an early warning system to support mental health–oriented interventions, extending beyond traditional classification-based approaches.

## 2. Related Works

The rapid growth of online social platforms has significantly increased exposure to cyberbullying across different age groups, particularly among teenagers. This exposure has been linked to serious psychological consequences, including depression and suicidal ideation [7, 8, 9, 10, 11, 12]. The World Health Organization recognizes bullying as a major public health concern due to its long-term educational, physical, and mental health impacts [13].

## 2.1. Traditional and Machine Learning Approaches

Early studies on cyberbullying detection mainly relied on traditional machine learning methods such as Support Vector Machines (SVM), Naive Bayes, and rule-based approaches. These methods use handcrafted features such as word frequency, n-grams, and lexical patterns. While they are simple and interpretable, they have limited ability to capture complex language patterns, sarcasm, and contextual information in social media text.

Table 1: Comparison of existing cyberbullying detection studies and the proposed CareGuard framework.
<table><tr><td>Study</td><td>Main Focus</td><td>Main Limitation</td><td>Key Insight</td><td>Early Warn- ing / Inter- vention</td></tr><tr><td>[14]</td><td>tection</td><td>Cyberbullying de- Requires labeled data and may have limited cross- domain generalization</td><td>Combining ML, DL, and BERT can improve de- tection performance over traditional baselines</td><td>No</td></tr><tr><td>[19]</td><td>Detection classification cyberbullying</td><td>and Computationally demand- of ing and dependent on la- beled datasets</td><td>Combining BiLSTM and No BERT improves contex- tual representation and supports multi-category classification</td><td></td></tr><tr><td>[22]</td><td>Cyberbullying classification</td><td>Performance may depend strongly on the dataset and can achieve strong clas- domain</td><td>1 BiLSTM-based models No sification performance for cyberbullying detec- tion</td><td></td></tr><tr><td>[23]</td><td>Cyberbullying using Twitter</td><td>Primarily focused on a during COVID-19 specific social-media con- CNN and MLP provides text and event</td><td>Combining BERT with No strong performance for context-specific cyber-</td><td></td></tr><tr><td>[20]</td><td>tweets</td><td>Large-scale anal- Focuses mainly on textual ysis of abusive abuse and does not provide proactive intervention</td><td>bullying detection NLP can support large- No scale analysis of abusive- language patterns and temporal trends</td><td></td></tr><tr><td>[21]</td><td>Relationship lying and suicidal early-warning system ideation</td><td>Does not provide an between cyberbul- automated cyberbullying</td><td>Highlights the impor- No tance of considering psychological and con- textual consequences of</td><td></td></tr><tr><td>CareGuard</td><td>Detection tion</td><td>&amp; Requires further valida- proactive preven- tion across diverse plat- ward proactive early forms and real-world set- tings</td><td>cyberbullying Extends detection to- Yes warning and interven- tion support</td><td></td></tr></table>

## 2.2. Deep Learning and Transformer-based Methods

Recent research has shifted toward deep learning models, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and transformerbased models. In [14], the authors propose an ensemble approach combining machine learning and deep learning methods, along with a BERT-based model, achieving higher accuracy compared to SVM baselines. Similarly, [19] uses BiL-STM and BERT to detect and classify cyberbullying into categories such as religion, age, and gender. In [22], different deep learning models are compared, showing that BiLSTM achieves better performance in terms of accuracy and F1- score. Despite these improvements, such models often require large labeled datasets and may not generalize well across different platforms.

## 2.3. Sentiment and Context-aware Approaches

Some researchers focus on sentiment analysis and contextual understanding to improve cyberbullying detection. In [23], the authors analyze cyberbullying during the COVID-19 pandemic using Twitter data and apply BERT combined with CNN and MLP models, achieving accuracy between 87.2% and 92.3%. In [20], a large-scale analysis of abusive tweets is conducted using NLP techniques to study trends over time. Additionally, [21] highlights psychological factors, showing that psychotic experiences can increase the impact of cyberbullying on suicidal ideation. Although these approaches improve detection, they often focus primarily on text sentiment and may miss deeper behavioral patterns.

## 2.4. Limitations ofExisting Methods

Despite significant progress, existing methods face several limitations. First, many models struggle to detect implicit or context-dependent cyberbullying, such as sarcasm or coded language. Second, models trained on specific datasets may not generalize well across different platforms or domains. Third, most approaches focus only on detection and do not provide early warning or intervention mechanisms. Unlike existing approaches that focus solely on classification, the proposed model, CareGuard, introduces an early warning mechanism to support proactive intervention. These advancements position CareGuard as a more comprehensive and practical solution for real-world cyberbullying detection and prevention.

## 2.5. Positioning of CareGuard

The comparison presented in Table 1 highlights several important gaps in existing cyberbullying detection research. Traditional machine learning methods provide relatively simple and interpretable solutions but rely heavily on handcrafted features and have limited ability to capture contextual and implicit forms of cyberbullying. Deep learning and transformer-based approaches improve contextual representation and classification performance; however, they commonly depend on large labeled datasets and are often evaluated within specific datasets or social-media platforms. Furthermore, most existing studies primarily focus on detecting or classifying cyberbullying rather than supporting proactive responses.

• In contrast, CareGuard is designed as a proactive cyberbullying detection and early-warning framework. Rather than limiting the task to binary or multi-class classification, CareGuard aims to identify potentially harmful interactions and provide an early indication of cyberbullying risk. This enables the system to move beyond retrospective detection toward proactive prevention and intervention support. The proposed framework therefore addresses an important gap in existing research by integrating cyberbullying detection with an early-warning mechanism.

• As highlighted in Table 1, CareGuard is distinguished from previous approaches by explicitly incorporating early warning and intervention support. This design is intended to make cyberbullying detection more practical for real-world applications, where identifying harmful behavior early can be more valuable than detecting it only after the bullying event has occurred. Nevertheless, further evaluation across diverse platforms, datasets, and realworld scenarios is required to assess the generalizability and effectiveness of the proposed framework.

## 3. Proposed Model

In this section, we present CareGuard, a multi-phase AI-driven framework for cyberbullying detection and early warning. The proposed model combines text pre-processing, emotion-aware semantic filtering, transformer-based classification, and large language model (LLM)-based post-analysis within a unified pipeline. The main objective of CareGuard is not only to identify cyberbullying content but also to provide interpretable and actionable outputs that can support timely intervention. The overall architecture of the proposed framework is illustrated in Figure 1.

<sub>l</sub> <sub>architecture</sub> <sub>of</sub> <sub>the</sub> <sub>proposed</sub> <sub>CareGuard</sub> <sub>fra</sub>m<sup>ework,</sup> <sup>structured</sup> <sup>i</sup>  
![](images/764b9664600200223e1cef60c22239649120e677a86ea722b392b09acadd0006.jpg)

## 3.1. Overview of the CareGuard Pipeline

CareGuard is designed as a four-phase framework: Phase A) Data Collection and Text Pre-processing, Phase B) Semantic Embedding and Emotion-Aware Filtering, Phase C) Transformer-Based Refinement, and Phase D) Post-Modeling LLM Analysis. This design enables CareGuard to operate as an end-to-end early warning system rather than a conventional single-stage classifier.

## 3.2. Phase A: Data Collection and Text Pre-processing

In the first phase, we use an existing dataset D collected from Kaggle, where $D = \{ T _ { 1 } , T _ { 2 } , . . . , T _ { N } \}$ denotes a set of N tweets related to cyberbullying. The objective of this phase is to prepare the raw textual data for subsequent semantic and classification analysis.

To improve text quality and reduce noise, several pre-processing operations are applied, including stemming, stop-word removal, and HTML cleaning. These steps standardize the input data and reduce irrelevant variations in the text.

Let $T _ { i }$ denote the i-th tweet in the dataset. The stemming operation is defined as:

$$
T _ { i } ^ { \prime } = \mathrm { s t e m } ( T _ { i } ) ,
$$

where stem(·) reduces each word to its root form. Next, stop-word removal is applied:

$$
\begin{array} { r } { T _ { i } ^ { \prime \prime } = T _ { i } ^ { \prime } - \mathrm { s t o p \_ w o r d s } , } \end{array}
$$

where stop\_words represents the set of common words with limited semantic contribution. Finally, HTML and noisy markup are removed:

$$
\begin{array} { r } { T _ { i } ^ { \prime \prime \prime } = \mathrm { c l e a n \_ h t m l } ( T _ { i } ^ { \prime \prime } ) . } \end{array}
$$

The processed dataset is then represented as:

$$
D ^ { \prime } = \{ T _ { i } ^ { \prime \prime \prime } | T _ { i } \in D \} .
$$

## 3.3. Phase B: Semantic Embedding and Emotion-aware Filtering

After pre-processing, the tweets are passed to an emotion-aware semantic filtering module. The goal of this phase is to reduce noise, improve efficiency, and retain tweets that are more likely to contain harmful or relevant content.

For each preprocessed tweet, negative sentiment and emotional cues are extracted. Let $E ( t )$ denote the emotion score of a tweet t and let $\theta _ { \mathrm { n e g } }$ represent the

Algorithm 1 Data Collection and Pre-processing   
Input: Dataset D with N cyberbullying-related tweets   
Output: Processed dataset D   
Initialization:   
Load dataset D   
for each tweet T ∈ D do   
Apply stemming: T<sup>′</sup> ← stem(T<sub>i</sub>)   
Remove stop-words: T<sup>′′</sup><sub>i</sub> ← T<sup>′</sup><sub>i</sub> − stop\_words   
Clean HTML/noisy markup: T<sup>′′′</sup> ← clean\_html(T<sup>′′</sup>)   
end for   
Construct processed dataset:   
D<sup>′</sup> ← {T<sup>′′′</sup> | T<sub>i</sub> ∈ D}   
Return: D<sup>′</sup>

negativity threshold. The filtering operation is formally modeled as a mapping function $\Phi ( t )$ that determines the data flow paths based on emotional intensity:

$$
\begin{array} { r } { \Phi ( t ) = \left\{ \begin{array} { l l } { \mathcal { D } _ { \mathrm { n o i s e } } \cup \{ t \} , } & { \mathrm { i f ~ } E ( t ) \geq \theta _ { \mathrm { n e g } } } \\ { f \big ( \mathcal { C } ( t ) , \mathrm { C o n t e x t } ( t ) , \mathcal { M } _ { \mathrm { L L M } } \big ) , } & { \mathrm { i f ~ } E ( t ) < \theta _ { \mathrm { n e g } } } \end{array} \right. } \end{array}
$$

where $\mathcal { C } ( t )$ denotes the localized structural content of tweet t, Context(t) denotes its surrounding contextual or metadata information, and $\mathcal { M } _ { \mathrm { L L M } }$ represents the fine-tuned large language model instance utilized for subsequent, deeper semantic evaluations.

This phase serves two important purposes. First, it removes tweets that are unlikely to contribute meaningful evidence for cyberbullying detection. Second, it prioritizes semantically and emotionally relevant content for deeper analysis in the next stage.

Algorithm 2 Emotion-aware Filtering Process   
Input: Preprocessed tweets   
Output: Noise dataset D and filtered tweets for advanced analysis   
for each tweet t do   
Extract emotion and sentiment features   
if E(t) < θ<sub>neg</sub> then   
Perform advanced semantic analysis using f Context(t), M   
else   
Add t to D<sub>noise</sub>   
end if   
end for

## 3.4. Phase C: Transformer-based Cyberbullying Refinement

In Phase C, we perform the main cyberbullying classification task using a hybrid transformer-based refinement strategy. The core classifier is a fine-tuned RoBERTa model enhanced with a Bidirectional Gated Recurrent Unit (Bi-GRU) layer. This design combines the contextual representation power of transformers with the sequential modeling capability of recurrent neural networks.

The RoBERTa-based classifier is used as the primary model due to its strong contextual embedding capabilities and robust pre-training strategy. In addition, BERT and DistilBERT are incorporated as supplementary models to further analyze tweets flagged as potentially harmful. This multi-model design improves robustness and provides flexibility under different computational constraints.

![](images/bf6da18a2e41fb199bbb1d9a6b309c22c739bb8c88c96f5c034f1d5524d755ea.jpg)  
Figure 2: Architecture of the transformer-based refinement module using fine-tuned RoBERTa with a Bi-GRU layer.

The selected transformer models serve complementary roles:

• RoBERTa is used as the primary classifier because of its strong contextual encoding and improved pre-training.

• BERT provides effective bidirectional semantic understanding for complex bullying expressions.

• DistilBERT offers a lightweight alternative suitable for resource-constrained environments.

Let $M _ { i }$ denote the i-th transformer model. Its performance is evaluated using accuracy $A _ { i } ,$ precision $P r _ { i } .$ , recall $R _ { i }$ , and F1-score $F 1 _ { i }$ . A weighted performance score is defined as:

$$
S _ { i } = \alpha A _ { i } + \beta P r _ { i } + \gamma R _ { i } + \delta F \sp { } \mathrm { l } _ { i } ,
$$

where $\alpha , \beta , \gamma$ , and $\delta$ are weighting coefficients.

The aggregated performance across n models is defined as:

$$
S _ { \mathrm { c o m b i n e d } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } S _ { i } .
$$

Finally, the Bi-GRU layer captures sequential dependencies in the textual input, helping the framework better identify implicit, context-dependent, and semantically connected bullying patterns.

## 3.5. Phase D: Post-modeling LLM-based Analysis

A key contribution of CareGuard is its post-modeling analysis layer. Rather than stopping at a class prediction, the framework further analyzes tweets flagged as cyberbullying using an LLM-based multi-task strategy. This phase improves interpretability and supports practical intervention.

Specifically, we use a LLaMA-based model (meta-llama/Llama-2-7b-chat-hf) to perform three post-modeling tasks:

1. Mental Health Analysis: estimates the emotional and psychological impact of the detected bullying language.

2. Bullying-Entity Extraction: identifies the targeted individuals or groups mentioned in the tweet.

3. Semantic Point Extraction and Summarization: extracts the most important harmful content and produces a concise explanation.

## 3.5.1. Task I: Mental Health Analysis

This task examines the tone and emotional characteristics of a tweet in order to estimate its potential impact on the victim. Aggression, humiliation, and emotional distress are treated as key indicators of harmful intent.

## 3.5.2. Task II: Bullying-Entity Extraction

This task identifies the entities involved in the harmful interaction, including potential targets of abuse. This enables a better understanding of the social and contextual structure of the bullying event.

## 3.5.3. Task III: Semantic Point Extraction and Summarization

This task extracts the core harmful content and generates a concise summary explaining why the tweet was flagged. This improves interpretability for human analysts and platform moderators.

![](images/b0623634271771369944eca218cde74c8638bd337fa28811994073dd5d3e0b8c.jpg)  
Figure 3: Example prompt used in the LLaMA-based post-modeling analysis module.

To improve reasoning quality, we incorporate prompt engineering techniques such as Chain-of-Thought (CoT) reasoning and zero-shot/few-shot learning. Chainof-Thought (CoT) helps the model reason through tone, context, and interaction cues in a step-by-step manner, while zero-shot and few-shot prompting improve generalization to unseen cyberbullying patterns without requiring extensive retraining.

Mathematically, let $T _ { \mathrm { t w e e t } }$ denote an input tweet classified as cyberbullying. The overall evidence generation process aims to produce robust evidence $E _ { \mathrm { c y b e r } }$ through three tasks $T _ { 1 } , T _ { 2 } , T _ { 3 }$

$$
\tau = f _ { \mathrm { t o n e } } ( T _ { \mathrm { t w e e t } } )
$$

$$
M _ { H } = f _ { \mathrm { i m p a c t } } ( \tau )
$$

$$
E _ { \mathrm { t a r g e t } } = f _ { \mathrm { e n t i t y } } ( T _ { \mathrm { t w e e t } } )
$$

$$
P _ { \mathrm { s e m } } = f _ { \mathrm { s e m } } ( T _ { \mathrm { t w e e t } } )
$$

$$
S _ { \mathrm { s h o r t } } = f _ { \mathrm { s u m m a r y } } ( P _ { \mathrm { s e m } } )
$$

The final output of the post-modeling stage is represented as:

$$
\mathcal { O } _ { \mathrm { t o t a l } } = \sum _ { i = 1 } ^ { 3 } T _ { i } ( M _ { 1 } , \mathcal { C } _ { \mathrm { r e a s o n } } , \mathcal { Z } ) ,
$$

where $M _ { 1 }$ denotes the LLaMA model, $\mathcal { C } _ { \mathrm { r e a s o n } }$ denotes Chain-of-Thought reasoning, and $\mathcal { Z }$ denotes zero-/few-shot prompting.

## 3.6. Why the Proposed Model is Different

The proposed CareGuard framework differs from existing cyberbullying detection studies in three major ways. First, it introduces an emotion-aware filtering stage prior to classification, which reduces noise and improves efficiency. Second, it integrates multiple transformer models in a refinement stage rather than relying on a single classifier. Third, it extends beyond conventional detection by incorporating LLM-based post-analysis for interpretability, mental health-oriented assessment, and semantic explanation. As a result, CareGuard functions not only as a classifier but also as an early warning and decision-support system.

## 4. Experiment and Settings

## 4.1. Fine-Tuned Candidate Models

In Phase D, we implement and evaluate a hybrid cyberbullying detection framework that combines pre-trained transformer models with a Bidirectional Gated Recurrent Unit (BiGRU) layer. The BiGRU component is introduced to enhance the modeling of sequential dependencies in tweets, complementing the contextual representations learned by transformers. We evaluate three widely used transformer architectures—BERT-base, RoBERTa-base, and DistilBERT—each integrated with a BiGRU layer to ensure a fair architectural comparison. All models are fine-tuned using a publicly available cyberbullying dataset from Kaggle [42]. Training is performed with a learning rate of $2 \times 1 0 ^ { - 5 }$ , a batch size of 32, and early stopping based on validation loss to mitigate overfitting. Model evaluation is conducted on a held-out test set of 1,137 samples. To ensure consistency across models, class labels are aligned by mapping gender/sexual to gender prior to analysis.

<table><tr><td colspan="9">Table 2: Comparison among fine-tuned transformer models for cyberbullying detection (RoBERTa-base as reference)</td></tr><tr><td colspan="9">Overall Performance (Single-model evaluation)</td></tr><tr><td></td><td colspan="2">Accuracy</td><td colspan="2">Macro F1</td><td colspan="2">Weighted F1</td><td colspan="2">Macro Recall</td></tr><tr><td>Model</td><td>Estimate</td><td>∆</td><td>Estimate</td><td>∆</td><td>Estimate</td><td>∆</td><td>Estimate</td><td>∆</td></tr><tr><td>RoBERTa-base</td><td>0.91</td><td>Ref.</td><td>0.90</td><td>Ref.</td><td>0.9129</td><td>Ref.</td><td>0.88</td><td>Ref.</td></tr><tr><td>BERT-base</td><td>0.91</td><td>0.00</td><td>0.90</td><td>0.00</td><td>0.91</td><td>-0.0029</td><td>0.88</td><td>0.00</td></tr><tr><td>DistilBERT</td><td>0.8777</td><td>0.0323</td><td>0.8526</td><td>0.0474</td><td>0.8748</td><td>0.0381</td><td>0.8369</td><td>0.0431</td></tr><tr><td colspan="9">Class-level Performance (F1-score)</td></tr><tr><td>Model</td><td colspan="2">ethnicity/race</td><td colspan="2">gender</td><td colspan="2">not_cyberbullying</td><td colspan="2">religion</td></tr><tr><td></td><td>F1</td><td>∆</td><td>F1</td><td>∆</td><td>F1</td><td>△</td><td>F1</td><td>Δ</td></tr><tr><td>RoBERTa-base BERT-base</td><td>0.89</td><td>Ref. 0.02</td><td>0.90 0.88</td><td>Ref.</td><td>0.95</td><td>Ref.</td><td>0.85</td><td>Ref.</td></tr><tr><td>DistilBERT</td><td>0.87</td><td></td><td>0.84</td><td>0.02</td><td>0.95</td><td>0.00</td><td>0.88</td><td>-0.03</td></tr><tr><td></td><td>0.84</td><td>0.05</td><td></td><td>0.06</td><td>0.93</td><td>0.02</td><td>0.80</td><td>0.05</td></tr><tr><td colspan="9">Error Rate Comparison</td></tr><tr><td>Model</td><td colspan="2">Accuracy</td><td colspan="2">Error Rate</td><td colspan="2">∆ Error Rate</td><td></td><td></td></tr><tr><td>RoBERTa-base</td><td colspan="2">0.91</td><td colspan="2">0.09</td><td colspan="2"></td><td>Ref.</td><td></td></tr><tr><td>BERT-base</td><td colspan="2">0.91</td><td colspan="2">0.09</td><td colspan="2"></td><td>0.00</td><td></td></tr><tr><td>DistilBERT</td><td colspan="2">0.8777</td><td colspan="2">0.1223</td><td colspan="2">0.0323</td><td></td><td></td></tr><tr><td colspan="10">fine-tuned RoBERTa-based model is used as the reference model. ∆ denotes the absolute difference relative to the reference. Positive values indicate worse performance than RoBERTa-base. All models are evaluated on the same test set of 1,137 samples with aligned</td></tr></table>

![](images/5d34d089d1a4c189936837df6447a9da0acb046b1e0a25d0d9ecc741c06d7b17.jpg)  
Figure 4: Confusion flow visualization of misclassifications for the fine-tuned BERT-based model.

## 4.2. Confusionflow analysis ofmisclassifications.

Figures 4, 5, and 6 illustrate the misclassification behavior of the fine-tuned transformer models using complementary visualizations, including top misclassification statistics. Figure 6 presents a confusion flow visualization that highlights only the misclassification pathways of the fine-tuned DistilBERT-based cyberbullying detection model. In this diagram, rectangular nodes represent true class labels, while elliptical nodes correspond to predicted labels.

Directed edges connect true and predicted classes, and the numerical annotations on each edge indicate the number of misclassified instances along each path. By excluding correct predictions, the figure provides a focused view of model failures, revealing systematic error patterns rather than overall accuracy. Notably, several minority and sensitive categories (e.g., religion and ethnicity/race) are frequently misclassified as not\_cyberbullying, indicating a tendency of the model to default to the majority class when semantic cues are weak or ambiguous.

![](images/321c17af2202b170d5b2a0e32d75ea7b4e872da44fb76f6830bf5dd555f4d5e0.jpg)  
Figure 5: Confusion flow visualization of misclassifications for the fine-tuned RoBERTa-based model.

## 4.3. Performance ofTransformer Models

Table 2 provides a comprehensive comparison of the fine-tuned transformer models using RoBERTa-base as the reference. In terms of overall performance, RoBERTa-base and BERT-base achieve identical accuracy (0.91) and Macro F1- score (0.90), indicating comparable global predictive capability. However, RoBERTabase exhibits a slightly higher weighted F1-score, suggesting improved robustness under class imbalance.

At the class level, RoBERTa-base demonstrates superior performance on sensitive cyberbullying categories, particularly ethnicity/race and gender, where it achieves the highest F1-scores. These improvements are critical for cyberbullying detection systems, as errors in protected or sensitive categories can have disproportionate social impacts. While BERT-base marginally outperforms RoBERTabase on the religion class, this advantage is offset by its weaker performance on other sensitive categories.

DistilBERT consistently underperforms compared to the other models, exhibiting a notable degradation in Macro F1-score and Macro Recall. The higher error rate observed for DistilBERT indicates a clear trade-off between computational efficiency and detection reliability. This suggests that although DistilBERT may be suitable for resource-constrained environments, it is less appropriate for safety-critical cyberbullying detection tasks.

![](images/1471af00e4a34aaa1d32c95d622147b5e10e968675bb11a7dc2bba7c93bbb373.jpg)  
Figure 6: Confusion flow visualization of misclassifications for the fine-tuned DistilBERT-based model.

Overall, RoBERTa-base emerges as the most balanced and robust model, achieving strong global performance while maintaining superior class-level fairness across sensitive categories. Consequently, RoBERTa-base is selected as the primary model for downstream analysis and deployment in the proposed cyberbullying monitoring framework.

## 4.3.1. Row-normalized confusion matrices

Figures 7–9 show the row-normalized confusion matrices for BERT-base, DistilBERT, and RoBERTa-base, all fine-tuned on the cyberbullying dataset. All models achieve high accuracy on the not\_cyberbullying class, while gender-based cyberbullying is classified reliably across models. Religion-based cyberbullying remains the most challenging category, often being confused with non-bullying content. RoBERTa-base demonstrates the most balanced performance, particularly for ethnicity/race-based cyberbullying, whereas DistilBERT exhibits higher overall misclassification rates.

![](images/1600f038c9cb276e8145e68633717a4a94f3d64df1160b90fa7e072dd379a7c6.jpg)  
Figure 7: Row-normalized confusion matrix of the fine-tuned BERT-base mode

DistilBERT: Confusion Matrix (Row-Normalized %)  
![](images/6c99a60dd119a759cf495881982fcb3baaa9b8980fd4b2f0563ca8d5ed566c99.jpg)  
Figure 8: Row-normalized confusion matrix of the fine-tuned DistilBERT model

![](images/cde57e708d17465bd4deb67bc54fc4e955cacae6bc162909739cd1db136b3c61.jpg)  
Figure 9: Row-normalized confusion matrix of the fine-tuned RoBERTa-base model.

## 5. Importance of the Application of CareGuard in Real Life

We believe that the proposed CareGuard model has significant real-world applications in addressing cyberbullying and online violence. Educational institutions such as schools and universities can use this system to monitor online interactions and enhance student safety. Social media platforms, including Facebook and Instagram, can integrate such tools to detect harmful content and enforce community guidelines [39, 40]. In addition, technology companies such as Google and Microsoft can incorporate cyberbullying detection features into their products to improve user experience and safety.

Furthermore, non-profit organizations and government agencies can use Care-Guard to develop policies and awareness programs aimed at preventing online harassment. Employers can also utilize such systems to detect and mitigate workplace harassment in digital communication channels. In the healthcare sector, hospitals and medical organizations can benefit from cyberbullying detection tools to protect patients, staff, and institutional reputation. These systems enable proactive monitoring of online interactions, helping to maintain a respectful and safe environment for all stakeholders.

## 6. Discussion

The experimental results demonstrate that transformer-based models can provide effective performance for cyberbullying detection, while also revealing several challenges associated with sensitive categories, class imbalance, and the practical deployment of automated detection systems. In particular, the comparison among RoBERTa-base, BERT-base, and DistilBERT shows that model architecture and representational capacity influence the balance between classification performance and computational efficiency. The findings also provide insights into how CareGuard can extend conventional cyberbullying classification toward a more proactive early-warning framework.

## 6.1. Interpretation ofExperimental Results

The experimental results indicate that RoBERTa-base provides the most balanced overall performance among the evaluated transformer models. RoBERTabase achieved an accuracy of 0.91, a Macro F1-score of 0.90, a weighted F1-score of 0.9129, and a Macro Recall of 0.88. BERT-base achieved comparable overall performance, whereas DistilBERT showed a noticeable reduction across the major evaluation metrics. These results suggest that the stronger contextual representations provided by the full-sized transformer architectures are beneficial for distinguishing between cyberbullying and non-cyberbullying expressions.

The comparable performance of RoBERTa-base and BERT-base is consistent with previous studies that have reported strong performance from transformerbased architectures for cyberbullying detection [14, 18, 19]. In particular, the ability of transformer models to capture contextual relationships between words is useful for social-media content, where harmful expressions can vary substantially in vocabulary, structure, and context. However, the results also indicate that increasing model efficiency through distillation can involve a measurable performance trade-off. DistilBERT achieved an accuracy of 0.8777 and a Macro F1- score of 0.8526, which were lower than those of both RoBERTa-base and BERTbase.

The class-level results provide additional insight into model behavior. RoBERTabase achieved F1-scores of 0.89 for ethnicity/race, 0.90 for gender, 0.95 for not cyberbullying, and 0.85 for religion. The relatively lower performance for the religion category suggests that some forms of cyberbullying may be more difficult to distinguish from non-bullying or ambiguous language. Similar challenges have been identified in cyberbullying research, where the linguistic expression of harmful behavior can vary across categories and contexts [1, 10, 22]. These findings emphasize the importance of evaluating cyberbullying detection systems at the class level rather than relying exclusively on aggregate accuracy.

The confusion-flow analysis further demonstrates that minority and sensitive categories can be more frequently misclassified as not\_cyberbullying. This behavior may be influenced by class imbalance as well as the semantic ambiguity of some harmful expressions. Consequently, a model with high overall accuracy may still produce meaningful errors for specific categories. This observation is particularly important for safety-oriented applications, where errors affecting sensitive categories may have greater practical consequences than errors involving the majority class.

## 6.2. Comparison with Previous Studies

The findings of this study are broadly consistent with previous research demonstrating the effectiveness of machine learning, deep learning, and transformerbased methods for cyberbullying detection. Saini et al. [14] demonstrated that combining conventional machine learning, deep learning, and BERT-based approaches can improve cyberbullying detection compared with traditional baselines. Similarly, Gupta et al. [19] investigated BiLSTM and BERT for textual cyberbullying identification and demonstrated the value of contextual representations for multi-category classification. Iwendi et al. [22] also showed that deep learning architectures can provide effective performance for cyberbullying detection.

The results obtained in the present study further support these observations, as RoBERTa-base and BERT-base achieved strong classification performance across the evaluated categories. However, CareGuard differs from conventional classificationoriented approaches by integrating multiple processing stages before and after classification. In particular, the proposed framework combines emotion-aware filtering, transformer-based classification, and LLM-based post-modeling analysis. Therefore, the contribution of CareGuard is not limited to improving classification performance; it is designed to provide additional contextual information that can support the interpretation of detected harmful interactions.

Previous research has also examined cyberbullying within specific social-media contexts and through contextual or psychological perspectives. For example, Sen et al. [23] investigated BERT-based approaches for cyberbullying detection using Twitter data, while Perez and Karmakar [20] examined cyberbullying trends through large-scale analysis of abusive tweets. Fekih-Romdhane et al. [21] further highlighted the relationship between cyberbullying and suicidal ideation. These studies demonstrate that cyberbullying should not be considered solely as a text-classification problem. Instead, its potential psychological and social consequences should also be considered when designing practical detection systems.

In this context, CareGuard extends existing approaches by incorporating an early-warning perspective. Rather than treating the classification result as the final output, the framework performs additional analysis of detected content, including emotional characteristics, targeted entities, and semantic information. This design provides a pathway for transforming a classification result into a more interpretable warning signal that could subsequently be reviewed by human moderators or other responsible decision-makers.

## 6.3. Sensitive Categories and Error Analysis

The class-level evaluation shows that cyberbullying categories are not equally difficult to detect. The relatively strong performance on the not\_cyberbullying class indicates that the models can effectively identify many non-harmful instances. However, the lower performance observed for categories such as religion and ethnicity/race indicates that harmful expressions associated with sensitive topics may be more difficult to classify reliably.

One possible explanation is that cyberbullying related to sensitive attributes can be expressed implicitly rather than through explicit abusive terminology. A harmful statement may therefore appear linguistically similar to ordinary discussion when considered without sufficient contextual information. Previous research has similarly emphasized the challenges associated with contextual, implicit, and domain-dependent cyberbullying expressions [1, 10, 14]. This finding reinforces the importance of contextual modeling and motivates the use of additional semantic and emotion-aware analysis within CareGuard.

## 6.4. Implicationsfor Early Warning and Mental Health Support

An important objective of CareGuard is to move cyberbullying detection beyond retrospective classification toward proactive monitoring and early warning.

Existing approaches often focus primarily on determining whether a given message belongs to a cyberbullying category [14, 19, 22]. Although accurate classification is necessary, practical deployment may require additional information regarding the nature, emotional characteristics, and potential target of the harmful interaction. The emotion-aware filtering stage in CareGuard is intended to prioritize content that contains potentially relevant emotional signals before applying more computationally intensive analysis.

This capability is particularly relevant in contexts where cyberbullying may be associated with psychological distress and reduced well-being. Previous research has reported associations between cyberbullying victimization and adverse psychological outcomes [8, 12, 21]. Therefore, an early-warning system could potentially support timely review and intervention by identifying potentially harmful interactions before they develop into more persistent patterns. However, CareGuard should be considered a decision-support and early-warning framework rather than a clinical diagnostic system. Any real-world healthcare or mental-health application would require additional validation, human oversight, privacy safeguards, and domain-specific evaluation.

## 6.5. Practical Implications

The findings have implications for the design of automated cyberbullying monitoring systems. First, the strong performance of RoBERTa-base suggests that contextual transformer architectures can serve as effective core classifiers. Second, the lower performance of DistilBERT illustrates the trade-off between computational efficiency and classification reliability. Therefore, the choice of model should depend on the deployment environment and the acceptable balance between latency, computational cost, and detection performance. Third, the classlevel results indicate that practical systems should not rely solely on overall accuracy. Finally, the additional analysis provided by CareGuard may help human reviewers understand why content was flagged, potentially supporting more informed moderation and intervention decisions.

## 6.6. Limitations

Despite the promising results, several limitations should be acknowledged. First, the framework is trained and evaluated using a single publicly available dataset. Consequently, the observed performance may not fully represent the linguistic diversity, and evolving characteristics of cyberbullying across different social-media platforms. Differences in vocabulary, slang, communication styles, and community norms may introduce domain-shift challenges during real-world deployment. Second, class imbalance remains a challenge, particularly for minority and sensitive categories such as religion and ethnicity/race.

## 6.7. Future Work

Future research will extend CareGuard in several directions. First, the framework will be evaluated across multiple social-media platforms and multilingual datasets to assess its robustness under domain and language variation. Crossdomain transfer learning and continual learning will be explored to reduce the impact of domain drift [43, 44, 45, 46].

Second, advanced data augmentation, cost-sensitive learning, and debiasing strategies will be investigated to improve performance on underrepresented and sensitive categories [47, 48]. Fairness-oriented evaluation will also be conducted to assess consistency across demographic and linguistic groups.

Third, future versions of CareGuard will incorporate multimodal learning and conversational context modeling to improve the detection of implicit and contextdependent cyberbullying. Finally, the complete early-warning pipeline will be evaluated in realistic scenarios, considering latency, computational cost, human review, privacy preservation, and responsible intervention [49, 50]. Recent datasets, studies, and state-of-the-art LLM and GenAI approaches will also be incorporated to ensure the continued relevance and generalizability of the framework.

## 7. Conclusion

In this study, we introduced CareGuard, an innovative early warning system designed to detect harmful online interactions related to cyberbullying and mental health concerns. The proposed approach integrates emotion-aware filtering, transformer-based classification, and LLM-based post-analysis to improve detection accuracy and interpretability. Experimental results show that RoBERTabased models achieve the most balanced performance, particularly for sensitive categories. The framework also supports early warning and decision-making, making it suitable for real-world online safety applications. Future work will focus on improving generalization, fairness, and multimodal analysis to enhance robustness across diverse environments.

## References

[1] T. Mahmud, M. Ptaszynski, J. Eronen, F. Masui, Cyberbullying detection for low-resource languages and dialects: Review of the state of the art, Inf. Process. Manag. 60 (5) (2023) 103454, https://doi.org/10.1016/j.ipm.2023.103454.

[2] D.P. Farrington, I. Zych, M.M. Ttofi, H. Gaffney, Cyberbullying research in Canada: A systematic review of the first 100 empirical studies, Aggress. Violent Behav. 69 (2023) 101811, https://doi.org/10.1016/j.avb.2022.101811.

[3] A. Al-Marghilani, Artificial intelligence-enabled cyberbullying-free online social networks in smart cities, Int. J. Comput. Intell. Syst. 15 (2022) 9, https://doi.org/10.1007/s44196-022-00063-y.

[4] A. Aggarwal, S. Gaur, Applying artificial intelligence to explore online harassment and cyberbullying prevention, in: Impact of AI on Advancing Women’s Safety, IGI Global, 2024, pp. 104–120, https://doi.org/10.4018/979-8-3693-2679-4.ch007.

[5] A. Saeid, D. Kanojia, F. Neri, Decoding cyberbullying on social media: A machine learning exploration, in: 2024 IEEE Conference on Artificial Intelligence (CAI), IEEE, 2024, pp. 425–428, https://doi.org/10.1109/CAI59869.2024.00084.

[6] Ç.O. Aliyeva, M. Yagano ˘ glu, Deep learning approach to detect cyber-˘ bullying on Twitter, Multimed. Tools Appl. 84 (2025) 20497–20520, https://doi.org/10.1007/s11042-024-19869-3.

[7] M. Alkasassbeh, A. Almomani, A. Aldweesh, A. Al-Qerem, M. Alauthman, K.M.O. Nahar, B. Mago, Cyberbullying detection using deep learning: A comparative study, in: 2024 2nd International Conference on Cyber Resilience (ICCR), IEEE, 2024, pp. 1–6, https://doi.org/10.1109/ICCR61006.2024.10533166.

[8] S. Halliday, A. Taylor, D. Turnbull, T. Gregory, The relationship between traditional and cyber bullying victimization in early adolescence and emotional wellbeing: A cross-sectional, population-based study, Int. J. Bullying Prev. 6 (2024) 110–123, https://doi.org/10.1007/s42380-022-00144-8.

[9] L. López-Castro, P.K. Smith, S. Robinson, A. Görzig, Age differences in bullying victimisation and perpetration: Evidence from cross-cultural surveys, Aggress. Violent Behav. (2023) 101888.

[10] P. Yi, A. Zubiaga, Session-based cyberbullying detection in social media: A survey, Online Soc. Netw. Media 36 (2023) 100250, https://doi.org/10.1016/j.osnem.2023.100250.

[11] S. Hu, W. Lei, H. Zhu, C. Hsu, Cyberbullying perpetration on social media: A situational action perspective, Inf. Manag. 61 (6) (2024) 104013, https://doi.org/10.1016/j.im.2024.104013.

[12] A. John, A.C. Glendenning, A. Marchant, P. Montgomery, A. Stewart, S. Wood, K. Lloyd, K. Hawton, Self-harm, suicidal behaviours, and cyberbullying in children and young people: Systematic review, J. Med. Internet Res. 20 (4) (2018) e129, https://doi.org/10.2196/jmir.9044.

[13] United Nations Office of the High Commissioner for Human Rights, Cyberbullying and children, 2023. Available at: https://www.ohchr.org/en/statements/2023/09/cyberbullying-children.

[14] H. Saini, H. Mehra, R. Rani, G. Jaiswal, A. Sharma, A. Dev, Enhancing cyberbullying detection: A comparative study of ensemble CNN–SVM and BERT models, Soc. Netw. Anal. Min. 14 (1) (2024) 1.

[15] T. Mahmud, M. Ptaszynski, J. Eronen, F. Masui, Cyberbullying detection for low-resource languages and dialects: Review of the state of the art, Inf. Process. Manag. 60 (5) (2023) 103454, https://doi.org/10.1016/j.ipm.2023.103454.

[16] S. Pericherla, E. Ilavarasan, Transformer network-based word embeddings approach for autonomous cyberbullying detection, Int. J. Intell. Unmanned Syst. 12 (1) (2024) 154–166, https://doi.org/10.1108/IJIUS-02-2021-0011.

[17] S. Sihab-Us-Sakib, M.R. Rahman, M.S.A. Forhad, M.A. Aziz, Cyberbullying detection of resource constrained language from social media using transformer-based approach, Nat. Lang. Process. J. 9 (2024) 100104, https://doi.org/10.1016/j.nlp.2024.100104.

[18] R. Sujud, W. Fahs, R. Khatoun, F. Chbib, Cyberbullying detection using Bidirectional Encoder Representations from Transformers (BERT), in: 2024

IEEE International Mediterranean Conference on Communications and Networking (MeditCom), IEEE, 2024, pp. 257–262.

[19] S. Gupta, U. Vadgama, T.R. Vedhavathy, Identification and labeling of textual cyberbullying using BiLSTM and BERT, in: 2023 International Conference on Networking and Communications (ICNWC), IEEE, 2023, pp. 1–5.

[20] C. Perez, S. Karmakar, An NLP-assisted Bayesian time-series analysis for prevalence of Twitter cyberbullying during the COVID-19 pandemic, Soc. Netw. Anal. Min. 13 (1) (2023) 51, https://doi.org/10.1007/s13278-023- 01053-4.

[21] F. Fekih-Romdhane, D. Malaeb, N. Farah, M. Stambouli, M. Cheour, S. Obeid, S. Hallit, The relationship between cyberbullying perpetration/victimization and suicidal ideation in healthy young adults: The indirect effects of positive and negative psychotic experiences, BMC Psychiatry 24 (1) (2024) 121, https://doi.org/10.1186/s12888-024-05552-2.

[22] C. Iwendi, G. Srivastava, S. Khan, P.K.R. Maddikunta, Cyberbullying detection solutions based on deep learning architectures, Multimed. Syst. 29 (3) (2023) 1839–1852, https://doi.org/10.1007/s00530-020-00701-5.

[23] M. Sen, J. Masih, R. Rajasekaran, From Tweets to Insights: BERT-enhanced models for cyberbullying detection, in: 2024 ASU International Conference on Emerging Technologies for Sustainability and Intelligent Systems (ICETSIS), IEEE, 2024, pp. 1289–1293.

[24] M.A. Hedderich, N.N. Bazarova, W. Zou, R. Shim, X. Ma, Q. Yang, A Piece of Theatre: Investigating how teachers design LLM chatbots to assist adolescent cyberbullying education, in: Proceedings of the CHI Conference on Human Factors in Computing Systems, 2024, pp. 1–17, https://doi.org/10.1145/3613904.3642379.

[25] A. Bunce, L. Hashemi, C. Clark, S. Stansfeld, C.-A. Myers, S. Mc-Manus, Prevalence and nature of workplace bullying and harassment and associations with mental health conditions in England: A crosssectional probability sample survey, BMC Public Health 24 (2024) 1147, https://doi.org/10.1186/s12889-024-18614-7.

[26] S. Thimmanayakanapalya, S. Das Smith, G. Sanders, Digital risk considerations across generative AI-based mental health apps, 2024.

[27] Z. Hu, H. Hou, S. Ni, Grow with Your AI Buddy: Designing an LLM-based conversational agent for the measurement and cultivation of children’s mental resilience, in: Proceedings of the 23rd Annual ACM Interaction Design and Children Conference, ACM, 2024, pp. 811–817.

[28] M. Qorich, R. El Ouazzani, Advanced deep learning and large language models for suicide ideation detection on social media, Prog. Artif. Intell. 13 (2) (2024) 135–147, https://doi.org/10.1007/s13748-024-00326-z.

[29] A. Khan, R. Ali, Unraveling minds in the digital era: A review on mapping mental health disorders through machine learning techniques using online social media, Soc. Netw. Anal. Min. 14 (1) (2024) 78.

[30] S.M. Fahim, R.M. Butt, S. Munawar, N.A. Siddiqui, M.K. Lohana, Addressing cyberbullying in healthcare: Causes, consequences, and policy reforms—a theory-driven approach, in: Workplace Cyberbullying and Behavior in Health Professions, IGI Global, 2024, pp. 32–58.

[31] I. Yosep, R. Hikmat, A. Mardhiyah, Nursing intervention for preventing cyberbullying and reducing its negative impact on students: A scoping review, J. Multidiscip. Healthc. 16 (2023) 261–273, https://doi.org/10.2147/JMDH.S400779.

[32] J. Rababah, M.M. Al-Hammouri, A. Awawdeh, The association between undergraduate nursing students’ health literacy and bullying and cyberbullying victimization, J. Prof. Nurs. 52 (2024) 15–20, https://doi.org/10.1016/j.profnurs.2024.03.002.

[33] NBC News, Texas child died by suicide after online game and cyberbullying, 2023. Available at: https://www.nbcnews.com/news/us-news/texaschild-died-suicide-online-game-cyberbullying-authorities-said-rcna129247.

[34] Y. Liu, RoBERTa: A robustly optimized BERT pretraining approach, arXiv preprint arXiv:1907.11692, 2019.

[35] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, arXiv preprint arXiv:1810.04805, 2018.

[36] V. Sanh, L. Debut, J. Chaumond, T. Wolf, DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter, arXiv preprint arXiv:1910.01108, 2019.

[37] L.R. Huesmann, E.F. Dubow, P. Boxer, S.F. Landau, S.D. Gvirsman, K. Shikaki, Children’s exposure to violent political conflict stimulates aggression at peers by increasing emotional distress, aggressive script rehearsal, and normative beliefs favoring aggression, Dev. Psychopathol. 29 (1) (2017) 39–50, https://doi.org/10.1017/S0954579416001115.

[38] M. Laurer, W. van Atteveldt, A. Casas, K. Welbers, Building efficient universal classifiers with natural language inference, arXiv preprint arXiv:2312.17543, 2023.

[39] Z. Zhang, A. Zhang, M. Li, A. Smola, Automatic chain of thought prompting in large language models, arXiv preprint arXiv:2210.03493, 2022, https://doi.org/10.48550/arXiv.2210.03493.

[40] P. Peprah, M.S. Oduro, R. Okwei, C. Adu, B.Y. Asiamah-Asare, W. Agyemang-Duah, Cyberbullying victimization and suicidal ideation among in-school adolescents in three countries: Implications for prevention and intervention, BMC Psychiatry 23 (1) (2023) 944, https://doi.org/10.1186/s12888-023-05268-9.

[41] Meta, Llama 2 7B Chat, 2023. Available at: https://huggingface.co/metallama/Llama-2-7b-chat-hf.

[42] Kaggle, Cyberbully Detection Dataset. Available at: https://www.kaggle.com/datasets/momo12341234/cyberbully-detectiondataset.

[43] J. He, B. Wang, Y. Li, C. Jin, Cyberbullying perpetration and externalizing problem behaviors in adolescents: Spreading effects based on developmental cascades, J. Res. Adolesc. 36 (3) (2026) e70220, https://doi.org/10.1111/jora.70220.

[44] A. Klocek, L. Kollerová, I. Ropovik, T. Lintner, Specific negative outcomes of bullying or victimization forms on psychological adjustment, J. Emot. Behav. Disord. 34 (2) (2026) 63–77, https://doi.org/10.1177/10634266261417596.

[45] H. Yang, Y. Zhou, J. Wu, H. Liu, L. Yang, C. Lv, Human-guided continual learning for personalized decision-making of autonomous driving, IEEE Trans. Intell. Transp. Syst. 26 (4) (2025) 5435–5447, https://doi.org/10.1109/TITS.2024.3524609.

[46] A.K. Mahmood, S.W. Kareem, Continual learning for enhancing personal assistants using machine learning, Adv. Hum.-Comput. Interact. 2026 (2026) 7942440, https://doi.org/10.1155/ahci/7942440.

[47] M.Q. Shatnawi, Q.Q.M. Abuein, H.A. Abualasal, T.M. Al Qudah, R. Atassi, H. Al-Aqrabi, Enhancing the early detection of depression among university students by using ensemble modeling, in: Proc. 17th Int. Conf. Inf. Commun. Syst., 2026, pp. 1–8.

[48] M.K. Jangir, K. Singh, T. Khan, A domain-adapted deep learning model for early detection of nomophobia using Tab Transformer model, SN Comput. Sci. 7 (5) (2026) 464, https://doi.org/10.1007/s42979-026-05018-0.

[49] A. Philipo, J. Ding, D. Sarwatt, J. Mohamed, A. Yusufu, M. Daneshmand, H. Ning, Sentiment-enhanced cyberbullying detection models on social media platforms, ACM Trans. Web 20 (1) (2026) 1–26, https://doi.org/10.1145/3766075.

[50] T. Mokheleli, T. Makaba, P. Ndayizigamiye, N. Ndlovu, H. Twinomurinzi, Artificial intelligence in suicide risk assessment: a systematic literature review, Discov. Artif. Intell. 6 (1) (2026) 296, https://doi.org/10.1007/s44163- 026-01031-7.