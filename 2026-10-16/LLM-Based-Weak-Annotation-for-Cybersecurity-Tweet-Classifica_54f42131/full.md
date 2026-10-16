# LLM-Based Weak Annotation for Cybersecurity Tweet Classification: A Teacher–Student Framework

Ferry Afit Kurniawan, Kazushi Okamoto, Koki Karube, Kei Harada, Atsushi Shibata

Department of Informatics, Graduate School of Informatics and Engineering, The University of Electro-Communications 1-5-1 Chofugaoka, Chofu, Tokyo 182-8585, Japan

{ferry.afit, kazushi, karubekoki, harada, shibata-atsushi}@uec.ac.jp

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Abstract—Cyber threat intelligence (CTI) relies on the timely identification of emerging threats, enabling organizations and security analysts to support situational awareness and decisionmaking. Social media platforms such as X (formerly Twitter) have become valuable CTI sources because vulnerabilities and attack activities are often discussed there before appearing in formal reports. However, the rapid growth of cybersecurity discussions on social media has created more content than domain experts can realistically review and annotate, making labeled data a major bottleneck for developing supervised learning approaches. While large language models (LLMs) have shown promise as weak annotators in other domains, their effectiveness for cybersecurity tweet classification and downstream training of domain-specific pre-trained language models (PLMs) remains underexplored. To address this challenge, we propose a Teacher– Student weak supervision framework in which a selected LLM serves as a Teacher to generate weak labels for unlabeled cybersecurity tweets, enabling the training of cybersecurityspecific PLMs without large-scale manual annotation. Experimental results show that PLMs trained solely on LLM-generated labels consistently achieve F1 scores above 0.92, remaining within 3 percentage points of fully human-supervised models across all evaluated architectures. These findings suggest that LLM-based weak annotation offers a practical alternative for constructing cybersecurity-specific training data when expert annotation resources are limited.

Index Terms—Cyber threat intelligence, weak annotation, large language models, teacher-student framework, tweet classification, pre-trained language models, weak supervision.

## I. INTRODUCTION

Cyber threat intelligence (CTI) supports cybersecurity decision-making by providing actionable information about emerging threats, vulnerabilities, and adversarial activity at strategic, operational, and tactical levels [1], [2]. Modern CTI systems combine heterogeneous information sources, knowledge bases, and analytical tools to transform raw threat data into insights that support security analysts and organizational decisions [1]. Among these sources, social media platforms such as X (formerly Twitter) have become valuable channels for collecting timely CTI. Cybersecurity researchers, vendors, and practitioners often share newly discovered vulnerabilities, malware campaigns, exploit activities, and indicators of compromise (IOCs) on these platforms before the same information appears in formal advisories or public databases [3]– [7]. Automated analysis of social media streams can therefore complement traditional threat intelligence pipelines and improve cyber situational awareness.

Despite this potential, cybersecurity-related tweets are noisy, brief, and produced at a volume that far exceeds what experts can realistically review and annotate manually [5], [7]. This creates a bottleneck for supervised learning approaches, which depend on high-quality labeled data for training. Although numerous models have been proposed for cybersecurity tweet classification and related CTI tasks, their effectiveness depends on carefully curated annotated datasets [3], [4], [8]–[10], [13].

Weak supervision offers a promising way to reduce the reliance on exhaustive manual annotation. Additionally, recent advances in natural language processing (NLP) have demonstrated that large language models (LLMs) can generate weak annotations for training downstream classifiers in a variety of text classification tasks [12], [17]–[20]. However, the reliability of LLM-generated weak labels for cybersecurity tweet classification, and their usefulness for training cybersecurity-specific pre-trained language models (PLMs), remain insufficiently explored.

To address this gap, we propose a Teacher-Student weak supervision framework for cybersecurity tweet classification. First, multiple candidate LLMs are evaluated as Teacher models. Next, the most reliable model is selected to annotate a large corpus of unlabeled cybersecurity tweets. Finally, these weakly labeled data are used to fine-tune cybersecurity-specific PLMs, and their performance is compared under humanonly, LLM-only, and hybrid supervision settings. This design enables us to examine both the annotation quality of LLM Teachers and the downstream value of their weak labels for domain-specific Student models. Accordingly, this study is guided by three research questions:

RQ1 Which candidate LLM provides the most reliable weak annotations for cybersecurity tweets?

RQ2 Can LLM-generated weak labels effectively supervise the training of cybersecurity-specific PLMs?

RQ3 How do LLM-only, human-only, and hybrid supervision compare in downstream classification performance?

The main contributions of this work are summarized as follows:

• We systematically evaluate three LLMs as weak annotators for cybersecurity tweet classification across public benchmark datasets and a newly collected gold-standard dataset.

• We introduce a scalable Teacher-Student pipeline that uses LLM-generated weak labels to construct a training set for cybersecurity-specific PLMs, reducing the need for extensive manual annotation.

• Across four cybersecurity-specific PLMs, we show that LLM-only training recovers approximately 98% of human-only F1 on average, indicating that weak annotation can provide competitive downstream performance under limited annotation resources.

## II. RELATED WORK

## A. Cybersecurity Tweet Classification

Initial work on cybersecurity tweet classification centered on automatically identifying cybersecurity-related content in social media streams. Dion´ısio et al. [3] applied deep neural networks to cyber threat detection on Twitter, while Alves et al. [4] investigated large-scale tweet processing for cybersecurity threat awareness. Subsequent studies explored novelty detection [5], contrastive word embeddings [8], and hybrid IDCNN–BiLSTM architectures [9] to improve classification performance. More recently, transformer-based approaches have become increasingly prominent. Marinho and Holanda [13] applied NLP techniques to automated CTI, whereas Bayer et al. [14], Kebeli and Kwon [15], and You et al. [16] explored fine-tuning, self-training, and promptbased learning to reduce dependence on large labeled datasets while maintaining competitive performance. These studies demonstrate the effectiveness of modern NLP methods for CTI-oriented tweet classification, but they still primarily focus on model design or task-specific supervision rather than LLMgenerated weak labels for cybersecurity-specific PLMs.

## B. LLM-Based Weak Supervision

Recent advances in LLMs have enabled their use as automatic annotators for generating weak labels in NLP tasks. Gilardi et al. [17] reported that ChatGPT can achieve annotation quality comparable to or better than that of human crowd workers across several classification tasks. He et al. proposed AnnoLLM [18], which improves annotation quality through explanation-guided prompting, while Pangakis and Wolken [12] demonstrated that LLM-generated labels can effectively supervise downstream classifiers through knowledge distillation. Similarly, Bansal and Sharma [19] showed that LLM-based annotation can improve model generalization at relatively low labeling cost. Beyond fully automated annotation, hybrid annotation methods have also attracted increasing attention. MEGAnno+ [20] provides an interactive human–LLM collaborative annotation environment, and Wang et al. [21] proposed verification mechanisms that combine

LLM-generated labels with selective human review to improve reliability and efficiency.

## C. Research Gap and Positioning

Although prior studies have advanced cybersecurity tweet classification and LLM-based weak annotation, the intersection of these two research directions remains largely unexplored. Existing cybersecurity studies mainly focus on developing classification architectures or reducing supervision through self-training and prompt-based learning, whereas LLM-based annotation has been evaluated predominantly on general-domain NLP tasks. Consequently, it remains unclear whether LLM-generated weak labels are reliable enough to train cybersecurity-specific PLMs for CTI-oriented tweet classification. Recent Teacher–Student frameworks have demonstrated the feasibility of training classifiers from LLMgenerated labels without manual annotation [22]; however, their applicability to cybersecurity-specific PLMs and to supervision regimes that combine human and LLM labels has not been systematically investigated. This work addresses this gap by evaluating multiple LLMs as Teacher models and assessing whether their weak labels can effectively train cybersecurity-specific PLMs under human-only, LLM-only, and hybrid supervision settings.

Table I summarizes representative prior studies and positions the proposed framework relative to existing research.

## III. METHODOLOGY

## A. Overall Framework

The framework consists of five sequential stages, as illustrated in Fig. 1. First, data sources are prepared by combining public benchmark datasets with a newly collected cybersecurity tweet corpus, including a manually annotated subset for evaluation and supervised baselines. Second, multiple candidate LLMs are evaluated as potential Teacher models, and the most reliable model is selected based on annotation performance. Third, the selected Teacher assigns weak labels to previously unlabeled cybersecurity tweets, producing a large weakly labeled corpus. Fourth, this corpus is used to fine-tune multiple cybersecurity-specific PLMs, which serve as Student models. Finally, the resulting classifiers are evaluated and compared across human-only, LLM-only, and hybrid supervision settings.

## B. Datasets

1) Data Collection: To construct a new cybersecurity tweet corpus, we collected tweets from 112 cybersecurity-related X accounts between November 2024 and January 2025. The selected accounts represent diverse and reputable sources of CTI, including 49 cybersecurity researchers, 39 security vendors, 7 cybersecurity organizations and agencies, and 17 cybersecurity news outlets. A total of 17,581 tweets were collected. After removing non-English tweets and duplicates and normalizing the text, 13,451 unique English-language tweets remained. To build a gold-standard benchmark, we selected a stratified sample of 2,000 tweets for manual annotation. Each tweet was independently labeled by two annotators with expertise in cybersecurity, following predefined annotation guidelines. Inter-annotator agreement was measured using Cohen’s Kappa (κ = 0.89), which indicates high agreement. To ensure high label quality, only tweets for which both annotators assigned the same label were retained. As a result, 103 tweets with annotation disagreements were excluded, yielding a final goldstandard dataset of 1,897 tweets.

TABLE I  
COMPARISON OF REPRESENTATIVE STUDIES AND THE PROPOSED WORK.
<table><tr><td>Study</td><td>Primary Focus</td><td>Cybersecurity Domain</td><td>LLM-based Weak Labels</td><td>Trains on Weak Labels</td><td>Human/ LLM/Hybrid</td></tr><tr><td>Dionísio et al. [3]</td><td>DNN classification</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Shin et al. [8]</td><td>Contrastive embedding</td><td></td><td></td><td></td><td></td></tr><tr><td>Bayer et al. [14]</td><td>Few-shot learning</td><td>√</td><td></td><td></td><td></td></tr><tr><td>TI-Prompt [16]</td><td>Prompt tuning</td><td>√</td><td></td><td></td><td></td></tr><tr><td>CTI-ANN [15]</td><td>Self-training annotation</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Pangakis and Wolken [12]</td><td>LLM-generated labels</td><td></td><td></td><td>V</td><td></td></tr><tr><td>AnnoLLM [18]</td><td>LLM annotation framework</td><td></td><td></td><td></td><td></td></tr><tr><td>MEGAnno+ [20]</td><td>Human-LLM collaboration</td><td></td><td></td><td></td><td></td></tr><tr><td>Kuzman and Ljubešić [22]</td><td>Teacher-Student learning</td><td></td><td></td><td></td><td></td></tr><tr><td>This Work</td><td>Teacher-Student weak supervision</td><td></td><td></td><td>√</td><td></td></tr></table>

![](images/3cd8afeb1dce72b852a223755d5102f47002b4300f2d445928b7822ffaad9734.jpg)  
Fig. 1. Teacher–Student framework

2) Data Splitting: Among the annotated tweets, 400 were reserved as an independent held-out test set and were not used for Teacher selection, weak annotation, or Student model training. The remaining 1,497 manually labeled tweets were used in the human-only and hybrid supervision settings. For Teacher selection, 400 tweets were sampled from this 1,497- tweet pool. The 11,451 tweets outside the annotated subset remained unlabeled and were annotated automatically by the selected Teacher model. These weakly labeled tweets were subsequently used to train the Student models.

3) Public Benchmark Datasets: In addition to the newly collected dataset, we used the publicly available Behzadan [23] and CybAttT [24] datasets during Teacher model selection. Evaluating the candidate Teachers on multiple datasets helps assess performance across different data distributions and reduces the risk of selecting a model that performs well on only a single benchmark.

4) Label Harmonization: The three datasets used in this study were originally annotated using different label schemas. To enable a consistent evaluation of Teacher models and downstream Student training, all datasets were converted into a unified binary classification task consisting of Relevant and Irrelevant classes. Tweets were annotated as Relevant if they contained concrete cyber threat information, such as CVEs and other vulnerabilities, exploits, malware or ransomware, active attacks, security incidents, IOCs, or attributed threat actors and campaigns. Tweets that did not contain concrete threat information were labeled as Irrelevant. The newly collected dataset was already annotated using the binary labels Relevant and Irrelevant and therefore required no label harmonization. For the CybAttT dataset, both Normal News and High Risk News were mapped to the Relevant class, whereas Not News was mapped to the Irrelevant class. For the Behzadan dataset, the Threat category was mapped to the Relevant class, while the Business and Irrelevant categories were merged into the Irrelevant class. Table II summarizes the label harmonization process across the three datasets.

TABLE II  
LABEL HARMONIZATION ACROSS DATASETS.
<table><tr><td>Dataset</td><td>Original Label (s)</td><td>Binary Label</td><td>Count</td></tr><tr><td>Our Data (Gold)</td><td>Relevant</td><td>Relevant</td><td>1,169</td></tr><tr><td>Our Data (Gold)</td><td>Irrelevant</td><td>Irrelevant</td><td>728</td></tr><tr><td>CybAttT</td><td>Normal News + High Risk News</td><td>Relevant</td><td>4,840</td></tr><tr><td>CybAttT</td><td>Not News</td><td>Irrelevant</td><td>31,231</td></tr><tr><td>Behzadan</td><td>Threat</td><td>Relevant</td><td>8,280</td></tr><tr><td>Behzadan</td><td>Business + Irrelevant</td><td>Irrelevant</td><td>8,929</td></tr></table>

## C. Teacher Model Selection

We evaluated three candidate LLMs: GPT-4.1-nano, GPT-4o-mini, and Llama-3.3-70B Instruct. The evaluation was conducted using manually labeled samples drawn from the three benchmark datasets described previously. Specifically, 150 Relevant and 150 Irrelevant tweets were sampled from each of Behzadan and CybAttT, and 400 tweets were sampled from the gold-data pool described above, resulting in 1,000 evaluation samples. All candidate models were evaluated under the same zero-shot prompting setting. For each tweet, the model was asked to determine whether the content was

Zero-shot prompt (universal across datasets, T = 0):   
Role: Senior Cyber Threat Intelligence Analyst.   
Task: Classify the tweet as Relevant (1) or Irrelevant (0). A tweet is Relevant (1) if it contains CONCRETE information about a cyber threat – a specific CVE or vulnerability, a named exploit, a named malware family, an active attack or security incident with a named victim or target, an indicator of compromise (IP, hash, domain, file path), or an attributed threat actor or campaign.   
A tweet is Irrelevant (0) if it contains NO concrete threat instance – including general cybersecurity discussion, security news commentary, marketing, job postings, opinions, educational content, policy discussions, software releases, or statistical trends about cybersecurity without naming a specific incident.   
Operative test: does this tweet reference a SPECIFIC, NAMED threat instance that a CTI analyst could act on? If not, label 0. When genuinely uncertain, default to 0.   
Constraint: Output ONLY the digit 0 or 1.

Relevant or Irrelevant to cybersecurity. The predicted labels were then compared with the human-annotated ground truth. The complete prompt template is shown in Fig. 2.

## D. Weak Annotation

The selected Teacher model was applied to 11,451 unlabeled cybersecurity tweets to generate weak labels for downstream model training. The Teacher produced 7,275 Relevant (63.5%) and 4,176 Irrelevant labels (36.5%). The resulting weakly labeled corpus served as the training data for the LLM-only setting and as an additional supervision source in the hybrid setting. Thus, the downstream experiments evaluate the utility of the raw Teacher-generated labels without manual correction.

## E. Student PLM Training

Four cybersecurity-specific PLMs were selected as Student models: CTI-BERT [11], CySecBERT [25], SecureBERT-2.0 [26], and SecRoBERTa [27]. These models were chosen because they were pre-trained on cybersecurity-related corpora and represent diverse architectures and training strategies within the cybersecurity NLP domain. Using multiple Student models reduces the likelihood that the findings are tied to a particular architecture and provides a broader assessment of the proposed framework.

• Human-only: trained using the 1,497 manually annotated tweets.

• LLM-only: trained using the 11,451 weak labels generated by the selected Teacher model.

• Hybrid: trained using the combination of 1,497 manually annotated tweets and 11,451 LLM-generated labels.

## F. Evaluation Procedure

Teacher model performance was measured on the 1,000- tweet evaluation set described in Sec. III-C using accuracy, precision, recall, and F1-score. The model achieving the strongest overall F1 across the three datasets is selected as the Teacher and used for all subsequent weak annotation. Student PLM performance was measured on the 400-tweet held-out test set, which was set aside before any training and never used for model selection. For comparisons among supervision regimes, we applied an exact permutation Friedman test to the five paired seed-level F1-scores for each PLM. Pairwise Wilcoxon signed-rank comparisons with Holm correction were reported descriptively because five paired seeds limit the attainable exact p-values.

Fig. 2. Universal zero-shot annotation prompt, used identically across all Teacher LLM candidates and all three datasets.  
TABLE III  
ZERO-SHOT LLM ANNOTATION QUALITY PER DATASET
<table><tr><td>Dataset</td><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>Behzadan</td><td>GPT-4.1-nano</td><td>0.7000</td><td>0.8125</td><td>0.5200</td><td>0.6341</td></tr><tr><td rowspan="4">CybAttT</td><td>GPT-4o-mini</td><td>0.6567</td><td>0.8310</td><td>0.3933</td><td>0.5339</td></tr><tr><td>Llama-3.3-70B</td><td>0.7500</td><td>0.8261</td><td>0.6333</td><td>0.7170</td></tr><tr><td>GPT-4.1-nano</td><td>0.8533</td><td>0.8193</td><td>0.9067</td><td>0.8608</td></tr><tr><td>GPT-4o-mini</td><td>0.8267</td><td>0.8500</td><td>0.7933</td><td>0.8207</td></tr><tr><td rowspan="4">Our Data</td><td>Llama-3.3-70B</td><td>0.7967</td><td>0.7214</td><td>0.9667</td><td>0.8262</td></tr><tr><td>GPT-4.1-nano</td><td>0.8975</td><td>0.9518</td><td>0.8785</td><td>0.9137</td></tr><tr><td>GPT-4o-mini</td><td>0.8500</td><td>0.9652</td><td>0.7854</td><td>0.8661</td></tr><tr><td>Llama-3.3-70B</td><td>0.9025</td><td>0.9063</td><td>0.9393</td><td>0.9225</td></tr></table>

## IV. EXPERIMENT AND RESULTS

## A. Teacher Model Selection

Table III reports per-dataset performance. On CybAttT, all three models achieved F1 scores above 0.82, while F1 scores decreased for all three candidates on Behzadan. This decrease was mainly associated with lower recall, suggesting that some Behzadan tweets mapped as Relevant after label harmonization may have fallen outside the concrete cyber threat information criterion used in our prompt. On the gold-standard evaluation sample, Llama-3.3-70B achieved the highest F1 score (0.9225), closely followed by GPT-4.1-nano (0.9137).

Across the full 1,000-tweet evaluation set, Llama-3.3-70B achieved the highest F1 score (0.8436), followed by GPT-4.1-nano (0.8312) and GPT-4o-mini (0.7758). McNemar’s test confirmed that both top-performing models significantly outperformed GPT-4o-mini $( \chi ^ { 2 } \ge 9 . 5 1 , p < 0 . 0 1$ for both comparisons). Although the difference between Llama-3.3- 70B and GPT-4.1-nano was not significant $( \chi ^ { 2 } ~ = ~ 0 . 0 1$ $p = 0 . 9 2 )$ , Llama-3.3-70B was selected as the Teacher based on its highest F1 score. It also achieved higher recall on Our Data (0.9393 vs. 0.8785), which is desirable for CTI prefiltering. However, GPT-4.1-nano incurred a lower annotation cost than Llama-3.3-70B (\$0.0100 versus \$0.0158) for 400 tweets, indicating a cost–performance trade-off.

## B. Student PLM Classification

Table IV reports PLM performance on the 400-tweet held-out test set. Each training configuration was run five times with different random seeds; results are reported as mean ± standard deviation across the five runs. Across all four PLMs, LLM-only training recovered approximately 98% of the human-only F1 score on average (range 97.6%–98.8%).

TABLE IV  
PLM CLASSIFICATION PERFORMANCE
<table><tr><td>Model</td><td>Regime</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>CTI-BERT</td><td>Human-only</td><td> $\mathbf { 0 . 9 4 4 5 \pm 0 . 0 0 6 2 }$ </td><td> $\mathbf { 0 . 9 6 5 2 \pm 0 . 0 0 9 7 }$ </td><td> $\mathbf { 0 . 9 4 3 9 \pm 0 . 0 1 5 0 }$ </td><td> $\mathbf { 0 . 9 5 4 3 \pm 0 . 0 0 5 4 }$ </td></tr><tr><td></td><td>LLM-only</td><td> $0 . 9 1 4 5 \pm 0 . 0 0 3 3$ </td><td> $0 . 9 2 2 0 \pm 0 . 0 0 7 2$ </td><td> $0 . 9 4 0 7 \pm 0 . 0 0 6 8$ </td><td> $0 . 9 3 1 2 \pm 0 . 0 0 2 5$ </td></tr><tr><td></td><td>Hybrid</td><td> $0 . 9 1 5 5 \pm 0 . 0 0 4 1$ </td><td> $0 . 9 2 6 2 \pm 0 . 0 0 6 9$ </td><td> $0 . 9 3 7 4 \pm 0 . 0 1 4 0$ </td><td> $0 . 9 3 1 7 \pm 0 . 0 0 4 0$ </td></tr><tr><td>CySecBERT</td><td>Human-only</td><td> $\mathbf { 0 . 9 4 1 0 \pm 0 . 0 0 4 5 }$ </td><td> $\mathbf { 0 . 9 6 4 6 \pm 0 . 0 1 6 3 }$ </td><td> $0 . 9 3 9 0 \pm 0 . 0 2 2 8$ </td><td> $\mathbf { 0 . 9 5 1 3 \pm 0 . 0 0 4 5 }$ </td></tr><tr><td></td><td>LLM-only</td><td> $0 . 9 1 8 0 \pm 0 . 0 0 8 6$ </td><td> $0 . 9 3 2 8 \pm 0 . 0 0 9 8$ </td><td> $0 . 9 3 4 2 \pm 0 . 0 1 8 3$ </td><td> $0 . 9 3 3 3 \pm 0 . 0 0 7 5$ </td></tr><tr><td></td><td>Hybrid</td><td> $0 . 9 2 1 0 \pm 0 . 0 0 5 8$ </td><td> $0 . 9 2 8 2 \pm 0 . 0 0 6 4$ </td><td> $\mathbf { 0 . 9 4 4 7 \pm 0 . 0 1 5 4 }$ </td><td> $0 . 9 3 6 3 \pm 0 . 0 0 5 2$ </td></tr><tr><td>SecRoBERTa</td><td>Human-only</td><td> $0 . 9 2 0 5 \pm 0 . 0 1 0 5$ </td><td> $0 . 9 3 4 0 \pm 0 . 0 1 6 6$ </td><td> $0 . 9 3 7 4 \pm 0 . 0 2 3 1$ </td><td> $0 . 9 3 5 4 \pm 0 . 0 0 9 0$ </td></tr><tr><td></td><td>LLM-only</td><td> $0 . 9 0 6 5 \pm 0 . 0 0 5 8$ </td><td> $0 . 9 1 9 2 \pm 0 . 0 1 3 3$ </td><td> $0 . 9 3 0 1 \pm 0 . 0 2 0 2$ </td><td> $0 . 9 2 4 4 \pm 0 . 0 0 5 4$ </td></tr><tr><td></td><td>Hybrid</td><td> $\mathbf { 0 . 9 2 5 0 \pm 0 . 0 0 6 8 }$ </td><td> $\mathbf { 0 . 9 3 6 4 \pm 0 . 0 0 9 2 }$ </td><td> $\mathbf { 0 . 9 4 2 3 \pm 0 . 0 1 7 1 }$ </td><td> $\mathbf { 0 . 9 3 9 2 \pm 0 . 0 0 6 1 }$ </td></tr><tr><td>SecureBERT-2.0</td><td>Human-only</td><td> $\mathbf { 0 . 9 3 7 0 \pm 0 . 0 0 4 5 }$ </td><td> $\mathbf { 0 . 9 7 4 4 \pm 0 . 0 0 9 8 }$ </td><td> $0 . 9 2 2 0 \pm 0 . 0 1 2 3$ </td><td> $\mathbf { 0 . 9 4 7 4 \pm 0 . 0 0 4 0 }$ </td></tr><tr><td></td><td>LLM-only</td><td> $0 . 9 0 7 5 \pm 0 . 0 1 0 5$ </td><td> $0 . 9 1 3 2 \pm 0 . 0 1 3 3$ </td><td> $0 . 9 3 9 0 \pm 0 . 0 1 1 9$ </td><td> $0 . 9 2 5 9 \pm 0 . 0 0 8 3$ </td></tr><tr><td></td><td> $\mathrm { H y b r i d }$ </td><td> $0 . 9 3 2 5 \pm 0 . 0 0 4 7$ </td><td> $0 . 9 4 0 5 \pm 0 . 0 0 2 6$ </td><td> $\mathbf { 0 . 9 5 0 4 \ : \pm 0 . 0 1 0 5 }$ </td><td> $0 . 9 4 5 4 \pm 0 . 0 0 4 1$ </td></tr></table>

TABLE V  
SIZE-MATCHED LLM-ONLY ABLATION
<table><tr><td>Model</td><td>LLM-matched</td><td>LLM-full</td><td>Human-only</td></tr><tr><td>CTI-BERT</td><td> $0 . 9 2 4 0 \pm 0 . 0 0 6 2$ </td><td> $0 . 9 3 1 2 \pm 0 . 0 0 2 5$ </td><td> $0 . 9 5 4 3 \pm 0 . 0 0 5 4$ </td></tr><tr><td>CySecBERT</td><td> $0 . 9 2 2 0 \pm 0 . 0 0 6 2$ </td><td> $0 . 9 3 3 3 \pm 0 . 0 0 7 5$ </td><td> $0 . 9 5 1 3 \pm 0 . 0 0 4 5$ </td></tr><tr><td>SecRoBERTa</td><td> $0 . 9 0 2 0 \pm 0 . 0 1 3 4$ </td><td> $0 . 9 2 4 4 \pm 0 . 0 0 5 4$ </td><td> $0 . 9 3 5 4 \pm 0 . 0 0 9 0$ </td></tr><tr><td>SecureBERT-2.0</td><td> $0 . 9 1 7 1 \pm 0 . 0 0 6 2$ </td><td> $0 . 9 2 5 9 \pm 0 . 0 0 8 3$ </td><td> $0 . 9 4 7 4 \pm 0 . 0 0 4 0$ </td></tr></table>

The largest mean F1 gap was observed for CTI-BERT (0.9543 versus 0.9312). Hybrid training achieved a higher mean F1 than LLM-only training for all PLMs and achieved the highest mean F1 for SecRoBERTa (0.9392).

Exact permutation Friedman tests indicated significant overall differences among the supervision regimes for all PLMs (Table $\operatorname { I V } \colon Q = 6 . 4 – 8 . 4 , p = 0 . 0 0 8 – 0 . 0 3 9 ;$ Table V: $Q = 8 . 4 \AA$ $1 0 . 0 , p < 0 . 0 0 9 )$ . With five paired seeds, the smallest attainable two-sided exact Wilcoxon p-value is 0.0625. No Holmadjusted pairwise comparison can therefore reach the 0.05 threshold. Pairwise differences are consequently interpreted descriptively based on mean F1.

## C. Size-matched LLM-only Ablation

To disentangle the contribution of label quantity from label quality, each PLM was also trained on a size-matched LLMonly subset of $n = 1 , 4 9 7$ tweets. This subset matched the size of the human-only pool and was drawn from the full LLMonly corpus using stratified sampling. The subset was drawn once and held fixed across seeds. Thus, the reported variability does not reflect variation in subset selection. Table V compares these results with those obtained from full LLM-only and human-only training. When restricted to the same number of labels $( n = 1 , 4 9 7 )$ , human-only training yielded 2.9–3.3 percentage points higher mean F1 than size-matched LLMonly training across all four models, suggesting that manual labels provided a stronger supervision signal at equal scale. The full LLM-only corpus (11,451 tweets; 7.6× the humanonly pool) reduced this mean F1 gap to 1.1–2.3 percentage points. The complete workflow, comprising evaluation of three candidate Teachers on the 1,000-tweet evaluation set and weak annotation of the 11,451 unlabeled tweets, incurred total API charges of less than US\$0.70.

## D. Discussion

1) RQ1: Which candidate LLM provides the most reliable weak annotations for cybersecurity tweets?: Llama-3.3-70B was selected as the Teacher model because it achieved the highest overall F1-score and the highest recall on the goldstandard dataset. Although its F1-score was not significantly different from that of GPT-4.1-nano $( p \ : = \ : 0 . 9 2 )$ , its higher recall is particularly desirable for CTI pipelines, where a relevance classifier serves as the initial filtering stage and missing a genuine threat is generally more costly than flagging additional false positives for review.

2) RQ2: Can LLM-generated weak labels effectively supervise the training of cybersecurity-specific PLMs?: At matched training-set size, human-only training yielded higher mean F1 across all PLMs. The gap narrowed when the full weakly labeled corpus was used, suggesting that the larger training set contributed to the recovery. LLM-generated weak labels therefore provide an effective supervision signal, with scale rather than per-label accuracy driving the improvement.

3) RQ3: How do LLM-only, human-only, and hybrid supervision compare in downstream classification performance?: The three supervision settings revealed a trade-off between annotation quality and training data quantity. Human-only supervision yielded the highest mean F1 for most PLMs. Hybrid supervision showed a higher mean F1 than LLM-only training for all PLMs but rarely surpassed human-only training, suggesting that additional weak labels may also introduce label noise. Because the Teacher assigns class labels rather than generating text, this noise takes the form of unsupported label assignments rather than fabricated content.

## E. Limitations

This study has several limitations that provide opportunities for future work. First, the proposed framework focuses on binary classification, distinguishing Relevant from Irrelevant. This design was chosen because relevance filtering is typically the first stage of a CTI pipeline before more specific analyses, such as topic modeling or event detection. Second, only

English-language tweets were included in this study. Evaluating the framework on multilingual datasets would provide broader coverage, since threat intelligence is also shared in many other languages. Finally, the newly collected dataset was constructed from 112 curated cybersecurity-related X accounts to reduce noise and improve the efficiency of dataset construction. However, it may not fully capture cybersecurity discussions from newly emerging sources. Future research should explore broader collection strategies while maintaining data quality.

## V. CONCLUSION

In this study, we proposed and evaluated a Teacher-Student weak supervision framework for cybersecurity tweet classification, in which an LLM serves as the Teacher to generate weak labels for training domain-specific PLMs. The experimental results showed that Llama-3.3-70B was the most reliable Teacher model, while Student models trained solely on LLMgenerated labels achieved performance close to that of fully supervised models. Additional experiments suggested that scaling training data can partially offset lower label quality. These findings demonstrate that LLM-based weak annotation offers a practical approach to reducing manual annotation effort while maintaining competitive downstream performance. For cybersecurity applications, where expert annotation is both expensive and time-consuming, the proposed framework provides an effective alternative for constructing large-scale training datasets with limited human involvement. Future work should extend this framework by integrating topic modeling and event detection to construct structured CTI events from weakly labeled social media streams.

## REFERENCES

[1] N. Sun, M. Ding, J. Jiang, W. Xu, X. Mo, Y. Tai, and J. Zhang, “Cyber Threat Intelligence Mining for Proactive Cybersecurity Defense: A Survey and New Perspectives,” IEEE Commun. Surv. Tutor., vol. 25, no. 3, pp. 1748–1774, 2023, doi: 10.1109/COMST.2023.3273282.

[2] M. Kriesten, M. Thinyane, and D. Ormrod, “Measures, Metrics, and a Scale for Appraisal of Cyber Threat Intelligence-Informed Decision-Making,” in Proc. 24th Eur. Conf. Cyber Warfare and Security (ECCWS), pp.284–292, 2025, doi: 10.34190/eccws.24.1.3421.

[3] N. Dion´ısio, F. Alves, P. M. Ferreira, and A. Bessani, “Cyberthreat Detection from Twitter using Deep Neural Networks,” in Proc. 2019 Int. Joint Conf. Neural Netw. (IJCNN), pp. 1–8, 2019, doi: 10.1109/ijcnn.2019.8852475.

[4] F. Alves, A. Bettini, P. M. Ferreira, and A. Bessani, “Processing Tweets for Cybersecurity Threat Awareness,” Inf. Syst., vol. 95, pp. 101586, 2021, doi: 10.1016/j.is.2020.101586.

[5] B.-D. Le, G. Wang, M. Nasim, and M. Babar, “Gathering Cyber Threat Intelligence from Twitter Using Novelty Classification,” in Proc. 2019 Int. Conf. Cyberworlds (CW), pp. 316–323, 2019, doi: 10.1109/cw.2019.00058.

[6] A. Rodriguez and K. Okamura, “Generating Real Time Cyber Situational Awareness Information Through Social Media Data Mining,” in Proc. IEEE 43rd Annu. Comput. Softw. Appl. Conf. (COMPSAC), vol. 2, pp. 502–507, 2019, doi: 10.1109/compsac.2019.10256.

[7] O. Alsodi, X. Zhou, R. Gururajan, A. Shrestha, and E. Btoush, “From Tweets to Threats: A Survey of Cybersecurity Threat Detection Challenges, AI-Based Solutions and Potential Opportunities in X,” Appl. Sci., vol. 15, no. 7, pp. 3898, 2025, doi: 10.3390/app15073898.

[8] H.-S. Shin, H.-Y. Kwon, and S.-J. Ryu, “A New Text Classification Model Based on Contrastive Word Embedding for Detecting Cybersecurity Intelligence in Twitter,” Electronics, vol. 9, no. 9, pp. 1527, 2020, doi: 10.3390/electronics9091527.

[9] Y. Fang, J. Gao, Z. Liu, and C. Huang, “Detecting Cyber Threat Event from Twitter Using IDCNN and BiLSTM,” Appl. Sci., vol. 10, no. 17, pp. 5922, 2020, doi: 10.3390/app10175922.

[10] S. A. Murad, A. Dahal, and N. Rahimi, “Multilingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis,” IEEE Trans. Comput. Soc. Syst., vol. 13, no. 2, pp. 1758– 1772, 2025, doi: 10.1109/tcss.2025.3623021.

[11] Y. Park and W. You, “A Pretrained Language Model for Cyber Threat Intelligence,” in Proc. 2023 Conf. Empir. Methods Nat. Lang. Process. (EMNLP): Ind. Track, pp. 113–122, 2023, doi: 10.18653/v1/2023.emnlpindustry.12.

[12] N. Pangakis and S. Wolken, “Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels,” in Proc. 6th Workshop Nat. Lang. Process. Comput. Soc. Sci. (NLP+CSS), pp. 113–131, 2024, doi: 10.18653/v1/2024.nlpcss-1.9.

[13] R. Marinho and R. Holanda, “Automated Emerging Cyber Threat Identification and Profiling Based on Natural Language Processing,” IEEE Access, vol. 11, pp. 58915–58936, 2023, doi: 10.1109/access.2023.3260020.

[14] M. Bayer, T. Frey, and C. Reuter, “Multi-Level Fine-Tuning, Data Augmentation, and Few-Shot Learning for Specialized Cyber Threat Intelligence,” Comput. & Secur., vol. 134, pp. 103430, 2023, doi: 10.1016/j.cose.2023.103430.

[15] B. N. Kebeli and H.-Y. Kwon, “CTI-ANN: Self-Training-Based Annotation With Tailored Augmentation for Cyber Threat Intelligence Posts,” IEEE Trans. Ind. Inform., vol. 22, no. 3, pp. 1996–2007, 2026, doi: 10.1109/TIL2025.3633628.

[16] Y. You, Z. Jiang, K. Zhang, J. Jiang, X. Wang, Z. Zhang, S. Wang, and H. Feng, “TI-Prompt: Towards a Prompt Tuning Method for Fewshot Threat Intelligence Twitter Classification,” in Proc. IEEE 46th Annu. Comput. Softw. Appl. Conf. (COMPSAC), pp. 272–279, 2022, doi: 10.1109/COMPSAC54236.2022.00046.

[17] F. Gilardi, M. Alizadeh, and M. Kubli, “ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks,” Proc. Natl. Acad. Sci., vol. 120, no. 30, pp. e2305016120, 2023, doi: 10.1073/pnas.2305016120.

[18] X. He, Z. Lin, Y. Gong, A.-L. Jin, H. Zhang, C. Lin, J. Jiao, S. M. Yiu, N. Duan, and W. Chen, “AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators,” in Proc. 2024 Conf. North Am. Chapter Assoc. Comput. Linguist.: Hum. Lang. Technol. (NAACL-HLT), pp. 165–190, 2024, doi: 10.18653/v1/2024.naacl-industry.15.

[19] P. Bansal and A. Sharma, “Large Language Models as Annotators: Enhancing Generalization of NLP Models at Minimal Cost,” arXiv preprint arXiv:2306.15766, 2023, doi: 10.48550/arXiv.2306.15766.

[20] H. Kim, K. Mitra, R. L. Chen, S. Rahman, and D. Zhang, “MEGAnno+: A Human-LLM Collaborative Annotation System,” in Proc. 18th Conf. Eur. Chapter Assoc. Comput. Linguist. (EACL): Syst. Demonstr., pp. 168–176, 2024, doi: 10.18653/v1/2024.eacl-demo.18.

[21] X. Wang et al., “Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels,” in Proc. 2024 CHI Conf. Hum. Factors Comput. Syst., pp.1–21, 2024, doi: 10.1145/3613904.3641960.

[22] T. Kuzman and N. Ljubesiˇ c, “LLM Teacher-Student Framework for´ Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification,” IEEE Access, vol. 13, pp.35621– 35633, 2025, doi: 10.1109/ACCESS.2025.3544814.

[23] V. Behzadan, C. Aguirre, A. Bose, and W. Hsu, “Corpus and Deep Learning Classifier for Collection of Cyber Threat Indicators in Twitter Stream,” in Proc. 2018 IEEE Int. Conf. Big Data, pp. 5002–5007, 2018, doi: 10.1109/BigData.2018.8622506.

[24] H. Lughbi, M. Mars, and K. Almotairi, “CybAttT: A Dataset of Cyberattack News Tweets for Enhanced Threat Intelligence,” Data, vol. 9, no. 3, pp. 39, 2024, doi: 10.3390/data9030039.

[25] M. Bayer, P. Kuehn, R. Shanehsaz, and C. Reuter, “CySecBERT: A Domain-Adapted Language Model for the Cybersecurity Domain,” ACM Trans. Priv. Secur., vol. 27, no. 2, Art. no. 18, pp. 1–20, 2024, doi: 10.1145/3652594.

[26] E. Aghaei, S. Jain, P. Arun, and A. Sambamoorthy, “SecureBERT 2.0: Advanced Language Model for Cybersecurity Intelligence,” in Proc. 2025 Annu. Comput. Secur. Appl. Conf. Workshops (ACSAC Workshops), pp. 382–388, 2025, doi: 10.1109/ACSACW69556.2025.00071.

[27] Jackaduma, “SecRoBERTa,” Hugging Face Model Card, Available: https://huggingface.co/jackaduma/SecRoBERTa