# TTLab at StanceEval-2026: A Cloze-Style Prompting Approach for Arabic-Language Stance Detection (CLASP-Ar)

Bhuvanesh Verma, Ali Abusaleh, Alexander Mehler

Text Technology Lab (TTLab),

Goethe University Frankfurt

{verma,a.abusaleh,mehler}@em.uni-frankfurt.de

## Abstract

Arabic-language stance detection remains challenging, and previous shared-task systems have largely relied on multitask learning and ensembles. While these systems achieve state-of-theart performance, their applicability and transferability are limited by the additional complexity introduced by multitask learning. To reduce this complexity, we introduce CLASP-Ar, which reformulates the task as cloze-style masked language modeling. In this approach, the target, predicted sentiment, and text are combined into a single prompt whose [MASK] prediction is restricted to a verbalizer-constrained label vocabulary. When evaluated on the StanceEval-2026: Arabic Stance Detection Shared Task, our proposed approach achieves an F<sub>avg2</sub> score of 71.36% on Track 1 and 74.14% on Track 2 test sets. The code for CLASP-Ar is available at  TTLab at StanceEval-2026

## 1 Introduction

Social media platforms have become integral to daily life, facilitating real-time global connectivity and rapid communication, particularly during emergencies (Muniz-Rodriguez et al., 2020; Ghosh et al., 2018). While these platforms have broadened participation in public discourse, the rapid growth in user engagement has also accelerated the spread of misinformation, contributing to adverse societal consequences such as increased polarization (Van Bavel et al., 2021; Kubin and Von Sikorski, 2021). As a result, there is a critical and growing need for automated tools capable of effectively analyzing social media content.

One computational approach to examining online discourse is stance detection, which aims to infer a user’s stance toward a specific topic from their textual posts. Typically formulated as a pairwise classification task in which a target topic and its corresponding text are jointly processed, stance detection has attracted considerable research interest (Bar-Haim et al., 2017; Wei et al., 2016;

Lynn et al., 2019). However, most of these efforts have focused primarily on English (Zhang et al., 2024). To address this disparity in Arabic, Alturayeif et al. (2024) introduced the Arabic Stance Detection Shared Task in 2024, using the Mawqif dataset (Alturayeif et al., 2022) to benchmark models on posts from X, formerly Twitter, across three distinct topics. StanceEval 2026 (Albalawi et al., 2026a) builds on this foundation by introducing an extended version of the Mawqif dataset. In the previous edition of the task, simple encoderbased architectures achieved highly competitive performance (Badran et al., 2024; Hasanaath and Alansari, 2024).

Building on the success of these encoder architectures, we propose a strategy that leverages masked language modeling (MLM) and the bidirectional contextual representations of encoders while incorporating textual sentiment as an explicit feature. More specifically, we propose CLASP-Ar, an MLM-based approach that combines principles from pattern-exploiting training (PET) and constrained decoding. It restricts the prediction vocabulary for the masked token to the discrete label space of the stance detection task. The constrained generation strategy of CLASP-Ar yields substantial performance improvements over baseline models, achieving an average gain of approximately 2 F<sub>1</sub> points across all stance labels.

The present study is organized into the following sections. First, Section 2 reviews the theoretical background and describes the task at hand along with the dataset. Section 3 then presents the underlying methodology and system architecture. The experimental setup and implementation details are covered in Section 4. Next, Section 3 delivers an in-depth quantitative assessment of the system, further validated via ablation experiments. Lastly, Section 6 summarizes the central contributions to bring the paper to a close.

## 2 Background

Target-specific stance detection has been a prominent research area for over a decade. Early foundational efforts include the SemEval-2016 shared task for English tweets (Mohammad et al., 2016) and a subsequent shared task focusing on Chinese microblogs (Xu et al., 2016). Since then, the field has expanded significantly, yielding datasets across diverse languages including Turkish, Czech, Italian, and Arabic (Zhang et al., 2024). Following the same line, Albalawi et al. (2026a) introduced StanceEval 2026, a shared task for stance detection in Arabic.

## 2.1 Dataset

This shared task uses Mawqif-XT (Albalawi et al., 2026b), an extended version of Mawqif (Alturayeif et al., 2022). It is annotated with stance, sentiment, and sarcasm. Each instance in this dataset consists of a target topic under debate and a corresponding controversial text. The controversial text can express support (favor) for the target topic, opposition (against) to it, or a non-committal (none) stance.

## 2.2 Tracks

This shared task consists of two tracks. Track 1 evaluates models on targets observed during training whereas Track 2 evaluates models on unseen targets. To handle unseen targets under Track 2 and build a robust system for stance detection, we use another dataset. The second dataset is ArabicStance-X (Alkhathlan et al., 2025), which contains 14,477 tweets spanning 17 different topics, including economy, education, health, and religion. Structurally, each major topic is divided into subcategories, which contain lists of controversial texts and their corresponding stance labels. The full details of this dataset along with the shared task dataset are shown in Table 1.

The previous edition of Arabic Stance Detection Shared Task in 2024 (Alturayeif et al., 2024) saw systems utilizing multi-task learning to build robust system. The winning system, AlexUNLP-BH (Badran et al., 2024), used multitask learning over sentiment and sarcasm, weighted cross-entropy with data augmentation, and an ensemble of models trained with different contrastive loss functions. The strongest competitors adopted similarly resource-intensive strategies: the MGKM system (Alghaslan and Almutairy, 2024) was based on fine-tuned large language models such as GPT-3.5-Turbo, whereas StanceCrafters (Hasanaath and Alansari, 2024) combined two BERT-based encoders via an attention mechanism in a multitask setup. These results suggest that auxiliary signals such as sentiment are highly informative for Arabic stance detection. However, top-performing systems obtain these gains at the cost of target-specific models, multiple task heads, or ensembling. In contrast, our approach incorporates sentiment as a simple input feature in a single cloze-style prompt and replaces the standard classification head with a verbalizerconstrained MLM objective, thereby retaining a single model across all targets and labels. As a result, our approach largely avoids the complexity introduced by multitask learning and ensembling, making it more suitable for transfer across application scenarios.

<table><tr><td>Dataset</td><td>Property</td><td>Value</td></tr><tr><td rowspan="5">ArabicStance-X</td><td>Usage</td><td>Pre-training</td></tr><tr><td>Samples</td><td>11,500</td></tr><tr><td>Targets</td><td>17</td></tr><tr><td>Favor</td><td>4,980</td></tr><tr><td>Against/None</td><td>4,452 / 2,068</td></tr><tr><td rowspan="5">Mawqif-XT</td><td>Usage</td><td>Main task</td></tr><tr><td>Train samples</td><td>3,502</td></tr><tr><td>Dev samples</td><td>619</td></tr><tr><td>Train labels</td><td>2148 / 1021 / 333</td></tr><tr><td>Dev labels</td><td>380 / 180 / 59</td></tr><tr><td></td><td>Classes</td><td>Favor/Against/None</td></tr></table>

Table 1: Statistics of the datasets used in this work. ArabicStance-X (Alkhathlan et al., 2025) is used only for intermediate stance pre-training, while Mawqif-XT (Albalawi et al., 2026b) is used for model development and evaluation.

## 3 System Overview

CLASP-Ar’s stance detection methodology draws on prompt-based learning paradigms, particularly pattern-exploiting training (PET) (Schick and Schütze, 2021), and constrained decoding techniques (Willard and Louf, 2023). Specifically, we reformulate the standard sequence classification objective as a cloze-style masked language modeling (MLM) task. Given a target t, a contextual sentiment label s, and an input text x, we construct a prompt sequence $x _ { \mathrm { p r o m p t } }$ containing a single [MASK] token:

$$
\begin{array} { r } { x _ { \mathrm { p r o m p t } } = \mathrm { T a r g e t } ; \ t \mathrm { S e n t i m e n t } ; \ s } \\ { \mathrm { S t a n c e } ; \ [ \mathsf { M A S K } ] \ \mathrm { T e x t } ; \ x } \end{array}
$$

A BERT-based encoder processes this sequence, and stance is predicted by constraining the model’s MLM head to produce outputs only from a predefined subset of vocabulary tokens corresponding to the label space Y = Favor, Against, None. Because the natural-language class names may be split into multiple subword tokens by the tokenizer, we define a verbalizer $v : \mathcal { V } \to \mathcal { V }$ that maps each class to a single-character token corresponding to its initial letter:

![](images/313750d5a24f0c74bd66539e85919f97b19040051302d450dc4164ee79a9c8fb.jpg)  
Figure 1: Overview of CLASP-Ar: target t, sentiment s, and text x form a cloze-style prompt whose [MASK] representation is classified by the MLM head under a verbalizer-constrained softmax over $\begin{array} { r l } { \mathcal { V } _ { c } } & { { } = } \end{array}$ $\{ { \bf \ddot { F } } , { \bf \dot { A } } ^ { \mathrm { , } } , { \bf \dot { N } } ^ { \mathrm { , } } \}$

$$
v ( y ) = { \left\{ \begin{array} { l l } { \cdot \mathrm { F } ^ { \cdot } } & { { \mathrm { i f ~ } } y = \mathrm { F a v o r } } \\ { \cdot \mathrm { A } ^ { \cdot } } & { { \mathrm { i f ~ } } y = { \mathrm { A g a i n s t } } } \\ { \cdot \mathrm { N } ^ { \cdot } } & { { \mathrm { i f ~ } } y = { \mathrm { N o n e } } } \end{array} \right. }
$$

Let $\mathbf { h } _ { \mathtt { [ M A S K ] } }$ denote the final hidden-state representation of the [MASK] token. The probability assigned to a stance label $y \in \mathcal { V }$ is then computed by applying a softmax over the constrained vocabulary set $\mathcal { V } _ { c } = \{ { ^ { \bullet } \mathrm { F } } ^ { \prime } , { ^ { \bullet } \mathrm { A } } ^ { \prime } , { ^ { \bullet } \mathrm { N } } ^ { \prime } \}$

$$
P ( y \mid x _ { \mathrm { p r o m p t } } ) = \frac { \exp ( \mathbf { w } _ { v ( y ) } ^ { \top } \mathbf { h } _ { \mathrm { [ M A S K ] } } ) } { \sum _ { y ^ { \prime } \in \mathcal { V } } \exp ( \mathbf { w } _ { v ( y ^ { \prime } ) } ^ { \top } \mathbf { h } _ { \mathrm { [ M A S K ] } } ) }
$$

where $\mathbf { w } _ { v ( y ) }$ represents the weight vector associated with the token $v ( y )$ in the pre-trained language model’s MLM head. The model is optimized endto-end by minimizing the standard cross-entropy loss between the predicted probabilities and the ground-truth stance label $y ^ { * }$

$$
\mathcal { L } _ { \mathrm { C E } } = - \log P ( y ^ { * } \mid x _ { \mathrm { p r o m p t } } )
$$

## 4 Experimental Setup

To mitigate the class imbalance resulting from the scarcity of the None label, we adopt two strategies. First, we optimize our model using a classweighted cross-entropy loss function to ensure a balanced penalty distribution during training. Secondly, we expand the Mawqif-XT dataset. Specifically, we generate 1,815 additional None instances by randomly pairing controversial texts with mismatched, unrelated targets.

<table><tr><td>Track</td><td>Favg2</td><td>Favg3</td><td>Overall Accuracy</td></tr><tr><td>Track 1</td><td>0.7136</td><td>0.5313</td><td>0.6761</td></tr><tr><td>Track 2</td><td>0.7414</td><td>0.6564</td><td>0.7096</td></tr></table>

Table 2: Performance of CLASP-Ar on the final evaluation data of Track 1 (seen targets) and Track 2 (unseen targets).

For our base architecture, we utilize the aubmindlab/bert-base-arabertv02-twitter encoder (Antoun et al., 2020) based on the experiments shown in Appendix A. To enhance the model’s domain adaptation and task-specific comprehension, we employ a two-stage training paradigm:

• Phase 1: intermediate pre-training: The model is initially trained on the ArabicStance-X dataset (Alkhathlan et al., 2025) for two epochs. Because this auxiliary dataset natively lacks sentiment annotations, we apply SARF (Abusaleh et al., 2026), a dedicated Arabic sentiment analysis model, to generate the requisite sentiment values.

• Phase 2: fine-tuning: Subsequently, the encoder is fine-tuned on our augmented Mawqif-XT dataset for up to 20 epochs. To prevent overfitting, we apply early stopping with a patience of five epochs.

Across both training phases, we maintain a consistent hyperparameter configuration: a maximum sequence length of 512 tokens, a learning rate of 2e-5, and a batch size of 32. We evaluate model performance using overall classification accuracy and class-specific F1-scores for thefavor, against, and none categories. For our aggregate metrics, we use $F _ { \mathrm { a v g 2 } }$ (the macro-average of the two polar classes) as the primary measure, alongside $F _ { \mathrm { a v g 3 } }$ to account for all three classes.

## 5 Results and Discussion

Table 2 presents the official evaluation results across both tracks. Overall, CLASP-Ar demonstrates superior performance on Track 2 (unseen targets), achieving an $F _ { \mathrm { a v g 2 } }$ of 74.14% and an $F _ { \mathrm { a v g 3 } }$ of 65.64%. Conversely, on Track 1 (seen targets), the model yields an $F _ { \mathrm { a v g 2 } }$ of 71.36% but experiences a notable drop in $F _ { \mathrm { a v g 3 } }$ (53.13%). The substantially lower $F _ { \mathrm { a v g 3 } }$ indicates that CLASP-Ar struggles to reliably predict the None class for this unobserved target. Overall, we attribute ${ \mathsf { C L A S P - A r } } ^ { \prime } { \mathsf { s } }$ strong generalization to Track $2 \mathrm { { : } }$ unseen targets primarily to the pre-training stage, where exposure to a diverse array of target domains enhanced zero-shot target adaptability.

<table><tr><td>Method</td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td>Accuracy</td></tr><tr><td>Baseline</td><td>85.75</td><td>71.91</td><td>83.20</td></tr><tr><td colspan="4">Without Pre-training</td></tr><tr><td>Base Setup</td><td>85.95</td><td>73.62</td><td>83.94</td></tr><tr><td>+ Extended None</td><td>86.33</td><td>70.40</td><td>83.94</td></tr><tr><td>+ Class Weights</td><td>85.40</td><td>74.23</td><td>83.36</td></tr><tr><td colspan="4">With Pre-training</td></tr><tr><td>Base Setup</td><td>87.35</td><td>74.87</td><td>85.40</td></tr><tr><td>+ Extended None</td><td>87.34</td><td>74.49</td><td>85.40</td></tr><tr><td>+ Class Weights</td><td>87.26</td><td>75.78</td><td>85.07</td></tr></table>

Table 3: Impact of adding None instances and class weights to CLASP-Ar. Full experiment results with multiple seeds can be found in Appendix B

## 5.1 Impact of Class Imbalance Mitigation Strategies

To systematically evaluate the impact of our classimbalance mitigation strategies, we conducted multi-seed experiments on the training and development sets. Notably, the inclusion of augmented None instances (Extended None in Table 3) mostly improved performance on the Against label, while consistently harming performance on the None label. This could mean the augmentation strategy helped create a much clearer boundary between the Against and None labels. On the other hand, using class-weighted cross-entropy significantly improved model performance on the None label, evident from the large gains in $F _ { \mathrm { n o n e } } \left( \sim + 7 \% \right)$ and $F _ { \mathrm { a v g 3 } } \left( \sim + 2 \% \right)$ over the baseline. Overall, both approaches boosted performance compared with the baseline and the base configuration, underscoring the effectiveness of these techniques.

## 5.2 Impact of Auxilliary Sentiment and Sarcasm Information

To evaluate the contribution of auxiliary task signals present in the dataset, we conduct an ablation study across three random seeds by incrementally integrating sentiment and sarcasm features. As shown in Table 4, incorporating sentiment information yields the most pronounced performance boost, increasing $F _ { \mathrm { a v g 2 } }$ by 1.38 points over the baseline (86.28% vs. 84.90%) and achieving the highest overall accuracy (84.38%). While incorporating both auxiliary features yields the highest $F _ { \mathrm { a v g 3 } }$ (72.50%), sentiment alone provides the strongest consistent gain across metrics, validating its integration into CLASP-Ar. Detailed classand target-specific ablation results are provided in Appendix C.

<table><tr><td>Ablation Setting</td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td>Accuracy</td></tr><tr><td>Baseline (None)</td><td>84.90</td><td>70.83</td><td>82.61</td></tr><tr><td>+ Sentiment</td><td>86.28</td><td>72.36</td><td>84.38</td></tr><tr><td>+ Sarcasm</td><td>84.91</td><td>71.44</td><td>82.98</td></tr><tr><td>+ Both</td><td>85.85</td><td>72.50</td><td>83.74</td></tr></table>

Table 4: Ablation analysis on the development set reporting $F _ { \mathrm { a v g } 2 } , F _ { \mathrm { a v g } 3 }$ , and Overall Accuracy across 4 settings.

## 5.3 Sentiment Source

Since ArabicStance-X nor the official task test data contains no sentiment annotations for data, we generate them with an external Arabic sentiment classifier. We compare two sources: a frozen off-the-shelf sentiment-tuned XLM-RoBERTa based model (Barbieri et al., 2022), and SARF (Abusaleh et al., 2026), a multi-view Arabic sentiment analyzer that fuses surface, stemmed, and rooted morphological representations from a shared MARBERTv2 encoder via a hybrid CNN–BiLSTM– attention head. SARF improves the dev $F _ { \mathrm { a v g 2 } }$ from 80.96% to 81.50%, so we adopt it as the sentiment source.

## 6 Conclusion

In this work, we present CLASP-Ar, a cloze-style masked language modeling framework tailored for stance detection. Through extensive experiments, we demonstrate that incorporating auxiliary task signals, particularly sentiment features, substantially enhances model performance, improving $F _ { \mathrm { a v g 2 } }$ by up to 1.38 points. Furthermore, our targeted class-imbalance mitigation strategies, such as class weighting, effectively boost performance on the underrepresented None class by up to 9 $F _ { 1 }$ points. Evaluated on the official StanceEval-2026 benchmark, CLASP-Ar achieves strong generalization on unseen targets, securing $F _ { \mathrm { a v g 2 } }$ scores of 71.36% on Track 1 and 74.14% on Track 2.

## Limitations

While CLASP-Ar avoids the architectural overhead of multitask and ensemble architectures, it exhibits several limitations. First, performance on the minority None class is notably low and highly volatile, with the standard deviation for $F _ { \mathrm { n o n e } }$ exceeding 10 points in certain configurations. Second, augmenting None instances by pairing texts with unrelated targets represents a naive heuristic that may fail to capture genuine non-committal stances, potentially degrading precision boundaries for the Against class.

## Acknowledgments

This research is partially funded by the German Research Foundation within the Infrastructure Priority Programme New Data Spacesfor the Social Sciences (SPP 2431), Research-driven Infrastructure for Advanced Survey-related Data (CIRCLET) measure project number 539634240 and (Semi-)Automated thematic text classification as a basis for corpus-linguistic value-added services (Project number: 531750631).

## References

Ali Abusaleh, Bhuvanesh Verma, and Alexander Mehler. 2026. TTLab at AraSentEval: okSARF sentiment analysis via root-based fusion for multi-dialectal Arabic. In The 7th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT7) with 5 Shared Tasks, pages 262–268, Palma, Mallorca (Spain). Association for Computational Linguistics.

Rasha Albalawi, Nuha Albadi, Hamzah Luqman, Saad Ezzini, Maram Kurdi, Asma Yamani, Ahmed Ashraf, Maged Al-Shaibani, and Nora Alturayeif. 2026a. Stanceeval-2026: The second stance detection shared task. In Proceedings of the Fourth Arabic Natural Language Processing Conference (ArabicNLP 2026), Budapest, Hungary. Association for Computational Linguistics.

Rasha Albalawi, Nuha Albadi, Hamzah Luqman, Maram Kurdi, Saad Ezzini, Asma Yamani, and Ahmed Ashraf. 2026b. Mawqif-xt: An arabic benchmark dataset for cross-target stance detection. Preprint, arXiv:2608.09539.

Mamoun Alghaslan and Khaled Almutairy. 2024. MGKM at StanceEval2024 fine-tuning large language models for Arabic stance detection. In Proceedings of the Second Arabic Natural Language Processing Conference, pages 816–822, Bangkok, Thailand. Association for Computational Linguistics.

Ali Alkhathlan, Faris Alahmadi, Faris Kateb, and Hend Al-Khalifa. 2025. Constructing and evaluating ArabicStanceX: a social media dataset for arabic stance detection. Front. Artif. Intell., 8:1615800.

Nora Alturayeif, Hamzah Luqman, Zaid Alyafeai, and Asma Yamani. 2024. Stanceeval 2024: The first arabic stance detection shared task. In Proceedings ofthe Second Arabic Natural Language Processing Conference, pages 774–782.

Nora Saleh Alturayeif, Hamzah Abdullah Luqman, and Moataz Aly Kamaleldin Ahmed. 2022. Mawqif: A multi-label Arabic dataset for target-specific stance detection. In Proceedings ofthe The Seventh Arabic Natural Language Processing Workshop (WANLP), pages 174–184, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Wissam Antoun, Fady Baly, and Hazem Hajj. 2020. Arabert: Transformer-based model for arabic language understanding. In LREC 2020 Workshop Language Resources and Evaluation Conference 11–16 May 2020, page 9.

Mohamed Badran, Mo’men Hamdy, Marwan Torki, and Nagwa M El-Makky. 2024. Alexunlp-bh at stanceeval2024: Multiple contrastive losses ensemble strategy with multi-task learning for stance detection in arabic. In Proceedings ofthe Second Arabic Natural Language Processing Conference, pages 823–827.

Roy Bar-Haim, Indrajit Bhattacharya, Francesco Dinuzzo, Amrita Saha, and Noam Slonim. 2017. Stance classification of context-dependent claims. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 251–261.

Francesco Barbieri, Luis Espinosa Anke, and Jose Camacho-Collados. 2022. XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 258–266, Marseille, France. European Language Resources Association.

Saptarshi Ghosh, Kripabandhu Ghosh, Debasis Ganguly, Tanmoy Chakraborty, Gareth JF Jones, Marie-Francine Moens, and Muhammad Imran. 2018. Exploitation of social media for emergency relief and preparedness: Recent research and trends. Information Systems Frontiers, 20(5):901–907.

Ahmed Hasanaath and Aisha Alansari. 2024. Stancecrafters at stanceeval2024: Multi-task stance detection using bert ensemble with attention based aggregation. In Proceedings ofthe Second Arabic Natural Language Processing Conference, pages 811–815.

Emily Kubin and Christian Von Sikorski. 2021. The role of (social) media in political polarization: a systematic review. Annals ofthe International Communication Association, 45(3):188–206.

Veronica Lynn, Salvatore Giorgi, Niranjan Balasubramanian, and H Andrew Schwartz. 2019. Tweet classification without the tweet: An empirical examination of user versus document attributes. In Proceedings of the third workshop on natural language processing and computational social science, pages 18–28.

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. Semeval-2016 task 6: Detecting stance in tweets. In Proceedings ofthe 10th international workshop on semantic evaluation (SemEval-2016), pages 31–41.

Kamalich Muniz-Rodriguez, Sylvia K Ofori, Lauren C Bayliss, Jessica S Schwind, Kadiatou Diallo, Manyun Liu, Jingjing Yin, Gerardo Chowell, and Isaac Chun-Hai Fung. 2020. Social media use in emergency response to natural disasters: a systematic review with a public health perspective. Disaster medicine and public health preparedness, 14(1):139–149.

Timo Schick and Hinrich Schütze. 2021. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th conference ofthe European chapter ofthe associationfor computational linguistics: main volume, pages 255–269.

Jay J Van Bavel, Steve Rathje, Elizabeth Harris, Claire Robertson, and Anni Sternisko. 2021. How social media shapes polarization. Trends in cognitive sciences, 25(11):913–916.

Wan Wei, Xiao Zhang, Xuqin Liu, Wei Chen, and Tengjiao Wang. 2016. pkudblab at semeval-2016 task 6: A specific convolutional neural network system for effective stance detection. In Proceedings of the 10th international workshop on semantic evaluation (SemEval-2016), pages 384–388.

Brandon T Willard and Rémi Louf. 2023. Efficient guided generation for large language models. arXiv preprint arXiv:2307.09702.

Ruifeng Xu, Yu Zhou, Dongyin Wu, Lin Gui, Jiachen Du, and Yun Xue. 2016. Overview of nlpcc shared task 4: Stance detection in chinese microblogs. In International Conference on Computer Processing of Oriental Languages, pages 907–916. Springer.

Bowen Zhang, Genan Dai, Fuqiang Niu, Nan Yin, Xiaomao Fan, Senzhang Wang, Xiaochun Cao, and Hu Huang. 2024. A survey of stance detection on social media: New directions and perspectives. arXiv preprint arXiv:2409.15690.

## A Leveraging ArabicStance-X for Stance Detection

We investigate two strategies for incorporating ArabicStance-X into the training pipeline. In the pretrain setting, the encoder is first fine-tuned on ArabicStance-X and subsequently trained on Mawqif-XT. In the joint setting, ArabicStance-X

instances are mixed with the Mawqif-XT training data. Table 5 summarizes the results.
<table><tr><td>Encoder</td><td>Plain</td><td>Pretrain</td><td>Joint</td></tr><tr><td>AraBERT  $_ \mathrm { l a r g e - t w i t t e r } ^ { \dagger }$ </td><td>84.13</td><td>85.24</td><td>84.42</td></tr><tr><td>MARBERTv2</td><td>82.65</td><td>82.35</td><td>82.49</td></tr><tr><td>CAMeLBERT-mix</td><td>79.50</td><td>80.18</td><td>81.15</td></tr></table>

Table 5: Impact of transfer learning strategies on development $F _ { \mathrm { a v g 2 } }$ . Pretrain denotes intermediate finetuning on ArabicStance-X followed by training on Mawqif-XT. Joint denotes training on the union of Mawqif-XT and ArabicStance-X. <sup>†</sup> Original sharedtask baseline using the large model.

## B Class Imbalance Mitigation

We perform multi seed experiment to measure the impact of two strategies we adopted for class imbalance. We train model with base configuration, with extended None instances and adding class weights with three seeds (42,123,456). We evaluate these systems on development set (Table 6). Similarly we also report the impact of these strategies on different targets in Table 7.

## B.1 Data Augmentation vs. Loss Weighting

A comparison of data-level (Extended None) and loss-level (Class Weights) approaches shows a clear trade-off between detecting polar stances $( F _ { \mathrm { a v g } 2 } )$ and overall multi-class performance $( F _ { \mathrm { a v g 3 } } )$ . As we can see from Table 6, Class Weights consistently improves the minority none class, delivering the highest $F _ { \mathrm { n o n e } }$ (52.83% with pre-training) with low cross-seed variance. In contrast, data augmentation (Extended None) reduces $F _ { \mathrm { n o n e } }$ (falling to 38.55% without pre-training) and leads to pronounced instability $( \sigma = 1 0 . 8 1 )$ . This indicates that the augmented none instances inject feature noise, complicating convergence within the minority class space.

However, Extended None effectively isolates the polar stance classes. By artificially saturating the none distribution during training, the model establishes a stricter decision boundary for the favor and against categories, yielding the highest $F _ { \mathrm { a v g 2 } }$ (86.33% without pre-training). Class weighting demonstrates the inverse effect, trading fractional $F _ { \mathrm { a v g 2 } }$ reductions for global macro-average $( F _ { \mathrm { a v g 3 } } )$ improvements.

At the target level (Table 7), strategy efficacy strongly correlates with the underlying class distributions. Extended None optimally resolves the

<table><tr><td>Method</td><td> $F _ { \mathbf { f a v o r } }$ </td><td> $F _ { \mathrm { a g a i n s t } }$ </td><td> $F _ { \mathbf { n o n e } }$ </td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td>Accuracy</td></tr><tr><td>Baseline</td><td> $8 9 . 7 7$ </td><td>81.72</td><td> $4 4 . 2 5$ </td><td>85.75</td><td>71.91</td><td>83.20</td></tr><tr><td colspan="7">CLASP-Ar</td></tr><tr><td colspan="7">Without Pre-training</td></tr><tr><td>Base Setup</td><td> $8 9 . 9 9 \pm 0 . 8 4$ </td><td> $8 1 . 9 1 \pm 1 . 0 3$ </td><td> $4 8 . 9 6 \pm 4 . 6 0$ </td><td> $8 5 . 9 5 \pm 0 . 8 2$ </td><td> $7 3 . 6 2 \pm 1 . 7 8$ </td><td> $8 3 . 9 4 \pm 0 . 8 7$ </td></tr><tr><td>+ Extended None</td><td> $8 9 . 9 4 \pm 0 . 4 1$ </td><td> $8 2 . 7 3 \pm 0 . 2 2$ </td><td> $3 8 . 5 5 \pm 1 0 . 8 1$ </td><td> $8 6 . 3 3 \pm 0 . 3 0$ </td><td> $7 0 . 4 0 \pm 3 . 6 7$ </td><td> $8 3 . 9 4 \pm 0 . 4 2$ </td></tr><tr><td>+ Class Weights</td><td> $8 9 . 6 6 \pm 1 . 2 9 $ </td><td> $8 1 . 1 4 \pm 0 . 6 7$ </td><td> $5 1 . 8 8 \pm 2 . 1 3 $ </td><td> $8 5 . 4 0 \pm 0 . 8 6$ </td><td> $7 4 . 2 3 \pm 0 . 8 5$ </td><td> $8 3 . 3 6 \pm 1 . 3 3 $ </td></tr><tr><td colspan="7">With Pre-training</td></tr><tr><td>Base Setup</td><td> $9 0 . 7 1 \pm 0 . 2 6 $ </td><td> $8 4 . 0 0 \pm 0 . 6 9$ </td><td> $4 9 . 9 1 \pm 5 . 0 0$ </td><td> $8 7 . 3 5 \pm 0 . 4 6$ </td><td> $7 4 . 8 7 \pm 1 . 7 3$ </td><td> $8 5 . 4 0 \pm 0 . 4 9$ </td></tr><tr><td>+ Extended None</td><td> ${ \bf 9 0 . 8 3 \pm 0 . 2 2 }$ </td><td> $8 3 . 8 5 \pm 0 . 5 5$ </td><td> $4 8 . 7 8 \pm 6 . 0 7$ </td><td> ${ \bf 8 7 . 3 4 \pm 0 . 2 6 }$ </td><td> $7 4 . 4 9 \pm 1 . 9 8$ </td><td> $\mathbf { 8 5 . 4 0 \pm 0 . 2 7 }$ </td></tr><tr><td>+ Class Weights</td><td> $9 0 . 4 3 \pm 0 . 3 0$ </td><td> $\mathbf { 8 4 . 0 9 \pm 1 . 2 1 }$ </td><td> ${ \pm 2 . 8 3 \pm 1 . 7 5 }$ </td><td> $8 7 . 2 6 \pm 0 . 5 7$ </td><td> ${ \bf 7 5 . 7 8 \pm 0 . 9 3 }$ </td><td> $8 5 . 0 7 \pm 0 . 6 2$ </td></tr></table>

Table 6: Overall stance detection performance comparison. Our implementation is presented as mean and standard deviation over 5 runs using different seeds. All values are reported as percentages (%). Baseline uses same encoder as for the CLASP-Ar which is AraBERT-large-twitter
<table><tr><td rowspan="2">Method</td><td colspan="2">Covid Vaccine</td><td colspan="2">Digital Trans.</td><td colspan="2">Women Emp.</td></tr><tr><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td></tr><tr><td colspan="7">CLASP-Ar</td></tr><tr><td>Without Pre-training</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base Setup</td><td> $8 0 . 7 8 \pm 1 . 0 3$ </td><td> $6 4 . 5 2 \pm 2 . 8 3 $ </td><td> $8 6 . 3 0 \pm 2 . 8 9$ </td><td> $8 0 . 9 9 \pm 2 . 4 2$ </td><td> $8 8 . 8 7 \pm 1 . 1 9$ </td><td> $7 0 . 4 8 \pm 4 . 4 4$ </td></tr><tr><td>+ Extended None</td><td> $8 2 . 5 2 \pm 1 . 0 0$ </td><td> $6 4 . 6 3 \pm 2 . 3 1 $ </td><td> $8 4 . 8 2 \pm 2 . 1 0$ </td><td> $7 4 . 1 7 \pm 7 . 7 7$ </td><td> $8 8 . 1 5 \pm 0 . 5 5$ </td><td> $6 7 . 5 0 \pm 4 . 6 7$ </td></tr><tr><td>+ Class Weights</td><td> $7 8 . 8 6 \pm 1 . 7 1 $ </td><td> $6 4 . 3 8 \pm 1 . 6 5$ </td><td> $\mathbf { 8 7 . 0 8 \pm 0 . 9 2 }$ </td><td> ${ \bf 8 1 . 8 9 \pm 1 . 5 3 }$ </td><td> $8 8 . 8 2 \pm 0 . 9 9$ </td><td> $7 4 . 8 9 \pm 1 . 4 7$ </td></tr><tr><td colspan="7">With Pre-training</td></tr><tr><td>Base Setup</td><td> $8 2 . 3 0 \pm 1 . 0 7$ </td><td> $6 6 . 5 9 \pm 3 . 0 0$ </td><td> $8 6 . 1 5 \pm 2 . 7 5$ </td><td> $7 9 . 9 2 \pm 3 . 4 1$ </td><td> ${ \bf 9 0 . 4 7 \pm 0 . 7 2 }$ </td><td> $7 5 . 1 5 \pm 3 . 5 5$ </td></tr><tr><td>+ Extended None</td><td> $\mathbf { 8 4 . 2 3 \pm 1 . 0 3 }$ </td><td> ${ \bf 6 9 . 4 4 \pm 0 . 8 6 }$ </td><td> $8 4 . 9 8 \pm 1 . 6 5$ </td><td> $7 7 . 9 2 \pm 3 . 3 1$ </td><td> $8 9 . 1 9 \pm 0 . 9 3 $ </td><td> $7 1 . 8 1 \pm 3 . 9 3$ </td></tr><tr><td>+ Class Weights</td><td> $8 2 . 8 7 \pm 0 . 9 6$ </td><td> $6 8 . 3 9 \pm 1 . 0 8$ </td><td> $8 5 . 0 1 \pm 1 . 8 6$ </td><td> $7 9 . 5 5 \pm 2 . 3 0$ </td><td> $9 0 . 4 6 \pm 0 . 8 9$ </td><td> $7 7 . 2 0 \pm 3 . 0 0$ </td></tr></table>

Table 7: Target-specific stance detection performance comparison. Results are shown as mean and standard deviation over 5 runs. All values are reported as percentages (%).

Covid Vaccine target, which features a balanced polar distribution (∼43%favor and against), maximizing both $F _ { \mathrm { a v g 2 } }$ (84.23%) and $F _ { \mathrm { a v g 3 } } ( 6 9 . 4 4 \% )$ when pre-trained. Conversely, for heavily skewed targets where the none class is an extreme minority (e.g., ∼5% in Women Empowerment and ∼10% in Digital Transformation), Class Weights substantially outperforms augmentation by preventing minority class collapse. Ultimately, class weighting is the superior mechanism for highly imbalanced distributions to maximize global macro-performance, whereas data augmentation is primarily advantageous for isolating stances in targets with balanced polar classes.

## C Auxiliary Feature Ablations

The shared task dataset contained additional information like sentiment and sarcasm. We conducted 3 seeds (42, 123, 456) experiments to determine which feature can be helped for Stance detection. Table 8 shows results on experiments on development set.

<table><tr><td>Ablation</td><td> $F _ { \mathbf { f a v o r } }$ </td><td> $F _ { \mathrm { a g a i n s t } }$ </td><td> $F _ { \mathbf { n o n e } }$ </td><td> $F _ { \mathbf { a v g } 2 }$ </td><td> $F _ { \mathbf { a v g 3 } }$ </td><td>Accuracy</td></tr><tr><td>None</td><td> $8 9 . 4 1 \pm 0 . 3 0$ </td><td> $8 0 . 3 9 \pm 2 . 4 7$ </td><td> $4 2 . 6 7 \pm 1 . 2 1$ </td><td> $8 4 . 9 0 \pm 1 . 3 8$ </td><td> $7 0 . 8 3 \pm 0 . 6 1$ </td><td> $8 2 . 6 1 \pm 1 . 0 4$ </td></tr><tr><td>Sentiment</td><td> ${ \bf 9 0 . 9 1 \pm 0 . 2 9 }$ </td><td> ${ \bf 8 1 . 6 6 \pm 0 . 6 9 }$ </td><td> $4 4 . 5 1 \pm 4 . 7 7$ </td><td> ${ \bf 8 6 . 2 8 \pm 0 . 2 0 }$ </td><td> $7 2 . 3 6 \pm 1 . 7 3$ </td><td> ${ \bf 8 4 . 3 8 \pm 0 . 4 9 }$ </td></tr><tr><td>Sarcasm</td><td> $8 9 . 8 7 \pm 0 . 7 8$ </td><td> $7 9 . 9 5 \pm 1 . 4 8$ </td><td> $4 4 . 5 0 \pm 4 . 2 3$ </td><td> $8 4 . 9 1 \pm 0 . 9 4$ </td><td> $7 1 . 4 4 \pm 1 . 5 7$ </td><td> $8 2 . 9 8 \pm 0 . 9 2$ </td></tr><tr><td>Both</td><td> $9 0 . 2 8 \pm 1 . 2 0$ </td><td> $8 1 . 4 2 \pm 1 . 3 9$ </td><td> ${ \pm 5 . 8 0 \pm 1 . 0 3 }$ </td><td> $8 5 . 8 5 \pm 1 . 2 8$ </td><td> ${ \bf 7 2 . 5 0 \pm 1 . 1 0 }$ </td><td> $8 3 . 7 4 \pm 1 . 3 5$ </td></tr></table>

Table 8: Comprehensive evaluation showing label-specific $F _ { 1 }$ scores $( F _ { \mathrm { f a v o r } } , F _ { \mathrm { a g a i n s t } } , F _ { \mathrm { n o n e } } ) _ { \mathrm { : } }$ , macro-averaged scores $( F _ { \mathrm { a v g } 2 } , F _ { \mathrm { a v g } 3 } )$ , and overall Accuracy across multi-seed experiments on the development set.