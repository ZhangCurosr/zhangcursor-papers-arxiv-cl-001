# Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model

Artem Suslov Hokkai-Gakuen University, Sapporo, Japan

Takehiro Hashimoto Chuo University, Tokyo, Japan

## Abstract

The scarcity of human-annotated literary corpora limits the development of domain-specific sentiment analysis models. This study investigates whether LLM-generated annotations can supervise smaller models for literary sentiment analysis. We fine-tune five BERT models using ordinal regression on 10,889 literary sentences annotated by Gemma 4 31B on a ten-point sentiment scale. The models are evaluated against 1,923 human-annotated sentences from Hemingway’s The Old Man and the Sea, alongside the teacher model and conventional baselines.

BERT-large achieves a Kendall’s τ of 0.589, approaching the teacher’s 0.604. Although a constant-neutral predictor achieves the lowest overall mean absolute error (MAE), BERTbase outperforms the teacher on non-neutral sentences, achieving an MAE of 0.726 versus 1.116. These findings demonstrate the potential of LLM-generated supervision for developing smaller literary sentiment models, while highlighting the limitations of aggregate error metrics on predominantly neutral datasets.

## 1 Introduction

Sentiment analysis (SA) has become an established method in Digital Humanities (DH), particularly for investigating emotional patterns and narrative sentiment arcs in literary texts (Öhman et al., 2024). However, most available sentiment classifiers were trained on non-literary domains, such as product reviews and social media (Barbieri et al., 2020), while human-annotated literary datasets remain scarce. Consequently, their application to literary texts presents challenges arising from implicit and context-dependent emotional expressions (Bizzoni and Feldkamp, 2024; Feldkamp et al., 2024).

Another limitation concerns sentiment representation. Most existing classifiers predict nominal categories, such as negative, neutral, and positive, which are often converted into continuous valence scores for literary analysis (Bizzoni and Feldkamp, 2024). Ordinal regression provides an alternative that explicitly models the ordering of sentiment categories (Cao et al., 2020). Meanwhile, LLMs offer opportunities for generating fine-grained sentiment annotations, although their repeated use for inference across large literary corpora can be computationally demanding (Ashir and Taha, 2024).

This study investigates whether LLM-generated annotations can provide effective supervision for smaller, domain-specific sentiment models. Specifically, we address two research questions:

• (RQ1) Can BERT models trained on LLMgenerated ordinal annotations generalize to independently human-annotated literary text?

• (RQ2) How does predictive performance vary across different BERT configurations and sentiment ranges?

To address these questions, we make three contributions: (1) we construct a corpus of 10,889 literary sentences annotated by a 31B-parameter LLM on a ten-point sentiment scale; (2) we fine-tune five BERT models using a shared ordinal regression objective; and (3) we evaluate their generalization against independent human annotations, comparing their performance with the teacher model, generaldomain classifiers, and a constant-neutral baseline. Our results demonstrate that the fine-tuned models approach the teacher’s rank correlation with human judgments while achieving lower prediction errors on non-neutral literary sentences.

## 2 Related Work

SA has become an established method in computational literary studies, particularly for examining narrative sentiment arcs and emotional patterns in literary texts (Öhman et al., 2024). Transformerbased models, including BERT and RoBERTa, have been increasingly employed for these tasks.

However, their effectiveness remains constrained by the complexity of literary language, where sentiment is frequently implicit, contextual, or conveyed through figurative expressions (Bizzoni and Feldkamp, 2024; Feldkamp et al., 2024).

Recent advances in LLMs have enabled their use both as sentiment classifiers (Krugmann and Hartmann, 2024; Zhang et al., 2024; Miah et al., 2024; Wang et al., 2024; Coutinho et al., 2026) and as annotators for constructing training datasets (Hellwig et al., 2026; Huang and Simpson, 2026; Zhang et al., 2025; Xie et al., 2025; Shvets, 2025; Ashir and Taha, 2024; Pangakis and Wolken, 2024). Previous experiments demonstrated that LLMs with approximately 8B parameters or more can achieve rank correlations comparable to or higher than conventional Transformer sentiment classifiers on literary texts (Suslov and Hashimoto, 2026). However, direct LLM inference is substantially more computationally demanding than using smaller encoderbased models.

In DH, custom SA models are being predominantly fine-tuned through nominal classification. A recently developed fiction-oriented RoBERTa model converts continuous sentiment annotations into three nominal classes and reports evaluations across several literary datasets (Pianzola, 2026). Ordinal regression offers an alternative by explicitly preserving the ordering of sentiment categories (Cao et al., 2020; Biasetton et al., 2026). Biasetton et al. demonstrated its application to BERT-based sentiment prediction using a five-point scale, although outside the literary domain. Building on these approaches, we investigate whether LLMgenerated annotations can provide effective supervision for a ten-level ordinal BERT model trained specifically for literary SA and evaluated against independent human judgments.

## 3 Data: LLM-Annotated Literary Sentiment

The training corpus consists of 10,889 sentences extracted from literary works in the Project Gutenberg collection. To ensure diversity across literary works, the extraction procedure randomly sampled one sentence per novel. To reduce the inclusion of paratextual material, such as titles, tables of contents, and section headings, we excluded very short sentences and structural entries (e.g., “Chapter $X ^ { \prime }$ “Section X”, and “Part X”), as well as the opening and closing sentences of each work.

The resulting corpus was annotated using Gemma 4 31B (Gemma Team, 2026), which assigned each sentence a sentiment score on a tenpoint scale, ranging from 1 (strongly negative) to 10 (strongly positive), with scores of 4–6 representing the near-neutral range. The annotation prompt, generation parameters, and post-processing procedures are described in Appendix A.

For independent evaluation, we employed the human-annotated corpus of Ernest Hemingway’s The Old Man and the Sea, introduced by Bizzoni and Feldkamp (2024). This dataset comprises 1,923 sentences evaluated by two human annotators on a comparable ten-point valence scale. The evaluation corpus was excluded from model training.

## 4 Method: Ordinal BERT

We fine-tuned five BERT configurations using an ordinal regression architecture inspired by McCullagh (1980); Cao et al. (2020); Wang et al. (2026). Unlike conventional sentiment classifiers that treat sentiment categories as independent labels, ordinal regression explicitly preserves their inherent ordering. This approach is particularly suitable for literary SA, where valence is expressed on a graded scale rather than through discrete polarities.

Given an input sentence x, BERT produces a contextual representation, which is projected onto a scalar latent sentiment score $s ( x )$ . For $K = 1 0$ sentiment levels, the model learns nine monotonically ordered thresholds $\theta _ { 1 } < \cdots < \theta _ { 9 }$ , enforced through softplus parameterization. Each threshold defines the cumulative probability:

$$
P ( Y > k \mid x ) = \sigma ( s ( x ) - \theta _ { k } ) , \quad k = 1 , \ldots , 9 ,\tag{1}
$$

where $\sigma$ denotes the sigmoid function. Training labels are converted into cumulative binary targets and optimized using binary cross-entropy loss. The ordered thresholds ensure that predictions remain rank-consistent.

At inference, the expected sentiment score is calculated as:

$$
\hat { y } = 1 + \sum _ { k = 1 } ^ { 9 } P ( Y > k \mid x ) .\tag{2}
$$

This produces a continuous sentiment estimate on the original 1–10 scale while preserving the ordinal structure of the training labels, without relying on post-hoc transformations of nominal sentiment probabilities.

## 5 Experimental Setup

We fine-tuned five pretrained BERT models of different sizes: Large, Base, Medium, Small, and Mini, described in Appendix B. The LLMannotated corpus was divided into training (n = 8, 711), validation (n = 1, 089), and test (n = 1, 089) sets. All models used identical optimization hyperparameters, detailed in Appendix C.

Baselines. We compared the models against Gemma 4 31B (teacher), two general-domain RoBERTa classifiers (twt\_roberta and twt\_roberta\_xlm) (Barbieri et al., 2020, 2022), and a constant-neutral baseline (score = 5). RoBERTa model scores were linearly scaled to 1-10 range. BERT-tiny also converged to constant predictions and is reported alongside this baseline.

Evaluation. Models were evaluated on 1,923 sentences from Hemingway’s The Old Man and the Sea, using the average scores of two human annotators. This corpus was excluded from training and validation. We calculated MAE, Kendall’s τ, and precision and accuracy within one sentiment level (±1). MAE and τ used original scores; tolerancebased precision was macro-averaged across observed predicted values.

To account for the predominance of neutral scores, we additionally evaluated 305 non-neutral sentences, defined by human scores below 4 or above 6.

## 6 Results

## 6.1 Non-Literary Domain SA

Table 1 presents the performance of the fine-tuned BERT models, the teacher LLM, general-domain RoBERTa classifiers, and the constant-neutral baseline against human sentiment annotations <sup>1</sup>.

Overall performance. All five fine-tuned BERT models achieved lower MAE than the generaldomain RoBERTa baselines. Among the student models, BERT-small obtained the lowest MAE (0.717) and the highest accuracy within one sentiment level (0.837).

However, none of the evaluated models outperformed the constant-neutral baseline, which achieved an MAE of 0.597 and an accuracy of 0.841. This result reflects the predominance of near-neutral sentiment in the evaluation corpus, where 84.1% of human reference scores fell within the interval from 4 to 6.

Performance on non-neutral sentences. To examine predictive performance beyond the dominant neutral range, we separately evaluated 305 sentences with human sentiment scores below 4 or above 6.

All five fine-tuned BERT models outperformed both the teacher LLM and the general-domain RoBERTa classifiers in MAE and accuracy within one sentiment level on this subset.

BERT-base achieved the strongest performance, with an MAE of 0.726 and an accuracy of 0.803, compared with 1.116 and 0.600 for the teacher model, respectively. The constant-neutral baseline achieved an MAE of 1.962 and an zero accuracy on this subset.

These results demonstrate that the performance advantage of the constant-neutral baseline on the complete corpus does not extend to sentences expressing sentiment outside the near-neutral range.

Rank correlation and model size. Rank-based evaluation revealed a different pattern. The teacher achieved the highest Kendall’s τ (0.604), followed by BERT-large (0.589) and BERT-base (0.580), both exceeding the general-domain RoBERTa baselines (0.497 and 0.505). Smaller BERT configurations showed weaker agreement, with τ declining from 0.489 for BERT-medium to 0.402 for BERTmini.

## 6.2 Literary Domain SA

Overall, performance depended on the evaluation metric and sentiment distribution. BERT-small achieved the lowest overall MAE among students, BERT-base performed best on non-neutral sentences, and BERT-large most closely approached the teacher’s rank correlation with human judgments.

## 7 Discussion

## 7.1 LLM Supervision and Ordinal Modeling

Trained on LLM-annotated sentences sampled across different literary works, the BERT models were evaluated on Hemingway’s stylistically distinctive The Old Man and the Sea. Despite this difference, BERT-large and BERT-base approached the teacher’s rank correlation with human judgments (τ = 0.589 and 0.580 versus 0.604), while BERT-base outperformed the teacher on nonneutral MAE (0.726 versus 1.116). These findings suggest that LLM-generated supervision can transfer meaningful sentiment information from a stylistically diverse training corpus to a distinct literary work.

<table><tr><td rowspan="2">Model</td><td colspan="4">All sentences (n = 1, 923)</td><td colspan="3">Non-neutral sentences (n = 305)</td></tr><tr><td>MAE</td><td>Kendall&#x27;s τ</td><td>Prec. (±1)</td><td>Acc. (±1)</td><td>MAE</td><td>Prec. (±1)</td><td>Acc. (±1)</td></tr><tr><td>Gemma 4 31B (teacher)</td><td>0.822</td><td>0.604</td><td>0.460</td><td>0.741</td><td>1.116</td><td>0.522</td><td>0.600</td></tr><tr><td>twt_roberta</td><td>0.972</td><td>0.497</td><td>0.176</td><td>0.695</td><td>1.656</td><td>0.306</td><td>0.223</td></tr><tr><td>twt_roberta_xlm</td><td>1.261</td><td>0.505</td><td>0.125</td><td>0.530</td><td>1.613</td><td>0.301</td><td>0.308</td></tr><tr><td>BERT-large</td><td>0.769</td><td>0.589</td><td>0.614</td><td>0.787</td><td>0.887</td><td>0.571</td><td>0.711</td></tr><tr><td>BERT-base</td><td>0.733</td><td>0.580</td><td>0.723</td><td>0.815</td><td>0.726</td><td>0.698</td><td>0.803</td></tr><tr><td>BERT-medium</td><td>0.752</td><td>0.489</td><td>0.659</td><td>0.808</td><td>0.998</td><td>0.669</td><td>0.675</td></tr><tr><td>BERT-small</td><td>0.717</td><td>0.468</td><td>0.708</td><td>0.837</td><td>0.982</td><td>0.668</td><td>0.698</td></tr><tr><td>BERT-mini</td><td>0.766</td><td>0.402</td><td>0.732</td><td>0.829</td><td>0.969</td><td>0.692</td><td>0.708</td></tr><tr><td>BERT-tiny (constant-neutral)</td><td>0.597</td><td></td><td>0.841</td><td>0.841</td><td>1.962</td><td>0.000</td><td>0.000</td></tr></table>

Table 1: Performance against human annotations (n = 1,923) and on non-neutral sentences (n = 305; scores <4 or >6). MAE denotes mean absolute error; precision and accuracy use a tolerance of ±1. Bold indicates the best result per metric. Kendall’s τ is undefined for constant predictions.

Nevertheless, evaluation on a single novel limits claims of broader stylistic generalization. Furthermore, the study did not examine alternative architectures, such as conventional regression or three-class nominal classification, or assess training variability across random seeds. Consequently, the specific contribution of ordinal modeling and the stability of the results remain to be established.

## 7.2 Sentiment Distribution and Evaluation

The results highlight a common challenge in SA: imbalanced sentiment distributions can favor models predicting the dominant neutral category (Jongeling et al., 2017). In our evaluation corpus, 84.1% of human annotations fell within the near-neutral range of 4–6. Consequently, a constant-neutral predictor achieved the lowest overall MAE, despite capturing no sentiment variation.

However, identifying non-neutral passages may have greater heuristic value for literary analysis than accurately predicting neutral sentences. On the non-neutral subset, all fine-tuned BERT models outperformed the teacher LLM and general-domain RoBERTa baselines. These findings suggest that aggregate error metrics may obscure models’ ability to estimate non-neutral sentiment, limiting their usefulness for analyzing emotional variation in literary narratives.

One possible improvement involves modifying the training objective to penalize errors on underrepresented sentiment categories more heavily. Class-balanced weighting (Cui et al., 2019)

and focal loss (Lin et al., 2020) offer established approaches, while distribution-aware methods address imbalance in continuous prediction (Yang et al., 2021). Adapting these approaches to ordinal sentiment modeling remains a direction for future research.

## 7.3 Model Size and Computational Efficiency

Model size influenced predictive performance, with larger BERT variants generally achieving stronger rank correlations. BERT-base performed best on non-neutral sentences, while BERT-large most closely approached the teacher’s rank correlation. Thus, model selection should consider whether absolute sentiment estimation or relative ordering is more important.

Transferring LLM-generated annotations to smaller encoder models may also reduce computational requirements for large-scale literary analysis. However, inference efficiency was not measured directly, and these potential benefits require experimental verification.

## 8 Conclusion

This study demonstrates that LLM-generated annotations can provide effective supervision for literary sentiment analysis, at least within the examined configuration. Using 10,889 LLM-annotated sentences, we fine-tuned BERT models of different sizes and evaluated them against independent human judgments.

The larger models approached the teacher’s rank correlation (BERT-large: $\tau = 0 . 5 8 9 ;$ Gemma 4 31B: 0.604), while BERT-base outperformed the teacher on non-neutral sentences (MAE: 0.726 versus 1.116). Even compact BERT configurations demonstrated meaningful predictive performance, suggesting that smaller encoders can effectively learn from LLM-generated annotations.

Nevertheless, the constant-neutral baseline achieved the lowest overall MAE, highlighting the influence of sentiment imbalance. Further research should examine alternative training objectives, prediction calibration, and generalization across literary genres.

## Limitations

The model was trained on randomly sampled sentences from Project Gutenberg, annotated using an LLM. No additional filtering was applied to exclude potentially bias-inducing contexts. Consequently, the resulting dataset may inherit biases from the annotating LLM.

In particular, we observed that Gemma 4 31B tended to assign disproportionately negative sentiment scores to sentences describing sensitive topics, such as violence, death, or criminal acts, even when their narrative context conveyed positive or contrasting sentiment. This suggests that the model may associate certain semantic topics with negative valence regardless of their contextual meaning.

This was the only specific annotation bias identified in the present study; however, other systematic biases cannot be excluded.

## Declaration of Generative AI Use

OpenAI’s ChatGPT 5.6 model was used for proofreading and minor stylistic refinements (paraphrasing and summarization under authors’ supervision) and assisting with LAT X layout adjustment. Also, the same model was used for debugging assistance and code review.

## References

Abubakar M. Ashir and Mohammed Abdulghani Taha. 2024. BERT with an augmented cross-attention decoder (BERT-ACD) for binary and fine-grained multiband sentiment detection. Intelligenza Artificiale: The international journal ofthe AIxIA, 19(1):3–16.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644–1650, Online. Association for Computational Linguistics.

Francesco Barbieri, Luis Espinosa Anke, and Jose Camacho-Collados. 2022. XLM-T: Multilingual language models in twitter for sentiment analysis and beyond. In Proceedings of the Thirteenth Language

Resources and Evaluation Conference, pages 258– 266. European Language Resources Association.

Nicolò Biasetton, Riccardo Ricciardi, Luigi Salmaso, and Paola Zuccolotto. 2026. How well do ratings reflect sentiment? Evidence from a large italian review corpus. Applied Stochastic Models in Business and Industry, 42(2):e70090.

Yuri Bizzoni and Pascale Feldkamp. 2024. Sentiment analysis for literary texts: Hemingway as a casestudy. Journal of Data Mining & Digital Humanities, NLP4DH:1–20.

Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka. 2020. Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140:325–331.

Daniel Coutinho, Breno Braga, Theo Canuto, Juliana Alves Pereira, Wesley K. G. Assunção, Igor Steinmacher, Marco Gerosa, and Alessandro Garcia. 2026. Leveraging large language models for sentiment analysis in GitHub pull request discussions. Empirical Software Engineering, 31(5):140.

Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. 2019. Class-Balanced Loss Based on Effective Number of Samples . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9260–9269, Los Alamitos, CA, USA. IEEE Computer Society.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Pascale Feldkamp, Jan Kostkan, Ea Overgaard, Mia Jacobsen, and Yuri Bizzoni. 2024. Comparing tools for sentiment analysis of Danish literature from hymns to fairy tales: Low-resource language and domain challenges. In Proceedings ofthe 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 186–199. Association for Computational Linguistics.

Gemma Team. 2026. Gemma 4 technical report. Preprint, arxiv:2607.02770.

Nils Constantin Hellwig, Jakob Fehle, Udo Kruschwitz, and Christian Wolff. 2026. LLM-as-an-annotator: Training lightweight models with LLM-annotated examples for aspect sentiment tuple prediction. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference, pages 7955–7972, Palma de Mallorca, Spain. ELRA Language Resource Association.

Wen-Fong (Xavier) Huang and Edwin Simpson. 2026. Efficient financial language understanding via distillation with synthetic data. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference, pages 10242–10254, Palma de Mallorca, Spain. ELRA Language Resource Association.

Robbert Jongeling, Proshanta Sarkar, Subhajit Datta, and Alexander Serebrenik. 2017. On negative results when using sentiment analysis tools for software engineering research. Empirical Software Engineering, 22:2543–2584.

Jan Ole Krugmann and Jochen Hartmann. 2024. Sentiment analysis in the age of generative ai. Customer Needs and Solutions, 11(3):1–19.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020. Focal loss for dense object detection. In IEEE Transactions on Pattern Analysis and Machine Intelligence, volume 42, pages 318–327.

Peter McCullagh. 1980. Regression models for ordinal data. Journal ofthe Royal Statistical Society: Series B (Methodological), 42(2):109–142.

Md Saef Ullah Miah, Md Mohsin Kabir, Talha Bin Sarwar, Mejdl Safran, Sultan Alfarhood, and M. F. Mridha. 2024. A multimodal approach to crosslingual sentiment analysis with ensemble of transformer and LLM. Scientific Reports, 14(1):9603.

Nicholas Pangakis and Sam Wolken. 2024. Knowledge distillation in automated annotation: Supervised text classification with LLM-generated training labels. In Proceedings ofthe Sixth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS 2024), pages 113–131, Mexico City, Mexico. Association for Computational Linguistics.

Federico Pianzola. 2026. sentiment-fiction.

Project Gutenberg. Project gutenberg.

Alexander Shvets. 2025. Emo pillars: Knowledge distillation to support fine-grained context-aware and context-less emotion classification. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 174–191, Vienna, Austria. Association for Computational Linguistics.

Artem Suslov and Takehiro Hashimoto. 2026. Traceable sentiment analysis with large language models out of the box. In Proceedings of JADH2026, volume 2026. JADH2026 Program Committee and Local Organizing Committee.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2020. Well-read students learn better: On the importance of pre-training compact models. In International Conference on Learning Representations.

Zengzhi Wang, Qiming Xie, Yi Feng, Zixiang Ding, Zinong Yang, and Rui Xia. 2024. Is chatgpt a good sentiment analyzer? a preliminary study. Preprint, arXiv:2304.04339.

Zhongyan Wang, Yuchen Cao, Shuo Xu, Zhanyi Ding, Yijun Gao, Qinkai Mao, Tongsong Qi, and Hailiang Wang. 2026. Ordinal sentiment classification in cancer support forums: a controlled benchmark of machine learning and transformer models. Frontiers in Digital Health, 8:1836536.

Guangyu Xie, Yice Zhang, Jianzhu Bao, Qianlong Wang, Yang Sun, Bingbing Wang, and Ruifeng Xu. 2025. Comprehensive and efficient distillation for lightweight sentiment analysis models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 22070–22091, Suzhou, China. Association for Computational Linguistics.

Yu-Zhe Yang, Kai-Wen Zha, Ying-Cong Chen, Hongya Wang, and Dina Katabi. 2021. Delving into deep imbalanced regression. In International Conference on Machine Learning.

Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Pan, and Lidong Bing. 2024. Sentiment analysis in the era of large language models: A reality check. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3881–3906, Mexico City, Mexico. Association for Computational Linguistics.

Yice Zhang, Guangyu Xie, Jingjie Lin, Jianzhu Bao, Qianlong Wang, Xi Zeng, and Ruifeng Xu. 2025. Targeted distillation for sentiment analysis. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 22158–22181, Suzhou, China. Association for Computational Linguistics.

Emily Öhman, Yuri Bizzoni, Pascale Feldkamp Moreira, and Kristoffer L Nielbo. 2024. EmotionArcs: Emotion arcs for 9,000 literary texts. In Proceedings of LaTeCH-CLfL 2024, pages 51–66. Association for Computational Linguistics.

## A Teacher Model Data Preparation

## A.1 Annotation Prompt

The sentiment annotations were generated using Gemma 4 31B through Ollama. Each sentence was processed independently using the following prompt, where INPUT\_SENTENCE was replaced with the corresponding sentence from the corpus:

You will perform the sentiment analysis of the sentence. Sentiment is the emotional tone, attitude, or opinion expressed in text.

Evaluate the sentence you will get as an input in the scale from 1 to 10 where 1 = most negative and 10 = most positive sentiment. Also, provide a brief explanation of your decision to score the sentiment in the sentence.

[   
{   
"Sentiment\_score": your score from 1 to 10,   
"Explanation": your explanation for the score   
}   
]   
Here is the sentence: INPUT\_SENTENCE

## A.2 Generation Parameters and Post-processing

Inference was performed locally using Ollama with the model’s default generation parameters in instant mode. No additional modifications to the default inference settings were applied.

To ensure consistent annotation output, the processing pipeline implemented a JSON parsing and validation procedure. Each sentence was allowed up to three generation attempts if the model failed to produce a valid, parsable JSON response.

However, Gemma 4 31B successfully generated valid JSON responses containing sentiment scores on the first attempt for all 10,889 sentences. Consequently, no additional generation attempts were required.

The resulting sentiment scores were extracted from the Sentiment\_score field and used as ordinal labels for subsequent BERT fine-tuning.

## B List of BERT Base Models for Fine-Tuning

Table 2 summarizes the six pretrained BERT configurations considered in this study. BERT-Large and BERT-Base follow the original architectures introduced by Devlin et al. (2019), while the Medium, Small, Mini, and Tiny variants are compact models introduced by Turc et al. (2020).

All models are English uncased BERT encoders. The configurations differ in the number of Transformer layers, hidden dimensions, attention heads, and trainable parameters. Parameter counts are approximate and refer to the pretrained architectures.

## C Fine-Tuning Training Parameters

All BERT variants were fine-tuned using the Hugging Face Trainer framework with the same optimization hyperparameters. Training was configured for up to three epochs, with a fixed random seed of 1111. BERT-large and BERT-base used a maximum sequence length of 256 tokens, while the remaining models used 128 tokens.

Model evaluation, checkpoint saving, and logging were scheduled every 100 training steps. The best checkpoint for each model was selected according to the lowest validation mean absolute error (MAE). At most six checkpoints were retained, and FP16 mixed precision was enabled when CUDA was available. The training configuration is summarized in Table 3.

<table><tr><td>Model</td><td>Layers Hidden</td><td>Heads</td><td>Params.</td></tr><tr><td>BERT-Large 24 google-bert/bert-large-uncased</td><td>1024</td><td>16</td><td>340M</td></tr><tr><td>BERT-Base</td><td>12 768</td><td>12</td><td>110M</td></tr><tr><td>google-bert/bert-base-uncased BERT-Medium</td><td>8 512</td><td>8</td><td>41.7M</td></tr><tr><td>google/bert_uncased_L-8_H-512_A-8</td><td></td><td></td><td></td></tr><tr><td>BERT-Small google/bert_uncased_L-4_H-512_A-8</td><td>4 512</td><td>8</td><td>29.1M</td></tr><tr><td>BERT-Mini</td><td>4 256</td><td>4</td><td>11.3M</td></tr><tr><td>google/bert_uncased_L-4_H-256_A-4 BERT-Tiny</td><td>2 128</td><td>2</td><td></td></tr></table>

Table 2: Architectural characteristics of the pretrained BERT models. Each Hugging Face model ID appears below its model. Parameter counts are approximate.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>2 × 10 3</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Warmup ratio</td><td>0.10</td></tr><tr><td>Maximum epochs</td><td>3</td></tr><tr><td>Training batch size</td><td>16</td></tr><tr><td>Evaluation batch size</td><td>32</td></tr><tr><td>Gradient accumulation</td><td>1</td></tr><tr><td>Evaluation interval</td><td>100 steps</td></tr><tr><td>Checkpoint interval</td><td>100 steps</td></tr><tr><td>Logging interval</td><td>100 steps</td></tr><tr><td>Checkpoint selection</td><td>Min. val. MAE</td></tr><tr><td>Maximum checkpoints</td><td>6</td></tr><tr><td>Random seed</td><td>1111</td></tr><tr><td>Mixed precision</td><td>FP16 (CUDA only)</td></tr><tr><td>Max. length (Large, Base)</td><td>256</td></tr><tr><td>Max. length (other models)</td><td>128</td></tr></table>

Table 3: Training hyperparameters shared across the ordinal BERT configurations.

## C.1 LLM-Annotated Label Distribution

The distribution of LLM-generated sentiment labels across the complete corpus (n = 10, 889) is presented in Table 4. The annotations were unevenly distributed, with score 5 accounting for 41.5% of all sentences. Overall, 57.9% of annotations fell within the near-neutral range (scores 4–6).

## C.2 Evaluation Metrics and Checkpoint Selection

During training, model performance was evaluated on the LLM-annotated validation set using MAE, root mean squared error (RMSE), accuracy, accuracy within one sentiment level, quadratic weighted Cohen’s kappa (QWK), and Spearman’s rank correlation (ρ).

<table><tr><td>Score</td><td>Count</td><td>Score</td><td>Count</td></tr><tr><td>1</td><td>239</td><td>6</td><td>960</td></tr><tr><td>2</td><td>800</td><td>7</td><td>637</td></tr><tr><td>3</td><td>1436</td><td>8</td><td>923</td></tr><tr><td>4</td><td>821</td><td>9</td><td>474</td></tr><tr><td>5</td><td>4519</td><td>10</td><td>80</td></tr></table>

Table 4: Distribution of LLM-generated sentiment labels across the complete annotated corpus (n = 10, 889), before the training, validation, and test split.
<table><tr><td>Model</td><td>Step Val. MAE</td><td>Test MAE</td><td>RMSE</td></tr><tr><td>BERT-large BERT-base BERT-medium</td><td>1600 0.847 600 0.971 1100 1.012</td><td>0.838 0.959 1.078</td><td>1.196 1.312 1.505</td></tr><tr><td>BERT-small BERT-mini</td><td>1100 1.079 1600 1.219</td><td>1.142 1.278</td><td>1.546 1.633</td></tr><tr><td>BERT-tiny Model</td><td>1600 Acc.</td><td>1.354 1.353 Acc. ±1 QWK</td><td>1.895</td></tr><tr><td>BERT-large</td><td>0.461</td><td>0.814 0.746 0.763</td><td>ρ</td></tr><tr><td>BERT-base</td><td>0.401</td><td>0.771 0.6800.711</td><td></td></tr><tr><td>BERT-medium</td><td>0.372</td><td>0.742 0.593</td><td>0.567</td></tr><tr><td>BERT-small</td><td>0.360</td><td></td><td></td></tr><tr><td></td><td></td><td>0.699 0.529</td><td></td></tr><tr><td>BERT-mini</td><td>0.264</td><td>0.661 0.4450.479</td><td>0.532</td></tr></table>

Table 5: Validation and test performance on LLMannotated sentences. Step identifies the selected checkpoint; ρ denotes Spearman’s rank correlation.

The checkpoint selection criterion was validation MAE, with lower values preferred. Following training, the models were also evaluated on the held-out test split of 1,089 LLM-annotated sentences. Table 5 reports the selected checkpoint steps, validation MAE, and test-set performance.

The models were subsequently evaluated against the independent human annotated Hemingway corpus, as reported in Section 6. These human annotations were not used for checkpoint selection.