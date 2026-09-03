# Improving Health Literacy through Lay Summarization of Radiological Reports: An Evaluation of BioNER and Retrieval-Augmented Generation

Egecan C¸ elik Evgin<sup>1</sup>, <sup>˙</sup>Ilknur Karadeniz<sup>3</sup>, Olcay Taner Yıldız<sup>1,2</sup>

<sup>1</sup>Department of Artificial Intelligence and Data Engineering, Ozye <sup>¨</sup> gin University, T ˘ urkiye¨

<sup>2</sup>Department of Computer Science, Ozye <sup>¨</sup> gin University, T ˘ urkiye ¨

<sup>3</sup>Department of Computer Engineering, Galatasaray University, Turkiye¨

egecan.evgin@ozu.edu.tr,

ikaradeniz@gsu.edu.tr,

olcay.yildiz@ozyegin.edu.tr

## Abstract

Radiology reports are written primarily for clinicians, and their specialized terminology often makes them difficult for patients to interpret. As a result, many patients turn to publicly available Large Language Models (LLMs) to help explain their reports, despite well-documented risks of factual inaccuracies and hallucinations. Automated lay-summary generation has emerged as a promising al ternative, yet the effectiveness of retrievalenhanced and clinically informed approaches for radiology-specific communication remains underexplored. This study investigates the extent to which Retrieval-Augmented Generation (RAG) and Named Entity Recognition (NER) improve the quality, factual consistency, and readability of automatically generated lay summaries compared with standard LLM-based generation. We develop a framework combining NER-based extraction of clinically relevant findings with a RAG mechanism for contextual grounding, evaluated across few-shot and fine-tuned variants of two models (Qwen, BioBART). Results show that NER consistently improves readability and overall quality, while RAG alone offers no benefit and can introduce hallucinations from irrelevant retrieved terms. Combining RAG with NER degrades performance in few-shot settings but improves readability when fine-tuned. Fine-tuned BioBART with NER achieves the best overall performance, highlighting entity-aware extraction as the primary driver of improved patient-friendly summaries.

## 1 Introduction

Radiological reports document the findings of medical imaging examinations, such as X-rays, computed tomography (CT), and magnetic resonance imaging (MRI), and serve as a primary means of communication between healthcare professionals. However, these reports are typically written using specialized biomedical terminology and complex clinical language, making them difficult for patients to understand. Consequently, many patients struggle to interpret their imaging results and fully comprehend the implications of the reported findings. To better understand their medical conditions and make informed decisions about treatment, patients often seek additional information. Traditionally, this involved searching online for medical terms and symptoms. Today, many patients use Large Language Model (LLM)-based chatbots, such as ChatGPT, Gemini, and DeepSeek, to obtain health-related information (OpenAI, 2022; Google, 2024; DeepSeek-AI, 2024). However, these systems can generate inaccurate or hallucinated content, potentially leading patients to misunderstand their radiological findings or place undue trust in incorrect information.

This communication gap can limit patient understanding and health literacy, motivating research into methods that translate radiological reports into patient-friendly language. Recent work has explored Retrieval-Augmented Generation (RAG) to improve the quality and factual consistency of lay summarization by grounding generated outputs in external knowledge (Guo et al., 2024). In parallel, Named Entity Recognition (NER) has been used to identify clinically relevant terms that can guide and constrain generation toward more accurate and relevant content. However, the comparative effectiveness of retrieval-based and entity-aware approaches for radiology-specific lay summarization remains underexplored, particularly across models of different scale and training regime.

To address this gap, we evaluate our approach across four public radiology report datasets spanning diverse clinical settings and imaging modalities: PadChest, BIMCV-COVID19+, Open-i, and MIMIC-CXR (Bustos et al., 2020; de la Iglesia Vaya et al.´ , 2020; Demner-Fushman et al.,

2012; Johnson et al., 2019).

The main contributions of this study are:

• A framework combining and comparing RAG-based and NER-enhanced approaches to radiology report lay summarization;

• A comparison between a state-of-the-art general-purpose LLM and a biomedical small language model;

• The use of few-shot baselines to systematically compare against fine-tuned model variants.

The remainder of the paper is organized as follows: Section 2 reviews related work, Section 3 describes the methodology, Section 4 presents the results and discussion, and Section 5 concludes the paper.

## 2 Related Work

Lay summaries differ from standard summaries in their emphasis on readability for non-expert audiences. In the biomedical domain, the BioLay-Summ shared task has been organized in 2023, 2024, and 2025 (Goldsack et al., 2023, 2024; Xiao et al., 2025), aiming to generate lay summaries that are relevant, readable, and factual. BioLay-Summ 2025 Shared Task 2 focused specifically on generating lay summaries from radiology reports, and several of the approaches discussed below were developed for this task.

Fine-tuning-based approaches: AEHRC achieved the best overall performance in both the open and closed subtasks of Shared Task 2 using fully supervised fine-tuning, comparing T5- Large with LLaMA-3.2-3B and finding T5-Large superior, without using LoRA, quantization, or RAG (Zhang et al., 2025; Raffel et al., 2020; Meta AI, 2024). KHU LDI, the second-best open-track system, used QLoRA fine-tuning on Qwen2.5-3B-Instruct and Qwen3-4B, combined with 3-shot prompting and a generate-feedbackrefine pipeline (Moriazi and Sung, 2025; Dettmers et al., 2023; Yang et al., 2024, 2025; Madaan et al., 2023). MetninOzU ranked third overall using an abstract-based summarization setup, showing that shorter inputs can still yield strong factuality scores (Evgin et al., 2025).

Prompting-based approaches: 5cNLP, the second-place closed-track system, relied on structured prompting rather than fine-tuning, testing Llama-3.3-70B-Instruct and GPT-4.1; their best result used GPT-4.1 with few-shot radiology examples selected via BERT-large embeddings (Lossio-Ventura et al., 2025). Proff et al. compared GPT-4o, Llama-3-70B, and Mixtral-8x22B for radiology report simplification, finding that all models improved readability, though open-weight models produced more high-risk errors than GPT-4o (Proff et al., 2026).

RAG-based approaches: CUTN Bio placed third in the closed track of Shared Task 2 using a RAG pipeline with Zephyr-7B-beta, extracting medical terms via SciSpacy and retrieving Wikipedia definitions stored in ChromaDB (Sivagnanam et al., 2025; Tunstall et al., 2023; Neumann et al., 2019; Chroma, 2026). The same team placed second in Subtask 1.2 (externalknowledge lay summarization) using a similar RAG approach with MedCAT for term extraction and LLaMA-3-8B-Instruct for generation (Kraljevic et al., 2021; Meta AI, 2024). Sun et al. proposed FactMM-RAG, which retrieves factually similar report content via RadGraph prior to generation to improve the accuracy of generated radiology reports (Sun et al., 2025). Lay-SummX applied retrieval-augmented fine-tuning, using abstracts to retrieve relevant full-text chunks before fine-tuning LLaMA 3.1 with LoRA (Lin and Yu, 2025). Guo et al. introduced Retrieval-Augmented Lay Language generation, retrieving UMLS and Wikipedia definitions to supply missing background explanations (Guo et al., 2024). UIUC BioNLP used an extract-then-summarize pipeline combining Wikipedia definition retrieval with DPR-based passage retrieval (You et al., 2024).

NER-based approaches: ISIKSumm used a BART-based system augmented with biomedical entity labels using Stanza NER to improve handling of technical terms (Colak and Karadeniz, 2023). Gupta and Krishnamurthy’s LayForge system used BioBERT NER to identify biomedical terms and incorporated UMLS definitions before summary rewriting, improving readability and factuality at a small cost to ROUGE scores (Gupta and Krishnamurthy, 2025; Lee et al., 2020; Bodenreider, 2004; Lin, 2004). Ming et al. used MeSH terms to guide LLMs toward more informative background context for lay readers (Ming et al., 2025).

Overall, prior work has largely explored RAGbased and NER-based strategies in isolation, with few studies directly comparing their effectiveness within a unified framework, or across models differing substantially in scale and training regime (few-shot vs. fine-tuned). This study addresses this gap by systematically comparing RAG-based and NER-enhanced lay summarization strategies for radiology reports.

## 3 Methodology

## 3.1 Models

Two models were used in this study: Qwen3.5- 0.8B (Qwen Team, 2026), a recent generalpurpose small language model, and BioBARTv2-large (Yuan et al., 2022), a model pretrained specifically on biomedical text. This pairing enables comparison between a strong generalpurpose model and a smaller model adapted for the biomedical domain.

## 3.2 Dataset and Evaluation Set

Four public radiology report datasets, spanning diverse clinical settings and imaging modalities, were used in this study: PadChest, BIMCV-COVID19+, Open-i, and MIMIC-CXR. PadChest contains over 160,000 chest X-ray images from approximately 67,000 patients, while BIMCV-COVID19+ includes COVID-19 X-ray and CT studies, comprising 21,342 CR, 34,829 DX, and 7,918 CT cases (Bustos et al., 2020; de la Iglesia Vaya et al.´ , 2020). Open-i is smaller, with 7,470 chest X-ray images and 3,955 reports, while MIMIC-CXR is the largest dataset, with 227,835 studies and 377,110 images derived from real clinical reports (Demner-Fushman et al., 2012; Johnson et al., 2019). Lay summaries were automatically generated from the clinical reports using the Layman’s RRG framework (Zhao et al., 2026), and the combined datasets were used in the BioLaySumm 2025 shared task (Xiao et al., 2025). However, the lay summaries for the shared task’s test set were not publicly available. To address this, the original training set was split to construct a new test set, with the goal of obtaining a larger evaluation set than the one used in the shared task. The split was performed randomly (seed = 42) to avoid bias; this is distinct from the sampling of the three few-shot exemplar reports described in §3.4, which used the same seed value for a separate sampling step. The resulting split comprises 168,036 training reports (89.38%), 14,971 validation reports (7.96%), 5,000 test reports (2.66%), and 3 few-shot exemplar reports, with average token counts summarized in Table 1.

Table 1: Dataset split statistics
<table><tr><td>Split Name</td><td>Split %</td><td>Rows</td><td>Rad. Reports</td><td>Lay</td></tr><tr><td>Training</td><td>89.38%</td><td>168,036</td><td>31.11</td><td>42.17</td></tr><tr><td>Validation</td><td>7.96%</td><td>14,971</td><td>34.27</td><td>45.49</td></tr><tr><td>Testing</td><td>2.66%</td><td>5,000</td><td>31.19</td><td>42.22</td></tr><tr><td>Few-Shot Samples</td><td>0.001%</td><td>3</td><td>11.00</td><td>20.33</td></tr></table>

Rad. Reports: Average tokens in radiological reports;  
Lay: Average tokens in layman summaries.

## 3.3 Evaluation Metrics

Generated summaries were evaluated along three dimensions: relevance, readability, and factuality. All metrics were scaled using min-max normalization to place their values on a comparable range (Han et al., 2011). No additional weighting was applied across the three metric groups, since each group contained an equal number of metrics. For relevance and factuality metrics, higher scores indicate better performance; for all readability metrics (FKGL, DCRS, SLE), lower scores indicate better performance (i.e., simpler, more accessible text).

Relevance: ROUGE (Lin, 2004) measures word overlap between predicted and gold lay summaries; we report the average F1 across ROUGE-1, ROUGE-2, and ROUGE-L. METEOR (Banerjee and Lavie, 2005) extends beyond exact word matches by accounting for stems and synonyms, offering a complementary view of relevance. BERTScore (Zhang et al., 2020) measures semantic similarity by comparing contextual word embeddings between predicted and gold summaries, computing precision, recall, and F1 based on closest token matches.

Readability: FKGL (Kincaid et al., 1975) estimates the grade level of a summary based on sentence and word length, with longer sentences and words yielding higher (less readable) scores. DCRS (Dale and Chall, 1948) complements FKGL by assessing word familiarity against a list of common words, capturing cases FKGL may miss, such as short but unfamiliar words (e.g., ”understand”). SLE (Cripwell et al., 2023) is a transformer-based metric that requires only the predicted summary, using a RoBERTa-base model with a regression head to produce a simplicity score.

Factuality: SummaC (Laban et al., 2022) evaluates sentence-level agreement between the source report and the predicted summary using entailment and contradiction scores, penalizing contradictory content. FENICE (Scire et al.\` , 2024) evaluates factuality at the claim level by extracting atomic claims from the predicted summary and verifying them against the source text, directly penalizing unsupported or contradicted claims. CheXbert-F1 (Smit et al., 2020), developed specifically for radiology reports, evaluates whether clinical findings in the generated summary match those in the reference report, penalizing missing or incorrect findings.

## 3.4 Baseline Strategies

Two baseline strategies were established for each model: (i) Few-shot prompting. Baselines were computed under 0-shot, 1-shot, and 3-shot settings. To construct the 1-shot and 3-shot exemplars, three radiology reports were randomly sampled (seed = 42) and excluded from the test set to prevent data leakage (Table 1). (ii) LoRA finetuning.

Both models were fine-tuned using LoRA with r=4, lora alpha=8, dropout of 0.05, and no bias term. Qwen was adapted on q proj and v proj; BioBART was adapted on q proj, v proj, and out proj. Training used 2 epochs, a batch size of 20 (evaluation batch size 16), gradient accumulation of 1, learning rate 2e-4, weight decay 0.01, 100 warmup steps, with bf16 enabled and fp16 disabled.

## 3.5 Enhancement Strategies

In addition to the baselines, three enhancement strategies were evaluated, each applied to both the few-shot and fine-tuned settings of both models:

• NER-enhanced (BioNER): Clinically relevant terms were extracted using Stanza’s radiology NER model (Zhang et al., 2021), which identifies five entity classes: ANATOMY, OBSERVATION, ANATOMY MODIFIER, OBSERVA-TION MODIFIER, and UNCERTAINTY. Extracted entities were used to guide summary generation toward clinically relevant content.

• RAG-enhanced: A retrieval-augmented pipeline was built in which an agent extracted candidate medical terms from the source report. Each term was first checked against a local term-description database; if not found, it was searched via the Wikipedia API, and the first sentence of the result was stored in the local database for reuse. Retrieved definitions were then provided as contextual grounding during summary generation.

• Combined (BioNER + RAG): Term extraction was performed using BioNER, and the resulting terms were used to query the RAG retrieval pipeline described above, combining entity-guided extraction with retrieval-based grounding.

This produces three conditions per model per learning setting (baseline, +BioNER, +RAG, +BioNER+RAG).

## 4 Results

Table 2 presents the overall results of the fewshot strategies. For the Qwen model, the BioNER strategy improved overall relevance, readability, and factuality compared with the 0-shot baseline. The RAG strategy did not outperform the baseline, while the combination of BioNER and RAG resulted in lower scores across all evaluation dimensions. A similar trend was observed for BioBART. Compared with Qwen, BioBART achieved lower performance in the few-shot setting across most evaluation metrics.

Table 3 summarizes the fine-tuning results. In contrast to the few-shot experiments, BioBART outperformed Qwen after fine-tuning. Similar to the few-shot setting, the BioNER strategy improved overall performance, particularly readability, for both models. The RAG strategy improved readability but reduced relevance in both models. For Qwen, the combined BioNER+RAG strategy achieved the best FKGL and DCRS scores together with the lowest SLE score.

Table 4 compares the best-performing few-shot and fine-tuning strategies for each model. For Qwen, the best few-shot strategy (0-shot BioNER) outperformed all fine-tuning configurations. In contrast, BioBART achieved its highest performance after fine-tuning. Comparing the bestperforming configurations of both models, finetuned BioBART with BioNER achieved better overall performance than Qwen with the 0-shot BioNER strategy across most evaluation metrics.

Table 2: Mean scores for Few-Shot Based Strategies
<table><tr><td rowspan="2">Str</td><td colspan="3">Relevance</td><td colspan="3">Readability</td><td colspan="3">Factuality</td><td rowspan="2">Mean↑</td></tr><tr><td>ROUGE↑</td><td>METEOR↑</td><td>BERTScore↑</td><td>FKGL↓</td><td>DCRS↓</td><td>SLE↓</td><td>SummaC↑</td><td>FENICE↑</td><td>CHEX↑</td></tr><tr><td>Qwen3.5 0-Shot</td><td>0.357</td><td>0.3952</td><td>0.9146</td><td>6.59</td><td>9.8097</td><td>1.4909</td><td>0.5784</td><td>0.3209</td><td>0.8918</td><td>0.8221</td></tr><tr><td>Qwen3.5 1-Shot</td><td>0.3272</td><td>0.3581</td><td>0.9077</td><td>6.71</td><td>9.7976</td><td>1.3763</td><td>0.6567</td><td>0.2032</td><td>0.8977</td><td>0.7865</td></tr><tr><td>Qwen3.5 3-Shot</td><td>0.3688</td><td>0.4029</td><td>0.914</td><td>6.26</td><td>9.3057</td><td>1.5591</td><td>0.7451</td><td>0.4018</td><td>0.9056</td><td>0.9066</td></tr><tr><td>Qwen3.5 BioNER 0-Shot</td><td>0.3589</td><td>0.399</td><td>0.9183</td><td>6.35</td><td>9.3841</td><td>1.0687</td><td>0.6028</td><td>0.4127</td><td>0.8941</td><td>0.9092</td></tr><tr><td>Qwen3.5 RAG 0-Shot</td><td>0.3565</td><td>0.3898</td><td>0.9089</td><td>7.1</td><td>9.6396</td><td>1.3106</td><td>0.5855</td><td>0.1742</td><td>0.889</td><td>0.7917</td></tr><tr><td>Qwen3.5 BioNER + RAG 0-Shot</td><td>0.2474</td><td>0.3019</td><td>0.865</td><td>9.61</td><td>10.7426</td><td>1.8543</td><td>0.542</td><td>0.0507</td><td>0.8141</td><td>0.5172</td></tr><tr><td>BioBART 0-Shot</td><td>0.194</td><td>0.2807</td><td>0.8448</td><td>15.39</td><td>11.9016</td><td>1.045</td><td>0.3</td><td>0</td><td>0.7708</td><td>0.3698</td></tr><tr><td>BioBART 1-Shot</td><td>0.0845</td><td>0.1298</td><td>0.8136</td><td>15.2</td><td>12.8661</td><td>0.5296</td><td>0.4173</td><td>0.3341</td><td>0.1074</td><td>0.2879</td></tr><tr><td>BioBART 3-Shot</td><td>0.0872</td><td>0.1418</td><td>0.8137</td><td>14.46</td><td>13.0713</td><td>0.5408</td><td>0.5829</td><td>0.3498</td><td>0.0368</td><td>0.3303</td></tr><tr><td>BioBART BioNER 0-Shot</td><td>0.0899</td><td>0.1241</td><td>0.802</td><td>18.39</td><td>12.1376</td><td>0.6565</td><td>0.416</td><td>0.3841</td><td>0.7531</td><td>0.3541</td></tr><tr><td>BioBART RAG 0-Shot</td><td>0.1069</td><td>0.1654</td><td>0.8186</td><td>14.31</td><td>11.8315</td><td>1.314</td><td>0.6266</td><td>0.0322</td><td>0.7686</td><td>0.344</td></tr><tr><td>BioBART BioNER + RAG 0-Shot</td><td>0.0918</td><td>0.1351</td><td>0.8052</td><td>14.66</td><td>11.2866</td><td>0.9849</td><td>0.5033</td><td>0.1442</td><td>0.778</td><td>0.3543</td></tr></table>

Table 3: Mean scores for Fine-Tuning Based Strategies
<table><tr><td rowspan="2">Str</td><td colspan="3">Relevance</td><td colspan="3">Readability</td><td colspan="3">Factuality</td><td rowspan="2">Mean↑</td></tr><tr><td>ROUGE↑</td><td>METEOR↑</td><td>BERTScore↑</td><td>FKGL↓</td><td>DCRS↓</td><td>SLE↓</td><td>SummaC↑</td><td>FENICE↑</td><td>CHEX↑</td></tr><tr><td>Qwen3.5 Fine-Tuning</td><td>0.4071</td><td>0.4583</td><td>0.9249</td><td>10.58</td><td>11.3545</td><td>1.1557</td><td>0.6754</td><td>0.4162</td><td>0.8918</td><td>0.5425</td></tr><tr><td>Qwen3.5 Fine-Tuning + BioNER</td><td>0.4548</td><td>0.5233</td><td>0.9244</td><td>7.67</td><td>10.1513</td><td>1.0948</td><td>0.3083</td><td>0.5531</td><td>0.8904</td><td>0.6499</td></tr><tr><td>Qwen3.5 Fine-Tuning + RAG</td><td>0.3379</td><td>0.4253</td><td>0.9055</td><td>9.75</td><td>10.7824</td><td>1.0682</td><td>0.4933</td><td>0.3684</td><td>0.879</td><td>0.4476</td></tr><tr><td>Qwen3.5 Fine-Tuning BioNER + RAG</td><td>0.2826</td><td>0.3332</td><td>0.8733</td><td>12.71</td><td>11.3036</td><td>1.4295</td><td>0.5741</td><td>0.2832</td><td>0.8088</td><td>0.1501</td></tr><tr><td>BioBART Fine-Tuning</td><td>0.5438</td><td>0.5953</td><td>0.9422</td><td>7.29</td><td>10.2835</td><td>1.0497</td><td>0.6129</td><td>0.6629</td><td>0.9344</td><td>0.9126</td></tr><tr><td>BioBART Fine-Tuning + BioNER</td><td>0.5335</td><td>0.5845</td><td>0.9399</td><td>6.09</td><td>10.038</td><td>1.0017</td><td>0.6005</td><td>0.6301</td><td>0.9255</td><td>0.9194</td></tr><tr><td>BioBART Fine-Tuning + RAG</td><td>0.452</td><td>0.4856</td><td>0.9278</td><td>6.3</td><td>9.9726</td><td>1.4031</td><td>0.4452</td><td>0.4062</td><td>0.9219</td><td>0.6662</td></tr><tr><td>BioBART Fine-Tuning + BioNER + RAG</td><td>0.3634</td><td>0.3792</td><td>0.9133</td><td>5.93</td><td>9.6512</td><td>2.0637</td><td>0.6138</td><td>0.293</td><td>0.906</td><td>0.522</td></tr></table>

Table 4: Mean scores for the best FT and few-shot strategies for each model
<table><tr><td rowspan="2">Str</td><td colspan="3">Relevance</td><td colspan="3">Readability</td><td colspan="3">Factuality</td><td></td></tr><tr><td>ROUGE↑</td><td>METEOR↑</td><td>BERTScore↑</td><td>FKGL↓</td><td>DCRS↓</td><td>SLE↓</td><td>SummaC↑</td><td>FENICE↑</td><td>CHEX↑</td><td>Mean↑</td></tr><tr><td>BioBART Fine-Tuning + BioNER</td><td>0.5335</td><td>0.5845</td><td>0.9399</td><td>6.09</td><td>10.038</td><td>1.0017</td><td>0.6005</td><td>0.6301</td><td>0.9255</td><td>0.9703</td></tr><tr><td>Qwen3.5 BioNER 0-Shot</td><td>0.3589</td><td>0.399</td><td>0.9183</td><td>6.35</td><td>9.3841</td><td>1.0687</td><td>0.6028</td><td>0.4127</td><td>0.8941</td><td>0.7059</td></tr><tr><td>Qwen3.5 Fine-Tuning + BioNER</td><td>0.4548</td><td>0.5233</td><td>0.9244</td><td>7.67</td><td>10.1513</td><td>1.0948</td><td>0.3083</td><td>0.5531</td><td>0.8904</td><td>0.623</td></tr><tr><td>BioBART 0-Shot</td><td>0.194</td><td>0.2807</td><td>0.8448</td><td>15.39</td><td>11.9016</td><td>1.045</td><td>0.3</td><td>0</td><td>0.7708</td><td>0.0595</td></tr></table>

## 4.1 Discussion

The experimental results partially support the initial hypotheses. The BioNER strategy consistently improved readability and generally enhanced overall performance in both few-shot and fine-tuning settings. This suggests that explicitly providing biomedical entity information helps the models better identify important concepts while generating lay summaries.

In contrast, the RAG strategy did not consistently improve performance. Manual inspection showed that the retrieval system occasionally returned Wikipedia entries corresponding to terms with identical surface forms but different meanings, introducing irrelevant background information into the generation process. Although the prompt instructed the model to ignore unrelated retrieved content, the FENICE scores indicate that hallucinated information was still introduced in some summaries.

The combination of BioNER and RAG did not produce the expected improvements. Analysis revealed that several multi-word biomedical entities extracted by the BioNER system could not be matched by the Wikipedia API, resulting in missing or incomplete retrieved knowledge. Consequently, the potential benefits of retrieval were diminished, leading to lower overall performance.

The comparison between the two language models highlights the importance of domainspecific pretraining. Although Qwen demonstrated stronger few-shot capabilities, BioBART benefited substantially from fine-tuning, ultimately achieving the best overall results. This finding suggests that biomedical pretraining provides a stronger foundation for task-specific adaptation, whereas larger general-purpose language models can remain competitive in low-resource settings without additional training.

## 4.2 Positive Impact

This study shows that radiology reports can be made easier for patients to understand without removing the main clinical information. Lay summaries may help patients understand their results better and ask more useful questions during medical appointments. The findings also suggest that using biomedical entities can help the model focus on the most important parts of a report and explain them in clearer language.

Since the study uses small language models and LoRA fine-tuning, the proposed setup may also be practical for institutions with limited computing resources. At the same time, the RAG results show that adding external information is not always helpful. Wrong term matches or unrelated definitions can introduce information that is not supported by the report. This points to the need for more reliable medical knowledge sources and careful checking before these systems are used in practice.

These summaries should be used as an aid for patients and clinicians, not as a replacement for medical advice. Further testing with both patients and healthcare professionals is still needed before patient-facing use.

## 5 Conclusion

This study investigated the effects of BioNERand RAG-based strategies on radiology report lay summarization under both few-shot inference and fine-tuning settings. The proposed approaches were evaluated using nine metrics covering relevance, readability, and factuality with equal weighting.

The experimental results show that the BioNER strategy consistently improved the baseline models, particularly in terms of readability, while also maintaining competitive relevance and factuality. In contrast, the RAG strategy did not consistently improve performance, and combining BioNER with RAG did not yield the expected gains. Overall, the findings partially support the initial hypotheses: BioNER proved to be an effective enhancement for lay summarization, whereas the effectiveness of RAG was limited by the quality of the retrieved knowledge.

These results demonstrate that providing explicit biomedical entity information is a simple yet effective approach for improving the readability of automatically generated lay summaries. Future work will focus on improving the retrieval component by exploring domain-specific knowledge bases, biomedical knowledge graphs, and more robust entity linking methods to reduce retrieval errors. In addition, investigating alternative BioNER models and retrieval strategies may further improve the quality and factual consistency of gen-

erated lay summaries.

## References

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72. Association for Computational Linguistics.

Olivier Bodenreider. 2004. The unified medical language system (UMLS): integrating biomedical terminology. Nucleic Acids Research, 32(Suppl. 1):D267–D270.

Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vaya. 2020.´ Padchest: A large chest x-ray image dataset with multilabel annotated reports. Medical Image Analysis, 66:101797.

Chroma. 2026. ChromaDB: The open-source search infrastructure for ai. https://www. trychroma.com/products/chromadb. Accessed: 2026-07-08.

Cagla Colak and Lknur Karadeniz. 2023. ISIKSumm at BioLaySumm task 1: BART-based summarization system enhanced with bio-entity labels. In Proceedings ofthe 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks, pages 636–640, Toronto, Canada. Association for Computational Linguistics.

Liam Cripwell, Joel Legrand, and Claire Gardent.¨ 2023. Simplicity level estimate (SLE): A learned reference-less metric for sentence simplification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12053–12059. Association for Computational Linguistics.

Edgar Dale and Jeanne S. Chall. 1948. A formula for predicting readability. Educational Research Bulletin, 27(1):11–20.

Maria de la Iglesia Vaya, Jose Manuel Saborit,´ Joaquim Angel Montell, Antonio Pertusa, Aurelia Bustos, Miguel Cazorla, Joaquin Galant, Xavier Barber, Domingo Orozco-Beltran, Francisco´ Garc´ıa-Garc´ıa, Marisa Caparros, Germ´ an Gonz´ alez,´ and Jose Mar´ıa Salinas. 2020. Bimcv covid-19+: a large annotated dataset of rx and ct images from covid-19 patients. arXiv preprint arXiv:2006.01174.

DeepSeek-AI. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Dina Demner-Fushman, Sameer Antani, Matthew Simpson, and George R. Thoma. 2012. Design and development of a multimodal biomedical information retrieval system. Journal of Computing Science and Engineering, 6(2):168–177.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs.

Egecan Evgin, Ilknur Karadeniz, and Olcay Taner Yıldız. 2025. MetninOzU at BioLaySumm2025: Text summarization with reverse data augmentation and injecting salient sentences. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 179–184, Vienna, Austria. Association for Computational Linguistics.

Tomas Goldsack, Zheheng Luo, Qianqian Xie, Carolina Scarton, Matthew Shardlow, Sophia Ananiadou, and Chenghua Lin. 2023. Overview of the biolaysumm 2023 shared task on lay summarization of biomedical research articles. In Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks, pages 468–477, Toronto, Canada. Association for Computational Linguistics.

Tomas Goldsack, Carolina Scarton, Matthew Shardlow, and Chenghua Lin. 2024. Overview of the BioLaySumm 2024 shared task on the lay summarization of biomedical research articles. In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, pages 122–131, Bangkok, Thailand. Association for Computational Linguistics.

Google. 2024. An overview of the gemini app. https://gemini.google/overview/.

Yue Guo, Wei Qiu, Gondy Leroy, Sheng Wang, and Trevor Cohen. 2024. Retrieval augmentation of large language models for lay language generation. Journal ofBiomedical Informatics, 149:104580.

Aaradhya Gupta and Parameswari Krishnamurthy. 2025. Shared task at biolaysumm2025 : Extract then summarize approach augmented with umls based definition retrieval for lay summary generation. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 185– 189, Vienna, Austria. Association for Computational Linguistics.

Jiawei Han, Micheline Kamber, and Jian Pei. 2011. Data Mining: Concepts and Techniques, 3 edition. Morgan Kaufmann.

Alistair E. W. Johnson, Tom J. Pollard, Seth J. Berkowitz, Nathaniel R. Greenbaum, Matthew P. Lungren, Chih-ying Deng, Roger G. Mark, and Steven Horng. 2019. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data, 6(1):317.

J. Peter Kincaid, Robert P. Fishburne, Richard L. Rogers, and Brad S. Chissom. 1975. Derivation of new readability formulas (automated readability index, fog count and flesch reading ease formula) for navy enlisted personnel. Technical Report Research Branch Report 8-75, Naval Technical Training Command.

Zeljko Kraljevic, Thomas Searle, Anthony Shek, Lukasz Roguski, Kawsar Noor, Daniel Bean, Aurelie Mascio, Leilei Zhu, Amos A. Folarin, Angus Roberts, Rebecca Bendayan, Mark P. Richardson, Robert Stewart, Anoop D. Shah, Wai Keong Wong, Zina Ibrahim, James T. Teo, and Richard J. B. Dobson. 2021. Multi-domain clinical natural language processing with medcat: The medical concept annotation toolkit. Artificial Intelligence in Medicine, 117:102083.

Philippe Laban, Tobias Schnabel, Paul N. Bennett, and Marti A. Hearst. 2022. SummaC: Re-visiting NLIbased models for inconsistency detection in summarization. Transactions of the Association for Computational Linguistics, 10:163–177.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. BioBERT: a pretrained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81. Association for Computational Linguistics.

Fan Lin and Dezhi Yu. 2025. LaySummX at BioLaySumm: Retrieval-augmented fine-tuning for biomedical lay summarization using abstracts and retrieved full-text context. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 202–214, Vienna, Austria. Association for Computational Linguistics.

Juan Antonio Lossio-Ventura, Callum Chan, Arshitha Basavaraj, Hugo Alatrista-Salas, Francisco Pereira, and Diana Inkpen. 2025. 5cNLP at BioLay-Summ2025: Prompts, retrieval, and multimodal fusion. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 215–231, Vienna, Austria. Association for Computational Linguistics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback.

Meta AI. 2024. Llama 3.2 Model Card. https://github.com/meta-llama/ llama-models/blob/main/models/ llama3\_2/MODEL\_CARD.md.

Shufan Ming, Yue Guo, and Halil Kilicoglu. 2025. Towards knowledge-guided biomedical lay summarization using large language models. In Proceedings of the Second Workshop on Patient-Oriented Language Processing, pages 285–297, Albuquerque, New Mexico. Association for Computational Linguistics.

Nur Alya Dania binti Moriazi and Mujeen Sung. 2025. KHU LDI at BioLaySumm2025: Fine-tuning and refinement for lay radiology report generation. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 256– 268, Vienna, Austria. Association for Computational Linguistics.

Mark Neumann, Daniel King, Iz Beltagy, and Waleed Ammar. 2019. ScispaCy: Fast and robust models for biomedical natural language processing. In Proceedings of the 18th BioNLP Workshop and Shared Task, pages 319–327, Florence, Italy. Association for Computational Linguistics.

OpenAI. 2022. Introducing chatgpt. https:// openai.com/index/chatgpt/.

Annemarie Katharina Proff, Babak Salam, Mohammed Hayawi, Dmitrij Kravchenko, Narine Mesropyan, Taraneh Aziz-Safaie, Tatjana Dell, Maike Theis, Claus Christian Pieper, Alois Martin Sprinkart, Daniel Kutting, Julian Alexander Luetkens, Sebas-¨ tian Nowak, and Alexander Isaak. 2026. Simplifying radiology reports with large language models: privacy-compliant open- versus closed-weight models. European Radiology.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

Alessandro Scire, Karim Ghonim, and Roberto Nav-\` igli. 2024. FENICE: Factuality evaluation of summarization based on natural language inference and claim extraction. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14148–14161. Association for Computational Linguistics.

Bhuvaneswari Sivagnanam, Rivo Krishnu C H, Princi Chauhan, and Saranya Rajiakodi. 2025. CUTN Bio at BioLaySumm: Multi-task prompt tuning with external knowledge and readability adaptation for layman summarization. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 269–274, Vienna, Austria. Association for Computational Linguistics.

Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y. Ng, and Matthew P. Lungren. 2020. CheXBert: Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 1500–1519. Association for Computational Linguistics.

Liwen Sun, James Jialun Zhao, Wenjing Han, and Chenyan Xiong. 2025. Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 643–655. Association for Computational Linguistics.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clementine´ Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. Zephyr: Direct distillation of lm alignment.

Chenghao Xiao, Kun Zhao, Xiao Wang, Siwei Wu, Sixing Yan, Tomas Goldsack, Sophia Ananiadou, Noura Al Moubayed, Liang Zhan, William K. Cheung, and Chenghua Lin. 2025. Overview of the Bio-LaySumm 2025 shared task on lay summarization of biomedical research articles and radiology reports. In Proceedings ofthe 24th Workshop on Biomedical Language Processing, pages 365–377, Vienna, Austria. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, et al. 2025. Qwen3 technical report.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, et al. 2024. Qwen2.5 technical report.

Zhiwen You, Shruthan Radhakrishna, Shufan Ming, and Halil Kilicoglu. 2024. UIUC BioNLP at BioLaySumm: An extract-then-summarize approach augmented with Wikipedia knowledge for biomedical lay summarization. In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, pages 132–143, Bangkok, Thailand. Association for Computational Linguistics.

Hongyi Yuan, Zheng Yuan, Ruyi Gan, Jiaxing Zhang, Yutao Xie, and Sheng Yu. 2022. Biobart: Pretraining and evaluation of a biomedical generative language model.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations.

Wenjun Zhang, Shekhar Chandra, Bevan Koopman, Jason Dowling, and Aaron Nicolson. 2025. AEHRC at BioLaySumm 2025: Leveraging t5 for lay summarisation of radiology reports. In Proceedings of the

24th Workshop on Biomedical Language Processing (Shared Tasks), pages 171–178, Vienna, Austria. Association for Computational Linguistics.

Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, and Curtis P. Langlotz. 2021. Biomedical and clinical english model packages for the stanza python nlp library. Journal ofthe American Medical Informatics Association, 28(9):1892–1899.

Kun Zhao, Chenghao Xiao, Sixing Yan, Haoteng Tang, William K. Cheung, Noura Al Moubayed, Liang Zhan, and Chenghua Lin. 2026. X-ray made simple: Lay radiology report generation and robust evaluation. In Findings of the Association for Computational Linguistics: ACL 2026, pages 34583–34598. Association for Computational Linguistics.