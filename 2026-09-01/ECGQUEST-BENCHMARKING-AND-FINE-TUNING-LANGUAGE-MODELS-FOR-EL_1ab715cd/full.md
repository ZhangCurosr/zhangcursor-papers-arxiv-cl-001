# ECGQUEST: BENCHMARKING AND FINE-TUNING LANGUAGE MODELS FOR ELECTROCARDIOGRAPHY

Mohammadsina Hassannia   
Department of Biomedical Informatics   
Emory University   
Atlanta, GA 30322, USA   
sina.hassannia@dbmi.emory.edu   
Reza Sameni   
Department of Biomedical Engineering   
Georgia Institute of Technology   
and Department of Biomedical Informatics   
Emory University   
Atlanta, GA 30332, USA   
rsameni@emory.edu   
Matthew A. Reyna   
Department of Biomedical Informatics   
Emory University   
Atlanta, GA 30322, USA   
matthew@dbmi.emory.edu

## ABSTRACT

Electrocardiogram (ECG) interpretation requires knowledge of cardiology, electrophysiology, clinical diagnosis, ECG waveforms, signal acquisition, and ECG instrumentation. While large language models have proven effective for such applications, existing language-models primarily assess broad medical knowledge or the interpretation of individual ECG signals and images, rather than comprehensive context knowledge required for ECG interpretation. We built ECGQuest, a literature-grounded resource for evaluating and fine-tuning ECG-specific language models. An automated GPT-4o-based pipeline generated questions from 23 ECG references and proceedings of the Computing in Cardiology conference between 2003–2025. The final dataset contains 10,904 unique True/False questions paired with their negated form derived from the same source (21,808 Q&A in total). We evaluated three commercial and 20 open-source language models on the held-out ECGQuest test set in a zero-shot setting. We also fine-tuned five open-source models with 7–14 B parameters using Low-Rank Adaptation and included BERT and BiomedBERT as supervised encoder baselines. Generalization was evaluated using ECG-related subsets of MedMCQA and MedQA datasets converted to binary True/False questions using their official examination answer keys. Zero-shot accuracy on ECGQuest ranged from 49.5% to 74.4%, with GPT-5 achieving the highest accuracy. General-purpose models outperformed all medically specialized models, and several models showed strong directional True/- False biases. Encoder baselines performed near chance. Fine-tuning improved all open-source models on ECGQuest by 6.5–14.1%. The fine-tuned DeepSeek-R1-Distill-Qwen-14 B model achieved 76.3% accuracy, and a five-model voting ensemble achieved the highest overall accuracy of 78.5%. On MedMCQA and MedQA, fine-tuning primarily benefited weak or class-biased models and did not consistently improve already strong base models. ECGQuest is a reproducible benchmark for evaluating contextual ECG knowledge in language models and demonstrates that targeted, parameter-efficient fine-tuning can make small language models competitive with substantially larger commercial models.

## 1 INTRODUCTION

The electrocardiogram (ECG) is among the most widely used diagnostic tools in cardiovascular medicine, routinely used across emergency, inpatient, and outpatient settings to detect arrhythmias, ischemia, conduction abnormalities, and a wide range of other cardiac conditions. Despite the broad use of the ECG, its accurate interpretation is challenging and requires substantial training and experience in cardiac electrophysiology. Moreover, even experienced ECG annotators show significant inter-rater and intra-rater variability, leading to missed diagnoses, inappropriate or unnecessary interventions, and delayed treatments (Cook et al., 2020; Rafie et al., 2021). The combination of high clinical volume and persistent interpretive challenges has motivated the development of computational systems capable of reliable and consistent ECG interpretation.

ECG interpretation is a multifaceted process rather than a binary or multiclass classification skill. It requires basic knowledge of the electrophysiology of the heart, ECG acquisition technologies, analog and digital frontend requirements, sampling and filtering specifications. It also involves quantitative measurements derived from the waveform, such as heart rate, rhythm, and interval durations including the PR, QRS, and QT intervals. Moreover, ECG diagnosis is descriptive and multi-labeled, where patterns are identified, associated with their electrophysiological mechanisms, and integrated with the clinical and personalized context to form a conclusion. The measurement component is primarily programmatic and conducted automatically via software, whereas the descriptive component relies on holistic knowledge of electrophysiology, cardiology and the subject's background.

Overall, the knowledge base required for accurate ECG diagnosis is broad and cross-disciplinary, motivating the use of methods that can integrate information across domains. LLMs have been promising in similar applications. They encode broad knowledge spanning natural sciences to engineering (Brown et al., 2020; Hendrycks et al., 2020), and have reached close to expert-level performance on similar clinical tasks (cf. Section 2). Despite this progress, current evaluations of clinical language models focus either on broad medical knowledge or on the interpretation of specific ECG anomalies. As a result, the background and contextual knowledge of the ECG, beyond specific ECG anomalies, is often not considered by existing LLMs. Therefore, it remains unclear how much a language model actually “understands" an ECG, rather than only classifying or describing patterns. To address this gap, we made the following contributions: (i) We used commercial LLMs to parse ECG-related literature and constructed a text-only database of general ECG questions and answers comprising 10,904 True/False question pairs (21,808 questions in total). The questions were derived from 23 scientific ECG resources and specialized ECG conference proceedings, spanning various ECG-related topics such as clinical, electro-physiological, biomedical signal processing, ECG diagnosis, and ECG hardware and electronics. (ii) We evaluated a diverse set of proprietary, general-purpose, biomedical, and open-access language models on this benchmark in a zero-shot setting. (iii) We explored whether smaller models in the 7 to 14 billion parameter range can be specialized for ECG knowledge using low-rank adaptation. (iv) We extended the evaluation to external ECG-related test sets extracted from MedQA (USMLE style) (Jin et al., 2021) and MedMCQA (Pal et al., 2022), whose labels come from official examination keys and are therefore independent of our generation pipeline; this analysis is reported in Supplement B.

## 2 EXISTING MEDICAL Q&A AND DOMAIN-SPECIFIC LANGUAGE MODELS

Multiple-choice benchmarks, such as MedQA (Jin et al., 2021), MedMCQA (Pal et al., 2022), and PubMedQA (Jin et al., 2019) have been used as references to evaluate medical language models. On these benchmarks, both general-purpose and domain-adapted models, including Med-PaLM (Singhal et al., 2023), Clinical Camel (Toma et al., 2023), and BioInstruct (Tran et al., 2024), have demonstrated strong, and, in some cases expert-level, accuracy through prompting, instruction tuning, and parameter efficient fine-tuning. However, these benchmarks assess broad clinical knowledge rather than ECG knowledge; Health-LLM, for instance, focuses on physiological time-series from wearable sensors rather than the conceptual knowledge required for ECG interpretation. Accuracy on these benchmarks therefore provides limited insight into a model's understanding of ECG.

Recent research have increasingly integrates language models in ECG applications. ECG-QA (Oh et al., 2023) paired the PTB-XL and MIMIC-IV ECG datasets, with seventy expert-validated question templates spanning verify, choose, and query formats, and evaluates language models by translating signal model outputs into text. Q-HEART (Pham et al., 2025), addressed ECG-related question answering through a knowledge-informed multimodal LLM that combines ECG representations, retrieved clinical context, and instruction tuning, achieving state-of-the-art performance on ECG-QA. Using ECG images, ECGInstruct (Liu et al., 2024) compiled over one million ECG image instruction-tuning samples and trains PULSE, an open multimodal model for ECG image interpretation. PULSE is evaluated on the ECGBench dataset across feature recognition, rhythm analysis, morphology assessment, and report generation. MEIT (Wan et al., 2025) instruction-tuned nine language models for ECG report generation by integrating signal representations into the model's latent space and benchmarked them on more than 800,000 reports.

![](images/807d471d7aa897fc766fd792dfacc4ead97c5f86d581ac84519806e50489c604.jpg)  
Figure 1: Overview of ECGQuest creation framework. (1) ECG reference materials were anonymized, split into pages, routed by content to a text or vision-based generation model for text extraction, and passed to GPT-4o to generate paired True/False questions. (2) Closed, open, and medically specialized models were tested on the held-out test set (n=1,050), and external ECG-focused MedQA and MedMCQA questions, extracted by GPT-4o ECG-topic filtering and converted to True/False for independent validation. (3) small open models (7–14 B parameters) were fine-tuned with LoRA adapters and re-evaluated.

## 3 METHODS

Figure 1 summarizes the three stages of our study: (i) Constructing ECGQuest, a literature-grounded benchmark question and answer (Q&A) resource from the ECG literature; (ii) evaluating a broad set of closed source, open source, and medically specialized language models in a zero-shot setting on a held-out test set, and, as an independent check of generalization, on two external exam-labeled ECG question sets whose construction and results are given in Supplement B; (iii) fine-tuning compact open source models using Low-Rank Adaptation (LoRA) (Hu et al., 2022).

## 3.1 ECGQUEST DATASET CONSTRUCTION

An ECG-specific question-answer benchmark was constructed by converting a curated corpus of electrocardiography reference materials into balanced, labeled True/False items. The construction pipeline consists of four stages: (i) corpus assembly and text extraction, (ii) question generation, (iii) label balancing through paired negation, (iv) and validation and deduplication. Each stage is fully automated, and each generated item is linked to its source, ensuring that question correctness depends on the reference text itself, and not on the prior knowledge of the model.

Data Source: The input corpus used for Q&A extraction comprises 23 electrocardiography reference documents from five source categories: (1) textbooks (Hall & Hall, 2020; Gertsch & Cannon, 2013; Kamath et al., 2012; Hampton, 2013; Thaler, 2017; Jevon, 2009; Surrena, 2010), (2) clinical guidelines and manuals (Vera, 2024; Rawshani, 2017; LifeSaver CPR, 2026; Philips Medical Systems, 2016; GE Healthcare, 2019), (3) ECG device manuals (Philips Medical Systems, 2023; 2019; GE Healthcare, 2016; Nihon Kohden, 2016; Cardiac Insight, Inc., 2025; Welch Allyn, Inc., 2023; SCHILLER AG, 1992; 2020; Jackson Heart Study Central ECG Reading Center and EPICARE, 2009), (4) online clinical tutorials and teaching materials (Yanowitz, 2018; DoctorProfessorPatrick, 2026), and (5) full-text conference proceedings from the Computing in Cardiology (CinC) conference published between 2003 and 2025 (Cin, 2003; 2004; 2005; 2006; 2007; 2008; 2009; 2010; 2011; 2012; 2013; 2014; 2015; 2016; 2017; 2018; 2019; 2020; 2021; 2022; 2023; 2024; 2025). These resources span the full range of ECG literature, from foundational cardiac electrophysiology and clinical interpretation texts to manufacturer documentation. Together, they cover ECG knowledge at multiple levels, including waveform morphology, clinical diagnosis, electrophysiological mechanisms, acquisition procedures, instrumentation, and device-specific technical specifications.

Data Preprocessing: All source files were in vector PDF format. In order to keep track of the integrity of the dataset, each source PDF was renamed to its 128-bit MD5 (Message-Digest Algorithm 5; RFC 1321) checksum before processing, computed over the raw file bytes and rendered as a 32- character hexadecimal string. Because all intermediate stages operated on hashed identifiers, the generation pipeline was blind to the source title. Each hashed document was programmatically split into separate single-page PDF files. From each page, we extracted the text using pdfplumber (Singer-Vine, 2015). The extracted text was then serialized into XML files that include source reference MD5 checksum, reference category, page number, character count, and text content. The character count was calculated at this stage and subsequently used as an indicator to decide how each page should be processed for question generation (due to LLM and SLM token size limitations). Pages without extractable text, typically image-only pages such as full-page ECG tracings, were logged separately and directed to the vision-based generation path described below.

Q&A Generation: Questions were generated from the extracted text using a GPT-4o, which operates in one of three modes determined automatically by the character count of each page extracted in Section 3.1. Pages containing 400 or more characters were processed in text mode, using only the extracted page text. Pages with 100 to 399 characters were processed in both modes, utilizing both the extracted text and a rendered image of the page, as sparse pages frequently contain essential information in figures, tables, or captions. Pages with fewer than 100 characters were processed in vision mode, utilizing only the rendered page image, which is appropriate for image-dominant pages with minimal extractable text. In all three modes, the model generated factual statements limited to four ECG topics: (i) waveform features (e.g., P wave, QRS complex, ST segment, T and U waves, intervals, morphology, etc.); (ii) clinical interpretation and diagnostic significance of ECG patterns (e.g., ischemia, infarction, arrhythmia, conduction blocks, hypertrophy, electrolyte effects, and normal versus abnormal decision criteria, etc.); (iii) the physiological basis of ECG findings (e.g., conduction system, depolarization and repolarization, action potentials, automaticity, refractory periods, etc.); and (iv) ECG device specifications (e.g., leads, electrodes, gain, filters, paper speed, calibration, artifacts, etc.). Each generated item was combined into a structured form including the page number, the question, a True/False label, and the original source text.

As shown Supplement C, GPT-4o was instructed to deliver questions from the source text alone, without reliance on external medical knowledge or requiring viewing an image (for example, “the rhythm shown, below", or similar that appears in many references and manuals). This constraint was to ensure that every question can be answered using clinical knowledge available in the provided text alone, independent of an image. Since the instructions in the prompt are not necessarily followed by the language model, the generated items were passed through a postprocessing verification stage (again conducted by a language model) to evaluate each item against these criteria: an item was rejected if its source text could not be found on the source page; if its question referenced the source figure as detected through a dictionary of image-referencing terms such as shown, figure, or this tracing; or if it was malformed, missing a required field, or contained an ambiguous label other than True or False. In total, 638 prompt responses were rejected at this stage. Representative examples and the reasons for their removal are provided in the Supplement C. For pages processed in both modes, text- and image-conditioned outputs were de-duplicated by another batch post-processing to remove redundant items. Each remaining item retains its origin, including source document MD5, page number, and generation mode, ensuring that every question can be traced to its originating page. The prompts used to generate the dataset are provided in Supplement C.

True-False Label Balancing: True-False questions that passed the verification stages in Section 3.1 had an imbalanced label distributions, as individual pages can contain true or false statements depending on their content. To balance the label distribution, each validated question was paired with its negative form derived from the same source text, to ensure that the dataset contains an equal number of true and false questions. For each true question, GPT-4o was prompted to generate a false counterpart that contradicts the source text by altering a minor clinical detail or by paraphrasing while preserving the source text (by negating an assertion or substituting an incorrect value, direction, or anatomical or temporal term). Conversely, for each false question, the model was prompted to produce the corresponding true statement by restating the fact as expressed in the source text. In both cases, the counterpart was derived solely from the source text, without introducing any concepts not present in the original; so that the pair differs primarily in the detail determining the label. The complete label-balancing prompt is provided in Supplement C.

To verify the negation process, each counterpart question was verified programmatically: (i) The source text must remain identical to the original item; (ii) the counterpart must have the correctly flipped label, differ from the original. Counterparts failing any of these criteria were discarded. This process resulted in question-answer pairs with balanced label distribution, where every true/false statement is paired with a minimally edited false/true version of the same question. This pairing prevents models from exploiting irrelevant clues, such as true/false prevalence or topic, as shortcuts to the label. The resulting pairs were de-duplicated in a final stage to remove near-identical facts repeated across documents. Two pairs were considered duplicates if the Jaccard similarity of their normalized source text (in lowercase and stripped of punctuation and whitespace) was at least 0.90. Because the sources overlap in content, this de-duplication step, prevented redundant questions across training/testing data splits.

The final ECGQuest corpus includes 10,904 unique question pairs (or 21,808 individual True/False questions) with a balanced 50% True-False label distribution. GPT-4o was further used to categorize the questions under four categories. As shown in Table 2 in the Supplement, 6735 pairs are related to ECG diagnosis, 1917 pairs to ECG waveform characteristics, 1200 pairs to ECG acquisition technologies, and 1052 pairs to basic cardiac physiology.

## 3.2 MODEL FINE-TUNING

For fine-tuning, the dataset was partitioned into training, validation, and test splits in an approximately 80:10:10 ratio, with both members of each True/False pair assigned to the same split. All models were fine-tuned targeting maximum macro-F1 score, which due to the balanced True/False set of ECGQuest questions is technically identical to maximizing accuracy. The two categories of fine-tuned models are detailed below.

Encoder-based model fine-tuning: As non-generative baselines, two pre-trained encoders were finetuned as binary True/False classifiers: BERT (Devlin et al., 2019) and BiomedBERT (Gu et al.. 2021). Each model was fine-tuned with a two-class classification on the same training split as all other models and evaluated on the identical test sets. In contrast to the generative models, no prompt, chat template, or text generation was involved. These classifiers establish a lower bound, indicating how much of the task can be solved by direct supervised classification.

Instruction-based model fine-tuning: BioMistral-7 B, Llama-3.1-8 B, Qwen2.5-14 B, Gemma-4- 12 B, and DeepSeek-R1-14 B were fine-tuned on the training split (8,803 question pairs) and evaluated on the held-out test split (n = 1,050). All models were adapted using LoRA (Hu et al., 2022) on the attention and MLP projections, with an effective batch size of 64, a dropout of 0.1, and LoRA α = 2r. BioMistral-7 B used a LoRA rank of 16 and a False-class weight of 1.5, whereas Llama-3.1-8 B, Qwen2.5-14 B, Gemma-4-12 B, and DeepSeek-R1-14 B used a rank of 64 and False-class weights of 3.0, 0.8, 1.0, and 1.5, respectively. The class-weighted loss was used to reduce each base model's tendency to overpredict either True or False.

## 4 RESULTS

## 4.1 ZERO-SHOT BENCHMARKING OF LLMS AND SLMS ON ECGQUEST

We evaluated a comprehensive set of language models on the held-out test split (n=1,050) in a zero-shot setting, with no weight updates, to measure how well existing models answer ECG True/- False questions. The models span three groups: closed-source general-purpose models, open-source general-purpose models, and medically specialized open-source models. Each question was presented with a short instruction and requests a single True or False answer. The models, their parameter counts, and performance across multiple metrics are presented in the upper part of Table 1.

Zero-shot accuracy ranges from 49.5% to 74.4%, with no model exceeding 75%. GPT-5 performs best (74.4% accuracy; 74.1% macro-F1), followed by GPT-4o (71.7%; 71.7%) and the leading opensource model, Gemma-4-31 B (71.0%; 70.9%). The gap between the best closed-source and best open-source model was 3.4% accuracy points. Gemma-4-31 B is within 0.7 points of GPT-4o at a small fraction of its estimated parameter count.

The gap between accuracy and macro-F1 identifies directional response biases. MedAlpaca-7 B (sensitivity 93.4, specificity 6.0; macro-F1 38.0 versus accuracy 50.2%) and BioMistral-7 B (sensitivity 90.2, specificity 18.7; macro-F1 48.0 versus accuracy 54.9%) are biased toward True, while Gemma-3-270 M (sensitivity 0.4, specificity 99.8; macro-F1 33.5 versus accuracy 49.5%) and Qwen2.5-Instruct-14 B (sensitivity 27.5, specificity 93.6; macro-F1 55.5 versus accuracy 60.2%) are biased toward False. In contrast, GPT-5, GPT-4o, Llama-3.1-Instruct-8 B, MedGemma-4 B,

DeepSeek-R1-Distill-Qwen-14 B, and the larger Gemma-3 and Gemma-4 models have accuracy and macro-F1 within 0.3 points of each other, indicating that their accuracy reflects discrimination between correct and incorrect ECG statements rather than a directional answering bias.

Between models of the same family, with varied number of parameters, accuracy increases consistently. Gemma-3 improves monotonically from 49.5% at 270 M, 50.2% at 1 B, 56.4% at 4 B, 64.6% at 12 B, to 67.1% at 27 B; Gemma-4 is non-decreasing over the same range (58.1% at 2 B, 62.9% at 4 B and 12 B, 66.5% at 26 B, 71.0% at 31 B), with a plateau between 4 B and 12 B; DeepSeek-R1-Distill-Qwen shows the same pattern (53.2% at 1.5 B, 59.0% at 7 B, 69.8% at 14 B). The same pattern is observed between closed-source models.

Interestingly, medical specialization of the models did not necessarily enhance ECG knowledge. The performance of medical models ranges from 50.2% (MedAlpaca-7 B) to 59.5% (MedGemma-4 B), with eight open general purpose models outperforming the best medical model. MedGemma-4 B is the only medical model that achieves the performance of mid sized general models, yet it remains below Gemma-4-4 B (62.9%) at a comparable size.

## 4.2 FINE-TUNED MODEL PERFORMANCES

Encoder-based models: As non-generative baselines, the fine-tuned versions of BERT (Devlin et al., 2019) and BiomedBERT (Gu et al., 2021) both perform near the 50% chance level. BERT achieves 52.1% and BiomedBERT achieves 54.0%. These near-chance results highlight the poor performance of these models for ECG-related tasks.

Fine-tuned small language models: The lower part of Table 1 presents results for the five fine-tuned language models and a voting-based ensemble built from them. Fine-tuning outperformed all models on ECGQuest, by between 6.5 and 14.1 percentage points, enabling 7–14 B models match or exceed GPT-5 and GPT-4o, the two strongest models in the zero-shot setting, despite substantially fewer parameters. DeepSeek-R1-14 B reaches 76.3% (macro-F1 76.1), outperforming GPT-5 (74.4%), while Qwen2.5-14 B (74.0%) and Gemma-4-12 B (73.3%) both surpass GPT-4o. Llama-3.1-8 B (71.7%) matches GPT-4o, and BioMistral-7 B, the weakest base in the group at 54.9%, gains 14.1 points to reach 69.0%, the largest improvement among the fine-tuned models. We also evaluated an ensemble model by majority voting between the five adapted models. This resulted in the best result on ECGQuest: 78.5% accuracy with macro-F1 78.4. All five are balanced, with closely aligned accuracy and macro-F1, and all outperform the encoder baselines reported in the previous section.

Generalization to external exam-labeled questions: To test whether these gains reflect transferable ECG knowledge rather than adaptation ECGQuest, we evaluated every model on ECG-related subsets of MedMCQA (Pal et al., 2022) and MedQA (Jin et al., 2021), converted to balanced True/- False items using their official examination keys. The construction procedure is described in Supplement B and the complete results are reported in Table 3 of Supplement B. The zero-shot ranking is broadly preserved across all three sets, but the effect of fine-tuning depends strongly on the base model: it improves weak or class-biased models, such as BioMistral-7 B (51.3% to 60.5% on MedMCQA) and Qwen2.5-Instruct-14 B (67.3% to 73.0%), while reducing accuracy for the already balanced Llama-3.1-Instruct-8 B and DeepSeek-R1-14 B. We discuss this asymmetry in Section 5.

## 4.3 AGREEMENT WITH BENCHMARK LABELS

We further assessed inter-model agreement on ECGQuest using pairwise Cohen's κ between model predictions and the benchmark reference labels (Figure 2). Because the questions and reference answers were generated and evaluated using language models, these values measure agreement with the benchmark or between models, not clinical correctness. Overall, agreement with the reference labels generally follows the accuracy ranking. Among models evaluated in zero-shot, the agreement is highest for GPT-5 (κ = 0.49), GPT-4o (0.43), followed by Gemma 4-31 B (0.42), and DeepSeek-R1-14 B (0.40). In contrast, strongly biased models show almost no agreement despite accuracies near 50%, including MedAlpaca-7 B (—0.01) and Gemma-3-270 M (0.00). Fine-tuning leads to the biggest improvements; the largest of these occured in the models that started as weakest: agreement rises from 0.09 to 0.38 for BioMistral-7 B, from 0.17 to 0.41 for Llama-3.1-Instruct-8 B, from 0.21 to 0.48 for Qwen2.5-Instruct-14 B, from 0.25 to 0.47 for Gemma-4-12 B, and from 0.40 to 0.49 for DeepSeek-R1-14 B. All five fine-tuned models consequently rank among the strongest in the panel, with DeepSeek-R1-14 B (FT) matching GPT-5 at 0.49 and Qwen2.5-Instruct-14 B (FT) and Gemma-4-12 B (FT) surpassing GPT-4o at 0.48 and 0.47. Inter-model agreement is likewise concentrated among the top performers. The two closed-models agree most strongly with each other (κ = 0.60), and Gemma-4-31 B agrees with GPT-4o and GPT-5 at 0.57 and 0.56, closely followed by DeepSeek-

Table 1: Zero-shot and fine-tuned language model performances on ECGQuest (n=1,050). Acc: accuracy; F1: macro-averaged F1 score; P: precision; R/Sen: recall/sensitivity; Spec: specificity Across all models, the best value per metric column is in bold and the runner-up is underlined.
<table><tr><td colspan="2"></td><td colspan="5">ECGQuest</td></tr><tr><td>Model</td><td>Params</td><td>Acc (%)</td><td>F1</td><td>P</td><td>R/Sen</td><td>Spec</td></tr><tr><td colspan="7">General purpose (closed source)</td></tr><tr><td>GPT-5</td><td>2T*</td><td>74.4</td><td>74.1</td><td>81.8</td><td>63.5</td><td>85.5</td></tr><tr><td>GPT-40</td><td>200 B*</td><td>71.7</td><td>71.7</td><td>72.8</td><td>70.4</td><td>73.0</td></tr><tr><td>GPT-3.5</td><td>175B*</td><td>63.1</td><td>61.7</td><td>60.0</td><td>81.4</td><td>44.5</td></tr><tr><td colspan="7">General-purpose (open source) Mistral-Instruct-v0.3</td></tr><tr><td></td><td>7B</td><td>57.6</td><td>57.4</td><td>57.2</td><td>63.6</td><td>51.4</td></tr><tr><td>Llama-3.1-Instruct</td><td>8B</td><td>58.4</td><td>58.4</td><td>59.3</td><td>56.3</td><td>60.5</td></tr><tr><td>Qwen2.5-Instruct</td><td>14 B</td><td>60.2</td><td>55.5</td><td>81.6</td><td>27.5</td><td>93.6</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>1.5B</td><td>53.2</td><td>52.9</td><td>53.6</td><td>59.1</td><td>47.0</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>7B</td><td>59.0</td><td>58.1</td><td>60.5</td><td>68.1</td><td>48.4</td></tr><tr><td>Zerrt a-nn DeepSeek-R1-Distill-Qwen Gemma-3</td><td>14B 270 M</td><td>69.8</td><td>69.8 33.5</td><td>71.4 66.7</td><td>70.5</td><td>69.1</td></tr><tr><td>Gemma-3</td><td></td><td>49.5</td><td>50.0</td><td>50.9</td><td>0.4</td><td>99.8</td></tr><tr><td>Gemma-3</td><td>1B 4B</td><td>50.2</td><td>53.2</td><td>54.6</td><td>43.3</td><td>57.2</td></tr><tr><td>Gemma-3</td><td>12 B</td><td>56.4 64.6</td><td>64.4</td><td>63.3</td><td>81.4 71.4</td><td>30.8</td></tr><tr><td>Gemma-3</td><td>27B</td><td>67.1</td><td>67.1</td><td>69.2</td><td>63.1</td><td>57.6</td></tr><tr><td>Gemma-4</td><td>2B</td><td>58.1</td><td>56.1</td><td>65.4</td><td>36.3</td><td>71.3</td></tr><tr><td>Gemma-4</td><td>4B</td><td></td><td>62.7</td><td>66.0</td><td>54.8</td><td>80.3 71.1</td></tr><tr><td>Gemma-4</td><td></td><td>62.9</td><td>62.7</td><td>62.9</td><td></td><td></td></tr><tr><td>Gemma-4</td><td>12B</td><td>62.9</td><td>66.4</td><td>68.4</td><td>64.4</td><td>61.1</td></tr><tr><td>Gemma-4</td><td>26B</td><td>66.5</td><td></td><td>72.8</td><td>62.7</td><td>70.3</td></tr><tr><td></td><td>31 B</td><td>71.0</td><td>70.9</td><td></td><td>68.0</td><td>74.0</td></tr><tr><td>Medical-specialized (open source)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedGemma</td><td>4B</td><td>59.5</td><td>59.5</td><td>59.8</td><td>61.0</td><td>58.0</td></tr><tr><td>MedAlpaca</td><td>7B</td><td>50.2</td><td>38.0</td><td>50.4</td><td>93.4</td><td>6.0</td></tr><tr><td>BioMistral</td><td>7B</td><td>54.9</td><td>48.0</td><td>53.2</td><td>90.2</td><td>18.7</td></tr><tr><td>MedAlpaca</td><td>13 B</td><td>55.1</td><td>53.5</td><td>54.2</td><td>73.3</td><td>36.6</td></tr><tr><td colspan="2">Encoder baselines BERT</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">BiomedBERT</td><td>110M</td><td>52.1</td><td>52.1 52.7</td><td>52.4</td><td>51.8</td></tr><tr><td colspan="2">Decoder LLMs</td><td>110M</td><td>54.0</td><td>53.5</td><td>55.8</td><td>43.5</td><td>64.7</td></tr><tr><td colspan="2">BioMistral</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Rin-nned</td><td>7B</td><td>69.0</td><td>68.9</td><td>68.7</td><td>70.8</td><td>67.1</td></tr><tr><td colspan="2">Llama-3.1-Instruct</td><td>8B</td><td>71.7</td><td>71.7</td><td>71.7</td><td>71.1</td><td>71.7</td></tr><tr><td colspan="2">Gemma-4</td><td>12B</td><td>73.3</td><td>73.0</td><td>69.6</td><td>84.0</td><td>62.4</td></tr><tr><td colspan="2">DeepSeek-R1-Distill-Qwen</td><td>14B</td><td>76.3</td><td>76.1</td><td>74.9</td><td>82.2</td><td>69.9</td></tr><tr><td colspan="2">Qwen2.5-Instruct</td><td>14B</td><td>74.0</td><td>74.0</td><td>74.0</td><td>75.0</td><td>73.0</td></tr><tr><td colspan="2">Ensemble Model</td><td>55B**</td><td>78.5</td><td>78.4</td><td>77.1</td><td>83.7</td><td>72.9</td></tr></table>

Parameter counts for GPT are not officially disclosed by OpenAI; the reported values are unofficial estimates. 55 B is the total number of parameters across the five fine-tuned models used for ensemble voting

R1-14 B at 0.56 and 0.50, indicating that the strongest models tend to make similar choices. The fine-tuned models form a separate cluster, agreeing more with one another (κ = 0.42 to 0.59) than with their base models. This shows that task-specific fine-tuning pushes different model types toward a shared, label-focused decision pattern.

Additional views of these results are provided in the supplement. Supplement A reports a questionlevel difficulty and clarity profile, in which approximately 73% of test items are answered correctly by at least half of evaluated models, with a small residual subset that almost none answers correctly. Supplement A reports category-level accuracy for the ten strongest models, where no single model leads across all four ECG knowledge categories and Fine-tuned models lead in most categories.

![](images/7762f6a0730a343858c00c08973410c8b41b9a0c51718169b39208ed0667a802.jpg)  
Figure 2: Pairwise Cohen's κ agreement between model predictions and the benchmark reference labels, computed on the held-out benchmark test set (n = 1,050). Higher values (darker) indicate stronger agreement. Values quantify agreement with the benchmark rather than clinical correctness.

## 4.4 PARETO FRONTIERS ON RECEIVER OPERATING CHARACTERISTICS AND PRECISIONRECALL CURVES

Any statistical classification task, including language-based tasks, is multiobjective in nature, and its performance reflects a balance between how sensitive and specific the model is. However, most language models do not provide us with access to their internal decision thresholds, which would allow us to change their operating points through the choice of parameters. Nonetheless, we can visualize their multiobjective tradeoff. Figure 3 shows each model as a single point in Receiver Operating Characteristic (ROC) space (left) and precision-recall (PR) space (right). We have also shown the models on the Pareto frontier, i.e., models for which no other model performs at least as well on both objectives and strictly better on at least one. Since the fine-tuning objective was to maximize the macro-F1 score (equivalently maximizing accuracy because of the balanced True/False questions), we can envision each fine-tuned model as moving toward the ideal classifier in ROC and PR spaces. In ROC space, this brings the models closer to the upper-left corner and increases Youden's J statistic (J = sensitivity + specificity — 1), increasing the diagonal distance of an operating point above the chance diagonal. In Figure 3, the gray arrows connect each base model to its fine-tuned counterpart. We can see that J increases in every case: from 0.211 to 0.480 for Qwen2.5-Instruct-14 B, whose recall rises from 27.5 to 75.0 at essentially unchanged specificity; from 0.09 to 0.38 for BioMistral-7 B, which moves in the opposite direction as its specificity climbs from 18.7 to 67.1; and from 0.40 to 0.52 for DeepSeek-R1-14 B. The ensemble model outperforms individual models, with a false-positive rate of 0.27 and a recall of 0.837, giving J=0.57, compared with GPT-5's 0.49.

![](images/e5c4d368e4d83c099daab86f44a2fa8d2cdc576da5c863d339b97d66d810082d.jpg)  
Figure 3: Operating points of all evaluated models in ROC (left) and PR (right) spaces on ECGQuest. As each model make hard true/false decisions, it contributes a single operating point rather than a curve. Marker color encodes the approximate parameter count; circles denote open source models and crosses closed source models. Grey arrows connect each base model to its fine-tuned counterpart (FT), red dashed lines trace the Pareto frontier, and red rings mark Pareto-optimal models. The black dashed lines indicate chance performance.

## 5 DISCUSSION AND FUTURE WORK

The results demonstrate that ECG knowledge varies across models and is not consistent even in medical-specific language models, highlighting the need for dedicated ECG benchmarks and ECGspecific language models, since general medical performance does not always predict how well a model will perform on ECG-specific tasks. Within the same architecture, performance generally improves with model size, as observed in both Gemma families. Across families, however, size does not always correlate with performance; pretraining data and base-model quality matter at least as much. Medical-specific models are a clear example: medically fine-tuned models under-performed larger general-purpose models, and MedGemma-4 B, the only competitive medical model, is outperformed by Gemma-4-4 B at identical parameter size. This suggests that medical-specific tuning does not necessarily provide the domain-specific ECG knowledge required for this task.

On a balanced True/False dataset, accuracy provides an overall measure of performance but may obscure class-specific imbalance. For example, a model that always predicts one class achieves 50% accuracy but only 33.3% macro-F1 because its F1 score for the other class is zero. We therefore report and compare macro-F1 alongside accuracy. A large gap between accuracy and macro-F1 indicates asymmetric performance across the True and False classes, although the direction of the bias must be determined from the class-specific results. Although several models exhibit a bias toward either True or False responses, GPT-5, GPT-4o, Llama-3.1-Instruct-8 B, MedGemma-4 B, and the larger Gemma-3 and Gemma-4 models have balanced performance across the two classes.

Fine-tuning SLMs closed the gap between open-source SLMs and commercial LLMs such as GPT-5 and GPT-4o, the two strongest models in the zero-shot evaluation. All five adapted models outperform the encoder baselines by a wide margin and match or exceed GPT-4o; while DeepSeek-R1-14 B and the majority vote over the five models exceed GPT-5. That a 14 B model matches a proprietary model of far larger scale indicates that the ECG knowledge can be transferred into a compact model. Also, because of the training setup and balanced dataset we created, adaptation mitigated the initial model class biases, moving their operating point toward the Youden's J-statistics.

On the external exam-labeled sets (Supplement B), LoRA improved performance and corrected directional bias in BioMistral-7 B and Qwen2.5-14 B, but for balanced base models such as Llama-3.1-Instruct-8 B or DeepSeek-R1-14 B it mainly specialized the model toward the ECGQuest distribution, reducing accuracy. We read this as a generalization gap rather than a training failure. The external sets also warrant caution: MedMCQA and MedQA are long-standing public resources. Their contents have likely entered the training corpora of existing models; so strong zero-shot scores there may partly reflect data leakage rather than ECG understanding.

As a limitation of our work, although ECGQuest questions and labels were generated by GPT-4o from ECG reference materials and grounded in the provided source text, the labels have not undergone expert over-read and should therefore be considered weak labels rather than clinical ground truth. To facilitate future expert validation, each ECGQuest item is linked to a quote from its source.

Future work should expand ECGQuest with additional public resources, including clinical guidelines, educational materials, open-access articles, and abstracts from cardiology conference proceedings. Furthermore, items should be over-read by human experts to identify ambiguous items, correct clinically inaccurate labels, and establish an expert-validated benchmark. Because each ECGQuest item is linked to its source text, expert review can be conducted systematically while maintaining full traceability. ECGQuest can then be paired with models that operate directly on waveforms or images, so that measurement-based models supply the quantitative findings while an ECG-specific language model interprets them using electrophysiological and clinical knowledge, with expert review closing the loop. Also, the data preparation, fine-tuning, and evaluation pipeline is model-agnostic and can be reapplied as new models become available. Future studies can extend this framework to additional models and cardiology modalities, including echocardiography, cardiac magnetic resonance imaging (MRI), and cardiac computed tomography (CT), enabling specialized models that integrate complementary cardiovascular information

## 6 CONCLUSION

We developed ECGQuest, a literature-grounded balanced True/False dataset derived from ECG references for evaluating and fine-tuning language models on conceptual ECG knowledge. Each item is source-linked and paired with a minimally modified opposite-label counterpart. The evaluation produced three main findings. First, ECG knowledge varies substantially across models and is not guaranteed by generic medical-specialized models. Second, within the same model family, performance generally improves with increasing model size. Third, parameter-efficient fine-tuning enables relatively small 7-14 B models to match or outperform substantially larger proprietary models on ECGQuest, though evaluation on external exam labeled ECG questions (Supplement B) shows that this benefit is largest for weak or class-biased base models. The ECGQuest Q&A set and the finetuned models presented in this work are publicly available1.

## REFERENCES

Proceedings of Computers in Cardiology 2003, volume 30 of Computers in Cardiology, Thessaloniki, Greece, 2003. IEEE. ISSN: 0276-6547/03.

Proceedings of Computers in Cardiology 2004, volume 31 of Computers in Cardiology, Chicago, IL, USA, 2004. IEEE. ISSN: 0276-6547/04.

Proceedings of Computers in Cardiology 2005, volume 32 of Computers in Cardiology, Lyon, France, 2005. IEEE. ISSN: 0276-6547/05.

Proceedings of Computers in Cardiology 2006, volume 33 of Computers in Cardiology, Valencia, Spain, 2006. IEEE. ISSN: 0276-6547.

Proceedings of Computers in Cardiology 2007, volume 34 of Computers in Cardiology, Durham, NC, USA, 2007. IEEE. ISSN: 0276-6574.

Proceedings of Computers in Cardiology 2008, volume 35 of Computers in Cardiology, Bologna, Italy, 2008. IEEE. ISSN: 0276-6574.

Proceedings of Computers in Cardiology 2009, volume 36 of Computers in Cardiology, Park City, UT, USA, 2009. IEEE. ISSN: 0276-6574.

Computing in Cardiology 2010, volume 37 of Computing in Cardiology, Belfast, UK, 2010. IEEE. ISSN: 0276-6574.

1Note: The dataset and developed models will be made public and their URL will be added to the paper upon manuscript acceptance.

Computing in Cardiology 2011, volume 38 of Computing in Cardiology, Hangzhou, China, 2011. IEEE. ISSN: 0276-6574

Proceedings of Computing in Cardiology 2012, volume 39 of Computing in Cardiology, Krakow, Poland, 2012. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2013, volume 40 of Computing in Cardiology, Zaragoza, Spain, 2013. IEEE. ISSN: 2325-8861.

Proceedings of Computing in Cardiology 2014, volume 41 of Computing in Cardiology, Cambridge, MA, USA, 2014. IEEE. ISSN: 2325-8861.

Proceedings of Computing in Cardiology 2015, volume 42 of Computing in Cardiology, Nice, France, 2015. IEEE. ISSN: 2325-8861.

Proceedings of Computing in Cardiology 2016, volume 43 of Computing in Cardiology, Vancouver, BC, Canada, 2016. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2017, volume 44 of Computing in Cardiology, Rennes, France, 2017. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2018, volume 45 of Computing in Cardiology, Maastricht, Netherlands, 2018. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2019, volume 46 of Computing in Cardiology, Singapore, 2019. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2020, volume 47 of Computing in Cardiology, Rimini, Italy, 2020. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2021, volume 48 of Computing in Cardiology, Brno, Czech Republic, 2021. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2022, volume 49 of Computing in Cardiology, Tampere, Finland, 2022. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2023, volume 50 of Computing in Cardiology, Atlanta, GA, USA, 2023. IEEE. ISSN: 2325-887X.

Proceedings of Computing in Cardiology 2024, volume 51 of Computing in Cardiology, Karlsruhe, Germany, 2024. IEEE. ISSN: 2325-887X.

Computing in Cardiology 2025, volume 52 of Computing in Cardiology, São Paulo, Brazil, 2025. IEEE. ISSN: 2325-887X.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Cardiac Insight, Inc. CARDEA 20/20 ECG Operator's Manual: Resting ECG Analysis System, Model CS-2020. Cardiac Insight, Inc., Bellevue, WA, USA, 2025. Part no. PN00261-01, revised Apr. 2025.

{David A.} Cook, {So Young} Oh, and {Martin V. } Pusic. Accuracy of physicians' electrocardiogram interpretations: A systematic review and meta-analysis. JAMA Internal Medicine, 180(11): 1461–1471, November 2020. ISSN 2168-6106. doi: 10.1001/jamainternmed.2020.3989. URL 10.1001/jamainternmed.2020.3989.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N 19-1423/.

DoctorProfessorPatrick. ECG's Step by Step: A Clinical Review / ECG Interpretation Guide, 2026.

GE Healthcare. MUSE v9 Cardiology Information System: Operator's Manual. General Electric Company, 2016. Document no. 2059568-009, Revision D, Apr. 25, 2016.

GE Healthcare. Marquette 12SL ECG Analysis Program: Physician's Guide. GE Healthcare, 2019. Version 23, Revision C, Document No. 2056246-002C.

M. Gertsch and C.P. Cannon. The ECG: A Two-Step Approach to Diagnosis. Medicine. Springer Berlin Heidelberg, 2013. ISBN 9783662103159. URL https://books.google.com/b ooks?id=pp5HBAAAQBAJ.

Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans. Comput. Healthcare, 3(1), October 2021. doi: 10.114 5/3458754.URLhttps://doi.org/10.1145/3458754.

J.E. Hall and M.E. Hall. Guyton and Hall Textbook of Medical Physiology. Guyton Physiology. Elsevier, 2020. ISBN 9780323597128. URL https://books.google.com/books?id= 1nUgzAEACAAJ.

J. Hampton. The ECG Made Easy E-Book. Made Easy. Elsevier Health Sciences, 2013. ISBN 9780702052439.URL https://books.google.com/books?id=MTeYAAAAQBAJ.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations (ICLR), 2022.

Jackson Heart Study Central ECG Reading Center and EPICARE. Electrocardiography Assessment Manual: Technicians'Manual, Jackson Heart Study. Department of Epidemiology and Prevention, Division of Public Health Sciences, Wake Forest University School of Medicine, 2009. Jan. 22, 2009.

P. Jevon. ECGs for Nurses. Essential Clinical Skills for Nurses. Wiley, 2009. ISBN 9781405151016. URLhttps://books.google.com/books?id=gPRL0qeap8QC.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14), 2021. ISSN 2076-3417. doi: 10.3390/app11146421. URL https://www.mdpi.com/2076-3417/11/14/6421.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL https://aclanthology.org/D19-125 9/.

M.V. Kamath, M. Watanabe, and A. Upton. Heart Rate Variability (HRV) Signal Analysis: Clinical Applications. Taylor & Francis, 2012. ISBN 9781439849804. URL https : / /books . goog le.com/books?id=2gSzQ7uTEwAC.

LifeSaver CPR. ECG rhythm study guide, 2026.

Ruoqi Liu, Yuelin Bai, Xiang Yue, and Ping Zhang. Teach multimodal llms to comprehend electrocardiographic images. arXiv preprint arXiv:2410.19008, 2024.

Nihon Kohden. Electrocardiograph ECG-1350, ECG-1350A, ECG-1350K: Operator's Manual. Nihon Kohden, 2016. Document no. 0614-010451H.

Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-Myoung Kwon, and Edward Choi. ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram. In Advances in Neural Information Processing Systems 36, NeurIPS 2023, pp. 66277–66288. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023. doi: 10.52202/075280-2893. URL http://dx.doi.orq/10.52202/075280-2893.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Gerardo Flores, George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann (eds.), Proceedings of the Conference on Health, Inference, and Learning, volume 174 of Proceedings of Machine Learning Research, pp. 248–260. PMLR, 07–08 Apr 2022. URL https : //proceedings.mlr.pr ess/v174/pal22a.html.

Hung Manh Pham, Jialu Tang, Aaqib Saeed, and Dong Ma. Q-heart: ECG question answering via knowledge-informed multimodal llms. arXiv preprint arXiv:2505.06296, 2025.

Philips Medical Systems. Philips DXL ECG Algorithm: Physician's Guide. Philips Medical Systems, 2016. Revision E, Publication No. 453564106411.

Philips Medical Systems. PageWriter TC50/TC70 Cardiograph: Instructions for Use. Koninklijke Philips N.V., June 2019. Release A.07, Revision A, PN 453564890031.

Philips Medical Systems. PageWriter TC10 Cardiograph: Instructions for Use. Koninklijke Philips N.V., May 2023. Release 1.1, Revision E, PN 453564461421.

Nikita Rafie, Anthony H. Kashou, and Peter A. Noseworthy. ECG Interpretation: Clinical Relevance, Challenges, and Advances. Hearts, 2(4):505–513, 2021. ISSN 2673-3846. doi: 10.3390/hearts2040039.URLhttps://www.mdpi.com/2673-3846/2/4/39.

Araz Rawshani. Pocket guide to ECG interpretation, 2017.

SCHILLER AG. CARDIOVIT AT-6: Operating Manual. SCHILLER AG, Baar, Switzerland, August 1992. Art. no. 2.510019.

SCHILLER AG. CARDIOVIT AT-102 G2: User Guide. SCHILLER AG, Baar, Switzerland, 2020. Art. no. 2.511296, Rev. d, Nov. 1, 2020.

Jeremy Singer-Vine. pdfplumber: Plumb a PDF for detailed information about each text character, rectangle, and line, 2015. URL https://github.com/jsvine/pdfplumber. Python package. Accessed 2026-07-10.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

H. Surrena. Handbook for Brunner and Suddarth's Textbook of Medical-Surgical Nursing. Lippincott Williams & Wilkins, 2010. ISBN 9780781785921. URL https ://books. google.c om/books?id=ZFYpJO2GPmUC.

M. Thaler. The Only EKG Book You'll Ever Need. Wolters Kluwer Health, 2017. ISBN 9781496377241.URL https://books.google.com/books?id=y-lDDwAAQBAJ.

Augustin Toma, Patrick R Lawler, Jimmy Ba, Rahul G Krishnan, Barry B Rubin, and Bo Wang. Clinical camel: An open expert-level medical language model with dialogue-based knowledge encoding. arXiv preprint arXiv:2305.12031, 2023.

Hieu Tran, Zhichao Yang, Zonghai Yao, and Hong Yu. Bioinstruct: instruction tuning of large language models for biomedical natural language processing. Journal of the American Medical Informatics Association, 31(9):1821–1832, 09 2024. ISSN 1527-974X. doi: 10.1093/jamia/ocae 122.URLhttps://doi.org/10.1093/jamia/ocae122.

Matt Vera. EKG interpretation and heart arrhythmias cheat sheet. Nurseslabs, 2024. Updated Aug. 9, 2024.

Zhongwei Wan, Che Liu, Xin Wang, Chaofan Tao, Hui Shen, Jing Xiong, Rossella Arcucci, Huaxiu Yao, and Mi Zhang. MEIT: Multimodal electrocardiogram instruction tuning on large language models for report generation. In Findings of the association for computational linguistics: ACL 2025, pp. 14510–14527, 2025.

Welch Allyn, Inc. Welch Allyn RScribe 12-Lead Electrocardiograph System: User Manual. Welch Allyn, Inc., Skaneateles Falls, NY, USA, January 2023. Software version V6.2.X, document no. 80030137, Ver. A.

Frank G. Yanowitz. Introduction to ecg interpretation, 2018. URL https : //ecg . utah. edu/. Educational tutorial.

## SUPPLEMENTARY MATERIALS

## A ECGQUEST QUESTION COMPOSITION AND LEVEL OF DIFFICULTY

We review ECGQuest in terms of dataset composition, question difficulty, and model performance. The analysis includes the distribution of question pairs across dataset splits and ECG knowledge categories, the percentage of cases in which each question is answered correctly across tested models, and the performance of the top-performing models within each knowledge category.

Dataset Composition: GPT4o was queried to categorize ECGQuest under four categories listed in Table 2. The table reports the number of ECGQuest question pairs in each category across the training, validation, and test splits.

Table 2: Distribution of benchmark question pairs across dataset splits and question categories.
<table><tr><td>Question Category</td><td>Train</td><td>Validation</td><td>Test</td><td>Total</td></tr><tr><td>ECG Diagnosis</td><td>5,300</td><td>703</td><td>732</td><td>6,735</td></tr><tr><td>ECG Characteristics</td><td>1,499</td><td>224</td><td>194</td><td>1,917</td></tr><tr><td>ECG Acquisition Technologies</td><td>1,045</td><td>74</td><td>81</td><td>1,200</td></tr><tr><td>Basic Cardiac Physiology</td><td>959</td><td>50</td><td>43</td><td>1,052</td></tr><tr><td>All Types</td><td>8,803</td><td>1,051</td><td>1,050</td><td>10,904</td></tr></table>

Question-Level Difficulty Profile: At the question level, some questions were easier or harder for the models to answer, as reflected in the degree of agreement across model responses. Without expert review, this can also be a reflection of more clear or more ambiguous questions. Figure 4 shows the fraction of models that answered each benchmark question correctly, with questions ordered from easiest/clear to hardest/ambiguous. At least half of the evaluated models answered approximately 73% of the 1,050 test questions correctly, whereas fewer than half answered the remaining 27% correctly. The gradual decline across the ranked questions indicates substantial variation in question difficulty/clarity, ranging from items answered correctly by nearly all models to a small subset that almost none answered correctly.

![](images/f02e0d30a0cdfa6018872443cd0a731b79de75bcc4f5cd233a797b36a5547136.jpg)  
Figure 4: Question-level difficulty/clarity profile on the ECGQuest test set (n = 1,050). Questions are ordered from easiest/clear to hardest/ambiguous according to the fraction of evaluated models that answered each question correctly. The green region represents questions answered correctly by at least half of the models, while the red region represents questions answered correctly by fewer than half of the models. The dashed vertical line indicates that approximately 73% of the questions were answered correctly by at least half of the evaluated models.

Category-Level Performance: Figure 5compares the category-level accuracy of the ten models with the highest overall performance. The results show that no single model achieved the best performance across all ECG knowledge categories. In ECG Diagnosis, the fine-tuned DeepSeek-R1- Distill-Qwen 14 B model achieved the highest accuracy at 76.8%, followed by GPT-5 at 76.1%. For ECG Acquisition Technologies, GPT-4o ranked first with 71.6%, while the fine-tuned Gemma-4 12 B model followed closely with 70.4%. In ECG Characteristics, the fine-tuned DeepSeek-R1-Distill-Qwen 14 B model achieved the highest accuracy of 76.5%, followed by the fine-tuned Qwen2.5- Instruct 14 B model at 71.1%. The strongest overall category-level performance was observed in

![](images/253c4440641135f6ee621b40a1135433a12138143ae24d84e03a853ff0ab435c.jpg)  
Figure 5: Category-level accuracy of the ten models with the highest overall performance across four ECG knowledge categories: ECG Diagnosis, ECG Acquisition Technologies, ECG Characteristics, and Basic Cardiac Physiology.

Basic Cardiac Physiology, where the fine-tuned Qwen2.5-Instruct 14 B model achieved the highest accuracy at 90.7%, followed closely by the fine-tuned DeepSeek-R1-Distill-Qwen 14 B model at 88.9%. Overall, these results highlight complementary strengths across models and suggest that fine-tuning was particularly beneficial.

## B EXTERNAL EXAM-LABELED EVALUATION

As supplementary analysis, in order to assess model generalization beyond ECGQuest, we evaluated all models on ECG-related questions from standard medical tests, as detailed below.

Dataset Preparation: ECG-related questions were extracted from multi-option Q&A exams MedM-CQA (Pal et al., 2022) and MedQA (Jin et al., 2021). Identical ECG-topic filtering was applied to both datasets. For this, GPT-4o was used as a strict topic classifier to identify questions within the four ECG topics covered by the benchmark. This resulted in 1,054 questions from 187,005 in MedMCQA and 150 out of 12,723 in MedQA. Questions with negated or except-style format were excluded, as these could not be converted into a single unambiguous True/False statement (12 removed from MedMCQA and 3 from MedQA). The multi-option questions were next appended with the correct option text (from the exam key). Next, to achieve a balanced dataset, a false version of each question was created by appending the question with a randomly picked wrong response from the false options. For example, the question “P wave is absent in:" with options A) Atrial fibrillation, B) Atrial asystole, C) Ventricular fibrillation, and D) Ventricular tachycardia, with the correct answer A, was converted into the following two True and False question-answer pairs:

• True version: P wave is absent in Atrial fibrillation.

• False version: P wave is absent in Ventricular tachycardia.

where the false answer was picked randomly between the three incorrect options. This procedure produced 2,084 MedMCQA and 294 MedQA True/False questions. As the labels are derived from the official examination key, these sets provide a measure of ECG knowledge independent of the question generation process.

Zero-Shot and Fine-Tuned Results: Table 3 presents the results on ECG-related MedMCQA and MedQA Q&A (Section B). The ranking of models are largely similar to the ECGQuest ordering. GPT-5 leads on both (86.5% on MedMCQA, 93.2% on MedQA), followed by GPT-4o (83.7%, 89.8%) and then Gemma-4-31 B (79.7%, 83.0%) and Gemma-4-26 B (75.4%, 81.0%), which both outperform GPT-3.5 (70.4%, 67.3%). Both Gemma families also show a monotonic increase in accuracy with model size on both sets. Among the medical models MedGemma-4 B is the only competitive medical model (66.8%, 60.9%), while MedAlpaca-7 B, BioMistral-7 B, and MedAlpaca-13 B perform near chance on both sets (from 49.4 to 52.3%).

On the external datasets, the effect of fine-tuning was highly dependent on the base model. For the low-performing biased BioMistral-7 B, fine-tuning improved accuracy on both sets (from 51.3% to

Table 3: Zero-shot and fine-tuned performance on the external, exam-labeled MedMCQA (n=1,042) and MedQA (n=147) True/False sets. Acc: accuracy; F1: macro-averaged F1 score, P: precision, and R: recall; Sen: sensitivity, Spec: specificity. Across all models, the best value per metric column is in bold and the runner-up is underlined. The results can be compared and contrasted with the results in Table 1 of the main text.
<table><tr><td colspan="2"></td><td colspan="5">MedMCQA (Pal et al., 2022)</td><td colspan="5">MedQA (Jin et al., 2021)</td></tr><tr><td>Model</td><td></td><td>Params</td><td>Acc (%)</td><td>F1</td><td>P</td><td>R/Sen Spec</td><td>Acc (%)</td><td>F1</td><td>P</td><td>R/Sen</td><td>Spec</td></tr><tr><td colspan="16" rowspan="19" rowspan="19">General purpose (closed source) GPT-5</td></tr><tr><td>2T* 86.5</td><td>86.5 88.8</td><td>83.5</td><td>89.4</td><td>93.2</td><td>93.2</td><td>97.0</td><td>89.1</td></tr><tr><td>GPT-40</td><td>200 B*</td><td>83.7</td><td>83.7</td><td>81.6 86.9</td><td>80.4</td><td>89.8</td><td>89.8</td><td>89.3</td><td>90.5</td><td>97.3 89.1</td></tr><tr><td>GPT-3.5</td><td>175B*</td><td>70.4</td><td>70.3 68.1</td><td>76.8</td><td>64.1</td><td>67.3</td><td>67.1</td><td>64.7 76.2</td><td></td><td>58.5</td></tr><tr><td>General-purpose (open source)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mistral-Instruct-v0.3 Llama-3.1-Instruct</td><td>7B</td><td>62.7</td><td>61.4</td><td>59.3</td><td>81.0</td><td>44.3</td><td>55.4</td><td>53.6 53.9</td><td>75.5</td><td>35.4</td></tr><tr><td>Qwen2.5-Instruct</td><td>8B</td><td>71.1</td><td>71.1</td><td>71.2</td><td>70.8</td><td>71.4</td><td>66.7</td><td>66.6 68.1</td><td>62.6</td><td>70.7</td></tr><tr><td></td><td>14B</td><td>67.3</td><td>64.6</td><td>88.1</td><td>39.9</td><td>94.6</td><td>66.3 46.9</td><td>62.9 91.4</td><td>36.1</td><td>96.6</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>1.5B</td><td>51.8</td><td>51.7</td><td>51.6</td><td>57.4</td><td>46.3</td><td></td><td>46.8 46.9</td><td>46.9</td><td>46.9</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>7B 14B</td><td>51.7</td><td>51.7</td><td>51.9</td><td>48.4 55.1</td><td>56.5 70.4</td><td>56.5</td><td>56.5</td><td>56.5</td><td>56.5</td></tr><tr><td>DeepSeek-R1-Distill-Qwen Gemma-3</td><td>270M</td><td>70.8 52.0</td><td>70.8 51.8</td><td>69.3 52.4</td><td>74.9</td><td>66.8</td><td>70.2</td><td>70.9 48.9</td><td>70.4</td><td>70.4</td></tr><tr><td>Gemma-3</td><td>1B</td><td>54.1</td><td>52.6</td><td>53.0</td><td>44.7 72.0</td><td>59.3 36.3</td><td>49.0 50.7</td><td>48.9 50.6 50.6</td><td>44.2 55.8</td><td>53.7</td></tr><tr><td>Zrt -nng Gemma-3</td><td>4B</td><td>61.8</td><td>61.8</td><td>62.1</td><td>60.5</td><td>63.1</td><td>53.1</td><td>53.0 52.8</td><td>57.1</td><td>45.6 49.0</td></tr><tr><td>Gemma-3</td><td>12B</td><td>67.3</td><td>66.8</td><td>64.0</td><td>78.9</td><td>55.7</td><td>66.3</td><td>66.3 66.2</td><td>66.7</td><td>66.0</td></tr><tr><td>Gemma-3</td><td>27B</td><td>70.4</td><td>69.8</td><td>65.7</td><td>85.4</td><td>55.5</td><td>69.0</td><td>68.9 66.9</td><td>75.5</td><td>62.6</td></tr><tr><td>Gemma-4</td><td>2B</td><td>61.5</td><td>60.9</td><td>65.5</td><td>48.8</td><td>74.3</td><td>59.2</td><td>56.3 69.0</td><td>33.3</td><td></td></tr><tr><td>Gemma-4</td><td>4B</td><td>68.7</td><td>67.9</td><td>77.3</td><td>53.0</td><td>84.5</td><td>66.7</td><td>65.9 73.8</td><td>51.7</td><td>85.0</td></tr><tr><td>Gemma-4</td><td>12B</td><td>74.4</td><td>74.3</td><td>77.5</td><td>68.7</td><td>80.0</td><td>68.7</td><td>67.6</td><td></td><td>81.6</td></tr><tr><td>Gemma-4</td><td>26B</td><td>75.4</td><td>75.4</td><td>73.5</td><td>79.5</td><td>71.4</td><td></td><td>79.6</td><td>50.3</td><td>87.1</td></tr><tr><td>Gemma-4</td><td>31B</td><td>79.7</td><td>79.6</td><td>78.0</td><td></td><td></td><td>81.0</td><td>80.9 78.6</td><td>85.0</td><td>76.9</td></tr><tr><td>Medical-specialized (open source)</td><td></td><td></td><td></td><td></td><td>82.5</td><td>76.8</td><td>83.0</td><td>82.9 87.0</td><td>77.6</td><td>88.4</td></tr><tr><td>MedGemma</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>4B</td><td>66.8</td><td>66.2</td><td>63.3</td><td>79.8</td><td>53.7</td><td>60.9</td><td>58.3</td><td>57.3 85.7</td><td>36.1</td></tr><tr><td>MedAlpaca BioMistral</td><td>7B</td><td>49.4</td><td>33.0</td><td>49.7</td><td>98.7</td><td>0.1</td><td>49.7</td><td>33.2 49.8</td><td>99.3</td><td>0.1</td></tr><tr><td>MedAlpaca</td><td>7B 13B</td><td>51.3</td><td>36.7</td><td>50.7</td><td>99.4</td><td>3.3</td><td>50.7</td><td>35.4 50.3</td><td>99.3</td><td>2.0</td></tr><tr><td>Encoder baselines BERT</td><td></td><td>52.3</td><td>44.7</td><td>51.3</td><td>89.4</td><td>15.2</td><td>48.3</td><td>42.1 49.0</td><td>81.0</td><td>15.6</td></tr><tr><td>Finn-uned</td><td>110M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BiomedBERT</td><td>110M</td><td>49.4 50.6</td><td>48.3 50.6</td><td>49.6</td><td>64.1</td><td>34.7</td><td>49.3</td><td>47.2 49.5</td><td>69.4</td><td>29.3</td></tr><tr><td>Decoder LLMs</td><td></td><td></td><td></td><td>50.6</td><td>50.9</td><td>50.3</td><td>48.0</td><td>44.9 48.6</td><td>71.4</td><td>24.5</td></tr><tr><td>BioMistral</td><td>7B</td><td>60.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama-3.1-Instruct</td><td>8B</td><td></td><td>60.3</td><td>59.2</td><td>67.6</td><td>53.4</td><td>58.5</td><td>58.4 59.5</td><td>53.1</td><td>63.9</td></tr><tr><td></td><td></td><td>63.2</td><td>63.2</td><td>64.6</td><td>58.7</td><td>67.8</td><td>55.8</td><td>53.1 61.0</td><td>32.0</td><td>79.6</td></tr><tr><td>Gemma-4</td><td>12 B</td><td>72.6</td><td>72.3</td><td>69.2</td><td>81.2</td><td>63.9 72.8</td><td>72.8</td><td>71.6</td><td>75.5</td><td>70.1</td></tr><tr><td>DeepSeek-R1-Distill-Qwen</td><td>14 B</td><td>65.5</td><td>65.4</td><td>63.9</td><td>71.4</td><td>59.7</td><td>62.6</td><td>62.6 62.6</td><td>62.6</td><td>62.6</td></tr><tr><td>Qwen2.5-Instruct</td><td>14 B</td><td>73.0</td><td>72.9</td><td>76.0</td><td>67.2</td><td>78.8</td><td>64.9</td><td>64.9 64.9</td><td>65.0</td><td>64.8</td></tr><tr><td>Ensemble Model</td><td>55B**</td><td>67.2</td><td>67.1</td><td>65.5</td><td>72.6</td><td>61.8</td><td>65.6</td><td>65.5 68.0</td><td>59.2</td><td>72.1</td></tr></table>

\* Parameter counts for the GPT models are not officially disclosed by OpenAI; the reported values are unofficial estimates. \*\* 55 B is the total number of parameters across the five fine-tuned models used for ensemble voting.

60.5% on MedMCQA; from 50.7% to 58.5% on MedQA) and roughly doubled macro-F1 (from 36.7 to 60.3 and from 35.4 to 58.4) by correcting the True class bias. Qwen2.5-Instruct-14 B, which had an opposite bias, improved on MedMCQA in both accuracy (67.3% to 73.0%) and macro-F1 (64.6 to 72.9), while remaining close to its starting point on MedQA (66.3% to 64.9%). Gemma-4-12 B remained close to its base on MedMCQA (74.4% to 72.6%) and improved on MedQA (68.7% to 72.8%). For the already strong Llama-3.1-8 B and DeepSeek-R1-14 B models, fine-tuning reduced accuracy on both external sets (from 71.1% to 63.2% and from 70.8% to 65.5% on MedMCQA; from 66.7% to 55.8% and from 70.4% to 62.6% on MedQA), and the ensemble built from all five models shows the same pattern (67.2% and 65.6%). This can be related to the fact that MedMCQA and MedQA are long-standing public resources whose items have likely entered the pretraining corpora of the evaluated models, so strong zero-shot scores may partly reflect data leakage rather than ECG understanding.

## C QUESTION-GENERATION, VALIDATION, AND LABEL-BALANCINGPROMPTS

For reproducibility, we provide the three prompts used to construct the Q&A sets described in Section 3.1. Generation used two modes depending on page content: a text prompt (Prompt 1) for pages processed from extracted text, and a vision prompt (Prompt 2) for pages processed from a

rendered image. A separate label-balancing prompt (Prompt 3) generated the paired opposite label counterpart for each item (Section 3.1).

Prompt 1: Text-mode generation prompt. GPT-4o reads only the extracted page text.   
You are an expert medical exam question writer specializing in ECG   
interpretation. Generate clinical TRUE/FALSE questions for medical   
students, biomedical engineers, cardiologists, and electrophysiologists   
Generate questions only about the following allowed topics:   
1. ECG waveform features, including P, QRS, ST, T, and U waves; QT, PR,   
and RR intervals; morphology; and axis deviation.   
2. Clinical interpretation and diagnostic significance of ECG patterns,   
including ischemia, infarction, arrhythmias, conduction blocks,   
hypertrophy, pericarditis, electrolyte effects on the ECG, and criteria   
for distinguishing normal from abnormal findings.   
3. The physiological basis of ECG findings, including the conduction   
system, depolarization and repolarization, action potentials,   
automaticity, and refractory periods.   
4. ECG device technical specifications, including leads, electrode   
placement, gain, filters, paper speed, calibration, and artifacts.   
STRICT RULES:   
\* Never hallucinate. Every question must come directly from the page   
text.   
\* Never use general medical knowledge. Use only facts stated in the   
text.   
\*'source\_text'must contain the exact sentence or sentences, limited   
to one to three sentences, copied VERBATIM from the text.   
\* The question must be a DIRECT TRANSFORMATION of 'source\_text'. For a   
TRUE question, every word, number, and fact must be traceable to   
source\_text'. For a FALSE question, only the single altered detail may   
differ. Do not introduce any other outside knowledge, inference, or   
generalization.   
\* For a TRUE question, restate the key fact from 'source\_text'.   
\* For a FALSE question, change ONE detail, such as a number, threshold,   
or direction, and keep the rest from 'source\_text'.   
\* Discard any question if it contains content from outside'source\_text   
', except for the single detail intentionally changed in a FALSE   
question.   
\* Questions must be clinically meaningful and self-contained.   
\* Never reference images, figures, strips, diagrams, or tables.   
\* Balance the number of True and False answers.   
\* Return'[]' if the page contains no ECG-relevant content.   
EXAMPLE:   
source\_text: "The normal PR interval ranges from 120 to 200 ms."   
TRUE: "The normal PR interval ranges from 120 to 200 ms."   
FALSE: "The normal PR interval ranges from 120 to 400 ms."   
WRONG (outside knowledge): "A prolonged PR interval indicates first  
degree AV block."   
OUTPUT:   
Return a Python list of dictionaries without Markdown:

[{"page":"page N", "question":"<transformation>", "label":"True", "   
source\_text":"<verbatim sentence(s)>"}]   
SELF-CHECK:   
Discard the question if any of the following checks fail:   
\* For a TRUE question, 'source\_text'must support every word, number,   
and fact.   
\* For a FALSE question, all content must be supported by'source\_text',   
except for the single intentionally altered detail.   
\*'source\_text' is copied verbatim.   
\* The label can be determined correctly from 'source\_text'alone.   
\* The question is self-contained.   
Extract up to 30 questions.   
[Extracted page text appended here.]

## Prompt 2: Vision-mode generation prompt. GPT-4o reads a rendered page image and uses only the text written on it.

You are an expert medical exam question writer specializing in ECG   
interpretation. You are analyzing a PAGE IMAGE from an ECG textbook,   
manual, or clinical guide. Extract clinical knowledge as TRUE/FALSE   
exam questions.   
Generate questions only about the following allowed topics:   
1. ECG waveform features, including P, QRS, ST, T, and U waves; QT, PR,   
and RR intervals; morphology; and axis deviation.   
2. Clinical interpretation and diagnostic significance of ECG patterns,   
including ischemia, infarction, arrhythmias, conduction blocks,   
hypertrophy, pericarditis, electrolyte effects on the ECG, and criteria   
for distinguishing normal from abnormal findings.   
3. The physiological basis of ECG findings, including the conduction   
system, depolarization and repolarization, action potentials,   
automaticity, and refractory periods.   
4. ECG device technical specifications, including leads, electrode   
placement, gain, filters, paper speed, calibration, and artifacts.   
CRITICAL IMAGE RULES:   
\* Extract only clinical knowledge EXPLICITLY WRITTEN in the image,   
including captions, labels, tables, and annotations. Do NOT visually   
interpret waveforms or infer information from their shapes.   
\* Never use words or phrases that reference the image, including shown,   
figure, fig., strip, diagram, image, picture, above, below, indicated,   
arrow, highlighted, annotated, this ECG, this tracing, this rhythm,   
this waveform, this lead, this recording, as seen, or as depicted.   
\* BAD example: "In the rhythm strip shown, the RR interval is regular."   
GOOD example: "Regular RR intervals are a defining feature of normal   
sinus rhythm." Use this question only if the statement is explicitly   
written in a caption, label, table, or annotation.   
STRICT RULES:   
\* Never hallucinate. Use only text written in the image. Do not use   
general medical knowledge.

\* The question must be a DIRECT TRANSFORMATION of source\_text. For a   
TRUE question, every word, number, and fact must be traceable to   
source\_text. For a FALSE question, only the single altered detail may   
differ. Do not introduce any other outside knowledge, visual inference,   
or generalization.   
\* For a TRUE question, restate the written fact from source\_text. For a   
FALSE question, change ONE detail, such as a number, threshold, or   
direction, and keep the rest from source\_text.   
\* source\_text must contain the exact text visible in the image, such as   
a caption, table cell, label, or annotation.   
\* Discard any question if it contains content from outside source\_text,   
except for the single detail intentionally changed in a FALSE question   
OUTPUT:   
Return a Python list of dictionaries without Markdown:   
[{"page":"page N", "question":"<transformation; no image reference>",   
"label":"True", "source\_text":"<exact text visible in image>"}]   
SELF-CHECK:   
Discard the question if any of the following checks fail: \* Source\_text   
is written in the image.   
\* For a TRUE question, source\_text supports every word, number, and   
fact.   
\* The question contains no outside facts, visual inference, or   
generalization.   
\* The label can be determined correctly from source\_text alone.   
\* The question is self-contained and does not reference the image.   
Extract up to 30 questions. Return [] if nothing relevant.

During verification, each candidate item is checked against the generation constraints and discarded if it fails, with the reason recorded. The following text box shows two representative rejections: one whose cited span is not grounded in the page, and one whose question refers to a figure. Representative items removed during verification.

% Rejection 1 -- reason: source\_text\_not\_grounded   
Question: "If a patient with atrial tachycardia is stable,   
synchronised electrical cardioversion is recommended."   
Label: False   
source\_text: "If the patient is stable, use vagal manoeuvres   
and adenosine."   
Reason: The statement is not supported by the cited text,   
which recommends vagal manoeuvres and adenosine   
rather than cardioversion. The question is discarded.   
% Rejection 2 -- reason: image\_reference\_in\_question   
Question: "The QRS rhythm is irregular in Figure 4.7."   
Label: False   
source\_text: "QRS rhythm: regular."   
Reason: The question refers to Figure 4.7 and cannot be   
answered without viewing the figure. The question   
is discarded.

Prompt 3: Label-balancing prompt. Given a validated item, GPT-4o produces the opposite-label counterpart from the same source\_text.

You are a medical exam question editor specializing in ECG   
interpretation.   
TASK (original label = True -> produce a FALSE counterpart):   
\* Make minimal changes (such as a number, threshold, direction, or   
condition) or by paraphrasing, to make the statement clinically false.   
\* Do not introduce any concept that is not present in source\_text.   
TASK (label = False -> produce the TRUE counterpart):   
\* Restate the key fact from source\_text as a correct true statement.   
\* Derive the statement only from source\_text.   
STRICT RULES:   
\* Keep source\_text IDENTICAL to the input. Do not change a single   
word.   
\* Derive the counterpart ONLY from source\_text. Do not introduce any   
outside fact or concept.   
\* The counterpart must be a self-contained clinical statement.   
\* Never reference images, figures, strips, or diagrams.   
INPUT:   
original\_question : <question>   
original\_label : <True|False>   
source\_text : <source\_text>   
OUTPUT (Python dict, no markdown):   
{"question":"<counterpart>", "label":"<target>",   
"source\_text":"<identical to input>"}   
SELF-CHECK:   
Regenerate the counterpart if any of the following fail; if it still   
fails after retries, discard it:   
\* source\_text is identical to the input.   
\* The counterpart differs from the original question.   
\* The label is the correct inverse of the original label.   
\* For a FALSE counterpart, the statement is made clinically false   
either by changing one detail or by paraphrasing, while source\_text is   
unchanged and no outside concept is introduced.   
\* For a TRUE counterpart, the statement is a faithful, correct   
restatement of source\_text (no new or unsupported claim).   
\* The label can be determined correctly from source\_text alone.   
\* The counterpart introduces no concept outside source\_text.   
\* The counterpart is a self-contained clinical statement.   
\* The counterpart does not reference any image, figure, strip, or   
diagram.