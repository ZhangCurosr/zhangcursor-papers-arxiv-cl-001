# Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports

Parsa Bakhtiari<sup>∗</sup> Hassan Bashiri<sup>†</sup> Alireza Khalilipour<sup>‡</sup> Masoud Nasiripour<sup>§</sup> Moharram Challenger<sup>¶</sup>

## Abstract

Industrial technical reports contain high-value knowledge for maintenance, troubleshooting, and product engineering, but their heterogeneous structure (dense prose, specifications, and tables) makes them dificult to index and reason over with standard retrieval and questionanswering pipelines, and no public instruction-tuning or benchmark datasets are built from such documents. We address this gap with Industrial-Instruction, contributing (i) two open question-answering datasets built from real industrial technical reports and (ii) the end-to-end pipeline that produces them. Using 906 publicly available Panasonic documents (7,525 pages) as a case study, we apply layout-aware extraction to preserve text and tabular content, construct a semantic index for retrieval, and synthesize multiple-choice QA instances grounded in retrieved evidence. Each dataset explicitly models five realistic query–document relationships— irrelevant retrieval, single-/multi-document support, and single-/multi-document answer—so that models can be trained and evaluated on robustness to retrieval noise and on multi-step evidence integration. After automated filtering of an initial 23.9k generated samples, each dataset provides ≈13.6k QA pairs with associated source documents and a held-out benchmark split. Experiments with small open LLMs (<10B parameters) show that full fine-tuning on Industrial-Instruction substantially improves performance on the Panasonic benchmark, increasing Set-Match Accuracy from 28.5% to 42.0% and F1 from 46.6% to 63.5%, with consistent gains both with and without RAG.

We release the dataset in two parallel versions built by the same pipeline—one generated with the open-weight Qwen3-30B-A3B-Instruct model and one with the closed, API-based Claude-Opus-4.6 model—which additionally enables a direct comparison of open- versus frontier-model data generation. The Claude-Opus-4.6-generated dataset produces a substantially cleaner raw corpus and larger downstream fine-tuning gains, at roughly two orders of magnitude higher cost. Finally, MMLU evaluation before and after fine-tuning shows that models trained on the Claude-Opus-4.6-generated data retain essentially all of their general knowledge, in contrast to a small but measurable forgetting efect for models trained on the Qwen-generated data. Together, these datasets and the pipeline behind them ofer a practical, reproducible path toward scalable industrial benchmarks and training data derived from real-world documentation.

Keywords: Large Language Model; Fine Tuning; Industrial Question Answering; Retrieval-Augmented Generation; Synthetic Data Generation; Small Language Models.

## 1 Introduction

In today’s industrial world, documents and industrial reports play a key role as vital and specialized knowledge repositories. This knowledge has been gathered over decades and is considered a valuable resource. These documents include a wide range of data such as technical information, product catalogs, maintenance manuals, and specialized guides, all of which carry deep expertise in their respective fields.[5] In the "Industry 4.0 framework," such documentation forms the foundation for critical applications like condition-based maintenance and Failure Modes and Efects Analysis (FMEA). For example, ISO standard documents and reports compiled by professionals provide a mapping between potential failures of industrial assets and sensor parameters used to detect faults and anomalies[9].

Despite the high value of the information embedded in these reports, organizations face significant challenges in managing and utilizing them. The structure of these documents is often complex and heterogeneous, integrating quantitative data, text, charts, and diagrams seamlessly. Additionally, the knowledge contained in these documents is usually scattered and decentralized; related information about a particular concept or process is distributed across diferent sections and formats (such as descriptive text, tables, and charts). This makes it dificult to categorize and efectively retrieve information. Traditional information retrieval systems may be able to retrieve relevant data, but they often cannot extract complex patterns and relationships between them.

Tabular data constitutes a significant portion of the specialized knowledge found in industrial and technical documentation. In datasets related to information and communication technology, the text contained in tables accounts for about 18% of the total word count. For example, in a 6-gigabyte dataset, tables contained 178 million words[20]. In industry, these tables are compiled by experts and contain very useful information.

These tables are generally valuable because:

1. Tables often appear alongside text as they are vital for a comprehensive understanding of the content, such as technical specifications.

2. Knowledge in these structures directly supports decision-making, risk management, and eficiency enhancement.

3. By leveraging large language models for reasoning over data and tabular mapping, it becomes possible to simultaneously analyze both the structure of the table and the meaning of its content. These models can automatically extract conceptual relationships and complex dependencies between variables by combining information in the rows and columns; for example, identifying links between diferent equipment failure modes and patterns recorded in sensor data, or vice versa.

The emergence of large language models, generative AI, and agents has shifted the knowledge management paradigm towards automated extraction and reasoning. Although large language models have been trained on broad and general datasets, they often lack the specialized expertise required for highly technical industrial reports. To bridge this gap, researchers have proposed methods such as fine-tuning and retrieval-augmented generation [17] to inject specialized knowledge hidden in technical manuals and industry-specific data into the models. In this approach, documents are transformed into dynamic knowledge bases, thereby boosting operational efectiveness[5, 9]. Results show that the industrial domain is particularly challenging for language models and requires a form of specialized reasoning that general datasets do not encompass. Leading models such as GPT-4[15] and LLama[26] have achieved, on average, only 53.5% accuracy in question answering on the FailureSensorIQ benchmark with a single correct answer, highlighting the dificulty of industrial data. Open-weight models that were fine-tuned on general datasets such as HotpotQA[32] have achieved, on average, 29% accuracy on industrial benchmarks[9].

Overall, large language models, by demonstrating capabilities in both general and specialized fields like medicine, have the capacity for improvement and development in the field of industrial knowledge as well. However, one of the main challenges in this field is the shortage of suitable datasets for training and evaluating these models. The industrial gap refers to the disconnect between the broad, general capabilities of large language models and the highly specialized and precise requirements of the industry. For example, in the semiconductor manufacturing industry, this gap appears as follows[22]:

1. Lack of deep knowledge: Large language models are trained on massive volumes of textual data and focus on expanding their coverage rather than depth of knowledge.

2. Mismatch in communication style: Standard models are often designed for general conversations, which creates problems when specialists communicate with each other. In industrial settings, experts do not need general, simple, or summarized explanations; instead, they require precise, concise, and actionable guidance—similar to what an experienced engineer would provide to another experienced engineer. General-purpose models are usually unable to deliver the level of practical applicability and immediate usability needed in real-world production environments.

3. Ineficiency of standard evaluation criteria: Standard criteria are not suitable for industrial domains and there is a need for specialized datasets and benchmarks.

Despite significant advances in the application of large language models in general domains and some specialized fields, studies show that the volume of research conducted on leveraging real industrial documents—especially focusing on tabular data and the scattered technical knowledge in the documentation of industrial companies—remains limited. Moreover, the majority of recent studies have concentrated on very large language models such as ChatGPT and their counterparts, while smaller models with fewer than ten billion parameters have received less attention. Yet, such smaller models are of great importance in terms of computational cost and deployability in industria environments.

The present study contributes primarily two datasets, and secondarily the reproducible pipeline that builds them, to bridge the gap between small-scale language models and industrial knowledge. Concretely, our contributions are:

1. Two industrial QA datasets. Building on publicly available Panasonic Corporation documents, we design and release the “Panasonic Dataset” in two parallel versions—one generated with an open-weight model (Qwen3-30B-A3B-Instruct) and one with a frontier API model (Claude-Opus-4.6)—each pairing instruction-tuning data with a held-out benchmark split for training and evaluating small (<10B-parameter) models on real industrial content, including the tabular knowledge that dominates such reports.

2. An end-to-end construction pipeline. We adapt a multi-scenario retrieval-augmented generation paradigm to real industrial PDFs, combining layout-aware extraction, semantic indexing, and automated quality filtering; while the generation paradigm builds on prior work, applying it to heterogeneous industrial documentation and releasing the resulting artifacts is, to our knowledge, novel.

3. An open- versus frontier-model study. Producing both dataset versions with the same pipeline lets us compare open-weight and closed API models as data generators along raw data quality, downstream fine-tuning gains, cost, and general-knowledge retention (MMLU).

The raw source corpus contains 906 documents across 7,525 pages, manually curated and downloaded from Panasonic’s website.

Our code, dataset generation pipeline, and evaluation scripts are publicly available at https: //github.com/parssky/industrial-instruction. The two released datasets are archived on Hugging Face at https://huggingface.co/datasets/Parssky/industrial-instruction-dataset (DOI: 10.57967/hf/10098) [6].

## 2 State of the Art

One of the important studies in the field of industrial large language models is the FailureSensorIQ research[9]. It is a specialized evaluation system based on multiple-choice questions and answers. This framework is designed to assess the capabilities of large language models in the field of "Industry 4.0" and focuses on the complex relationships between failure modes and sensor data in various industrial assets. This dataset includes 8,296 questions selected by experts, representing 10 distinct assets, including electric motors, steam turbines, and power converters. These multiple-choice questions were generated in an automated pipeline using ISO document data. In this study, two main tasks are addressed:

1. Failure Mode to Sensor: Identifying the most relevant sensors for detecting the early signs of a specific failure.

2. Sensor to Failure Mode: Determining which failure mode is likely based on abnormal sensor data.

The results of this evaluation indicate a high level of dificulty in this domain , with the best models achieving an average accuracy of 53%. This dataset also showed that stress and variations in the questions can reduce model accuracy to as low as 12%. In this study, a feature selection tool based on large language models was also introduced. This tool uses large language models to suggest relevant sensor variables for predictive modeling . This could be a paradigm shift in "Industry 4.0" where large language models act as knowledge producers to assist engineers in feature engineering and root cause failure analysis. This research shows that prompts focused on reasoning improve the performance of medium-sized models, while agent-based systems relying on external knowledge —which use knowledge bases such as Wikipedia— often cannot provide the same level of reliability. Overall, this benchmark determines whether the internal logic of large language models can withstand the noise and complexities of a real factory without getting confused by irrelevant data or minor changes in how a problem is phrased[9]. Another challenge in industrial data is the management of mixed data. Data that includes both text and structured and semi-structured tables. There are common approaches for processing this type of data , such as flattening tables, which causes the loss of structure , or mapping text and tables to separate vector spaces , which disrupts their semantic link. D. Min et al [20] addresses this gap by integrating the table-totext conversion process to enhance a question-and-answer system based on large language models.

This approach transforms structured tables into coherent natural language descriptions to preserve informational links and structural integrity. In this work, four main categories of table-to-text conversion methods are compared in two main industrial AI paradigms: domain-specific fine-tuning and retrieval-augmented generation:

• Markdown template: a simple, syntax-based method that renders tables in Markdown format.

• Template-based Serialization: Uses hand-designed templates based on table features to generate descriptions.

• Traditional Pre-trained Language Model: This method uses pre-trained traditional models like BART to generate text from tables.

• Large Language Model-Based Methods: In this method, large language models such as Chat-GPT are used to describe the table.

The experimental results show that the choice of the table-to-text conversion method has a significant impact on system performance, with diferences in various metrics ranging from 2.8% to 16% [20]. In the domain-specific fine-tuning paradigm, methods based on large language models and traditional pre-trained models have been shown to outperform others, as they generate a higher frequency of specialized terminology and diverse verbs—elements that are critical for acquiring genuine industrial knowledge during the pre-training process.In contrast, in the retrieval-augmented generation paradigm, while methods based on large language models remain suitable for table processing, the Markdown format has also demonstrated high efectiveness[20]. The reason is that Markdown provides a semantic representation compatible with retrieval, allowing query vectors to embed and achieve better alignment with diferent parts of the documents. Researchers propose language model–based and Markdown–based strategies for industrial data. Consequently, for large language models to play an efective role in industry, they must adopt strategies that bridge the gap between technical data and natural language–based reasoning[20]. In a study by H. Femmer et al.[11] in the field of industrial requirements engineering and the evaluation of large language models, the QuRE dataset comprises 2,111 built-industry requirements gathered from real automotive projects at Mercedes-Benz, which were labeled over a decade through an industrial review process. Unlike other datasets, this dataset provides a gold standard for the quality of natural language requirements and specifically focuses on weak words —words that may indicate ambiguity, inconsistency, or incompleteness. This research is significant for the industry because it enables a comparative analysis between real-world requirements and requirements generated by language models . QuRE research indicates that requirements generated by models like GPT-4 often exhibit a lower degree of realism compared to real-world industrial data. Synthetic requirements have been found to be syntactically simpler, with shorter sentences, fewer paragraphs, and shallower syntactic trees compared to industrial samples. Although the text generated by large language models is more diverse, it lacks structural complexity and nuance. Real requirements are typically less readable compared to the simple, book-like examples generated by large language models. QuRE studies show that engineering prompt methods , such as content and role, also introduce only a limited increase in requirement complexity, but the outputs still have a significant gap in complexity compared to the real industrial dataset. This dataset provides a realistic foundation for evaluating automated quality assessment tools. Because it includes human labeling of a word’s content weakness, it turns the dataset into a rigorous test for evaluating the context-based reasoning of language models. Using QuRE, researchers can answer the fundamental question of whether large language models can be used ly to generate high-quality, realistic industrial requirements that reflect the depth and structure of documents used in sensitive and high-risk domains such as automotive, aviation, and healthcare.

## 3 Methodology

Considering the research conducted in the industrial sector, this study aims to fill the gap of insuficient datasets for training and evaluating large language models using an information retrievalaugmented generation approach. In this approach, we seek to create an industrial dataset in the form of multiple-choice question-and-answer pairs and their associated documents, enabling optimal model evaluation and training. To this end, we collected documents related to Panasonic that are publicly available in PDF format and ultimately succeeded in generating 13,557 question-and-answer pairs.

## 3.1 Information extraction methods

Extracting data and information from PDF files is considered a critical bottleneck, especially in retrieval-augmented generation tasks used by large language models. Unlike formats like HTML, the PDF format typically only stores character placement instructions, which makes it challenging to preserve document layout, paragraph coherence and continuity, and word order during information extraction. Existing methods can be divided into two categories: rule-based and learning-based , each ofering varying levels of capabilities such as Markdown output, image handling, and model compatibility.

• Rule-based methods for PDF processing: In these methods, algorithms are used to identify document elements. These tools are typically suitable for digital documents because they are computationally fast and easy to implement. Tools like PyMuPDF are widely used for basic text retrieval. Other tools, such as Camelot, also focus on tabular data . However, rulebased tools typically perform poorly when dealing with complex, nested tables. Libraries like PyPDF are also capable of extracting and managing images used in documents. Rule-based methods generally fail to preserve semantic structures, which can lead to text fragmentation and reduced performance in information retrieval[3, 4, 21, 24].

• Learning-based methods and vision-language models: In this approach, s leverage machine learning and deep neural network architectures such as transformers, especially for complex documents that include images and tables. The transformer-based Nougat [8] model is specifically designed to convert scientific documents into Markdown text. This model excels at analyzing mathematical equations and complex formulas. Its Markdown output is very useful for language models, as it preserves the semantic relationship between text and complex elements. The Table Transformer [25] model is an object detection model trained on large datasets like PubTables-1M. This model is more flexible in identifying types of tables, such as scientific and financial, compared to rule-based methods. Vision-language models like Dots.OCR and Florance-2 ofer an integrated paradigm where layout recognition, optical character recognition, and relationship understanding are learned simultaneously. These models are designed to handle spatial hierarchies and semantic granularity, and consequently provide a more robust solution for interpreting structured knowledge from complex visual documents[18, 30].

As shown in Table 1, for natural language model applications, using learning-oriented methods may be the best option, since they support Markdown, which is one of the bset ways to extrac tables[20]. Also, since technical documents used in industry include various types of images and tables, many of these documents are stored as scanned or unstructured formats and lack digital layers. Consequently, structured information extraction and maintaining the semantic relationship between text, tables, and images in these documents requires more advanced learning-based methods.

Table 1: Comparison between Rule-based and Learning-oriented approaches
<table><tr><td>Feature</td><td>Rule-based</td><td>Learning-oriented</td></tr><tr><td>Application in language models</td><td>Suitable for digital gov- Suitable for preserving ernment texts; may be content; can be con- fragmented for complex verted into a Markdown</td><td></td></tr><tr><td>Markdown sup-</td><td>documents. Limited; usually re- Native support; de-</td><td>document.</td></tr><tr><td>port Technical images</td><td>quires post-processing signed for outputs like after text extraction.</td><td>Markdown/LaTeX. Can extract images but Ability to recognize ob-</td></tr><tr><td></td><td>has no relation to the jects and link them ap- content.</td><td>propriately to the con- tent.</td></tr><tr><td>Reliability</td><td>Highly interpretable; Higher risk of hallucina- without producing hal- lucinatory errors.</td><td>tions in complex docu- ments.</td></tr></table>

## 3.2 Text extraction pipeline from PDF files

In this study, we used the Dots.OCR[18] model to extract text. This model achieved an OveralEdit score of 0.125 on the

English OmniDocBench[23] benchmark, which is significantly better than other tools such as MinerU with a score of 0.150 and Mathpix with a score of 0.191. The OveralEdit metric is a normalized error index based on the edit distance at the level of the entire document, simultaneously reflecting text recognition errors, reading order, and layout structure. This model also achieved a score of 0.177 on the XDocParse benchmark in this metric, compared to the multimodal Gemini-2 model5.- pro performed much better with a score of 0.251. Another competitor, MonkeyOCR-3B, performed worse in the XDocParse benchmark with a score of 0.483. On the other hand, the model’s 3 billion parameter size was more eficient in terms of accuracy-to-size ratio[18]. Like many vision-language models, the Dots.OCR model is also prone to phenomena such as hallucinations and low localization accuracy, especially when processing dense areas of documents. In other words, any inconsistency in sequence generation can lead to a distorted document layout. One of the main reasons for this issue is the use of downsampling methods in the image feature extraction stage, which removes some of the image’s spatial details. This removal of spatial information ultimately leads to the loss of precise alignment between document elements in the output sequence. For example, in documents that consisted entirely of tables, the model failed to correctly extract the structure of the cells and the relationships between them. Initially, we fed each page of the PDF documents to the model to have it directly generate that same page in Markdown. For each page, the following outputs are produced:

• Two Markdown documents (one including header/footer and the other without them).

• A JPG document, an example of which can be found at Figure 1. (Image of the page the model used for OCR)

• A structured JSON document containing an array of extracted regions, the type of each region, and the extracted text.

![](images/ba8698eca3feae47dd1cce8359feb993f110e1a3e738b94e559ed59e05d71103.jpg)  
Figure 1: An example output of a page from PDF documents by the Dots.OCR model; on the right, you can see the JPG document that has identified and categorized the various regions and sections, and the two images below are a sample created with the Markdown format, which has placed the images and various sections present on the page into this file.

Table 2: This presents the distribution of document sections that were not successfully processed, categorized by the type of structural element responsible for the processing error. The counts indicate the number of pages afected by each category.
<table><tr><td>Classification</td><td>Table</td><td>Text</td><td>Footnote</td><td>Index</td><td>Section</td><td>Page</td><td>Etc.</td></tr><tr><td>Count (pages)</td><td>236</td><td>14</td><td>6</td><td>4</td><td>2</td><td>1</td><td>28</td></tr></table>

These outputs, which can be viewed at and exemplified in Figure 1, allow for a qualitative review of the image-to-text extraction performance and the identification of missed regions. In the Markdown document, images are also stored in base64 format. Given the model’s weaknesses, pages from which text was not extracted must also be identified. In total, out of 906 documents with 7,525 pages, 291 pages had extraction issues, which on average represents 3.8

## 3.3 Knowledge Base Structuring and Preprocessing

In the data obtained from the previous stage, there are three types of data: images, text, and tables. The extracted tables are presented as text data in Markdown format, and the images are available as Base64-encoded data. However, the primary focus of this research is on text-based language models rather than vision-language models. Accordingly, during the dataset construction process, only text data and tables were retained, and all images were completely removed. The goal of this selection is to evaluate the ability of purely textual language models to understand and process technical documents without relying on visual information. The final dataset is organized in the standard Huggingface format. Additionally, during the initial preprocessing, pages that were empty, had very little content, or were duplicates were removed, which in total resulted in the deletion of 370 pages from the dataset.

The combination of the EmbeddingGemma[27] embedding model with FAISS[16] as the retrieval engine creates a highly eficient pipeline for semantic search and large-scale information retrieval. The EmbeddingGemma model is a model derived from Gemma3 [12] that has 300 million parameters and, despite its simplicity, performs very powerfully. It also supports dimension reduction from 768 to 128 without significant loss. This feature allows researchers to balance dimension size, storage space, and search speed. The FAISS library is also a specialized billion-scale search library that enables vector processing. This combination was chosen for its lightness and eficiency in retrieval. Figure 2 shows the distribution of tokens per page. Most documents have around 500 tokens, with longer documents reaching up to 2,500 tokens.

Retieval - Each Chunks  
![](images/75d6c54466f6ef7a9ca97ae4ee7e91e5a982c201d27a8c5e70c12ed3ec049055.jpg)  
Figure 2: Shows the distribution of tokens across document pages.

## 3.4 Dataset Creation

Now, with the information retrieval system in place, we aim to build a question-and-answer dataset to train the model on five real-world scenarios, which you can see in Figure 3. These scenarios illustrate the relationship between the user’s query q\* and the retrieved documents D\*:

1. Useless Document (r0): In this scenario, the retrieved document provides no help to the model in generating a response; it may even be completely irrelevant. The purpose of training this pattern is for when the model, upon seeing an irrelevant document, begins to hallucinate. Training this pattern helps reduce hallucination in the model.

2. Single Document Support (r1): In this case, a single document provides supporting information or clues related to the query, but does not contain an explicit and direct answer. The model must use these clues s to generate a comprehensive response.

3. Multi-Document Support (r2): In this scenario, multiple documents provide clues or supporting information without ofering an explicit answer to the query. This mode requires multi-step reasoning or the integration of scattered pieces of information to arrive at a final conclusion.

![](images/4a4dc88a8b18a353ab2629089af663c37c41770a97aca7d556bbdabbeeaedb61.jpg)  
Figure 3: Real-world information retrieval system scenarios and how they are used in an information retrieval-enhanced generation system.

4. Single-Document Answer (r3): In this case, a single document provides the complete answer to the question directly. This is the simplest information retrieval scenario, where the answer can be extracted entirely from one source.

5. Multi-document Answer (r4): In this scenario, multiple documents collectively provide the question’s explicit answer. Similar to the multi-document support case, this mode also requires multi-step reasoning or the integration of information across several documents to construct the final answer.

To generate a dataset based on these scenarios, we can use multimodal large language models. This approach ofers several advantages that lead to improved small models. Multimodal large language models have a strong ability to produce accurate, high-quality, and instruction-aligned responses, which can serve as a reliable source for training other models. Leveraging its ability to understand these same external source documents, a large language model can accurately and purposefully generate retrieval-aware relationships between a query and a document, for example, by recognizing that a document is intentionally useless [29]. We also use an instruction simulation method to avoid repetitive instructions. In the prompt we give the model to generate the question and answer, we use similar instructions. This method ensures that the resources the model references are not repetitive and include various forms of instructions.

Now, selecting a model for the instruction-following task is important. Large, multi-purpose language models like GPT-4o can be used, but alternatives can also be found. In the context of synthetic dataset generation, the Qwen3 model, particularly the Qwen3-30B-A3B-Instruct, serves as a

Listing 1: The prompt sample used for scenario r0, which is placed in the first part of the retrieved documents, followed by a random instruction in the next section to ensure the generated samples are appropriate.

<Documents>   
{doc}   
</Documents>   
Your task is to generate an English question q\* and a corresponding response a\* based on the provided <   
Documents>. Please note that the question q\* can take various forms, not limited to questions with a   
question mark, but also including statements, instructions, and other formats. You need to follow the   
requirements below to generate the q\* and a\* (RAG Paradigms):   
1. q\* should be related to the <Documents>, but the <Documents> can not provide any useful information for   
answering q\*.   
2. a\* should be able to answer q\*, ensuring that the response a\* is accurate, detailed, and comprehensive.   
3. q\* must be a standalone question. Strictly avoid using phrases that reference the source material, such as   
"Based on the provided context," "According to the documents," "In the text," or similar meta  
references.   
Additionally, to ensure diversity, richness, and high quality in the question q\* you generate, we will   
randomly provide a question for you to emulate. In other words, while satisfying the requirements above,   
make q\* similar in task requirement and expression to the <Simulated Instruction> below:   
<Simulated Instruction>   
{simulated\_instruction}   
</Simulated Instruction>   
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q\*   
MUST ALSO include exactly 5 options labeled A-E, and a\* MUST include the correct option(s) in the same   
JSON list format used in the simulated instruction. Always generate 5 options, even if the original   
document does not contain any option-like content.   
Please directly generate the question-answer-options (q\*, a\*, options\*) following all the rules above in the   
format of {"q\*": ..., "a\*": ..., "options\*": ...}, so for a\* select A-E correct option.   
Ensure the quality of the generated (q\*, a\*, options\*).

lightweight alternative to GPT-4o due to its combinatorial reasoning capabilities, high instructionfollowing accuracy, and extensive multilingual coverage. The following points can be noted when comparing these two models[31]:

1. Superior adherence to instructions and alignment: dataset generation requires strict adherence to formatting and complex logic. Benchmarks show that Qwen3-30B-A3B-Instruct scores 84.7 in IFEval (instruction following) achieved a score of 84.7, which is higher than the 83.9 score for GPT-4o. Additionally, this model reached a score of 69 on Arena-Hard-V2 (human-aligned), a benchmark for instruction-tuned models, which is better than the 61.9 recorded for GPT-4o. Also, on the WritingBench benchmark (Creative Writing) the Qwen model has reached a score of 85.5, which is higher than the 75.5 for GPT-4o.

2. Advanced Reasoning Through Hybrid Modes: Unlike traditional large models, Qwen3 introduces a hybrid reasoning framework that switches between "Thinking" and "Non-Thinking" modes. The "thinking" mode is specifically optimized for multi-step logical reasoning, mathematics, and programming, while the "non-thinking" mode is suitable for fast, general, and creative tasks.

Given its superior performance in evaluations and its open-source nature, the Qwen3-30B-A3B-Instruct model was used to generate the dataset. As shown in Figure 3 the various sections are separated by markers to help the model efectively identify the documents and random instructions.

To simulate multiple-choice questions that also allow for accuracy measurement, we used the SensorIQ dataset instructions from IBM. This dataset includes 5,334 multiple-choice questions and answers that are used for each of the five scenarios. As shown in the Figure 1 , the dataset generation pipeline first selects one of the SensorIQ datasets, then, based on the prompt, retrieves

![](images/dc59c0d6696526b7a0f0f94f9e633bd6f370c82f0b25c2e9c66a4cf9d3dfafa1.jpg)  
Figure 4: The pipeline for generating the Panasonic document dataset.

Listing 2: A sample generated by Qwen that includes the question, answer, options, and related documents.

```jsonl
{
"q*": "When selecting a current sensing resistor for use in a high-reliability automotive application,
which performance characteristic should be prioritized to ensure long-term stability under thermal
cycling conditions?",
"a*": ["A"],
"options*": ["A Thermal shock resistance",
"B Solderability",
"C Frequency characteristics",
"D Moisture resistance",
"E Overload capability"],
"documents": ["Current Sensing Resistors, Metal Plate Type\n\n## Performance (AEC-Q200) \n\n### ERJMS4S/
ERJMS4H\n\n ... "]
}
```

three documents from the knowledge base and feeds them to the large Qwen model to generate the corresponding question and answer for the scenario.

Thus, a total of 23,910 Q&As are generated. During sample generation, some samples were dropped by Qwen because, due to the probabilistic nature of large language models’ outputs, they did not follow the specified JSON structure. In Figure 4, this structure is in JSON format so that each section can be examined with code.

## 3.4.1 Question and Answer Preprocessing

Filter for detecting valid answer options. To enhance the quality of the input data and ensure the validity of the multiple-choice question structure, a rule-based preprocessing has been employed. This preprocessing is tasked with selecting samples whose options have a complete and meaningful structure. The goal is to determine whether the options truly consist of an identifier and an explanatory text, or are merely a collection of short, incomplete identifiers, such as single digits. In Listing 2, one can see examples of correct and incorrect options. First, it is checked that the number of options is appropriate; then each option is analyzed individually. Options that consist of only a short identifier such as A or 1 and lack explanatory text are considered incomplete options.

Listing 3: Correct and incorrect examples of options in samples generated during the dataset construction process.  
```markdown
# Correct
A. text
B) description
l: explanation
# Incorrect
A
l relay model
```

Listing 4: In this sample Q&A, the full descriptions of the options are provided immediately after the question, and in the option template only the letters are displayed without any explanation, which does not follow the original format.

```jsonl
{
"q*": "Please select the correct option(s) from the following options given the question: Which sensor
type is most suitable for detecting motionless objects and providing information about the direction
of movement, in addition to measuring temperature? Options: A PIR (Through Hole) B Grid EYE High Gain
Type C Grid EYE Low Gain Type D EKM-B E AMN",
"a*": ["B", "C"],
"options*": ["A", "B", "C", "D", "E"],
"documents": ["..."]
}
```

Options that begin with an identifier and end with explanatory text, separated by punctuation such as a period, parenthesis, or colon, are considered complete options. An example of a question with no explanation in the options, where only the option letter is written, can be seen in Figure 3.

Filter for identifying options embedded in the question text. Now, in addition to examining the options’ structure in the template, another filter is used to detect answers hidden in the question text. This filter is designed to identify instances where options are directly inserted into the question text. For this filter as well, rule-based methods are used, and the specifies whether a minimum number of answer options are structurally present in the question text, an example of which can be seen in Figure 3.

Finally, we add the list of first-filter samples to the options set and merge both the first and second filter lists together. This set includes $\mathrm { q } ^ { * }$ , which contains the question along with its options. We remove from the dataset the remaining samples that did not fall into either filter. In Listing 4, you see an example of this category where, in practice, there are no options and only letters are provided as choices. In Table 3, the number of samples is specified; with this preprocessing method, we ensure that high-quality samples are retained, as they can have a significant impact on the training and evaluation process.

Filter for checking the option response. Also, in each sample, there is a section titled $\mathrm { a } ^ { \ast }$ which contains the answer to that question. Sometimes the model used a keyword other than $\mathrm { a } ^ { \ast }$ for construction, or it used a complex form that depended on the technical identifier . This section must also be a list of characters; in some samples, this list is not in that format, and a diferent type of rule-based filter was used for this section to ensure the responses are in a standard format.

After applying the two initial filters, the option-response filter is also applied, and finally the distribution of the dataset in is shown in Figure 5. There are also no significant discrepancies in this distribution, and during the model training process it can almost learn the sample proportions.

Table 3: Total number of samples after preprocessing in each scenario.
<table><tr><td>Filter/Scenario</td><td>R0</td><td>R1</td><td>R2</td><td>R3</td><td>R4</td></tr><tr><td>Filter to detect healthy response options</td><td>2472</td><td>2526</td><td>2406</td><td>2476</td><td>2379</td></tr><tr><td>Filter for identifying options embedded in the question text</td><td>427</td><td>302</td><td>410</td><td>315</td><td>365</td></tr><tr><td>Rest (deleted)</td><td>1962</td><td>1961</td><td>1994</td><td>2015</td><td>2052</td></tr><tr><td>Total remaining</td><td>2899</td><td>2828</td><td>2816</td><td>2791</td><td>2744</td></tr></table>

Number of Samples per Split  
![](images/1eb5ccbbd7bae6bb148a22a2671e3eca8131a41dccd2fa904839d9cf79549760.jpg)  
Figure 5: This shows the final set of samples after the filtering process in each scenario.

At the outset of this study, approximately 24,000 samples were generated, which, after a stringent preprocessing step, were reduced to about 13,500 samples. Given that the behavior and output of language models are a direct reflection of their training data, and that the quality of the dataset plays a decisive role in the model’s final performance, we adopted a rigorous approach to data cleansing. Initially, to evaluate and benchmark the models, we randomly selected 200 samples from each defined scenario, the details of which are specified in Table 4. The remaining samples were then used as the training dataset, with a portion of these data also allocated to the test or validation set during the training process.

## 3.5 Dataset Regeneration with a Frontier API Model

To test whether the quality of the generator model materially afects the quality of the resulting Industrial-Instruction dataset, we repeated the entire generation pipeline described above using Claude-Opus-4.6 in place of Qwen3-30B-A3B-Instruct, keeping the retrieval pipeline, prompts, and the five scenario definitions (r0–r4) unchanged. This run produced 26,395 raw Q&A pairs, of which only 143 samples (0.5%) were removed by the same preprocessing filters described above, compared with roughly 10,353 samples (43%) removed from the Qwen3-30B-A3B-Instruct output (Table 3). Figure 6 shows the resulting per-scenario distribution, which remains balanced across the five scenarios (5,215–5,273 samples each). Table 5 reports the resulting training and benchmark splits.

Table 4: Data split for training and testing.
<table><tr><td>Sample Type</td><td>Sample Size</td></tr><tr><td>Training</td><td>12,557</td></tr><tr><td>Test (Benchmark)</td><td>1,000</td></tr></table>

Number of Samples per Split  
![](images/9d4ae89a8f0e1dd31f6e833de56f82d8d21635c5b0a1bff499d48712fc4e8b91.jpg)  
Figure 6: Per-scenario sample counts for the Panasonic dataset generated with Claude-Opus-4.6, after filtering.

The substantially lower filtering rate suggests that Claude-Opus-4.6 followed the required JSON schema and option-formatting constraints more reliably than the open-weight Qwen3-30B-A3B-Instruct model, consistent with its much larger parameter count and correspondingly stronger instruction-following ability. This came at a monetary cost, however: generating the Claude-Opus-4.6 dataset cost approximately \$330 in API usage, versus roughly \$3.2 in local compute for the Qwen3-30B-A3B-Instruct run (Table 15, Section 6).

## 4 Experimental Setup

To examine the extracted dataset, a training and evaluation framework has been set up. First, the models are evaluated on the test portion of the data to obtain the required metrics. Then, the models are trained on the other portion to investigate the dataset’s impact on the model. Additionally, in

Table 5: Data split for training and benchmarking on the Claude-Opus-4.6-generated dataset.
<table><tr><td>Sample Type</td><td>Sample Size</td></tr><tr><td>Training</td><td>25,252</td></tr><tr><td>Test (Benchmark)</td><td>1,000</td></tr></table>

this section, the models are also evaluated on the FailureSensorIQ dataset, and its impact on this data is also determined.

## 4.1 Benchmarking of base models on the test dataset

To accurately evaluate model performance, selecting an evaluation metric is essential. This metric can vary depending on the task and objective. In the question-answering task, two metrics, Exact-Match and Accuracy, are commonly used.

• Exact-Match criterion.A binary criterion that checks whether the predicted string is exactly the same as the reference string, character-by-character. This criterion is very sensitive to reordering and formatting. For example, if the correct answer is [A, B] is the correct answer and the model produces [B, A], the metric assigns a score of zero, even though the answer is logically correct.

• Accuracy metric. This metric is typically used for classification and calculates the fraction of samples where the predicted class is the same as the reference class. When applied to lists of string representations, Accuracy behaves similarly to Exact-Match. In this case, the entire string [A, B] is considered as a class label. This metric may also mark a correct sample as incorrect due to diferences in order.

Given the limitations of the above metrics in assessing responses that are inherently set-based and order-independent, using composite and set-based metrics provide a more accurate evaluation of the model’s performance. Accordingly, the following metrics are used in this study to evaluate the models.

• Set-Match-Accuracy.A domain-specific metric that converts string representations into unordered sets before comparison. This metric checks for set equality and is therefore robust to element order and formatting, such as whitespace. This metric aligns the evaluation with the problem’s logic and ensures that the model is judged on its ability to identify the correct option, not on a specific order.

• F1 Score. This standard metric for multi-label classification balances accuracy and recall. The accuracy metric measures how many of the options the model selected were correct, while the recall metric measures how many of the correct options available were selected by the model.

• Jaccard similarity metric. This metric measures the overlap between the predicted set and the reference set. Its interpretation is intuitive: what percentage of the total unique options involved were correct?

Thus, the Set-Match-Accuracy, F1-Score, and Jaccard similarity metrics were chosen as the final evaluation criteria to assess the models’ performance based on the logical correctness of the answers rather than on any specific ordering or formatting.

Table 6: Performance of base models on the Panasonic industrial dataset. Qwen-4B-Instruct, Phi-3-mini-4k-Instruct [1], RAG-Instruct-Llama3-8B
<table><tr><td>Model</td><td>Set-Match Acc.</td><td>F1-Score</td><td>Jaccard</td></tr><tr><td>Qwen-4B-Instruct</td><td>28.5%</td><td>46.65%</td><td>41.62%</td></tr><tr><td>Phi-3-mini-4k-Instruct</td><td>17.5%</td><td>31.27%</td><td>28%</td></tr><tr><td>RAG-Instruct-Llama3-8B</td><td>0.70%</td><td>0.90%</td><td>0.86%</td></tr></table>

The comparisons in Section 5.2 additionally evaluate models on the FailureSensorIQ benchmark, using four metrics reported by that benchmark’s own evaluation harness: AccOrgIBM, accuracy on FailureSensorIQ’s original, unperturbed questions; AccPerIBM, accuracy on the perturbed variant of the same questions, used to test robustness to rephrasing [9]; and F1-Macro/F1-Micro, the macro- and micro-averaged F1 scores across FailureSensorIQ’s question categories. On the Panasonic benchmark we continue to report Set-Match-Accuracy, F1-Score, and Jaccard Similarity as defined above; in the comparison figures these are labeled AccPana, F1Score, and JacSim respectively for brevity.

Also, language models are sensitive to the length of the generated outputs y, in that if you ask the model for only the correct option, it might select one option, but in an explanatory mode, if the model writes an explanation and specifies the reason for the selection, it might choose a diferent option. Also, in the desired response, only the option matters, and the preceding explanations are not relevant to the evaluation metric. Therefore, a balancing layer is needed to extract the model’s final answer. To address this issue, a normalization layer is applied to the model’s output. The goal of this layer is to extract the final answer independently of the model’s explanatory details. In this approach, the model is asked to provide the final answer in the form of a standard JSON block. Then, using regular expressions, this block is extracted and selected as the reference answer. In the normalization stage, common inconsistencies in the output of language models, such as the use of single quotation marks instead of double quotation marks, are corrected, and the JSON structure is converted into a processable format. This process ensures that the evaluation metric focuses solely on the model’s final answer and is not influenced by explanatory explanations or changes in the output’s format.

The FailureSensorIQ study addressed large models[9], whereas this research focuses on small language models. The purpose of this selection is to examine models that require fewer hardware resources, making them practical for small companies and research teams with limited computational resources. In this regard, the proposed models and their accuracy have been evaluated on the Panasonic industrial dataset, which can be viewed in Table 6.

The base model also considered is Qwen-4B-Instruct, and we will proceed to discuss the training and evaluation process for this model.

## 4.2 General-Knowledge Evaluation Protocol

A well-known risk of fine-tuning a pre-trained large language model on a narrow, domain-specific dataset is catastrophic forgetting, in which the model loses part of the general-purpose knowledge acquired during pre-training while adapting to the new data distribution [19]. The severity of forgetting depends on several factors, including the fine-tuning method, the nature of the training data, and the training duration; prior work shows that instruction-based fine-tuning tends to induce more forgetting than continued pre-training, that forgetting worsens as training progresses, and that parameter-eficient methods such as LoRA, while typically weaker than full fine-tuning on the target task, tend to better preserve out-of-domain performance [7]. Because the degree of forgetting cannot be predicted from the training method alone, it must be measured empirically with standard benchmarks; evaluating general knowledge before and after fine-tuning is a widely adopted protocol in the large-language-model literature [19].

Table 7: Evaluation results under diferent LoRA fine-tuning configurations.
<table><tr><td>Config</td><td>F1-Score</td><td>Jaccard Similarity</td><td>Set-Match Acc.</td></tr><tr><td>Original</td><td>46.57%</td><td>41.57%</td><td>28.50%</td></tr><tr><td>R16-A16</td><td>46.71%</td><td>41.66%</td><td>28.50%</td></tr><tr><td>R32-A16</td><td>46.89%</td><td>41.86%</td><td>28.70%</td></tr><tr><td>R8-A16</td><td>46.53%</td><td>41.54%</td><td>28.50%</td></tr><tr><td>R64-A16</td><td>46.27%</td><td>41.27%</td><td>28.20%</td></tr></table>

We use the Massive Multitask Language Understanding (MMLU) benchmark [13] to evaluate general knowledge and reasoning ability. MMLU spans 57 subjects across STEM, the humanities, the social sciences, and specialized domains such as law and medicine, with dificulty ranging from introductory to expert level, and is designed to test whether a model has acquired generalizable world knowledge during pre-training. For this reason, MMLU is one of the most commonly used measures of general-knowledge retention in the catastrophic-forgetting literature. In this study we measure accuracy on the full MMLU test set (57 subjects grouped into 4 top-level categories, 14,042 questions) both before and after fine-tuning on the Panasonic industrial dataset, to determine whether fine-tuning degrades the model’s ability to answer general-domain questions.

## 4.3 Training the Models on the Qwen-Generated Dataset

Now, we prepare the base model on the training dataset and measure the impact of that data on the model. Fine-tuning is the process of adapting a pre-trained large language model to a specific task or domain by continuing to train the model on a smaller, domain-specific dataset. Main fine-tuning methods include , in which all weights are updated, and , which only modifies a small portion of the weights. Among PEFT methods,LoRA is the most common approach, approximating a full weight update by applying low-rank matrix multiplications. Despite the high eficiency of the LoRA method, it experiences a performance drop compared to full fine-tuning, especially when the target task requires high levels of logical reasoning or learning entirely new skills[2, 10, 14, 28].

The results of the LoRA method do not show significant changes in the model. Four experiments were conducted to improve the Qwen3-4B-Instruct model, and the evaluation results on the Panasonic dataset can be found in Table 7.

Also, for a more detailed examination, the error rate chart for is available in Table 8 . In this table, the charts are nearly identical, and changing the LoRA-related parameters had no impact on the error rate and, consequently, did not improve the model.

Therefore, one can say that this method was not efective for training the Qwen model on the dataset. Of course, LoRA fine-tuning is for aligning the model’s behavior and cannot afect the model’s knowledge. For this reason, we pursued full fine-tuning and trained all of the model’s parameters. The results are available at and in Table10 . With full fine-tuning, the Set-Match Accuracy metric increased from 28.5% to 42%. The F1-Score and Jaccard-Similarity metrics also reached 63.48% and 57.95%, respectively, which are better than the baseline results in Table 6.

Table 8: Loss reduction under diferent LoRA rank configurations.
<table><tr><td>R Alpha</td><td colspan="6">Loss Changes</td></tr><tr><td>8</td><td colspan="6">Evaluation Loss</td></tr><tr><td></td><td>16 1.4 12</td><td></td><td></td><td></td><td></td><td>R=16,4=16 K=RA=2O R=32.4=16 R=54,4=16</td></tr><tr><td>16</td><td>16 14</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>32</td><td>16</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>64</td><td>16 anβax, ea]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>3</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Performance comparison before and after full fine-tuning, with and without RAG, on the Claude-Opus-4.6-generated Panasonic dataset.

<table><tr><td>Configuration</td><td>F1-Score</td><td>Jaccard Similarity</td><td>Set-Match Acc.</td></tr><tr><td>Original with RAG</td><td>58.55%</td><td>54.00%</td><td>40.90%</td></tr><tr><td>Fully Fine-Tuned with RAG</td><td>72.66%</td><td>68.88%</td><td>56.40%</td></tr><tr><td>Original</td><td>59.24%</td><td>54.71%</td><td>41.60%</td></tr><tr><td>Fully Fine-Tuned</td><td>72.72%</td><td>68.93%</td><td>56.40%</td></tr></table>

## 4.4 Training the Models on the Claude-Generated Dataset

Following the same full-fine-tuning protocol described above — identical hyperparameters, optimizer settings, and training duration — we fine-tuned Qwen3-4B-Instruct on the Claude-Opus-4.6-generated Panasonic dataset (Table 5) to allow a controlled comparison between the two data sources. Table 9 reports the results on the corresponding held-out test split.

Fine-tuning on the Claude-Opus-4.6-generated data produces larger absolute gains than finetuning on the Qwen3-30B-A3B-Instruct-generated data (Table 10): Set-Match Accuracy improves by roughly 15 percentage points with RAG (40.9%→56.4%), versus about 13.5 points for the Qwengenerated dataset (28.5%→42.0%). Note, however, that the two evaluations use diferent held-out test splits (each generated by its respective model), so the “Original” baseline scores also difer between Table 10 and Table 9 (28.5% vs. 40.9–41.6%); the two improvement magnitudes are therefore indicative rather than strictly controlled, and should not be over-interpreted as a head-tohead comparison.

## 5 Results and Discussion

This study uses two benchmarks to evaluate model performance: FailureSensorIQ and the Panasonic industrial dataset introduced above. In all evaluation scenarios, the retrieval component draws from the Panasonic knowledge base described in Section 3.5. We first examine the efect of fine-tuning on each model’s general knowledge (Section 5.1), then compare the base Qwen3-4B-Instruct model, its Panasonic-fine-tuned variants, and RAG-Instruct-Llama3-8B across both benchmarks for each of the two dataset-generation approaches used in this study (Section 5.2).

## 5.1 General-Knowledge Evaluation Results

Table 11 reports overall MMLU accuracy for the base model (Qwen3-4B-Instruct-2507, no finetuning), the model fine-tuned on the Qwen3-30B-A3B-Instruct-generated dataset (finetuned\_v1\_qwen), and the model fine-tuned on the Claude-Opus-4.6-generated dataset (finetuned\_v2\_claude).

finetuned\_v2\_claude retains essentially all of the base model’s general knowledge (a 0.05-point drop), whereas finetuned\_v1\_qwen loses 1.26 points overall. Neither drop is severe on its own, but a clear pattern emerges once results are broken down by top-level category (Table 12).

Table 10: Performance comparison before and after full fine-tuning, with and without RAG, on the Panasonic dataset.
<table><tr><td>Configuration</td><td>F1-Score</td><td>Jaccard Similarity</td><td>Set-Match Acc.</td></tr><tr><td>Original with RAG</td><td>46.57%</td><td>41.57%</td><td>28.50%</td></tr><tr><td>Fully Fine-Tuned with RAG</td><td>63.48%</td><td>57.95%</td><td>42.00%</td></tr><tr><td>Original</td><td>46.63%</td><td>41.61%</td><td>28.50%</td></tr><tr><td>Fully Fine-Tuned</td><td>63.14%</td><td>57.60%</td><td>41.70%</td></tr></table>

Table 11: MMLU accuracy of the base model and the two fine-tuned variants.
<table><tr><td>Model</td><td>MMLU Overall</td></tr><tr><td>Original (base)</td><td>72.13%</td></tr><tr><td>finetuned_v1_qwen</td><td>70.87%</td></tr><tr><td>finetuned_v2_claude</td><td>72.08%</td></tr></table>

finetuned\_v2\_claude performs on par with, or marginally better than, the base model in every category, with only a small dip in the “Other” category — one of the clearest cases of negligible catastrophic forgetting observed in this study. finetuned\_v1\_qwen, by contrast, shows forgetting concentrated in the Humanities category, driven in particular by a roughly 10-point drop on moralreasoning-related subjects (see Appendix C for the subject-level breakdown). finetuned\_v2\_claude additionally shows small but consistent improvements over the base model on several quantitativeand legal-reasoning subjects, including global\_facts, college\_mathematics, machine\_learning, security\_studies, and international\_law. This indicates that fine-tuning on the higher-quality Claude-Opus-4.6-generated dataset can, in this case, improve certain general-reasoning capabilities without sacrificing others — a much more favorable trade-of than the one observed for finetuned\_v1\_qwen.

## 5.2 Comparison on Industrial Benchmarks

In this section we compare each fine-tuned model against both the FailureSensorIQ and Panasonic benchmarks. In all cases, the retrieval component is implemented using the Panasonic knowledge base described in Section 3.5.

## 5.2.1 Qwen-Generated Dataset

Figure 7 compares Qwen3-4B-Instruct (base), Qwen3-4B-Instruct-Panasonic-Qwen (fine-tuned on the Qwen3-30B-A3B-Instruct-generated dataset), and RAG-Instruct-Llama3-8B across four FailureSensorIQ metrics and three Panasonic metrics.

As with the Claude-generated dataset, fine-tuning on the Qwen3-30B-A3B-Instruct-generated dataset substantially improves all three Panasonic metrics (Set-Match Accuracy 28%→42%, consistent with Table 10), while RAG-Instruct-Llama3-8B remains essentially unusable on Panasonic (1% across all three metrics), reproducing the severe drop already reported in Table 6. On Failure-SensorIQ, however, the Qwen-generated dataset produces the opposite trade-of from the Claudegenerated one: fine-tuning on Qwen data reduces accuracy on the original, unperturbed questions (AccOrgIBM, 34%→27%) while improving both macro- and micro-averaged F1 (F1-Macro 40%→43%; F1-Micro 66%→74%) — a mirror image of the Claude-fine-tuned model, which gained

Table 12: MMLU accuracy by top-level category for the base model and the two fine-tuned variants.
<table><tr><td>Category</td><td>Original</td><td>fnetuned 1 v1 qwen</td><td>finetuned v2 claude</td></tr><tr><td>Humanities</td><td>63.66%</td><td>61.13%</td><td>63.85%</td></tr><tr><td>Social Sciences</td><td>81.38%</td><td>80.44%</td><td>81.28%</td></tr><tr><td>STEM</td><td>72.53%</td><td>72.41%</td><td>72.44%</td></tr><tr><td>Other</td><td>75.41%</td><td>74.57%</td><td>75.09%</td></tr></table>

Qwen-ORGINAL vs Qwen-Pana-Qwen vs RAG-Instruct

![](images/a290f0259f26b9a2202436382f68af005a0162759e60cb8fc78d982b497a7909.jpg)  
Figure 7: A side-by-side comparison of FailureSensorIQ and Panasonic metrics for Qwen3-4B-Instruct, Qwen3-4B-Instruct-Panasonic-Qwen, and RAG-Instruct-Llama3-8B.

AccOrgIBM but lost F1-Macro/F1-Micro (Section 5.2.2). As with the Claude comparison, neither Qwen variant answers any perturbed FailureSensorIQ question correctly (AccPerIBM = 0%), while RAG-Instruct-Llama3-8B is the strongest model on that metric (33%) as well as on F1-Macro. This contrast between the two fine-tunes’ efect on FailureSensorIQ is itself a notable finding: the choice of generator model does not just change the magnitude of downstream gains, but can change which FailureSensorIQ metrics improve versus regress.

## 5.2.2 Claude-Generated Dataset

Figure 8 compares Qwen3-4B-Instruct (base), Qwen3-4B-Instruct-Panasonic-Claude (fine-tuned on the Claude-Opus-4.6-generated dataset), and RAG-Instruct-Llama3-8B across four FailureSensorIQ metrics and three Panasonic metrics.

Fine-tuning on the Claude-Opus-4.6-generated dataset improves Panasonic performance across all three metrics by a wide margin (e.g., Set-Match Accuracy 40.9%→56.4%), consistent with Table 9. On FailureSensorIQ, the picture is mixed: fine-tuning raises accuracy on the original, unperturbed questions (AccOrgIBM, 34.0%→49.6%), but reduces both macro- and micro-averaged F1 (F1-Macro 40.0%→33.5%; F1-Micro 66.0%→50.3%), and neither the base nor the fine-tuned Qwen model answers any perturbed FailureSensorIQ question correctly (AccPerIBM = 0% for both). RAG-Instruct-Llama3-8B shows the opposite profile: it is the strongest model on three of the four FailureSensorIQ metrics (AccPerIBM, F1-Macro, F1-Micro) yet collapses on Panasonic, scoring roughly 3–5× lower than either Qwen variant on all three Panasonic metrics (10.7% Set-Match Accuracy vs. 40.9–56.4%). This reinforces the pattern observed throughout this study: strength on one benchmark does not transfer to the other, and a model’s retrieval-augmented performance is highly dependent on whether its underlying knowledge base and fine-tuning data match the target domain.

Table 13: Underlying values for Figure 7.
<table><tr><td>Metric</td><td>Qwen-Org</td><td>Qwen-Pana-Qwen</td><td>RAG-Instruct</td></tr><tr><td>AccOrgIBM</td><td>34%</td><td>27%</td><td>28%</td></tr><tr><td>AccPerIBM</td><td>0%</td><td>0%</td><td>33%</td></tr><tr><td>F1-Macro (IBM)</td><td>40%</td><td>43%</td><td>62%</td></tr><tr><td>F1-Micro (IBM)</td><td>66%</td><td>74%</td><td>68%</td></tr><tr><td>Set-Match Acc. (Panasonic)</td><td>28%</td><td>42%</td><td>1%</td></tr><tr><td>Jaccard Similarity (Panasonic)</td><td>41%</td><td>58%</td><td>1%</td></tr><tr><td>F1-Score (Panasonic)</td><td>46%</td><td>63%</td><td>1%</td></tr></table>

Qwen-ORGINAL vs Qwen-Pana-Claude vs RAG-Instruct  
![](images/c7c4850acfe25ecae47ada53550c0667f4f593c18f839c3266dd2ceae17ef42a.jpg)  
Figure 8: A side-by-side comparison of FailureSensorIQ and Panasonic metrics for Qwen3-4B-Instruct, Qwen3-4B-Instruct-Panasonic-Claude, and RAG-Instruct-Llama3-8B.

## 5.2.3 Robustness to Perturbation: A Shared Limitation

Across both comparisons, one result stands out for what fine-tuning could not fix: AccPerIBM accuracy on the perturbed, rephrased variant of FailureSensorIQ’s questions — was 0% for the base Qwen3-4B-Instruct model and remained 0% after fine-tuning on either the Qwen- or Claudegenerated dataset (Tables 13 and 14). RAG-Instruct-Llama3-8B, by contrast, scored 33% on this metric in both comparisons, despite performing far worse than either Qwen variant on the Panasonic benchmark. This suggests the failure is not simply a matter of domain knowledge — both finetuned models gained substantial Panasonic-specific knowledge — but reflects a more fundamental brittleness to question rephrasing in the Qwen3-4B backbone itself, one that neither of our generator models’ training data was designed to address. This mirrors, and appears to sharpen, a pattern already reported for larger models on FailureSensorIQ, where perturbation was shown to reduce accuracy by as much as 41 percentage points [9]; here, for a small 4B-parameter model, the efect is total rather than partial. We view this as a limitation of the present study rather than a conclusion about fine-tuning data quality, and return to it in Section 6.

Table 14: Underlying values for Figure 8.
<table><tr><td>Metric</td><td>Qwen-Org</td><td>Qwen-Pana-Claude</td><td>RAG-Instruct</td></tr><tr><td>AccOrgIBM</td><td>34.0%</td><td>49.6%</td><td>28.0%</td></tr><tr><td>AccPerIBM</td><td>0.0%</td><td>0.0%</td><td>33.0%</td></tr><tr><td>F1-Macro (IBM)</td><td>40.0%</td><td>33.5%</td><td>62.0%</td></tr><tr><td>F1-Micro (IBM)</td><td>66.0%</td><td>50.3%</td><td>68.0%</td></tr><tr><td>Set-Match Acc. (Panasonic)</td><td>40.9%</td><td>56.4%</td><td>10.7%</td></tr><tr><td>Jaccard Similarity (Panasonic)</td><td>54.0%</td><td>68.9%</td><td>15.24%</td></tr><tr><td>F1-Score (Panasonic)</td><td>58.55%</td><td>72.66%</td><td>16.88%</td></tr></table>

## 6 Conclusion and Future Directions

This research provides a framework for constructing an appropriate dataset of industrial documents for model training and use in RAG. During this study, Panasonic product documents were collected and information and tables were extracted via an automated pipeline. We then artificially generated a dataset from these documents and evaluated its impact on the model. In this study, we used opensource models such as Dots.OCR and Qwen-30B-Instruct-2507, as well as the closed, API-based Claude-Opus-4.6, for dataset extraction and generation, and for evaluating impact, we employed Qwen-4B-Instruct, Phi-3-mini-4k, and the Llama3-based RAG-Instruct-8B. Also, in all of these studies, due to industry requirements for systems that retrieve all states, the simplest RAG method was implemented. The models used for the RAG system include Gemma3-300m for the embedding model and FAISS for retrieval. The results showed that this dataset can be efectively trained on small models with fewer than 10 billion parameters and that th s can improve model performance on evaluation metrics. Additionally, for dataset generation, it is possible to use open-source models, eliminating the need for massive commercial models like ChatGPT or Gemini.

Generating the dataset with the open-weight Qwen3-30B-A3B-Instruct model on local hardware cost approximately \$3.2 in compute, compared with roughly \$330 in API usage for Claude-Opus-4.6 (Table 15) — roughly two orders of magnitude more expensive. While Claude-Opus-4.6 produced a cleaner raw dataset (a far lower filtering rate, Section 3.5) and yielded larger downstream finetuning gains (Section 4.4), the improvement was not proportional to the cost diference, suggesting that open-weight models remain a highly cost-efective choice for this type of dataset construction. Moreover, the open-weight model ecosystem is advancing rapidly, and larger open-weight models are increasingly capable of approaching the quality of closed, API-based frontier models at a fraction of the cost.

For future work, we plan to evaluate a broader range of language models to further analyze generalization across architectures and parameter scales. In addition, we aim to expand the dataset by incorporating documents from diverse industrial knowledge sources in order to construct a largescale, comprehensive industrial benchmark. Future research will also explore more advanced RAG architectures beyond the basic implementation used in this study. We also intend to investigate multimodal industrial documents that integrate textual and visual information, enabling the extraction and utilization of embedded images, diagrams, and structured visual content alongside text to improve model robustness in real-world industrial scenarios. Most notably, every model evaluated in this study — regardless of fine-tuning data — completely failed to answer any perturbed FailureSensorIQ question correctly (Section 5.2), a limitation our current dataset construction pipeline does not address since it does not include rephrased or perturbed variants of its own generated questions. We consider closing this gap, for instance by explicitly generating paraphrased or adversarially rephrased variants of each Q&A pair during dataset construction, one of the most promising directions for improving robustness in future iterations of Industrial-Instruction.

Table 15: Compute/monetary cost of generating the Panasonic dataset with each generator model.
<table><tr><td>Generator</td><td>Compute Time</td><td>Cost</td><td>Notes</td></tr><tr><td>Qwen3-30B-A3B-</td><td>1h 43m</td><td>$3.2</td><td>Local Pro 6000</td></tr><tr><td>Instruct Claude-Opus-4.6</td><td></td><td>$330</td><td>WS Token-based</td></tr><tr><td></td><td></td><td></td><td>billing</td></tr></table>

## A Dataset Creation Process

The dataset construction process was completed in a total of 1 hour and 43 minutes on a Pro 6000 WS system.

## B Model Training Settings and Time

The model was trained for 12 hours, 3 minutes, and 3 seconds using two NVIDIA RTX 5090 GPUs. The target batch size during training was set to 64. Due to GPU memory limitations, directly implementing this batch size was not feasible; therefore, gradient accumulation was employed. In this configuration, the per-device batch size was set to 2 with 32 gradient accumulation steps, resulting in an efective batch size of 64. It should be noted that, in this setup, the computed loss represents an accumulated estimate and may slightly difer from the exact large-batch gradient.

## C Additional MMLU Detail

Table 16 and Table 17 report the largest subject-level MMLU changes for finetuned\_v1\_qwen and finetuned\_v2\_claude, respectively, relative to the base model. Full 57-subject scores for all three models are available in our released evaluation logs.

Notably, public\_relations and anatomy regress under both fine-tunes, which suggests these two subjects may be inherently sensitive to fine-tuning on this base model regardless of the training data source, rather than an artifact specific to either dataset. Conversely, a handful of subjects — college\_mathematics, machine\_learning, and moral\_disputes among them — improve under both fine-tunes, suggesting the fine-tuning process itself contributes some dataset-independent generalization alongside whatever forgetting occurs elsewhere. We recommend manually inspecting the moral\_scenarios log samples for finetuned\_v1\_qwen specifically, since a -10.7 point swing on a single subject is large enough to warrant ruling out answer-format drift as opposed to genuine reasoning loss.

Table 16: Largest MMLU regressions for finetuned $\_ \mathrm { v 1 }$ \_qwen (subjects with the biggest accuracy drop vs. the base model).
<table><tr><td>Subject</td><td>Original</td><td>finetuned  $\mathbf { \nabla } _ { \cdot } \mathbf { v } \mathbf { 1 }$  qwen</td><td> $\Delta$ </td></tr><tr><td>moral scenarios</td><td>51.17%</td><td>40.45%</td><td>-10.72</td></tr><tr><td>public_relations</td><td>69.09%</td><td>62.73%</td><td>-6.36</td></tr><tr><td>marketing</td><td>91.88%</td><td>87.18%</td><td>-4.70</td></tr><tr><td>high_school_biology</td><td>90.65%</td><td>86.13%</td><td>-4.52</td></tr><tr><td>high_school_microeconomics</td><td>91.60%</td><td>87.82%</td><td>-3.78</td></tr><tr><td>human_aging</td><td>73.54%</td><td>69.96%</td><td>-3.58</td></tr><tr><td>anatomy</td><td>71.11%</td><td>68.15%</td><td>-2.96</td></tr><tr><td>astronomy</td><td>87.50%</td><td>84.87%</td><td>-2.63</td></tr></table>

Table 17: Largest MMLU changes for finetuned\_ $\mathrm { \Delta _ { \cdot } v 2 }$ \_claude vs. the base model: regressions (top) and improvements (bottom).
<table><tr><td>Subject</td><td>Original</td><td>finetuned  $\mathbf { v 2 }$  claude</td><td> $\Delta$ </td></tr><tr><td>public_relations</td><td>69.09%</td><td>63.64%</td><td>-5.45</td></tr><tr><td>anatomy</td><td>71.11%</td><td>66.67%</td><td>-4.44</td></tr><tr><td>college_chemistry</td><td>54.00%</td><td>50.00%</td><td>-4.00</td></tr><tr><td>high_school_computer_science</td><td>89.00%</td><td>85.00%</td><td>-4.00</td></tr><tr><td>marketing</td><td>91.88%</td><td>88.46%</td><td>-3.42</td></tr><tr><td>global_facts</td><td>43.00%</td><td>51.00%</td><td>+8.00</td></tr><tr><td>college_mathematics</td><td>48.00%</td><td>55.00%</td><td>+7.00</td></tr><tr><td>machine_learning</td><td>60.71%</td><td>66.07%</td><td>+5.36</td></tr><tr><td>security_studies</td><td>74.29%</td><td>78.78%</td><td>+4.49</td></tr><tr><td>international law</td><td>80.17%</td><td>83.47%</td><td>+3.30</td></tr></table>

## D Prompt Templates for the Five RAG Scenarios

Each of the five dataset-generation scenarios (r0–r4, Section 3.4) uses a prompt built from a shared template that instructs Qwen3-30B-A3B-Instruct to generate a question $q ^ { * }$ , an answer $a ^ { * }$ , and, when the simulated instruction is multiple-choice, five options labeled A–E. All five prompts share the same document placeholder(s), instruction-simulation mechanism, and output format; they difer only in how many documents are supplied and in the single requirement rule that defines the query–document relationship, as summarized in Table 18.

Listing 5: Full prompt template for scenario r0 (Useless Document).

<Documents>   
{doc}   
</Documents>   
Your task is to generate an English question q\* and a corresponding response a\* based on the provided <   
Documents>. Please note that the question q\* can take various forms, not limited to questions with a   
question mark, but also including statements, instructions, and other formats. You need to follow the   
requirements below to generate the q\* and a\* (RAG Paradigms):   
1. q\* should be related to the <Documents>, but the <Documents> can not provide any useful information for   
answering q\*.   
2. a\* should be able to answer q\*, ensuring that the response a\* is accurate, detailed, and comprehensive.   
3. q\* must be a standalone question. Strictly avoid using phrases that reference the source material, such as   
"Based on the provided context," "According to the documents," "In the text," or similar meta  
references.   
Additionally, to ensure diversity, richness, and high quality in the question q\* you generate, we will   
randomly provide a question for you to emulate. In other words, while satisfying the requirements above,   
make q\* similar in task requirement and expression to the <Simulated Instruction> below:   
<Simulated Instruction>   
{simulated\_instruction}   
</Simulated Instruction>   
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q\*   
MUST ALSO include exactly 5 options labeled A-E, and a\* MUST include the correct option(s) in the same   
JSON list format used in the simulated instruction. Always generate 5 options, even if the original   
document does not contain any option-like content.   
Please directly generate the question-answer-options (q\*, a\*, options\*) following all the rules above in the   
format of {"q\*": ..., "a\*": ..., "options\*": ...}, so for a\* select A-E correct option.   
Ensure the quality of the generated (q\*, a\*, options\*).

Table 18: How each scenario prompt difers from the shared template.
<table><tr><td>Scenario</td><td>Docs</td><td>Core requirement (rule 1)</td></tr><tr><td>Useless Document (r0)</td><td>1</td><td>Document is related but provides no useful informa- tion for answering q*.</td></tr><tr><td>Single-Doc. Support (r1) 1</td><td></td><td>Document provides sup- porting clues but not an explicit answer.</td></tr><tr><td>Multi-Doc. Support (r2) ≥2</td><td></td><td>Multiple documents jointly provide clues but not an ex- plicit answer.</td></tr><tr><td>Single-Doc. Answer (r3) 1</td><td></td><td>Document contains the complete, explicit answer.</td></tr><tr><td>Multi-Doc. Answer (r4) ≥2</td><td></td><td>Answer requires multi-hop reasoning across documents.</td></tr></table>

Listing 5 shows the full prompt used for the Useless Document (r0) scenario as a representative example; the remaining four prompts follow the identical structure with only the document count and rule 1 changed as listed in Table 18.

Full source code for all five prompt templates, along with the dataset generation pipeline, is available in our public code repository.

## Acknowledgments

We thank Hamedan University of Technology and the University of Antwerp / Flanders Make Strategic Research Center for their institutional support of this work.

## References

[1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, et al. Phi-3 technical report: A highly capable language model locally on your phone, 2024.

[2] PhD D.M. Anisuzzaman, PhD Jefrey G. Malins, MD Paul A. Friedman, and PhD Zachi I. Attia. Fine-tuning large language models for specialized use cases. Mayo Clinic Proceedings: Digital Health, 3, 2024. doi: 10.1016/j.mcpdig.2024.11.005.

[3] Salvador D. Atagong, Henri E. Z. Tonnang, K. Senagi, M. Wamalwa, K. Agboka, and John Odindi. A review on knowledge and information extraction from pdf documents and storage approaches. Frontiers in Artificial Intelligence, 8, 2025. doi: 10.3389/frai.2025.1466092.

[4] Omar El Bachyr, Yewei Song, Saad Ezzini, Jacques Klein, Tegawendé F. Bissyandé, Anas Zilali, Ulrick Ble, and Anne Goujon. Empirical evaluation of pdf parsing and chunking for financial question answering with rag. Proceedings of the IEEE/ACM 48th International Conference on Software Engineering: Software Engineering in Practice, 2026. doi: 10.1145/3786583.3786911.

[5] Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, and Moharram Challenger. Knowledge extraction from technical reports based on large language models: An exploratory study. In 2024 15th International Conference on Information and Knowledge Technology (IKT), pages 265–271, Iran, 2024. IEEE, IEEE.

[6] Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, and Moharram Challenger. industrial-instruction-dataset (revision 7eadea0), 2026. URL https://huggingface. co/datasets/Parssky/industrial-instruction-dataset.

[7] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John P. Cunningham. Lora learns less and forgets less. Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/2405.09673. Also available as arXiv:2405.09673.

[8] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents, 2023.

[9] Christodoulos Constantinides, Dhaval Patel, Shuxin Lin, Claudio Guerrero, Sunil Dagajirao Patil, and Jayant Kalagnanam. Failuresensoriq: A multi-choice qa dataset for understanding sensor relationships and failure modes. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https://arxiv.org/abs/ 2506.03278. Also available as arXiv:2506.03278.

[10] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023. URL https://arxiv.org/abs/2305.14314. Also available as arXiv:2305.14314.

[11] Henning Femmer, Frank Houdek, Max Unterbusch, and Andreas Vogelsang. Description and comparative analysis of QuRE: A new industrial requirements quality dataset. In 2025 IEEE 33rd International Requirements Engineering Conference (RE), 2025. URL https: //arxiv.org/abs/2508.08868. Also available as arXiv:2508.08868. Exact page numbers/DOI not confirmed – please verify against your IEEE Xplore access before submission.

[12] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, et al. Gemma 3 technical report, 2025.

[13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2009. 03300. Also available as arXiv:2009.03300.

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

[15] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

[16] Jef Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2021. doi: 10.1109/TBDATA.2019.2921572. URL http://arxiv.org/abs/1702.08734. Also available as arXiv:1702.08734.

[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020.

[18] Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. dots.ocr: Multilingual document layout parsing in a single vision-language model, 2025. URL https://arxiv.org/ abs/2512.02498.

[19] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 33:3776, 2025. URL https://arxiv. org/abs/2308.08747. Also available as arXiv:2308.08747. Exact ending page number/DOI not confirmed – please verify against your IEEE Xplore access before submission.

[20] Dehai Min, Nan Hu, Rihui Jin, Nuo Lin, Jiaoyan Chen, Yongrui Chen, Yu Li, Guilin Qi, Yun Li, Nijun Li, and Qianren Wang. Exploring the impact of table-to-text methods on augmenting LLM-based question answering with domain hybrid data. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 464–482, Mexico City, Mexico, jun 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-industry.41. URL https://aclanthology.org/2024.naacl-industry.41/.

[21] Rohaan Nadeem, Tahir Iqbal, Noor Fatima, Junaid Altaf, Asma Irshad, and Asif Farooq. Extraction of user-defined information from pdf. 2024 International Conference on Decision Aid Sciences and Applications (DASA), pages 1–6, 2024. doi: 10.1109/dasa63652.2024.10836169.

[22] Christopher Nguyen, William Nguyen, Atsushi Suzuki, Daisuke Oku, Hong An Phan, Sang Dinh, Zooey Nguyen, Anh Ha, Shruti Raghavan, Huy Vo, Thang Nguyen, Lan Nguyen, and Yoshikuni Hirayama. Semikong: Curating, training, and evaluating a semiconductor industryspecific large language model, 2024. URL https://arxiv.org/abs/2411.13802.

[23] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, Jin Shi, Fan Wu, Pei Chu, Minghao Liu, Zhenxiang Li, Chao Xu, Bo Zhang, Botian Shi, Zhongying Tu, and Conghui He. OmniDocBench: Benchmarking diverse PDF document parsing with comprehensive annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24838– 24848, 2025. URL https://arxiv.org/abs/2412.07626. Also available as arXiv:2412.07626.

[24] Ray Smith. Tesseract OCR Engine, 2007. URL https://web.archive.org/web/ 20160819190257/tesseract-ocr.googlecode.com/files/TesseractOSCON.pdf.

[25] Brandon Smock, Rohith Pesala, and Robin Abraham. PubTables-1m: Towards comprehensive table extraction from unstructured documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4634–4642, 2022.

[26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[27] Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, et al. Embeddinggemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354, 2025.

[28] Luping Wang, Sheng Chen, Linnan Jiang, Shuaiqun Pan, Runze Cai, Sen Yang, and Fei Yang. Parameter-eficient fine-tuning in large language models: a survey of methodologies. Artificial Intelligence Review, 58, 2024. doi: 10.1007/s10462-025-11236-4.

[29] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 13484–13508, 2023.

[30] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4818–4829, 2024. URL https://arxiv.org/abs/2311.06242. Also available as arXiv:2311.06242.

[31] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[32] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multihop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium, October–November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https:// aclanthology.org/D18-1259/.