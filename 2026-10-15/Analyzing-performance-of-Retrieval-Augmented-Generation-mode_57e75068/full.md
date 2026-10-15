# Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks

1<sup>st</sup> Fazle Mohammed Tasfiq Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh fazle.mohammed.tasfiq@g.bracu.ac.bd

3<sup>rd</sup> Sheikh Md. Tahmid Hossain Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh sheikh.md.tahmid.hossain@g.bracu.ac.bd

5<sup>th</sup> Tanzim Ahmmed Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh tanzim.ahmmed@g.bracu.ac.bd

7<sup>th</sup> Anika Ahmed Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh anika.ahmed@bracu.ac.bd

9<sup>th</sup> Dr. Farig Yousuf Sadeque Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh farig.sadeque@bracu.ac.bd

2<sup>nd</sup> Sabab Attin Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh sabab.attin@g.bracu.ac.bd

4<sup>th</sup> Raiat Reza Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh raiat.reza@g.bracu.ac.bd

6<sup>th</sup> Nafis Chowdhury Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh ext.nafis.chowdhury@bracu.ac.bd

8<sup>th</sup> Moinul Haque Dept. of Computer Science and Engineering BRAC University Dhaka, Bangladesh ext.moinul.haque@bracu.ac.bd

Abstract—Retrieval-Augmented Generation (RAG) models have become almost a cornerstone of opendomain question answering tasks; in which the quality of answers generated in the process, greatly relies on the efectiveness of retrieval mechanisms. In the realm of LLM, the last two years have seen tremendous progress in RAG architectures, with many models reporting to have improved performance through various design innovations. The cultural question and answering systems of South Asia, in particular, present a specific challenge since the region is home to the vast number of languages and traditions, which are closely interrelated. Along with these deeply intertwined historical narratives increase the need for culturally grounded and context-aware interpretation. Despite major architectural advances in LLMs, claiming significant improvements, challenges remain when answering these region specific questions in native languages. In this study, we aim to extract high-performing factors to guide the development of optimized hybrid RAG systems. We further explore whether the integration of RAG changes performance diferently for cultur-

ally grounded and non-cultural questions in a South Asian multilingual setup, and whether RAG produces a higher overall percentage of correct answers than contexted answering. We also examine how LLM performance changes when RAG methods are integrated into the question-answering pipeline in comparison with non-contexted and contexted answering and, evaluate whether the introduction of RAG benefits popular or non-popular culturally grounded questions more. Not only does our work evaluate the current methods but also carves a path for future designs of more accurate and eficient question-answering pipelines.

Index Terms—Adaptive Retrieval, Benchmarking, Domain-specific QA, Knowledge Graph, Large Language Model, Multimodal, Question Answering, Relevance Estimation, Retrieval-Augmented Generation, Transformer.

## I. Introduction

The rapid advancements in large language models (LLMs) have inaugurated a new era of natural language processing, enabling unprecedented capabilities in complex tasks such as summarization and Question Answering (QA). These models acquire their powerful, generalpurpose language understanding through a process controlled by scaling laws, involving the training of billions of parameters on massive textual corpora [1]. Prominent LLM families, including GPT, LLaMA, and PaLM, demonstrate this property by internalizing a vast set of parametric knowledge within their internal weights. Yet, the operational utility of these systems for knowledgeintensive applications is fundamentally constrained by the static nature of their training data [2]. This structural limitation creates a defined knowledge cutof, rendering models temporally bound and incapable of accessing or processing information created subsequent to their final training iteration. More critically, LLMs are system ically prone to hallucination, which is the generation of false, misleading, or unfounded information [3]. This phenomenon is often classified as intrinsic hallucination, where the output contradicts the immediate input context, or extrinsic hallucination, which arises when the model attempts to fill internal knowledge gaps by fabricating details inconsistent with its training data [4]. For factually demanding applications like QA, these inherent limita tions severely compromise system reliability and diminish user trust. To overcome these fundamental barriers and augment LLMs with access to real-time, verifiable, and vastly expanded knowledge, the paradigm of Retrieval-Augmented Generation (RAG) was introduced in 2020 by Lewis et al [5]. RAG fundamentally transforms the LLM architecture by fusing its pre-trained parametric memory with a dedicated non-parametric memory, an external, updatable knowledge base. This technique enables the LLM to dynamically retrieve relevant evidence from a vast corpus, conditioning its generation on verifiable, external context [6]. By explicitly grounding its responses, RAG efectively expands the model’s factual scope beyond its original training boundary and significantly mitigates the risk of extrinsic hallucination. Consequently, RAG has rapidly become a central focus for research, proving capable of converting a general LLM into a more accurate, knowledgeable, and reliable system, particularly for complex QA tasks. The adoption of RAG models ofer substantial benefits over traditional parametric LLMs, especially by generating language that is more specific, diverse, and factual compared to state-of-the-art parametriconly sequence-to-sequence baselines [5]. Moreover, RAG is a strategic and eficient way of specialization. The ability to create and leverage a knowledge graph of external knowledge sources enables RAG to efectively transform a general LLM into a domain-specific expert [7]. This domain adaptation ability is a great benefit in terms of resource eficiency over the intensive and expensive finetuning that is needed to add equivalent specialized knowl edge as parameters in the model. RAG has spread quickly, sparking an unprecedented surge of architectural research that has produced a new field of architecture, with a wide array of methods. Currently, RAG systems are widely categorized according to the innovations that are made in the retriever—how documents are selected, the generator— how context is synthesized, or hybrid approaches that combine coordinated innovations in both their respective components [8]. This multi-component system, however, does have inherent complexity and potential areas for failure. Some of the basic tasks of RAG, including chunking large bodies of knowledge, embedding them in vectors, and indexing them eficiently in a database, are vital [7]. Even with the most powerful LLM, if these initial processes are not correct, the retriever will provide irrelevant or insuficient context. Thus, the analysis of performance for tasks in QA should go beyond linguistic fluency. Rather, evaluation requires specialized, diagnostic measures for factual consistency and verifiability. They include Groundedness (or Faithfulness): Does the response keep to the provided context? Context Relevance: Are the retrieved documents relevant to the question? Answer Relevance: How well does the final response answer the user’s question? [9] This will be a systematic assessment that will help to record progress and pinpoint the most successful RAG combinations. Now it is time to examine the performance dynamics of Retrieval-Augmented Generation models for QA tasks systematically, by looking at the influence of these design decisions and assessing their success by applying challenging, RAG-specific evaluations. This analysis can yield an actionable insight into building strong, reliable, and successful knowledge-empowered language models.

![](images/07f844516c704342c496c0c6941a509707e7b30344ab333781f77309b81e0f19.jpg)  
Fig. 1. Workflow of the RAG model evaluation framework.

## II. Related Works

Based on the overall finding, the assessment of RAG models in Question Answering(QA) needs to be expanded to beyond retrieval coverage. Rather, the three dimensions of structural knowledge management, execution eficiency and strong verification mechanisms should also be emphasized. The CRAG benchmark is the benchmark most commonly used as a performance benchmark, which quantifies the persistent Trust Gap, indicating that, in complex real-world QA settings, state-of-the-art RAG systems only achieve approximately 63% truthfulness without hallucination [10]. The need for this has been met by recent advances in the architecture, which repeatedly show that structured knowledge fares much better than naive vector matching for complex queries. To illustrate, GraphRAG outperformed vector RAG in comprehensiveness with up to 83% accuracy when it comes to the global contextual understanding of unstructured data as Knowledge Graphs (KGs) [11]. This structural precision is realized in a useful and domain-specific manner by DO-RAG, which combines KG traversal with vector retrieval, resulting in the joint score gains of up to 33.38% in technical domains [12]. The eficiency challenge in dynamic corpora is solved by EraRAG [13], whose LSH-based incremental update mechanism slashes KG reconstruction time by 77.5%, validating that eficient structured storage is essential for modern RAG performance.

The second major pillar is performance optimization through adaptive and eficient execution. On more complex multi-hop queries, Collab-RAG [14] proposes a breakthrough in query decomposition performance by training a small language model (SLM) to break down queries, showing that a 3B SLM can outperform a frozen 32B LLM in query decomposition performance. This highlights the importance of specialised, aligned orchestration models reducing the cost of running queries significantly [14]. The achievable data and results of ARAGOG [15] indicate that the maximum benefits of optimisation of certain elements like the use of Hypothetical Document Embedding (HyDE) and reranking with LLM provide the highest benefits to the retrieval accuracy. This therefore confirms that the RAG performance is extremely sensitive to the specific combination and optimisation of high-impact elements.

## III. Dataset and Framework

## A. Wikipedia webscraping

The dataset construction process began with the selection of a subset of South Asian languages. Our choices included our native Bangla, along with Hindi, Urdu, Tamil and Nepali, covering the wider range of South Asia. However, since the languages we are focusing on are low-resource languages so there are no appropiate datasets for our experiments since most of the low-resource datasets are machine translated [16]. Upon selecting the languages, our task was to find appropriate datasets for these languages. The main motivation for constructing this multilingual dataset is inspired from the prior work by [17]. To get the dataset for those languages, we needed accurate regional data. So we webscraped from the regional wikipedia pages including: bn.wikipedia.org, ur.wikipedia.org, hi.wikipedia.org, ta.wikipedia.org and finally ne.wikipedia.org for native articles on Bangla, Urdu, Hindi, Tamil and Nepali respectively. Our approach began by selecting the Wikipedia article of the country or state with the highest number of native speakers for a given language. This article served as the level-0 (landing) page. From this page, we followed in-page hyperlinks that redirected to separate Wikipedia articles, which were treated as level-1 terms. The process was then repeated for links found within level-1 pages to obtain level-2 terms. In total, this resulted in a three-level hierarchy of terms, comprising the landing page and two successive levels of hyperlinkbased expansions. The levels were noted and to identify the common redirects, for each term a parent list was kept so that duplicate redirects can be avoided. In order to facilitate our tracking of the terms, all the corresponding URLs redirecting to the term were kept as well.

After collecting the titles and their corresponding article links, we observed a large number of Wikipedia page edit links as well as duplicate titles. Links containing edit&redlinks=1 were filtered out along with duplicate entries, and were removed. Additionally, some links redirected to Wikipedia calendar date pages, which were also discarded. Furthermore, stopwords were eliminated from our datasets with the help of publicly available stopword lists in kaggle and/or hugging face.

In this way, for all the languages, the amount of data present in each dataset before and after preprocessing:

TABLE I  
Dataset size before and after preprocessing
<table><tr><td>Language</td><td>Direct webscraping</td><td>After preprocessing</td></tr><tr><td>Bangla</td><td>17,794</td><td>10,769</td></tr><tr><td>Hindi</td><td>16,362</td><td>9,541</td></tr><tr><td>Tamil</td><td>13,258</td><td>11,310</td></tr><tr><td>Urdu</td><td>8,795</td><td>7,805</td></tr><tr><td>Nepali</td><td>2,641</td><td>2,574</td></tr></table>

## B. Context generation

Context was collected from Wikipedia for question generation. Our approach was that we used the links of the wikipedia pages from the database and scraped the first four lines from body paragraph content as context. However, in some cases, some pages did not contain a total of 4 lines, rather less. Some pages, specifically known as redlinks, had no content or included just a list of items and no body paragraph content. Contexts for such pages came up empty or zero. Along with this, we used Wikipedia Pageviewer API to list down the page view counts of each title in the time period from start to end of 2025 and normalized it between scores of 0 and 1 (otherwise the gap between counts are too sparse). Finally, the contents were taken and a counter variable was kept to note the number of context lines available. The generated files now had the title, url, context and number of lines and normalized popularity count as present in the sample table below :

## C. Question dataset construction

The context we extracted was then used for question generation. Qwen 3 32b has been used to generate the question. We designed the prompts in such a way that the model would design the question from the context given the answer of the question should be the term we collected and the term should not be present in the question. Two lines of context were used to generate the questions. The questions were generated by giving the same prompt translated to the other regional languages to keep consistency. The prompts for all the respective languages are translated into respective languages from this prompt- “From the given context ’{context}’, write exactly one name of language question whose answer would be only ’{title}’. The answer itself, or any part of it, must not be used in the question. There is no need to write anything other than the complete question that is generated.” During the experimental phase, a problem arose that all the terms did not have 4 line contexts. But in most cases around 2 lines of context were available. We used 2 lines as context to retain maximum dataset and discarded those titles with only 1 line of context. Another case was observed that impacted this decision was the tendency of model hallucination when feeding too little (1 line) context or too much (4 line) context. 1 line context questions were almost too vague and 4 line context provided additional information for the model.

## D. Assessment of Question clarity

A test sample of 250 questions on both 1-line and 2- line were analysed to determine how many lines of context provided better question. We used the following factors to evaluate:

Relevance: The question correctly targets the core ideas present in the given answers.

Factual Accuracy: The question does not introduce incorrect assumptions or false premises from the answers.

Question Clarity: The question is clearly and unambiguously phrased based on the answers.

Creativity: The question frames the existing answers in a novel, insightful, or non-trivial way.

Coherence: The question is logically structured and internally consistent with the information in the answers.

Task-Specific Appropriateness: title not repeated in question, question has only the complete title as answer, no other answer is possible except the title and question is in expected language or not.

All the ratings were done by manual human input out of 5, and then an average was calculated. Upon the evaluation, it was found out that in case of all languages, better questions were observed for 2 line contexts. In case of other languages, the respective questions were translated to Bangla first via google translator model call and then evaluated. Since these questions were all judged on the basis of translation, we could not judge the coherence. So no coherence evaluation was done and the values might vary if judged by a native speaker. It was found that the performance is better when there are 2 lines of context used while question generation. After the confirmation of using 2 line contexts, we proceeded towards full set question generation and answer generation. All the questions were generated by the same prompt as mentioned before. For answer generation, we used four models, 3 open and 1 closed source for diversity. Llama 4 Maverick 17B by Meta (open source), DeepSeek V3-0324 (open source), Gemma 3 27B by Google (open source) and Grok 4.1 Fast by xAI (closed source).

## E. LLM-Based Semantic Classification

We followed an LLM-assisted semantic classification method. The categorization scheme from Chowdhury et al.’s [17] Bengali cultural knowledge evaluation provided a peer-reviewed taxonomy for organizing titles into meaningful semantic groups that worked as a reference structure for this stage. Each title was treated as an individual instance and passed to Meta Llama Maverick with a wellconstrained instruction to return only one label from the predefined set of labels. This LLM was selected upon evaluating its stronger performance in question-answering tasks in our prior analysis. The model outputs were then stored as semantic labels for each title.

The following definitions were provided in the model instructions when passing the titles:

Table II  
Category definitions, classes, and examples
<table><tr><td>Category</td><td>Class</td><td>Definition</td><td>Example</td></tr><tr><td>Event</td><td>Cultural</td><td>Specific occurrences such as festivals, battles, conferences, natural disasters, ceremonies, or incidents.</td><td>92022 [Pohela Boishakh]</td></tr><tr><td>Religion</td><td>Cultural</td><td>Religious beliefs, practices, deities, scriptures, rituals, or religious movements.</td><td>司 [Buddhism]</td></tr><tr><td>Food</td><td>Cultural</td><td>Dishes, ingredients, cuisines, beverages, or culinary items.</td><td>211122 [Rasgulla]</td></tr><tr><td>Entertainment</td><td>Cultural</td><td>Movies, music, television, games, arts, celebrities, or media-related content.</td><td>年0 [Folk music]</td></tr><tr><td>Historical</td><td>Cultural</td><td>Historical eras, dynasties, treaties, or historically significant topics not classified as specific events.</td><td>[Gupta Empire]</td></tr><tr><td>People</td><td>Cultural</td><td>Individual persons, real or fictional, including public figures, scholars, artists, and athletes.</td><td>可小G5000 [Kazi Nazrul Islam]</td></tr><tr><td>Politics</td><td>Non-cultural</td><td>Political parties, ideologies, political offices, public policies, governance, or political movements.</td><td>[Constitution of Bangladesh]</td></tr><tr><td>Locations</td><td>Non-cultural</td><td>Countries, cities, regions, landmarks, geographical entities, or physical places.</td><td>TR-Cn [Bangladesh]</td></tr><tr><td>Language</td><td>Non-cultural</td><td>Languages, dialects, scripts, linguistic concepts, or grammar-related topics.</td><td>[Bangla Language]</td></tr><tr><td>Material</td><td>Non-cultural</td><td>Physical substances, chemicals, elements, minerals, fabrics, or tangible materials.</td><td>[Muslin]</td></tr><tr><td>Organization</td><td>Non-cultural</td><td>Companies, institutions, NGOs, agencies, universities, clubs, or structured groups.</td><td>22 [The Daily Prothom Alo]</td></tr><tr><td>Health</td><td>Non-cultural</td><td>Diseases, medical conditions, treatments, anatomy, healthcare systems, or psychological conditions.</td><td>心 [Dengue Fever]</td></tr><tr><td>Temporal</td><td>Non-cultural</td><td>Time-related concepts such as months, years, calendars, historical periods when referring to time itself, or dates.</td><td>C0 [Chaitra]</td></tr><tr><td>Economics</td><td>Non-cultural</td><td>Financial systems, markets, economic theories, currencies, trade, or economic indicators.</td><td>TRaCnC可A [Economy of Bangladesh]</td></tr><tr><td>Sports</td><td>Non-cultural</td><td>Sports types, tournaments, teams, competitions, or athletic disciplines.</td><td>a [Cricket]</td></tr><tr><td>Miscellaneous</td><td>Non-cultural</td><td>Titles that do not clearly belong to any of the above categories.</td><td>同 [Rickshaw]</td></tr></table>

TABLE III  
Amount of data present in the final dataset
<table><tr><td>Language</td><td>Final data count</td></tr><tr><td>Bangla</td><td>10,278</td></tr><tr><td>Hindi</td><td>8,694</td></tr><tr><td>Tamil</td><td>8,553</td></tr><tr><td>Urdu</td><td>5,806</td></tr><tr><td>Nepali</td><td>2,405</td></tr></table>

## F. Dataset Analysis

After the classification was done, the full dataset was formulated and, since we pivoted to 2 line context, the dataset size reduced from the previous ones of Table I to:

Our final dataset hence had 4 columns: Title, context 2 line, question 2 line and cluster i.e. the categories.

TABLE IV  
Sample of multilingual QA dataset
<table><tr><td>Title</td><td>Context</td><td>Question</td><td>Cluster</td></tr><tr><td>交回 The Daily Ittefaq</td><td>? REGR 31 The Daily Ittefaq is a national daily newspaper pub- lished in the Bengali language in Bangladesh. It also has an English online edition.</td><td>What is a national daily newspaper in the Bengali language of Bangladesh?</td><td>organization</td></tr><tr><td>TRAR Muzaffarnagar</td><td>RR市市市市R Muzaffarnagar is a city in Uttar Pradesh, India, and in a city with the same name? the headquarters of Muzaffarnagar district.</td><td>Which district of Uttar Pradesh has its headquarters</td><td>locations</td></tr><tr><td>Kashmiri language</td><td>The Kashmiri language is an important language of India and Pakistan, with slightly over 1.5 million speakers.</td><td>What language is spoken in India and Pakistan by over 1.5 million people?</td><td>language</td></tr><tr><td>I Pallankuzhi</td><td>LT, @T . L L1600T60f60T 历の . Pallanguzhi is a traditional game generally played by women.</td><td>? σfWhich traditional game is played by women?</td><td>entertainment</td></tr><tr><td>H Himalayas</td><td>,    The Himalayas separate the Indian subcontinent from the Tibetan Plateau and include Mount Everest.</td><td>What mountain range separates the Indian subconti- nent from the Tibetan Plateau and includes Mount Everest?</td><td>locations</td></tr></table>

## IV. RAG Setup and Analysis

The next part of our work adopted Retrieval-Augmented Generation as the main framework for context-supported answer generation. We integrated RAG to our pipeline because it combines a parametric LLM with a nonparametric retrievable memory base, thus allowing generation to be constrained on relevant retrieved contexts or ground truths instead of only relying on the model’s base knowledge [5] . In our setup, the retrieved context served as the external knowledge base or ground truth, while the LLM generated the final answer from the query and retrieved passages. This phase began with selecting three frameworks for implementation of the pilot evaluation of RAG-based answer generation- LangChain, LangGraph, and LlamaIndex. Each of these frameworks provides a dif ferent approach towards organizing retrieval-based generation. LangChain was included because it supports chainbased composition of LLM workflows, where retrieval, prompting, model invocation, and output parsing can be connected sequentially [24] . In a slightly diferent approach, LangGraph represents applications through a shared state of nodes and edges to update the state and define next execution steps [25] . LlamaIndex provides retrieval-augmented LLM applications with data indices and query engines, where each engine provides naturallanguage access to indexed data [26] . A 100-sample pilot test was carried out across the three frameworks under similar retrieval and generation settings for each language using each LLM. The results showed weaker performance of LangChain based pipeline compared to the other two, specifically for lower performing language settings such as Nepali, Tamil, and Urdu. As a result, further work was carried out using LangGraph and LlamaIndex as RAG frameworks.

TABLE V  
Performance comparison across languages and frameworks
<table><tr><td>Lang.</td><td>LLM</td><td>LC</td><td>LG</td><td>LI</td></tr><tr><td rowspan="4">Bangla</td><td>DeepSeek V3</td><td>64.0</td><td>64.0</td><td>57.0</td></tr><tr><td>Gemma 3</td><td>67.0</td><td>66.0</td><td>70.0</td></tr><tr><td>LLaMA 4</td><td>66.0</td><td>66.0</td><td>68.0</td></tr><tr><td>Grok 4.1</td><td>61.0</td><td>65.0</td><td>60.0</td></tr><tr><td rowspan="4">Hindi</td><td>DeepSeek V3</td><td>37.0</td><td>38.0</td><td>34.0</td></tr><tr><td>Gemma 3</td><td>46.0</td><td>46.0</td><td>44.0</td></tr><tr><td>LLaMA 4</td><td>63.0</td><td>63.0</td><td>64.0</td></tr><tr><td>Grok 4.1</td><td>56.0</td><td>60.0</td><td>58.0</td></tr><tr><td rowspan="4">Nepali</td><td>DeepSeek V3</td><td>51.0</td><td>53.0</td><td>55.0</td></tr><tr><td>Gemma 3</td><td>53.0</td><td>55.0</td><td>53.0</td></tr><tr><td>LLaMA 4</td><td>64.0</td><td>74.0</td><td>76.0</td></tr><tr><td>Grok 4.1</td><td>60.0</td><td>67.0</td><td>61.0</td></tr><tr><td rowspan="4">Tamil</td><td>DeepSeek V3</td><td>40.0</td><td>43.0</td><td>41.0</td></tr><tr><td>Gemma 3</td><td>49.0</td><td>46.0</td><td>51.0</td></tr><tr><td>LLaMA 4</td><td>63.0</td><td>66.0</td><td>68.0</td></tr><tr><td>Grok 4.1</td><td>44.0</td><td>41.0</td><td>44.0</td></tr><tr><td rowspan="4">Urdu</td><td>DeepSeek V3</td><td>38.0</td><td>39.0</td><td>42.0</td></tr><tr><td>Gemma 3</td><td>11.0</td><td>10.0</td><td>15.0</td></tr><tr><td>LLaMA 4</td><td>57.0</td><td>68.0</td><td>70.0</td></tr><tr><td>Grok 4.1</td><td>61.0</td><td>63.0</td><td>63.0</td></tr></table>

LC = LangChain, LG = LangGraph, LI = LlamaIndex. Values are percentages.

Throughout the rest of our work, LangGraph and LlamaIndex organized the same RAG task following different pipeline structures. LangGraph pipeline modeled the process as a state-based workflow to store retrieved contexts and generated answers as a part of a shared RAG state. The retrieval and generation were executed as separate nodes that were dependent directly on the context returned by the retriever. On the other hand, LlamaIndex connected the documents to a Chroma vector space and queried through a query-engine following an index-based approach. In both cases, the pipelines followed similar retrieval to generation logic, but the implementation difered in how context was retrieved, passed and used during answer generation processes.

## V. RAG Pipeline: ChromaDB, Embeddings, and Retrieval

Our RAG pipeline used ChromaDB as the vector store since the retrieval stage required semantic similarity search rather than conventional keyword lookup. ChromaDB stores document chunks as embeddings along with their corresponding documents and metadata making it suitable for RAG-based AI workflows where query embeddings must be matched with semantically relevant stored passages [27]. In our pipeline, the query was embedded at runtime and compared against the stored chunk embeddings so that the most relevant contexts could be retrieved and passed as ground truth for answer generation. The knowledge base was prepared from the rag\_database.csv file, where each entry contained a title and a short twoline contextual description. Each row was converted into a document by combining the title with its corresponding context before chunking and embedding. A sample of the dataset structure is shown below:

Table VI  
Sample Bangla Titles with Contextual Descriptions
<table><tr><td colspan="2">Title</td><td>Context</td><td>Language</td></tr><tr><td colspan="2">TET Bangladesh</td><td>G Bangladesh is an independent sovereign state in South Asia. The constitutional name of Bangladesh is the People&#x27;s Republic of Bangladesh.</td><td>bn</td></tr><tr><td>③司 2 Airport</td><td>Osmani International is located in Sylhet.</td><td>G GR R GR Osmani International Airport is one of the major international airports in Bangladesh. It</td><td>bn</td></tr><tr><td>2R可 The Daily Prothom Alo</td><td></td><td>R8 22月2利8羽1 Prothom Alo is a daily newspaper published in Bengali from Bangladesh. It was first published on 4 November 1998.</td><td>bn</td></tr></table>

The documents were further split into smaller chunks before embedding. For embedding generation, we used intfloat/multilingual-e5-large-instruct, a multilingual embedding model designed for text retrieval tasks and released as a part of the Multilingual-E5 embedding family [28] . Separate ChromaDB directories were maintained for each framework implementation in order to maintain framework-specific indexing behavior independent of each other aligning with reproducible research practices [29].

## VI. Result Collection

## A. Initial Answer Generation

Answer generation was done by api calls and the same prompt for generating answers were translated to their respective languages. In order to evaluate the baseline knowledge of the models in each language, and their ability to utilize contextual information and also evaluate the change in performance, we generate answers in two stages. At first, answers were generated without providing context to the models (non-contexted answer) and then by providing context (contexted answers). Similarly answer generation was done by api calls and the same prompt for generating answers were translated to their respective languages. In order to evaluate the baseline knowledge of the models in each language, and their ability to utilize contextual information and also evaluate the change in performance, we generate answers in two stages. At first, answers were generated without providing context to the models (non-contexted answer) and then by providing context (contexted answers). The prompts used in this case was in english as followed for Non-contexted answer generation. “According to the question ‘{question}’, provide only the direct correct answer to the question. Try to limit the answer to one word, and under no circumstances should you use more than five words. Do not add any explanation, full sentence, punctuation, extra words, introduction, or conclusion—only the final answer should be given in the output.” and for contexted answer generation the prompt was- “According to the question ‘{question}’ and the given context ‘{context}’, provide only the direct correct answer to the question. Try to limit the answer to one word, and under no circumstances should you use more than five words. Do not add any explanation, full sentence, punctuation, extra words, introduction, or conclusion— only the final answer should be given in the output.” these two prompts were translated into respective languages.

## B. RAG-Based Answer Collection

RAG-based answer collection was conducted after the final selection of LangGraph and LlamaIndex. For each language, model, and framework combination, the generated questions were processed through the corresponding RAG pipeline, and the model outputs were saved for evaluation. The same answer-generation instruction was used across all settings, with language-specific translations to maintain consistency.

During the answer generation stage, the input question was embedded and matched against the corresponding ChromaDB vector store, and the top retrieved chunks were passed to the LLM along with the question. The prompt was designed to produce short title-like answers rather than explanatory responses. Since the questions were constructed so that the expected answer was the corresponding title, each model was instructed to return only the final answer. The collected outputs were stored with their language, model, framework, question, retrieved context, generated answer, and reference title. The following prompt was used in respective languages to generate RAG-Based Answer for each language- ”You are a {Language} cultural question-answering assistant. Answer the question from the context below. Give the answer in {Language}. Limit the answer to one word, and under no circumstances should you use more than five words; there is no need to write anything extra or a complete sentence. If the answer is not found in the context, say: “Information not found.” Context: {context} Question: {question}.”

## VII. Model

## A. Question Generation

Qwen 3 32B by Alibaba is used exclusively for question generation. With 32 billion parameters and support for over 100 languages, it operates in two modes: a structured ”thinking” mode for carefully constructed prompts and a faster ”non-thinking” mode for fluent question variations. Its 131,072-token context window enables diverse, contextaware question creation across topics and languages.

## B. Answer Generation

Four models handle answer generation, each bringing distinct strengths. Meta Llama 4 Maverick 17B is a multimodal open-source model with 400 billion parameters and 128 experts, handling both text and image inputs with a 128K context window, excelling in natural language processing and long-context applications. DeepSeek Chat V3-0324 is a massive 685-billion-parameter mixture-ofexperts model with a 163,840-token context window, built for high-performance natural language understanding and complex dialogues. Google Gemma 3 27B supports visionlanguage input and text output across 140+ languages, performing strongly in math, reasoning, and structured outputs with a 128K context window and function calling capabilities. xAI Grok 4.1 Fast is a closed-source agentic model featuring a remarkable 2-million-token context window, optimized for customer support and deep research with an optional reasoning mode for enhanced decisionmaking.

## C. Translation

OpenAI GPT-OSS 120B is an open-weight, 117-billionparameter mixture-of-experts model used solely for translation across five languages. With a 131K context window, native MXFP4 quantization for eficient inference, and support for function calling and structured outputs, it brings high-reasoning capacity to multilingual translation workflows.

## VIII. Result and Analysis

## A. Result Analysis

![](images/55b1cdff3ddc975d76f9438ab9c1220bea63d106534da1443b56bfb0f105129f.jpg)  
For each language, groups of three columns signify Meta Llama 4, Google Gemma 3, Deepseek v3 0324, and Grok 4.1 Fast respectively  
Fig. 2. Overall accuracy comparison across non-contexted, contexted, and best RAG settings for five languages. Model groups within each language represent Meta Llama 4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 Fast, respectively.

The overall accuracy results show a clear gap between non-contexted and context-supported answer generation. As shown in Fig. 2, both contexted answering and best RAG perform substantially better than non-contexted answering across all five languages. Meta Llama 4 Maverick 17B generally achieves the strongest performance across the evaluated settings, while DeepSeek V3-0324 and Grok 4.1 Fast show weaker performance depending on the language and answering condition. However, several language- and category-specific anomalies are observed. In Nepali, Google Gemma 3 27B performs weakest in the general non-contexted settings, while DeepSeek V3- 0324 produces the weakest performance for overall nonpopular titles and non-cultural non-popular titles under non-contexted answering. Another exception appears in culturally grounded non-popular contexted answering, where Grok 4.1 Fast performs better than Meta Llama 4 Maverick. Similar deviations are also visible in Hindi and Urdu. In Hindi cultural non-contexted answering, Google Gemma 3 27B performs weakest instead of Grok. In Urdu, Gemma performs weakest for popular culturally grounded non-contexted answering and popular non-cultural contexted answering.

![](images/f68fc5e05e503bf5681e4e5f57eb896b8f3ae4ae5d9dd1c733a21c2733c21304.jpg)  
Fig. 3. Accuracy comparison across cultural and non-cultural question categories under contexted and best RAG settings for five languages. Model groups within each language represent Meta Llama 4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 Fast, respectively.

The cultural subdivision results in Fig. 3 show that performance does not vary uniformly between cultural and non-cultural questions. In several languages, cultural questions achieve comparable or higher accuracy than non-cultural questions, while in other cases the opposite pattern is observed. This indicates that cultural relevance alone does not determine answer-generation accuracy. Instead, the results vary depending on the language, model, and type of context used. The figure also shows that RAG does not consistently improve both cultural and noncultural categories for every model, which supports the need for language-wise and model-wise analysis.

![](images/ace93c56a87600d726f762e165aa71ac4eb676ba1adc3c8ebb615b994c89c7a0.jpg)

Fig. 4. Accuracy comparison for popular titles across cultural and non-cultural question categories under contexted and best RAG settings. Model groups within each language represent Meta Llama 4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 Fast, respectively.  
![](images/eb72afc76fc7dea6ed677783a5a89a68295a26b9b61350cf782a9ecca5ad49c2.jpg)  
Fig. 5. Accuracy comparison for non-popular titles across cultural and non-cultural question categories under contexted and best RAG settings. Model groups within each language represent Meta Llama 4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 Fast, respectively.

The popular and non-popular title comparisons in Fig. 4 and Fig. 5 show that popular titles generally achieve higher accuracy than non-popular titles. This trend is visible across most languages and categories, suggesting that title popularity afects answer correctness. However, the figures also show some model-level variations. For example, in some non-popular settings, the gap between contexted answering and best RAG becomes larger, indicating that retrieval support is more useful when the baseline performance is weaker. For RAG-based answering, the overall model ranking remains similar to the preliminary answergeneration results. Meta Llama 4 Maverick 17B remains the strongest model in most settings, while the weaker model varies across languages and categories. The comparison between LangGraph and LlamaIndex shows only marginal diferences, with LlamaIndex performing slightly better in some cases. Therefore, the main performance diferences are more strongly associated with language, model choice, cultural subdivision, and title popularity than with the specific RAG framework.

## IX. Discussion

The first research question examines whether RAG afects culturally grounded and non-cultural questions diferently in the multilingual setup. As shown in Figure 6, the efect of RAG on culturally grounded questions varies across languages and models. Bangla and Tamil show the clearest RAG advantage, with most models improving over contexted answering. In contrast, Hindi and Nepali are more context-favored, while Urdu shows a mixed pattern. This indicates that culturally grounded questions do not automatically benefit from retrieval. Instead, the improvement depends on the language, model, and relevance of the retrieved evidence.

![](images/4ec6cba8a23e4b65af8d2be746084eee6d306eb6865645b5ca0bd6c0a1ada9f0.jpg)

Fig. 6. Percentage-point improvement of best RAG performance over contexted answering for culturally grounded questions across five languages and four models.  
![](images/1a9728ec00f1600405f07ed6ae0869b79ab99efa70a6264b4e67aad498bb894d.jpg)  
Fig. 7. Percentage-point improvement of best RAG performance over contexted answering for non-cultural questions across five languages and four models.

Figure 7 shows that the non-cultural category follows a related but not identical pattern. Bangla shows the most consistent RAG improvement, while Urdu performs better with RAG for most models. However, Hindi and Nepali remain largely context-favored, and Tamil becomes less consistent compared to its cultural-question performance. Therefore, the findings for the first research question suggest that RAG is not simply category-sensitive. Its efect is both language-sensitive and model-sensitive, with stronger cultural-question benefits mainly observed in Bangla and Tamil.

![](images/4d93e9ae8a5a7b6bb1e02767094fb819f5a62d2a14c1edb1960d5b987b1e2071.jpg)  
Fig. 8. Overall accuracy comparison between contexted answering and best RAG performance across five languages.

The second research question evaluates whether RAG achieves higher overall accuracy than contexted answering. Figure 8 shows that RAG produces higher average accuracy than contexted answering in Bangla, Urdu, and Tamil. However, contexted answering remains stronger in Hindi and Nepali. This means that RAG cannot be treated as an automatically superior method for multilingual QA. Retrieval improves performance when the retrieved passages are useful and well aligned with the question, but it may underperform when the retrieved context is weaker than the directly supplied context.

![](images/7ae9449b3d53df928e53d3e1a01a8786e3e92365d7c349f08d7709decfe4c512.jpg)  
Fig. 9. Average accuracy across five languages for non-contexted answering, contexted answering, LangGraph RAG, and LlamaIndex RAG.

The third research question compares the four answering settings: non-contexted, contexted, LangGraph RAG, and LlamaIndex RAG. As shown in Figure 9, noncontexted answering produces the lowest average accuracy across all five languages. This shows that the models’ internal parametric knowledge alone is not suficient for reliable answer generation in this South Asian multilingual QA task. Both directly supplied context and retrievalsupported context improve performance substantially. Although prior work has shown that long or excessive context can reduce LLM performance, our results show that, in this setup, the absence of external information is the larger limitation.

RAG Gain for Popular and Non-popular Culturally Grounded Questions  
![](images/a12988b29d499fa2c0b9a234ef590134f3a05794f4d55263fd523440579ae7fd.jpg)  
Fig. 10. RAG gain over non-contexted answering for popular and non-popular culturally grounded questions across five languages.

The fourth research question examines whether RAG benefits popular or non-popular culturally grounded questions more. Figure 10 shows that non-popular culturally grounded questions receive higher RAG gain than popular culturally grounded questions across the evaluated languages. This does not mean that non-popular questions always achieve the highest final accuracy. Rather, it shows that retrieval provides greater relative support when the model starts from a weaker non-contexted baseline. Popular cultural questions appear to be better represented in the models’ internal knowledge, so the additional improvement from retrieval is smaller. Overall, the findings suggest that RAG should be treated as a context-dependent enhancement rather than a universally superior answering strategy. It reduces knowledge gaps, especially for weaker baseline settings and non-popular cultural questions, but it does not consistently outperform direct contexted answering across all languages. The efectiveness of RAG in multilingual cultural QA depends on retrieval quality, language-specific data alignment, and the ability of each model to use retrieved evidence correctly.

## X. Conclusion

This study found that LLMs perform better in multilingual cultural knowledge evaluation when supported by retrieved contextual information than being asked to answer from its parametric knowledge base. Earlier experiments have shown that modern LLMs often struggle with culturally nuanced questions when no prior contextual knowledge is provided, especially for less popular or regionally specific languages. LLM-assisted classification helped organize the titles into structured categories, while the later RAG experiments showed that retrieved context could improve answer generation across the multilingual dataset. Overall, the study suggests that multilingual cultural evaluation benefits from combining structured semantic labeling with retrieval-supported generation, especially for short, context-dependent titles and culturally specific questions.

## Acknowledgment

We express our gratitude and gratefulness to our supervisor, Dr. Farig Yousuf Sadeque for his guidance and support throughout our thesis. We are also grateful to our families and friends who have provided constant encouragement and support.

## References

[1] S. Minaee et al., “Large language models: A survey,” arXiv preprint arXiv:2402.06196, 2024. [Online]. Available: https:// arxiv.org/abs/2402.06196

[2] A. Brown, M. Roman, and B. Devereux, “A systematic literature review of retrieval-augmented generation: Techniques, metrics, and challenges,” arXiv preprint arXiv:2508.06401, 2025. [Online]. Available: https://arxiv.org/abs/2508.06401v1

[3] S. Amirshahi, A. Bigdeli, C. L. A. Clarke, and A. Ghenai, “Evaluating the robustness of retrieval-augmented generation to adversarial evidence in the health domain,” arXiv preprint arXiv:2509.03787, 2025. [Online]. Available: https://arxiv.org/ abs/2509.03787

[4] Y. Bang et al., “HalluLens: LLM hallucination benchmark,” 2025.

[5] P. Lewis et al., “Retrieval-augmented generation for knowledgeintensive NLP tasks,” arXiv preprint arXiv:2005.11401, 2020. [Online]. Available: https://arxiv.org/abs/2005.11401

[6] M. Cheng et al., “A survey on knowledge-oriented retrievalaugmented generation,” arXiv preprint arXiv:2503.10677, 2025. [Online]. Available: https://arxiv.org/abs/2503.10677

[7] S. Wu et al., “Retrieval-augmented generation for large language models: A survey,” 2024.

[8] A. Sharma et al., “A survey of retrieval-augmented generation systems,” 2025.

[9] S. Roychowdhury et al., “Metrics for evaluating retrievalaugmented generation,” 2024.

[10] X. Yang et al., “CRAG: Comprehensive RAG benchmark,” Advances in Neural Information Processing Systems, 2024.

[11] D. Edge et al., “From local to global: A graph RAG approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024.

[12] D. O. Opoku, M. Sheng, and Y. Zhang, “DO-RAG: A domain-specific QA framework using knowledge graphenhanced retrieval-augmented generation,” arXiv preprint arXiv:2505.17058, 2025.

[13] F. Zhang et al., “EraRAG: Eficient and incremental retrieval augmented generation for growing corpora,” arXiv preprint arXiv:2506.20963, 2025.

[14] R. Xu et al., “Collab-RAG: Boosting retrieval-augmented generation for complex question answering via white-box and blackbox LLM collaboration,” arXiv preprint arXiv:2504.04915, 2025.

[15] M. Eibich, S. Nagpal, and A. Fred-Ojala, “ARAGOG: Advanced RAG output grading,” arXiv preprint arXiv:2404.01037, 2024.

[16] T. Mahfuz et al., “Too late to train, too early to use? A study on necessity and viability of low-resource Bengali LLMs,” 2025.

[17] N. Chowdhury et al., “From facts to folklore: Evaluating large language models on Bengali cultural knowledge,” 2025.

[18] A. Yang et al., “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[19] Hugging Face, “meta-llama/Llama-4-Maverick-17B-128E-Instruct,” 2025, model card accessed Feb. 5, 2026.

[20] xAI, “Grok 4.1,” 2025. [Online]. Available: https://x.ai/news/ grok-4-1. Accessed: Feb. 5, 2026.

[21] Gemma Team et al., “Gemma: Open models based on Gemini research and technology,” 2024. [Online]. Available: https:// arxiv.org/abs/2403.08295

[22] DeepSeek-AI et al., “DeepSeek-V3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2412.19437

[23] OpenAI et al., “gpt-oss-120b & gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

[24] H. Chase, “LangChain,” 2022. [Online]. Available: https:// github.com/langchain-ai/langchain

[25] LangChain AI, “LangGraph: Building stateful, multi-actor applications with LLMs,” 2024. [Online]. Available: https:// github.com/langchain-ai/langgraph

[26] J. Liu, “LlamaIndex,” 2022. [Online]. Available: https://github. com/run-llama/llama\_index

[27] Chroma, “Chroma: The AI-native open-source embedding database,” 2025. [Online]. Available: https://github.com/ chroma-core/chroma

[28] L. Wang et al., “Multilingual E5 text embeddings: A technical report,” arXiv preprint arXiv:2402.05672, 2024. [Online]. Available: https://arxiv.org/abs/2402.05672

[29] G. K. Sandve, A. Nekrutenko, J. Taylor, and E. Hovig, “Ten simple rules for reproducible computational research,” PLoS Computational Biology, vol. 9, no. 10, 2013.