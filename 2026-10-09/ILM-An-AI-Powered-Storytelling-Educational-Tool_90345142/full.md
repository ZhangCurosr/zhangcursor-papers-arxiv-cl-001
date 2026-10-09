# ILM: An AI-Powered Storytelling Educational Tool

Suhaila Mohammed Abdelaziz Serour Suhaila Mohammed Abdelaziz Serour Allison Lahnala Allison Lahnala

Department of Computing and Software, Faculty of Engineering McMaster University, Hamilton, ON, Canada {suhai1, seroura, lahnalaa}@mcmaster.ca

## Abstract

Digital technologies have made Islamic narratives more accessible, but existing platforms provide limited support for structured learning and comprehension of these stories, particularly in Arabic and multilingual settings. We present ILM, an interactive educational platform for Stories of the Prophets that combines Arabic natural language processing, structured knowledge representation, and retrieval based question generation. Admin-approved Arabic narratives are processed by a Knowledge Graph (KG) Constructor Engine that identifies entities and narrative relationships and stores them as structured knowledge, enabling learners to explore stories through a visual story map and answer entity- and relation-based questions generated from the KG. Separately, a multilingual retrieval pipeline retrieves relevant passages from the original narratives to generate multiple-choice and open-ended comprehension questions. For open-ended questions, an LLM-as-a-Judge evaluates learners’ answers against the retrieved passages and reference answers to determine correctness. The platform also incorporates Quranic content as a separate enrichment layer, allowing selected narratives to be supplemented with source-supported information. By combining structured knowledge with passage-based retrieval, ILM supports narrative exploration, comprehension, and assessment across Arabic and multilingual content. The system demonstrates the feasibility of combining structured knowledge representation and retrieval-based generation to support interactive learning of Islamic narratives. A demo is available at anonymous.4open.science/r/mml-5FCF.

## 1 Introduction

Knowledge, or ilm, holds an important position in Islam and is highly valued by Muslims. Islamic knowledge has historically been passed from one generation to the next through teaching, study, narration, and direct engagement between teachers and learners. Narration has an important role in Islamic knowledge transmission, particularly through a framework and method called qasas (Qur’anic narratives). The Qur’an uses narrative to convey guidance and lessons, highlighting its pedagogical significance within Islamic education [27].

In today’s world, the way knowledge is transmitted has changed significantly with the rise of digital technologies where information is now more widely available and accessible than before. On the other hand, digital platforms have become an increasingly common medium through which people learn. However, the availability of information through digital platforms does not necessarily ensure meaningful engagement or learning. Recent studies have shown that digital storytelling of prophetic narratives can support moral and religious character formation, as well as learner motivation and engagement, particularly when stories are well designed and appropriately mediated by adults [4, 20, 21].

Therefore, we developed a storytelling platform for children and young adults that combines reading, visualization, and interactive learning. We begin by focusing on prophetic stories. Through an interactive story map, users can visualize key entities in each story, such as the prophets and other people, locations, and important objects, and explore the relationships between them. They can also evaluate their understanding of the stories through quizzes that assess both factual knowledge and story comprehension.

## 2 Related Work

Existing apps and AI systems: A range of apps and AI systems have risen to support the learning and exploration of prophetic and Islamic stories. Apps such as Qissah[6] and Qalam[5] provide 40+ narrated Qur’an and prophet stories with professional voiceover and multilingual support, while platforms such as Thurayya[17], Miraj Stories[16], Quran Stories 4 Kids [23], and Kisah Nabi [22] combine animated or interactive stories, audio narration, and games to present s¯ırah and qis<sub>.</sub>as<sub>.</sub> al-anbiya’ in engaging formats. Conversational AI has also begun to be used in Islamic education,¯ with general-purpose systems such as ChatGPT [19] and specialized systems such as MuslimGPT [28], TheoAI [15], providing sourced answers grounded in the Qur’an and authenticated hadith.

Knowledge graph and RAG systems Knowledge graphs (KGs) have been used in Islamic texts to structure entities and relations in Qur’anic and hadith corpora for semantic search and question answering [13, 26, 25]. More generally, narrative-centric KGs represent events, entities, and temporal or causal links to support story understanding and interactive exploration [7, 18, 8]. Retrievalaugmented generation (RAG) frameworks further combine LLMs with retrieved context, with recent work exploring temporally coherent passage assembly for narrative QA [14].

Despite these developments, most work remains English-focused, with limited support for Arabic and educator control. We address these gaps through Arabic- and English-based learning that enables learners to explore narrative structure and relationships, assess factual and comprehension-based knowledge, receive story-grounded feedback, and learn from educator-controlled content.

## 3 System Architecture and Design

The platform comprises end-user and administrative web portals built with React.js and connected to a FastAPI backend. PostgreSQL serves as the primary data store, with pgvector supporting vector storage and similarity search. The backend integrates a KG Constructor Engine (KG Engine) for structured knowledge extraction from Arabic narratives approved by educators with admin privileges, and a multilingual retrieval pipeline for question generation and answer evaluation. Figure 1 presents the overall architecture.

## 3.1 Knowledge Graph Constructor Engine

The KG Engine constructs a story-specific KG from Arabic paragraphs through named entity recognition (NER), story-scoped entity linking, candidate pair generation, relation extraction, and semantic resolution. Extracted entities and relations are converted into structured KG records.

Each paragraph is processed using a four-expert nested NER ensemble comprising AraBERTV2 [3], CAMELBERT-Mix[12], MARBERTV2[1], and an AraBERTv2 span-based expert. An entity span is retained when at least two experts produce an exact-span match. XLM-R[10] is additionally used as a complementary safe-addition component for mentions not sufficiently supported by the primary ensemble. The resulting mentions are passed to a story-scoped Islamic entity linker, which maps surface forms, aliases, titles, and pronouns to canonical entities within the story.

After entity linking, candidate entity pairs are passed to a two-stage relation extraction pipeline. An AraBERT-based binary classifier first identifies related pairs, followed by a relation classifier using the WOJOOD relation scheme[2]. The top-5 relations are then evaluated by an NLI model using the original text as the premise to resolve semantic ambiguities, such as birth place versus lives in. Selected relation types are further refined by relation-specific resolvers for family, prophetic mission, and major events before being added to the knowledge graph.

The extraction pipeline operates over the broader WOJOOD ontology[2], while the educational application exposes nine controlled predicates: child\_of, father\_of, sibling\_of, sent\_to, thrown\_into, taken\_to, imprisoned\_in, appointed\_over, and reunited\_with. Although the KG Engine extracts relations from the full WOJOOD ontology[2], only nine selected relations are used in the educational application for learner activities and deterministic assessment.

![](images/dd8acf5a153c8d969331aa744986a38367cee66ed44953d72925baedf5d6024a.jpg)  
Figure 1: Overall architecture of the proposed ILM platform.

The extracted entities, mentions, and relations are converted into structured knowledge graph records before being stored in database. Each relation is validated against its entity endpoints and stored together with information about its source and extraction process. This includes the supporting paragraph, source information, original relation prediction, and, where applicable, the semantic resolution rule used to obtain the final relation. The graph is maintained in synchronization with the approved narrative content. When a paragraph is modified, the KG records previously associated with that paragraph are retracted and the updated paragraph is processed again.

A separate Quran enrichment layer is derived from the Tanzil Quran Text–Simple Clean v1.1 (CC BY 3.0) [24]. Relations are manually verified against the Quranic source and stored with their corresponding verses. This layer adds relations such as interpreted\_for, tempted, and summoned, but remains separate from the nine-predicate educational ontology and is excluded from deterministic quiz generation. Entity representation is conservative and limited to information explicitly supported by the source, distinguishing verified Quranic knowledge from automatically extracted narrative knowledge.

## 3.2 Multilingual Retrieval and RAG Pipeline

The retrieval pipeline operates independently of the KG Engine and supports passage-based openended question generation and evaluation. Each approved narration is divided into paragraphs, with approved translations maintained across languages. Following human review, each paragraph translation is indexed as a retrieval chunk, preserving a direct mapping between retrieved content and its source paragraph. Chunks are embedded using Cohere’s embed-multilingual-v3.0 model[9], producing 1024-dimensional representations stored in PostgreSQL with pgvector and an HNSW index for efficient cosine-similarity search. Arabic text is normalized before embedding and retrieval to reduce surface-form variation.

For question generation, the administrator specifies the story, language, question type, and number of questions. The query is embedded and matched against the chunk embeddings, with the top k = 3 chunks provided to Gemini 3.6 Flash[11] as source context. The model is instructed<sup>1</sup> to generate questions solely from the retrieved passages. Post-generation checks verify answer support against the source passages, detect within-chunk duplicates using Jaccard word overlap (≥ 0.6), and flag potentially redundant questions across the story using embedding similarity (> 0.78) for manual review. Duplicate generation is retried once before the question is discarded.

## 3.3 Question Evaluation

The end-user portal combines two assessment mechanisms: knowledge-graph-based questions and retrieval-based open-ended questions. After completing all paragraphs of a story, the learner can attempt the corresponding quiz. The two question types use different evaluation strategies because they rely on different representations of the underlying knowledge.

## 3.3.1 Knowledge-Graph-Based Evaluation

Knowledge-graph-based questions use the story knowledge graph as the source of truth. Questions are generated from graph triples representing entities and relations, with each question linked to its source triple and expected answer. This enables deterministic evaluation without a language model: answers are normalized and matched against the canonical graph value and registered aliases, while multiple-choice answers are determined directly from the corresponding graph entity or relation.

## 3.3.2 Retrieval-Based Answer Evaluation

For retrieval-based open-ended questions, the learner’s question is embedded and used to retrieve the top k = 3 relevant chunks from the same story and language. The retrieved passages, together with the learner’s answer and the reference answer, are provided to Gemini[11] for evaluation. The retrieved passages serve as the evidence available to the evaluator, ensuring that the assessment remains grounded in the approved narrative content. Gemini[11] determines whether the learner’s answer is correct based on the retrieved evidence while allowing for reasonable differences in phrasing, spelling, and grammar. When the retrieved passages do not contain sufficient information to establish that an answer is incorrect, the system follows a benefit-of-the-doubt policy rather than penalizing the learner based on information unavailable in the retrieved context. The model returns a correct or incorrect and learner-facing feedback.

## 4 Conclusion

We presented ILM, an interactive educational platform that combines structured knowledge representation and multilingual retrieval to support learning from Islamic narratives. ILM establishes a framework for connecting narrative content with interactive exploration and comprehension assessment while maintaining links to the underlying source material. This work provides a foundation for extending Arabic and multilingual educational applications and for future empirical evaluation with learners and educators. The platform can support accessible and interactive learning of Islamic narratives. However, automatically extracted relations and generated questions may contain errors, which we mitigate through approved narrative content, human review, and source-verified Quranic enrichment. A demonstration video showcasing the ILM platform and its interactive learning capabilities is available here: demo video.

## 5 Future Work

Several directions can further extend the platform. First, we plan to expand the story corpus to include a wider range of prophetic narratives and evaluate the system across different texts, as the current study is limited by its evaluation on a single story. Second, we aim to conduct formal empirical studies with children and educators to measure the efficacy of the platform in improving narrative comprehension, engagement, and knowledge retention. Third, we could also further improve engagement by incorporating multimedia such as GIFs, animations, and short videos, along with more interactive assessment activities such as matching characters and relationships, ordering events, and completing story maps.

## Acknowledgments and Disclosure of Funding

Computing and API costs were supported by a start-up fund awarded to Allison Lahnala by McMaster University. The authors declare no competing interests. The views expressed are those of the authors and do not necessarily reflect the views of McMaster University.

## References

[1] Muhammad Abdul-Mageed, AbdelRahim Elmadany, and El Moatez Billah Nagoudi. Arbert and marbert: Deep bidirectional transformers for arabic. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 7088–7105. Association for Computational Linguistics, 2021.

[2] Alaa Aljabari, Mohammed Khalilia, and Mustafa Jarrar. Wojood<sup>Relations</sup>: Arabic relation extraction corpus and modeling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 34342–34360. Association for Computational Linguistics, 2025.

[3] Wissam Antoun, Fady Baly, and Hazem Hajj. Arabert: Transformer-based model for arabic language understanding. In Proceedings ofthe 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection, pages 9–15. European Language Resources Association, 2020.

[4] Yahya Aziz and Idris Afandi. Digital storytelling of the stories of the prophets for developing religious character in early childhood. GHAITSA: Islamic Education Journal, 7(2):169–179, 2026.

[5] Baris Bingor. Qalam: Quran, deen & stories. iOS App, 2025. 40+ animated prophet stories; Quranic AI buddy. See https://apps.apple.com/app/qalam-quran-deen-stories/id6744072027.

[6] Baris Bingor. Qissah: Quran stories & ai. iOS App, 2025. 40+ narrated Quran and prophet stories; AI Islamic assistant. See https://qissahapp.com.

[7] Inès Edith Laurent Blin. Narrative Understanding with Knowledge Graphs. PhD thesis, Vrije Universiteit Amsterdam, 2026.

[8] Yi-Chun Chen. Hierarchical knowledge graphs for story understanding in visual narratives. In Interactive Storytelling—18th International Conference on Interactive Digital Storytelling, ICIDS 2025, Proceedings, volume 16374 of Lecture Notes in Computer Science, pages 198–218. Springer, 2026.

[9] Cohere. Cohere embed models. https://docs.cohere.com/docs/cohere-embed, 2024. Accessed September 14, 2026.

[10] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451. Association for Computational Linguistics, 2020.

[11] Google. Gemini 3.6 flash, 2026.

[12] Go Inoue, Bashar Alhafni, Nurpeiis Baimukan, Houda Bouamor, and Nizar Habash. The interplay of variant, size, and task type in arabic pre-trained language models. In Proceedings of the Sixth Arabic Natural Language Processing Workshop. Association for Computational Linguistics, 2021.

[13] Amna Binte Kamran, Nigar Azhar Butt, and Amna Basharat. Semantic enrichment of hadith corpus—knowledge graph generation from islamic text. Semantic Web, 17(2):1–27, 2026.

[14] Byeongjeong Kim, Jeonghyun Park, Joonho Yang, and Hwanhee Lee. Chronological passage assembly for retrieval-augmented generation in narrative question answering. Computers, Materials & Continua, 88(3), 2026.

[15] MindHYVE.ai. Theoai: Evidence-based islamic q&a. Online, 2026. Agentic AI grounded in Quran and authenticated hadith with full citations. See https://www.theogrid.ai.

[16] Miraj Audio. Miraj stories: Quran for kids. Android App, 2025. Islamic stories, prophets’ stories, animations, audiobooks. See https://play.google.com/store/apps/details?id=com.mirajaudio.mirajstorybook.

[17] Mohammad Shaker. Thurayya: Quran for kids. iOS App, 2025. Prophets’ stories, interactive narratives, AI recitation feedback. See https://apps.apple.com/app/thurayya-quran-forkids/id1571034432.

[18] Brian Keith Norambuena. Interactive narrative analytics: Bridging computational narrative extraction and human sensemaking. arXiv preprint arXiv:2601.11459, 2026.

[19] OpenAI. Chatgpt. Online, 2026. General-purpose conversational AI used for Islamic Q&A. See https://chat.openai.com.

[20] Reni Prastiwi. Digitalization of the prophet’s story to strengthen islamic values from an early age. Jurnal Teknologi Pembelajaran, 1(1):1–12, 2024.

[21] Faiz Azizi Rohman and Ach. Nurfuad Al-Fajri. Learning with light: Integrating islamic character education into multimedia storytelling at rural elementary schools. International Journal of Instructional Technology, 4(1):1–14, 2025.

[22] Solite Kids. Kisah nabi: Cerita anak muslim. Android App, 2025. Stories of the Prophets for kindergarten and elementary. See https://play.google.com/store/apps/details?id=com.solitekids.kisahnabi.

[23] Spark / MWM. Quran stories 4 kids: Interactive prophet stories & games. Mobile App, 2026. 20+ interactive prophet stories in 16 languages. See https://spark.mwm.ai/en/apps/quran-stories-4-kids-prophets/1578082367.

[24] Tanzil Project. Tanzil Quran Text (Simple-Clean, Version 1.1), 2021. Creative Commons Attribution 3.0 License.

[25] Shasha Arzila Tarmizi and Saidah Saad. Named entity recognition on islamic texts: A systematic review. International Journal ofAdvanced Science Computing and Engineering, 7(3):98–111, 2025.

[26] Kemas Rahmat Saleh Wiharja, Danang Triantoro Murdiansyah, Muhammad Zakiyullah Romdlony, Tiwa Ramdhani, and Muhammad Ramadhan Gandidi. A questions answering system on hadith knowledge graph. Journal ofICT Research and Applications, 16(2):184–196, 2022.

[27] Muhammad Fawwaz Bin Muhammad Yusoff. Narrative pedagogy for spiritual empowerment: A typological figuration of qa¸sa¸s in the quran. JRLA Journal of Peace Education and Islamic Studies, 6(2):51–60, 2023.

[28] Zaid Koradia. Muslimgpt: Authentic islamic ai. Mobile App, 2026. AI-powered answers to Islamic questions with verifiable sources. See https://mwm.ai/apps/muslimgpt-authenticislamic-ai/6762242577.

## A Gemini 3.6 Flash Prompts

## A.1 Open-Ended Question Generation Prompt

You write reading-comprehension questions for children aged age.

Use ONLY the SOURCE provided below. Do not add names, events, places, or details that are not present in the SOURCE. If the SOURCE does not provide enough information to answer a question, do not ask about it.

Write ONE open-ended question that requires the child to answer in their own words, using one or two sentences. The question must be answerable from the SOURCE alone.

Ask about the story itself, not about the writing or the source. Do not use phrases such as according to the text” or the story states.”

Also provide the expected answer as one short sentence that a child might reasonably write.

Write everything in language.

## avoid

Return only valid JSON in the following format:

```json
{"prompt": "...", "reference_answer": "...", "explanation": "..."}
```

## A.2 Open-Answer Judge Prompt

A child aged age was asked a question about a story and wrote an answer.

Decide whether the child’s answer correctly answers the QUESTION using ONLY the PASSAGES provided below. Do not use any outside knowledge or information about the story.

If the PASSAGES do not provide enough information to determine whether the answer is correct, give the child the benefit of the doubt and mark the answer as correct.

Be generous about wording, spelling, grammar, and sentence structure. The child does not need to use the same words as the expected answer. Mark an answer as correct when its meaning is consistent with the PASSAGES and it demonstrates understanding of what the question asks.

Mark an answer as incorrect only when:

• it contradicts information in the PASSAGES;

• it gives information that is incompatible with the PASSAGES; or

• it fails to answer the main point of the QUESTION.

Do not require every detail from the reference answer if the child’s answer is otherwise correct. Do not mark an answer wrong merely because it is incomplete in a minor way, uses different wording, or contains harmless spelling or grammar mistakes.

Write feedback for the child in language, using one short, encouraging sentence. If the answer is correct, briefly say what they got right. If the answer is incorrect, kindly provide the correct answer based only on the PASSAGES. Never mention passages, sources, texts, evaluation, or judging.

```jsonl
QUESTION: {question}
THEIR ANSWER: {answer}
{reference}
PASSAGES:
{passages}
{avoid}
Return only valid JSON:
{"is_correct": true, "feedback": "..."}
```

## A.3 Shared Block for Question Generation Prompt

Shared injected block (avoid)

These questions have already been asked about this same passage. Ask about something else in it, a different detail, person, action, or consequence. Do not rephrase or repeat any of them:

{existing}