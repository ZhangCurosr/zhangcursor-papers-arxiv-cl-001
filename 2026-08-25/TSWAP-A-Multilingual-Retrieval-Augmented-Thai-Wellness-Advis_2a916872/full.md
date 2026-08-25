# TSWAP: A Multilingual Retrieval-Augmented Thai Wellness Advisor

Pornthep Ukosaramig<sup>∗</sup> Digital Touch Point Co., Ltd. aun\_@hotmail.com

Kobkrit Viriyayudhakorn iApp Technology Co., Ltd. kobkrit@iapp.co.th

August 25, 2026

## Abstract

We present TSWAP, a deployed eight-language conversational wellness advisor grounded, via retrieval-augmented generation, in a verified knowledge base of Thai traditional medicine and certified wellness providers. An unmodified open-weight LLM (Qwen3.6-35B-A3B on vLLM) is grounded on a 30.6K-chunk Thai index by a hybrid dense–sparse retriever with cross-encoder reranking; a first-turn query classifier forces tool-based retrieval for entity lookups; a rule-based safety layer enforces medical scope and Thai emergency routing; and all eight languages are served zero-shot with translate-then-retrieve. We release the first Thai traditional-medicine/wellness retrieval benchmark (50 questions with gold document IDs; Recall@5 = 0.88), production QA logs (91.1% test–retest pass over 259 cases), and a 71-question frontier no-retrieval probe showing what each grounding pillar contributes: without the safety prompt the backend model family produced a full drug-dosing schedule and complied with out-of-scope requests, and without the knowledge base it produced zero verifiable provider recommendations. We further report two transferable deployment findings: English-calibrated 4-bit AWQ quantization corrupts Thai tone marks, and forced-retrieval routing is necessary for reliable grounding.

Keywords: retrieval-augmented generation, Thai NLP, health chatbots, traditional medicine, LLM safety, benchmarks.

## 1 Introduction

Wellness tourism is among the fastest-growing segments of the global economy; Thailand’s wellness economy was valued at US\$40.5 billion ( 1.4 trillion THB) in 2023, with wellness-tourism spending of US\$12.3 billion [4]. Yet information about Thai wellness and beauty providers— services, packages, prices, certification status, and the traditional-medicine and herbal knowledge that diferentiates Thai wellness—remains fragmented across channels, with no trustworthy, verifiable, multilingual central source. Foreign visitors in particular face a language barrier and uncertainty about provider standards.

The Thai Smart Wellness Advisor Platform (TSWAP) was built to address this gap: a web, mobile, and chatbot platform that consolidates verified provider data and Thai traditionalmedicine knowledge and exposes it through a multilingual conversational advisor. This paper focuses on the conversational AI engine.

From an NLP standpoint, TSWAP occupies an empty niche. A survey of Thai and Southeast-Asian NLP (Section 2) finds a crowded field of general-purpose Thai LLMs [16, 17, 10, 13, 12, 2] and a small body of Thai clinical NLP [3], but no public NLP benchmark or dataset for Thai traditional or herbal medicine (แพทย์แผนไทย / สมุนไพรไทย), and only one prior Thai traditional-medicine language model—AppHerb [1], a Gemma-2 fine-tune that generates treatment and recipe text from two textbooks, with no deployment, no multilingual support, no retrieval grounding, and no reusable benchmark. The mature analogue is Traditional Chinese Medicine, which has a developed benchmarking subfield [9, 14]; Thai has essentially no equivalent.

## Contributions.

1. The first public Thai traditional-medicine/wellness retrieval benchmark (50 questions, gold document IDs, Recall@5 = 0.88), released together with production QA logs and frontier-probe logs (Section 4).

2. A grounding-first RAG recipe whose central element—a query classifier that forces retrieval for entity lookups—we show is necessary to prevent silent hallucination, with transferable serving findings: English-calibrated AWQ quantization corrupts Thai output, and hidden-reasoning budgets silently truncate answers (Sections 3.2, 4).

3. A production evaluation with a Thai-localized safety probe quantifying what the safety prompt and the knowledge base each contribute (Section 4.4).

TSWAP’s LLM is deliberately not fine-tuned and its safety layer is prompt-and-routing only;   
we treat these as design choices with measurable consequences.

## 2 Related Work

Thai and SEA LLMs. Open Thai/SEA foundation models include Typhoon [16, 17], OpenThaiGPT [10], SEA-LION [13], Sailor [12], and Chinda [2]. These are general-purpose; none targets traditional or herbal medicine. TSWAP uses an unmodified multilingual base model (Qwen3.6-35B-A3B) and invests in grounding rather than pretraining.

Thai medical and traditional-medicine NLP. Eir [3] is a Thai clinical LLM evaluated on clinical tasks; evaluations of frontier models on the Thai national medical licensing examination exist [15]. General Thai benchmarks (ThaiExam [16], M3Exam [7], WangchanThaiInstruct [19]) are exam- or general-domain. The only Thai traditional-medicine LM is AppHerb [1]. None evaluate open-ended, culturally specific herbal/wellness advisory quality, and none is deployed and multilingual.

Traditional-medicine benchmarks (analogue). Traditional Chinese Medicine has multiple benchmarks—MTCMB [9], TCM-Ladder [14]—which we take as templates for a future Thai herbal benchmark.

Multilingual health chatbots and evaluation. Cross-lingual health-QA studies show substantial non-English quality degradation [20]; multilingual health assistants have been studied in other low-resource settings [6]. For RAG we adopt faithfulness/relevance/context metrics [11] and medical-RAG evaluation practice [8]; for open-ended health-answer quality and safety we follow rubric-based grading and safe/problematic/unsafe labeling [5, 18]. These inform our proposed protocol (Section 4.6).

## 3 The TSWAP Engine

TSWAP is a production platform, publicly launched 28 June 2026 and generally available since August 2026 on the web and as Wellness Chatbot on Google Play and the App Store.<sup>1</sup> This paper concerns only its AI engine, which receives each user message with a data-minimized profile (wellness-relevant fields only; no direct identifiers, for PDPA compliance) and owns the retrieval tools (Section 3.2), the multilingual behavior (Section 3.3), and the safety prompt (Section 3.4).

Table 1: Live retrieval index (chunks per collection, 2026-07-15).
<table><tr><td>Collection</td><td>Chunks</td><td>Collection</td><td>Chunks</td></tr><tr><td>restaurants</td><td>11,212</td><td>thai_herbs</td><td>870</td></tr><tr><td>hotels</td><td>8,704</td><td>providers</td><td>484</td></tr><tr><td>attractions</td><td>8,169</td><td>staygold_catalogs</td><td>18</td></tr><tr><td>packages</td><td>1,153</td><td>wellness_facilities</td><td>15</td></tr></table>

Total  30,625 chunks / 8 collections

Table 2: Schema of a herb monograph in thai\_herbs.
<table><tr><td>Group</td><td>Fields</td></tr><tr><td>Identity</td><td>thai_name, english_name, scientific_name, category</td></tr><tr><td>Content</td><td>properties, benefits, usage_instructions, botanical_characteristics, pharmacology</td></tr><tr><td>Safety</td><td>contraindications</td></tr><tr><td>Evidence</td><td>scientific_references, research_references</td></tr><tr><td>Metadata</td><td>origin_region, usage_type, images</td></tr></table>

## 3.1 Knowledge base

The retrieval index is a single Milvus (v2.6.4) vector store holding eight collections (Table 1): provider and venue directories (restaurants, hotels, attractions, wellness facilities, providers), curated packages and catalogs, and—central to the paper’s novelty—a Thai herbal-medicine collection (thai\_herbs). As of 2026-07-15 the live index holds 30,625 chunks; venue collections are one chunk per entity, while thai\_herbs and attractions are multi-chunk, so the number of distinct herb monographs is smaller than the collection’s chunk count.

Each herb entry is a structured monograph (Table 2). Two field groups matter most for safe advisory behavior: contraindications (vulnerable-group cautions) and the two reference fields (surfaced as citations, Section 3.2). Ingestion is deterministic, with per-collection SHA-256 change detection, guarded blue–green index rebuilds, and geocoding enrichment for venues.

Provenance. The herbal collection is compiled from reference materials of the Department of Thai Traditional and Alternative Medicine (DTAM, กรมการแพทย์แผนไทยและการแพทย์ทางเลือก), Ministry of Public Health—Thailand’s authoritative body for Thai traditional medicine—and each entry carries structured scientific\_references/research\_references fields pointing back to source literature. Chunking is character-based with a maximum of 512 tokens per chunk and 50-token overlap.

## 3.2 Retrieval pipeline

Base model and serving. The generator is Qwen3.6-35B-A3B, a mixture-of-experts model ( 35B total / 3B active parameters), not fine-tuned, served self-hosted with vLLM behind an OpenAI-compatible endpoint; “thinking” mode is disabled in production. The deployed checkpoint is a community 4-bit AWQ quantization; we observed Thai token corruption attributable to English-calibrated AWQ (Section 4) and are migrating to the oficial FP8 checkpoint.

Hybrid retrieval. Retrieval is hybrid dense–sparse. Dense embeddings use BAAI/bge-m3 (1024-d, multilingual); sparse retrieval uses Milvus’s built-in BM25 over raw text. Per query, dense ANN (cosine) and BM25 (drop\_ratio\_search = 0.1) results are fused with Reciprocal Rank Fusion; 10 candidates are retrieved, the top 7 are reranked by a cross-encoder (Qwen/Qwen3-Reranker-8B), reranking is skipped when the top similarity  0.85, and the final top\_k is 3. Beyond semantic search, the engine applies structured filters—province, food type, and price ranges with schema normalization—and geospatial filtering: a nullable WKT GEOMETRY field with an RTREE index supports ST\_DWITHIN radius queries, and a near parameter resolves a place name to coordinates server-side.

Grounding-first routing. A first-turn LLM query classifier routes each message to one of five intents: wellness lookup, general advice, emergency, out-of-scope, or small talk. For wellness lookups the engine forces the retrieval tool via a named tool\_choice, guaranteeing that any entity lookup (herb, provider, spa, package, hotel, restaurant, place) calls rag\_search before the model answers. This fixes a concrete failure mode: with tool\_choice=auto, the model silently skipped retrieval on short noun-phrase queries and answered from parametric memory. The system prompt further constrains the model to recommend only items present in retrieval results, to never invent entities, and to say “no match found” rather than fabricate. Only pure general-lifestyle advice with no lookup target is answered without retrieval; the system is thus grounded-for-entities rather than strictly closed-book. For the longevity advisory variant, the prompt mandates a trailing “Citation:” block of 3–5 items drawn from the KB’s scientific\_references/research\_references fields.

## 3.3 Multilingual strategy

TSWAP serves eight languages—Thai, English, Chinese, Japanese, Malay, Russian, Korean, and Hindi—with one multilingual model and a single English system prompt; there are no per-language templates. The model replies in the language of the user’s most recent message. Because the knowledge base is Thai, retrieval uses a translate-then-retrieve strategy: the model translates the query to Thai before calling rag\_search, then answers in the user’s language (bge-m3’s multilinguality provides additional cross-lingual tolerance). All eight languages are served zero-shot; per-language output quality therefore rests entirely on the base model’s pretraining, which motivates the per-language evaluation we propose for the secondary languages (Malay, Russian, Korean) in Section 4.6.

## 3.4 Safety layer

The engine’s server-owned system prompt implements a rule-based safety layer with four elements:

• Scope. Wellness topics only (Thai herbs, nutrition, lifestyle, sleep, exercise, stress, providers, packages, wellness travel); anything else is politely declined and redirected.

• Medical safety. “You are NOT a doctor”: no diagnosis, no prescribing, no drug dosages; guidance is framed as general wellness; the model recommends consulting a qualified professional, explicitly for children, pregnant/breastfeeding women, the elderly, and people with chronic illness or on medication.

• Emergencies. Chest pain, breathing dificulty, severe bleeding, stroke signs, or fainting trigger an immediate instruction to call 1669; self-harm or suicidal ideation triggers an empathetic response with hotline 1323. Emergencies are answered immediately, bypassing retrieval (enforced both by the prompt and by the query classifier’s emergency category).

• Additive grounding. Safety rules are additive and never replace retrieval, preventing safety text from being used to dodge grounding.

Two server-side mechanisms complement the prompt: the query classifier (above), and a postgeneration Thai quality guard that detects corrupted Thai via combining-mark heuristics (mark-to-consonant ratio threshold 0.28 plus two hard signals) and regenerates once with official instruct-mode sampling. We stress that this is a quality filter, not a safety filter: there are no post-generation safety classifiers, regex filters, or blocklists. Herb–drug interaction risk is handled by deferral plus the KB’s contraindications field rather than a dedicated interaction checker. We report this plainly as a design choice and limitation: the harm categories addressed are exactly those encoded in the prompt and routing above (medical scope, emergencies, self-harm, vulnerable groups), and we make no coverage claim beyond them. No adversarial red-teaming has been performed to date; the safety suite of the production QA campaign (Section 4.3) is the only systematic safety testing so far, and we identify structured red-teaming as immediate future work.

## 4 Evaluation

## 4.1 Retrieval benchmark (released)

We release a 50-question Thai retrieval golden set:<sup>2</sup> 50 Thai questions grounded to gold document identifiers across the live collections (built 2026-07-15), scored by Recall@K with an accompanying harness, alongside a sample of 50 realistic user questions. On this set the deployed retriever attains Recall@5 = 0.88 (44/50), with per-collection weak spots on thai\_herbs and packages ( 0.75)—indicating that the herbal collection, the paper’s domain of interest, is also where retrieval most needs improvement. The repository also contains a before/after harness for the metadata/range-filter change (a constraint-violation@k metric), a regression gate, and a geospatial smoke suite.

## 4.2 Deployment evaluation

Four-week user acceptance testing (n = 120: foreign and Thai tourists, wellness providers, tourism operators) reported overall satisfaction 87.2%, user-assessed chatbot accuracy 86.5%, Thai herbal-knowledge accuracy 90.2%, and provider-information trust 91.0% (5-point Likert instrument—user-assessed, not expert-graded; the objective correctness evidence is Section 4.3). Load testing sustained 200 concurrent users at 0% error (platform-API latency  527 ms avg; this excludes LLM generation, which the QA campaign measured at 14.1 s median per answer), and a security assessment passed all 55 OWASP Top-10 (2025) cases.

## 4.3 Production QA campaign (answer correctness)

Separately from the UAT, the platform team maintains a regression test list of 429 cases in 15 groups covering the whole platform; the engine-correctness subset spans six groups: core wellness content (ENG), colloquial-Thai geospatial provider search (LOC; 28 sub-categories such as district/road/landmark/transit phrasing), multilingual behavior (LANG), conversational context (CTX), safety and scope (SAFE), and input robustness (INP). Each case specifies an expected behavior, and outcomes are graded pass / odd (anomalous) / fail against it, executed manually against the production web client.

In a second-round campaign (August 2026), the 259 cases that passed round one were reexecuted: 236 (91.1%) reproduced pass, 21 became odd, and 2 fail—a direct test–retest consistency measurement that also exposes sampling nondeterminism (the campaign log notes identical questions receiving diferent answers across runs). Per group, the safety suite passed 19/21, multilingual 13/14, input robustness 11/11, core wellness content 22/23, context 8/9, and geospatial search 163/181. A companion log tracks the 56 problematic cases across both rounds (round 1: 46 odd / 7 fail / 3 pass; round 2: 13 pass / 40 odd / 3 fail); 47 of 56 (84%) fall in the colloquial geospatial group, with documented failure modes of false proximity claims, unresolved landmarks and zones, and non-reproducible answers. End-to-end failures thus concentrate exactly where retrieval is structurally hardest—colloquial place-name grounding—consistent with the retrieval-stage weak spots of the golden set.

## 4.4 Frontier no-retrieval probe

Engine-side ablations (disabling retrieval or the safety prompt inside the deployed stack) require serving-side switches; as a first controlled measurement we instead probed the engine’s designated alternative backend family—Gemini (2.5 Flash, Vertex AI)—without retrieval, on 71 real questions drawn from the QA campaign (all 22 SAFE, all 23 ENG, 12 LOC, 14 LANG), in two conditions: vanilla (no system prompt) and +safety (a reconstruction of the TSWAP safety prompt of Section 3.4). Outputs were scored by scripted surface checks (hotline presence and position, referral/caution language, reply-language match) plus a manual review of every SAFE and LOC answer by the authors; we label this a probe, not a benchmark, and release the per-answer logs.<sup>3</sup>

The safety prompt changes safety-critical behavior. Vanilla, the frontier model answered the dosage question (กินพาราได้ครั้งละกี่เม็ด) with a complete paracetamol dosing schedule including pediatric doses, and complied with both out-of-scope requests (writing Python code; discussing politics): 3/22 clear violations of the expected safety behavior. With the safety prompt, 0/22: the dosing request is refused and deferred to a pharmacist, and both out-ofscope requests are declined with a redirect to wellness topics. On emergencies the diference is prominence: vanilla does eventually mention the Thai emergency number, but buries it after a long diferential-diagnosis-style explainer (1669 first appears at character 2,354 of a 3,596- character answer), whereas the +safety answer leads with it (character 141 of 280); the selfharm hotline 1323 shows the same pattern (character 330 of 1,599 vs. 166 of 361). For a user with chest pain, prominence is the metric that matters. Professional-referral language on herbal content also rises from 14/23 to 22/23.

Provider recommendation does not work without the knowledge base. On the 12 colloquial geospatial queries, the no-retrieval model produced zero verifiable provider recommendations: 8/12 answers fell back to naming well-known national spa chains (Health Land, Let’s Relax, Oasis, Divana) with no address, hours, certification status, or confirmation that a branch exists in the asked-about neighborhood or province, and 4/12 declined specifics entirely, deferring the user to Google Maps or hotel staf. The deployed system answers the same query family from 484 verified providers with geospatial filtering (163/181 pass in the QA campaign). Multilingual reply behavior, by contrast, was unafected by the prompt (13/14 in both conditions), consistent with it being inherited from the base model in TSWAP as well.

Caveats: one frontier model, small n, author-graded surface metrics, and a reconstructed rather than production prompt; the probe complements, not replaces, engine-side ablations.

## 4.5 Serving findings

Two deployment findings transfer beyond TSWAP. (1) English-calibrated quantization degrades Thai: the community 4-bit AWQ checkpoint measurably corrupts Thai vowel and tone-mark generation, motivating the post-generation quality guard (Section 3.4) and a migration to FP8. (2) Forced retrieval is necessary: with automatic tool selection the model silently skipped retrieval on short queries; classifier-forced tool\_choice restored reliable grounding.

## 4.6 Toward an answer-quality benchmark

The released harness scafolds the protocol we propose for a full Thai herbal answer-quality benchmark: an expert-authored QA set tagged by topic and expected safety behavior, translated into the eight languages with native back-translation [20]; rubric-based grading [5] by an LLM jury validated against native-speaker experts with chance-corrected agreement; RAG faithfulness metrics per query-language [11, 8]; and safe/problematic/unsafe labeling with redteaming [18]. The engine’s swappable OpenAI-compatible backend makes the corresponding engine-side ablations cheap to run; they remain future work.

## 5 Discussion and Limitations

TSWAP shows that a grounding stack around an unmodified multilingual LLM can deliver a deployed wellness advisor in a culturally specific, low-resource domain without fine-tuning. Limitations, plainly: per-language quality is inherited from pretraining (a risk for the secondary languages); the safety layer has no trained classifier or interaction checker; engine-side ablations and per-language answer-quality results do not yet exist; retrieval is weakest exactly on the herbal collection; end-to-end failures concentrate on colloquial geospatial queries and answers are not fully reproducible across runs; and evaluation covers a single deployment. These set the future-work agenda: the full answer-quality benchmark, engine-side ablations, FP8 migration, thai\_herbs retrieval improvements, and structured red-teaming.

## 6 Ethics and Data Statement

TSWAP forwards only data-minimized, wellness-relevant profile fields to the LLM (PDPA compliance). It is positioned as general wellness guidance, not medical advice, and routes emergencies to Thai hotlines (1669, 1323). Informed consent was obtained from all 120 UAT participants. The 50-question retrieval golden set, the production QA test logs, and the frontier-probe logs are released with this paper; the herbal knowledge base itself is not part of the release.

## 7 Conclusion

TSWAP fills an empty niche in Thai NLP: no prior deployed, multilingual, retrieval-grounded Thai traditional-medicine advisor existed, and no public Thai herbal benchmark. We release the first such benchmark with production evaluation logs, and contribute grounding and serving findings we expect to transfer to other tool-using RAG assistants in low-resource languages.

## Acknowledgments

This research project is financially supported by the Program Management Unit for Competitiveness (PMUC) under grant number C05F680144.

## References

[1] Piyasawetkul, T., Tiyaworanant, S., Srisongkram, T. AppHerb: Language Model for Recommending Traditional Thai Medicine. AI (MDPI), 6(8):170, 2025. https://doi.org/10.3390/ai6080170

[2] iApp Technology. OpenThai Chinda 4B: Thai Sovereign AI Language Model. Hugging Face, 2025. https://huggingface.co/iapp/chinda-qwen3-4b

[3] Thiprak, Y., Ngodngamthaweesuk, R., Ngodngamtaweesuk, S. Eir: Thai Medical Large Language Models. arXiv:2409.08523, 2024.

[4] Global Wellness Institute. The Global Wellness Economy: Thailand. February 2025. https:// globalwellnessinstitute.org/wellness-in-thailand/

[5] Arora, R. K., Wei, J., Soskin Hicks, R., et al. HealthBench: Evaluating Large Language Models Towards Improved Human Health. arXiv:2505.08775, 2025.

[6] Gumma, V., Raghunath, A., Jain, M., Sitaram, S. HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings. arXiv:2410.13671, 2024.

[7] Zhang, W., Aljunied, S. M., Gao, C., Chia, Y. K., Bing, L. M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models. NeurIPS 2023, Datasets and Benchmarks Track, 2023. arXiv:2306.05179.

[8] Xiong, G., Jin, Q., Lu, Z., Zhang, A. Benchmarking Retrieval-Augmented Generation for Medicine. Findings of ACL 2024, pp. 6233–6251, 2024. arXiv:2402.13178.

[9] Kong, S., Yang, X., Wei, Y., et al. MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine. arXiv:2506.01252, 2025.

[10] Yuenyong, S., Viriyayudhakorn, K., Piyatumrong, A., Jaroenkantasima, J. OpenThaiGPT 1.5: A Thai-Centric Open Source Large Language Model. arXiv:2411.07238, 2024.

[11] Es, S., James, J., Espinosa-Anke, L., Schockaert, S. RAGAs: Automated Evaluation of Retrieval Augmented Generation. EACL 2024, System Demonstrations, 2024. arXiv:2309.15217.

[12] Dou, L., Liu, Q., Zeng, G., et al. Sailor: Open Language Models for South-East Asia. EMNLP 2024, System Demonstrations, pp. 424–435, 2024. arXiv:2404.03608.

[13] Ng, R., Nguyen, T. N., Huang, Y., et al. SEA-LION: Southeast Asian Languages in One Network. IJCNLP–AACL 2025, 2025. arXiv:2504.05747.

[14] Xie, J., Yu, Y., Zhang, Z., et al. TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine. arXiv:2505.24063, 2025.

[15] Saowaprut, P., Wabina, R. S., Yang, J., Siriwat, L. Performance of large language models on Thailand’s national medical licensing examination: a cross-sectional study. J. Educ. Eval. Health Prof., 22:16, 2025. https://doi.org/10.3352/jeehp.2025.22.16

[16] Pipatanakul, K., Jirabovonvisut, P., Manakul, P., et al. Typhoon: Thai Large Language Models. arXiv:2312.13951, 2023.

[17] Pipatanakul, K., Manakul, P., Nitarach, N., et al. Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models. arXiv:2412.13702, 2024.

[18] Draelos, R. L., Afreen, S., Blasko, B., et al. Large language models provide unsafe answers to patient-posed medical questions. npj Digital Medicine, 9:241, 2026. arXiv:2507.18905.

[19] Limkonchotiwat, P., Tuchinda, P., Lowphansirikul, L., et al. WangchanThaiInstruct: An Instruction-Following Dataset for Culture-Aware, Multitask, and Multi-domain Evaluation in Thai. EMNLP 2025, pp. 3535–3558, 2025. arXiv:2508.15239.

[20] Jin, Y., Chandra, M., Verma, G., Hu, Y., De Choudhury, M., Kumar, S. Better to Ask in English: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries. The Web Conference (WWW), 2024. https://doi.org/10.1145/3589334.3645643