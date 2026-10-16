# Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence

Owen Walters\* Lihu Chen Francesca Toni Department of Computing, Imperial College London, UK {lihu.chen, ft}@imperial.ac.uk

## Abstract

Web search allows LLMs to access up-todate information, but it remains unclear why they sometimes produce outdated answers even when relevant evidence is available. To study this problem, we construct temporal facts whose values changed after the models’ training cutoffs. We evaluate one proprietary model under deployed web search and six open-weight model configurations under controlled retrieval. We find that models often answer from search snippets without opening the source pages, which is associated with higher rates of outdated answers and unfaithful citations. Our controlled experiments and layer-wise analysis reveal a gap between the availability of evidence and its effective use. Even when retrieved evidence influences internal predictions, models may still favour outdated information in their final answers. We hope these findings will help advance research on the reliable use of retrieved evidence in search-augmented LLMs.

## 1 Introduction

Deployed LLM assistants such as ChatGPT (OpenAI, 2024) often use web search to access up-todate information. However, retrieving relevant search results does not guarantee that models use the information correctly. Even when updated evidence is available, search-augmented LLMs may still produce outdated answers or cite sources that contradict their responses (Figure 1). This matters because citations lend answers provenance. This raises an important question: Why do searchaugmented LLMs fail to use available evidence?

Prior work has studied temporal knowledge (Mousavi et al., 2024; Zhu et al., 2025), citation faithfulness (Wallat et al., 2025b) and failures in retrieval-augmented generation (RAG) (Chen et al., 2024; Xie et al., 2024). However, how searchaugmented LLMs use web search results has received less attention, and it is unclear why these models fail even when relevant evidence is available.

![](images/2612c838f0a4d4f02e21e8d64e359b1030e9e380cbb0bc5f0f795303c6e4e801.jpg)  
Figure 1: An example of the “retrieved but not read” phenomenon in a deployed search-augmented LLM. The model searches once, opens no page, endorses a claim five months out of date and attributes it to an outdated article on the company’s own site.

To this end, we construct a dataset of post-cutoff temporalfacts, whose values changed after the deployed model’s training cutoff. We build a deterministic and regenerable pipeline that generates these facts from primary sources, and draw 225 facts from its output, across six domains including sports, finance and medicine.<sup>1</sup> Each fact has an outdated value and a current value. This lets us identify outdated answers, test whether models use updated evidence correctly and check whether the cited pages support the models’ claims.

We first study a proprietary LLM under its native web search to identify failures in a deployed setting. We then test whether these failures recur under controlled retrieval, giving the same proprietary LLM and six open-weight model configurations an identical menu of frozen search results. Our experiments reveal two main findings. First, models frequently answer from result snippets without opening a page, and most stale answers and unfaithful citations occur in these episodes. These failures are also shared across models (§4). Second, to understand why these failures happen, we run controlled experiments and a mechanistic analysis. Providing the source page directly reduces some errors. In the remaining failures, two layer-wise readouts show that the retrieved value is promoted, but the stale value is not suppressed enough to be displaced (§5).

## 2 Related Work

Temporal knowledge and web search. Prior benchmarks measure whether LLMs answer timesensitive questions correctly (Mousavi et al., 2024; Zhu et al., 2025; Wallat et al., 2025a; Cheng and Dou, 2025), but not whether the answers come from the retrieved evidence. Kale (2026) finds that four models from two families invoke native search on 87.5–93.9% of questions that need it, without checking whether the cited pages support the answers. Premature stopping and top-result reliance also surface in evaluations of search agents and LLM-based search engines (Gupta et al., 2026; Zhang et al., 2025). We extend these studies by examining snippet reliance in both proprietary and open-weight LLMs.

Citation faithfulness. Citation quality is usually scored as whether a cited passage supports a claim (Gao et al., 2023). Wallat et al. (2025b) separate this correctness from faithfulness, that is, whether the source actually produced the answer. On a single open model in a controlled pipeline with adversarially modified documents, they report up to 57% post-rationalisation. Other studies examine how LLMs handle conflicts between parametric knowledge and external evidence, from external behaviour (Xie et al., 2024; Chen et al., 2024) to internal representations (Ortu et al., 2024; Halawi et al., 2024). We use post-cutoff facts to study citation faithfulness under both live and controlled web search.

## 3 Setup

Post-cutoff temporal facts. To study evidence use in a controllable setting, we construct postcutoff temporal facts, whose values changed after the models’ training cutoffs. As a common threshold, we use 16 February 2026, the latest reported training cutoff among the evaluated models (OpenAI, 2026a). We exclude a change if it took effect on or before this date or was announced before 1 February 2026, and we exclude FDA claims that can be answered from standard medical knowledge. We draw 225 treatment facts across six domains (FDA 62, finance 52, sports 47, people 43, SEC 13, chemical 8), plus control facts that did not change across the cutoff. To estimate how much can be inferred without search, we report Luna’s closed-book accuracy on open questions (23.3%, 24/103). All extraction is deterministic. The one model-assisted step, rewriting FDA label clauses as atomic claims, is reviewed manually for every fact (Appendix B).

Models and retrieval. We use two search settings to examine whether failures observed in deployment recur across models. First, we evaluate GPT-5.6 Luna (OpenAI, 2026b), a proprietary LLM, under its native web search (the deployed run). We then evaluate the same model and six open-weight model configurations under controlled retrieval: GLM-5.3-Flash (Z.ai, 2026), gpt-oss-120b and gpt-oss-20b (OpenAI, 2025), gemma-4- 31b (with and without reasoning; Google Deep-Mind, 2026) and qwen3-8b (Qwen Team, 2025). Frozen retrieval covers 223 of the 225 facts, and for each one every model receives the same ten-result search menu, built once from Brave Search (Brave, 2026). Models navigate these results with the same search, open-page and find tools. The ground-truth page keeps its rank if it is in the organic top ten; otherwise, it is inserted at a seeded position, so it is always reachable. Temperature is 0 where exposed, and serving providers are pinned (Appendix A).

Tasks. We test each fact with two main prompt types. An open question (“Which club does Bernardo Silva playfor?”) is asked bare and with a source requested. A claim check (“Is the following accurate? . . . ”) carries either the current or the outdated value, so endorsement of the stale claim is read against its true twin. The pooled rates below also draw on four further prompt types (Appendix C): cued open questions, quote requests, citation requests (attach a source to a supplied claim) and fact-sheet checks (find the one stale line among true ones). Every prompt states the date, and most end with a required answer line that allows CAN-NOT VERIFY and can be graded deterministically.

Measures. We use three measures. Reading depth comes from the tool trace: snippet-only episodes open no page, integrated episodes cite a page they opened, and the rest open a page without citing it. Accuracy is the share of correct answers among those that are correct, stale or another value; abstentions are excluded. A citation is unfaithful when the cited page contradicts the value the reply asserts. Unfaithful rates are per graded answer, and control facts give each model’s floor. On-claim rates count only citations attached to the claim under test. Grading is deterministic where possible. An LLM marker may only assign passing faithfulness outcomes, each backed by a verbatim span of the archived page. Every adverse faithfulness label that needs judgement is assigned by a human (Appendix D).

<table><tr><td>Model</td><td>Acc.</td><td>Snippet</td><td>Unfaith.</td><td>Floor</td></tr><tr><td>Luna (deployed)</td><td>93.5</td><td>53.6</td><td>2.6</td><td>0.0</td></tr><tr><td>Luna</td><td>91.8</td><td>9.9</td><td>1.0</td><td>0.0</td></tr><tr><td>GLM-5.3-Flash</td><td>94.4</td><td>13.4</td><td>1.8</td><td>0.0</td></tr><tr><td>gpt-oss-120b</td><td>92.4</td><td>2.6</td><td>3.6</td><td>0.3</td></tr><tr><td>gpt-oss-20b</td><td>95.4</td><td>7.9</td><td>3.7</td><td>0.3</td></tr><tr><td>gemma-4-31b</td><td>92.0</td><td>64.6</td><td>4.9</td><td>0.5</td></tr><tr><td>gemma-4-31b-think</td><td>89.6</td><td>56.5</td><td>6.4</td><td>0.3</td></tr><tr><td>qwen3-8b</td><td>85.8</td><td>76.0</td><td>9.4</td><td>4.0</td></tr></table>

Table 1: Model performance under frozen retrieval, with Luna’s deployed run for reference. Acc.: accuracy on bare open questions with search. Snippet: share of search episodes that open no page. Unfaith.: share of graded answers with an unfaithful citation. Floor: unfaithful-citation rate on control facts. All values are percentages; the last three columns pool all prompt types (Appendix C).

## 4 Results

We investigate how search-augmented LLMs use retrieved evidence and where failures occur. We first examine their search behaviour and the reliability of their answers and citations. We then test whether these failures are shared across models and whether providing evidence directly reduces them.

## 4.1 Search-Augmented LLMs Often Rely on Snippets

Search helps every configuration: under frozen retrieval, enabling search raises claim-check accuracy by 27–42 points (Appendix E), and openquestion accuracy reaches 85.8–95.4% (Table 1). What differs is how deeply models read. Given the same search results, models answer 2.6–76.0% of episodes without opening any page. Under its native search, Luna answers 53.6% of episodes from snippets alone.

<table><tr><td></td><td>Snippet</td><td>Opened</td><td>Integr.</td></tr><tr><td>Stale answer</td><td>6.4</td><td>1.7</td><td>1.3</td></tr><tr><td>Unfaithful citation</td><td>6.4</td><td>1.6</td><td>2.3</td></tr></table>

Table 2: Adverse-event rates (%) by reading depth under frozen retrieval, pooled over all seven configurations and all prompt types: snippet-only, page opened but not cited, and opened page cited. Counts: stale 87/1366, 17/990, 8/632; unfaithful 62/962, 11/683, 29/1258.

Snippet-only episodes fail more often. Pooled over all seven configurations, they are stale 6.4% of the time, against 1.3% when an opened page is cited. They also carry unfaithful citations 6.4% of the time, against 2.3% (Table 2). The gap is particularly large for sports: among the open models, snippet-only answers are stale 20.2% of the time, against 2.1% for integrated ones (Appendix F).

Unfaithful citations are also strongly associated with stale answers. On sourced open questions, 81.8% (27/33) of on-claim citations are unfaithful when answers are stale, compared with only 0.9% (6/694) when answers are current. Similar patterns appear in fact-sheet checks (Appendix G).

## 4.2 Failures Are Shared across Models

We next test whether prompts that Luna fails under native search are more likely to fail under frozen retrieval. Seven of the nine failure types show significant positive associations (odds ratios 1.84–6.46, p < 0.05), while a negative control shows no association (odds ratio 1.05; Appendix I).

## 4.3 Direct Access to Evidence Reduces Failures

We first test whether providing evidence directly reduces failures. On 50 sports claims, gemma-4- 31b and gpt-oss-20b endorse the stale claim 14/50 and 11/47 times when they have to find the page in the search menu, but only 0/50 and 1/50 times when the same Wikipedia page is in the prompt. In all 14 of its failures, gemma-4-31b opens no page; qwen3-8b benefits less (from 16/49 to 8/49; Appendix H).

In an agentic loop, the question comes before everything the model retrieves. We therefore test whether restating the full prompt after each opened page helps, keeping the menu, pages and sampling settings unchanged. This recovers 41 of 157 failed fact-sheet checks (26.1%; 4.5–57.1% across the six configurations re-run; Appendix K).

![](images/7ee60410c87e447877b83b500563de0cb7ed98e39275f6ada5cc8b88dbf057b3.jpg)  
Figure 2: Layer-wise ranks of the retrieved (green) and outdated (orange) clubs for qwen3-8b in successful (left) and failed (right) cases. Solid and dashed lines represent J-lens and logit lens, respectively.

Here is what you know about Anthony Gordon:   
1. Anthony Gordon plays for Newcastle United   
F.C. (the asserted club)   
[lines 2–4: three truefactsfrom his page]   
Which club does Anthony Gordon play for?   
[Source 1] Anthony Michael Gordon . . . plays   
primarily as a winger for La Liga club   
Barcelona . . . (page prompt only)   
Anthony Gordon plays for (the answer slot, read at   
every layer)  
Figure 3: The two readout prompts for one fact. Both open with the date and a fact sheet in which one line asserts the old club; the page prompt adds the player’s current page, which names the new one. Each club is scored at the answer slot.

## 5 Mechanistic Analysis of Evidence Use

Providing evidence directly reduces failures but does not eliminate them. We investigate why models may still give outdated answers when the correct evidence is available.

We use player transfer facts from the sports domain and three open-weight models: gpt-oss-20b, gemma-4-31b and qwen3-8b. For each fact, we construct two prompts that assert an outdated club and ask for the player’s current club. One prompt also includes the player’s current Wikipedia page (Figure 3). Comparing the two prompts shows how models choose between the outdated and current answers. We use the logit lens (nostalgebraist, 2020) and the Jacobian lens (J-lens; Gurnee et al., 2026) to probe how the model represents the two candidate answers across layers. Both methods project intermediate hidden states at the answer position onto token-level logits. Each club’s score is the highest logit among selected first tokens of its name, so a higher score indicates stronger support for that club. We track the ranks of both clubs across layers and compare their scores with and without the page (Table 12 in Appendix J).

Figure 2 shows how the two candidate answers compete across layers in qwen3-8b. In successful cases, the retrieved club gains an advantage in the upper layers and keeps it until the final layers. In failed cases, the retrieved club ranks above the outdated club early in the upper layers, but the outdated club overtakes it in the final layers. This suggests that retrieved evidence can influence internal predictions even when the model fails. The same crossover appears in gpt-oss-20b, while in gemma-4-31b the outdated club rises first and stays ahead (Appendix J). Averaged over the upper third of layers, the page still promotes the retrieved club in failures, though less than in updates $( \Delta _ { \mathrm { n e w } } > 0 $ $p \leq 0 . 0 4 )$ , but suppresses the outdated club far less, and in gemma-4-31b not at all (Table 13).

## 6 Conclusion

We study why search-augmented LLMs fail to use available evidence correctly. We find that models often retrieve relevant evidence but answer from search snippets without opening the source pages. Such answers are more likely to be stale and to carry unfaithful citations. Providing relevant evidence directly reduces failures but does not eliminate them. To understand the remaining failures, we analyse how models process retrieved evidence across layers. The retrieved answer can gain an early advantage in the upper layers but lose it to the outdated answer in the final layers. Future work can explore strategies that help models make more effective use of source pages when needed. Causal interventions and latent debate (Chen et al., 2026) can be used to better understand how retrieved evidence competes with outdated knowledge and how to improve it. These findings may help guide the development of search-augmented LLMs that use retrieved evidence better and give more reliable responses.

## Limitations

We study only one proprietary model, the lightest of the three GPT-5.6 tiers, chosen for cost. A natural extension is to run the experiments on additional frontier models, which may demonstrate different behaviour.

Z.ai’s GLM-5.3-Flash discloses no training cutoff, so we check for contamination through its behaviour instead. We find its control-fact performance and closed-book accuracy match those of the models with stated cutoffs (Appendix B). We include it because it is one of the most capable open-weight models available.

Most prompts require a fixed answer line, so that answers can be graded mechanically rather than by an LLM. This ‘contract’ can result in different model behaviour compared to exclusively free prose. In the study’s pre-registered answerline check, it lowered Luna’s false endorsement by 17.5 points (Appendix C). Luna’s deployed failures, which anchor the carry-over analysis, are therefore likely to be lower bounds. We keep the answer line because it gives standardised, reliable grading across the more than 30,000 replies in the deployed and frozen runs.

Unfaithful citations are rare for some models under frozen retrieval (1.0% of Luna’s graded answers and 1.8% of GLM-5.3-Flash’s), so a few grading errors could shift these rates substantially, in both absolute and relative terms. We therefore took particular care with grading. An LLM marker (DeepSeek-V4-Flash; DeepSeek-AI, 2026) helps grade that the deterministic grader cannot decide, but it can only pass a reply, never assign an adverse label. Every unfaithful citation, and every other adverse label that needs judgement, is assigned or confirmed by a human (Appendix D).

Failures also vary substantially across domains: sports drives the severe rates, while SEC (13 facts) and chemical (8) are underpowered because few usable post-cutoff facts exist for them. We include them because they add variety to the study. That sports drives so many failures is a finding in its own right, since the answer usually sits in the first line of the player’s Wikipedia page, whereas FDA questions require reading long label sections and reasoning about often complex medical knowledge.

The layer-wise analysis with the logit lens and Jlens uses sports transfers only, because the answer, a club name, is only a few tokens long. It characterises the failure but does not test a mechanism causally, for example by activation patching.

## Ethical Considerations

Our finding that search-augmented LLMs often answer from snippets alone could help malicious actors spread misinformation, since misleading text that appears in a snippet may be repeated without the model opening the page. To limit this risk, we report only aggregate failure rates and release no tools for manipulating search snippets. All facts in the study are public. Most come from moderated sources (Wikipedia and Wikidata) or from government records (FDA drug labels and SEC filings); every finance value we use is the figure reported in an SEC filing, and we re-derive all 52 from SEC EDGAR. The people domain names individuals whose deaths are recorded on Wikipedia. These facts cannot be anonymised, because each concerns a named person, but they are already public, and the outdated claims about them are used only as test items. Outdated drug-label information carries clinical risk, which is one of the main reasons we include FDA label changes. Nothing in this study, including the drug-label facts, should be used for medical purposes. We queried all models in line with their providers’ terms of use, and no human participants were involved beyond the authors’ own grading.

## Acknowledgments

This paper is based on the first author’s MSc dissertation at Imperial College London.

## References

Brave. 2026. Brave Search API. https://brave.com/ search/api/. Accessed August 2026.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large language models in retrieval-augmented generation. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):17754–17762.

Lihu Chen, Xiang Yin, and Francesca Toni. 2026. Latent debate: A surrogate framework for interpreting LLM thinking. Preprint, arXiv:2512.01909.

Jiehan Cheng and Zhicheng Dou. 2025. DailyQA: A benchmark to evaluate web retrieval augmented

LLMs based on capturing real-world changes. Preprint, arXiv:2505.17162.

DeepSeek-AI. 2026. DeepSeek-V4-Flash. https://huggingface.co/deepseek-ai/ DeepSeek-V4-Flash. Model card, accessed October 2026.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488. Association for Computational Linguistics.

Google DeepMind. 2026. Gemma 4 model card. https://ai.google.dev/gemma/docs/ core/model\_card\_4. Accessed October 2026.

Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, Sasha Goldshtein, and Dipanjan Das. 2026. DeepSearchQA: Bridging the comprehensiveness gap for deep research agents. Preprint, arXiv:2601.20975.

Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. 2026. Verbalizable representations form a global workspace in language models. Preprint, arXiv:2607.15495.

Danny Halawi, Jean-Stanislas Denain, and Jacob Steinhardt. 2024. Overthinking the truth: Understanding how language models process false demonstrations. In The Twelfth International Conference on Learning Representations.

Sahil Kale. 2026. Look it up: Analysing internal web search capabilities of modern LLMs. Preprint, arXiv:2511.18931.

Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe Riccardi. 2024. DyKnow: Dynamically verifying time-sensitive factual knowledge in LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8014–8029. Association for Computational Linguistics.

nostalgebraist. 2020. interpreting GPT: the logit lens. LessWrong.

OpenAI. 2024. Introducing ChatGPT search. https://openai.com/index/ introducing-chatgpt-search/. Accessed October 2026.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

OpenAI. 2026a. GPT-5.6 Luna. https: //developers.openai.com/api/docs/models/ gpt-5.6-luna. Model documentation, accessed October 2026.

OpenAI. 2026b. Improving GPT-5.6 Sol in ChatGPT—and expanding access to GPT-5.6 Luna for free users. https://openai.com/index/ improving-gpt-5-6-sol-in-chatgpt/. Accessed October 2026.

Francesco Ortu, Zhijing Jin, Diego Doimo, Mrinmaya Sachan, Alberto Cazzaniga, and Bernhard Schölkopf. 2024. Competition of mechanisms: Tracing how language models handle facts and counterfactuals. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8420–8436. Association for Computational Linguistics.

Qwen Team. 2025. Qwen3-8B. https:// huggingface.co/Qwen/Qwen3-8B. Model card, accessed October 2026.

Jonas Wallat, Abdelrahman Abdallah, Adam Jatowt, and Avishek Anand. 2025a. A study into investigating temporal robustness of LLMs. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 15685–15705. Association for Computational Linguistics.

Jonas Wallat, Maria Heuss, Maarten de Rijke, and Avishek Anand. 2025b. Correctness is not faithfulness in retrieval augmented generation attributions. In Proceedings ofthe 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR), pages 22–32. ACM.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2024. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In The Twelfth International Conference on Learning Representations.

Z.ai. 2026. GLM-5.3-Flash. https://huggingface. co/zai-org/GLM-5.3-Flash. Model card, accessed October 2026.

Peixian Zhang, Qiming Ye, Zifan Peng, Kiran Garimella, and Gareth Tyson. 2025. Source coverage and citation bias in LLM-based vs. traditional search engines. Preprint, arXiv:2512.09483.

Zhiyuan Zhu, Yusheng Liao, Zhe Chen, Yuhao Wang, Yunfeng Guan, Yanfeng Wang, and Yu Wang. 2025. EvolveBench: A comprehensive benchmark for assessing temporal awareness in LLMs on evolving knowledge. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16173–16188. Association for Computational Linguistics.

## A Models

Table 3 lists the seven configurations. All frozenretrieval runs go through OpenRouter, with the serving provider pinned because providers serve different quantisations. Temperature is 0 where the provider exposes it, and Luna runs at the provider default. The layer-wise readouts in Section 5 use the open weights of gpt-oss-20b, gemma-4-31b and qwen3-8b.

Frozen retrieval. The open models have no provider-native search tool, so our tool loop supplies one, and Luna uses the same loop under frozen retrieval. It offers three functions in standard function-calling format, search, open page and find in page, which each model’s own chat template renders; in the closed-book condition no tools are offered. Every search call returns the fact’s frozen menu, so the query is recorded but has no effect. A menu always has exactly ten entries, each with a title, URL and snippet. Where the ground-truth page is not in the organic top ten, it is inserted at a seeded position and the tenth entry is dropped (62 of the 223 menus), and the tool result gives no sign of this. Opening a page returns its frozen text in windows of 40,000 characters, and pages not on the menu are refused. Each episode allows at most five searches, five page opens and eight assistant turns.

Deployed run. Luna’s deployed run uses OpenAI’s Responses API with the provider’s own web search at default settings. Luna searched in every one of the 1,747 treatment episodes with search enabled, 1.58 times on average with 23 results per search, and opened at least one page in 46.4% of them.

## B Fact Sources

Table 4 lists the six sources. The pipeline generates facts using a single cutoff of 1 February 2026; the evaluated set then keeps only changes after the evaluated model’s reported cutoff. GLM-5.3-Flash discloses no training cutoff, so contamination is ruled out by behaviour instead. Closed book, it judges post-cutoff claims correctly 53.6% of the time, within the 45.7–58.4% of the other models (Table 7), and with search it answers 98.3% of control questions correctly, against 97.0–100% for the others.

Facts are stratified by how predictable the change is. Deaths and transfers cannot be announced in advance. Earnings, officer changes and label changes are partly predictable. IPO listings and ticker changes are usually announced, so the announcement-date gate removes most of them.

For FDA facts, each label’s Recent Major Changes field certifies the drug, the section and the month of revision. We fetch the current section and compare it, sentence by sentence, with the latest archived version from before the cutoff; the clause that is new becomes the fact. A clause already present in the earlier label becomes a control, and a claim answerable from standard medical knowledge is excluded.

Control facts. Control facts did not change across the cutoff, so their failure rates give each model’s floor. Sports controls are players who have been at the same club since 31 March 2024 or earlier, matched to the treatment players by league tier, tenure and prominence. People controls are living people, matched to the treatment people by prominence. FDA controls are label clauses already present before the cutoff. The deployed run uses 120 control facts (people 47, sports 39, FDA 34), and frozen retrieval uses 119.

Exclusions. Of the 233 treatment facts run, eight were withdrawn after review: four had malformed before or after values, two named people who could not be pinned to a single individual, and two had unreliable ground truth. This leaves 225. Narrower exclusions apply only to the prompts they affect: 27 people fact sheets whose planted line stays true after the death, so it is not false; one finance fact whose false claim reads the same as its true claim, in the false-claim prompts only; two FDA control claims that drop a qualifier the label itself carries, in the claim prompts only; and a few prompts with known rendering or grading defects. An early pilot run is excluded entirely.

## C Prompt Types

Table 5 lists the six prompt types. Claim checks, quote requests and citation requests come in pairs: a false twin carrying the outdated value and a true twin carrying the current one. Accuracy in Table 1 uses only the bare open question, while readingdepth and citation rates pool every type. In the deployed run, every prompt with search enabled runs its own search. Under frozen retrieval, some prompts reuse the evidence opened by a twin that differs only in wording or in the claim, so readingdepth rates pool only the prompts that search for themselves. Citation requests use a binary answer line with no CANNOT VERIFY option. The answer line is not neutral: in a pre-registered closed-book check, adding it lowered Luna’s false endorsement by 17.5 points compared with free prose.

<table><tr><td>Model</td><td>Developer</td><td>Parameters</td><td>Pinned provider</td><td>Used in</td></tr><tr><td>GPT-5.6 Luna</td><td>OpenAI</td><td>undisclosed</td><td>OpenAI</td><td>deployed, frozen</td></tr><tr><td>GLM-5.3-Flash</td><td>Z.ai</td><td>320B MoE (18B active)</td><td>stealth endpoint</td><td>frozen</td></tr><tr><td>gpt-oss-120b</td><td>OpenAI</td><td>117B MoE (5.1B active)</td><td>DeepInfra (bf16)</td><td>frozen</td></tr><tr><td>gpt-oss-20b</td><td>OpenAI</td><td>21B MoE (3.6B active)</td><td>DeepInfra (bf16)</td><td>frozen, readouts</td></tr><tr><td>gemma-4-31b</td><td>Google</td><td>31B dense</td><td>CoreWeave (bf16)</td><td>frozen, readouts</td></tr><tr><td>gemma-4-31b-think</td><td>Google</td><td>31B dense, reasoning on</td><td>CoreWeave (bf16)</td><td>frozen</td></tr><tr><td>qwen3-8b</td><td>Alibaba</td><td>8B dense</td><td>Alibaba (fp8)</td><td>frozen, readouts</td></tr></table>

Table 3: The seven model configurations. Pinned provider is the serving provider for the frozen-retrieval runs; Luna’s deployed run uses OpenAI’s own API with native web search. Used in: the deployed run, frozen retrieval, and the layer-wise readouts.

<table><tr><td>Source</td><td>Captures</td><td>Primary ground truth</td><td>Corroboration</td><td>Included</td></tr><tr><td>SEC</td><td>Senior-officer changes, S&amp;P 500</td><td>8-K Item 5.02 filings; SOX §302 signatures</td><td>none (filing authoritative)</td><td>20 / 652</td></tr><tr><td>Finance</td><td>Quarterly revenue; ticker changes; IPOs</td><td>SÉC EDGAR</td><td>none (sole source)</td><td>1,444 / 2,273</td></tr><tr><td>Sports</td><td>Athlete club transfers New clause in a revised label</td><td>Wikipedia infobox openFDA + DailyMed vs. archived</td><td>Wikidata P54 none (label</td><td>212 / 959</td></tr><tr><td>FDA</td><td>section</td><td>prior</td><td>authoritative)</td><td>112 /318</td></tr><tr><td>People Chemical</td><td>Deaths (date; place or cause) Carcinogenicity reclassifications</td><td>Wikipedia infobox IARC, Prop 65, EPA TSCA</td><td>Wikidata P570 pinned reference table</td><td>3,473 / 6,409 8/8</td></tr></table>

Table 4: Fact sources. Included counts are records in the full dataset, out of the candidates screened; the evaluation draws 225 treatment facts from them.

<table><tr><td>Prompt type</td><td>The prompt supplies</td><td>The model must</td><td>Failure looks like</td><td>Domains</td></tr><tr><td>Open question (bare, or with a source requested)</td><td>the question only</td><td>produce the value</td><td>states the stale value</td><td>all six</td></tr><tr><td>Cued open question</td><td>the question plus a temporal or source cue</td><td>produce the value</td><td>stale despite the cue</td><td>FDA, sports</td></tr><tr><td>Claim check Quote request</td><td>a claim carrying one value a claim and a named</td><td>return a verdict give a verdict and the</td><td>endorses the stale claim endorses, and supplies a</td><td>all six FDA, people, sports</td></tr><tr><td></td><td>authority a claim the model never</td><td>supporting sentence</td><td>sentence not in the document the source does not</td><td></td></tr><tr><td></td><td>asserted</td><td></td><td>support it passes the sheet as</td><td></td></tr><tr><td>Fact-sheet check</td><td>a claim among true lines</td><td>find the false line</td><td>accurate</td><td>FDA, people, sports</td></tr></table>

Table 5: Prompt types. The main text analyses the open question and the claim check; the pooled reading-depth and citation rates also draw on the other types.

Prompts for one fact. Table 6 gives each prompt type for one fact, word for word as sent in the deployed run.

## D Grading

Every reply is graded in three tiers, recorded in one append-only grade book. First, a deterministic engine decides everything mechanical. It matches the value the reply asserts against terms derived from the fact, requires a quote to be an exact substring of the archived page, and checks each citation against the value the reply asserts rather than against the ground truth. Second, what the engine cannot decide goes to an LLM marker, DeepSeek-V4-Flash served through OpenRouter. The marker is not among the evaluated models and shares no model family with them. It matches paraphrases rather than judging: it reads the archived page without seeing the reply, must back every finding with a verbatim quote that code verifies against the page, and may assign only passing outcomes. Adverse findings, splits and unverified quotes go to a human. A route can use the marker only once it agrees with human labels on at least 90% of its human-decided cards, with no contradicted pass (citation support: 30 of 31). Answer-value matching, which only sports and finance answers reach, was calibrated on every card the study could supply and agreed on all 24. Third, humans decide everything else, and only humans assign adverse labels that need judgement. Every unfaithful citation is read by a human before it counts, and finding nothing on a page is never graded unfaithful automatically. As further checks, a fixed 5% of cards never reaches the marker and is graded by hand throughout, and a seeded 10% of the marker’s passes is re-graded blind by a human until 300 have passed cleanly.

In total, the authors hand-graded 4,108 answers on 3,856 cards. Two blind checks measure their consistency: a seeded re-code agreed with the original label on 28 of 31 cards (Cohen’s $\kappa = 0 . 8 4 8 )$ , and a blind re-ask of 103 cards first answered while the grading engine was still escalating them agreed on 96 (κ = 0.865). In the blind audit of the marker’s passes, the human agreed with the marker on 181 of 194 cards. A further 41 cards exposed a defect on living-person control facts, where the marker marks faithful citations as unsupported; this changes no rate reported here.

## E Accuracy with and without Search

Table 7 gives each configuration’s accuracy on claim checks closed book and with search, and on the bare open question with search. Accuracy is correct/(correct + stale + other), excluding abstentions. Closed book, the models sit within a narrow band; the lift from search ranges from 27.0 points (qwen3-8b) to 42.1 (gpt-oss-120b).

## F Snippet Content by Domain

To see which values a model can find without opening a page, we measure what the snippets carry (Table 8). We use each evaluated fact’s frozen menu, built from one canonical query: the search results the models see, without the inserted ground-truth page and with HTML markup removed. A snippet counts as carrying a value when it would satisfy the grading criteria for the statement. Names and labels match with their aliases, financial figures within the grader’s tolerance (1% for quarterly revenue), listings by their ticker, and FDA clauses only when every key-term group is present. For people, the current value is the death, and the stale value is an entity-anchored present-tense description. Among the open models, sports episodes answered from snippets alone are stale 20.2% of the time (53/262), against 2.1% (1/48) when an opened page is cited.

## G Fact-Sheet Outcomes by Model

Table 9 gives each configuration’s fact-sheet failures under frozen retrieval. It also splits the onclaim unfaithful-citation rate by whether the model failed the sheet, that is, missed the stale line or passed the sheet as accurate. Pooled over the seven configurations, citations are unfaithful 1.4% of the time (7/485) when the verdict holds and 58% (53/91) when it fails.

## H Handed-Over versus Searched-For Pages

This experiment tests whether giving models the page directly reduces failures. Each of the 50 facts concerns a footballer or basketball player who changed club in 2026. They are all the sports facts with a frozen menu, including three that the main analysis later withdrew after reviewing their ground truth, so it has 47. The claim names the former club, so the correct verdict is always INACCURATE. Table 10 gives how often each model endorses the stale claim in each condition, with and without a request to cite sources; denominators vary where a model returns an empty reply. Opened pages are served from the frozen-retrieval page cache, which holds about four of the ten listed URLs per fact, so a listed page cannot always be opened. This limit never affects gemma, because it never opens a page.

<table><tr><td>Prompt type</td><td>Prompt as sent</td><td>Answer line</td></tr><tr><td rowspan="2">Open question with a source</td><td>Which club does Bernardo Silva play for?</td><td>(a)</td></tr><tr><td>Which club does Bernardo Silva play for? Please provide sources inline, with links.</td><td>(a)</td></tr><tr><td>Cued open question</td><td>Which club does Bernardo Silva currently play for? or According to Bernardo Silva&#x27;s Wikipedia page, which club does Bernardo Silva play for? or both cues together</td><td>(a)</td></tr><tr><td>Claim check</td><td>Is the following accurate? &#x27;Bernardo Silva plays for Real Madrid CF.&#x27;</td><td>(b)</td></tr><tr><td>Quote request</td><td>Does Bernardo Silva&#x27;s Wikipedia page support this statement: &#x27;Bernardo Silva plays for Real Madrid CF.&#x27;? If yes, quote the exact sentence verbatim, in quotation marks.</td><td>(c)</td></tr><tr><td>Citation request</td><td>If not, say NOT SUPPORTED. Add a citation from Bernardo Silva&#x27;s Wikipedia page to support this statement:</td><td>(d)</td></tr><tr><td>Fact-sheet check</td><td>&#x27;Bernardo Silva plays for Real Madrid CF.&#x27; Is the following information about Bernardo Silva accurate? [Please provide sources inline for the claims.] 1. He is mainly known for his technique, acceleration, agility, passing, ball control, ... 2. Silva has a French Bulldog named John after the</td><td>(e)</td></tr></table>

(a) End your reply with a final line: ANSWER: <short answer> or ANSWER: CANNOT VERIFY. (b) End your reply with a final line containing exactly one of: FINAL: ACCURATE / FINAL: INACCURATE / FINAL: CANNOT VERIFY. (c) As (b), with FINAL: SUPPORTED / FINAL: NOT SUPPORTED / FINAL: CANNOT VERIFY. (d) End your reply with exactly two final lines: first SOURCE: <url> (or SOURCE: NONE if there is none), then a last line containing exactly one of: FINAL ANSWER: CITATION / FINAL ANSWER: NOT SUPPORTED. (e) As (b), with FINAL: ACCURATE / FINAL: INACCURATE: <numbers of the false lines> / FINAL: CANNOT VERIFY.

Table 6: Every prompt type for one fact (Bernardo Silva, Manchester City F.C. to Real Madrid CF), word for word as sent in the deployed run. Every prompt opens with “Today’s date is 2026-08-08.” and closes with the answer line listed. Claim checks, quote requests and citation requests also have a false twin carrying Manchester City F.C.; closed-book versions are identical, with tools withheld. The fact-sheet check is sent as a verdict request, with the bracketed source request, or as a source request alone (“Please provide sources inline for the following information about Bernardo Silva.”, with no answer line). The stale line’s doubled full stop is as sent.
<table><tr><td>Model</td><td>Claims, closed book</td><td>Claims, with search</td><td>Lift (points)</td><td>Open question, with search</td></tr><tr><td>Luna (deployed)</td><td>192/345 (55.7)</td><td>432/443 (97.5)</td><td>41.9</td><td>188/201 (93.5)</td></tr><tr><td>Luna</td><td>184/337 (54.6)</td><td>387/432 (89.6)</td><td>35.0</td><td>179/195 (91.8)</td></tr><tr><td>GLM-5.3-Flash</td><td>104/194 (53.6)</td><td>361/385 (93.8)</td><td>40.2</td><td>201/213 (94.4)</td></tr><tr><td>gpt-oss-120b</td><td>128/251 (51.0)</td><td>394/423 (93.1)</td><td>42.1</td><td>159/172 (92.4)</td></tr><tr><td>gpt-oss-20b</td><td>177/340 (52.1)</td><td>401/434 (92.4)</td><td>40.3</td><td>165/173 (95.4)</td></tr><tr><td>gemma-4-31b</td><td>118/258 (45.7)</td><td>271/351 (77.2)</td><td>31.5</td><td>160/174 (92.0)</td></tr><tr><td>gemma-4-31b-think</td><td>138/290 (47.6)</td><td>290/374 (77.5)</td><td>30.0</td><td>163/182 (89.6)</td></tr><tr><td>qwen3-8b</td><td>129/221 (58.4)</td><td>303/355 (85.4)</td><td>27.0</td><td>133/155 (85.8)</td></tr></table>

Table 7: Accuracy (%) on claim checks closed book and with search, the lift between them, and accuracy on the bare open question with search. All rows use frozen retrieval except the first. Under frozen retrieval, the false claim reuses the evidence its true twin opened (Appendix C).

<table><tr><td>Domain</td><td>Current in a snippet</td><td>Stale in a snippet</td></tr><tr><td>Sports</td><td>72% (34/47)</td><td>70% (33/47)</td></tr><tr><td>People</td><td>91% (39/43)</td><td>53% (23/43)</td></tr><tr><td>Chemical</td><td>50% (4/8)</td><td>13% (1/8)</td></tr><tr><td>SEC</td><td>69% (9/13)</td><td>62% (8/13)</td></tr><tr><td>FDA</td><td>8% (5/62)</td><td>0% (0/62)</td></tr><tr><td>Finance</td><td>90% (45/50)</td><td>44% (22/50)</td></tr></table>

Table 8: Share of the 223 evaluated facts with a frozen menu whose snippets carry the current and the stale value.

## I Shared Failures across Models

This analysis tests whether prompts that Luna fails under native search are more likely to fail under frozen retrieval. For each prompt (fact, task and grading route), we join Luna’s deployed outcome to the frozen-retrieval outcomes for the same prompt. Carried is the share of Luna-failed prompts that also fail under frozen retrieval. New is the share of Luna-passed prompts that fail, which serves as the base rate (Table 11). Confidence intervals for the odds ratios are 95% Woolf intervals.

<table><tr><td>Model</td><td>Stale line missed</td><td>Sheet rubber-stamped</td><td>Unfaithful, sheet failed</td><td>Unfaithful, sheet held</td></tr><tr><td>GLM-5.3-Flash</td><td>3/161 (1.9%)</td><td>5/81 (6.2%)</td><td>4/6 (67%)</td><td>2/147 (1.4%)</td></tr><tr><td>Luna</td><td>8/162 (4.9%)</td><td>10/81 (12.3%)</td><td>0/12 (0%)</td><td>1/135 (0.7%)</td></tr><tr><td>gemma-4-31b-think</td><td>5/135 (3.7%)</td><td>19/81 (23.5%)</td><td>16/17 (94%)</td><td>1/69 (1.4%)</td></tr><tr><td>gemma-4-31b</td><td>20/154 (13.0%)</td><td>21/81 (25.9%)</td><td>10/14 (71%)</td><td>0/42 (0%)</td></tr><tr><td>gpt-oss-120b</td><td>23/132 (17.4%)</td><td>29/44 (65.9%)</td><td>6/7 (86%)</td><td>0/17 (0%)</td></tr><tr><td>gpt-oss-20b</td><td>33/147 (22.4%)</td><td>23/49 (46.9%)</td><td>4/4 (100%)</td><td>0/8 (0%)</td></tr><tr><td>qwen3-8b</td><td>51/154 (33.1%)</td><td>22/81 (27.2%)</td><td>13/31 (42%)</td><td>3/67 (4.5%)</td></tr></table>

Table 9: Fact-sheet checks under frozen retrieval, by configuration, with on-claim unfaithful citations split by whether the model failed the sheet.

<table><tr><td>Condition</td><td>gemma</td><td>gpt-oss</td><td>qwen</td></tr><tr><td>No evidence</td><td>25/50</td><td>5/30</td><td>1/50</td></tr><tr><td>+ sources asked</td><td>24/50</td><td>5/26</td><td>3/50</td></tr><tr><td>Page in prompt</td><td>0/50</td><td>1/50</td><td>8/49</td></tr><tr><td>+ sources asked</td><td>0/50</td><td>0/47</td><td>12/49</td></tr><tr><td>Search menu</td><td>14/50</td><td>11/47</td><td>16/49</td></tr><tr><td>+ sources asked</td><td>13/50</td><td>13/48</td><td>16/50</td></tr></table>

Table 10: Endorsements of the stale claim by evidence condition (gemma-4-31b, gpt-oss-20b, qwen3-8b).
<table><tr><td>Event</td><td>Carried</td><td>New</td><td>OR [95% CI]</td></tr><tr><td>Sheet rubber-stamped</td><td>16/23</td><td>103/394</td><td>6.46 [2.58, 16.14]</td></tr><tr><td>Unfaithful citation</td><td>18/90</td><td>122/2676</td><td>5.23 [3.03, 9.05]</td></tr><tr><td>Stale line missed</td><td>18/41</td><td>117/842</td><td>4.85 [2.54, 9.26]</td></tr><tr><td>False claim sourced</td><td>18/53</td><td>30/258</td><td>3.91 [1.97, 7.75]</td></tr><tr><td>Wrong rejection</td><td>168/372</td><td>400/2268</td><td>3.85 [3.05, 4.85]</td></tr><tr><td>Stale answer</td><td>12/93</td><td>61/1215</td><td>2.80 [1.45, 5.42]</td></tr><tr><td>Endorsement</td><td>155/584</td><td>335/2043</td><td>1.84 [1.48, 2.29]</td></tr><tr><td>Negative control</td><td>10/178</td><td>39/729</td><td>1.05 [0.52, 2.15]</td></tr></table>

Table 11: Carry-over of Luna’s deployed failures into the frozen runs: Luna-failed prompts that fail again (carried), Luna-passed prompts that fail (new), and the odds ratio [95% CI].

Besides the negative control, the table lists every event whose interval excludes 1. For citing a URL absent from the model’s own search session, the interval spans 1 (1.93 [0.97, 3.82]), and fabricated quotes are too rare to test. Stale answers are joined on the route that matches the answer value, which covers sports, finance, SEC and chemical facts; on the people and FDA routes no stale answer carried over (0/18 and 0/12), and pooled over all three routes the odds ratio is 2.66 [1.42, 4.96]. The two fact-sheet events are sheet rubber-stamped (sources requested, stale line never flagged) and stale line missed (sheet judged accurate, or the wrong line flagged). False claim sourced is a citation request answered with a citation rather than a refusal. The negative control is a model asserting that a named document backs a false claim. It occurs at the same rate whether or not Luna failed the prompt, so the join does not produce spurious carry-over.

## J Layer-Wise Readout Details

Table 12 defines the two readout metrics.
<table><tr><td>Metric</td><td>Definition</td><td>What it asks</td></tr><tr><td> $\Delta _ { \mathrm { n e w } }$ </td><td>read(new | page) —</td><td>Did the page promote</td></tr><tr><td rowspan="3"> $\Delta _ { \mathrm { o l d } }$ </td><td>read(new | no page)</td><td>the retrieved club?</td></tr><tr><td>read(old | page) –</td><td>Did it suppress the</td></tr><tr><td>read(old | no page)</td><td>asserted club?</td></tr></table>

Table 12: The two readout metrics. read(c | ·) is club c’s score at the answer slot under a lens, per layer, and each metric is averaged over the upper third of layers. Both terms come from prompts that differ only in the page, so anything specific to the fact cancels.

Club scores. Club names span several tokens, so each club is represented by a set of candidate first tokens. The name is split into words of three or more letters, generic words (e.g. fc, club, united, city) are dropped, and each remaining word contributes its first token, with and without a leading space. At each layer, a club’s logit is the maximum over its set, and its rank is the minimum. In roughly a third to a half of replies, the set misses the token the model actually writes, typically a bold-markup fragment or a generic first word such as FC. At the penultimate layer, the readout agrees with the model’s own answer on 85.1% of episodes under the logit lens and 85.6% under the J-lens.

Layers. $\Delta _ { \mathrm { n e w } }$ and $\Delta _ { \mathrm { o l d } }$ are averaged per episode over the upper third of the stack (layers 40–59 of 61 for gemma-4-31b, 16–23 of 25 for gpt-oss-20b and

<table><tr><td></td><td></td><td></td><td></td><td colspan="3"> $\Delta _ { \mathrm { n e w } }$ </td><td colspan="3"> $\Delta _ { \mathrm { o l d } }$ </td></tr><tr><td>Model</td><td>Lens</td><td>Updated</td><td>Failed</td><td>updated</td><td>failed</td><td>p</td><td>updated</td><td>failed</td><td>p</td></tr><tr><td>gemma-4-31b</td><td>logit</td><td>62</td><td>9</td><td>19.03</td><td>5.67</td><td>0.0054</td><td>-21.19</td><td>-0.48</td><td>0.0001</td></tr><tr><td></td><td>J-lens</td><td>62</td><td>9</td><td>27.40</td><td>10.27</td><td>0.0010</td><td>-21.22</td><td>3.80</td><td>0.0002</td></tr><tr><td>gpt-oss-20b</td><td>logit</td><td>23</td><td>34</td><td>2.34</td><td>0.92</td><td>0.0003</td><td>-2.39</td><td>-0.81</td><td>0.0001</td></tr><tr><td></td><td>J-lens</td><td>23</td><td>34</td><td>2.46</td><td>1.29</td><td>0.0014</td><td>-2.59</td><td>-0.43</td><td>0.0003</td></tr><tr><td>qwen3-8b</td><td>logit</td><td>40</td><td>13</td><td>9.46</td><td>3.42</td><td>&lt;0.0001</td><td>-7.39</td><td>-2.41</td><td>0.0004</td></tr><tr><td></td><td>J-lens</td><td>40</td><td>13</td><td>12.49</td><td>5.13</td><td>&lt;0.0001</td><td>-8.12</td><td>-1.87</td><td>0.0004</td></tr></table>

Table 13: Promotion of the retrieved club $\left( \Delta _ { \mathrm { n e w } } \right)$ and suppression of the asserted one $( \Delta _ { \mathrm { o l d } } )$ when the page is added, averaged over the upper third of layers. Episodes are split by whether the model updated or repeated the asserted club (failed), with episode counts. p: Mann–Whitney U, updates against failures. In failures, $\Delta _ { \mathrm { n e w } }$ is above zero in every row $( p \leq 0 . 0 4 )$ $\Delta _ { \mathrm { o l d } }$ is below zero for gpt-oss under both readouts and for qwen under the logit lens $( p \leq 0 . 0 1 4 ) .$ , but not for gemma $( p \ge 0 . 0 7 )$ or for qwen under the J-lens $( p = 0 . 0 7 )$ . The tests against zero are exact one-sample Wilcoxon signed-rank tests. Compare magnitudes within a model only.

24–35 of 37 for qwen3-8b), then across episodes within each outcome. Groups are compared with the Mann–Whitney U test. We exclude layer 0 and the final layer; at the final layer, the J-lens is the identity. The J-lens is severely ill-conditioned: for qwen3-8b, its condition number is $9 . 8 \times 1 0 ^ { 7 }$ , and 90% of the squared spectrum lies in 48 of 4,096 directions. We therefore compare magnitudes only within a model. Ranks are read at the penultimate layer (Table 14).

<table><tr><td>Model</td><td>Outcome</td><td>n</td><td>Retr.</td><td>Stale</td><td>Retr. top-15</td></tr><tr><td rowspan="2">gemma</td><td>updated</td><td>62</td><td>1</td><td>&gt;3000</td><td>69%</td></tr><tr><td>failed</td><td>9</td><td>69</td><td>5</td><td>33%</td></tr><tr><td rowspan="2">gpt-oss</td><td>updated</td><td>23</td><td>1</td><td>47</td><td>87%</td></tr><tr><td>failed</td><td>34</td><td>134</td><td>50</td><td>35%</td></tr><tr><td rowspan="2">qwen</td><td>updated</td><td>40</td><td>1</td><td>9</td><td>83%</td></tr><tr><td>failed</td><td>13</td><td>11</td><td>2</td><td>54%</td></tr></table>

Table 14: Median rank of the retrieved and stale clubs at the penultimate layer under the logit lens, and the share of facts with the retrieved club in the top 15.

Worked example. Take qwen3-8b on Anthony Gordon (Figure 3). At layer 35, adding the page raises Barcelona’s logit-lens score by 15.59 and lowers Newcastle United’s by 5.96. Thus $\Delta _ { \mathrm { n e w } } =$ $+ 1 5 . 5 9 , \Delta _ { \mathrm { o l d } } = - 5 . 9 6$ , and Barcelona gains 21.55 on Newcastle (Table 15). The J-lens moves both clubs the same way, but further. With the page, Barcelona’s first token is the model’s top prediction at that layer under both lenses, and the model answers Barcelona; without it, the model answers Newcastle United. On the same prompt, gpt-oss-20b keeps Newcastle. As in other failures, the retrieved club still rises, but the asserted one falls too little to be overtaken (Table 13).

<table><tr><td>Layer 35</td><td>Logit lens</td><td>J-lens</td></tr><tr><td> $\Delta _ { \mathrm { n e w } }$  (Barcelona)</td><td>+15.59</td><td>+30.37</td></tr><tr><td> $\Delta _ { \mathrm { o l d } }$  (Newcastle United)</td><td>-5.96</td><td>-11.50</td></tr><tr><td>Barcelona&#x27;s gain on Newcastle</td><td>21.55</td><td>41.87</td></tr></table>

Table 15: Worked example: the change in each club’s score when the page is added, at layer 35 for qwen3-8b on Anthony Gordon (old club Newcastle United, new club Barcelona).

Ranks by layer. Figure 4 traces the median rank of both clubs through the stack.

## K Re-Asking after the Evidence

This experiment tests whether restating the question after each opened page helps models use the evidence. The re-run takes every fact-sheet check with a required answer line, with and without a source request, whose original reply under frozen retrieval did not name exactly the stale line: 222 episodes as run, of which 157 remain after the exclusions applied throughout the paper. Each open-page result returns the page text followed by the full prompt, so the question is the most recent thing the model has read; a repeated open-page call carries it too. Nothing else differs: the menu, page bytes, sampler, models and episode identifiers are the same. A reply counts as correct only if it names exactly the stale line, and unparseable verdicts count as failures. Table 16 lists the six configurations re-run, after exclusions.

Answered correctly n = 40 · median, IQR band (J-lens)  
![](images/e41a2272f10118b971a739f95fb00ddd21b01dba3252dd1ae440e35cd6f2c475.jpg)

Failed: repeated the asserted club n = 13 · median, IQR band (J-lens)  
![](images/2728ca306e31302bad6d6c5240fde69aab426ba1641287475675cf0ce03888a1.jpg)

Answered correctly n = 23 · median, IQR band (J-lens)  
![](images/6cd9611337f6a8aeecf7db29002ee1dee909608b229feafaee355f223088b8a2.jpg)  
Failed: repeated the asserted club n = 34 · median, IQR band (J-lens)

![](images/bb630e8e3f9c89c49887a619e5b137a2d00ce39a70e69c12ce44dec12419156c.jpg)

![](images/f45ff76b18a18681d2e4896dd56474ad2919ca3d1d1c22a75ebb23698e21c855.jpg)

Failed: repeated the asserted club n = 9 · median, IQR band (J-lens)  
![](images/7a90dd44496388b616ea8e513068e1ea4d43dca54b86289ac8b630f27716e61d.jpg)  
retrieved club (on the page) club the fact sheet asserted J-lens logit lens

Figure 4: Median rank of the retrieved club (green, circles) and the asserted club (orange, squares) by layer, for qwen3-8b, gpt-oss-20b and gemma-4-31b (top to bottom), in updates (left) and failures (right). Solid lines: J-lens; dashed: logit lens; shaded band: J-lens interquartile range; grey: the upper third of layers used for the averages; 3000+: outside the recorded window. Both clubs sit at or near the floor for most of the stack and separate only in the final third: in updates the retrieved club overtakes and reaches the top, while in failures the asserted club ends ahead.

<table><tr><td>Model</td><td>Failures</td><td>Recovered</td><td>Rate</td></tr><tr><td>gemma-4-31b-think</td><td>7</td><td>4</td><td>57.1</td></tr><tr><td>Luna</td><td>9</td><td>3</td><td>33.3</td></tr><tr><td>gpt-oss-20b</td><td>34</td><td>11</td><td>32.4</td></tr><tr><td>qwen3-8b</td><td>61</td><td>16</td><td>26.2</td></tr><tr><td>gpt-oss-120b</td><td>24</td><td>6</td><td>25.0</td></tr><tr><td>gemma-4-31b</td><td>22</td><td>1</td><td>4.5</td></tr><tr><td>Total</td><td>157</td><td>41</td><td>26.1</td></tr></table>

Table 16: Failed fact-sheet checks recovered when the question is restated after each opened page (rate in %), after the exclusions used throughout; as run, before exclusions, 55 of 222 (24.8%).