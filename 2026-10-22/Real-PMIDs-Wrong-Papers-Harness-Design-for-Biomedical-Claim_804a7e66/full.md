# Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents

Mohammed Saqlain Independent Researcher Bangalore, India saqlain.contact@gmail.com

## Abstract

Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49–59% for the other realistic harnesses), although these differences are not significant at our sample size and agents use fewer than half of their search budget. With gold abstracts, the same models reach 79–86% accuracy, a significant 17–24 point gap showing that retrieval, not reasoning, is the main bottleneck. Finally, a citation audit shows that closed-book models cite PubMed IDs that almost always exist (327/335) but are almost never relevant: a cross-family LLM judge rates only 3 of 327 existing citations as relevant to the claim, versus 67% for citations produced by the search agent. Existence checks are therefore insufficient safeguards for scientific agents. We release code, prompts, and all cached tool calls at https://github.com/saqlain2204/real-pmids-wrong-papers.

## 1 Introduction

Language agents that search the biomedical literature are quickly becoming a default component of scientific assistants: they formulate queries, read abstracts, and return verdicts with citations [Lála et al., 2023, Skarlinski et al., 2024, Jin et al., 2024]. The design space of such harnesses is large, but the empirical basis for choosing among designs is thin. Three questions arise for anyone building a literature agent for the life sciences: (i) Does giving a frontier model access to PubMed improve its verdicts over closed-book answering, or does parametric knowledge suffice? (ii) Is an agentic loop, in which the model writes and refines its own queries [Yao et al., 2023, Nakano et al., 2021], worth its additional cost over a fixed one-shot retrieval pipeline [Lewis et al., 2020]? (iii) When agents cite papers, are those citations faithful to what they actually read?

We study these questions in a controlled setting: expert-written biomedical claim verification on SciFact [Wadden et al., 2020], using the live PubMed E-utilities API as the only tool. We hold the backbone model, prompt format, and output schema fixed and vary only the harness: closed-book, single-shot retrieval, a ReAct-style multi-query agent, and an oracle that receives the gold abstracts. We run the same protocol with two backbones and decompose errors into retrievalfailures (the gold paper was never seen) and reasoningfailures (the gold paper was seen but the verdict was wrong).

Contributions. (1) A small, fully reproducible harness-ablation protocol for literature agents built on a public benchmark and a public API (code and cached tool calls released). (2) An empirical comparison of four harnesses across two backbones, with bootstrap confidence intervals and paired analyses. (3) A citation audit showing how closed-book models cite real but irrelevant papers, a failure mode that existence-based checks alone cannot catch.

## 2 Related Work

Scientific claim verification. SciFact [Wadden et al., 2020] introduced expert-written claims paired with evidence abstracts; subsequent systems such as MultiVerS [Wadden et al., 2022] operate over a fixed corpus of about 5K abstracts. We instead place the model in the open-world setting it would face in deployment, searching all of PubMed. Retrieval-augmented and agentic biomedical QA. MedRAG [Xiong et al., 2024] benchmarks retrieval-augmented generation for medical QA, GeneGPT [Jin et al., 2024] teaches LLMs to call NCBI APIs, and PaperQA/PaperQA2 [Lála et al., 2023, Skarlinski et al., 2024] build full literature agents. LAB-Bench [Laurent et al., 2024] evaluates biology-research capabilities including literature tasks. These works typically evaluate full systems; our aim is a minimal, controlled ablation of the harness itself. Citation fabrication. LLMs are known to fabricate bibliographic references [Walters and Wilder, 2023]. We extend this analysis to PubMed IDs and distinguish non-existent citations from existing but irrelevant ones.

## 3 Experimental Setup

Task and data. Each SciFact claim is labeled SUPPORT, CONTRADICT, or NOT ENOUGH INFO (NEI) with respect to its cited abstracts. Because SciFact test labels are hidden, we use the labeled dev split (300 claims) and, due to computational budget constraints, evaluate a fixed random sample of N=100 claims (seed 0), shared by all conditions. We report accuracy and macro-F1 over the three labels, with 95% bootstrap confidence intervals (2,000 resamples).

Harnesses. All conditions share the same instructions and require a JSON output {label, cited\_pmids, rationale}, at temperature 0.

• Closed-book: the model answers from parametric knowledge and may cite PMIDs it believes are relevant.

• Single-shot RAG: the claim, with stopwords removed, is issued as one PubMed query (relevance sort, top-5). If PubMed returns nothing (terms are ANDed), we relax to an OR over content words. Titles and abstracts (truncated to 1,500 characters) are given to the model.

• Agentic search: a ReAct-style loop [Yao et al., 2023] in which, at each step, the model either issues a PubMed query of its own design or answers. It has a budget of 4 searches (top-5 each, deduplicated) and is told PubMed’s query semantics.

• Oracle: the model receives the SciFact-cited gold abstracts directly. This upper-bounds performance given perfect retrieval and isolates reasoning over evidence.

Backbones. We use two efficient frontier models from the same provider, gemini-3.1-flash-lite and gemini-3.5-flash-lite, chosen for their low latency and cost. Tool calls and model outputs are cached to make every number reproducible from the released logs.

Retrieval and citation diagnostics. SciFact documents are identified by Semantic Scholar IDs, not PMIDs, so we map a retrieved PubMed record to a gold document by normalized title match (exact or ratio > 0.9). Gold recall is the fraction of SUPPORT/CONTRADICT claims for which at least one gold abstract was retrieved. For retrieval conditions, a citation is unfaithful if its PMID was never returned by the tool. For closed-book citations, we check (a) existence in PubMed, (b) whether it matches a gold paper, and (c) topical relevance, judged by a model from a different family (qwen3.8-27b) given the claim and the cited title and abstract. We apply the same relevance judge to agent citations for comparison.

## 4 Results

Table 1 and Figure 1 summarize the results. We highlight five findings.

Table 1: Claim verification on 100 SciFact dev claims (95% bootstrap CIs in brackets). Gold recall: fraction of SUPPORT/CONTRADICT claims (n=64) for which a SciFact gold abstract was retrieved. #Searches: mean PubMed queries per claim.
<table><tr><td>Backbone</td><td>Harness</td><td>Acc. (%)</td><td>Macro-F1 (%)</td><td>Gold recall (%)</td><td>#Searches</td></tr><tr><td rowspan="4">Gemini-3.1-Flash-Lite</td><td>Closed-book</td><td>55.0 [45,64]</td><td>46.5 [38,55]</td><td>一</td><td></td></tr><tr><td>Single-shot RAG</td><td>53.0 [44,63]</td><td>52.5 [41,62]</td><td>6</td><td>1.00</td></tr><tr><td>Agentic search</td><td>62.0 [52,71]</td><td>60.3 [50,69]</td><td>19</td><td>1.42</td></tr><tr><td>Oracle abstracts</td><td>86.0 [79,92]</td><td>85.8 [78,92]</td><td>一</td><td></td></tr><tr><td rowspan="4">Gemini-3.5-Flash-Lite</td><td>Closed-book</td><td>59.0 [49,69]</td><td>53.6 [44,63]</td><td>一</td><td></td></tr><tr><td>Single-shot RAG</td><td>49.0 [39,59]</td><td>46.0 [35,57]</td><td>6</td><td>1.00</td></tr><tr><td>Agentic search</td><td>62.0 [52,71]</td><td>61.1 [50,70]</td><td>20</td><td>1.70</td></tr><tr><td>Oracle abstracts</td><td>79.0 [71,87]</td><td>78.6 [70,86]</td><td>一</td><td></td></tr></table>

![](images/68d6b47eab7d8c430a8d29bcf8f281c6bd0531b48bee93bc49ba0640b20e2c43.jpg)

![](images/0067d0b082093ed02f5b1d974ca065db218e17c12544f8f466386ccde7c69a7d.jpg)  
Figure 1: (a) Macro-F1 by harness with 95% bootstrap CIs. (b) Accuracy on SUPPORT/CONTRADICT claims, split by whether a gold abstract was retrieved (gold-hit groups are small: n=4 for RAG and n=12/13 for the agent).

F1: Naive one-shot retrieval does not help, and changes how the model fails. Issuing the claim itself as a PubMed query retrieves a gold abstract for only 6% of evidence-bearing claims (4/64 for both backbones). Given mostly off-target abstracts, both models abstain heavily: 59% and 70% of their predictions are NEI, against 36% NEI in the gold labels. Accuracy is unchanged or lower relative to closed-book (53.0 vs. 55.0 and 49.0 vs. 59.0; exact McNemar p=0.89 and p=0.24). Retrieval converts overconfident errors into abstentions rather than into correct verdicts. When the gold abstract was retrieved, both models were correct on all 4 such claims.

F2: Letting the model write its own queries triples gold recall and gives the best verdicts, but the gains are modest. The agentic harness raises gold recall to 19–20% and attains the highest accuracy (62.0 for both backbones) and macro-F1 (60.3 and 61.1) of the non-oracle harnesses. Its queries are short, entity-centric reformulations (e.g., “chenodeoxycholic acid energy expenditure” instead of the full claim), and they can repair misspellings in the claim (“Chenodeosycholic” → “chenodeoxycholic”), which silently break literal retrieval. In paired comparisons, however, the improvement is not statistically significant at N=100: agent vs. single-shot, 24 vs. 15 discordant claims (p=0.20) and 29 vs. 16 (p=0.07); agent vs. closed-book, 13 vs. 6 (p=0.17) and 11 vs. 8 (p=0.65). The direction is consistent across both backbones.

F3: Agents under-use their search budget. Despite a budget of 4 searches, the agents issue 1.42 and 1.70 queries per claim on average and answer after a single search on 71% and 55% of claims; only 3% and 6% of claims use the full budget. Among evidence-bearing claims where the agent never saw a gold abstract, it still answered SUPPORT/CONTRADICT correctly 75% of the time, suggesting that it draws on other relevant PubMed papers or on parametric knowledge. But it also abstains less on true NEI claims (33% and 42% NEI recall, versus 72% and 81% for single-shot). Budget use is therefore a clear axis for harness improvement, for example through a minimum-evidence criterion before answering.

F4: Retrieval, not reasoning, is the main bottleneck. Given the gold abstracts, the same models reach 86.0% and 79.0% accuracy (macro-F1 85.8 and 78.6), 24 and 17 points above the best realistic harness. Unlike the differences among realistic harnesses, this gap is highly significant (agent vs. oracle, 5 vs. 29 discordant claims, $p { < } 1 0 ^ { - 4 }$ ; 9 vs. 26, p=0.006). The oracle also restores calibrated abstention (NEI recall 78% and 86%) without sacrificing SUPPORT/CONTRADICT accuracy, so the models reason well over evidence once they are shown the right papers.

F5: Closed-book citations are real papers, but almost never relevant ones. Closed-book models cited PMIDs for 97 and 80 of the 100 claims. Almost all cited PMIDs exist in PubMed (186/190 and 141/145), so existence checks would pass them. None matched a SciFact gold paper, however, and the cross-family judge rated only 1 of 186 and 2 of 141 as relevant to the claim. Typical examples: a claim about TNF-α/IL-1 and IL-6/IL-10 regulation is supported by “[The proper eyeglassesfor children]”, and a claim about headache and cognitive impairment by “Improving access to appointments at Laurie Pike Health Centre”. The judge itself is well calibrated on controls: it labels 21/25 gold abstracts relevant and 0/25 mismatched gold abstracts relevant. Grounding changes this sharply. The same judge rates 66% (141/215) and 70% (128/184) of agent-cited PMIDs as relevant, and 77% and 84% of agent verdicts that cite anything cite at least one relevant paper. Agents almost never cite outside what they retrieved (0% and 2.6% of citing claims). Closed-book models are also overconfident: they predict NEI for only 4% and 9% of claims, recovering 6% and 17% of true NEI claims, so their competitive accuracy rests entirely on SUPPORT/CONTRADICT claims.

## 5 Discussion and Limitations

Implications for harness design. (1) Query formulation appears to be where the model adds the most value in a literature agent: letting the LLM reformulate the query triples recall of the right paper and gives the best verdicts on both backbones, whereas passing the claim straight to the search engine does not help. (2) The largest remaining headroom lies in retrieval, as the 17–24 point gap to the oracle shows, so iterative search that is actually used (F3), citation chasing, and domain retrievers are more promising than stronger verdict prompting. (3) Citations from ungrounded generation should never be surfaced, even when the IDs resolve: validating that a PMID exists is not sufficient, and relevance must be checked against the retrieved text.

Limitations. This is a pilot study. N=100 claims per condition gives wide confidence intervals, and none of the accuracy differences between realistic harnesses is significant. Both backbones are small, efficient models from a single provider, chosen for their low latency and cost, so results may differ for larger or open-weight models. SciFact labels are defined with respect to specific cited abstracts, whereas our harnesses search all of PubMed as of 2026. Some “errors” may therefore reflect newer or different evidence, and title-matched gold recall is a lower bound on useful retrieval. Relevance judgments come from a single LLM judge, validated only on 50 controls. We plan to scale to the full dev set, add open-weight and biomedical-specialist backbones, and study budget-forcing strategies for agentic search.

Reproducibility. Code, prompts, all raw model outputs, and the cached PubMed responses are available at https://github.com/saqlain2204/real-pmids-wrong-papers. The released cache reproduces every reported number without API access.

## Acknowledgments and Disclosure of Funding

The author received no specific funding for this work and declares no competing interests.

## References

Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. GeneGPT: Augmenting large language models with domain tools for improved access to biomedical information. Bioinformatics, 40(2):btae075, 2024. doi: 10.1093/bioinformatics/btae075.

Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, and Andrew D. White. PaperQA: Retrieval-augmented generative agent for scientific research. arXiv preprint arXiv:2312.07559, 2023.

Jon M. Laurent, Joseph D. Janizek, Michael Ruzo, Michaela M. Hinks, Michael J. Hammerling, Sid dharth Narayanan, Manvitha Ponnapati, Andrew D. White, and Samuel G. Rodriques. LAB-Bench: Measuring capabilities of language models for biology research. arXiv preprint arXiv:2407.10362, 2024.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language agents achieve superhuman synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740, 2024.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, 2020. doi: 10.18653/v1/2020.emnlp-main.609.

David Wadden, Kyle Lo, Lucy Lu Wang, Arman Cohan, Iz Beltagy, and Hannaneh Hajishirzi. MultiVerS: Improving scientific claim verification with weak supervision and full-document context. In Findings ofthe Associationfor Computational Linguistics: NAACL 2022, pages 61–76, 2022. doi: 10.18653/v1/2022.findings-naacl.6.

William H. Walters and Esther Isabelle Wilder. Fabrication and errors in the bibliographic citations generated by ChatGPT. Scientific Reports, 13:14045, 2023. doi: 10.1038/s41598-023-41032-5.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 6233–6251, 2024. doi: 10.18653/v1/2024.findings-acl.372.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

## A Prompts

Shared task instruction. “You are a biomedical fact-checking assistant. Decide whether the scientific literature SUPPORTS the claim, CONTRADICTS it, or provides NOT ENOUGH INFO (NEI). Use NEI when the evidence does not directly address the claim.” All conditions require the JSON output {"label", "cited\_pmids", "rationale"}.

Closed-book adds: “You have no access to tools; rely on your own knowledge. If you know specific papers, cite their PubMed IDs.” Single-shot RAG adds: “Base your verdict only on the retrieved abstracts below; cite only PMIDs that appear in them.” Oracle adds: “Base your verdict only on the abstracts below.”

Agentic search adds: “You can search PubMed. PubMed search is keyword-based (terms are ANDed), so short focused queries of 2–6 key terms work best; you may use OR and field tags like [tiab]. You have at most 4 searches.” At each step the model emits either {"action":"search","query":...} or {"action":"answer",...}. Only unseen results are shown after each search, and the model is forced to answer once the budget is exhausted.

Citation relevance judge (qwen3.8-27b): “Answer ‘yes’ only if the paper studies the specific entities and relationship in the claim (so it could plausibly support or refute it); answer ‘no’ if it is off-topic or only loosely related.”

## B Implementation details

PubMed is accessed through NCBI E-utilities (esearch with sort=relevance, retmax=5; efetch for titles and abstracts). All LLM calls use temperature 0 and JSON output mode. Every tool response and model output is cached, and the released cache reproduces all tables without API access. Experiments query live PubMed, so retrieval results reflect the index at the time of the run.