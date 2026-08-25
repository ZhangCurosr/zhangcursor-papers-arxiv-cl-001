# Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing

Nazib Sorathiya Daniel Zhang Bardiya Akhbari

Amazon AGI

Seattle, WA, USA {nsorath, nyudan, bardiyaa}@amazon.com

## Abstract

Agent ecosystems now include thousands of MATS components (Models, Agents, Tools, and Skills), yet their discovery still relies on in-context routing. These systems read a registry (names, hints, or descriptions, as context budget permits), pick a candidate, invoke it, and retry on failure. This pattern degrades with scale, and registries are growing fast. We recast capability discovery as search over a registry by defining an offline enrichment step that turns sparse metadata into searchable profiles, and an online retrieve-then-rank pipeline that returns a ranked shortlist without invoking any candidates online. We show that from N = 10 to 7,278 capabilities, in-context routing’s top-1 accuracy (Match@1) collapses (0.85 → 0.12), while retrieve-then-rank degrades more gently (0.81 → 0.39) because its reranker still ranks the right capability first 0.70–0.87 of the time once retrieval finds it. In the Nova Micro sweep, the crossover is around N = 500. We compare against two in-context baselines. Full-Ctx puts the whole registry in the prompt and asks the LLM to pick. Search&Pick gives the LLM a search tool to narrow candidates before it picks. At full scale the pipeline leads Search&Pick by 6.5 percentage points (pp) on Match@1 at about half the cost, and reduces the cost 70× versus Full-Ctx. We use the same configuration (same enrichment, retriever, and scorer weights) across agent, tool, and skill registries. The pipeline runs in production as the default capability-discovery layer of a largescale multi-agent platform.

## 1 Introduction

Recent agentic systems (Anthropic, 2025; Cursor, 2024; LangChain, 2024; Amazon Web Services, 2024) expose registries of thousands of components. Our production registry unifies all MATS components under one schema and already exceeds 500 entries in every MATS type; the largest benchmark we study holds 7,278 tools. At that scale an orchestrator cannot place every candidate in context, so choosing what to call requires first narrowing what to consider.

Today’s systems discover capabilities by incontext routing. The large language model (LLM) sees a representation of the registry (tool names, one-line hints, or full descriptions as budget permits), picks a candidate, invokes it, and retries on failure. Some systems truncate alphabetically; others use a regular expression tool to narrow candidates before picking. All couple cost and accuracy to registry size. Every wrong invocation costs tokens and latency, and may call an untrusted endpoint just to learn what it does.

We frame capability discovery as web search over a capability registry rather than the open web. A search engine does not make users click through every indexed page, and an orchestrator should not present every capability and rely on the LLM to pick. We retrieve ranked candidates without invoking them and return a shortlist. The advantage over in-context routing widens with registry size, so it matters most where the field is heading.

We decompose end-to-end performance into retrieval recall and reranker conditional accuracy. The reranker reaches near-peak accuracy at modest registry sizes and stays flat beyond that, so at larger scales most misses originate in retrieval. At full scale, our pipeline leads Search&Pick in match accuracy and uses half the tokens.

We apply one pipeline to agent, tool, and skill registries under a single configuration and deploy it in production as the capability-discovery layer for a multi-agent platform. The pipeline includes offline enrichment (an LLM rewrites sparse metadata into rich profiles once at registration time, not per query). Its value scales with metadata sparsity. On well-documented public data the effect is neutralto-negative (§4.3). We include it as a production design choice, not a source of benchmark gains.

![](images/82b4c74d8426cd3dbb6ef404bb4d57168d0be6435c785ffcd33b5e012b728370.jpg)  
Figure 1: The Enrich-Retrieve-Rank pipeline. Phase 1 transforms sparse capability metadata into rich profiles. Phase 2 retrieves top-k candidates from the enriched index. Phase 3 ranks with a single LLM call. The entire online path requires one LLM call at bounded latency. Dashed boxes are data, and solid boxes are computation.

## 2 Related work

Tool retrieval. Toolformer (Schick et al., 2023) self-supervises a fixed API set; Gorilla (Patil et al., 2024) trains end-to-end on APIs; ToolLLM (Qin et al., 2024) scales the endpoints with a retriever and depth-first-search caller. All fix the API pool at train time, leaving retrieval implicit. ToolRet (Shi et al., 2025) strips the pre-annotated tool list and shows that general information retrieval (IR) models perform poorly on the resulting task. CRAFT (Yuan et al., 2024a) and EasyTool (Yuan et al., 2024b) are the closest enrichment-side methods. CRAFT abstracts code into per-task toolsets; Easy-Tool rewrites API docs into instructions. PLUTO (Huang et al., 2024a) pairs planning with description editing and is the closest prior recipe to ours. It does not quantify the retrieve-vs-rerank decomposition or study scale. AnyTool (Du et al., 2024) uses a hierarchical agent over 16k APIs but does not separate retrieval from MATS invocation. Meta-Tool (Huang et al., 2024b) and Qu et al. (2024) confirm tool selection is the dominant failure mode but propose no retrieve-then-rank pipeline.

Agent routing. AppWorld (Trivedi et al., 2024) keeps agent selection implicit because the registry fits in context (n = 8). HuggingGPT (Shen et al., 2023) and Visual ChatGPT (Wu et al., 2023) pick from a static menu. Our Nova Micro sweep shows that this pattern breaks around N = 500. AFlow (Zhang et al., 2024) and Guo et al. (2024) frame the challenge as workflow construction over heterogeneous agents; we contribute the retrieval primitive.

Capability discovery. The Model Context Protocol (MCP) (Anthropic, 2024), Agent2Agent (Google, 2025), and the GPT Store (OpenAI, 2024) standardize capability enumeration and invocation but assume the client already knows which server to call. ToolkenGPT (Hao et al., 2023) embeds tools as soft tokens but ties embeddings to a specific checkpoint. Mulang’ et al. (2026) instead use a knowledge-graph index over MCP servers; their graph-based filtering improves discovery at 269 tools but not on smaller curated menus.

IR, dense embeddings, and LLM reranking. Capability discovery mirrors web search as the system matches a user intent against a large corpus of documents (capabilities) indexed offline. Our pipeline reuses the retrieve-then-rerank architecture web search engines refined over decades, adapted to capability metadata rather than web pages. BM25 (Robertson and Zaragoza, 2009) is our lexical baseline. doc2query (Nogueira et al., 2019) expands documents at index time with predicted queries; our enrichment writes typed structured fields instead, consumed by both retriever and reranker. HyDE (Gao et al., 2022) generates a hypothetical document at query time; we do the analogous work once at registration. For dense retrieval we use BGE-large-en-v1.5 (Xiao et al., 2024) and Amazon Titan Embed V2 (1024d); E5- Mistral (Wang et al., 2024) and GTE (Li et al., 2023) are left to future work. RankGPT (Sun et al., 2023) showed listwise LLM rerankers match supervised cross-encoders. Two concurrent works apply enrich-then-retrieve to tools independently: Tool-DE (Lu et al., 2025) and Multi-Field Tool Retrieval (Tang et al., 2026). Neither studies how performance changes as the registry grows.

Our contribution. This is a systems-and-scaling study, not a new-model study. The pipeline composes established components (offline enrichment (Gao et al., 2022; Nogueira et al., 2019; Yuan et al., 2024a,b; Huang et al., 2024a); BM25/BGE retrieval; listwise reranking (Sun et al., 2023)); three contributions are new. First, we map the full degradation curve from N = 10 to 7,278 and locate the crossover around N = 500 in the Nova Micro sweep. Prior work reports degradation at a single scale (Shi et al., 2025; Du et al., 2024). Second, we decompose retrieval from reranking. The fixed-k reranker holds a 0.70–0.87 conditional-accuracy band, while ∼70% of large-registry misses occur at retrieval. Third, we compare our results against trial-and-error baselines grounded in how deployed harnesses route, measuring accuracy, tokens, latency, and cost (absent from prior literature).

## 3 Method

## 3.1 Problem formulation

Given a query $q$ and a capability registry $C =$ $\{ c _ { 1 } , \ldots , c _ { N } \}$ , we produce a ranked list $\{ ( c , s _ { c } ) \}$ without invoking any $c _ { i }$ online. Each $c _ { i }$ belongs to one type; our experiments cover Tools, Agents, and Skills, leaving model routing to existing work. The orchestrator consumes the top-k list and picks.

## 3.2 Offline enrichment

At registration time (once per capability, never per query), an LLM rewrites sparse metadata into a structured profile with five typed fields of a capability summary, an action-verb-led description, differentiating keywords, and positive and negative usage examples. We add a trust score and selfreported capability tags for production registries.

These typed fields feed both the retriever and the reranker, unlike HyDE (Gao et al., 2022), which generates a hypothetical document per query at inference time, or doc2query (Nogueira et al., 2019), which expands documents with surrogate queries for a single retrieval index. The retriever indexes them lexically and densely; the reranker reads them verbatim when scoring candidates. We concatenate these fields into the BM25 index and encode with BGE-large-en-v1.5 or Amazon Titan Embed V2.

## 3.3 Online retrieve-then-rank

We retrieve the top $k \in \{ 1 5 , 2 5 \}$ candidates using BM25, neural, or hybrid retrieval. We then combine four scores into [0, 1]: LLM (weight 0.50, one API call), BM25 (0.05), Quality (0.30), and Intent (0.15). Quality and Intent drop out on public registries because they lack trust and type metadata.

The two production-only signals use fields from the enriched profile (§3.2). Quality is the normalized offline trust score; Intent matches a queryinferred capability type to the candidate’s selfreported type tags. A signal drops out when its fields are missing, and the remaining weights are renormalized. Thus, all public results use only LLM and BM25 at a fixed 10:1 ratio. We set the four weights once according to each signal’s role and do not tune them by capability type or dataset.

## 4 Experiments

## 4.1 Setup

We evaluate our pipeline across three capability types and five dataset/type combinations (Table 1). We report Match@k $( k \in \{ 1 , 3 , 5 \} )$ as the fraction of queries for which any ground-truth capability appears in the top-k returned list, mean reciprocal rank (MRR) as the average of 1/rank of the first correct result, and Recall@k as the fraction of ground-truth capabilities retrieved in the top-k (relevant when a query maps to multiple ground-truth). We compute bootstrap 95% confidence intervals (CIs) via $n = 1 0 0 0$ resamples; all reported deltas ≥ 5 pp exceed the 95% CI width (±1.5–3 pp on ToolRet, ±5–8 pp on the smaller AppWorld data).

Unless noted, we use Nova Micro as the reranker. A four-model experiment (Nova Micro, Nova Lite, Claude 3.5 Haiku, Claude Sonnet 4) confirms the method ordering is stable across rerankers.

<table><tr><td>Type</td><td>Benchmark</td><td>Reg.</td><td>Queries</td></tr><tr><td>Tools</td><td>ToolRet</td><td>7,278</td><td>7,961</td></tr><tr><td>Tools</td><td>AppWorld</td><td>332</td><td>147</td></tr><tr><td>Agents</td><td>ToolRet</td><td>5,885</td><td>7,961</td></tr><tr><td>Agents</td><td>AppWorld</td><td>8</td><td>147</td></tr><tr><td>Skills</td><td>MCP</td><td>859</td><td>1,627</td></tr></table>

Table 1: Capability × benchmark coverage.

## 4.2 Results

We report Match@1 across the benchmarks and methods (Table 2). Each baseline mirrors a routing strategy used in production: Regex (patternmatch, no LLM); Full-Ctx (full registry in-context, LLM picks); Search&Pick (LLM with a search tool to narrow candidates before picking); Trial&Err (LLM iteratively lists and tries).<sup>1</sup> We also include two retriever-only baselines (BM25 and BGE-largeen-v1.5), and three full-pipeline configurations (Ours+BM25, Ours+BGE-5f, and Ours+Titan).

We use Match@1 as our primary metric because it allows direct comparison with the single-pick baselines. The pipeline itself returns a ranked shortlist $( k \in \{ 1 5 , 2 5 \}$ , with k = 15 in production), so we also report Match@3, Match@5, MRR, and Recall@15 to measure whether the correct capability reaches a downstream selector. On Tools-ToolRet, for example, Ours+Titan reaches Match@3 0.523 and Match@5 0.558. Full results are reported in Appendix B (Tables 7, 8).

On Tools-ToolRet (7,961 queries, N=7,278 capabilities), Ours+Titan reaches Match@1 0.397, a 6.5 pp lead over Search&Pick (0.332) at half the token cost. The gain is broad as the pipeline beats Search&Pick on 13/16 Tools and 12/16 Agents sources with $n \geq 1 0 0$ (Appendix F). ToolRet includes near-duplicate items, and the error decomposition shows 70% of failures occur at retrieval. On small registries the pipeline loses its advantage. Full-Ctx leads on Tools-AppWorld (n=332, 0.76 vs. 0.74) and Skills-MCP. The Nova Micro sweep places the crossover near $N = 5 0 0 . ^ { 2 }$

Agents-ToolRet shares queries and 89% of its registry with Tools-ToolRet, so it is a nearreplicate rather than independent evidence. Agents-AppWorld saturates at 1.00 across its eight-agent registry. Skills-MCP’s query set paraphrases capability descriptions, giving lexical retrievers an edge. Public benchmarks do not yet support an independent agent- or skill-retrieval win at scale. Even on Skills-MCP, Ours+Titan (0.942) closes most of the gap to Full-Ctx (0.968), up from 0.807 with Ours+BM25. Titan’s retrieval recall accounts for the gain (R@15 0.994).

## 4.3 Ablations

We compare raw vs. LLM-enriched registries with everything else held fixed (same query set, registry size, retriever, k, weights, model; full numbers in Appendix A, Table 6). Enrichment is neutralto-negative on public data: ±0.1 pp Match@1 on ToolRet, 0.0 pp on AppWorld (with +3.4 pp Match@3), and −4.4 pp on MCP where added keywords dilute already-strong lexical overlap. A per-field ablation (ToolRet n = 500) shows no single field raises Match@1 by more than +0.6 pp (keywords, $0 . 8 4 2 \  \ 0 . 8 4 8 )$ Public registries already have clean, detailed descriptions, so enrichment fills no gap. Its production value scales with metadata sparsity, which public benchmarks cannot stress-test. We therefore construct a controlled stress test by degrading the same 1,000- tool ToolRet pool from full descriptions to firstsentence hints and names only, then comparing raw and enriched registries with the pipeline fixed. Enrichment improves Match@1 by +5.8, +9.1, and +25.6 pp, respectively (paired McNemar p < $1 0 ^ { - 6 } )$ . At name-only, Recall@15 rises from 0.134 to 0.467. The end-to-end gain therefore originates in retrieval (Figure 4; Appendix E).

## 4.4 Retrieval analysis

On large-registry ToolRet, ∼70% of misses occur at retrieval because BM25’s top 15 excludes the ground-truth capability (68% of Tools-ToolRet and 70% of Agents-ToolRet misses; Cat 1, Appendix D). AppWorld and MCP misses occur during reranking. Both neural retrievers raise Recall@15 over BM25 by +4.7–7.6 pp on the Tool-Ret rows (BGE-5f and Titan Embed V2; full numbers in Appendix B). The gain does not propagate cleanly to end-to-end Match@1. The three pipeline configurations sit within 0.9 pp on Tools-ToolRet (0.388, 0.389, 0.397) and 2.1 pp on Agents-ToolRet (0.397, 0.412, 0.418) because the fixed-k reranker’s conditional accuracy is flat across retrievers. Titan attains the best end-to-end Match@1 on Tools-ToolRet (0.397); on Agents-ToolRet, BM25 retrieval leads (0.418). We adopt Titan as the production retriever for its stronger recall (R@15 0.625 vs BM25 0.549) and its leading Match@1 on the Tools and Skills rows. The remaining recall gap is real, but the dense encoders close little of it end-toend. Doing so requires a much stronger first-stage retriever, not a better reranker. ToolBench-IR, a BERT-base retriever fine-tuned for tool-API relevance (Qin et al., 2024), does not improve the candidate pool. Its Recall@15 is 0.608 on Tools and 0.596 on Agents, below the general encoders (0.625 and 0.626); end-to-end Match@1 is lower (0.378 vs. 0.397 and 0.381 vs. 0.412; Appendix G). Existing domain-specific fine-tuning therefore does not remove the first-stage recall bottleneck.

## 4.5 Scale and latency

We ran a registry-size scan on a fixed 100-query Tools-ToolRet slice (178 unique tools) over twelve subsampled registries from $N = 1 0 \mathrm { t o } 7 , 2 7 8 . ^ { 3 }$ We compare Regex, Full-Ctx (Nova Micro full-context picker, 480k character budget, alphabetic truncation when exceeded), and Ours+BM25 (k=15 rerank). We show Match@1 vs. N and decompose the pipeline into retrieval recall and reranker conditional accuracy (Figure 2).

<table><tr><td>Row (n / q)</td><td>Regex</td><td>Full Ctx</td><td>Search &amp;Pick</td><td>Trial &amp;Err</td><td>BM25</td><td>BGE 5f</td><td>Ours +BM25</td><td>Ours +BGE</td><td>Ours +Titan</td></tr><tr><td>Tools / ToolRet (7,278 / 7,961)</td><td>0.231</td><td>0.166</td><td>0.332</td><td>0.054</td><td>0.311</td><td>0.346</td><td>0.388</td><td>0.389</td><td>0.397</td></tr><tr><td>Tools / AppWorld (332 / 147)</td><td>0.408</td><td>0.762</td><td>0.476</td><td>0.061</td><td>0.483</td><td>0.429</td><td>0.741</td><td>0.735</td><td>0.741</td></tr><tr><td>Agents / ToolRet (5,885 / 7,961)</td><td>0.228</td><td>0.148</td><td>0.341</td><td>0.045</td><td>0.336</td><td>0.360</td><td>0.418</td><td>0.412</td><td>0.397</td></tr><tr><td>Agents / AppWorld (8 / 147)</td><td>0.993</td><td>0.986</td><td>0.973</td><td>0.014</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>Skills / MP, leaky (859 / 1,627)</td><td>0.648</td><td>0.968</td><td>0.829</td><td>0.624</td><td>0.880</td><td>0.892</td><td>0.807</td><td>0.806</td><td>0.942</td></tr></table>

Table 2: Match@1 splits by registry size (n) and query count (q): the pipeline leads on large registries while in-context routing wins for the smaller sizes. Ours+BGE is the 5-field BGE config. Bold marks the best result in each row. Ours+Titan uses Amazon Titan Embed V2 as the dense retriever. Full-Ctx uses an $n = 5 0 0$ sub-sample on Tools-ToolRet, Agents-ToolRet, and Skills-MCP because the full registry exceeds Nova Micro’s context window; every other cell is full-n. Trial&Err scores its final try and can fall below the 1/n random-pick rate when it abstains rather than guessing; see footnote. Other metrics are in Appendix B.

Below $N \ = \ 5 0 0 .$ , full-context LLM picking works; above it, Full-Ctx drops −13 pp per doubling and collapses once alphabetic truncation triggers at $N \ge 5 , 0 0 0$ (4,023 of 7,278 tools silently dropped). The crossover is around $N = 5 0 0$ in this Nova Micro sweep and shifts with context size. Native-size AppWorld and MCP points are consistent with the pattern but do not establish a shared threshold (Figure 3).

The scale decomposition explains the gap since the single-stage picking collapses from Match@1 0.85 to 0.12 as N grows, while the reranker’s conditional accuracy holds steady at 0.70–0.87 across all registry sizes. The pipeline therefore degrades far more gracefully (0.81 to 0.39). Further gains require better retrieval (Appendix C).

## 4.6 Cross-type generalization

We use one configuration (same enrichment template, retriever stack, and scorer weights) to produce every pipeline cell in Table 2 with no pertype tuning. Three choices make this possible: the enrichment template is type-agnostic (summary, action-led description, keywords, examples, regardless of target), the retrieval signal is typeblind (BM25 and the dense encoder score any text field the same way), and the reranker reads the same prompt shape (a ranked list of enriched profiles) across all three registries. The results are consistent with registry size determining the ad vantage, but the benchmarks do not separate size from capability type. Current benchmarks limit the cross-type claim. Agents-AppWorld saturates, Agents-ToolRet overlaps Tools-ToolRet, and Skills-MCP queries paraphrase capability descriptions. A stronger test requires agent and skill registries above 500, real user intents, and multi-ground-truth trajectory labels (Appendix D).

## 4.7 Cross-model robustness

We test four rerankers on Tools, with Nova Micro as the reference. The model ordering is stable from n = 500 to full n. The ordering is Claude Sonnet 4 > Nova Micro ≈ Nova Lite ≫ Claude 3.5 Haiku. At full $n ,$ Claude Sonnet 4 improves Match@1 by only 3 pp, consistent with conditional accuracy inside the 0.70–0.87 band (Qwen3 Next and Mistral Large 3, Table 5). Claude 3.5 Haiku is the exception, collapsing from 0.496 to 0.280 on output truncation. Nova Micro is the production choice; Claude Sonnet 4 is acceptable only on the smaller registry (Table 4; Appendix A).

## 4.8 Efficiency

We compare retrieve-then-rank against the incontext routing baselines on accuracy and cost. Full per-row breakdowns (tokens, dollars, and latency) appear in Appendix A, Table 3. All pricing uses Nova Micro managed-endpoint retail rates. Baseline rows reflect real token counts. Pipeline rows are post-hoc estimates (±5–10%). On Tools-ToolRet, Ours+BM25 is more accurate than every LLM-using in-context baseline. It is ∼2× cheaper than Search&Pick and ∼70× cheaper than Full-Ctx (\$0.066 vs. \$4.48 per 1,000 queries). On both large ToolRet rows, it is more accurate and uses fewer tokens than Full-Ctx and Search&Pick. Full-Ctx is cheaper on Agents-AppWorld, and lexical baselines are more accurate on Skills-MCP. Four failure examples are in Appendix D.

## 5 Deployment

The deployment evidence is architectural rather than an online evaluation, and our operational evidence is limited to cost and latency (Table 3). We deploy the pipeline in production as the capabilitydiscovery layer for an internal multi-agent platform. All four MATS types coexist in a single index with shared enrichment schema and scorer weights. Offline enrichment runs at registration time. An LLM rewrites sparse metadata into the five-field profile. The profile is stored in a keyvalue store and encoded with Titan Embed V2. Generation cost is paid once per capability update, never per query. The online path is a serverless AWS Lambda: BM25 retrieval over the enriched corpus, neural re-scoring via the dense index, and a single Nova Micro reranker call over k = 15 candidates, with all four scoring signals active where the registry includes trust and type metadata.

![](images/70e486eb82c728087a867ced658092da6aa776b904a6928e2a65a495a310204b.jpg)

![](images/482a73a3e117352cf27c1aae0ecb39d67a33612ab92c422c74885ebb6d392b1a.jpg)  
Figure 2: Within-benchmark scaling. Match@1 vs. N (Left) and pipeline decomposition (Right).

The production registry exceeds 500 across all four MATS types, in the regime where single-stage LLM routing fails (−13 pp Match@1 per doubling past the crossover), so retrieve-then-rank is an architectural requirement, not a preference.

The public configuration uses two signals (LLM + BM25); production also uses Quality and Intent. On an internal registry with the required trust and type metadata, these signals add +4.5 pp Match@1 (p = 0.031), with retrieval fixed (Appendix H). We retain enrichment because production metadata includes the sparse conditions under which the ablation in §4.3 shows a retrieval gain.

## 6 Conclusion

We recast capability discovery as information retrieval. An offline enrichment step converts sparse registry metadata into searchable profiles, and an online retrieve-then-rank pipeline returns a ranked shortlist without invoking any candidate. In-context routing’s Match@1 collapses (0.85 → 0.12), while retrieve-then-rank degrades gently (0.81 → 0.39), with the crossover around 500 entries in the Nova Micro sweep. At full scale the pipeline leads Search&Pick by 6.5 pp at about half the cost. It reduces cost 70× versus Full-Ctx. The pipeline is in production.

## Limitations

Our strongest conclusions come from ToolRet (n = 7,278). The agent and skill benchmarks have construction limits (§4.2, §4.3) that prevent independent cross-type claims. Enrichment is neutralto-negative on well-documented registries. Sparse metadata does not occur naturally in the evaluated benchmarks, so we reproduce it through controlled degradation. Quality and Intent require trust and type metadata absent from public registries; public numbers reflect the two-signal pipeline only. The production registry indexes models alongside the other types under the same schema, but we report no model-retrieval numbers. Existing modelrouting literature addresses that case, and no public benchmark isolates model-as-capability retrieval.

## Ethical considerations

The discovery stage ranks candidates without invoking them, so it does not run candidate code. Provider-controlled metadata can bias retrieval, so operators should version profiles and monitor exposure. The public benchmarks involve no human subjects or private data.

## References

Amazon Web Services. 2024. Amazon bedrock agents. https://aws.amazon.com/bedrock/agents/.

Anthropic. 2024. Model context protocol. https://mo delcontextprotocol.io.

Anthropic. 2025. Claude code. https://www.anthro pic.com/claude-code.

Cursor. 2024. Cursor: The AI code editor. https: //cursor.com.

Yu Du, Fangyun Wei, and Hongyang Zhang. 2024. Any-Tool: Self-reflective, hierarchical agents for largescale API calls. ArXiv:2402.04253.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Precise zero-shot dense retrieval without relevance labels. ArXiv:2212.10496.

Google. 2025. Announcing the Agent2Agent protocol (A2A). Google Developers Blog, 2025-04-09. http s://github.com/google/A2A.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. ArXiv:2402.01680.

Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. 2023. ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2305.11554.

Tenghao Huang, Dongwon Jung, and Muhao Chen. 2024a. Planning and editing what you retrieve for enhanced tool learning. In Findings of the Association for Computational Linguistics: NAACL. ArXiv:2404.00450.

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and Lichao Sun. 2024b. MetaTool benchmark for large language models: Deciding whether to use tools and which to use. ArXiv:2310.03128.

LangChain. 2024. LangGraph: Building stateful, multiactor applications with LLMs. https://github.c om/langchain-ai/langgraph.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. ArXiv:2308.03281.

Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, and Xiaoyu Shen. 2025. Tools are underdocumented: Simple document expansion boosts tool retrieval. ArXiv:2510.22670.

Isaiah Onando Mulang’, Johannes Thaller, Tushar Trivedi, Lars Heling, and Felix Sasaki. 2026. Representing agentic tools in knowledge graphs for structure-aware tool discovery under tool overload. In GENAIK-NORA 2026: Joint Workshop on Generative AI and Knowledge Graphs and Knowledge Graphs & Agentic Systems Interplay, IJCAI-ECAI 2026 Workshops, Bremen, Germany. OpenReview: https://openreview.net/forum?id=7MVoH9Y3 mi.

Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. ArXiv:1904.08375.

OpenAI. 2024. Introducing the GPT store. https://op enai.com/index/introducing-the-gpt-store.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2024. Gorilla: Large language model connected with massive APIs. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2305.15334.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations (ICLR). ArXiv:2307.16789.

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024. Tool learning with large language models: A survey. ArXiv:2405.17935.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2302.04761.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugging-GPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2303.17580.

Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, and Zhaochun Ren. 2025. Retrieval models aren’t tool-savvy: Benchmarking tool retrieval for large language models. In Findings of the Association for Computational Linguistics (ACL), pages 24497–24524. ArXiv:2503.01763.

Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at search? investigating large language models as re-ranking agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). ArXiv:2304.09542.

Yichen Tang, Weihang Su, Yiqun Liu, and Qingyao Ai. 2026. Multi-field tool retrieval. ArXiv:2602.05366.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). ArXiv:2407.18901.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving text embeddings with large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). ArXiv:2401.00368.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023. Visual ChatGPT: Talking, drawing and editing with visual foundation models. ArXiv:2303.04671.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). ArXiv:2309.07597; home of the BGE / FlagEmbedding family.

Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R. Fung, Hao Peng, and Heng Ji. 2024a. CRAFT: Customizing LLMs by creating and retrieving from specialized toolsets. In International Conference on Learning Representations (ICLR). ArXiv:2309.17428.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. 2024b. EASYTOOL: Enhancing LLM-based agents with concise tool instruction. ArXiv:2401.06201.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. 2024. AFlow: Automating agentic workflow generation. ArXiv:2410.10762.

## A Detailed results

The cost breakdown (Table 3), four-reranker robustness sweep (Table 4), and raw-versus-enriched ablation (Table 6) support the claims in §4.8, §4.7, and §4.3, respectively.

## B Extended ranking metrics

The single-pick baselines emit one capability per query, so M@3, M@5, MRR, and R@15 are undefined. We omit them from both tables. Regex emits a deterministic ranked list (regex hits first, then alphabetic). Its lists are long enough for M@3, M@5, and MRR (Table 7) but not for R@15, so it is absent from that table (Table 8).

## C Retrieval vs. reranking

1. Two stages prevent the single-stage collapse. Single-stage LLM picking (Full-Ctx) collapses at scale because the LLM has to consider thousands of items at once. Our pipeline’s reranker always sees exactly k = 15. Full-Ctx goes 0.85 → 0.12 across the scan; our pipeline’s reranker stays 0.85 → 0.70.

2. The reranker is not the constraint. ∼0.87 conditional accuracy at N = 100 with full retrieval recall is the highest value we measure.

3. The remaining end-to-end gains must come from retrieval. On the scan slice at N = 7,278, Match@1 is ∼0.39. A GT capability reaches the top 15 for ∼56% of queries, and the reranker ranks it first with ∼0.70 accuracy $( 0 . 3 9 \approx 0 . 5 6 \times 0 . 7 0 )$ . This factorization shows that retrieval limits further gains.

4. The pipeline’s advantage is largest at mid-N (50–1,000). $N \leq 2 5$ retrieval is identity and Ours ≈ Full-Ctx; N ≥ 5,000 Ours is bottlenecked by BM25 recall, not its own design; in between, Ours leads by 7–17 pp.

5. Why a ≥ 5 pp BGE-over-BM25 Match@1 gain does not materialize. At the registry sizes we test, the reranker’s conditional accuracy is flat across stages, so a +6.9 pp Recall@15 advantage for BGE does not propagate to endto-end Match@1. Recall@15 and end-to-end Match@3/Match@5 separate the retrievers.

Comparisons are consistent with the sweep, but only Figure 2 estimates the crossover at N = 500.

## D Failure modes

Category 1 – Retrieval miss. Tools-ToolRet. A query asks to “develop an emotion analysis system for customer satisfaction over the phone for a Russian telecom company.” The ground-truth tool is a Russian speech-emotion model; the system predicts a generic customer-satisfaction scorer. The query shares no keywords with the ground-truth tool name, and BM25 misses it. This failure mode motivates the neural retriever in §4.4.

<table><tr><td>Row</td><td>Method</td><td>M@1</td><td>Tok-in/q</td><td>Tok-out/q</td><td>$/1k q</td><td>p50 (ms)</td><td>Source</td></tr><tr><td rowspan="5">Tools — ToolRet</td><td>Full-Ctx</td><td>0.166</td><td>128,067</td><td>8</td><td>4.4835</td><td>5,841</td><td>real</td></tr><tr><td>Search&amp;Pick</td><td>0.332</td><td>3,141</td><td>48</td><td>0.1166</td><td>1,556</td><td>real</td></tr><tr><td>Trial&amp;Err</td><td>0.054</td><td>1,875</td><td>29</td><td>0.0697</td><td>1,532</td><td>real</td></tr><tr><td>Ours+BM25</td><td>0.388</td><td>1,597</td><td>75</td><td>0.0664</td><td>1,290</td><td>est.</td></tr><tr><td>Ours+BGE-5f</td><td>0.389</td><td>1,586</td><td>75</td><td>0.0660</td><td>1,116</td><td>est.</td></tr><tr><td rowspan="5">Tools — AppWorld</td><td>Full-Ctx</td><td>0.762</td><td>6,997</td><td>10</td><td>0.2463</td><td>447</td><td>real</td></tr><tr><td>Search&amp;Pick</td><td>0.476</td><td>2,505</td><td>42</td><td>0.0936</td><td>2,085</td><td>real</td></tr><tr><td>Trial&amp;Err</td><td>0.061</td><td>2,457</td><td>29</td><td>0.0900</td><td>1,978</td><td>real</td></tr><tr><td>Ours+BM25</td><td>0.741</td><td>1,867</td><td>75</td><td>0.0758</td><td>565</td><td>est.</td></tr><tr><td>Ours+BGE-5f</td><td>0.735</td><td>1,842</td><td>75</td><td>0.0750</td><td>972</td><td>est.</td></tr><tr><td rowspan="5">Agents — ToolRet</td><td>Full-Ctx</td><td>0.148</td><td>122,672</td><td>7</td><td>4.2946</td><td>5,637</td><td>real</td></tr><tr><td>Search&amp;Pick</td><td>0.341</td><td>3,172</td><td>47</td><td>0.1176</td><td>1,621</td><td>real</td></tr><tr><td>Trial&amp;Err</td><td>0.045</td><td>1,747</td><td>26</td><td>0.0648</td><td>1,663</td><td>real</td></tr><tr><td>Ours+BM25</td><td>0.418</td><td>2,061</td><td>75</td><td>0.0826</td><td>1,153</td><td>est.</td></tr><tr><td>Ours+BGE-5f</td><td>0.412</td><td>2,232</td><td>75</td><td>0.0886</td><td>1,068</td><td>est.</td></tr><tr><td rowspan="5">Agents — AppWorld</td><td>Full-Ctx</td><td>0.986</td><td>553</td><td>5</td><td>0.0201</td><td>360</td><td>real</td></tr><tr><td>Search&amp;Pick</td><td>0.973</td><td>5,487</td><td>41</td><td>0.1978</td><td>1,946</td><td>real</td></tr><tr><td>Trial&amp;Err</td><td>0.014</td><td>1,441</td><td>23</td><td>0.0536</td><td>1,792</td><td>real</td></tr><tr><td>Ours+BM25</td><td>1.000</td><td>26,906</td><td>60</td><td>0.9501</td><td>1,543</td><td>est.</td></tr><tr><td>Ours+BGE-5f</td><td>1.000</td><td>26,906</td><td>60</td><td>0.9501</td><td>1,628</td><td>est.</td></tr><tr><td rowspan="5">Skills — MCP (leaky)</td><td>Full-Ctx</td><td>0.968</td><td>29,110</td><td>6</td><td>1.0197</td><td>1,195</td><td>real</td></tr><tr><td>Search&amp;Pick</td><td>0.829</td><td>2,472</td><td>35</td><td>0.0915</td><td>1,492</td><td>real</td></tr><tr><td>Trial&amp;Err</td><td>0.624</td><td>1,125</td><td>23</td><td>0.0426</td><td>1,212</td><td>real</td></tr><tr><td>Ours+BM25</td><td>0.807</td><td>2,290</td><td>75</td><td>0.0906</td><td>141</td><td>est.</td></tr><tr><td>Ours+BGE-5f</td><td>0.806</td><td>2,290</td><td>75</td><td>0.0906</td><td>176</td><td>est.</td></tr></table>

Table 3: Retrieve-then-rank costs ∼70× less than the full-context picker at higher Match@1. Columns report Match@1, tokens per query, dollars per 1,000 queries, and p50 latency. Trial-and-error rows report real measured token counts; Ours rows are post-hoc prompt-shape estimates (±5–10% of real).
<table><tr><td>Row</td><td>Reranker</td><td>M@1</td><td>p50 (ms)</td></tr><tr><td>Tools-ToolRet Nova Lite (n=500)</td><td>Nova Micro Cl. 3.5 Haiku Cl. Sonnet 4</td><td>0.682 0.678 0.496 0.722</td><td>750 1,316 11,196 3,775</td></tr><tr><td>Tools-ToolRet Nova Lite (full)</td><td>Nova Micro Cl. 3.5 Haiku Cl. Sonnet 4</td><td>0.389 0.395 0.280 0.419</td><td>807 1,453 11,605 3,909</td></tr><tr><td>Tools- AppWorld</td><td>Nova Micro Nova Lite Cl. 3.5 Haiku Cl. Sonnet 4</td><td>0.782 0.864 0.503 0.946</td><td>553 965 11,628 8,833</td></tr></table>

Table 4: Cross-model robustness of the reranker. The large-registry gain from a stronger reranker is small, consistent with accuracy inside the 0.70–0.87 band.

Category 2 – Picked-more-generic. Skills-MCP. A query asks to “list running code-sandbox-mcp containers.” The ground-truth tool is the named sandbox server, retrieved at rank 3; the system instead promotes a generic Docker tool. It overweights the surrounding intent (“running containers”) despite the query naming the correct tool.

Category 3 – Picked outside ground-truth set. Tools-AppWorld. A query asks to “add a comment to all Venmo payments I received from coworkers and like those payments.” The ground truth is a 9-tool trajectory; the system predicts a single tool that treats the comment and the like as one action. Match@1 penalizes this; Recall@15 does not.

<table><tr><td>Row</td><td>Ranker LLM</td><td>Search &amp;Pick</td><td>Ours +BM25</td><td>Ours +BGE</td></tr><tr><td>Tools- ToolRet</td><td>Nova Micro Qwen3 Next Mistral L3</td><td>0.332 0.368 0.365</td><td>0.388 0.416 0.408</td><td>0.389 0.419 0.415</td></tr><tr><td>Agents- ToolRet</td><td>Nova Micro Qwen3 Next Mistral L3</td><td>0.341 0.385 0.388</td><td>0.418 0.451 0.451</td><td>0.412 0.452 0.458</td></tr></table>

Table 5: Cross-model generalization of the pipeline (not just the reranker). Match@1 at full n. Ours beats the strongest in-context baseline (Search&Pick) under all LLMs: $+ 5 . 6 / + 5 . 1 / + 5 . 0$ pp on Tools-ToolRet and $+ 7 . 7 / + 6 . 7 / + 7 . 0$ pp on Agents-ToolRet. The Qwen3 Next ranker is ∼ 3 pp of Nova Micro at full n, so the prompt is not tuned. Bold marks the best result.

Category 4 – Full-Ctx miss. Tools-ToolRet, n=500. A query asks “show me my history for today.” The ground-truth tool exists in the registry but Full-Ctx returns null. At full scale, prompt truncation drops 55% of the registry (4,023 of 7,278 tools) and creates additional misses. This truncation causes the collapse in §4.5.

<table><tr><td>Benchmark</td><td>Cond.</td><td>M@1</td><td>M@3</td><td>R@k</td></tr><tr><td rowspan="2">AppWorld (332)</td><td>Raw</td><td>0.782</td><td>0.884</td><td>0.420</td></tr><tr><td>Enriched</td><td>0.782</td><td>0.918</td><td>0.400</td></tr><tr><td rowspan="2">ToolRet (1,000)</td><td>Raw</td><td>0.124</td><td>0.137</td><td>0.135</td></tr><tr><td>Enriched</td><td>0.125</td><td>0.139</td><td>0.138</td></tr><tr><td rowspan="2">MCP Skills (859)</td><td>Raw</td><td>0.929</td><td>0.977</td><td>0.988</td></tr><tr><td>Enriched</td><td>0.885</td><td>0.939</td><td>0.951</td></tr></table>

Table 6: Enrichment ablation behind §4.3. LLM enrichment adds no measurable gain on well-documented public registries. Raw vs. LLM-enriched registry, all else held fixed. The ToolRet rows use a 1,000-tool subset, so absolute numbers are low but the raw-vs-enriched delta is the quantity of interest.

![](images/48de5472abece272594537fde40dc063c46d9d8d6121342fd2e59e82809d4403.jpg)  
Figure 3: Cross-sectional benchmark comparison. Full-Ctx wins or ties on the small rows, while Ours+BM25 leads on the large ToolRet rows. Full-Ctx uses 500- candidate subsamples on ToolRet and Skills-MCP (Table 2), so these points do not estimate a shared threshold.

## E Enrichment under sparse metadata

We reuse the 1,000-tool ToolRet pool from Table 6 and vary only its input metadata. Full retains the upstream name and description, hint retains the name and first sentence (at most 150 characters), and name-only removes the description. For each level, the raw arm indexes that input directly and the enriched arm first applies the same enrichment procedure. Both use BM25 at k = 15, the Nova Micro reranker, and the weights in §3.3.

We evaluate the 1,252 queries whose groundtruth capability occurs in the pool. This differs from the original ablation in Table 6, which uses the full query set and a raw registry that already contains keywords added during preprocessing. The present comparison starts from the upstream fields and isolates how much enrichment recovers as those fields are removed.

At name-only, enrichment raises Recall@15 from 0.134 to 0.467, while rerank conditional accuracy changes from 0.63 to 0.73. Enriched inputs perform similarly (Match@1 0.776 and 0.778), so one descriptive sentence supplies most of what this enricher uses. For 867 of 1,000 name-only entries, it leaves the capability summary empty rather than inferring one from an opaque name.

![](images/2e27f3809a3c4152f0774b957b227de3273a629b8405271ec8c5fb259aed7c1c.jpg)  
Figure 4: Enrichment acts primarily through retrieval. As input metadata is removed, the raw-versus-enriched gap widens far more in Recall@15 (right) than in Match@1 (left).

## F Results by ToolRet source

ToolRet combines queries from 35 upstream sources. We group the existing full-n results by metadata.subset to test whether a single source accounts for the aggregate gain. Each group remains part of the same benchmark and is evaluated against the same registry.

A broken source. The ultratool source (500 queries; 6.3% of ToolRet) is malformed. Every label embeds the same tool document, so 287 referenced ground-truth tools are absent from the registry, and all methods score near zero on it. Excluding it raises every method’s Tools Match@1 by ∼2 pp and leaves the ordering unchanged.

## G Tool-tuned retriever

We test ToolBench-IR, the BERT-base retrieval model released with ToolBench and fine-tuned on tool-API relevance (Qin et al., 2024), as the first stage on both large rows. All retrievers use the same five-field documents and k = 15.

## H Contribution of quality and intent

Public benchmarks lack the trust and type metadata that Quality and Intent need, so these two signals drop out (§3.3). We compare the two- and four-signal configurations on an internal singleturn agent benchmark (n = 221) whose registry includes both fields. We hold retrieval fixed and report only paired performance differences.

<table><tr><td>Row</td><td>Metric</td><td>Regex</td><td>BM25</td><td>BGE-5f</td><td>Ours+BM25</td><td>Ours+BGE-5f</td><td>Ours+Titan</td></tr><tr><td rowspan="3">Tools — ToolRet</td><td>M@3</td><td>0.314</td><td>0.423</td><td>0.472</td><td>0.493</td><td>0.502</td><td>0.523</td></tr><tr><td>M@5</td><td>0.354</td><td>0.464</td><td>0.520</td><td>0.523</td><td>0.539</td><td>0.558</td></tr><tr><td>MRR</td><td>0.285</td><td>0.378</td><td>0.424</td><td>0.446</td><td>0.454</td><td>0.467</td></tr><tr><td rowspan="3">Tools — AppWorld</td><td>M@3</td><td>0.687</td><td>0.728</td><td>0.721</td><td>0.884</td><td>0.898</td><td>0.918</td></tr><tr><td>M@5</td><td>0.755</td><td>0.837</td><td>0.823</td><td>0.898</td><td>0.932</td><td>0.946</td></tr><tr><td>MRR</td><td>0.576</td><td>0.631</td><td>0.594</td><td>0.820</td><td>0.819</td><td>0.832</td></tr><tr><td rowspan="3">Agents — ToolRet</td><td>M@3</td><td>0.324</td><td>0.445</td><td>0.479</td><td>0.523</td><td>0.533</td><td>0.519</td></tr><tr><td>M@5</td><td>0.368</td><td>0.485</td><td>0.528</td><td>0.546</td><td>0.567</td><td>0.550</td></tr><tr><td>MRR</td><td>0.288</td><td>0.402</td><td>0.435</td><td>0.475</td><td>0.479</td><td>0.464</td></tr><tr><td rowspan="3">Agents — AppWorld</td><td>M@3</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>M@5</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>MRR</td><td>0.997</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td rowspan="3">Skills — MCP</td><td>M@3</td><td>0.811</td><td>0.964</td><td>0.958</td><td>0.879</td><td>0.880</td><td>0.981</td></tr><tr><td>M@5</td><td>0.854</td><td>0.980</td><td>0.974</td><td>0.902</td><td>0.911</td><td>0.988</td></tr><tr><td>MRR</td><td>0.736</td><td>0.923</td><td>0.928</td><td>0.852</td><td>0.855</td><td>0.962</td></tr></table>

Table 7: Ours+Titan leads on Tools-ToolRet, AppWorld, and Skills-MCP; on Agents-ToolRet, Ours+BGE-5f leads. The saturated Agents-AppWorld row ties. The single-pick baselines (Full-Ctx, Search&Pick, and Trial&Err) emit one capability per query, so these metrics are undefined for them and their columns are omitted.
<table><tr><td>Row</td><td>BM25 M@15</td><td>BGE-5f M@15</td><td>Titan M@15</td><td>Ours+BM25 R@15</td><td>Ours+BGE-5f R@15</td></tr><tr><td>Tools — ToolRet</td><td>0.549</td><td>0.618</td><td>0.625</td><td>0.485</td><td>0.517</td></tr><tr><td>Tools — AppWorld</td><td>0.952</td><td>0.966</td><td>0.993</td><td>0.262</td><td>0.261</td></tr><tr><td>Agents — ToolRet</td><td>0.569</td><td>0.626</td><td>0.616</td><td>0.508</td><td>0.538</td></tr><tr><td>Agents — AppWorld</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>Skills — MCP</td><td>0.991</td><td>0.992</td><td>0.994</td><td>0.951</td><td>0.992</td></tr></table>

Table 8: Retrieval at $k = 1 5 ,$ . Standalone columns report Match@15, the query-level any-hit rate. Ours columns report Recall@15, the fraction of all ground-truth capabilities retrieved. Single-pick baselines are omitted.
<table><tr><td>Input level</td><td>Raw</td><td>Enriched</td><td>R@15 raw → enr.</td></tr><tr><td>Full description</td><td>0.720</td><td>0.778</td><td> $0 . 9 0 5  0 . 9 3 3$ </td></tr><tr><td>First-sentence hint</td><td>0.685</td><td>0.776</td><td> $0 . 8 8 8  0 . 9 2 9$ </td></tr><tr><td>Name-only</td><td>0.085</td><td>0.341</td><td> $0 . 1 3 4  0 . 4 6 7$ </td></tr></table>

Table 9: Metadata-sparsity ablation $( n = 1 , 2 5 2$ per cell). Match@1 gains are +5.8, +9.1, and +25.6 pp from full to name-only input.

<table><tr><td>Source</td><td>n Ours+Titan</td><td>S&amp;Pick</td><td>∆</td></tr><tr><td>Tools — ToolRet toolbench 1,100 apigen 1,000 toolace 1,000</td><td>0.518 0.702 0.694</td><td>0.490 0.611 0.620</td><td>+0.028 +0.091 +0.074</td></tr><tr><td>toolink apibank gorilla-huggingface long tail (19 src)</td><td>497 101 500</td><td>0.579 0.614 0.148</td><td>0.348 +0.231 0.426 +0.188 0.180 -0.032</td></tr><tr><td></td><td>798 0.325</td><td>0.246</td><td>+0.079</td></tr><tr><td>Agents — ToolRet</td><td></td><td></td><td></td></tr><tr><td>toolace</td><td>1,000 0.715</td><td>0.637</td><td>+0.078</td></tr><tr><td>apigen</td><td>1,000 0.663</td><td>0.610</td><td>+0.053</td></tr><tr><td>toolbench craft-math-algebra</td><td>1,100 0.475 280 0.482</td><td>0.524 0.257</td><td>-0.048 +0.225</td></tr></table>

Table 10: Match@1 for representative ToolRet sources. Across all sources with $n \geq 1 0 0 .$ , Ours+Titan leads Search&Pick on 13/16 Tools sources and 12/16 Agents sources. “Long tail” pools 19 smaller Tools sources.

<table><tr><td>Row</td><td>Retriever</td><td>Standalone M@1</td><td>Standalone R@15</td><td>E2E M@1</td></tr><tr><td rowspan="3">Tools</td><td>BM25</td><td>0.311</td><td>0.549</td><td>0.388</td></tr><tr><td>ToolBench-IR</td><td>0.354</td><td>0.608</td><td>0.378</td></tr><tr><td>Titan V2</td><td>0.358</td><td>0.625</td><td>0.397</td></tr><tr><td rowspan="3">Agents</td><td>BM25</td><td>0.336</td><td>0.569</td><td>0.418</td></tr><tr><td>ToolBench-IR</td><td>0.362</td><td>0.596</td><td>0.381</td></tr><tr><td>BGE-large 5f</td><td>0.360</td><td>0.626</td><td>0.412</td></tr></table>

Table 11: Tool-tuned and general retrievers on the two full ToolRet query sets $( n = 7 { , } 9 6 1 )$ . ToolBench-IR improves standalone Match@1 over BM25 but does not exceed the best general encoder on Recall@15 or endto-end (E2E) Match@1.

<table><tr><td>Metric</td><td>Four-signal minus two-signal</td></tr><tr><td>Match@1</td><td>+4.5 pp</td></tr><tr><td>Match@3</td><td>+5.0 pp</td></tr><tr><td>Recall@15</td><td>0.0 pp</td></tr></table>

Table 12: Paired effect of adding Quality and Intent on the internal benchmark. Fourteen queries change from miss to hit at rank 1 and four change from hit to miss (exact McNemar $p = 0 . 0 3 1 )$ . Recall is unchanged because retrieval is shared.