# MemUse: Moving Memory Evaluation from Direct QA to Natural Integration in Long-Term Human-AI Conversation

Ryuichi Sumida Koji Inoue Tatsuya Kawahara

Graduate School of Informatics, Kyoto University

sumida.ryuichi.65m@st.kyoto-u.ac.jp

inoue@sap.ist.i.kyoto-u.ac.jp, kawahara@i.kyoto-u.ac.jp

## Abstract

Memory systems for conversational LLMs are conventionally evaluated by direct, factseeking questions about prior dialogue (Direct QA): can the model recall fact X from a prior conversation? We tested whether higher Direct QA accuracy correlates with higher user satisfaction in a 4-month deployment (40 users, 1,872 sessions, 7 memory conditions). Existingbenchmark Direct QA varies from 19.7% to 70.1% across the 7 conditions, but satisfaction does not change. We hypothesize that existing benchmarks and user satisfaction are tracking different capabilities: benchmarks measure elicited retrieval (recall when asked), while conversation requires natural integration (detecting relevance and naturally weaving prior context into a response). To examine this, we introduce MEMUSE, a set of real user-cued memory moments drawn from the deployment, scored by an integration-aware judgment of the natural conversational response. Holding the model and context fixed, the same system that scores 78.8% on Direct QA references only 7.9% of those facts in conversation—a 71-point gap. Within these moments, Natural Integration is associated with satisfaction, whereas Direct QA is not. We release the deployment corpus and MEMUSE together with all judgments and scoring prompts at https: //github.com/ryuichi-sumida/memuse.

## 1 Introduction

As LLMs are deployed in long-term conversational settings—personal assistants, therapy bots, diary companions—their memory of past interactions is assumed critical for user satisfaction (Skjuve et al., 2021; Brandtzæg et al., 2022). Recent benchmarks like LoCoMo (Maharana et al., 2024), LUFY (Sumida et al., 2025), and RealTalk (Lee et al., 2025) evaluate memory through direct, fact-seeking questions about prior dialogue (we refer to this evaluation format as Direct QA throughout): can the model recall fact X from a prior conversation? Memory systems have accordingly focused on expanding capacity through longer contexts (Chen et al., 2023) and retrieval augmentation (Lewis et al., 2020). All share an untested structural assumption: that recall-when-asked (elicited retrieval) reflects whether a model will use prior knowledge in conversation (natural integration).

But does benchmark performance on Direct QA actually translate to user satisfaction in real-world settings? We conducted a 4-month longitudinal study where 40 users interacted daily with an AI diary companion across 7 memory conditions, ranging from summary-only to full long-context and RAG systems. At the population level, higher Direct QA accuracy does not raise satisfaction (§4): condition-level mean differences against Summaryonly are all below 0.06 within-user SD.

The null is the puzzle, not the conclusion. We hypothesize that existing benchmarks and userexperience measures are simply tracking different capabilities: benchmarks measure elicited retrieval, while conversation requires natural integration. To examine this, we introduce MEMUSE (§5), a benchmark of 72 real moments where users explicitly referenced or tested the system’s memory, scored by a single Natural Integration judgment of whether the system’s natural response demonstrates memory of the referenced topic.

Contributions. We make three contributions:

• A randomized 4-month deployment (40 users, 1,872 sessions, 7 memory conditions) in which Direct QA accuracy varies from 19.7% to 70.1% across conditions, yet user satisfaction does not change (§4).

• MEMUSE, a benchmark of 72 real usercued memory moments, scored by whether the model’s conversational response actually draws on the referenced memory—a measure that, unlike Direct QA, is associated with satisfaction (§5, §6.1).

<table><tr><td>Benchmark</td><td>Authoring</td><td>Users</td><td>Sess./u.</td><td>Turns/u.</td><td>Date span</td><td>Mem. %†</td><td>Nat. cued</td><td>Sat.</td></tr><tr><td colspan="9">Synthetic / LLM-simulated dialogues</td></tr><tr><td>MemoryBank (Zhong et al., 2024)</td><td>authors</td><td>15</td><td>10</td><td>37.7</td><td>10 days</td><td>17.7%</td><td>X</td><td>X</td></tr><tr><td>LoCoMo (Maharana et al., 2024)</td><td>authors</td><td>20</td><td>27.2</td><td>294.1</td><td> ${ \sim } 2 \mathrm { m o }$ </td><td>24.0%</td><td>X</td><td>X</td></tr><tr><td>LongMemEval (Wu et al., 2025)</td><td>experts</td><td>500</td><td>47.7</td><td>244.8</td><td></td><td>0.4%</td><td>X</td><td>X</td></tr><tr><td colspan="9">Real human conversation</td></tr><tr><td>LUFY (Sumida et al., 2025)</td><td>users</td><td>17</td><td>4</td><td>123.1</td><td>~9 days</td><td>23.8%</td><td>X</td><td>××</td></tr><tr><td>RealTalk (Lee et al., 2025)</td><td>crowd</td><td>20</td><td>21.9</td><td>193.7</td><td>~3 weeks</td><td>14.6%</td><td>X</td><td></td></tr><tr><td>MEMUSE (Ours)</td><td>detected</td><td>40</td><td>46.8</td><td>269.9</td><td>~4 months</td><td>1.4%</td><td>√</td><td>V</td></tr></table>

Table 1: Long-term memory benchmarks. MEMUSE is the only one combining multi-month real human-AI deployment, evaluation items detected from the conversation rather than authored as QA, and per-session satisfaction ratings. Column abbreviations: Sess./u. = sessions per user; Turns/u. = user turns per user; Mem. $\% ^ { \dag } = \%$ of user turns paired with an evaluation item; Nat. cued = evaluation items naturally cued by users (vs. externally authored); Sat. = per-session user satisfaction rating available. LoCoMo and RealTalk pair the same script across two speakers, so their 20 users represent 10 distinct dialogues.

• Evidence that retrieval and integration are dissociable: the same system answers 78.8% of MEMUSE items correctly under Direct QA, but references those facts in conversation only 7.9% of the time—a 71-point gap (§6.2).

## 2 Related Work

Memory Benchmarks. Existing memory benchmarks—LoCoMo (Maharana et al., 2024), RealTalk (Lee et al., 2025), LUFY (Sumida et al., 2025), and LongMemEval (Wu et al., 2025)—rely on synthetic or externally authored questions in a Direct QA format and none validate whether their scores predict user experience (Table 1).

Memory-Augmented Systems. Personalized dialogue systems combine summarization (Lu et al., 2023), retrieval (Lewis et al., 2020), and longcontext prompting (Chen et al., 2023) with explicit memory management (Packer et al., 2023; Chhikara et al., 2025; Ong et al., 2025); selective forgetting can preserve performance at lower cost (Sumida et al., 2025), which we use to instantiate our LC/RAG capacity range. Long-context prompting carries integration risk—models often ignore or degrade on relevant evidence (Liu et al., 2024)—a phenomenon our retrieval–integration gap (§6.2) extends to natural conversation. Evaluation across these systems still centers on recall accuracy; we instead test these strategies against longitudinal user satisfaction.

Dialogue Evaluation Beyond Accuracy. Disconnects between automatic metrics and human judgment are well documented in dialogue (Deriu et al., 2021), with prior work on satisfaction prediction (Bodigutla et al., 2020; Santana et al., 2025) and long-term human-chatbot relationships (Skjuve et al., 2021) highlighting user experience beyond benchmark accuracy.

## 3 Experimental Setup

Forty proficient English speakers from diverse geographic backgrounds (approximately half based in East Asia, remainder from North America, Europe, the Middle East, and other regions; 37F, 3M; ages 20s–60s) interacted daily with “Luke,” an AI diary companion built on GPT-4.1-mini, from November 2025 to February 2026. Users accessed the system through a web interface (compatible with both mobile and desktop) where they could review summaries of prior conversations before starting each session (Appendix A.2). Participants were required to interact for at least 1 hour and write around 2,000 English words per month; content was unconstrained—users wrote about daily life, personal reflections, and any topic of their choosing. The study yielded 1,872 sessions (21,575 turns); users rated each session on a 1–7 satisfaction scale.

Each session was randomly assigned one of 7 memory conditions: SUMMARY, LC-10/50/100%, or RAG-10/50/100%. Summaryonly supplies only a summary of prior conversations; all other conditions add capacity on top of this same summary. LC-k% prepends the top k% most important prior turns to the conversation history. RAG-k% retrieves the top-10 turns from a vector index over that same top-k% pool. Importance is scored by a RoBERTa model finetuned on human annotations from the LUFY dataset (Sumida et al., 2025). On a held-out evaluation (trained on two annotator groups, tested on a third), the model exceeds the human annotator average on both important-utterance identification (56.3% vs. 41.4%) and not-important identification (93.1% vs. 91.3%), ranking 7th among 18 raters (17 humans + the model). Conditions were randomly rotated to ensure balanced exposure (Appendix A.7); engagement remained stable across the 4-month study (Appendix A.6). All seven conditions share the same persona and per-session framing; verbatim generation prompts are in Appendix A.3.

For satisfaction analysis, we applied a predefined filtering pipeline (Appendix $\mathbf { A . 7 } ) \colon$ from 1,872 sessions, we excluded 11 users—9 with near-constant ratings $( \mathrm { S D } \leq 0 . 5$ on the 1–7 scale) plus 2 flagged for content quality (template/AI-paste content)— yielding 1,270 sessions from 29 users. We analyze within-user z-scored satisfaction.

## 4 Capacity Improves Direct QA but Not Satisfaction

Our research question is simple: does higher Direct QA performance—obtained by raising memory capacity beyond a summary baseline—translate into higher user satisfaction? Under the prevailing assumption that benchmark scores reflect user value, LC-100% should rank first and Summaryonly last; we test that prediction directly.

Existing-benchmark accuracy rises sharply with capacity. Across the 7 conditions, average existing-benchmark QA accuracy climbs from 19.7% (Summary) to 70.1% (LC-100%) (full percondition numbers in Appendix Table 13, perbenchmark breakdown in Appendix C.2).

User satisfaction does not follow. User ratings do not change with capacity. Every condition’s mean satisfaction differs from Summary-only by less than 0.06 within-user SD (Appendix Table 13). Numerically, LC-100% is not better than Summary; the directional ranking has Summary slightly ahead $( + 0 . 0 2 \mathrm { v s . } - 0 . 0 4 \mathrm { S D } )$ ). Because every condition includes the same summary, this is a null about the marginal value of LC/RAG capacity on top of a summary, not about memory in general. This flat curve is not just rating noise. In a session-level linear mixed-effects model, we entered three engagement features—response length, response specificity (the proper-noun rate in the system’s reply, as a proxy for whether responses name concrete user details), and cross-session continuity (how much the user’s current turn references prior-session topics)—alongside memory condition. Each engagement feature independently predicts higher satisfaction, while memory condition does not (full models in Appendix $\mathrm { F } ;$ robustness in Appendix C). The null is also not latency-mediated: on the same 29-user subset, neither mean, max, nor min total response latency—nor their TTFT counterparts—correlates with within-user z-scored rating $( | \rho | < 0 . 0 6$ , all $p \geq . 0 7$ ; per-condition latencies in Table 2, full breakdown in Appendix G).

<table><tr><td>Cond.</td><td>Tok. Gr. vs Sum.</td><td></td><td>Mean Lat.</td><td>Mean TTFT</td></tr><tr><td>Summary</td><td>1.0×</td><td>1.0×</td><td>1.93</td><td>1.04</td></tr><tr><td>LC-10</td><td>2.0×</td><td>2.9×</td><td>2.00</td><td>1.12</td></tr><tr><td>LC-50</td><td>3.5×</td><td>11.9×</td><td>2.52</td><td>1.41</td></tr><tr><td>LC-100</td><td>5.8×</td><td>22.0×</td><td>3.27</td><td>1.79</td></tr><tr><td>RAG-10</td><td>1.8×</td><td>2.3×</td><td>2.03</td><td>1.05</td></tr><tr><td>RAG-50</td><td>1.0×</td><td>2.5×</td><td>2.03</td><td>1.09</td></tr><tr><td>RAG-100</td><td>1.4×</td><td>2.1×</td><td>2.13</td><td>1.12</td></tr></table>

Table 2: Per-condition cost and latency on the 29-user / 1,270-session subset. Tok. Gr.: within-condition inputtoken growth from month 1 to month 4. vs Sum.: mean session input tokens relative to the Summary baseline. Mean Lat.: per-session mean total response latency (s). Mean TTFT: per-session mean time-to-first-token (s).

## 5 Detecting and Scoring Memory Moments

The deployment puzzle (§4) motivates a closer look at the rare moments where memory is actually at stake: if memory matters to users, the effect should appear when users explicitly cue or encounter memory use—not averaged across sessions where memory was never invoked. We audit the 1,872-session corpus for these moments and package the usercued subset as the MEMUSE benchmark.

## 5.1 Detecting Memory Moments

Detection. We extract memory-relevant moments using GPT-5.4 (temperature=0) on each of 1,872 session transcripts, detecting four event types through explicit linguistic signals: user memory probes (“Do you remember X?”), user reprovisions (“As I mentioned before, X”), system proactive recalls (“How was your trip to the flower shop?”), and user memory reactions (“You remembered!”). Detected events were independently verified by human annotators, who reviewed each candidate alongside (i) the trigger utterance, (ii) the surrounding conversation context (the current session plus summaries of the three preceding sessions), and (iii) the detector-supplied memory rationale (the prior-session content the detector identified as the referent and its justification for flagging the moment). This yielded 147 true positives— 11 probes, 62 re-provisions, 66 proactive recalls, 8 reactions—at 95.5% precision (full procedure and per-type breakdown in Appendix B.1).

<table><tr><td colspan="2">Source sessions (prior context): In sessions 2–3, the system (“Luke&quot;) recommended Bruno Mars (upbeat) and Ryuichi Sakamoto, Hikaru Utada, Yumi Matsutoya, and Haruka Nakamura (Japanese relaxing music). Trigger (user turn, current session): “I&#x27;m thinking I’ll try to relax by listening to the music Luke recommended the other</td></tr><tr><td colspan="2">day.&quot; (i) Natural Integration (primary metric: does the system&#x27;s natural response demonstrate memory of the referenced topic?)</td></tr><tr><td>LC-100% response</td><td>“That sounds like a perfect way to unwind. I hope the music brings you some calm and comfort while your little one rests. Enjoy this peaceful moment—you deserve it!&quot; X GENERIC RESPONSE</td></tr><tr><td>Ideal response (au- thored)</td><td>“Glad you’re going to try the music I recommended! Ryuichi Sakamoto&#x27;s Merry Christmas Mr. Lawrence is my favorite—the solo piano eases you down.&#x27; √ INTEGRATES CONTEXT</td></tr><tr><td colspan="2">(ii) Direct QA (same model, same reconstructed context, each fact-seeking question asked literally; excerpt of 4)</td></tr><tr><td rowspan="2">Q1 Q2</td><td>Q: “What upbeat artist did the system recommend?&#x27; A: “Bruno Mars.&quot;</td></tr><tr><td>Q: “What Japanese artists did the system recommend for relaxing music?&quot; A: “Ryuichi Sakamoto, Hikaru Utada, Yumi Matsutoya, and Haruka Nakamura.&quot;</td></tr><tr><td>(iii) Reference (do those facts actually appear in the natural response above?)</td><td></td></tr><tr><td>Reference 0 / 4 facts referenced in the LC-100% response.</td><td>X</td></tr></table>

Table 3: One real MEMUSE instance scored under all three metrics (GPT-4.1-mini, LC-100% condition, same reconstructed context throughout). The same model that fails (i) Natural Integration and (iii) Reference answers the underlying questions correctly under (ii) Direct QA—the retrieval–integration gap of §6.2.

Memory moments are rare. Across the 1,872 sessions (10,795 user turns total; mean 5.8 per session), the 147 verified events arise in only ∼3.5% of sessions and 1.4% of user turns—one detected memory moment per ∼73 user turns. This 1.4% rate is the natural frequency at which users invoke memory in deployment; existing benchmarks instead pair externally authored QA with much higher fractions of user turns (15–24% across MemoryBank, LoCoMo, LUFY, and RealTalk; Table 1).

Reactive vs. proactive split. The four event types separate cleanly by who initiates the memory use. Reactive (user-cued) moments are userinitiated invitations to use prior context (analyzed in §6.1). Proactive (system-initiated) recalls are turns where the system refers to prior-session content unprompted (analyzed in §6.3). We package only the reactive cases as a re-runnable benchmark, for two reasons. First, each reactive moment has a clear per-instance target—the user has explicitly elicited a specific piece of prior context, so any candidate memory system can be re-tested on the same trigger and scored against the same target.

Proactive recalls have no such fixed target—there is no canonical “correct” next turn—so they cannot be re-played across alternative memory systems in a comparable way. Second, only reactive integration is related to session-level satisfaction in our deployment; neither proactive recall accuracy nor appropriateness is (§6.1, §6.3). Improvements on this benchmark therefore have a plausible path to user-visible benefit.

## 5.2 The MEMUSE Benchmark

MEMUSE is the reactive benchmark packaged for re-evaluation: 62 re-provisions + 10 probes = 72 instances. For each instance we extracted ground-truth facts from the source sessions and decomposed them into 3–5 fact-seeking questions (316 total, avg. 4.4 per instance), phrased in the same format used by existing QA benchmarks. Given an instance and a candidate memory system, the benchmark scores three metrics on the same reconstructed context (Table 3): (i) Natural Integration (the primary metric)—a per-instance binary judgment of whether the system’s natural conversational response to the trigger demonstrates memory of the referenced topic, produced by GPT-5.4-nano at temperature 0 following Zheng et al. (2023); (ii) Direct QA—the same Direct QA format used by existing benchmarks (§4). Each factseeking question is asked literally with the same reconstructed context, making this the apples-toapples analogue of existing benchmarks on real memory moments; (iii) Reference—a per-question check that the facts elicited by Direct QA are actually referenced in the natural conversational response. The metrics (ii) and (iii) operate on the same fact-seeking questions and together diagnose the retrieval–integration gap (§6.2); Natural Integration is the primary metric because it captures the user-side question—did the response acknowledge what I just referenced?—rather than a fact-checker’s question (verbatim prompts in Appendix B.1).

![](images/a80824bbace50361cd101e82c8448b5925ac347e8bb483c848583b19eb629ce1.jpg)  
Existing QA Acc MemUse Direct QA MemUse Natural Integration MemUse Reference Satisfaction (z)  
Figure 1: Across 7 randomized memory conditions (40 users, 1,872 sessions; LC prepends the top-k% most important prior turns, RAG retrieves from the same top-k% pool; full definitions in §3), the two retrieval metrics (Existing QA Acc; MEMUSE Direct QA) rise sharply with capacity, while the integration metrics (MEMUSE Natural Integration; MEMUSE Reference) stay flat. Satisfaction (lower panel; 1,270 rating-valid sessions, 29 users) is shown for comparison. Numerical values in Table 13 (Appendix C.1).

Judge validation. Because LLM judges can be biased (Panickssery et al., 2024), we validate ours against humans. On a stratified 56-item subset, human–human agreement on Natural Integration was substantial (Cohen’s κ = 0.57, 95% bootstrap CI [0.34, 0.78]), and the LLM judge matched the humans on positive rate (LLM 51.8% vs. humans 51.8% / 55.4%). Per-question judgments showed strong three-rater agreement on the full 316-item set (Fleiss’ κ = 0.65; pairwise human–human Cohen’s κ = 0.70; strictness analyses and per-rater breakdown in Appendix B.4).

## 6 Results

Across the 7 memory conditions, retrieval rises sharply with capacity while integration stays flat and satisfaction tracks neither (Figure 1). The deployment puzzle (§4) and the memory-moment audit (§5) together motivate three questions about this gap. Q1: at the rare moments where memory is at stake, is any measurable system behavior associated with satisfaction? Q2: is that integration ability correlated with what existing benchmarks measure? Q3: does system-initiated memory use deliver the same satisfaction benefit as user-cued integration? We answer them in turn. All inferential models are linear mixed-effects models (LMMs) with user random intercepts; rank correlations are Spearman’s ρ, and effect-size confidence intervals are user-clustered bootstrap. Full specifications in Appendix A.9.

## 6.1 Q1: User-Cued Integration Is Associated With Satisfaction

The puzzle (§4) is that capacity-driven retrieval gains do not change ratings on average. Restricting attention to the rare sessions where users explicitly cued memory tells a different story (real excerpts in Table 4). On the 48 reactive memory-moment sessions (after filtering), we evaluate GPT-4.1-mini on two metrics—MEMUSE Direct QA and Natural Integration—using the same instances and the same context.

Only Natural Integration is associated with satisfaction. Direct QA accuracy is uncorrelated with session-level satisfaction $( \rho = + 0 . 0 3 )$ ; Natural Integration is associated $( \rho = + 0 . 2 9 , p =$ .046), with successful integration carrying +0.56 within-user SD higher satisfaction (user-clustered bootstrap 95% CI $[ + 0 . 1 2 , + 0 . 9 8 ]$ ; robust to aggregation rule and engagement covariates, Appendix C.4, F.3). The two metrics also disagree on condition rankings: LC-100% ranks highest under Direct QA (78.8%) and lowest under Natural Integration (22.2%; visible as the LC-100% bars in Figure 1, per-condition numbers in Appendix Table 13). A within-user regression with user-level clustering, entering both predictors jointly, finds that Direct QA adds no predictive value once Natural Integration is included (full specification in Appendix A.9).

The user-side failure mode: re-provisioning burden. Across all 1,270 rating-valid sessions, user word count interacts negatively with cross-session continuity $( \beta = - 0 . 0 8 6 , p < . 0 0 1$ ; Appendix F.5): in high-continuity sessions (those where users reference topics from prior sessions), longer user utterances predict lower satisfaction. This corroborates the integration result with complementary observational evidence: when memory is integrated successfully, satisfaction is higher (above); when the user is forced to re-supply context the system has lost, satisfaction is lower.

## 6.2 Q2: Why Direct QA Misses It: Retrieval and Integration Are Dissociable

The integration-aware result above raises a sharper question: are Direct QA accuracy and Natural Integration two views of the same underlying ability, or are they dissociable? We test the stronger claim that they are dissociable—the same model, with the same context, retrieves and integrates the same facts with no detectable correlation.

A concrete example. Table 3 shows the gap on a single MEMUSE instance: under LC-100%, GPT-4.1-mini’s natural response is generic and ignores the user’s music-recommendation cue, while the same model with the same context, asked the corresponding fact-seeking question directly, names four of the recommended artists.

Available ̸= conversationally integrated. In the LC-100% condition, by design, every prior turn is included in context, and the same model achieves existing-benchmark QA accuracy of 70.1% on LoCoMo/LUFY/RealTalk with this context, confirming the information is retrievable when explicitly queried. Yet in natural conversation, GPT-4.1- mini demonstrates MEMUSE Natural Integration of only 22.2% of real memory moments (LC-100% column of Figure 1; per-condition numbers in Appendix Table 13). The same model recovers the fact when explicitly asked, yet does not reference it when the user implicitly cues it. Classifying all 72 LC-100% responses by failure mode reveals the dominant patterns: 38.9% are generic (empathetic responses ignoring context), 23.6% hallucinate recall with fabricated details (Huang et al., 2025), 26.4% show partial (gist-only) recall, 2.8% are honest admissions of not remembering, and only 8.3% fully integrate prior details into their response (Appendix D.1).

![](images/8e7fd17d3053b0f428788ef0e2de9bda8650dfb9871e34a66334038bdf2d1c36.jpg)  
Figure 2: The retrieval–integration dissociation replicates across stronger models. Direct QA: per-question accuracy when each MEMUSE question is asked literally with the reconstructed context. Natural Integration: reader-level binary judgment of whether the conversational response demonstrates memory of the topic. Numerical values in Table 18.

A same-model QA ablation. To quantify whether the system references the same facts in natural conversation that it can answer when asked directly, we now compare MEMUSE Direct QA against MEMUSE Reference on the same 316 items: GPT-4.1-mini answers each question in Direct QA format using the identical reconstructed contexts. The gap is stark: MEMUSE Direct QA (78.8%) vs. MEMUSE Reference (7.9%)—a 71-point gap with the same model and context on LC-100% (the gap between the Direct QA and Reference bars at LC-100% in Figure 1; per-condition results in Appendix Table 13). The two metrics also pull condition rankings apart: condition differentiation invisible in natural responses (22–28% Natural Integration) emerges clearly in Direct QA (45–79%). Per-instance, the Spearman correlation between Direct QA correctness and whether the samefact is referenced in the natural response is $\rho = - 0 . 0 0 9$ (48 deployed sessions): retrieval and integration are uncorrelated.

The dissociation is not a property of GPT-4.1-mini. Running MEMUSE Direct QA and

Natural Integration across all 7 memory conditions on two frontier models—GPT-5.5 and Gemini 3.1 Pro, with the judge held fixed at GPT-5.4- nano—replicates the same shape (Figure 2): Direct QA rises by 32–34 points across capacity for every model, while Natural Integration spans at most 8.2 points within any model and never tracks capacity. Stronger models lift the integration baseline (Natural Integration at Summary: 23.6% / 32.9% / 53.4% for GPT-4.1-mini / Gemini 3.1 Pro / GPT-5.5) but capacity-driven gains in retrieval do not translate into capacity-driven gains in integration for any of them. Full numbers in Appendix Table 18.

Modern memory systems raise integration but not fact surfacing. We also evaluate two widely used memory systems on MEMUSE: Mem0 (Chhikara et al., 2025), a retrieval-layer memory, and Letta (MemGPT) (Packer et al., 2023), an end-to-end memory agent (Appendix Table 19). For each instance we ingest the full prior conversation, keep GPT-4.1-mini as the generation model, and score with the same judge; Mem0 retrieves memories that the deployed pipeline then generates from, whereas Letta answers through its own agent loop over archival memory, so its generation is not perfectly matched to ours (Appendix E.2). Both systems lift Natural Integration to 58.3% (Mem0) and 56.9% (Letta), versus 22.2– 27.8% across our seven provisioning conditions—a gain capacity scaling alone did not produce. Yet neither closes the fact-level gap: Direct QA vs. Reference is 41.5% vs. 7.3% (Mem0) and 61.4% vs. 10.8% (Letta), so at least 34 and 51 points respectively are answerable on direct query yet absent from the natural reply. The dissociation thus extends beyond our Summary/LC/RAG strategies; as this is an offline evaluation, we make no claim that the higher integration would raise satisfaction.

The bottleneck is generation, not retrieval. To test whether the integration ceiling is a prompting artifact rather than a deeper limitation, we evaluated four prompt-level interventions on top of the deployed pipeline, each targeting a different hypothesized cause, on GPT-4.1-mini across all 7 conditions:

• CoT prepends a chain-of-thought scaffold instructing the model to check for priorconversation references and weave specific named details into the response.

![](images/60b04ddcab57ff7d0645b3417672fbf2717a41dd3ce583004e94235f6e4976df.jpg)  
Figure 3: Prompt-level interventions raise the level of Natural Integration but do not induce capacitysensitivity (GPT-4.1-mini): every variant stays roughly flat across the 7 memory conditions. GPT-5.5 panel, per-condition numbers, and verbatim prompts in Appendix E.3.

• Cue-aware adds a one-line patch telling the model to reference specific remembered details whenever the user refers to an earlier conversation.

• Two-step splits generation into two calls—a separate extraction step lists up to 5 specific named details (proper nouns, places, dates, prior recommendations) tied to the user’s trigger, which are then handed to the deployed pipeline as “Notes from your prior conversations” (Appendix E.3.5).

• Query-rewrite rewrites the user’s trigger into multi-aspect search queries, runs vector-based retrieval with those queries against the prior-turn pool, and prepends the retrieved passages before generation (Appendix E.3.6).

None induce capacity-sensitivity: every integration variant spans at most 21.9 pts across the 7 conditions versus Direct QA’s ∼33-pt span, and Natural Integration stays roughly flat with capacity for every variant (Figure 3; full results, GPT-5.5 replication, and verbatim prompts in Appendix E.3). The Two-step ablation localizes the failure: even when the extraction step successfully names the ground-truth details, the conversational generation fails to reference them 77% of the time (37/48 cases on LC-100%; Appendix E.3.5)—the bottleneck is downstream of retrieval, in the conversational gen-

<table><tr><td>Mode &amp; outcome</td><td>Excerpt (real deployment)</td></tr><tr><td>Reactive integration √ succeeds</td><td>User: “I inally went to Vienna with my coworker. We followed the plan we made together. ... Just like AI suggested, it wasn&#x27;t crowded because it was a weekday. We were impressed by how accurate the plan was.&quot; Luke:  $^ { 6 6 } \cdot \cdot .$  awesome your plan worked out so well and you could enjoy the views and that tasty</td></tr><tr><td>Reactive integration X fails</td><td>schnitzel without the crowds. Figlmüller is a great pick—juicy veal with that crisp golden breading sounds mouthwatering.&#x27; User: “Do you remember what we were talking about earlier about Brittany?&quot; Luke: “We haven’t talked about Brittany yet in our chats—but it sounds like a fascinating topic!</td></tr><tr><td>Proactive recall √welcomed</td><td>What about Brittany caught your interest?&quot; Luke:  $\because I$  hope your trip to Sapporo, Hakodate, and back to Sendai goes safely. It sounds like a busy day with driving and linguistic tasks—remember to take little breaks to refresh your mind!&quot;</td></tr><tr><td>Proactive recall X walked past</td><td>User:  $^ { i } { \cal O } h ,$  you remember our previous conversation! I am so glad that you remember my previous business  $t r i p . \ '$  Luke: “Lovely to hear from you. How did your little superhero&#x27;s day  $g o \mathrm { ? } \mathrm { " }$  User: “Today I had a serious fight with my wife, and the whole situation has been sitting heavily in</td></tr></table>

Table 4: Real deployment excerpts illustrating the two modes. Reactive integration (top): the user invites memory; the system either weaves prior context into a natural response (✓) or hallucinates absence of the referenced topic (✗). Proactive recall (bottom): the system references a prior topic unprompted; the user either welcomes the callback (✓) or simply continues with their own agenda (✗).

eration itself.

## 6.3 Q3: Proactive Memory: No Symmetric Reward, Clear Timing Risk

A natural follow-up to the reactive results (§6.1) is whether system-initiated proactive memory use— callbacks the system volunteers without a user cue— yields the same satisfaction benefit. Proactive recall is a central design move in long-horizon agent architectures (e.g., the reflection/recall loops of generative agents and self-reflective agents) and proactive conversational systems (Park et al., 2023; Shinn et al., 2023; Deng et al., 2023), but its userside benefit in deployed long-term companions is rarely measured directly. We analyze 70 detectorflagged candidates (66 verified true positives; 64 in rating-valid sessions).

Scoring proactive recalls. Using GPT-5.4 as an LLM judge at temperature 0, we score each of the 70 detector-flagged candidates on two axes. Grounding: given the system’s recall turn and the referenced topic, are the specific details actually present in a prior-session summary? Appropriateness: given the conversation leading up to the recall, does the user’s immediately preceding turn naturally invite this callback? Both use a 0/1/2 scale (0 = not grounded / mistimed; 1 = partial / tangential; 2 = grounded / clearly invited; verbatim prompts in Appendix F.6). The deployed system’s proactive recalls are predominantly grounded (46/70 fully, 21 partial, 3 hallucinated), but timing is more variable (32 invited, 29 tangential, 9 mistimed).

![](images/0bff73b6287e2a3ae03c1da910e43516b8716386f4ce8c190ed677e68aee31ae.jpg)  
Figure 4: Asymmetric satisfaction landscape: only failure modes carry negative signal; reactive integration straddles zero. 95% CIs on z-rating; Appendix F.6.

No reliable positive satisfaction signal. Treating grounding as the symmetric success axis, proactive recalls show no satisfaction signal (Figure 4): $\rho ~ = ~ - 0 . 0 5$ , against the reactive baseline $\rho ~ =$ +0.29. Neither grounding nor appropriateness predicts satisfaction at the event level (both $| \rho | < 0 . 1 ;$ Appendix F.6).

Mistimed recalls show the clearest negative signal. One directional signal does emerge, and it is negative: the 8 mistimed cases in rating-valid sessions sit at $\bar { z } = - 0 . 4 8 \mathrm { S D }$ (Figure 4)—when the system forces an ill-timed callback that overrides what the user is currently doing (e.g., interrupting a different topic the user just opened, or pivoting away from a closing turn), satisfaction drops by roughly half a within-user standard deviation.

Interpretation: users walk past most callbacks. Users continue the recalled topic in only 30% of cases (19/64), and appropriateness does not predict whether they do $( \rho = - 0 . 0 2 ;$ Appendix F.6). A well-timed callback can be walked past when it doesn’t match the user’s agenda, while a mistimed one derails (Table 4, bottom rows).

## 7 Conclusion

We present a 4-month, 40-user, 7-condition deployment of memory-augmented dialogue systems paired with the MEMUSE benchmark, which scores memory under integration-aware, user-cued conditions. Across our 7 conditions, Direct QA accuracy rises from 19.7% to 70.1% while user satisfaction does not change. A plausible explanation is a retrieval–integration gap: the same model that scores 78.8% on Direct QA references those facts in conversation only 7.9% of the time, and Natural Integration, not Direct QA, is associated with satisfaction within these moments. Realigning evaluation with user experience will require benchmarks that score integration under user-cued conditions. We release MEMUSE and the deployment artifacts to support this realignment of memory evaluation.

## Limitations

Scope of the null. All seven conditions include a summary of prior conversations, so our null result pertains to additional memory capacity beyond this summary baseline, not to memory writ large; cost and latency by condition, and a check that the null is not latency-mediated, are reported in Appendix G.

Memory moments are explicit-cue moments. MEMUSE detects memory-relevant moments through explicit linguistic signals (probes, reprovisions, recalls, reactions). The ∼3.5% rate at which these arise is therefore a lower bound: implicit moments where memory matters but is not flagged in the dialogue surface are missed by construction. The true prevalence may be substantially higher, which would only strengthen the case for evaluating integration rather than retrieval.

The integration–satisfaction link is small-N and observational. The within-instance result that successful integration is associated with higher satisfaction comes from 48 memory-moment sessions across 16 users; the integration variable is observational rather than randomized. We treat this as moderate evidence rather than a causal estimate. Full power, multiple-comparison, and clusteringrobust analyses—including Bonferroni correction, post-hoc power, and within-user permutation tests— are in Appendix C.7.

Automated measurement. Several constructs— Natural Integration scoring, response specificity, cross-session continuity, and prior-context dependence—are computed automatically. The Natural Integration judge was validated against two human annotators (Appendix B.4) and shows sensitivity to strictness level, but the key finding (no correlation with satisfaction) holds regardless of judge choice. Future work should similarly validate specificity and continuity measures. Natural Integration is also a binary judgment; the Reference score and the failure-mode taxonomy (§6.2) capture partial integration but were not designed or validated as ordinal measures, so a human-validated graded metric remains future work.

Sample and setting. Our 40 participants are geographically diverse proficient English speakers but predominantly female (92.5%); the diary setting may not generalize to other longitudinal interaction types (e.g., task-oriented assistants).

Privacy and release. Conversations were screened for residual identifiers before release; consent procedures and a one-month withdrawal window are described in the Ethics Statement. Data-release constraints—summarization in lieu of raw text in some cases—may limit downstream replicability.

## Ethics Statement

This study was approved by our institutional review board. All participants provided informed consent prior to participation and were compensated for their time; consent explicitly authorized public release of their conversations and satisfaction ratings for research use. Participants were instructed to self-curate their disclosures, sharing only content they were comfortable releasing publicly. A onemonth withdrawal window at the start of the 4- month deployment allowed participants to revoke consent; one participant did so and their data is excluded from the reported sample. Prior to release, the full deployment corpus was screened with GPT-5.4 for residual direct identifiers—personal names, locations, workplaces, and contact information referring to participants or to third parties— and flagged spans were reviewed by the research team and redacted. Raw conversation data was stored on encrypted institutional servers with access restricted to the research team during the study; human annotators who verified memory events accessed only de-identified transcripts. We release the full deployment corpus (1,872 sessions across 40 users and 7 memory conditions, with persession satisfaction ratings), the MEMUSE benchmark (72 de-identified instances with ground-truth facts and questions), and all evaluation and analysis code at https://huggingface.co/datasets/ RuiSumida/memuse and https://github.com/ ryuichi-sumida/memuse (release notes in Appendix H).

## Acknowledgments

This work was supported by JST BOOST Grant Number JPMJBS2407, JST Moonshot R&D JP-MJPS2011, and NEXUS JPMJNX25C1.

## References

Praveen Kumar Bodigutla, Aditya Tiwari, Spyros Matsoukas, Josep Valls-Vargas, and Lazaros Polymenakos. 2020. Joint turn and dialogue level user satisfaction estimation on multi-domain conversations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3897–3909.

Petter Bae Brandtzæg, Marita Skjuve, and Asbjørn Følstad. 2022. My AI friend: How users of a social chatbot understand their human–AI friendship. Human Communication Research, 48(3):404–429.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. In ECAI 2025 – 28th European Conference on Artificial Intelligence, volume 413 of Frontiers in Artificial Intelligence and Applications, pages 2993– 3000. IOS Press.

Yang Deng, Wenqiang Lei, Wai Lam, and Tat-Seng Chua. 2023. A survey on proactive dialogue systems: Problems, methods, and prospects. In Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI), pages 6583–6591.

Jan Deriu, Alvaro Rodrigo, Arantxa Otegi, Guillermo Echegoyen, Sophie Rosset, Eneko Agirre, and Mark Cieliebak. 2021. Survey on evaluation methods for

dialogue systems. Artificial Intelligence Review, 54(1):755–810.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2).

Daniël Lakens. 2017. Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. Social Psychological and Personality Science, 8(4):355– 362.

Dong-Ho Lee, Adyasha Maharana, Jay Pujara, Xiang Ren, and Francesco Barbieri. 2025. REALTALK: A 21-day real-world dataset for long-term conversation. arXiv preprint arXiv:2502.13270.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459–9474.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. 2023. Memochat: Tuning LLMs to use memos for consistent long-range open-domain conversation. arXiv preprint arXiv:2308.08239.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870.

Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seungwon Hwang, Dongha Lee, and Jinyoung Yeo. 2025. Towards lifelong dialogue agents via timeline-based memory management. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8631–8661.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their

own generations. In Advances in Neural Information Processing Systems (NeurIPS).

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology (UIST).

Ricardo Santana, Bahar Irfan, Erik Lagerstedt, Gabriel Skantze, and Andre Pereira. 2025. Speech-to-joy: Self-supervised features for enjoyment prediction in human–robot conversation. In Proceedings of the 27th International Conference on Multimodal Interaction (ICMI ’25), pages 238–248.

Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and statistical modeling with Python. In Proceedings ofthe 9th Python in Science Conference, pages 92–96.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS).

Marita Skjuve, Asbjørn Følstad, Knut Inge Fostervold, and Petter Bae Brandtzæg. 2021. My chatbot companion—a study of human–chatbot relationships. International Journal of Human-Computer Studies, 149:102601.

Ryuichi Sumida, Koji Inoue, and Tatsuya Kawahara. 2025. Enhancing long-term RAG chatbots with psychological models of memory importance and forgetting. Dialogue & Discourse, 16(2):74–110.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR).

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems 36: Datasets and Benchmarks Track (NeurIPS).

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large language models with long-term memory. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

## A Deployment Details and Filtering

## A.1 Participant and Annotator Recruitment and Compensation

Participants were recruited through a third-party participant-pool platform serving the Japanese consumer market. Each participant was paid JPY 3,000 per hour for the contracted minimum engagement of one hour per month, totaling JPY 12,000 per participant over the 4-month deployment (approximately USD 80 at the exchange rate during the study period). This rate is roughly 2.5–3× the Japanese statutory minimum wage (approximately JPY 900–1,163 per hour, varying by prefecture), and is above standard compensation guidelines for crowdsourced study participation.

Human annotators used for judge validation (Direct QA full set, Natural Integration recalibration pilot; Appendix B.4) and prior-context-dependence calibration (Appendix F.4) were recruited through the same platform and compensated at the same JPY 3,000 per hour rate, prorated to the time required to complete their respective annotation tasks.

## A.2 Study Interface and Session Structure

Users accessed the AI Diary system through a responsive web interface compatible with both mobile and desktop browsers. Before starting each session, users could review summaries of prior conversations.

## A.3 Verbatim Generation Prompts

Table 5 reports the verbatim per-condition generation prompts used during the deployment.

## A.4 Summary Generation Prompts

After every session, three summaries are generated by GPT-4.1-mini (gpt-4.1-mini-2025-04-14, max\_output\_tokens = 512, same temperature setting as Table 5): a running per-session summary, a one-sentence daily summary, and a 100- word monthly rollup. The per-session summary is the artifact fed back into the next session’s framing prompt as {summary} (Table 5); the daily and monthly summaries are shown to users on the session-start review screen (§A.2, Figure 5) but are not passed back into the model’s context. The three prompts are reproduced verbatim in Figure 6; placeholders “...” inside angle-bracketed tags stand for the prior-session summary and the relevant conversation history (the just-completed session for persession/daily; all sessions in the current month for monthly), with <previous\_summary> initialized to “No summaries available.” before the first session.

<table><tr><td rowspan=1 colspan=3>Block          Verbatim text</td></tr><tr><td rowspan=2 colspan=3>PERSONA      You are Luke, a friendly, kind, and understanding diary AI chatbot. Your goal is to make(shared)        conversations feel like chatting with a friend. You are also a reflective companionwho helps the user think about their day. - Ask questions only a third of the time. -Adapt your tone based on the user&#x27;s mood. - Aim for one or two sentences, like a realconversation. - You are also here to act as a personal diary.</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=3>FRAMING     This is the {N}th conversation. Summary of previous conversations: {summary}(shared)</td></tr><tr><td rowspan=1 colspan=3>SUMMARY-    System prompt: PERsONA  FRAMING ⊕ Generate the response using the conversationONLY          history. Messages: current-session turns only.</td></tr><tr><td rowspan=2 colspan=2>LC-k%</td><td rowspan=4 colspan=1>System prompt: PERsONA ⊕ FRAMING ⊕ Generate the response using the followingconversation history if necessary. Messages: {lc_history} (top k% user turns by importance,with their assistant pair-turns)  current-session turns.System prompt: PERsONA ⊕ FRAMING ⊕ Generate the response using the following relatedmemories if necessary: {mem_text} Messages: current-session turns only.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>RAG-k%</td></tr></table>

Table 5: Verbatim generation prompts for the deployed conditions. {N} is the session index; {summary} is the previous-session summary (generation prompt in Appendix A.4); {lc\_history} is the importance-filtered LC turn list; {mem\_text} for RAG is the top-10 cosine-retrieved memories formatted as “Memory from {date}: {text}”. Generation: GPT-4.1-mini, temperature 0, max\_completion\_tokens = 500. Offline reconstruction details in Appendix A.5.

![](images/14f9edd7dcc3cc461a543244883af2b614e8501eb9821d6b1294e82ee79a36f5.jpg)  
Figure 5: The AI Diary chat interface (mobile view). The system references prior conversations in real time.

## A.5 Deployed vs. Offline Settings

The deployed system used GPT-4.1-mini (temperature=0) with the identical persona prompt, condition-specific memory context, and response constraints as the offline MEMUSE reconstructions. The only difference is the generation model: offline evaluation additionally uses GPT-5.4 as a counterfactual capability probe (Appendix F.3). Judging uses GPT-5.4-nano (Zheng et al., 2023), validated against two human annotators (Appendix B.4).

## A.6 Engagement Patterns Over Time

Users averaged 46.8 sessions over the 4-month study (median 44, range 25–68). Monthly engagement remained stable: session frequency drifted slightly downward (13.4 → 10.3 per user per month from Nov to Feb), but words per session rose from 152 to 245, keeping total monthly word output near the 2,000-word target throughout. The pattern is consistent with deepening engagement rather than fatigue.

## A.7 Data Flow and Aggregation

Condition balance. Condition assignment was balanced by random rotation. A chi-squared test confirms no significant deviation from expected balance $( \chi ^ { 2 } = 1 4 6 . 0 , p = . 8 9 , d f = 1 6 8 )$ . Per-user condition counts average 6.3 sessions per condition (SD = 2.7).

## A.8 Operationalization Details

Response specificity. Response specificity is measured as the rate of proper noun references in system responses (capitalized words not at sentence boundaries, excluding first-person pronouns), normalized by response length. An alternative operationalization—shared named entities between user and system turns—independently confirms the specificity pattern (within-user ρ = .088, p < .001).

Cross-session continuity. Cross-session continuity is a composite score capturing the degree to which a user’s current utterance references prior conversations. We extract content words (lowercased tokens >3 characters, excluding 130 stop words) from the current session’s user text and all prior sessions’ user text, then compute four components: (1) full-history word overlap: |current ∩ all prior|/|current|; (2) recent overlap: the same ratio restricted to the 3 most recent prior sessions; (3) entity overlap: shared capitalized tokens (non-sentence-initial, >2 characters) between current and all prior sessions; and (4) temporal reference count: regex matches for explicit backward references (e.g., “last time,” “as I mentioned,” “do you remember”), capped at 3. The composite score is $0 . 3 \times ( 1 ) + 0 . 3 \times ( 3 ) + 0 . 2 \times ( 2 ) + 0 . 2 \times ( 4 )$ , yielding a [0, 1] score. We additionally isolate the temporalreference component as a separate measure of explicit memory cueing. Explicit memory cueing independently predicts satisfaction (p < .01). All three operationalizations independently predict satisfaction $( p < . 0 1 )$ .

![](images/6993fd204336bff5b34f19fd596ef211611737bc9865c0039be1a22e9414e8cf.jpg)  
Figure 6: Verbatim post-session summary-generation prompts used by the deployed system.

<table><tr><td>Stage</td><td>Users</td><td>Sessions</td></tr><tr><td>Enrolled</td><td>40</td><td>1,872</td></tr><tr><td>Valid conditions (0–6)</td><td>40</td><td>1,770</td></tr><tr><td>Rating &gt; 0</td><td>40</td><td>1,713</td></tr><tr><td>— Adversarial users</td><td>38</td><td>1,635</td></tr><tr><td>— Low-variance raters</td><td>29</td><td>1,270</td></tr><tr><td>Subsets</td><td></td><td></td></tr><tr><td>MemUse detection</td><td>40</td><td>1,872</td></tr><tr><td>MemUse instances</td><td></td><td>72</td></tr><tr><td>Memory-moment sessions</td><td>16</td><td>48</td></tr><tr><td>MemUse instances (filtered)</td><td></td><td>55</td></tr></table>

Table 6: Data flow from enrollment to analysis. MemUse detection was run on all 1,872 enrolled sessions, regardless of rating status. Detection yielded 73 reactive instances, one of which (from annotator\_40, flagged for pasting AI-generated content; see §5) is excluded from the released benchmark, giving 72 benchmark instances. Excluding instances from the remaining 10 rating-filtered users yields 55 instances, distributed across 48 unique memory-moment sessions with ratingvalid ratings (the within-instance analysis slice of §6.1). Table 7 maps each reported analysis to its unit of aggregation.

Prior-context dependence. Prior-context dependence (PCD) measures how much response quality would suffer without memory of prior conversations (0–3 scale); see Appendix F.4 for annotation procedure and judge calibration. A model with length and specificity outperforms QA accuracy on AIC (∆AIC = −3.5), and adding PCD provides no further improvement: engagement dimensions capture the session-level satisfaction variance that QA accuracy misses, while memory demand does not independently contribute.

## A.9 Statistical Analysis: Full Specification

All inferential models are linear mixed-effects models (LMMs) fitted with the Python statsmodels package (Seabold and Perktold, 2010). The general

<table><tr><td>Analysis</td><td>Section</td><td>Unit of analysis (one N row per...)</td><td></td><td>What is averaged / computed per unit</td></tr><tr><td colspan="5">Population / deployment level (all filtered sessions)</td></tr><tr><td>Existing-benchmark and §4 matched-MemUse null vs. satisfaction (Table 13)</td><td></td><td>Session (filtered, rat- 1,270 ing &gt; 0)</td><td>sions users)</td><td>ses- Session z-rating paired with the condition- (29 level mean of QA Acc, Direct QA, and matched-MemUse Natural Integration for that session&#x27;s condition.</td></tr><tr><td>Engagement LMM (Appendix F.2)</td><td>predictors App. F.2 </td><td>Session (filtered)</td><td>1,270 sions</td><td>ses- rating_z ~ length + specificity + cross- session continuity + cond + (1|user); covari- ates are session-level.</td></tr><tr><td colspan="5">Benchmark scoring (every instance re-run in every condition)</td></tr><tr><td>MemUse Natural Integra- §5 tion (Table 13, column NI)</td><td>response</td><td>504</td><td>sponses</td><td>Instance × condition 72 × 7 = Binary Natural Integration judgment per re- re- sponse; condition means reported.</td></tr><tr><td>MemUse Direct QA (col- §5, §6.2 umn Direct QA in Table 13)</td><td></td><td>Instance × condition 504 response</td><td>sponses covering  $3 1 6 \times { \bar { 7 } } =$  2,212</td><td>re- Per-response question accuracy (fraction cor- rect over that instance&#x27;s 3–5 questions); con- dition means reported.</td></tr><tr><td>ment (Table 12)</td><td>Multi-judge / human agree- App. B.4 </td><td>Instance × condition 56 Natural Pairwise Cohen&#x27;s κ and Fleiss’κ between response</td><td>question judgments Integration (recalibra- QA (full)</td><td>the LLM judge and two human annotators; Natural Integration re-annotated on a strati- tion pilot); fied 56-item sub-sample with newly recruited 316 Direct annotators after the original round showed leniency divergence.</td></tr><tr><td colspan="5">Memory-moment subset (deployed-condition responses only)</td></tr><tr><td>Per-instance integration decoupling (ρ = −0.009)</td><td>retrieval- §6.2</td><td>Question (deployed 207 ques- Paired Direct QA vs. MEMUsE Reference condition)</td><td>stances, 48 computed across the 207 pairs. sessions, 16</td><td>tions nested correctness on the same (model, recon- in 55 in- structed context, question) triple; Spearman</td></tr><tr><td>Within-instance head-to- §6.1 head (ρNI = +.29; LMM βNI = +0.60)</td><td></td><td>sion (deployed condi- (16 users; Direct tion)</td><td>users sessions by integrates).</td><td>Memory-moment ses- 48 sessions Session y: z-rating. Session x: mean  $\mathrm { \bar { Q A } }$  accuracy, mean Reference rate, 55 instances and binary Natural Integration (Natural collapsed to Integration=1 if any instance in the session</td></tr><tr><td>Successful-integration main §6.1 effect (β = +0.556, p=.082)</td><td></td><td>sion</td><td>averaging) Integration=1;the head-to-head. 37 Natural Integration=0)</td><td>Memory-moment ses- 48 sessions LMM rating_z ~ Natural Integration + (11 Natural (1user), same session-level aggregation as</td></tr></table>

Table 7: What is averaged where. Each row names an analysis in the paper and its unit of aggregation, sample size, and aggregation rule. The benchmark comprises 72 instances and 316 questions; every instance is re-run in every condition, producing 504 Natural Integration judgments and 316 question judgments (columns NI, Direct QA in Table 13). The memory-moment subset restricts to each instance’s deployed condition only, yielding 55 filtered instances → 48 sessions → 207 questions. Session-level analyses collapse within-session instances by averaging question accuracy and OR-ing Natural Integration before correlating with the session’s rating.

form is

$$
\begin{array} { c } { { y _ { i j } = \beta _ { 0 } + \sum _ { k } \beta _ { k } x _ { i j k } + u _ { i } + \varepsilon _ { i j } , } } \\ { { u _ { i } \sim \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } ) , \varepsilon _ { i j } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } } \end{array}\tag{1}
$$

where $y _ { i j }$ is the within-user z-scored satisfaction rating for session $j$ of user $i , x _ { i j k }$ are the fixedeffect predictors specific to each analysis (continuous predictors z-standardized), $\beta _ { k }$ are the corresponding coefficients, u<sub>i</sub> is a per-user random intercept, and $\varepsilon _ { i j }$ is residual error. Within-user zscoring removes individual baseline differences: $y _ { i j } = ( r _ { i j } - { \bar { r } } _ { i } ) / s _ { i }$ , where $r _ { i j }$ is the raw rating and r¯ , s are the per-user mean and standard deviation. Reported $\beta ^ { \bullet } { \bf s }$ denote coefficients from this family unless stated otherwise; Spearman $\rho \mathbf { \ ' } \mathbf { s }$ appear as the non-parametric analogue. Confidence intervals default to asymptotic Wald CIs from the LMM; for small-N specifications we additionally report userclustered percentile bootstrap CIs (the 2.5–97.5% range of 10,000 user-resampled point estimates), which are more reliable when the Wald approximation is suspect. Model comparisons use likelihoodratio $\chi ^ { 2 }$ tests on nested LMMs and ∆AIC for nonnested comparisons (lower AIC indicates better fit). When reporting null effects we report Cohen’s d alongside TOST equivalence (Lakens, 2017) at δ = ±0.25 z-units to make a positive equivalence claim rather than rely on failure-to-reject. Individual models are fitted with restricted maximum likelihood (REML); model comparisons use maximum likelihood (ML) with likelihood ratio tests and AIC.

<table><tr><td>Cond.</td><td>Nov</td><td>Dec</td><td>Jan</td><td>Feb</td></tr><tr><td>Summary</td><td>61</td><td>47</td><td>38</td><td>50</td></tr><tr><td>LC-10</td><td>48</td><td>45</td><td>45</td><td>48</td></tr><tr><td>LC-50</td><td>35</td><td>52</td><td>42</td><td>51</td></tr><tr><td>LC-100</td><td>30</td><td>52</td><td>45</td><td>64</td></tr><tr><td>RAG-10</td><td>38</td><td>43</td><td>56</td><td>38</td></tr><tr><td>RAG-50</td><td>30</td><td>54</td><td>50</td><td>41</td></tr><tr><td>RAG-100</td><td>36</td><td>57</td><td>50</td><td>24</td></tr><tr><td>Total</td><td>278</td><td>350</td><td>326</td><td>316</td></tr></table>

Table 8: Per-month condition counts (29 filtered users). Balance is maintained across time.

## B MemUse Construction and Judge Validation

## B.1 Detection and Judging Pipelines

Event detection. MEMUSE instances were detected using GPT-5.4 (temperature=0) applied to each of 1,872 session transcripts independently. No prior-session summaries were provided—detection relies solely on explicit linguistic signals within each transcript. The prompt requires signals such as “do you remember,” “as I mentioned before,” or “you recommended X last time,” and rejects vague or ambiguous cases.

Judge prompts. Table 10 shows the verbatim user-side prompts for both judging modes. All judges use the system prompt Answer ONLY ‘yes’ or ‘no’. at temperature 0; the Direct QA system prompt additionally adds “Does the response show knowledge of the stated fact?” after the yes/no instruction. The same prompts are used across the judge model (GPT-5.4-nano) and for human annotators.

<table><tr><td>Event Type</td><td>Det.</td><td>Ver.</td><td>Prec.</td></tr><tr><td>User memory probe</td><td>12</td><td>11</td><td>91.7%</td></tr><tr><td>User re-provision</td><td>63</td><td>62</td><td>98.4%</td></tr><tr><td>System proactive recall</td><td>70</td><td>66</td><td>94.3%</td></tr><tr><td>User memory reaction</td><td>9</td><td>8</td><td>88.9%</td></tr><tr><td>Total</td><td>154</td><td>147</td><td>95.5%</td></tr></table>

Table 9: Detection results. All events individually verified against raw conversation transcripts.

Generation prompts. The verbatim percondition generation prompts are reported in Appendix A.3 (Table 5).

## B.2 Anatomy of a MEMUSE Instance: Memory Probe Example

Table 3 (main text) shows a re-provision instance (N=62). For completeness, here is one user memory probe (N=10): trigger “Remember when I told you I got a lot of lemons?” → ground-truth from session 14 (the user’s family was sick with flu and a friend gave her many lemons; she first made lemon jelly, which the whole family ate and said was delicious) → questions Q: Who gave the user the lemons? A: A friend. Q: What did the user first make with the lemons? A: Lemon jelly. Q: What was thefamily’s reaction? A: Everyone ate it and said it was delicious. The benchmark pairs 72 such de-identified moments with ground-truth facts and 3–5 questions each (316 total).

## B.3 Benchmark Evaluation Pipeline

QA benchmark evaluation (Table 14). Lo-CoMo, LUFY, and RealTalk were evaluated using GPT-4.1-mini for both generation and judging. For each benchmark, we applied the same 7 memory conditions by constructing context according to each condition’s specification: Summaryonly provides only the conversation summary; LC-k% includes the top k% of prior user utterances (ranked by the importance model described in §3) along with their preceding assistant turns; RAG-k% indexes importance-filtered utterances and retrieves the top-10 most similar documents via BAAI/bge-small-en embeddings with cosine similarity. No explicit token budget was applied for LC conditions; all qualifying turns were included. The generation prompt instructed the model to answer in fewer than 5 words, with <SUMMARY>, <RELEVANT\_MEMORIES>, and <CONVERSATION\_HISTORY> sections. Each answer was scored by 10 independent GPT-4.1-mini judge calls producing correct (1), incorrect (0), or “I don’t know” (2); accuracy = correct / total across all judge calls. All conditions used the same precomputed summaries.

<table><tr><td>Mode</td><td>Judge inputs</td><td>Judge prompt (verbatim; system prompt: A “yes&quot; means Answer ONLY ‘yes&#x27;or ‘no&#x27;.)</td><td></td></tr><tr><td>Natural Integra- tion (Mode 1)</td><td>trigger quote, ground-truth facts, system&#x27;s natural re- Expected: {ground_truth_facts} sponse</td><td>1 User said: {trigger_quote} Assistant responded: {response} Does the response demonstrate memory of the referenced topic?</td><td>Judge makes a single overall yes/no decision that the response shows memory of the refer- enced topic. No rubric or worked examples are</td></tr><tr><td>Direct QA (Mode 2)</td><td>system&#x27;s response, fact- Assistant&#x27;s response: {response} seeking question, expected Question: {question} answer</td><td>Expected answer: {expected_answer} Does the assistant&#x27;s response demonstrate knowledge of this fact?</td><td>provided. The specific target fact (paraphrase accepted) is present in the response.</td></tr></table>

Table 10: How each mode is judged. All prompts are used verbatim across all judge models and for human annotators. Natural Integration is the primary benchmark metric; Direct QA is a fact-anchored diagnostic (§6.2). Natural Integration scoring deliberately does not specify what “memory of the referenced topic” requires—by design, it measures whether integration appears to a reader rather than whether any particular token matches.

MEMUSE evaluation (Table 13). MEMUSE instances were evaluated using GPT-4.1-mini (temperature=0, max 500 tokens) and GPT-5.4 for generation, with GPT-5.4-nano for binary yes/no judging (temperature=0). Context was reconstructed from the actual AI Diary conversation data using the same importance model and condition logic as the deployed system. RAG retrieval used text-embedding-3-small embeddings with cosine similarity (top-k = 10, documents truncated to 500 characters for embedding, 300 characters for retrieval display). The summaries used offline were the same summaries stored during the live deployment. Question accuracy is the fraction of questions judged “yes”; Natural Integration accuracy is the binary yes/no rate across instances.

Summary coverage. Since all conditions include a summary, a natural concern is that summaries already satisfy most memory needs, reducing the marginal value of additional memory. We assess summary coverage by computing content-word overlap between MEMUSE ground-truth facts and the summary available before each session. Of 72 instances, only 1 (1.4%) has ≥50% coverage, 60 (83.3%) have 10–50% partial coverage, and 11 (15.3%) have <10%. Summaries average

127 words (range: 3–246) and capture the gist of prior conversations but not the specific details needed for MEMUSE questions. This suggests the null result is not primarily driven by summaries already meeting memory needs—rather, the specific factual details tested by MEMUSE are largely absent from summaries, yet satisfaction remains unaffected by whether the system has access to those details.

## B.4 Judge Sensitivity and Multi-Judge Agreement

Human annotator validation. We validated the LLM judge against two human annotators on the 316 per-question judgments (full set), and on a 56- item recalibration pilot for the Natural Integration judgment (8 items per condition, stratified across the 7 conditions; Table 12). The recalibration pilot was motivated by the structure of agreement in the original round, described below.

The structure of agreement is informative. In the original validation round, the same two annotators agreed substantially on concrete question-level judgments (κ = 0.70) but diverged sharply on the reader-level Natural Integration judgment (κ = 0.19, positive rates 72.2% / 22.5%). Question-level verification is largely unambiguous (does the response state fact X?), whereas Natural Integration— a holistic judgment of whether a natural response demonstrates memory of the referenced topic—is more sensitive to where each rater sets their leniency threshold. We therefore ran a recalibration pilot on a stratified 56-item sub-sample (8 items per condition) with two newly recruited annota-

<table><tr><td>Metric</td><td>Strict</td><td>Lenient</td></tr><tr><td>GPT-4.1-mini generation</td><td></td><td></td></tr><tr><td>Natural Integration</td><td>25.4%</td><td>76.1%</td></tr><tr><td>Direct QA</td><td>7.6%</td><td>24.0%</td></tr><tr><td>GPT-5.4 generation</td><td></td><td></td></tr><tr><td>Natural Integration</td><td>49.5%</td><td>84.3%</td></tr><tr><td>Direct QA</td><td>11.8%</td><td>28.3%</td></tr></table>

Table 11: Strict (GPT-5.4-nano) vs. lenient (GPT-5.4) judge on the same generated responses.

<table><tr><td>Rater</td><td>NI</td><td>Direct QA</td></tr><tr><td>LLM judge (GPT-5.4-nano)</td><td>51.8%</td><td>11.9%</td></tr><tr><td>Human Annotator 1</td><td>55.4%</td><td>13.6%</td></tr><tr><td>Human Annotator 2</td><td>51.8%</td><td>8.9%</td></tr><tr><td>Pairwise Cohen&#x27;s κ</td><td></td><td></td></tr><tr><td>LLM-Human 1</td><td>0.43</td><td>0.62</td></tr><tr><td>LLM-Human 2</td><td>0.21</td><td>0.65</td></tr><tr><td>Human 1–Human 2</td><td>0.57</td><td>0.70</td></tr><tr><td>Fleiss’κ (3 raters)</td><td>0.40</td><td>0.65</td></tr></table>

Table 12: Agreement between the production LLM judge (GPT-5.4-nano, used throughout the paper) and two human annotators. Natural Integration: $N = 5 6$ recalibration pilot; two newly recruited annotators. Direct $Q A \colon N = 3 1 6$ original full validation; two original annotators. Both constructs reach substantial human–human agreement $( \kappa = 0 . 5 7$ , 95% bootstrap CI [0.34, 0.78], $N { = } 5 6$ for Natural Integration; $\kappa = 0 . 7 0$ for Direct QA). LLM–human κ on Natural Integration is asymmetric (LLM is closer to Annotator 1 than to Annotator 2), reflecting the residual interpretive latitude of a reader-level judgment. As a judge-stability check, re-running the Natural Integration pilot with GPT-5.5 $( \mathsf { g p t } - 5 . 5 - 2 0 2 6 - 0 4 - 2 3 )$ yields nearly identical labels (LLM–LLM $\kappa = 0 . 8 9$ , 94.6% agreement on the 56 items).

tors. Each annotator was first walked through 3 worked examples by one author and given the opportunity to raise any concerns about the rubric before beginning independent annotation; this calibration step was absent in the original round. The recalibrated round recovered matched positive rates (55.4% / 51.8%) and substantial human–human agreement (Cohen’s $\kappa = 0 . 5 7$ , 78.6% raw; Fleiss’ $\kappa = 0 . 4 0$ across the two humans and the GPT-5.4-nano judge), confirming that the construct is reliably annotatable when raters are calibrated. As a rater-robustness check on the GPT-5.4 capability probe, the condition-level Spearman correlation between Natural Integration accuracy and mean satisfaction is uniformly non-positive across all three raters (using original-round annotators on the parallel GPT-5.4 set): LLM $\rho = - 0 . 2 5$ , Human 1 $\rho = - 0 . 5 8$ , Human $2 \rho = - 0 . 3 4 ( \mathrm { a l l } N = 7 .$ none significant). This is a counterfactual comparison— users never experienced GPT-5.4 outputs—so we treat it only as evidence that judge choice does not flip the sign; the headline matched-model satisfaction analysis is reported in §4.

## C Population-Level Null: Robustness

## C.1 Detailed Per-Condition Results

The deployment-level null reported in §4 is supported by a battery of tests beyond the headline LMM coefficient. An LMM regressing zscored satisfaction on condition-level QA accuracy with a user random intercept yields $\beta = - 0 . 0 0 1$ $( N = 1 , 2 7 0 )$ . Kruskal–Wallis across the 7 conditions and a within-user permutation test (10,000 permutations) both agree. All pairwise condition effect sizes against Summary-only are negligible $( | d | < 0 . 0 6 )$ , and equivalence testing with $\delta \ : = \ : \pm 0 . 2 5$ on the z-rating scale confirms practical equivalence for every memory condition vs. Summary (all TOST $p < . 0 5 )$ .

Matched-model MEMUSE scores are similarly null at the population level. The per-session Spearman correlation between the deployed GPT-4.1- mini’s Natural Integration score and the session’s z-rating is $\rho = + 0 . 0 0 8 \ : ( N = 1 , 2 7 0 )$ . Fitting an LMM of the form

$$
\begin{array} { r l } { y _ { i j } \sim \mathsf { M E M O R Y C O N D I T I O N } } & { } \\ & { \times \mathrm { H A S M E M O R Y M O M E N T } } \\ & { + \left( 1 \mid \mathsf { u s e r } _ { i } \right) } \end{array}
$$

finds no significant effects for memory moments, conditions, or their interaction: even when we condition on whether a session contains a detected memory moment, condition-level capacity does not move satisfaction.

The per-session correlation footnote from the main table summarizes the same picture in three numbers: existing QA Acc vs. Sat $\rho = . 0 0 4 ;$ Direct QA vs. Sat $\rho = - . 0 3 ;$ matched-model MemUse Natural Integration vs. Sat $\rho = +$ .008. None of the four metric columns in Table 13 moves with satisfaction at the deployment level.

## C.2 Per-Benchmark QA Accuracy

## C.3 Sensitivity Analyses

The capacity null is robust across user-filtering choices: Table 15 reports the omnibus and mixedeffects tests on three nested subsets (full $N { = } 4 0$ drop two adversarial users, original 29-user filter), Figure 7 plots the per-condition z-scored ratings within each subset, and Figure 8 shows that condition-vs-Summary effect sizes are uniformly negligible with 95% intervals straddling zero.

<table><tr><td>Cond.</td><td>Existing (%)</td><td>Direct QA Acc Direct QA Ref. NI (%) (%) (%)</td><td>Sat.  $( \mathbf { Z } \pm \mathbf { S } \mathbf { E } )$ </td></tr><tr><td>Summary</td><td>19.7</td><td>44.9 7.9</td><td> $2 3 . 6 + . 0 2 \pm . 0 7$ </td></tr><tr><td>LC-10</td><td>40.4</td><td>53.0 5.6</td><td> $2 5 . 0 + . 0 5 \pm . 0 7$ </td></tr><tr><td>LC-50</td><td>65.9</td><td>72.4</td><td>7.2  $2 5 . 0 + . 0 2 \pm . 0 7$ </td></tr><tr><td>LC-100</td><td>70.1</td><td>78.8</td><td>7.9  $2 2 . 2 \ : - . 0 4 \pm . 0 7$ </td></tr><tr><td>RAG-10</td><td>40.6</td><td>49.2</td><td>7.0  $2 7 . 8 + . 0 4 \pm . 0 8$ </td></tr><tr><td>RAG-50</td><td>61.0</td><td>57.7</td><td>8.0  $2 5 . 0 - . 0 4 \pm . 0 8$ </td></tr><tr><td>RAG-100</td><td>63.7</td><td>62.2</td><td>7.3  $2 6 . 4 \ : - . 0 2 \pm . 0 8$ </td></tr></table>

Table 13: Numerical values for Figure 1. Existing QA Acc: average accuracy on existing benchmarks (Lo-CoMo, LUFY, RealTalk; per-turn QA; Appendix C.2). MemUse Direct QA: GPT-4.1-mini accuracy when each MEMUSE question is asked literally with the same reconstructed context—the apples-to-apples analogue of existing benchmarks on real memory moments. Ref.: MEMUSE Reference—whether each fact appears in the conversational response (question level). NI: MEMUSE Natural Integration—a reader-level binary judgment of whether the response demonstrates memory of the topic (instance level; the benchmark’s primary metric). Sat.: z-scored satisfaction ± SE. $N = 1 7 0 – 2 0 7$ per condition. Per-session correlations: Existing QA Acc vs. Sat $\rho { = } . 0 0 4 \left( p { = } . 8 6 \right)$ ; Natural Integration vs. Sat $\rho { = } + . 0 0 8$ $( p \mathrm { = } . 7 7 )$ ; within-user permutation $p { = } . 9 1 ;$ all $d { < } 0 . 0 6$
<table><tr><td>Condition</td><td>LUFY</td><td>LoCoMo</td><td>RealTalk</td><td>Avg.</td></tr><tr><td>Summary</td><td>34.0</td><td>15.3</td><td>9.8</td><td>19.7</td></tr><tr><td>LC-10</td><td>53.8</td><td>41.1</td><td>26.4</td><td>40.4</td></tr><tr><td>LC-50</td><td>83.1</td><td>63.4</td><td>51.2</td><td>65.9</td></tr><tr><td>LC-100</td><td>85.1</td><td>69.7</td><td>55.4</td><td>70.1</td></tr><tr><td>RAG-10</td><td>55.2</td><td>37.1</td><td>29.5</td><td>40.6</td></tr><tr><td>RAG-50</td><td>80.5</td><td>57.3</td><td>45.3</td><td>61.0</td></tr><tr><td>RAG-100</td><td>83.8</td><td>60.6</td><td>46.8</td><td>63.7</td></tr></table>

Table 14: QA accuracy (%) on three existing benchmarks. All three show the same pattern: accuracy increases with memory capacity but none predicts satisfaction. The “QA Acc” column in Table 13 reports the average.

## C.4 Natural Integration Aggregation Rule (Session-Level)

The 55 filtered MEMUSE instances reduce to 48 memory-moment sessions because a single session can contain more than one reactive memory moment (e.g., a user memory probe followed by a re-provision, or several re-provisions threaded through a longer chat). Because satisfaction is reported once per session, we aggregate withinsession before pairing with the session’s z-rating: continuous metrics (Direct QA accuracy, Reference rate) are averaged across the session’s instances, and the binary MEMUSE Natural Integration score is collapsed by an any-success rule (NI = 1 if at least one instance in the session was judged to integrate, else 0). In practice, the multi-instance subset is small: 45 sessions contain a single instance, 1 contains two, and 2 contain three.

<table><tr><td>Sample</td><td> $N _ { u }$ </td><td> $N _ { s }$ </td><td> $\mathbf { K W } p$ </td><td> $\mathbf { M E } \mathbf { \Xi } \beta$ </td><td> $\mathbf { M E } p$ </td></tr><tr><td>All 40 users</td><td>40</td><td>1,713</td><td>.93</td><td>-.001</td><td>.71</td></tr><tr><td>No adversarial</td><td>38</td><td>1,635</td><td>.93</td><td>-.001</td><td>.62</td></tr><tr><td>Original filter</td><td>29</td><td>1,270</td><td>.91</td><td>-.001</td><td>.47</td></tr></table>

Table 15: Null result across user subsets. KW: Kruskal-Wallis. ME: mixed-effects model (z-scored rating ∼ QA accuracy + (1|user)).

The within-instance head-to-head and successful-integration tests (§6.1) pair each session’s z-rating with this session-level Natural Integration label. Because three of the 48 memory-moment sessions contain more than one reactive instance, the session-level label depends on the aggregation rule. Our default is any-success (NI=1 if at least one instance in the session is judged integrated). We re-ran the analysis under three alternatives: majority-vote (ties broken to 1), all-success (every instance must integrate), and mean-NI (continuous 0–1). The hinge is a single session (annotator\_34, session 32) in which 1 of 3 instances was judged integrated—this flips the label between any-success and the stricter rules. The other two multi-instance sessions have all-zero Natural Integration vectors and are unaffected by rule choice.

## C.5 Cross-Session Continuity Confound Control

Cross-session continuity is partly mechanically confounded with cumulative history size. Figure 9 re-operationalizes the continuity score under three definitions (full-history, fixed 10-session window, history-size-normalized) and shows that all three independently predict satisfaction even after the mechanical trend is removed.

## C.6 User Heterogeneity

Figure 10 plots each user’s per-condition memoryvs-summary contrast: only 11 of 29 users show a directional preference for memory conditions, with

![](images/eeabf65b4c7798c902a597904b3d03047cbcda7ba4fc1da7633d59c80072e0b7.jpg)  
Figure 7: Sensitivity analysis: condition effects on z-scored satisfaction across three user subsets. The null result is robust to all filtering decisions.

![](images/84ce68adca3af5f0d04968a3137bd4abd4541504b62a124f24eedfac9e833aea.jpg)  
Figure 8: Effect sizes (Cohen’s d) for each condition vs. Summary-only with 95% CIs. All effects are negligible and span zero.

<table><tr><td>Rule</td><td>#NI=1</td><td> $\rho$ </td><td> $p _ { \rho }$ </td><td> $\beta _ {  { \mathbf { N I } } }$ </td><td> $p _ { \beta }$ </td></tr><tr><td>Any-success (default)</td><td>11/48</td><td></td><td></td><td>+.289 .046 +.556 .082</td><td></td></tr><tr><td>Majority-vote</td><td>10/48</td><td> $+ . 2 4 9$ </td><td>.088</td><td> $+ . 5 4 3$ </td><td>.098</td></tr><tr><td>All-success</td><td>10/48</td><td> $+ . 2 4 9$ </td><td>.088</td><td> $+ . 5 4 3$ </td><td>.098</td></tr><tr><td>Mean-NI (0–1)</td><td></td><td> $+ . 2 8 3$ </td><td>.051</td><td> $+ . 5 6 3$ </td><td>.088</td></tr></table>

Table 16: Sensitivity of the successful-integration main effect to the session-level Natural Integration aggregation rule. $\rho \colon$ Spearman Natural Integration vs. rating ${ \bf \Xi } _ { - } { \bf \Lambda } _ { Z } ; ~ \beta _ { \mathrm { N I } } ;$ LMM coefficient from rating $_ - z \sim$ Natural Integration + (1|user) fit with ML. $N { = } 4 8$ memorymoment sessions, 16 users. The point estimate is robust $( \beta _ { \mathrm { N I } } \in [ + . 5 4 3 , + . 5 6 3 ]$ , a range of 0.02); the Spearman p-value crosses the conventional 0.05 threshold depending on the rule, driven by the single multi-instance session described above (annotator\_34 s32). The LMM $p$ hovers in [.082, .098] across all rules and does not cross 0.05. We report any-success as the default because it treats any in-session demonstration of memory integration as a positive outcome, consistent with the reader-level framing of the Natural Integration metric.

random-slope variance 0.042, indicating that the population-level null does not mask a uniformlyshaped subgroup effect.

![](images/136c80e0256048b7a68fd33775cc20bbe63ab3a027bcf9c9b8d07909f28c6635.jpg)  
Figure 9: Cross-session continuity over time under three operationalizations. Left: Original (full history) shows strong increase $( \rho = . 6 8 )$ . Center: Fixed-window (last 10 sessions) reduces but does not eliminate the trend $( \rho = . 2 7 )$ . Right: Normalized by history size shows slight decrease $( \rho = - . 1 4 )$ , confirming partial mechanical confound. All three variants independently predict satisfaction $( p < . 0 1 )$

## C.7 Strength of Evidence: Multiple-Comparison and Power

Our findings span a wide range of statistical power, and we interpret them accordingly. The populationlevel null—existing-benchmark QA accuracy and matched-model MEMUSE scores do not predict user satisfaction—is well powered $( N = 1 { , } 2 7 0$ sessions across 29 users, equivalence-confirmed; §4). The per-instance retrieval–integration decoupling $( \rho = - 0 . 0 0 9$ on 207 paired questions nested in 48 sessions; §6.2) is likewise well powered and robust: the same model reading the same reconstructed context answers 79% of questions when queried directly but references only 8% when each question is evaluated on its natural conversational response. These two findings form the empirical core of the paper and survive Bonferroni correction at $\alpha / 6 = 0 . 0 0 8 3$ (a six-test family-wise threshold), as does the engagement-covariate model comparison reported in Appendix F.2 $( p < . 0 0 1 )$

Memory Summary Satisfaction Difference (z-scored)  
Per-User Memory Condition Effect (11/29 prefer memory)  
![](images/4377e5529163ad58a32c7703b850074a237a76fa10eea0b3ac9d38286127d558.jpg)  
Figure 10: Per-user memory condition effect (memory − summary z-scored satisfaction). Only 11 of 29 users show a positive preference for memory conditions. Random slope variance = 0.042.

The within-instance integration-to-satisfaction association (§6.1, $N \ = \ 4 8$ memory-moment sessions, 16 users) is a lower-power observational test: LMM $\beta ~ = ~ + 0 . 5 5 6 , ~ p ~ = ~ . 0 8 2$ 95% Wald CI roughly $[ - 0 . 0 7 , + 1 . 1 9 ]$ , straddling zero; the non-parametric Spearman analogue $( p =$ .046) does not survive Bonferroni correction at $\alpha / 6 .$ A user-clustered nonparametric bootstrap (10,000 iterations) gives a tighter percentile CI of $[ + 0 . 1 2 , + 0 . 9 8 ]$ that excludes zero (one-sided $p _ { \mathrm { b o o t } } = . 0 0 4 ) ;$ a within-user permutation is more conservative $\left( p = . 2 3 \right.$ , with only 11 informative clusters of 16 users). We offer the finding as conditional evidence consistent with the mode-contrast reading rather than a standalone causal claim: the bootstrap excludes zero and the relationship is robust to user clustering, but the asymptotic Wald and the permutation analogue both indicate the evidence is moderate rather than strong, consistent with $N = 4 8$ . The point estimate is robust to analyst choice $( \beta ~ \in ~ [ + 0 . 5 4 3 , + 0 . 5 6 3 ]$ across four session-level aggregation rules; Appendix C.4).

Causal interpretation. The 7-condition randomization supports causal claims about the null direct effect at the population level. The matched-model integration-to-satisfaction association (§6.1) and the engagement-dimension analyses involve nonrandomized variables and should be interpreted as observational associations, not established causal pathways.

Post-hoc power. The study achieves 84.5% power for $d = 0 . 2 5$ but only $2 0 . 9 \%$ for $d = 0 . 1 0 ;$ detecting the latter would require ${ \sim } 7 0$ users.

## D Retrieval–Integration Gap: Diagnostics

## D.1 Failure Mode Analysis

## E Cross-Model and Prompt Ablations

## E.1 Cross-Model MEMUSE: Figure and Full Numbers

Figure 2 plots Direct QA and Natural Integration across the 7 conditions for three models; Table 18 reports the full numbers, including MEMUSE $R e f -$ erence—per-question whether the fact appears in the conversational response.

N=73 vs. released N=72. The cross-model and prompt-ablation analyses (this appendix and E.3) were conducted on the full set of 73 detected reactive instances. The released benchmark of 72 (§5, Table 6) excludes one probe (annotator\_40, session 13) flagged in our post-hoc rating-reliability audit for likely AI-paste contamination on the user side—exactly the construct the benchmark is built to avoid. This single-instance difference shifts any per-condition Natural Integration cell by at most $1 / 7 2 \approx 1 . 4 \ : \mathrm { p p }$ , leaves all reported conclusions and significance patterns unchanged, and preserves the 71-point Direct QA vs. Reference gap.

The cross-model evaluation uses the identical pipeline as the main paper (same 73 groundtruth instances, same context-construction code per condition, same judge prompt and judge model). Only the generation model varies. GPT-5.5 and Gemini 3.1 Pro are reasoning models; we allowed max\_completion\_tokens = 4,000 to accommodate internal reasoning while leaving room for short conversational responses. Reproduction scripts and per-instance outputs are included in the released analysis code.

## E.2 Modern Memory Systems on MEMUSE

Table 19 (§6.2) reports Mem0 and Letta on the 72 released instances, scored with the same releaseformat judge pipeline (GPT-5.4-nano, temp=0) and the same 316 fact targets as the paper. Mem0 (mem0ai 2.0.11, open-source Memory with a local

<table><tr><td>Failure Mode</td><td>N</td><td>%</td></tr><tr><td>Generic (ignores context)</td><td>28</td><td>38.9</td></tr><tr><td>Partial recall (gist only)</td><td>19</td><td>26.4</td></tr><tr><td>Hallucinated recall</td><td>17</td><td>23.6</td></tr><tr><td>Correct recall</td><td>6</td><td>8.3</td></tr><tr><td>Honest admission</td><td>2</td><td>2.8</td></tr></table>

Table 17: LC-100% failure modes (GPT-4.1-mini, N = 72). Despite having all prior turns available, 39% of responses are generic and 24% hallucinate recall.
<table><tr><td>Model</td><td>Cond.</td><td>Direct (%)</td><td>Ref. (%)</td><td>NI (%)</td></tr><tr><td rowspan="5">GPT-4.1-mini</td><td>Summary</td><td>44.9</td><td>7.9</td><td>23.6</td></tr><tr><td>LC-10</td><td>53.0</td><td>5.6</td><td>25.0</td></tr><tr><td>LC-50</td><td>72.4</td><td>7.2</td><td>25.0</td></tr><tr><td>LC-100</td><td>78.8</td><td>7.9</td><td>22.2</td></tr><tr><td>RAG-10 RAG-50</td><td>49.2 57.7</td><td>7.0 8.0</td><td>27.8 25.0</td></tr><tr><td rowspan="6">GPT-5.5</td><td>RAG-100 Summary</td><td>62.2 50.0</td><td>7.3 15.4</td><td>26.4 53.4</td></tr><tr><td>LC-10</td><td>56.6</td><td>13.0</td><td>53.4</td></tr><tr><td>LC-50</td><td>77.8</td><td>19.0</td><td>57.5</td></tr><tr><td>LC-100</td><td>82.9</td><td>13.6</td><td>52.1</td></tr><tr><td>RAG-10</td><td>54.4</td><td>13.4</td><td>49.3</td></tr><tr><td>RAG-50 RAG-100</td><td>63.2 64.5</td><td>15.2</td><td>54.8</td></tr><tr><td rowspan="6">Gemini 3.1 Pro</td><td>Summary</td><td>37.9</td><td>12.9</td><td>54.8</td></tr><tr><td>LC-10</td><td>52.3</td><td>8.2 9.2</td><td>32.9 31.5</td></tr><tr><td>LC-50</td><td>67.7</td><td>8.8</td><td>30.1</td></tr><tr><td>LC-100</td><td></td><td></td><td></td></tr><tr><td>RAG-10</td><td>72.3</td><td>7.6</td><td>37.0</td></tr><tr><td>RAG-50</td><td>45.3</td><td>7.7</td><td>31.5</td></tr><tr><td></td><td></td><td>55.1</td><td>10.1</td><td>37.0</td></tr><tr><td></td><td></td><td></td><td></td><td>32.9</td></tr><tr><td>RAG-100</td><td>57.1</td><td></td><td>10.1</td><td></td></tr></table>

Table 18: Cross-model MEMUSE accuracy across all 7 memory conditions, N = 73 instances each. Direct: MEMUSE Direct QA (per-question accuracy). Ref.: MEMUSE Reference (per question, against the natural response). NI: MEMUSE Natural Integration (holistic on the natural response, primary metric). GPT-4.1-mini values are from Table 13; GPT-5.5 (gpt-5.5-2026-04-23) and Gemini 3.1 Pro (gemini-3.1-pro-preview) were re-run for this comparison with judge held fixed at GPT-5.4-nano. Direct QA spans 32.9–34.4 points across capacity for every model; NI spans at most 8.2 points within model and never tracks capacity.

Qdrant store; text-embedding-3-small; GPT-4.1-mini as Mem0’s extraction LLM): prior sessions are ingested incrementally per user; on the trigger turn we retrieve once (top-10 memories) and reuse that memory context, under the verbatim deployed persona prompt, for both the natural reply and every Direct QA answer—the same “identical reconstructed context” protocol as the paper. Letta (letta 0.16.8, self-hosted server; memgpt\_agent agent type with the model pinned to

<table><tr><td>Memory system</td><td>Direct QA</td><td>Ref.</td><td>NI</td></tr><tr><td>Summary/LC/RAG (7 cond., range)</td><td>44.9-78.8</td><td>5.6-8.0</td><td>22.2–27.8</td></tr><tr><td>Mem0</td><td>41.5</td><td>7.3</td><td>58.3</td></tr><tr><td>Letta (MemGPT)</td><td>61.4</td><td>10.8</td><td>56.9</td></tr></table>

Table 19: Modern memory systems on MEMUSE (72 instances, 316 questions; GPT-4.1-mini generator; judge fixed at GPT-5.4-nano; all values %). Both raise Natural Integration (NI) well above the provisioning baselines, yet the Direct QA vs. Reference gap persists (§6.2).

GPT-4.1-mini and text-embedding-3-small): a fresh agent per instance loads the prior sessions into archival memory as passages; tools are restricted to archival\_memory\_search and send\_message so the agent actually consults archival memory; the natural reply is the agent’s response to the trigger, and each Direct QA question is asked as a separate explicit probe. Because Letta generates through its own agent loop (with its own system prompt and tool calls), its generation is not perfectly matched to the deployed pipeline; Mem0 is the cleaner like-for-like comparison. Exact configurations and per-instance outputs are released with the analysis code.

## E.3 Prompt and Architectural Ablations

§6.2 reports that prompt-level and retrieval-side interventions fail to close the retrieval–integration gap. This subsection gives the cross-condition figure (Figure 11), the difficulty continuum across event types, the verbatim prompt texts, percondition tables, and decomposition diagnostics. All variants use the same 73 reactive MemUse instances, the same reconstructed contexts per condition, and the same Natural Integration / Reference judge (GPT-5.4-nano, temp=0).

## E.3.1 Cross-Variant Overview

The 71-point Direct QA / Natural Integration gap conceals a difficulty continuum: Direct QA (78.8%) → user memory probe “Do you remember X?” (51%, n=10) → user re-provision “As I mentioned, $\mathrm { X } ^ { \prime \prime } ( 2 1 \% , n { = } 6 2 ; \mathrm { O R } { = } 4 . 0 6 , p { = } . 0 0 1 $ in a user-random-effect logistic GLM). Probes share the explicit-elicitation form of Direct QA but require integration into a conversational response, locating them midway on the continuum (full perevent-type tables in §E.3.9). Variants V2-CoT, V5, V6, V7 are defined and evaluated in §§E.3.4–E.3.6; the headline span diagnostic (Table 20) shows that across both models, every integration variant spans at most 21.9 pts versus the ∼33-pt Direct QA span—some interventions raise the level of Natural Integration (V5 on GPT-5.5 reaches the high 70s–80s) but the rise is flat across capacity, not capacity-driven. Two additional nuances: V5’s lift is uniform across all 7 conditions including Summary-only where no prior facts are in context (judge fragility, §E.3.8); and V7’s rewritten queries retrieve worse than raw triggers because in a longitudinal personal corpus user phrasings are already lexically aligned with prior turns (§E.3.6). Prompt-level fixes (V2-CoT, failure-mode, fewshot, cue-aware) add ∼+30 pts to probe NI but at most +8 pts to re-provision NI at LC-100%; the V6 two-step shows the inverse pattern. Prompt-level instruction amplifies semi-explicit recall it could already partly do; closing the implicit-integration gap requires architectural intervention.

![](images/e895199fe5b8fae0908b885339992ba818a5548d3cb130403b76be77ae8edb85.jpg)

![](images/144c8e0434babb13ff951125d7195775c860a10725dce10eb84df6fb65dc0b2f.jpg)

![](images/c2ccc4b77560fc8ca534b497f4150946580e88a4b54a1a5da88bc641d76e1fc1.jpg)

![](images/6bdde588d977c793546cc24738a80e816bb639cac9a245f5869c4ea224b4b79a.jpg)  
Figure 11: Prompt interventions do not induce capacity-sensitivity in the integration channel. Top: Natural Integration (%) by memory condition for V0 (deployed) and four interventions, with Direct QA on the same axis as a retrieval baseline. Bottom: span across the 7 conditions (max−min, pts). Direct QA spans 32.9–33.9 pts in both models; every integration variant spans at most 21.9 pts, and the V0 deployed pipeline spans only 6.8–11.0 pts. Variants: V0 deployed pipeline (no extra prompt); V2-CoT chain-of-thought reasoning prepended; V5 one-line cue-aware patch; V6 two-step extract-then-integrate (a separate call extracts named details from context, then the deployed pipeline generates with those details handed in); V7 query rewrite + targeted RAG (the user trigger is rewritten into explicit search queries before retrieval). Per-condition numbers in §E.3.4.

## E.3.2 Deployed Persona (V0 Baseline)

The deployed system prompt (verbatim, used for every condition during the 4-month study and reused as V0 control in every ablation):

You are Luke, afriendly, kind, and understanding diary AI chatbot. Your goal is to make conversationsfeel like chatting with afriend. You are also a reflective companion who helps the user think about their day.

\- Ask questions only a third of the time.

\- Adapt your tone based on the user’s mood.

\- Aimfor one or two sentences, like a real conversation.

\- You are also here to act as a personal diary.

A condition-specific framing is appended (“This is the k-th conversation. Summary of previous conversations: . . . . Generate the response using the following conversation history if necessary.”), followed by the reconstructed prior-turn context per condition (Summary-only, LC-k%, RAG-k%). All ablation variants below preserve this baseline and append additional instructions before the framing.

## E.3.3 Per-Variant Capacity-Span Summary (Both Models)

Table 20 reports the headline diagnostic of the V1–V7 prompt-variant ablation across all 7 memory conditions on both generation models: the Natural-Integration-rate span across the 7 capacity conditions remains well below the ∼33-pt Direct QA span for every variant on either model, confirming that prompt-level interventions do not make the integration channel capacity-sensitive. V0 baselines are canonicalized per (model, condition) cell to a single deployment+judging run (the V5 cross-condition study for GPT-4.1-mini; the V2- CoT@GPT-5.5 followup for GPT-5.5), eliminating the ≤ 2-instance V0 drift that arose from independent per-variant judge re-runs (GPT-5.4-nano at temp=0 is not bit-exact across reruns). Each variant’s ∆ and McNemar p are computed against this canonical V0 paired by instance. Full per-cell results (V0 NI, variant NI, ∆ NI, McNemar p, V0 Reference, variant Reference for all $7 \times 7 \times 2 = 9 8$ cells) are released with the analysis code.

## E.3.4 Single-Call Prompt Variants (V1–V5)

V1–V4 are appended to the deployed persona before the framing, on LC-100% only (the strongest existing condition); V5 is a one-line cue-aware patch evaluated across all 7 conditions. Verbatim text:

V1 (light). When the user references prior conversations or topics, integrate specific named details from your context (proper nouns, places, prior recommendations). Avoid generic empathetic responses. Do not invent details that are not in your context.

V2-CoT. Before responding, follow these steps internally: (1) check if the user is referencing prior conversations; (2) ifso, locate the relevant content in your context; (3) weave specific named details from that content into a natural conversational response; (4) if you cannot find the referenced content, acknowledge that rather than inventing details. Output only the final natural response.

V3 (failure-mode). Threefailure modes to avoid: (a) generic empathetic responses that ignore the user’s reference to prior conversations; (b) inventing details not in context; (c) referring vaguely without naming specifics. When the user mentions prior topics, weave concrete named details from your context into a natural response. V4 (few-shot). Two integration exemplars prepended (Vienna/Figlmüller and Sapporo/Hakodate casesfrom real deployment, also used in Table 4), followed by the instruction to apply the same pattern. The full prompt is included in the released analysis code.

<table><tr><td>Variant</td><td>GPT-4.1-mini</td><td>GPT-5.5</td></tr><tr><td>V0 (deployed)</td><td>6.8</td><td>13.7</td></tr><tr><td>V1</td><td>9.6</td><td>8.2</td></tr><tr><td>V2-CoT</td><td>20.5</td><td>15.1</td></tr><tr><td>V3</td><td>9.6</td><td>13.7</td></tr><tr><td>V4</td><td>19.2</td><td>11.0</td></tr><tr><td>V5</td><td>9.6</td><td>21.9</td></tr><tr><td>V6</td><td>8.2</td><td>16.4</td></tr><tr><td>V7</td><td>8.2</td><td>11.0</td></tr><tr><td>Direct QA capacity span†</td><td>33.9</td><td>32.9</td></tr></table>

Table 20: Natural Integration-span diagnostic (capacity insensitivity). Each cell is max−min Natural Integration (%) across the 7 memory conditions for the indicated variant on the indicated model. <sup>†</sup>Samemodel MEMUSE Direct QA capacity span across the 7 conditions, reported in §6.2: 32.9 / 33.9 / 34.4 pts for GPT-5.5 / GPT-4.1-mini / Gemini 3.1 Pro. Across every variant on either model, NI span remains $\leq 2 2 p t s -$ well below the ∼33-pt Direct QA span—confirming that prompt-level interventions do not make the integration channel capacity-sensitive in the way the retrieval channel is. Full per-cell numbers (V0/variant NI, ∆, McNemar p) are released with the analysis code.

## V5 (cue-aware). If the user refers to something from an earlier conversation, identify the relevant prior topic and respond with one or two specific remembered details. If unsure, say so.

V1–V4 results. On LC-100% with GPT-4.1- mini $( N = 7 3 )$ , Natural Integration: V0 24.7%, V1 28.8%, V2-CoT 34.2%, V3 31.5%, V4 27.4%. McNemar exact two-sided p vs. V0: 0.51, 0.12, 0.23, 0.77 respectively. None significant after Bonferroni at $\alpha / 4 = . 0 1 2 5$ . Reference rates barely move $( + 0 . 6 \mathrm { t o } + 1 . 9 \mathrm { p t s } )$ . The V5 cue-aware variant is broken out across all 7 conditions in Table 21.

## E.3.5 Two-Step Extract-then-Integrate Pipeline (V6)

V6 splits the task into two GPT-4.1-mini calls. Step 1 extracts named facts from the reconstructed context; Step 2 generates the response with those facts piped in as “Notes from your prior conversations.”

Pipeline. Step 1 (extraction): a separate GPT-4.1- mini call (system: “You extract relevant facts from prior conversation context.”) is shown the reconstructed context and the user trigger and asked to “list up to 5 specific named details from the context that relate to what the user is referencing or might want acknowledged” (proper nouns, places, prior recommendations, dates, activities) as a bullet list, or output NO\_RELEVANT\_CONTEXT if nothing relates. Step 2 (generation): the deployed persona prompt (Appendix E.3.2) is augmented with a [Notes from your prior conversations] block containing the Step-1 output (replaced by “(none relevant)” if NO\_RELEVANT\_CONTEXT), followed by the current conversation. Per-condition Natural Integration and Reference for V6 vs. V0 are reported in Table 22.

<table><tr><td>Condition</td><td>V0</td><td>V5</td><td>∆</td><td>McNemar p</td></tr><tr><td>Summary-only</td><td>24.7</td><td>41.1</td><td>+16.4</td><td>.004</td></tr><tr><td>LC-10%</td><td>20.5</td><td>31.5</td><td>+11.0</td><td>.096</td></tr><tr><td>LC-50%</td><td>26.0</td><td>39.7</td><td>+13.7</td><td>.053</td></tr><tr><td>LC-100%</td><td>21.9</td><td>32.9</td><td>+11.0</td><td>.039</td></tr><tr><td>RAG-10%</td><td>24.7</td><td>31.5</td><td>+6.8</td><td>.302</td></tr><tr><td>RAG-50%</td><td>21.9</td><td>35.6</td><td>+13.7</td><td>.031</td></tr><tr><td>RAG-100%</td><td>27.4</td><td>41.1</td><td>+13.7</td><td>.041</td></tr></table>

Table 21: V5 cue-aware prompt vs. V0 deployed persona on Natural Integration % (GPT-4.1-mini, $N = 7 3$ instances per cell). Variant × condition interaction $\chi ^ { 2 } = 2 . 5 , \stackrel { \textstyle } { d f } = 6 , p = . 8 6$ (homogeneity of discordantpair distribution). Reference rates barely move (+0 to +5 pts).

Extraction-quality decomposition (LC-100%). We score each Step-1 extraction against the instance’s ground-truth facts and cross with whether Step-2 referenced those facts in its natural response (Reference = yes/no; Table 23):

The 77% drop rate is the cleanest mechanistic localization we obtain: the model has been prehanded exactly the named details it needs; generation refuses to use them.

## E.3.6 Query-Rewrite + Targeted Retrieval (V7)

V7 tests whether the deployed RAG conditions underperform because raw user-trigger embeddings are bad retrieval queries. We replace the deployed retrieval query with multi-aspect rewrites generated by GPT-4.1-mini.

Pipeline. Step 1 (query rewriting): a separate GPT-4.1-mini call (system: “You generate memoryretrieval queries from a user’s conversational message.”) is shown the user trigger and asked, if it implicitly references prior conversations, to generate 1–3 explicit self-contained search queries suitable for embedding-based retrieval (one per line; output NO\_QUERIES if no reference). Step 2 (retrieval and generation): each rewritten query is embedded with text-embedding-3-small (deployed model) and used to retrieve top-5 from the per-user importance-thresholded prior-turn pool, dedupe by (session, turn) up to 10 unique passages. The retrieved passages are prepended to the persona prompt as a [Notes from your prior conversations, retrieved by your memory system] block before generation. For LCk% cells, V7 also retains the full LC-k% context dump (V7-LC tests rewrite-notes-on-top-of-longcontext); V7-RAG-k% replaces the deployed RAG retrieval query only. Per-condition Natural Integration and retrieval recall are reported in Table 24.

<table><tr><td>Cond.</td><td>V0 NI</td><td>V6NI ∆NI</td><td>V6 Ref.</td><td>McN. p</td></tr><tr><td>Summary-only</td><td>24.7</td><td>24.7</td><td>+0.0 10.0</td><td>1.00</td></tr><tr><td>LC-10%</td><td>20.5</td><td>23.3</td><td>+2.7 6.5</td><td>.77</td></tr><tr><td>LC-50%</td><td>26.0</td><td>28.8</td><td>+2.7 9.0</td><td>.82</td></tr><tr><td>LC-100%</td><td>21.9</td><td>27.4</td><td>+5.5 8.1</td><td>.29</td></tr><tr><td>RAG-10%</td><td>26.0</td><td>26.0</td><td>+0.0 7.5</td><td>1.00</td></tr><tr><td>RAG-50%</td><td>21.9</td><td>31.5</td><td>+9.6 8.7</td><td>.09</td></tr><tr><td>RAG-100%</td><td>26.0</td><td>30.1</td><td>+4.1 8.4</td><td>.63</td></tr></table>

Table 22: V6 two-step pipeline vs. V0 deployed persona (GPT-4.1-mini, $N = 7 3 )$ . Natural Integration and Reference in %. V6 is comparable to or worse than V2-CoT (34.2% on LC-100%; Appendix E.3.4).

<table><tr><td>Step-1 found GT?</td><td>Step-2 referenced? Count</td><td></td></tr><tr><td>Yes</td><td>Yes</td><td>11</td></tr><tr><td>Yes</td><td>No</td><td>37</td></tr><tr><td>No</td><td>Yes</td><td>9</td></tr><tr><td>No</td><td>No</td><td>16</td></tr><tr><td>NO_RELEVANT_CONTEXT</td><td></td><td>2</td></tr></table>

Table 23: V6 extraction-quality decomposition on LC-100% $( N = 7 3 )$ . Step-1 mean GT-coverage = 0.51. When Step-1 successfully named the ground-truth facts, Step-2 still failed to integrate them in $3 7 / 4 8 = 7 7 \%$ of cases.

Why query rewriting hurts here. Standard RAG benchmarks assume a corpus where naive retrieval is bad: short user questions over Wikipediastyle declarative passages, where rewriting (e.g., HyDE) helps because hypothetical answers embed closer to passage content than questions do. In a longitudinal personal corpus the situation reverses. The user’s trigger (“the music Luke recommended the other day”) already contains the lexical anchors of the relevant prior turn (music, recommend). Rewriting into question form (“What music did Luke recommend?”) replaces a conversational match with a Q-form query against an A-form corpus, while spreading the top-10 retrieval budget across 1–3 distinct queries that may individually rank the relevant passage lower than the raw trigger does. The generic-empathetic failure mode persists at ∼48%, identical to V0, regardless of whether retrieval recall improves.

<table><tr><td rowspan="2">Cond.</td><td colspan="2">NI (%)</td><td colspan="2">Recall (%)</td></tr><tr><td>V0</td><td>V7</td><td>V7</td><td>V0</td></tr><tr><td>Summary-only</td><td>24.7</td><td>26.0</td><td>8.5</td><td>9.0</td></tr><tr><td>LC-10%</td><td>26.0</td><td>24.7</td><td>29.9</td><td>32.7</td></tr><tr><td>LC-50%</td><td>26.0</td><td>24.7</td><td>70.1</td><td>75.4</td></tr><tr><td>LC-100%</td><td>23.3</td><td>24.7</td><td>67.3</td><td>78.7</td></tr><tr><td>RAG-10%</td><td>27.4</td><td>31.5</td><td>29.9</td><td>32.7</td></tr><tr><td>RAG-50%</td><td>24.7</td><td>32.9</td><td>70.1</td><td>75.4</td></tr><tr><td>RAG-100%</td><td>26.0</td><td>32.9</td><td>67.3</td><td>78.7</td></tr></table>

Table 24: V7 query-rewrite RAG vs. V0 (GPT-4.1-mini, $N = 7 3 )$ . Natural Integration (NI) and retrieval recall (token-overlap @0.5 between retrieved passages and ground-truth fact spans), both in %. V7 lifts NI modestly on RAG-50/100% (+8.2/ + 6.8 pts; McNemar n.s.) but rewritten queries actually retrieve worse than raw user triggers at high k% (LC/RAG-100% recall 67.3 vs. 78.7).

## E.3.7 GPT-5.5 + V2-CoT

To rule out a model-capacity ceiling for the promptablation null, we re-ran V0 vs. V2-CoT (Appendix E.3.4) on GPT-5.5 across all 7 conditions (Table 25). V0 generations are reused from the cross-model run (Appendix E.1); V2-CoT generations are fresh; all judgments are re-run for applesto-apples comparison.

## E.3.8 Holistic Judge Fragility on Recall-Style Prompts

V5 (cue-aware) and V2-CoT both lift Natural Integration on Summary-only by +16.4 pts even though Summary-only contains no prior-turn facts to integrate, with no corresponding rise in factbased hallucination $( \Delta = + 0$ in failure-mode classification). The lift magnitude is statistically indistinguishable from the lift on context-bearing cells (variant × condition interaction $\textit { p } = \ . 8 6$ for V5, $p = . 9 0$ for V2-CoT on GPT-5.5). The holistic Natural Integration judge—a single binary read of whether the response demonstrates memory of the topic—appears partly responsive to recallstyle framing language (“As you mentioned. . . ”, “I remember you said. . . ”) without verifying substantive integration. This is consistent with the moderate inter-judge agreement reported in Appendix B.4 (Fleiss $\kappa = . 2 6$ on holistic Natural Integration vs. .65 on per-question Reference). The Summary-fragility lift is therefore best read as judge fragility plus a smaller real component; the LC-100%-specific +24.7 pt lift on GPT-5.5 is suggestive of a real capacity-conditional gain on top of fragility, but the asymptotic ceiling at ∼73% remains ∼10 pts below the 83% Direct QA ceiling on the same instances.

<table><tr><td>Cond.</td><td>V0NI V2-CoT NI</td><td>∆</td><td> $\mathbf { M e N } , p$ </td></tr><tr><td>Summary-only</td><td>54.8 71.2</td><td>+16.4</td><td>.023</td></tr><tr><td>LC-10%</td><td>47.9 64.4</td><td>+16.4</td><td>.023</td></tr><tr><td>LC-50%</td><td>61.6 71.2</td><td>+9.6</td><td>.19</td></tr><tr><td>LC-100%</td><td>47.9 72.6</td><td>+24.7</td><td>.0005</td></tr><tr><td>RAG-10%</td><td>50.7 67.1</td><td>+16.4</td><td>.023</td></tr><tr><td>RAG-50%</td><td>61.6 79.5</td><td>+17.8</td><td>.011</td></tr><tr><td>RAG-100%</td><td>58.9 72.6</td><td>+13.7</td><td>.031</td></tr></table>

Table 25: V2-CoT on GPT-5.5 across all 7 conditions $( N = 7 3 )$ . On LC-100%, +24.7 pt lift closes ∼70% of the V0 → Direct QA gap (47.9 → 72.6 vs. 83.0 ceiling), leaving +10.3 pts of headroom. Variant × condition interaction $\chi ^ { 2 } = 2 . 2 ,$ , df = 6, p = .90—lift is uniform across conditions including Summary-only where no facts are in context, replicating the cue-aware Summary-fragility pattern (Appendix E.3.8). Reference remains low (LC-100% 24.6%) confirming the dissociation between holistic Natural Integration and item-level Reference seen on GPT-4.1-mini.

## E.3.9 Event-Type Split (Probe vs. Re-provision)

The 73 reactive instances comprise 11 user memory probes (“Do you remember X?”) and 62 user reprovisions (“As I mentioned, X”). The deployed V0 baseline already shows the predicted continuum: probe Natural Integration ∼50% vs. re-provision Natural Integration ∼20% (logistic GLM with user random effect: $\mathrm { O R } = 4 . 0 6 $ , 95% CI [1.71, 9.63], $p = . 0 0 1 )$ . Table 26 reports the LC-100% breakdown across all prompt variants.

The split clarifies what each intervention family does: prompt-level fixes amplify what the model could already partly do (semi-explicit “Do you remember X?” cues), without repairing the implicitintegration failure mode (“As I mentioned X”). Architectural retrieval-restructuring (V6) helps the harder implicit cases at small cost to the easier ones, but the absolute improvement remains small. Prompt-level instruction amplifies semi-explicit recall; closing the implicit-integration gap requires architectural intervention.

## F Exploratory Engagement and Proactive Analyses

The analyses in this appendix—predictors of satisfaction beyond memory condition, system proactive recall, and user re-provisioning—are exploratory: they involve non-randomized variables (response length, specificity, integration success, recall timing) and should be interpreted as observational associations, not causal claims.

<table><tr><td>Variant (LC-100%)</td><td>Overall</td><td>Probe</td><td>Re-prov</td></tr><tr><td>V0 baseline</td><td>23.6</td><td>40.0</td><td>21.0</td></tr><tr><td>V1 (light)</td><td>27.8</td><td>40.0</td><td>25.8</td></tr><tr><td>V2-CoT</td><td>33.3</td><td>70.0</td><td>27.4</td></tr><tr><td>V3 (failure-mode)</td><td>30.6</td><td>70.0</td><td>24.2</td></tr><tr><td>V4 (few-shot)</td><td>26.4</td><td>70.0</td><td>19.4</td></tr><tr><td>V5 (cue-aware)</td><td>31.9</td><td>70.0</td><td>25.8</td></tr><tr><td>V6 (two-step)</td><td>26.4</td><td>50.0</td><td>22.6</td></tr><tr><td>V7 (query-rewrite RAG)</td><td>23.6</td><td>50.0</td><td>19.4</td></tr></table>

Table 26: Natural Integration % on $L C { - } 1 0 0 \%$ , split by event type $( n _ { \mathrm { p r o b e } } = 1 1 , n _ { \mathrm { r e - p r o v } } = 6 2 ;$ full 73-instance detection set, see Appendix E.1). Reasoning/framing prompts (V2-CoT, V3, V4, V5) lift probe Natural Integration by $\sim + 3 0$ pts but re-provision by $\leq + 8$ pts. The architectural V6 two-step shows the inverse pattern (mean across 7 conditions: −5.7 pts on probes, +4.4 pts on re-provisions). Probe $n = 1 1$ implies ±25–30 pp 95% CIs per single cell; the pattern’s reliability comes from directional consistency across multiple variants and 7 conditions.

## F.1 Engagement Predictors of Satisfaction

The deployment-level null (§4) raises an obvious follow-up: what does predict per-session satisfaction if memory condition does not? Fitting an LMM (Eq. 1) on all 1,270 rating-valid sessions with four z-standardized fixed-effect predictors— response length (LEN), response specificity (SPEC; system proper-noun rate, capturing whether responses reference user-specific personal details), cross-session continuity (CONT; lexical and entity overlap with prior sessions plus explicit user backward references), and memory condition (MEMCOND)—three engagement predictors are independently significant (Figure 12): response length $( \beta = 0 . 1 7 9 , p < . 0 0 1 )$ , cross-session continuity $( \beta ~ = ~ 0 . 1 0 5 , ~ p ~ < ~ . 0 0 1 )$ , and specificity $( \beta = 0 . 0 7 9 , p = . 0 0 1 )$ . Memory condition contributes nothing $( \beta = 0 . 0 1 1 )$

Long-but-generic responses are the failure mode. Cross-session continuity interacts negatively with response length $( \beta = - 0 . 0 6 7 , p = . 0 0 2 )$ : when users reference prior topics, long responses that fail to address those references reduce satisfaction. The negative continuity-×-length effect is offset by specificity (three-way interaction $\beta _ { \mathrm { C o n t \times L e n \times S p e c } }$ = 0.035, p = .084; binary-condition specification $\beta \ : = \ : 0 . 0 7 3 , p \ : = \ : . 0 0 1 )$ . “Longer” helps when paired with specificity; without it, length backfires precisely on the sessions where memory is at stake.

![](images/d4fdc4b84747a48f2dd869876b7f9cf9d51e847c8fd4300cfffd5cb3b2d410ed.jpg)  
Figure 12: Predictors of z-scored satisfaction $( N =$ $^ { 1 , 2 7 0 }$ sessions, 29 users). Bars show 95% Wald CIs around point estimates. Three engagement metrics each independently predict higher satisfaction; memory condition does not.

## F.2 Engagement Dimensions: Full LMM Specification and Operationalization

This subsection supplements Appendix F.1. The full model is $\operatorname { E q . }$ 1 with fixed-effect predictors {LEN, SPEC, CONT, MEMCOND}. Variable definitions: $\mathrm { L E N } _ { i j }$ is mean system response length (words) over the session; $\operatorname { S P E C } _ { i j }$ is specificity, computed as the proper-noun rate in system responses; $\mathrm { C o N T } _ { i j }$ is cross-session continuity, combining lexical overlap with prior sessions, entity overlap, and explicit user backward references (full operationalization in Appendix A.8); $\mathbf { M E M C O N D } _ { i j }$ is the memory condition assigned to session $j$ for user i.

Engagement vs. existing benchmarks (model comparison). We compare two LMMs fitted with maximum likelihood. Model A uses only condition-level QA benchmark accuracy: $y _ { i j } \sim$ $\beta { \bf Q } { \bf A } _ { i j } + ( 1 \ | \ { \bf u } { \bf s e r } _ { i } )$ . Model $\boldsymbol { B }$ uses the three engagement dimensions {LEN, SPEC, CONT} (without memory condition). Model B significantly outperforms Model $\mathcal { A } \left( \Delta \mathrm { A I C } = 1 4 . 1 \right.$ ; likelihood ratio $\chi ^ { 2 } ( 2 ) = 1 8 . 1 4 , p < . 0 0 1 )$ . Adding QA accuracy on top of Model B provides no additional predictive value $( \chi ^ { 2 } ( 1 ) = 1 . 0 2 )$ : QA accuracy captures none of the variance that engagement dimensions explain.

Specificity is not a hidden capacity effect. Specificity does not differ across memory conditions $( F = 0 . 4 3 )$ , so its coefficient is not absorbing variance attributable to capacity. The engagement story is genuinely orthogonal to memory condition.

## F.3 Memory Integration and Engagement: Extended Analyses

Matched-model main effect details. Extending the base result reported in $\ S 6 . 1 \ ( \beta \ = \ + 0 . 5 5 6$ $\textit { p } = \ . 0 8 2 )$ , we examine sensitivity to engagement covariates. Adding response length as a control attenuates the Natural Integration coefficient only slightly $( \beta = + 0 . 5 7 4 , p = . 0 9 1$ , length $\beta = + 0 . 4 2 0 , p = . 0 2 8 )$ , reflecting partial overlap between integration and substantive responding. Adding both response specificity and priorcontext dependence as controls further attenuates the Natural Integration coefficient to $\beta = + 0 . 4 3 0$ $p = . 1 2 1$ , with prior-context dependence emerging as significant $( \beta = + 0 . 2 3 6 , p = . 0 1 7 )$ . The pattern is consistent: integration is positively associated with satisfaction on memory-moment sessions, with the effect partially mediated by—but not eliminated by—engagement covariates. Of the 48 sessions, 11 received Natural Integration = 1 (successful integration) and 37 received Natural Integration = 0; the small successful-integration cell limits power for stricter specifications. Critically, the same exercise on Direct QA accuracy yields a null effect (Spearman $\rho = + 0 . 0 3 3 ; \mathrm { L M M }$ $\beta = - 0 . 0 9 3 )$ , and the head-to-head LMM (§6.1) confirms that Direct QA contributes nothing once Natural Integration is included $( \beta _ { \mathrm { D Q } } = - 0 . 3 5 3 )$

Robustness: leave-one-user-out and influence diagnostics. The matched-model result (§6.1) is not driven by individual users or outlier sessions. Figure 13 shows leave-one-user-out crossvalidation: the interaction coefficient is positive in all 16 iterations and significant $( p < . 0 5 )$ in 15 of 16. Figure 14 plots Cook’s distance on the $N = 5 3$ memory-moment fit (53 of the 55 filtered instances; 2 dropped for missing engagement covariates); four observations exceed the $4 / N$ threshold, and after removing all four the interaction remains significant $( \beta = 0 . 3 5 , p = . 0 1 8 )$

Cross-session continuity × length interaction. A separate length interaction characterizes how user-initiated reference to prior topics moderates the length–satisfaction relationship. On all 1,270 sessions, cross-session continuity interacts negatively with length $( \beta = - 0 . 0 6 7 , p = . 0 0 2 )$ : when users reference prior topics, long responses that fail to address those references are associated with reduced satisfaction. A three-way interaction resolves this apparent contradiction: the negative continuity × length relationship is offset by response specificity (β<sub>continuity×length×specificity</sub> = 0.035, p = .084), confirmed with a stronger effect using an alternative specification with binary memory condition $( \beta = 0 . 0 7 3 , p = . 0 0 1 )$ . Long, specific responses remain beneficial even in high-continuity sessions, while long, generic responses that ignore the user’s conversational history are detrimental.

![](images/46782b6278842de6c187b4eb31add45ec800eabc4ddd0be12095bd842b64eb02.jpg)  
Figure 13: Leave-one-user-out cross-validation of the memory × engagement interaction. Each bar shows the interaction coefficient when one user is withheld. The effect is positive in all 16 iterations and significant $( p < . 0 5 )$ in 15 of 16 (green). The vertical line marks the full-model estimate $( \beta = 0 . 5 7 )$

Capability probe (GPT-5.4 generator). For completeness, we report a parallel analysis on offline GPT-5.4 generations on the same 53 (perinstance) memory-moment events. Note that these outputs were never seen by users, so they cannot be tied directly to user experience and are reported here only as a capability characterization. Under this counterfactual generator, the matched main effect of Natural Integration disappears $( \beta = - 0 . 0 0 2 )$ , but an NI × Length interaction emerges $( \beta = + 0 . 5 7 0 , p = . 0 1 3 ;$ simple slopes: length → rating $\rho = . 4 5 , p = . 0 2 8$ when Natural Integration = 1 vs. $\rho = . 1 5$ when Natural Integration = 0). We interpret this as suggestive evidence that with stronger model capability, memory integration may compound with substantive responding rather than acting as an additive main effect; we do not include this finding in the main text because the underlying responses do not correspond to anything users actually rated.

![](images/a4e4717c8c0c694e92305b836c8b5206d5fe1742a05a7e40e90d34b4c1d11954.jpg)  
Figure 14: Cook’s distance for the memory × engagement model $( N = 5 3 )$ . Four observations exceed the $4 / N$ threshold (dashed line). After removing all four, the interaction remains significant $( \beta = 0 . 3 5 , p = . 0 1 8 )$

## F.4 Prior-Context Dependence (PCD): Detector and Null Result

To complement cross-session continuity (a userside measure), we developed prior-context dependence (PCD), a demand-side judge-scored measure (0–3) of how much a session requires prior conversational context. The detector is a GPT-5.4 judge selected by sweeping prompt structure (three variants) and prior-summary context size $( k \in \{ 1 , 3 , 5 , 1 0 \} )$ against 253 expert-annotated stratified sessions; the selected v2 prompt at $k { = } 5$ achieves binary F1=.835, quadratic-weighted $\kappa =$ .711, and 85.0% binary accuracy. Scaled to the 1,251 filtered sessions, PCD distribution is 64.6%/33.0%/2.3%/0.1% across scores 0–3. In an LMM with response length, specificity, and PCD, the PCD coefficient is null $( \beta = - 0 . 0 1 1 ) \colon$ memory demand per se does not predict satisfaction. An exploratory check on a separately judged integration quality (IQ; 0–3) shows the same pattern as the matched-model NI effect (§6.1): on the 30 sessions with PCD ≥ 2, IQ=2 carries $+ 0 . 2 1 ~ z$ over IQ≤ 1 (Spearman $\rho { = } . 0 9$ , underpowered at $N { = } 3 0 )$ , but no IQ effect emerges among the larger PCD=1 set. Full prompt-sweep numbers and IQ tables are released with the analysis code.

## F.5 Re-Provisioning Burden

We find a significant negative interaction between user word count and cross-session continuity $( \beta =$ $. 0 8 6 , p < . 0 0 1 )$ . In high-continuity sessions— where users reference prior topics—longer user utterances predict lower satisfaction. When users must re-explain previously shared information because the system fails to remember, the effort diminishes satisfaction. This interaction is robust to controlling for number of turns $( \beta = - . 0 7 6$ $\textit { p } = \ . 0 0 2 )$ , user sentiment $( \beta ~ = ~ - . 0 8 8 , ~ p ~ <$ .001), and all controls combined $( \beta ~ = ~ - . 0 7 4 .$ $p \ = \ . 0 0 4 )$ ; it also survives winsorization at the 1st/99th percentiles $( \beta = - . 0 9 2 , p < . 0 0 1 )$ . This is corroborated by a marginal difference between user-initiated and system-initiated memory events $( p = . 0 8 6 )$ : user re-provisions (mean $z = + 0 . 1 7 )$ are associated with higher satisfaction than system proactive recalls (mean $z = - 0 . 1 4 )$

## F.6 Proactive Recalls: Methods and Extended Analyses

This appendix supplements the main-text proactive analysis (§6.3) with verbatim judge prompts, per-condition breakdowns, the aggregate reactive– proactive contrast, robustness checks, and the canonical well-timed / ill-timed examples. The 70 system-initiated proactive recalls are detected by the same event-detection pipeline as the reactive moments (§5); each is scored on grounding and appropriateness using GPT-5.4 at temperature 0.

Aggregate reactive vs. proactive contrast. Figure 15 shows the session-level aggregate underlying the main-text analysis: sessions containing reactive (user-initiated) memory events have higher zscored satisfaction than sessions containing proactive system recalls.

Judge prompts (0/1/2). Both judges are GPT-5.4, temperature 0, with system prompts framing the rater as an expert annotator. Grounding is shown the system’s recall turn, the annotated referenced topic, and the prior-session summaries (newest last), and asked: 0 = not grounded (hallucination, or too vague to be a factual claim); 1 = partially grounded (general topic present but specifics wrong/missing); 2 = fully grounded (specific content clearly present in prior summaries). Appropriateness is shown the current session up to and including the recall turn, and asked: 0 = nonsequitur/forced (derails or ignores what the user just said); 1 = tangential (loosely related but not clearly invited); 2 = natural/invited (fits the flow and is clearly relevant). Both prompts end with “Answer with ONLY a single digit.” Verbatim text is in the released code.

![](images/ea49a66c0e32fe3e3aab5604a6cd25854caed40a1ca416d62738b14da27b1998.jpg)  
Figure 15: Satisfaction by memory event type (session-level aggregate). Reactive events (user-initiated probes and re-provisions) are associated with higher satisfaction than proactive system recalls (z¯ = +0.09 vs. −0.14, p = .086 on sessions; §6.3 refines this by scoring the recalls themselves).

Rating z by score bin (event-level, N=64). Table 27 cross-tabulates mean satisfaction against the grounding and appropriateness scales.

Proactive events by condition. Table 28 confirms that the null holds across every deployed memory condition.

Detection summary. The 70 proactive recalls are predominantly grounded (46 fully, 21 partial, 3 hallucinated), but timing is more variable (32 invited, 29 tangential, 9 mistimed).

Asymmetric-landscape numerics. The three effects shown in Figure 4 (main text): reactive integration success has the largest point estimate (β = +0.56) but its 95% interval (−0.07, +1.19) and the mistimed-recall interval (−1.17, +0.21) both straddle zero, while the re-provisioning burden (β = −0.086, CI [−0.126, −0.046]) is the only effect whose interval excludes zero—the small but reliable penalty users incur when the system fails to remember.

<table><tr><td></td><td colspan="3">Grounding</td><td colspan="3">Appropriateness</td></tr><tr><td>Score</td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>2</td></tr><tr><td>N</td><td>1</td><td>20</td><td>43</td><td>8</td><td>25</td><td>31</td></tr><tr><td>Z</td><td>+0.09</td><td>-0.14</td><td>-0.08</td><td>-0.48</td><td>+0.03</td><td>-0.10</td></tr></table>

Table 27: Mean z-scored satisfaction by Proactive-MEMUSE score bin (64 events, 24 users, 63 unique sessions with TALKING\_INDEX ≤ 6). The only pronounced effect is the low-appropriateness bin; better scores on either axis do not translate into higher satisfaction.
<table><tr><td>Condition</td><td>N</td><td>Ground.</td><td>Approp.</td><td>Z</td></tr><tr><td>Summary-only</td><td>3</td><td>1.67</td><td>1.33</td><td>+0.14</td></tr><tr><td>LC-10%</td><td>21</td><td>1.48</td><td>1.14</td><td>-0.31</td></tr><tr><td>LC-50%</td><td>17</td><td>1.76</td><td>1.24</td><td>+0.12</td></tr><tr><td>LC-100%</td><td>9</td><td>1.78</td><td>1.33</td><td>-0.18</td></tr><tr><td>RAG-10%</td><td>4</td><td>1.75</td><td>2.00</td><td>+0.20</td></tr><tr><td>RAG-50%</td><td>6</td><td>1.67</td><td>1.67</td><td>-0.02</td></tr><tr><td>RAG-100%</td><td>4</td><td>1.75</td><td>2.00</td><td>-0.28</td></tr></table>

Table 28: Proactive-MEMUSE scores and mean zscored satisfaction by deployed memory condition (event-level, N=64). Both axes are stable across conditions; RAG conditions achieve uniformly high appropriateness (mean ≈ 2.0 on RAG-10/100) without a corresponding satisfaction gain.

Robustness of the proactive null. The reactive– proactive gap holds apples-to-apples with identical judge and prompt: scoring the 64 proactive recalls with the same GPT-5.4 Natural Integration metric used on reactive memory moments yields $\rho = - 0 . 2 0 \ : ( p = . 1 2 ; 6 2$ of 64 judged yes), against the reactive baseline $\rho = + 0 . 2 9 \ : ( p = . 0 4 6 ) \cdot$ —the asymmetry is not an artifact of the 0/1/2 Proactive-MEMUSE scales. A second judge (Gemini 3.1 Pro) replicates the null on both axes $( \rho _ { \mathrm { g r o u n d } } = - 0 . 0 2 ;$ $\rho _ { \mathrm { a p p r o p } } = + 0 . 0 8 )$ , and the only directional signal is again at appropriateness=0 $( \bar { z } = - 1 . 7 0 , N { = } 3 )$ Inter-judge agreement is substantial on grounding (weighted $\kappa = 0 . 7 3 )$ and fair on appropriateness $( \kappa = 0 . 2 8 )$ , as expected for a more subjective judgment; the rating null is robust across both. TOST equivalence at $\delta = 0 . 2 5$ is underpowered at $N { = } 6 4$ (all $p > . 0 9 )$ ; we therefore read this as a directional null with a robust apples-to-apples asymmetry, not a formal equivalence claim. On the joint LMM, neither grounding nor appropriateness survives conditioning on the other $( \beta _ { \mathrm { g r o u n d } } = + 0 . 2 9 , p = . 1 2 ;$ $\beta _ { \mathrm { a p p r o p } } = + 0 . 1 6 )$

Topical-continuation coding. To check whether users actually engage with the recalled topic, we code each recall’s immediate next user turn as topically engaging if it shares any content word with the referenced topic. From the topic we extract alphabetic tokens of length > 4 with a small stopword list (system/role and meta terms) removed; from the user turn we extract lowercased alphabetic tokens of length ≥ 3. The two sets overlap in 19 of 62 recalls (31%), and the binary topicalcontinuation indicator is uncorrelated with appropriateness $( \rho ~ = ~ - 0 . 0 2 )$ : users walk past welltimed callbacks at the same rate as poorly-timed ones.

What well-timed and ill-timed recalls look like. A well-timed (appropriateness=2) recall fills a slot the user has just opened:

User: “There’s a movie I’ve been looking forward to, and it’s finally coming out, so I want to go see it.”

System: “Is it the one you mentioned before, like the new Predator, or something different this time?”

A mistimed recall (appropriateness=0) is factually grounded but disconnected from what the user just said:

User: “Thank you for saying that! See you next time! Good night.”

System: “Even though you’re feeling exhausted from caring for your son’s illness, your approach to taking small steps and focusing on self-care, like eating more vegetables, is really inspiring. . .

Both reference real prior-session content; what differs is whether the user’s immediately preceding turn invited the callback.

## G Cost and Latency

Per-condition token usage and latency are summarized in main-text Table 2. This appendix provides the methodology and the full latencyvs-satisfaction breakdown supporting the latencyconfound check in §4.

Compute provenance. All experiments use commercial inference APIs (OpenAI gpt-4.1-mini, gpt-5.4-nano, and gpt-5.5); no local GPU training or fine-tuning was performed, and exact parameter counts for these closed-weight models are not publicly disclosed. Token-throughput cost (Table 2) and wall-clock latency therefore serve as the computational-budget summary in lieu of GPUhour figures.

Methodology. Cost numbers in Table 2 are computed using the OpenAI gpt-4.1-mini inputtoken rate over reconstructed per-session contexts; latency is wall-clock total per session in the deployed system. Token growth from month 1 to month 4 reflects the linear context-accumulation term that drives long-context cost: LC-100% accumulates roughly 5.8× more input tokens by month 4, while the importance-filtered RAG conditions stay flat or near-flat. Cost-per-memorymoment is computed as session cost divided by the observed 4% moment rate, treating non-memorymoment sessions as cost without memory benefit.

Latency does not confound the capacity null. One alternative explanation for the satisfaction null is that low-latency conditions (Summary, RAGk%) might be rated higher because of their shorter response times, masking a true capacity effect. The data do not support this. On the same 29- user / 1,270-session filtered subset used for satisfaction analysis, we computed per-session mean, maximum, and minimum total response latency and time-to-first-token (TTFT) across system turns and correlated each with within-user z-scored rating. Overall (Spearman): mean total latency $\rho = - 0 . 0 3 5 \ : ( p = . 2 2 )$ , max total latency $\rho =$ $- 0 . 0 1 2 \left( p = . 6 7 \right)$ , min total latency $\rho = - 0 . 0 5 1$ (p = .07); mean TTFT $\rho = - 0 . 0 2 6 \ ( p = . 3 6 )$ max TTFT ρ = −0.001 $( p ~ = ~ . 9 6 )$ , min TTFT $\rho ~ = ~ - 0 . 0 5 1 ~ ( p ~ = ~ . 0 7 )$ Within every condition: $| \rho | \ \leq \ 0 . 0 8 .$ , all $\textit { p } > \ . 3 3$ (Summaryonly − $- 0 . 0 4 ; \mathrm { L C - 1 0 \% - 0 . 0 3 ; L C - 5 0 \% - 0 . 0 3 ; L C - }$ $1 0 0 \% - 0 . 0 6 ; \mathrm { { R A G - 1 0 \% + 0 . 0 3 ; R A G - 5 0 \% - 0 . 0 8 ; } }$ $\mathrm { R A G - } 1 0 0 \% ~ + 0 . 0 5 )$ . Notably, per-session maximum latency—which captures the worst-case turn (single-turn maximum reaches 34.3 s in our deployment, $p _ { 9 5 } = 6 . 5 \mathrm { s } )$ —is the least predictive of all six features: a “one slow turn anchors the impression” confound would predict max-turn latency to track satisfaction most strongly, whereas observed $\rho \ = \ - 0 . 0 1 2 .$ The capacity null is not latencymediated at the scale spanned by our deployment (∼1.3 s of mean-latency variance from Summary to LC-100%, with single-turn maxes reaching tens of seconds).

## H Data Release Notes

We release (https://huggingface.co/ datasets/RuiSumida/memuse; code at https: //github.com/ryuichi-sumida/memuse): (i) the full deployment corpus (1,872 sessions across 40 users, 7 memory conditions, with per-session satisfaction ratings); (ii) the MEMUSE benchmark (72 de-identified instances with ground-truth facts and 316 questions, plus Natural Integration judgments and scoring prompts); and (iii) all evaluation and analysis code. Consent procedures, redaction process, and privacy considerations are described in the Ethics Statement (main text).

License and intended use. The deployment corpus and MEMUSE benchmark are released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license; the evaluation and analysis code are released under the MIT license. Both data artifacts are intended exclusively for non-commercial research use, consistent with the participant consent procedures (Ethics Statement); commercial redistribution of the conversational content is prohibited. Existing benchmarks compared against in this work (MemoryBank, LoCoMo, Long-MemEval, LUFY, RealTalk) are used under their respective public-research licenses as released by their original authors. Commercial inference APIs (OpenAI’s GPT family) are used in accordance with OpenAI’s research-use terms of service.