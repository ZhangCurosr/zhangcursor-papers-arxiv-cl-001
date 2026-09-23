# Knowledge Pull Requests for Continual Document Authoring

Alexander Martin Benjamin Van Durme Johns Hopkins University {amart233, vandurme}@jhu.edu

## Abstract

We introduce Knowledge Pull Requests (KPRs), a framework for continual document authoring that makes each change interpretable. Documents require ongoing revision as new knowledge surfaces from other sources, languages, or times, but existing approaches either edit with no account of what knowledge changed or regenerate from scratch. A KPR integrates new knowledge into a document by extracting claims, filtering and routing them to sections, and flagging conflicts with existing content, producing a ChangeLog that separates what knowledge changes (claim proposal) from how the text changes (document diff). We evaluate KPRs on revising Wikipedia across languages and updating query-driven reports on RAGTIME. KPRs integrate more information and better preserve existing content than rewriting from sources or regenerating from scratch, while adding the most information per token generated. A KPR-revised article also grounds question answering better than a frontier model with search, which does not surface knowledge documented only in other languages.<sup>1</sup>

## 1 Introduction

Many important documents maintained by humans and systems are never truly finished. Wikipedia articles, technical documentation, intelligence reports, and more require ongoing maintenance as knowledge surfaces from new sources, other languages, or later time periods. To incorporate these changes, one may prompt a language model with an existing document alongside new sources (Logan IV et al., 2022; Gangi Reddy et al., 2026), but this gives no account of what facts are added or overwritten. Query-driven systems do not attempt updating, instead generating documents from scratch for each query (Shao et al., 2024; OpenAI, 2025; Google,

![](images/7a28e75ef9d88431c00930e6df685acf99ab4f49c95bdcf3cab0fffe52162496.jpg)  
Figure 1: Knowledge Pull Requests update a main document with information from sources as a reviewable ChangeLog: a Claim Proposal (the absent or conflicting source claims) and a Document Diff (the text changes).

2025). If the query is asked again or new sources surface, the entire generation pipeline reruns.

Even Wikipedia, the canonical continually authored knowledge base, tracks edits only as raw text diffs, recording which words changed and not what knowledge did. With millions of edits per month and a finite pool of volunteers, reviewers struggle to assess whether an edit introduces new facts, corrects outdated ones, or conflicts with existing content (Franzmeyer et al., 2024).

Reducing this burden requires moving from textlevel diffs to knowledge-level proposals — an interpretable account of what claims are being added and how they relate to existing content. Software engineering solved an analogous problem with the Pull Request, a proposal of changes to a code base that can be inspected, discussed, and merged. We introduce Knowledge Pull Requests (KPRs), the equivalent for documents (Figure 1). A KPR extracts claims from a set of sources, filters and routes them into a main document, and flags conflicts, producing a ChangeLog: a reviewable artifact separating what knowledge changes (claim proposal) from how the text changes (document diff).

We study continual document authoring in two settings: revision, adding information about an unchanged world, and update, synchronizing a document with a changed world (Katsuno and Mendelzon, 1991). For revision, we integrate knowledge into an English Wikipedia article from its counterparts in other languages, where coverage is often uneven, so facts documented in one language may be absent from the English article and hard to reach through English search. For update, query-driven reports must be kept current as new sources extend, supersede, or contradict them. We study this with three constructed variants of RAGTIME (Lawrie et al., 2026): temporal, conflict, and balanced.

We compare KPRs against three baselines. The first rewrites the document conditioned on the new sources, but never makes explicit what an edit adds or overwrites (Gangi Reddy et al., 2026). The second conditions on claims extracted from those sources, but without a claim proposal to filter them. The third regenerates the document from scratch over all sources, discarding the existing one. We evaluate these methods intrinsically, by the faithfulness and completeness of the rewrite, extrinsically, by its usefulness as a grounding source for downstream question answering, and by the cost of adding and reviewing its changes.

Across both settings, KPRs outperform these baselines, integrating more of the new information and better preserving the existing document. When revising Wikipedia, the revised article is a stronger grounding source than the original, any baseline, or a frontier model with web search, which does not surface this knowledge. When updating RAG-TIME, KPRs flag conflicts between sources rather than silently resolving them, holding precision where conditioning on raw text is misled. Across tasks, we find that conditioning on claims rather than raw source text yields more complete documents. Additionally, filtering those claims before the rewrite raises both precision and recall, while concentrating the changes into contiguous edits a reviewer can approve.

We contribute: (1) Knowledge Pull Requests, which add knowledge to a document as a reviewable ChangeLog, and (2) evidence that KPRs outperform existing document-updating approaches.

## 2 Related Work

Belief Revision and Epistemology of Documents. Formal epistemology studies how to incorporate new information into an existing body of belief (Gärdenfors, 1988; Fermé and Hansson, 2011). The AGM framework (Alchourrón et al., 1985) operates over belief sets closed under logical consequence, but requiring the belief state to contain every sentence its members entail is poorly suited to a document. This is refined by belief bases (Hansson, 1992, 1999), which model finite sets of explicitly held sentences and move closer to how belief is expressed in language. Belief revision further distinguishes revision, learning more about a static world, from update, where the world itself changes (Katsuno and Mendelzon, 1991). These frameworks, however, model belief change over sentences and their logical relations alone, without reference to the justifications behind them, operating at too high a level of abstraction (Pollock, 1987; Pollock and Gillies, 2000). Our KPRs take this a step further, operating over natural-language claims rather than formal sentences and producing a ChangeLog that separates the knowledge-level claim proposal from the text-level diff. Our experiments cover both settings: revising English Wikipedia from other-language editions (revision) and updating reports as sources evolve (update).

Knowledge Cutoffs and Editing. A language model’s knowledge is fixed at a training cutoff and grows stale as the world changes (Jang et al., 2022; Cheng et al., 2024; Vu et al., 2024). A common approach updates the model’s parameters: either locating and overwriting factual associations in the weights (Meng et al., 2022, 2023) or isolating knowledge in modular adapters that can be added, removed, or swapped (Pfeiffer et al., 2021; Fleshman et al., 2025; Fleshman and Durme, 2025). A complementary line instead updates the source text models rely on. Here Wikipedia is a natural target: it is a standard ingredient of LLM pretraining corpora (Soldaini et al., 2024; Groeneveld et al., 2024) and among the most frequently cited domains grounding the answers of LLMs and AI search (Semrush, 2025; theStacc, 2026; Wikimedia Foundation, 2023, 2025). Logan IV et al. (2022) and Gangi Reddy et al. (2026) rewrite Wikipedia articles to reflect new sources, and Spangher et al. (2022) model the revision histories of news articles. However, editing weights and text alike treat updating as end-to-end rewriting, yielding a revised artifact without surfacing which claims changed or how they conflict with existing content, and offering a reviewer no interpretable record of the update.

Claims and Factuality. Claims—atomic, independently verifiable propositions—have become a standard unit for reasoning about factual content, for two reasons. First, claims are more interpretable than sentences. Building on the Pyramid method (Nenkova and Passonneau, 2004), which scores content by atomic facts (Summary Content Units) rather than whole sentences, FActScore (Min et al., 2023) and VeriScore (Song et al., 2024) decompose model outputs into subclaims and verify each against evidence, showing that atomic claims support more reliable and interpretable factuality judgments than sentence- or passage-level assessment. Second, claims are more universal and independent of the form of their context than task-specific alternatives (e.g., nuggets; Voorhees, 2004), transferring even to other modalities (Jing et al., 2024; Martin et al., 2026). These two properties motivate operating over claims rather than whole documents. Claims expose exactly which fact is being asserted, distinguishing our approach from methods that update articles conditioned on raw text (Gangi Reddy et al., 2026). Reducing sources to claims first makes explicit what knowledge each change contributes.

## 3 Knowledge Pull Requests

A Knowledge Pull Request (KPR) is a structured process for integrating new knowledge from a set of source documents into a main document. We define the core components below.

Main Document and Sources. A KPR operates over two inputs: a main document and a set of sources. The main document (hereafter, main) is a previously written document on a given topic. The sources are documents containing potentially new knowledge relevant to the main’s topic. The goal of a KPR is to integrate information from the sources into the main, working with its existing content rather than overwriting it.

Claims. KPRs operate at the level of claims: atomic, decontextualized factual statements (see Gunjal and Durrett, 2024) extracted from the sources and the main. Operating at the claim level, rather than the passage level, enables precise tracking of what knowledge is being proposed, where in the main it should be placed, and whether it conflicts with existing content.

ChangeLog. The key artifact produced by a KPR is a ChangeLog: a structured, reviewable record of the proposed changes to the main. A ChangeLog consists of two components:

![](images/e0efe01ab84cf6bea075966d73a4e85b04080249670354f49c2b67108c84cbe3.jpg)  
Figure 2: The main and sources are decomposed into atomic, decontextualized claims. Non-English sources are decomposed directly into English, and the main’s claims are cached offline.

Claim Proposal. A mapping of new claims extracted from the sources to the sections of the main document<sup>2</sup> where they are proposed to be added, including new ones where needed. A candidate claim is mapped if it is not filtered by coverage (main already contains it) or against the document’s authoring criterion (query relevance or authoring guidelines<sup>3</sup>). The claim proposal is also where the KPR flags knowledge conflicts (merge conflicts) (intercontext conflicts; Xu et al., 2024): cases where a proposed claim contradicts an existing claim in main. Rather than silently resolving these conflicts, the KPR surfaces them for review (Thorne et al., 2018). A human reviewer can adjudicate flagged conflicts, approve or reject claims, and reverse filtering decisions made in the proposal.

Document diff. A record of the proposed textual changes to the main, showing how the document would read after the proposed claims are integrated. Together, the claim proposal and diff let a reviewer inspect both what knowledge is being added (the claim proposal) and how the document text changes as a result (the diff).

The ChangeLog is what distinguishes a KPR from a simple rewrite. By separating the knowledge-level proposal from the text-level changes, it enables collaborative continual authoring. A human reviewer can assess proposed claims on their merits, resolve conflicts, and approve or reject changes before they are merged into main, analogous to a code review in software engineering.

![](images/34182003ccd3cced549ebcdc96ac0e9fa050d055b991632275f1450e2fe90a03.jpg)  
Figure 3: Source claims are classified against the main’s claims as absent, conflicting, or supported. Absent claims are filtered for relevance and routed to a section, conflicting claims are flagged for review, and supported and irrelevant claims are dropped.

## 4 Method for KPRs

We introduce a three-stage baseline for producing KPRs. The first stage decomposes the sources into claims. The second produces a claim proposal by filtering claims already covered by or irrelevant to the main, flagging knowledge conflicts, and routing the remainder to sections. Finally, the third rewrites the new and affected sections to produce the diff.

Claim Decomposition (Figure 2). Using an LLM, we decompose the sources into sets of atomic, decontextualized claims, giving a set of candidate claims to be potentially added to the main. The main is decomposed with the same method, but its claims are decomposed offline and cached as an index rather than online as with the sources. For non-English sources, we decompose directly into English rather than translating first, which yields more faithful claims (Appendix C).

Claim Proposal (Figure 3). For each candidate claim, the KPR makes four decisions: (1) coverage, whether the claim is already covered by the main’s claims; (2) conflict, whether it contradicts an existing claim in the main; (3) relevance, whether the claim meets the document’s authoring criterion; and (4) routing, which section a new, nonconflicting, relevant claim belongs in. Coverage and conflict are resolved in a single classification pass labeling source claims as covered, conflicting, or absent. Relevance is then applied to the absent claims, against the information request in the RAGTIME setting, and left unfiltered in the Wikipedia setting, where candidates already come from articles authored with the same guidelines. Routing maps absent, relevant claims to an existing section or proposes a new one. Covered and irrelevant claims are dropped and conflicting claims are flagged for review.

![](images/c4636a1e10082548e3036c80d99701f6f4b6e72978cf8029c819b215c19f2708.jpg)  
Figure 4: The approved claim proposal is integrated into each new and affected section, and diffing the result against the original gives the document diff. Sections with no proposed claims are left unchanged.

Document diff (Figure 4). We then apply the claim proposal to the main, one section at a time. Each section (new or existing) that receives one or more claims is rewritten to integrate them, while sections with no proposed claims are left unchanged. Diffing the resulting document against the original main gives the document diff.

## 4.1 Baselines

We compare KPRs against three methods that integrate the sources without a claim proposal.

ConText. ConText (Concatenate Text) is our adaptation of WiNELL (Gangi Reddy et al., 2026). It conditions each section’s rewrite on raw source text and performs no claim decomposition. In place of WiNELL’s retrieval step, an LLM classifies whether a source contains information relevant to that section, also to mirror our claim routing.

ConClaim. ConClaim (Concatenate Claims) differs from ConText only in conditioning on source claims rather than source text. Exactly as in KPR, sources are decomposed into claims and those claims are routed to sections. The only difference from KPR is that ConClaim applies no coverage, conflict, or relevance filtering, conditioning each section’s rewrite on all claims routed to it. Con-Claim is therefore equivalent to a KPR without claim review.

Scratch. Scratch regenerates the document from all sources in one pass, discarding the standing document entirely. This follows how a query-driven or deep-research system answers an updated query. We use it in the RAGTIME setting only, where regeneration is the standard alternative to updating when new sources surface.

All baselines produce their document diff the same way as KPR, by diffing the rewritten document against the original main.

## 4.2 Implementation Details

All uses of an LLM—classification, routing, and rewriting—use Qwen3.5-27B (Team, 2026). Because ConText, ConClaim, and KPRs operate over document sections, the two evaluation settings differ in how sections are obtained. For Wikipedia, articles are already organized into sections, so all methods operate on the native section structure. For RAGTIME, system-generated reports do not generally have a section structure, but for our experiments we impose one on the round-1 reports so that every method can operate section by section. A KPR requires only a span the document can be rewritten in, not a pre-existing header, so imposing an outline is sufficient.

Our experiments have no human reviewer, so any claims flagged as conflicts are withheld from the rewrite rather than resolved. Resolving a conflict means deciding which source to believe, and adjudicating source trust remains an open problem, so we withhold conflicts as a default.

## 5 Revising Cross-lingual Knowledge Across Wikipedia with KPRs

Our first application, cross-lingual knowledge revision, integrates knowledge from one language edition of Wikipedia into another. Events, entities, and locations are often documented unevenly across editions, covered more thoroughly in the language of the region they concern and sparsely elsewhere, depending on the distribution of volunteer editors. Each edition is therefore a source of human-authored knowledge that may be missing from or in conflict with another.

Data and Task. We treat English Wikipedia as the main and its counterparts in other languages as sources. Our task is thus to revise the English article with knowledge from the other sources while preserving its existing content. This tests whether a KPR can work from existing, curated text rather than newly surfaced information. Our documents come from MegaWika 2.0 (Barham et al., 2025), whose collections (Barham et al., 2023, 2025) are built specifically for broad multilingual Wikipedia coverage with aligned articles across languages.

Evaluation. We evaluate these rewrites along three dimensions: the quality of the rewritten article, the edit cost of accepting it, and its value as a grounding source for downstream question answering. Appendix D gives more evaluation details.

Quality. We measure quality with Mi-RAGE (Martin et al., 2026). Information precision (InfoP) is a source-constrained variant of FActScore: each claim in the rewrite is verified against the documents used to produce it, checking that added claims faithfully reflect their sources rather than being distorted during rewriting. We adapt information recall (InfoR) to measure the coverage of two distinct claim sets: (1) InfoR-R (retain), the original English claims preserved in the rewrite, and (2) InfoR-A (add), source claims that are added in the rewrite. These capture whether the rewrite preserves existing content and adds the intended new content, respectively.

Grounding Source. We test how well each rewritten article serves as a grounding source for question answering. To build the QA set, we take the decomposed claims from the English and multilingual articles, generate QA pairs from them, and filter for quality (answerable, not context-dependent, wellanswered), yielding English and multilingual splits. With the two splits, we measure whether the rewrite loses existing knowledge (En-QA) and whether it adds the new cross-lingual knowledge (Multi-QA).

Edit Cost. We measure edit cost from the perspective of a reviewer who must review and approve a method’s changes to a document. From a word-level diff between the main M and the rewrite R, we report five quantities: word edit rate (WER), the word-level Levenshtein distance per source word; Click, the number of contiguous edit blocks the reviewer must approve, regardless of their size; added tokens (Tok), the number of new tokens generated; preservation (Presv), the fraction of the main untouched; and expansion (Add), the length ratio <sup>R</sup> . WER, Click, and Tok measure the cost of producing and reviewing a rewrite, while Presv and Add describe the shape of the rewrite.

<table><tr><td rowspan=1 colspan=1>Language   Model</td><td rowspan=1 colspan=1>CB  EW</td><td rowspan=1 colspan=1>ConText  ConClaim KPR</td></tr><tr><td rowspan=1 colspan=1>Q3.5-27B</td><td rowspan=1 colspan=1>33.6 95.7</td><td rowspan=1 colspan=1>91.5        91.5      90.9</td></tr><tr><td rowspan=1 colspan=1>Q3-30B</td><td rowspan=1 colspan=1>27.7 92.3</td><td rowspan=1 colspan=1>88.3        88.7      88.1</td></tr><tr><td rowspan=1 colspan=1>G4-31B</td><td rowspan=1 colspan=1>33.3 95.1</td><td rowspan=1 colspan=1>91.1        91.0      90.2</td></tr><tr><td rowspan=4 colspan=1>En-QA      L3.3-70BL4-ScoutM-8x7BN3-120B</td><td rowspan=1 colspan=1>45.4 95.1</td><td rowspan=1 colspan=1>91.9        92.1      91.6</td></tr><tr><td rowspan=1 colspan=1>32.6 77.2</td><td rowspan=1 colspan=1>67.3        70.1      71.0</td></tr><tr><td rowspan=1 colspan=1>36.1 93.3</td><td rowspan=1 colspan=1>90.4        90.2      89.7</td></tr><tr><td rowspan=1 colspan=1>42.1 95.3</td><td rowspan=1 colspan=1>92.5        92.6      92.2</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>35.8 92.0</td><td rowspan=1 colspan=1>87.6        88.0      87.7</td></tr><tr><td rowspan=1 colspan=1>Q3.5-27B</td><td rowspan=1 colspan=1>36.3 15.5</td><td rowspan=1 colspan=1>41.7        54.6      67.8</td></tr><tr><td rowspan=1 colspan=1>Q3-30B</td><td rowspan=1 colspan=1>31.0 29.4</td><td rowspan=1 colspan=1>49.1        59.9      70.7</td></tr><tr><td rowspan=1 colspan=1>G4-31B</td><td rowspan=1 colspan=1>35.9 14.6</td><td rowspan=1 colspan=1>39.5        52.7      66.3</td></tr><tr><td rowspan=3 colspan=1>Multi-QA  L3.3-70BL4-ScoutM-8x7B</td><td rowspan=1 colspan=1>43.5 31.9</td><td rowspan=1 colspan=1>52.9        62.8      74.0</td></tr><tr><td rowspan=1 colspan=1>33.9 34.5</td><td rowspan=1 colspan=1>42.3        48.9      54.8</td></tr><tr><td rowspan=1 colspan=1>36.4 39.0</td><td rowspan=1 colspan=1>55.7        64.2      74.5</td></tr><tr><td rowspan=1 colspan=1>N3-120B</td><td rowspan=1 colspan=1>41.6 41.3</td><td rowspan=1 colspan=1>57.1        66.1      76.6</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>36.9 29.5</td><td rowspan=1 colspan=1>48.4        58.5      69.2</td></tr></table>

Table 1: QA accuracy, conditioned on the original English article (EW), each rewrite (ConText, ConClaim, KPR), or no document (CB). Q: Qwen, G: Gemma, L: Llama, M: Mixtral, N: Nemotron.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>InfoPInfoR-R InfoR-A</td></tr><tr><td rowspan=1 colspan=1>ConText</td><td rowspan=1 colspan=1>0.837  0.946    0.716</td></tr><tr><td rowspan=1 colspan=1>ConClaim</td><td rowspan=1 colspan=1>0.853  0.943    0.786</td></tr><tr><td rowspan=1 colspan=1>KPR</td><td rowspan=1 colspan=1>0.878  0.950    0.891</td></tr></table>

Table 2: Article-quality evaluation of the Wikipedia rewrites, measured with MiRAGE. InfoP: information precision; InfoR-R: Information Recall Retained from the English article; InfoR-A: Information Recall Added by the sources.

We evaluate the QA accuracy of Qwen3.5- 9B/27B (Q3.5-XB; Team, 2026), Qwen3- 8B/30B (Q3-XB; Yang et al., 2025), Gemma-4- 31B (G4-31B; Team et al., 2026), Llama-3.3-70B, Llama-3.1-8B, and Llama-4-Scout (L3.X-XB; L4-Scout; Grattafiori et al., 2024; Meta AI, 2025), Mixtral-8x7B (M-8x7B; Jiang et al., 2024), OLMo-3-7B (OLMo-3-7B; Olmo et al., 2026), Nemotron 3 (N3-120B; NVIDIA et al., 2025)

<table><tr><td>Model</td><td>CB</td><td>EW</td><td>KPR</td></tr><tr><td>Qwen3.5-9B</td><td>37.4</td><td>15.5</td><td>44.8</td></tr><tr><td>Qwen3-8B</td><td>33.5</td><td>35.8</td><td>61.4</td></tr><tr><td>Llama-3.1-8B</td><td>38.9</td><td>33.8</td><td>61.5</td></tr><tr><td>OLMo-3-7B</td><td>25.5</td><td>25.1</td><td>51.1</td></tr></table>

Table 3: Smaller models on multilingual QA.  
and GPT Sol 5.6 (Sol-5.6; OpenAI, 2026) when conditioning on each rewritten article.

## 5.1 Article Quality

Table 2 reports the InfoP and the two InfoR variants. We find that conditioning on claims rather than raw text (ConText vs. ConClaim) raises both the precision and the recall of added information. When adding the claim proposal (ConClaim vs. KPR), both precision and recall rise again, with the larger gain in added information. Retention is high for every method, so the methods differ mainly in how much of the source knowledge they integrate, where KPRs see the largest benefit.

## 5.2 Grounding Source

Table 1 reports each model’s accuracy under five conditions: no document (closed-book), the original main (EW), and the main after each rewrite (ConText, ConClaim, and KPR).

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=2>CBCB-TEW</td><td rowspan=1 colspan=1>KPR</td><td rowspan=1 colspan=1>WSWS-T</td></tr><tr><td rowspan=1 colspan=1>Q3.5-9B</td><td rowspan=1 colspan=1>2    3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>49</td><td rowspan=4 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Q3-8B</td><td rowspan=1 colspan=1>4    4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>56</td></tr><tr><td rowspan=2 colspan=1>L3.1-8BO3-7B</td><td rowspan=1 colspan=1>2    1</td><td rowspan=1 colspan=1>7</td><td rowspan=2 colspan=1>5650</td></tr><tr><td rowspan=1 colspan=1>2    2</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>Q3.5-27B</td><td rowspan=1 colspan=1>0    5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>52</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Q3-30B</td><td rowspan=1 colspan=1>0    3</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>54</td></tr><tr><td rowspan=1 colspan=1>G4-31BL3.3-70B</td><td rowspan=1 colspan=1>0    40    5</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>5451</td></tr><tr><td rowspan=1 colspan=1>Sol-5.6</td><td rowspan=1 colspan=1>13   20</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>38   41</td></tr></table>

Table 4: Model performance on the hardest QA set: 100 multilingual questions no dense open model answers closed-book. CB: closed-book; CB-T: closed book translated; EW: original English article; KPR: KPRrevised article; WS: web search; WS-T: web search translated.

Multilingual QA. The unchanged article is a poor grounding source, averaging below even closed-book, because the queried facts are absent from the English article, and some models abstain rather than guess.<sup>4</sup> All three rewrites recover some of this missing knowledge, but the KPR method recovers substantially more than either baseline. This shows the downstream impact of KPR’s high InfoR (add), demonstrating KPRs incorporate multilingual information in a way that grounding articles can properly utilize.

English QA. On questions about content already in the English article, the unchanged article scores highest (EW). All three rewrites degrade QA performance slightly, but stay within half a percent of one another, so integrating the cross-lingual knowledge does not meaningfully cost existing content for any method. KPR therefore obtains its multilingual gains without giving up English performance relative to the baselines.

Small Models. The KPR advantage does not require a large model to exploit the rewritten article. On smaller models (Table 3), we still find that conditioning on the KPR-revised article provides a strong lift in performance. Small models running locally are exactly where a personal, grounded wiki is most useful.<sup>5</sup> A KPR gives such a setup a document worth grounding on, without a frontier model or a search service in the loop.

<table><tr><td>Method</td><td>WER</td><td>Click</td><td>Tok</td><td>Presv</td><td>Add</td></tr><tr><td>KPR</td><td>131</td><td>11.2</td><td>1,299</td><td>97.3</td><td>2.3</td></tr><tr><td>ConClaim</td><td>92</td><td>25.3</td><td>898</td><td>94.6</td><td>1.8</td></tr><tr><td>ConText</td><td>138</td><td>14.7</td><td>1,374</td><td>96.9</td><td>2.3</td></tr></table>

Table 5: Cost of accepting the rewrite from the original Wikipedia article. WER: word edit rate, Click: number of contiguous edits a reviewer must approve, Tok: added tokens, Presv: percentage of the original document preserved in the rewrite, Add: length of the rewrite relative to the original document.

KPRs vs. Search. In Table 4, we test whether grounding on a KPR is worth it over a frontier model with search. We create a QA set of the hardest questions by taking a random sample of 100 multilingual QA instances that no dense model could answer. On this set, we again see that grounding on KPR recovers the most answers at every scale.<sup>6</sup> We additionally give Sol-5.6 access to web search, and it recovers only marginally more answers than EW. Because the questions are posed in English, while the answer may only be documented in its source language, we translate each question to its source language for closed book (CB-T) and web search (WS-T). This helps both, but neither approaches KPR, and even the smallest 7B model grounded on the KPR article outperforms the frontier model with search in either condition.

The sources are other-language Wikipedia editions, so the knowledge is public and indexed, but not surfaced by search. Having access to a KPRrevised document provides a more reliable grounding source than relying on search at query time.

## 5.3 Edit Cost

Table 5 reports the cost of authoring and reviewing the revisions and their shape. We find that KPRs preserve the most of the article and make changes in the smallest number of contiguous edits. While both ConText and KPRs add a similar number of new tokens (2.3× context increase), KPRs integrate substantially more of the source knowledge and require a reviewer looking at the text to approve fewer separate changes. ConClaim makes the smallest change to the article overall, but requires more than twice the approvals of a KPR. Without a claim proposal, every claim routed to a section is passed to the rewrite, so the model revises content that did not need to change.

<table><tr><td rowspan=1 colspan=2>Round-2</td><td rowspan=1 colspan=1>InfoP InfoR-R InfoR-A</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>R1 Main</td><td rowspan=1 colspan=1>0.928   0.416</td></tr><tr><td rowspan=3 colspan=1>Temp.</td><td rowspan=2 colspan=1>ScratchConText</td><td rowspan=1 colspan=1>0.882   0.475     0.223</td></tr><tr><td rowspan=2 colspan=1>0.620   0.649    0.6320.729   0.709    0.682</td></tr><tr><td rowspan=1 colspan=1>KPR</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>R1 Main</td><td rowspan=1 colspan=1>0.904   0.430</td></tr><tr><td rowspan=3 colspan=1>Conf.</td><td rowspan=1 colspan=1>Scratch</td><td rowspan=1 colspan=1>0.875   0.456     0.286</td></tr><tr><td rowspan=1 colspan=1>ConText</td><td rowspan=2 colspan=1>0.666   0.625    0.4290.811   0.782     0.571</td></tr><tr><td rowspan=1 colspan=1>KPR</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>R1 Main</td><td rowspan=1 colspan=1>0.961   0.453</td></tr><tr><td rowspan=2 colspan=1>Bal.</td><td rowspan=1 colspan=1>Scratch</td><td rowspan=1 colspan=1>0.862   0.558     0.326</td></tr><tr><td rowspan=1 colspan=1>ConText</td><td rowspan=1 colspan=1>0.692   0.739    0.537</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>KPR</td><td rowspan=1 colspan=1>0.841   0.856    0.632</td></tr></table>

Table 6: RAGTIME results across each setting. InfoR-R: information recall retained from the round-1 report; InfoR-A: information recall added from the round-2 sources. For the round-1 report (R1 Main), InfoR-R is the recall of round-1 nuggets in the original report.

## 6 Updating Query-Driven Report Generation with KPRs

Our second application moves from a single evolving article to query-driven report generation, where the document is a report synthesized to answer a query and the sources are documents retrieved for it. This setting is harder than the Wikipedia revision in two ways. First, the sources are no longer aligned versions of the same article. Instead, they are independent documents that may overlap with, extend, or contradict one another and the existing report. Second, reports are regenerated as new sources surface—a query asked today may be re-answered next month against updated evidence—which is precisely where query-driven systems are weakest. Current deep research and report-generation systems regenerate from scratch (Google, 2025), discarding prior content as wasted compute. A KPR instead treats each regeneration as an incremental update to the standing report.

Data and Task. We build our experiments on RAGTIME (Lawrie et al., 2026), a multilingual report-generation task in which a system is given a persona and query and performs retrievalaugmented generation over a collection of multilingual documents.<sup>7</sup> From RAGTIME we construct three two-round variants of the task and report their details in Appendix B.

<table><tr><td>Method</td><td>WER</td><td>Click</td><td>Tok</td><td>Presv</td><td>Add</td></tr><tr><td>Scratch</td><td>130</td><td>56.1</td><td>770</td><td>37.3</td><td>1.2</td></tr><tr><td>ConText</td><td>330</td><td>12.5</td><td>3,091</td><td>98.2</td><td>4.3</td></tr><tr><td>KPR</td><td>372</td><td>20.5</td><td>3,475</td><td>96.3</td><td>4.7</td></tr></table>

Table 7: Cost of accepting the rewrite from Round-1 RAGTIME document. WER: word edit rate, Click: number of contiguous edits a reviewer must approve, Tok: added tokens, Presv: percentage of the original document preserved in the rewrite, Add: length of the rewrite relative to the original document.

Temporal: new information arrives over time. RAGTIME metadata contains the document date, so we set a per-topic knowledge cutoff such that half the relevant documents surface only in round 2, simulating the same query asked at two times.

Conflict: a round-2 source contradicts round-1 content. RAGTIME nuggets, QA evaluation units similar to atomic facts (Voorhees, 2004), sometimes take different values across documents (e.g., a margin of victory reported as 2.7% in one and 3% in another). We construct these rounds so that conflicting nuggets appear in each.

Balanced: information is split evenly, with an equal number of nuggets appearing in round-1 and round-2, maximizing how much a method must add while still preserving round-1.

Evaluation. To generate the round-1 report, we condition on the text the way a RAG system might put the relevant documents in context with the query to write the report. We evaluate report quality with MiRAGE (Martin et al., 2026), which has higher human agreement than AutoArgue (Walden et al., 2026) on this task, using the same information metrics from the Wikipedia experiments. InfoP measures precision against the sources used in generation, InfoR-R (retain) the round-1 nuggets preserved in the updated report, and InfoR-A (add) the round-2 nuggets newly incorporated. We omit ConClaim in this setting, as the unfiltered claim sets from RAGTIME’s source documents exceed the context window.

## 6.1 Report Quality

Table 6 reports the InfoP and the two InfoR variants across all three settings. KPR achieves the highest recall of both round-1 and round-2 nuggets in every setting, and leads on precision among the rewrites. Scratch adds the least information between rounds, while ConText has the lowest precision of any method in every setting. The round-1 report scores highest on precision, only needing to be faithful to the round-1 sources.

Every method recalls more round-1 nuggets than the round-1 report. The incremental methods never revisit the round-1 documents, but some round-1 information recurs in round-2, so integrating those sources recovers nuggets the original missed.

Conflict. When sources disagree across rounds, ConText integrates less of the round-2 information than it did in either other setting. Without a mechanism to flag conflicts it must resolve them implicitly during the rewrite, and doing so appears to suppress how much of the incoming information it incorporates. KPR instead withholds conflicting claims, adding new facts at a rate comparable to the other settings while holding the highest precision of any method that integrates a meaningful amount.

## 6.2 Edit Cost

Table 7 reports the cost of authoring and reviewing each update. We find that Scratch’s high precision makes sense in the context of its edit shape, generating short, but high-confidence reports. This also makes it the most expensive to review, as only coincidental n-gram overlaps are preserved in the generation. ConText and KPR instead make similarly sized updates to the round-1 document, but ConText struggles to add as much information to the updates as KPR.

## 7 Findings Across Settings

Claims as the unit for information. We demonstrate that claims are a better unit of information to operate over than text when integrating knowledge. In both experiments, conditioning the rewrite on claims rather than source text improves both the precision and the recall of added information, with the same trend showing in the QA experiments. Claim-level representations are known to be the appropriate unit for factuality evaluation (Min et al., 2023; Song et al., 2024) and can even boost retrieval performance (Chen et al., 2024). Our results extend this finding to knowledge integration, where conditioning generation on claims rather than source text yields a more faithful grounding document.

Claim proposals improve the rewrite. We find that claim proposals consistently improve the quality of the rewrite over the baselines, maintaining precision and raising the recall of integrated information. The proposal also helps shape the rewrite into contiguous, easy to review edits. A KPR also adds the most information per token generated, as baselines writing a comparable volume of text integrate substantially less of the source information.

## 8 Conclusion

We introduce Knowledge Pull Requests, which reframe continual document authoring as a reviewable, knowledge-level operation rather than an uninterpretable rewrite. By decomposing sources into claims, filtering and routing them, and surfacing conflicts, a KPR produces a ChangeLog that separates what knowledge changes from how the text changes. Across cross-lingual Wikipedia revision and query-driven report updating, KPRs integrate new information more completely and preserve existing content better than other methods, while concentrating their changes into contiguous edits a reviewer can approve. By making the unit of change a reviewable claim rather than an opaque edit, KPRs open a path toward collaborative, human-in-theloop authoring.

## Limitations

Computational Efficiency. Our pipeline performs every step—decomposition, classification, routing, and rewriting—with an LLM. While a KPR rewrites only the sections that change rather than regenerating the entire document, each intermediate step still requires an LLM call, and cheaper alternatives exist. For example, a lightweight encoder could classify claim containment or route claims to sections in place of prompting. Our editcost metrics measure the size and shape of the resulting rewrite, not the compute spent producing it. Reducing this per-step cost is important for deploying KPRs at scale (e.g., continually updating Wikipedia) and is left to future work.d

Human Review. A central motivation for KPRs is that the ChangeLog is inspectable and reviewable. A human can assess proposed claims, resolve flagged conflicts, and approve or reject changes before they are merged, analogous to a code review. However, our experiments run the pipeline automatically, withholding flagged conflicts from the rewrite, and thus do not evaluate the review process itself. Human studies that investigate whether a ChangeLog makes an editor faster and more accurate than a text diff, and how claim proposals should be adjudicated by a human, are both necessary future work.

## References

Carlos E. Alchourrón, Peter Gärdenfors, and David Makinson. 1985. On the logic of theory change: Partial meet contraction and revision functions. The Journal ofSymbolic Logic, 50(2):510–530.

Samuel Barham, Chandler May, and Benjamin Van Durme. 2025. Megawika 2: A more comprehensive multilingual collection of articles and their sources. Preprint, arXiv:2508.03828.

Samuel Barham, Orion Weller, Michelle Yuan, Kenton Murray, Mahsa Yarmohammadi, Zhengping Jiang, Siddharth Vashishtha, Alexander Martin, Anqi Liu, Aaron Steven White, Jordan Boyd-Graber, and Benjamin Van Durme. 2023. Megawika: Millions of reports and their sources across 50 diverse languages. Preprint, arXiv:2307.07049.

Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15159–15177, Miami, Florida, USA. Association for Computational Linguistics.

Jeffrey Cheng, Marc Marone, Orion Weller, Dawn Lawrie, Daniel Khashabi, and Benjamin Van Durme. 2024. Dated data: Tracing knowledge cutoffs in large language models. In First Conference on Language Modeling.

Eduardo Fermé and Sven Ove Hansson. 2011. Agm 25 years. Journal ofPhilosophical Logic, 40(2):295– 331.

William Fleshman and Benjamin Van Durme. 2025. Lora-augmented generation (lag) for knowledge-intensive language tasks. Preprint, arXiv:2507.05346.

William Fleshman, Aleem Khan, Marc Marone, and Benjamin Van Durme. 2025. Adapterswap: Continuous training of llms with data removal and accesscontrol guarantees. Preprint, arXiv:2404.08417.

Tim Franzmeyer, Aleksandar Shtedritski, Samuel Albanie, Philip Torr, Joao F. Henriques, and Jakob Foerster. 2024. HelloFresh: LLM evalutions on streams of real-world human editorial actions across X community notes and Wikipedia edits. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12702–12716, Bangkok, Thailand. Association for Computational Linguistics.

Revanth Gangi Reddy, Tanay Dixit, Jiaxin Qin, Cheng Qian, Daniel Lee, Jiawei Han, Kevin Small, Xing Fan, Ruhi Sarikaya, and Heng Ji. 2026. Winell: Wikipedia never-ending updating with llm agents. In Proceedings of the ACM Web Conference 2026, WWW ’26, page 4396–4406, New York, NY, USA. Association for Computing Machinery.

Google. 2025. Gemini deep research.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, and 24 others. 2024. OLMo: Accelerating the science of language models. In Proceedings ofthe 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15789–15809, Bangkok, Thailand. Association for Computational Linguistics.

Anisha Gunjal and Greg Durrett. 2024. Molecular facts: Desiderata for decontextualization in LLM fact verification. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3751–3768, Miami, Florida, USA. Association for Computational Linguistics.

Peter Gärdenfors. 1988. Knowledge in Flux: Modeling the Dynamics of Epistemic States. MIT Press.

Sven Ove Hansson. 1992. In defense of base contraction. Synthese, 91(3):239–245.

Sven Ove Hansson. 1999. A textbook of belief dynamics - theory change and database updating. In Applied Logic Series.

Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, and Minjoon Seo. 2022. TemporalWiki: A lifelong benchmark for training and evaluating ever-evolving language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 6237–6250, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024. Mixtral of experts. Preprint, arXiv:2401.04088.

Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. 2024. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5042–5063, Miami, Florida, USA. Association for Computational Linguistics.

Hirofumi Katsuno and Alberto O. Mendelzon. 1991. On the difference between updating a knowledge base and revising it. In Proceedings ofthe Second International Conference on Principles of Knowledge Representation and Reasoning, KR’91, page 387–394, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc.

Dawn Lawrie, Sean MacAvaney, James Mayfield, Luca Soldaini, Eugene Yang, and Andrew Yates. 2026. Overview of the trec 2025 ragtime track. Preprint, arXiv:2602.10024.

Robert L. Logan IV, Alexandre Passos, Sameer Singh, and Ming-Wei Chang. 2022. FRUIT: Faithfully reflecting updated information in text. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3670–3686, Seattle, United States. Association for Computational Linguistics.

Alexander Martin, William Walden, Reno Kriz, Dengjia Zhang, Kate Sanders, Eugene Yang, Chihsheng Jin, and Benjamin Van Durme. 2026. Seeing through the mirage: Evaluating multimodal retrieval augmented generation. Preprint, arXiv:2510.24870.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA. Curran Associates Inc.

Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In The Eleventh International Conference on Learning Representations.

Meta AI. 2025. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. https://ai.meta.com/blog/ llama-4-multimodal-intelligence/. Accessed: 2026-09-21.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Ani Nenkova and Rebecca Passonneau. 2004. Evaluating content selection in summarization: The pyramid method. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pages 145–152, Boston, Massachusetts, USA. Association for Computational Linguistics.

NVIDIA, :, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Kondratenko, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alisa Liu, Amelia Barton, and 340 others. 2025. Nvidia nemotron 3: Efficient and open intelligence. Preprint, arXiv:2512.20856.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, and 50 others. 2026. Olmo 3. Preprint, arXiv:2512.13961.

OpenAI. 2025. Deep research system card. Technical report.

OpenAI. 2026. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/ index/gpt-5-6/. Accessed: 2026-08-03.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 487–503, Online. Association for Computational Linguistics.

John L. Pollock. 1987. Defeasible reasoning. Cognitive Science, 11(4):481–518.

John L. Pollock and Anthony S. Gillies. 2000. Belief revision and epistemology. Synthese, 122(1):69–92.

Semrush. 2025. The most-cited domains in AI: A 3-month study. Semrush Blog. https://www. semrush.com/blog/most-cited-domains-ai/, accessed 30 July 2026.

Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam. 2024. Assisting in writing Wikipedia-like articles from scratch with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6252–6278, Mexico City, Mexico. Association for Computational Linguistics.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, and 17 others. 2024. Dolma: an open corpus of three trillion tokens for language model pretraining research. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15725–15788, Bangkok, Thailand. Association for Computational Linguistics.

Yixiao Song, Yekyung Kim, and Mohit Iyyer. 2024. VeriScore: Evaluating the factuality of verifiable claims in long-form text generation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9447–9474, Miami, Florida, USA. Association for Computational Linguistics.

Alexander Spangher, Xiang Ren, Jonathan May, and Nanyun Peng. 2022. NewsEdits: A news article revision dataset and a novel document-level reasoning challenge. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 127–157, Seattle, United States. Association for Computational Linguistics.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, and 20 others. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

theStacc. 2026. Wikipedia gets 47.9% of ChatGPT citations. theStacc Blog. https://thestacc.com/ blog/wikipedia-chatgpt-citations/, accessed 30 July 2026.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

Ellen M. Voorhees. 2004. Overview of the trec 2003 question answering track. In Text Retrieval Conference.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. Fresh-LLMs: Refreshing large language models with search engine augmentation. In Findings ofthe Association for Computational Linguistics: ACL 2024, pages 13697–13720, Bangkok, Thailand. Association for Computational Linguistics.

William Walden, Marc Mason, Orion Weller, Laura Dietz, John Conroy, Neil Molino, Hannah Recknor, Bryan Li, Gabrielle Kaili-May Liu, Yu Hou, Dawn Lawrie, James Mayfield, and Eugene Yang. 2026. Auto-argue: Llm-based report generation evaluation. Preprint, arXiv:2509.26184.

Wikimedia Foundation. 2023. Wikipedia’s value in the age of generative AI. Wikimedia Foundation News. https:// wikimediafoundation.org/news/2023/07/12/ wikipedias-value-in-the-age-of-generative-ai/, accessed 30 July 2026.

Wikimedia Foundation. 2025. New user trends on Wikipedia. Wikimedia Foundation News. https://wikimediafoundation.org/news/ 2025/10/17/new-user-trends-on-wikipedia/, accessed 30 July 2026.

Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. 2024. Knowledge conflicts for LLMs: A survey. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 8541– 8565, Miami, Florida, USA. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

## A Wikipedia Experiments Implementation Details

Articles. We sample 599 articles from MegaWika 2.0 (Barham et al., 2025), selecting evenly from the distribution of cross-lingual links per document so that the source language editions and number of source articles vary across topics. We cover 49 of MegaWika’s 50 languages (all but English, our target): af, ar, az, bn, cs, de, es, et, fa, fi, fr, ga, gl, gu, he, hi, hr, id, it, ja, ka, kk, km, ko, lt, lv, mk, ml, mn, mr, my, ne, nl, pl, ps, pt, ro, ru, si, sl, sv, ta, th, tr, uk, ur, vi, xh, zh.

QA Generation. To build the questionanswering set that measures grounding, we convert every claim in an article into a question with the claim as its gold answer. For the multilingual split, we additionally discard claims that are contained by English Wikipedia, so that the remaining questions target knowledge genuinely available only in other language editions. We then filter the questions for quality and deduplicate them, keeping only questions that (i) are specific, well-formed factoid questions with a clear, verifiable answer; (ii) are answerable on their own, without external context (e.g., “who is the captain of the ship?” is discarded, as it depends on an unstated referent); and (iii) are cleanly and correctly answered by the gold answer, taking the claim as ground truth. Table 8 reports the number of questions remaining after each stage. We provide our prompts for filtering the data in Figure 10, Figure 11, and Figure 12.

<table><tr><td>Stage</td><td>Multilingual</td><td>English</td></tr><tr><td>Original</td><td>427,898</td><td>111,693</td></tr><tr><td>Absent/Conflict</td><td>81,490</td><td></td></tr><tr><td>Final</td><td>44,683</td><td>26,572</td></tr></table>

Table 8: Question counts at each stage of QA generation. Original: questions generated from all article claims. Absent/Conflict: multilingual questions retained after keeping only claims absent from or contradicting English Wikipedia. Final: questions remaining after quality filtering and deduplication.

100-Question Subset. To construct this set, we filter the multilingual QA questions down to those not answered correctly by any of our dense models in a closed-book, non translated setting, and then sample 100 at random. This isolates questions whose answers are absent from the models parametric knowledge, so that any performance must come from the grounding document. For the translated versions of this subset (CB-T, WS-T), we translate questions with Qwen3.5-27B.

Frontier Model Usage. We access GPT-5.6 (Sol) through the OpenAI API, setting reasoning effort to its maximum and retaining this setting in the web-search condition. In Table 9 we also show performance when conditioning on ConText and ConClaim.

## B RAGTIME Experiments Implementation Details

Split Creation. We construct all three splits programmatically from the RAGTIME relevance judgments, reporting the resulting novel nugget counts per round in Table 10.

• Temporal: we choose a per-topic knowledge cutoff date that divides the relevant documents into two equal halves by publication date, assigning the earlier half to round 1 and the later half to round 2.

• Conflict: RAGTIME distinguishes OR nuggets, which admit multiple acceptable answers across documents, from AND nuggets, whose supporting information is spread across documents. We place the documents supporting an OR nugget’s conflicting answers (or an AND nugget’s complementary pieces) in different rounds, so that a conflicting or complementary document surfaces in round 2.

• Balanced: we partition documents to make the number of nuggets appearing in each round as equal as possible.

## C Method Details

Claim Decomposition. Claim decomposition is less straightforward across languages than within a single one. A claim must end up as an atomic, decontextualized statement in the target language, faithful to its meaning in the source language and without propagating translation errors. We compare three strategies (Table 11):

• Translate-then-decompose: translate the source sentence into the target language, then decompose claims from the translation. We use both the MegaWika NLLB translations (Team et al., 2022) and Qwen3.5-9B.

• Native-then-translate: decompose the sentence into claims in the source language, then translate those claims to the target language.

• Cross-lingual decomposition: decompose the source sentence directly into claims in English.<sup>8</sup>

Translate-then-decompose is the least faithful under either translator, as translation errors compound with decomposition. Native-then-translate and cross-lingual decomposition achieve roughly the same faithfulness. We use cross-lingual decomposition because it uses a single LLM pass instead of two passes.

<table><tr><td>Model</td><td>CB</td><td>CB-T</td><td>English-Wiki</td><td>ConText</td><td>ConClaim</td><td>KPR</td><td>WS</td><td>WS-T</td></tr><tr><td rowspan="4">Q3.5-27B Q3-30B G4-31B</td><td>0</td><td>5</td><td>3</td><td>18</td><td>31</td><td>52</td><td></td><td></td></tr><tr><td>0</td><td>3</td><td>12</td><td>26</td><td>34</td><td>54</td><td></td><td></td></tr><tr><td>0</td><td>4</td><td>4</td><td>20</td><td>32</td><td>54</td><td></td><td></td></tr><tr><td>0</td><td>5</td><td>8</td><td>25</td><td>31</td><td>51</td><td></td><td></td></tr><tr><td>Sol-5.6</td><td>L3.3-70B 13</td><td>20</td><td>32</td><td>46</td><td>54</td><td>62</td><td>38</td><td>41</td></tr></table>

Table 9: Frontier and open model performance on the hardest QA set (Table 4), including the ConText and ConClaim rewrite conditions omitted from the main table. CB-T: closed-book translated; WS-T: web search translated.

<table><tr><td>Setting</td><td>Round-1</td><td>Round-2</td></tr><tr><td>Temporal</td><td>12.2</td><td>3.2</td></tr><tr><td>Conflict</td><td>14.7</td><td>0.5</td></tr><tr><td>Balanced</td><td>7.6</td><td>8.0</td></tr></table>

Table 10: Novel nuggets introduced per round in each RAGTIME variant (nuggets not present in the prior round).

Prompts for each method. The KPR prompts are given in Figure 12 (claim review), Figure 13 (routing), Figure 14 (new-section routing), Figure 15 (rewriting), and Figure 16 (new-section rewriting). ConClaim and ConText reuse the same routing and rewriting flow as KPR and differ from KPR only as follows:

• ConClaim omits the claim-review step (Figure 12), so claims are not filtered by absent/covered/conflicting/relevance before rewriting.

• ConText conditions each section’s rewrite on raw source text, with an LLM classifying section relevance in place of claim review and routing. Its rewriting prompts are reworded to take source text as input, which we omit, as they are nearly identical to the KPR versions

## D Metric Details

## D.1 Quality

We measure quality with MiRAGE (Martin et al., 2026), which scores a predicted text P against a set of reference evidence R by decomposing both into atomic subclaims and measuring support with a scoring function $\mathbf { \mathsf { s } } ( \cdot , \cdot ) \ \in \ [ 0 , 1 ]$ In every experiment s is a cross-lingual LLM support judge (Qwen3.5-27B). Following the definitions, we report Information Precision (InfoP), and two variations of Information Recall (InfoR-A, InfoR-R).

## D.1.1 Information Precision

InfoP is a source-constrained variant of FActScore (Min et al., 2023). For each $P ,$ we decompose the sentences into subclaims $C _ { P }$ and score each for support against R:

$$
\mathrm { I n f o P } ( C _ { P } , R ) = \frac { 1 } { | C _ { P } | } \sum _ { c \in C _ { P } } \mathsf { s } ( c , R ) .\tag{1}
$$

We use the Collection variant of InfoP, where the documents are used, instead of their underlying claims. For Wikipedia, P is the rewrite’s added sentences and R is the non-English sources. We do not do the full articles here because it is not expected for the unmodified English Article to be supported by the non-English sources. For RAG-TIME, P is the round-2 report and R is the union of round-1 and round-2 source documents.

## D.1.2 Information Recall

InfoR follows the same formulation, with P and R reverse, scoring claims decomposed from the reference $C _ { R }$ against P.

$$
\mathrm { I n f o R } ( C _ { R } , P ) = \frac { 1 } { | C _ { R } | } \sum _ { c \in C _ { R } } \mathsf { s } ( c , P ) .\tag{2}
$$

For both tasks, we introduce two variants of InfoR: retain (InfoR-R), the number of claims preserved in the rewrite, and add (InfoR-A), the number of source claims added in the rewrite. For Wikipedia, InfoR-R has $C _ { R }$ be the set of decomposed claims of the original English article, measuring how much of the original English Wikipedia the rewrite preserves; and InfoR-A has $C _ { R }$ be the set of decomposed claims of the non-English sources, measuring how much was added from the non-English sources during rewriting. For RAGTIME, InfoR-R has $C _ { R }$ be the set of nuggets from round-1, measuring how much of the round-1 report the rewrite preserves; and InfoR-A has $C _ { R }$ be the set of nuggets from round-2, measuring how much was added from the new sources during the rewriting.

<table><tr><td>Strategy</td><td>Claims</td><td>Supported</td><td>Contradicted</td><td>Hallucinated</td><td>Cannot-Det</td></tr><tr><td>English-to-English</td><td>175,267</td><td>88.7</td><td>0.7</td><td>10.0</td><td>0.7</td></tr><tr><td>NLLB Translate</td><td>431,098</td><td>79.8</td><td>2.3</td><td>16.8</td><td>1.1</td></tr><tr><td>LLM Translate</td><td>451,333</td><td>84.4</td><td>1.0</td><td>13.2</td><td>1.4</td></tr><tr><td>Native Decomp</td><td>376,331</td><td>89.8</td><td>1.1</td><td>8.3</td><td>0.8</td></tr><tr><td>Cross Decomp</td><td>357,460</td><td>89.0</td><td>0.9</td><td>9.2</td><td>0.9</td></tr></table>

Table 11: Claim decomposition strategies evaluated by verdict distribution. Each row shows the number of claims produced and the percentage judged supported, contradicted, hallucinated, or cannot-determine when verified against the source.

## D.2 Edit Cost

Each metric compares the original token sequence M to the rewritten one R via the word-level diff, which segments the change into maximal contiguous blocks: equal, insert, replace, delete. Ratio metrics are token-weighted over the document set D to not overweight short documents.

• Word edit rate (WER): the word-level Levenshtein distance (insertions, deletions, substitutions) between M and R,

$$
\mathrm { W E R } = \frac { \sum _ { d \in \mathcal { D } } \mathrm { e d i t s } ( M _ { d } , R _ { d } ) } { \sum _ { d \in \mathcal { D } } \left| M _ { d } \right| } \times 1 0 0 \% ,
$$

it can exceed 100% when a rewrite adds more than the original length.

• Click: one click of the “approve” button per maximal contiguous inserted or replaced span, regardless of its length,

$$
{ \mathrm { C l i c k } } = { \big | } { \{ i n s e r t { \mathrm { a n d } } r e p l a c e { \mathrm { b l o c k s } } \} } { \big | } ,
$$

reported as the per-document mean. Many scattered small edits cost more clicks than a few large contiguous blocks, even when the latter add more total text.

• Added tokens (Tok): the number of new word tokens,

$$
\mathrm { T o k } = \sum _ { i n s e r t , r e p l a c e \ b l o c k s } | R \mathrm { - s p a n } | ,
$$

the per-document mean of the rewrites token count.

• Preservation % (Presv): the percentage of the original document the method keeps verbatim, as contiguous equal spans,

$$
\mathrm { P r e s v } = \frac { \sum _ { d } \sum _ { e q u a l \ b l o c k s } \left| { \cal M } \mathrm { - s p a n } \right| } { \sum _ { d } \left| { \cal M } _ { d } \right| } \times 1 0 0 \%
$$

High preservation means the method left most of the original in place; low means it regenerated the content.

• Add (expansion): how many times larger the rewritten document is than the original,

$$
\mathrm { A d d } = { \frac { \sum _ { d } { | R _ { d } | } } { \sum _ { d } { | M _ { d } | } } }
$$

For the Wikipedia revisions M is the English article and R the reconstructed rewrite. For RAGTIME M is the round-1 seed report and R the round-2 report. WER, Click, and Tok measure the cost of producing and reviewing a rewrite, while Presv and Add describe its shape. Note that a good revision may legitimately rewrite much of a document, so neither is better in a fixed direction.

## E Acknowledgement of AI

We use AI for coding and to edit and condense writing.

```markdown
Instructions:
- You are given a paragraph and one sentence from the paragraph to decompose
- The text may be in any language — decompose the sentence into atomic claims
- Output the claims in the SAME LANGUAGE as the input (do not translate)
- You must output a JSON array: [{"claim": "..."}, {"claim": "..."}, ...]
##PARAGRAPH##: On 15 April 2019, just before 18:20 CEST, a structural fire broke out in the roof
space of Notre-Dame de Paris, a medieval Catholic cathedral in Paris, France. By the time the
fire was extinguished, the cathedral's wooden spire had collapsed, most of the wooden roof had
been destroyed, and the cathedral's upper walls were severely damaged.
##SENTENCE##: On 15 April 2019, just before 18:20 CEST, a structural fire broke out in the roof
space of Notre-Dame de Paris, a medieval Catholic cathedral in Paris, France.
##DECOMPOSITION##:
```json
[
{"claim": "A structural fire broke out"},
{"claim": "The fire broke out on 15 April 2019"},
{"claim": "The fire broke out just before 18:20 CEST"},
{"claim": "The fire broke out in the roof space"},
{"claim": "Notre-Dame de Paris is a medieval Catholic cathedral"},
{"claim": "Notre-Dame de Paris is located in Paris, France"}
##PARAGRAPH##: Le 15 avril 2019, peu avant 18h20 CEST, un incendie s'est déclaré dans la charpente
de la cathédrale Notre-Dame de Paris, une cathédrale catholique médiévale située à Paris, en
France. Au moment où l'incendie a été éteint, la flèche en bois de la cathédrale s'était
effondrée, la majeure partie de la toiture en bois avait été détruite et les murs supérieurs
de la cathédrale avaient été gravement endommagés.
##SENTENCE##: Le 15 avril 2019, peu avant 18h20 CEST, un incendie s'est déclaré dans la charpente
de la cathédrale Notre-Dame de Paris, une cathédrale catholique médiévale située à Paris, en
France.
##DECOMPOSITION##:
```json
[
{"claim": "Un incendie s'est déclaré"},
{"claim": "L'incendie s'est déclaré le 15 avril 2019"},
{"claim": "L'incendie s'est déclaré peu avant 18h20 CEST"},
{"claim": "L'incendie s'est déclaré dans la charpente"},
{"claim": "Notre-Dame de Paris est une cathédrale catholique médiévale"},
{"claim": "Notre-Dame de Paris est située à Paris, en France"}
]

##PARAGRAPH## [paragraph]
##SENTENCE## [sentence]
##DECOMPOSITION##:
```  
Figure 5: Native-then-translate. Decomposes a sentence into atomic claims in the source language.

Instructions:   
- You are given a JSON array of claims written in a non-English language   
- Translate each claim into English, preserving meaning exactly   
- Output a JSON array in the same format: [{"claim": "..."}, {"claim": "..."}, ...]   
- Do not add, remove, or merge claims — one-to-one translation only   
##CLAIMS##: [{"claim": "Un incendie s'est déclaré"}, {"claim": "L'incendie s'est déclaré le 15   
avril 2019"}, {"claim": "Notre-Dame de Paris est une cathédrale catholique médiévale"}, {"   
claim": "Notre-Dame de Paris est située à Paris, en France"}]   
##TRANSLATION##:   
\`\`\`json   
[   
{"claim": "A fire broke out"},   
{"claim": "The fire broke out on 15 April 2019"},   
{"claim": "Notre-Dame de Paris is a medieval Catholic cathedral"},   
{"claim": "Notre-Dame de Paris is located in Paris, France"}   
]   
##CLAIMS##: [claims]   
##TRANSLATION##:  
Figure 6: Claim-translation prompt. Translates native-language claims into English one-to-one, without adding, removing or merging claims.

Instructions:   
- You are given a paragraph and one sentence from the paragraph to decompose   
- The text may be in any language   
- Decompose the sentence into atomic claims and output them in ENGLISH regardless of the input   
language   
- You must output a JSON array: [{"claim": "..."}, {"claim": "..."}, ...]   
##PARAGRAPH##: Le 15 avril 2019, peu avant 18h20 CEST, un incendie s'est déclaré dans la charpente   
de la cathédrale Notre-Dame de Paris, une cathédrale catholique médiévale située à Paris, en   
France. Au moment où l'incendie a été éteint, la flèche en bois de la cathédrale s'était   
effondrée, la majeure partie de la toiture en bois avait été détruite et les murs supérieurs   
de la cathédrale avaient été gravement endommagés.   
##SENTENCE##: Le 15 avril 2019, peu avant 18h20 CEST, un incendie s'est déclaré dans la charpente   
de la cathédrale Notre-Dame de Paris, une cathédrale catholique médiévale située à Paris, en   
France.   
##DECOMPOSITION##:   
\`\`\`json   
[   
{"claim": "A structural fire broke out"},   
{"claim": "The fire broke out on 15 April 2019"},   
{"claim": "The fire broke out just before 18:20 CEST"},   
{"claim": "The fire broke out in the roof space"},   
{"claim": "Notre-Dame de Paris is a medieval Catholic cathedral"},   
{"claim": "Notre-Dame de Paris is located in Paris, France"}   
]   
  
##PARAGRAPH##: El huracán Irma fue un extremadamente poderoso huracán de Cabo Verde que causó una   
destrucción generalizada en su camino a principios de septiembre de 2017. Irma fue el primer   
huracán de categoría 5 en golpear las Islas de Barlovento, seguido por María dos semanas despu   
és.   
##SENTENCE##: El huracán Irma fue un extremadamente poderoso huracán de Cabo Verde que causó una   
destrucción generalizada en su camino a principios de septiembre de 2017.   
##DECOMPOSITION##:   
\` \`\`json   
[   
{"claim": "Hurricane Irma was a Cape Verde hurricane"},   
{"claim": "Hurricane Irma was extremely powerful"},   
{"claim": "Hurricane Irma caused widespread destruction"},   
{"claim": "Hurricane Irma occurred in early September 2017"}   
]   
  
##PARAGRAPH## [paragraph]   
##SENTENCE## [sentence]   
##DECOMPOSITION##:  
Figure 7: Cross-lingual decomposition. Decomposes a non-English sentence directly into English claims, with no separate translation step.

![](images/04de4960c85131dc373afea8e85ce240b037330503c9cae79c1f8b56e489daf9.jpg)  
Figure 8: Sentence-translation prompt. Translates a source-language sentence into English prior to decomposition.

You are a fact-checker. You are given a source sentence (which may be in any language) and one   
English claim derived from that sentence.   
Determine whether the claim is supported by the source sentence using your multilingual   
understanding.   
Verdict options:   
- SUPPORTED: The source sentence explicitly supports the claim.   
- CONTRADICTED: The claim states something that directly contradicts the source sentence (wrong   
fact, wrong name, wrong number, etc.).   
- HALLUCINATED: The claim contains information that is not present in the source sentence and   
cannot be inferred from it.   
- CANNOT\_DETERMINE: The source sentence does not contain enough information to judge the claim.   
Respond with a JSON object only:   
{"verdict": "<SUPPORTED|CONTRADICTED|HALLUCINATED|CANNOT\_DETERMINE>", "explanation": "<one   
sentence>"}   
##SOURCE SENTENCE##: Brussels Basketball is a professional basketball club based in Brussels,   
Belgium.   
##CLAIM (English)##: Brussels Basketball is a basketball club.   
##VERDICT##: {"verdict": "SUPPORTED", "explanation": "The source sentence explicitly states it is   
a basketball club."}   
##SOURCE SENTENCE##: Brussels Basketball is a professional basketball club based in Brussels,   
Belgium.   
##CLAIM (English)##: Brussels Basketball was founded in 1899.   
##VERDICT##: {"verdict": "CONTRADICTED", "explanation": "The source sentence does not mention a   
founding year of 1899; this contradicts information from other knowledge."}   
##SOURCE SENTENCE##: Brussels Basketball is a professional basketball club based in Brussels,   
Belgium.   
##CLAIM (English)##: Brussels Basketball has won three national championships.   
##VERDICT##: {"verdict": "HALLUCINATED", "explanation": "The source sentence contains no   
information about championships won."}   
##SOURCE SENTENCE##: Le Brussels Basketball est un club belge de basket-ball fondé en 1957, basé à   
Bruxelles.   
##CLAIM (English)##: Brussels Basketball is a Belgian basketball club.   
##VERDICT##: {"verdict": "SUPPORTED", "explanation": "The French source sentence states it is a   
Belgian basketball club (club belge de basket-ball)."}   
##SOURCE SENTENCE##: Le Brussels Basketball est un club belge de basket-ball fondé en 1957, basé à   
Bruxelles.   
##CLAIM (English)##: Brussels Basketball was founded in 1962.   
##VERDICT##: {"verdict": "CONTRADICTED", "explanation": "The source sentence states the club was   
founded in 1957 (fondé en 1957), not 1962."}   
##SOURCE SENTENCE##: [sentence]   
##CLAIM (English)##: [claim]   
##VERDICT##:  
Figure 9: LLM-judge prompt. Fact-checks an English claim against a (possibly non-English) source sentence, returning one of four verdicts.

![](images/61d1661c7ec487fcd4891ef29af341a5e5a3b2938b94fdea35d51446bc53abed.jpg)  
Figure 11: Standalone-answerability prompt. Judges whether a quiz question can be answered without its one hidden source document (SELF\_CONTAINED / BORDERLINE / CONTEXT\_DEPENDENT).

You are reviewing proposed additions to the English Wikipedia article "[article\_title]". Each   
numbered claim below was extracted from a non-English edition of the same article. For each   
claim, determine its relationship to the ARTICLE TEXT:   
- SUPPORTED — the article already states this information, either directly or through a more   
specific statement that entails it. The claim adds nothing new.   
- ABSENT — the article does not state this information, and nothing in the article conflicts with   
it. The claim is a candidate addition.   
- CONTRADICTED — the article states something incompatible with the claim (a different date,   
number, name, place, quantity, or relationship).   
Rules:   
- Judge ONLY against the article text below. Do not use outside knowledge about the topic.   
- A claim that is partially in the article: if the new part conflicts with the article, use   
CONTRADICTED; if the new part is simply not mentioned, use ABSENT.   
- For SUPPORTED and CONTRADICTED, copy the single most relevant sentence or span from the article   
verbatim into "evidence" (at most 40 words). For ABSENT, set "evidence" to null.   
ARTICLE TEXT:   
[article\_text]   
CLAIMS:   
[claims]   
Respond with ONLY a JSON array, one object per claim, in order:   
[{"idx": 1, "verdict": "SUPPORTED", "evidence": "..."}, {"idx": 2, "verdict": "ABSENT", "evidence":   
null}, ...]  
Figure 12: Claim-review. Classifies each candidate claim (extracted from a non-English edition) against the ful English article text as SUPPORTED, ABSENT, or CONTRADICTED.

![](images/33b19ebdc5f4a4584a0cbbab170e43eb50e71d130813d31fabc5e4065c685428.jpg)  
Figure 13: Section-routing. Assigns each new claim to the existing section it best fits, or to 0 when no existing section is suitable and a new one is needed.

You are editing the English Wikipedia article "[title]". The claims below did not fit any existing   
section (outline shown for context). Propose one or more NEW sections to hold them, grouping   
related claims together. Keep the number of new sections small; only propose a section when   
several claims share a topic, or a single claim is clearly its own topic.   
EXISTING OUTLINE:   
[outline]   
ORPHAN CLAIMS:   
[claims]   
Respond with ONLY a JSON array of proposed sections:   
[{"heading": "Reception", "after\_section": 4, "claim\_idxs": [1, 3, 5]}, ...]   
- "heading": the new section title (Wikipedia style, plain).   
- "after\_section": the [n] index of the existing section this new section should follow (use the   
highest sensible index; 0 to place right after the lead).   
- "claim\_idxs": the claim numbers (from the list above) that belong in this section.   
Every claim must appear in exactly one proposed section.  
Figure 14: New-section planning prompt. Groups the orphan claims that fit no existing section into a small set of proposed new sections, each with a heading and insertion point.

![](images/23269dec7fc17b50b6541c657401d05e65472b0ece2043af14def297edc4df71.jpg)  
Figure 15: Fluent section-rewrite prompt. Integrates new facts into an existing section as natural prose, merging facts that share a subject into single flowing sentences rather than one sentence per fact.

You are writing the "[heading]" section of the English Wikipedia article "[title]" using ONLY the   
facts below. Write FLUENT, natural encyclopedic prose — not a list of one-fact sentences.   
Rules — WRITE WELL, DON'T LIST:   
- MERGE facts that share a subject into single flowing sentences using coordination and lists.   
Example — these four facts:   
- The 1876 Constitution maintained Spain as a constitutional monarchy.   
- The 1876 Constitution granted the king the power to appoint members of the Senate.   
- The 1876 Constitution granted the king the power to repeal laws.   
- The 1876 Constitution granted the king the title of Commander-in-Chief of the Army.   
should become ONE sentence:   
"The 1876 Constitution maintained Spain as a constitutional monarchy, granting the king the   
power to appoint members of the Senate and repeal laws, as well as the title of Commander-in-  
Chief of the Army."   
- Use only the given facts; do not invent detail or add outside knowledge. Group related facts   
into coherent sentences and paragraphs.   
- CITATIONS: some facts end with a tag like [cite: 3]. Append the marker(s) at the END of the   
sentence ("...Army.[3]"); combine when merging ("...[3][5]"). Use only numbers given; never   
invent; never write the "[cite: ...]" text.   
- Objective, encyclopedic tone. Output ONE SENTENCE PER LINE (a merged multi-fact sentence is ONE   
line). Blank line between paragraphs. No heading, no commentary.   
FACTS:   
[claims]   
Respond with ONLY the section text.  
Figure 16: Fluent new-section prompt. Drafts a new section from only its facts as natural prose, merging related facts into flowing sentences.

![](images/2ce340a7af03a8f016fd176ac9b6ddad77dccb0638564086f4b68857651b2610.jpg)  
Figure 17: Query-relevance prompt. Judges whether each candidate fact is on-topic for the requested report (relevant true/false), independent of whether the fact is interesting or novel.