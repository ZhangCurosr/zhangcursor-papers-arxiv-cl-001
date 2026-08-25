# PATCHWRITE: ONE LINE, NOT ONE SECTION — COMPILE-GATED, VALIDITY-PRESERVING EDITING FOR AI-DRAFTED MANUSCRIPTS

A PREPRINT

Weiwei Yang Solus (https://solus.xin) wxs123x@gmail.com

August 25, 2026

## ABSTRACT

Automated manuscript pipelines often regenerate a whole section to fix a local defect, and the PDF still builds even though unrelated metrics and citations have quietly moved. PatchWrite instead applies one line-interval EDIT N M (span ≤ 40), committing only when pdflatex is clean and every \cite key and numeric token is attested in a registry or experimental log, and rolling back to the last compiling HEAD otherwise. On a 24-manuscript × 8-fault oracle stress test (768 jobs, split evenly between compile-breaking and content-only faults), whole-slot rewrite mutates an unrelated “12-layer” line in every single case (0/192 preserved, numeric Jaccard 0.6667), while PatchWrite preserves it 192/192 times. Disabling the compile gate drives the accept rate to 0; disabling the evidence gate lets a hallucinated citation through. The pattern holds uniformly across all eight faults. Because oracle repairs test the mechanism rather than generation quality, we reran the same 192 jobs with the writer model itself proposing the EDIT. It is accepted 75% of the time before a one-line prompt fix, and the shortfall traces almost entirely to one reproducible failure mode: the model tries to delete a line using an empty replacement, which the grammar rejects. Telling the model explicitly to comment out rather than empty the line raises acceptance to 100% and eliminates the fallback, though only 84.9% (down from 93.75%) of the now-accepted patches pass our strict fault-fixed check, since one fault class is resolved by commenting out the defect marker rather than removing it. Every accepted patch still clears both gates: no hallucinated citation and no unattested number gets through. The author and a second rater, both blind to condition, preferred PatchWrite on all sixteen PDF pairs sampled from the oracle corpus when the criterion is keeping lab-grounded facts intact (C1 Likert 5.0 vs. 2.0), and scored the prose nearly identically. Logs from 193 in-product drafting tasks show the same failure modes occurring in the wild.

Keywords automated scientific writing · LaTeX revision · compile gates · evidence locks · multi-agent systems

## 1 Introduction

Large language models can draft conference-shaped LAT<sub>E</sub>X [Song et al., 2026, Schmidgall et al., 2025, Yamada et al., 2025]. A remaining failure is non-monotonic revision: a reviewer asks to fix one undefined citation; the agent rewrites Results; the PDF still builds; a lab-grounded number has moved. Dashboards that only plot compile rate will celebrate the new PDF. The unit of mutation was the section, not the line.

This is one instance of a broader question the agent community is only now making a first-class research topic: not whether an agent’s output looks fluent, but whether it can be verified to satisfy a domain-specific correctness predicate before being trusted, the focus of an emerging line of work on agent reliability and verification (for instance the NeurIPS 2026 workshop “Who Verifies the Agents? Toward Reliable Agent Development,” https://verify-agents-workshop.github.io/, non-archival at the time of writing). This paper grounds that question in one narrow, high-stakes domain, LaTeX manuscript revision, where the correctness predicate is a compile gate plus an evidence lock rather than a hand-wavable quality score.

PaperOrchestra’s refinement loop historically emits a whole-slot fence. A slot is a named placeholder region of the venue template, for instance an entire Method or Results section, that the writer model regenerates in full each round rather than editing in place: compile failure is a continue, success replaces the whole slot [Song et al., 2026]. Agent Laboratory’s PaperSolver already uses EDIT N M with immediate pdflatex [Schmidgall et al., 2025], but on English ICLR-shaped shells (the outer document-class/template scaffold a paper is typeset into, as distinct from its content) and with LLM self-scores as quality. EasyPaper generates structured LAT X from metadata [EasyPaper, 2026] (software, not a peer-reviewed writing operator). We combine bounded interval editing, a compile gate that does not trust a nonstopmode PDF, cite/number locks, and PaperOrchestra’s venue shells into one invariant: thefile on disk is always the last compiling, evidence-clean manuscript.

PatchWrite sits in the refinement loop (Figure 1). The proposer sees numbered source and emits one EDIT N M. The runtime commits a compiling, evidence-clean T<sup>′</sup> or returns T. Illegal or over-long EDITs fall back to the legacy slot rewrite, so the operator is only as strong as the proposer’s interval. PatchWrite is deployed as the revision operator inside Solus (https://solus.xin), a research-writing workspace that formats drafts to a chosen venue’s template.

We ask five questions:

1. Does surgical EDIT keep unrelated tokens that slot rewrite mutates?

2. Does the compile gate reject a bad patch rather than a nonstopmode PDF?

3. Does the evidence gate catch a compiling patch that invents a cite key?

4. When the writer model itself proposes the EDIT instead of an oracle, how often is it legal, does it pass the gates, and does the accepted patch actually fix the fault?

5. When two readers compare the resulting PDFs, do they prefer the surgical draft if lab-grounded facts must not move?

Questions 1–3 are answered with scripted oracle repairs (24 mini-articles, eight faults spanning both compile-breaking and content-only defects, four conditions) so generation quality is not a confound. Question 4 is answered by rerunning the same 24×8 corpus with the writer model (qwen3.7-plus) proposing the patch itself, single call per job, no oracle (Section 4.2); this is exactly the measurement Questions 1–3 deliberately deferred by isolating the mechanism. Question 5 is answered with a dual-pass rater annotation of sixteen held-out PDF pairs from the oracle corpus (Section 4.3) Product logs from 193 drafting tasks motivate the gates but are not a substitute for either pass.

The contribution here is not a new edit grammar but a stricter, empirically validated acceptance predicate applied to an existing bounded-edit mechanism. Contributions: (i) a compile gate that scans the pdflatex log for fatals rather than trusting a nonstopmode PDF, applied to PaperSolver’s EDIT N M and rollback-onfailure mechanism (tex\_patch.py, ContentRefinementAgent); (ii) cite and numeric locks reused from PaperOrchestra, combined with (i) for the first time as a single acceptance criterion; (iii) an oracle stress protocol at $2 4 \times 8$ scale plus two ablations isolating each gate’s contribution, with table cells copied from evals/results/summary.json; (iv) a live-model companion at the same 24 × 8 scale across two independent writer models (qwen3.7-plus, Kimi K2.6) measuring parse, compile, gate, and fault-fixed rates for each model’s own EDIT proposals, including a one-line prompt fix that closes one model’s fallback gap while surfacing a new fault-fixed trade-off, cells copied from evals/results/llm\_edit\_summary.json and evals/results\_kimi/llm\_edit\_summary.json; (v) a sixteen-pair rater inspection with cells copied from raw\_materials/human\_eval.json.

## 2 Related Work

Tool use. PatchWrite’s EDIT N M is a narrow instance of the broader reasoning-plus-acting paradigm: a model interleaves generation with one constrained, well-typed action rather than free-form output [Yao et al., 2023], and can be taught to invoke that action only when it improves the outcome [Schick et al., 2023]. Neither line of work ties the action’s acceptance to a compile gate or an evidence lock; PatchWrite adds both as the acceptance criterion for the single tool it exposes.

Paper agents. PaperOrchestra maps idea plus experimental log onto venue templates [Song et al., 2026]: IEEE/arXiv shells, Chinese journal and thesis templates, OpenAlex citation policy, claim–evidence checks. Refinement historically rewrites a slot, not a line. The host workbench already runs those templates in production; PatchWrite replaces only the mutation operator inside ContentRefinementAgent. AI Scientist-v2 searches experiments then writes [Yamada et al., 2025]; we compare only the writing/revision module, not discovery or workshop acceptance. EasyPaper is a metadata-to-LAT X stack with optional typesetting [EasyPaper, 2026].

Table 1: Writing/revision modules only. “Compile gate”: failed pdflatex or a fatal log rolls back HEAD. “ZH shells”: does the system ship a Chinese-venue shell (defined in Section 2)? EasyPaper is listed as software.
<table><tr><td>System</td><td>Edit unit</td><td>Compile gate</td><td>Evidence lock</td><td>ZH shells</td></tr><tr><td>PaperOrchestra</td><td>slot fence</td><td>compile, no slot rollback</td><td>cite/claim</td><td>yes</td></tr><tr><td>PaperSolver</td><td>EDIT N M</td><td>yes</td><td>no (LLM score)</td><td>no</td></tr><tr><td>AI Scientist-v2 (writeup)</td><td>paper draft</td><td>typesetting</td><td>experiment search</td><td>no</td></tr><tr><td>EasyPaper (software)</td><td>section / paper</td><td>optional</td><td>citation tools</td><td>n/a</td></tr><tr><td>PatchWrite</td><td>EDIT N M≤40</td><td>yes + log scan</td><td>cite + numbers</td><td>surrogates</td></tr></table>

PaperSolver vs. PatchWrite. Agent Laboratory’s PaperSolver supplies the edit grammar we reuse (EDIT N M, compile, rollback) [Schmidgall et al., 2025]. The differences that matter for this paper are locks and templates, not the fence syntax. PatchWrite reuses bounded interval editing and rollback-on-failure from PaperSolver, but tightens the compile predicate — a fatal-log scan rather than treating any nonstopmode PDF as success — and adds an external evidence constraint (cite-key and numeric-token grounding against a registry and experimental log) to the acceptance criterion. Existing bounded edit + stricter compile predicate + evidence locks ⇒ validity-preserving acceptance. PaperSolver scores drafts with LLM reviewers and iterates on English ICLR-shaped shells. PatchWrite does not use an overall LLM score as a halt condition; it binds new \cite keys and empirical numbers to a registry and experimental log, scans the pdflatex log for fatals instead of treating a nonstopmode PDF as success, and runs on PaperOrchestra slot files (with Chinese-venue surrogates in the stress corpus). Table 1 summarizes writing/revision modules only.

Code editing. PatchWrite’s constrained patch generation is structurally the same problem SWE-bench poses for source code: given a defect description, produce a patch that resolves it against a fixed correctness signal [Jimenez et al., 2024]. SWE-agent’s finding, that a narrow, purpose-built edit interface outperforms raw file access for LLM patch generation [Yang et al., 2024], is the same design bet EDIT N M makes for LAT X manuscripts instead of a Git repository. The correctness signal differs: SWE-bench checks a held-out test suite, PatchWrite checks pdflatex plus a citation/number registry, so a head-to-head comparison across the two settings would need matched infrastructure we do not build here.

Concurrent work: PaperJury. PaperJury argues that “load-bearing safety and completion logic should reside in deterministic orchestration rather than model discretion” [Wang et al., 2026], the same HEAD-invariant instinct this paper builds around, reached independently and posted earlier in 2026. The target differs: PaperJury pre-submissionhardens human-authored CS papers through a due-process review–verdict–revise–verify loop with anchor-bounded edits and terminal outcomes (invalid-drop, valid-fixable, author-required), evaluated by expert review on held-out vision/NLP/ML papers against four baselines. PatchWrite sits earlier in the pipeline, inside LLM-drafted revision (PaperOrchestra’s refinement loop), with a narrower, more mechanical gate: one line-interval EDIT N M, a pdflatex log scan, and cite/number set-membership against a registry. There is no due-process trial and no author-required terminal state; in their place is an oracle-plus-live-model stress test (Section 4) rather than expert review. The two systems converge on a thesis and diverge on where in the authoring pipeline to enforce it and how heavy the gate should be. We have read only PaperJury’s public abstract, not run its benchmark, and not had it run on ours, so Table 1 deliberately leaves out a cell-by-cell row for it: a comparison built from an abstract alone would overstate what either side has verified about the other.

Long-form synthesis. STORM [Shao et al., 2024] and AutoSurvey [Wang et al., 2024] outline-then-write encyclopedic or survey text. They are not compile-gated LAT X editors and do not bind numeric tokens to a lab log. Their quality metrics (citation coverage, outline coherence) answer a different question than “did this revision move a number the injector never touched.”

Judges. Agent Laboratory showed automated reviewers over-score generated papers versus humans [Schmidgall et al., 2025]. We therefore report human preference on PDF pairs as a separate table (Section 4.3), and we do not use LLM overall-score as a halt condition.

## 3 Method

HEAD invariant. Let T be a compiling .tex file. PatchWrite replaces T only if every enabled gate returns true. Failed candidates leave the bytes on disk unchanged. Figure 1 is the control flow inside ContentRefinementAgent when use\_tex\_patch = True (disable with PATCHWRITE = 0).

PatchWrite in the PaperOrchestra refinement loop

![](images/51ea26231240aec5b3a2e1ab0195f7c6dd547436fbe169eb938d9f1b31f91467.jpg)  
Figure 1: PatchWrite in the PaperOrchestra refinement loop. An illegal or over-long EDIT falls back to legacy slot rewrite. Compile or evidence failure rolls back to HEAD; only both gates passing commits $T ^ { \prime }$

Numbered view. The proposer sees 1-based NNNN | text (number\_lines) and may emit one fence

‘‘‘EDIT N M   
replacement lines   
11

with $1 \leq N \leq M$ and $M - N + 1 \leq 4 0$ . The interval is replaced by the body (apply\_edit). Over-long, empty, non-integer, or past-EOF ranges are rejected before compile. Bare EDIT N M is also parsed, matching PaperSolver [Schmidgall et al., 2025].

Runtime (try\_patch). The runtime takes two inputs: the current compiling document T, and cmd, a candidate edit string supplied by a model or an oracle. It proceeds as follows.

1. Parse one EDIT block (the function parse\_edit). On failure, return T unchanged.

2. Apply the closed-interval replacement (apply\_edit) to obtain a candidate document $T ^ { \prime }$

3. If the compile gate is on: write a unique jobname, delete stale .aux/.pdf/.log files, and run PaperOrchestra’s compile\_latex. Accept only a PDF larger than 64 bytes whose log contains none of emergency stop, fatal error, ! latex error, or a forced-fail marker. pdflatex -interaction=nonstopmode often still writes a PDF after a syntax error, so “a PDF exists” is not by itself a passing compile gate.

4. If the evidence gate is on: a cite key is allowed if it appears in the bibliography, the citation map, or HEAD itself; any other key is flagged unknown\_cite:<sub>\*</sub>. Every empirical number is checked against the experimental log. A violation of either check rolls the candidate back.

5. On success, commit $T ^ { \prime }$ as the new HEAD; on any failure, HEAD stays T.

Fallback. If no legal EDIT is parsed or a gate fails, the loop falls back to slot-fence rewrite so jobs still finish. That fallback re-introduces whole-slot mutation; production logs should count it. Whole-slot REPLACE remains a human-explicit escape hatch. Line numbers are regenerated each round after a committed insert; otherwise N, M drift.

Worked example.

9 |The encoder uses 12 layers and a frozen tokenizer\~\cite{smith2020}.

Table 2: Oracle revision stress (24 × 8 faults = 192 per condition). Jaccard vs pre-fault HEAD. The 0.00 vs. 1.00 layer split is by construction of the oracles: it tests whether gates preserve HEAD, not whether an LLM finds the patch.
<table><tr><td>Condition</td><td>n</td><td>Acc.</td><td>Comp.</td><td>Gate</td><td>Cite J.</td><td>Num. J.</td><td>Layers</td></tr><tr><td>PO-slot rewrite</td><td>192</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.6667</td><td>0.00</td></tr><tr><td>PatchWrite</td><td>192</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>PW, no compile gate</td><td>192</td><td>0.00</td><td>0.00</td><td></td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>PW, no evidence gate</td><td>192</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.6667</td><td>1.00</td><td>1.00</td></tr></table>

10 |\section{Results}  
11 |Accuracy was 91.4 compared with the baseline 68.2\~\cite{lee2019}.  
A fake\_cite injector rewrites line 11 to \cite{fake2024}. The oracle emits EDIT 11 11 with the origina sentence. Slot rewrite also restores a valid Results line, but it also changes line 9 to “16 layers.” Both PDFs build; only one is monotonic (Figure 4).

## 4 Experiments

## 4.1 Oracle revision stress

Twenty-four mini-articles make up the corpus (16 English, 8 Chinese-venue pdflatex surrogates with ASCII titles). Each HEAD contains “The encoder uses 12 layers . . . ” and Results numbers drawn from a per-document log. Eight scripted faults split into two classes. Content-class faults leave the broken HEAD compiling: an undefined \cite{fake2024}, a 99.9 absent from the log, a TODO line, and swapped accuracy/baseline numbers. Compileclass faults break pdflatex outright: an unclosed \$, an unmatched brace next to a \cite, an unclosed quote environment, and an undefined control sequence. Four conditions times 192 copies gives 768 jobs.

• po\_slot: scripted whole-block rewrite that does invert the fault and also rewrites “12 layers”→“16 layers”. This is a PaperOrchestra-style slot mutation, not a live LLM rewrite of a conference paper.

• patchwrite: one-line EDIT with both gates on.

• patchwrite\_nocompile: replacement \errmessage{patchwrite-forced-fail}.

• patchwrite\_nogate: surgical fix plus \cite{hallucinated2024}, evidence gate off.

Metrics: accept, compile\_ok, gate\_ok, cite/numeric Jaccard vs pre-fault HEAD, layers\_preserved. Table 2 cells match evals/results/summary.json after wiping evals/results/work.

![](images/46992201ac4f954c2b0b2aa195ce7107e80918656bfd859cdf40f439c79bb22d.jpg)

![](images/79c909382bf70b7174444fae05b85f57f26be865307b137632f802d268bfbc65.jpg)

![](images/050195d77532369d26f3e92977dfb1ea6e69fa07f87e1a814da7316375587794.jpg)  
Figure 2: Oracle stress (n = 192 per bar). Left: accept rate. Middle: numeric Jaccard vs pre-fault HEAD. Right: whether the unrelated “12 layers” sentence survives. Compile-off never replaces HEAD; evidence-off keeps layers but drops cite Jaccard (not plotted here; Table 2).

What Table 2 shows. Slot rewrite compiles at a 1.00 rate while layers\_preserved sits at 0.00. The 0.6667 numeric Jaccard is exactly the 12 → 16 token swap. Compile-off never replaces HEAD, so accept stays at 0. Evidence-off keeps the layers line but drops cite Jaccard to $2 / 3 ,$ once smith2020 and lee2019 are joined by hallucinated2024. The pattern is identical on every one of the eight faults (n = 24 per fault-condition cell,

Figure 3), on the content-class/compile-class split (n = 96 per condition each), and on the EN/ZH subsets (n = 128/64 per condition); ZH here means a shared pdflatex engine, not ctex camera copies. Compile-class faults are mechanically the harder case for the oracle, since the broken HEAD does not build at all, so patchwrite has to restore compilation and stay surgical at the same time. It still lands at 1.00 on every preservation metric there. Mean latexmk wall time is ≈ 0.22 s in every condition (Appendix C); latency is not the claim.

Unrelated 12-layer line kept, by fault (n = 24 papers each)  
![](images/1bd52b5e8a8f87868126ef4fb882b16e6148aac7c099203853d814ade4f30d65.jpg)  
Figure 3: layers\_preserved broken out by fault $( n ~ = ~ 2 4$ per bar). PO-slot is 0.00 and PatchWrite is 1.00 on all eight faults, including the four compile-class faults added to widen the corpus beyond the original fake\_cite/bad\_number/broken\_math/todo set.

Case: en\_vision\_01 / bad\_number. Figure 4 shows the two compiling PDFs after the injector wrote “Accuracy was 99.9”. Both oracles restore 91.4. Only the slot rewrite also rewrites the Method sentence. A compile-rate dashboard cannot tell them apart.

![](images/93d331f435f4de55d53ed0091fcf14c8b7372c02ccbbbf1395d6c7373da8c003.jpg)  
Figure 4: Side-by-side source after repairing bad\_number on en\_vision\_01. Both PDFs build. PO-slot changes encoder depth 12 → 16; PatchWrite EDIT 11 11 does not.

## 4.2 Live-model EDIT proposals

Table 2 measures the operator with the correct patch already known; it says nothing about how often a real writer model finds a legal patch on its own. We reran the same $2 4 \times 8$ corpus with the oracle replaced by one live call to the project’s own writer model (Alibaba Cloud DashScope, model id qwen3.7-plus). Given the numbered broken source, the experimental log, and a one-line reviewer or compiler note describing the defect (not the fix), the model has to emit its own EDIT N M, single-shot. The prompt follows the same numbered-source-plussingle-EDIT-fence contract as ContentRefinementAgent.\_revise\_via\_tex\_patch’s production prompt, though it uses a standalone system prompt and a synthetic one-line defect note rather than a live reviewer pass and the full production system instruction. So this measures the EDIT-fence contract under realistic constraints, not a byte-identical replay of production. The resulting patch runs through the same compile and evidence gates as

Table 3: Live-model EDIT proposals, n = 192 (same corpus as Table 2, oracle replaced by qwen3.7-plus). fault\_fixed and layers\_preserved are conditioned on accepted.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Parse ok (legal ED IT emitted)</td><td>0.75</td></tr><tr><td>Compile ok, of parsed</td><td>1.00</td></tr><tr><td>Gate ok, of compiled</td><td>1.00</td></tr><tr><td>Accept rate</td><td>0.75</td></tr><tr><td>Fallback rate</td><td>0.25</td></tr><tr><td>Fault fixed, of accepted Layers preserved, of accepted</td><td>0.9375 0.9444</td></tr></table>

Table 2. Cells: evals/results/llm\_edit\_summary.json / llm\_edit\_slices.json (n = 192; code llm\_edit\_stress.py, https://github.com/Baiang/editnm).

One thing shapes how to read every number below: the OpenAI-compatible DashScope call path strips any temperature parameter, so each row is a single stochastic draw at the server default rather than a repeated or averaged measurement (unlike Table 2’s oracle cells, which are bit-identical on rerun). The clean fault-level split (six faults at 1.00/1.00, two at 0.00/0.00) reads as systematic per-fault behavior rather than a coin flip, but that is a reading, not a confirmed replication. Scope and caveats for this section are collected once, in full, in Section 6.

![](images/e8196eb839a0cbbdf468459d50cc1df2608905a1d554ac95894ca7a424239f8e.jpg)

![](images/de608dd167a0b915ded48b5ca964a8f1f54a8a7e3f7bfe0ceed200889760c723.jpg)  
Figure 5: Left: the six headline rates from Table 3. Right: parse and accept rate by fault (n = 24 each). todo and undefined\_cmd are 0.00 on both; the other six faults are 1.00 on both.

What Table 3 shows. Six of the eight faults, fake\_cite, bad\_number, broken\_math, unmatched\_brace, unclosed\_env, and swapped\_acc, parse and get accepted every single time. Every parsed patch that reaches the gates also compiles and clears the evidence gate 100% of the time (gate\_ok = 1.00 of compiled), matching zero hallucinated cites or unattested numbers. The 25% overall fallback rate is not spread across the eight faults at all; it belongs entirely to the other two. todo and undefined\_cmd fall back 24/24 times each, for exactly the same reason, and none of the remaining six faults contributes a single fallback job: 48 of the 48 fallback jobs are todo or undefined\_cmd, and 0 of the remaining 144 jobs fall back.

The reason is specific enough to name. The model’s instinct is to delete the injected line, so it submits EDIT N N with an empty body (‘‘‘EDIT 10 10\n‘‘‘). parse\_edit rejects an empty replacement (empty\_replacement), because the grammar has no delete verb, only replace-with-at-least-one-line, and a one-line reviewer note never tells the model the workaround the oracle uses: replace the line with a % comment. That single gap accounts for the entire observed fallback mass, all 48 of the 48/192 fallback jobs, not merely a large share of it. It reads as a fixable prompt-contract issue rather than evidence that gate architecture explains fallback in general. (broken\_math is accepted every time; its shortfall shows up in fault\_fixed instead, discussed next.)

fault\_fixed of accepted is 0.9375, below 1.00 for two distinct reasons visible in the per-fault cells. broken\_math accepts 24/24 but only fixes 16/24: the miss rewrites The encoder uses 12 layers to The encoder uses \$12\$ layers instead of removing the stray \$. That still compiles and still passes the evidence gate, since the numeric token 12 is unchanged, but it does not byte-match the original prose. It is a compiling, gate-clean patch that is nonetheless not the identity repair. fake\_cite fixes 23/24; the miss replaces \cite{fake2024} with \cite{smith2020}, a real key defined in this file’s own bibliography, so the evidence gate accepts it. But smith2020 is the Method-encoder citation, not the correct key for this Results-accuracy sentence. That is the evidence gate’s real boundary, stated with an example rather than an assertion: it checks that a cite key is attested, not that it is the correct attested key for this specific claim.

Mean latency rises from ≈ 0.22 s (Table 2, compile only) to ≈ 1.38 s (LLM call plus compile). The model call now dominates wall time, not the gate. Section 5 returns to what this result does and does not license us to claim.

Table 3b: Live-model EDIT proposals, second model (Kimi K2.6, $n = 1 9 2 ;$ same corpus, prompt-fixed harness).
<table><tr><td>Metric</td><td>Qwen3.7-plus (prompt-fixed)</td><td>Kimi K2.6</td></tr><tr><td>Parse ok</td><td>1.00</td><td>0.875</td></tr><tr><td>Compile ok, of parsed</td><td>1.00</td><td>1.00</td></tr><tr><td>Gate ok, of compiled</td><td>1.00</td><td>1.00</td></tr><tr><td>Accept rate</td><td>1.00</td><td>0.875</td></tr><tr><td>Fallback rate</td><td>0.00</td><td>0.125</td></tr><tr><td>Fault fixed, of accepted</td><td>0.849</td><td>0.9107</td></tr><tr><td>Layers preserved, of accepted</td><td>0.9792</td><td>0.9167</td></tr><tr><td>Mean latency (s)</td><td>1.5651</td><td>26.87</td></tr></table>

## 4.3 Rater inspection of sixteen PDF pairs

Oracle metrics do not say whether a reader would treat the two compiling PDFs as equally submission-ready. We therefore annotated sixteen pairs drawn from the stress work directory: en\_vision\_01 and zh\_med\_09 × all eight faults, up from four in an earlier four-fault pass (protocol and cells: raw\_materials/human\_eval.json). The author and a second rater scored each pair with condition names hidden (sheets labeled A/B). Rubric:

• C1 factual (1–5). Consistency with HEAD: encoder depth 12 and log-attested accuracy.

• C2 defect repaired (1–5). The injected fault is gone.

• C3 stub prose (1–5). Readability of the mini-article (both drafts are stubs).

• Preference. Which PDF is more submission-ready if lab-grounded facts must not move.

This is a designed rater dual-pass on real oracle PDFs, not a multi-PhD panel and not LLM-as-judge, and it is disclosed at that scale rather than dressed up as a validated study: standard guidance for human evaluation of generated text treats a two-rater, non-independent pool as exploratory rather than confirmatory [van der Lee et al., 2021], and raw preference agreement (here 16/16) is not a substitute for a chance-corrected agreement statistic computed over a larger, independent rater pool [Artstein and Poesio, 2008]. Likert means for C1/C2 follow the visible 12 → 16 rewrite (every PO-slot file) versus its absence (every PatchWrite file). C3 is near-tied by construction of the stub corpus; one annotator marked a single PatchWrite stub one point lower on prose, the same single disagreement as the earlier eight-pair pass, now diluted over twice as many pairs (3.94 → 3.97).

![](images/7afff07b33b8cf00ac69903fb51b3536167aa14bb944219ca11f348dcfdd1948.jpg)

![](images/9d5349f37b43b52e8af23127ba5c3c7fd309a563278dcd3c9e7fc34b3b6d8e25.jpg)  
Figure 6: Left: pairwise preference (n = 16). Right: mean Likert. Both oracles repair the injected defect (C2); readers split on facts (C1) and therefore on preference. Prose of these stubs is a near-tie.

Table 4: Rater dual-pass on sixteen oracle PDF pairs. Preference agreement 16/16. Likert is 1–5. Cells match human\_eval.json. Per-pair rows: Appendix E. Exploratory two-rater pool (author plus one unpaid rater; Section 4.3), not a validated human-evaluation study.
<table><tr><td></td><td>PO-slot</td><td>PatchWrite</td></tr><tr><td>Preferred as submission-ready (pairs)</td><td>0/16</td><td>16/16</td></tr><tr><td>Mean factual errors listed per PDF</td><td>1.00</td><td>0.00</td></tr><tr><td>C1 factual (mean Likert)</td><td>2.00</td><td>5.00</td></tr><tr><td>C2 defect repaired (mean Likert)</td><td>5.00</td><td>5.00</td></tr><tr><td>C3 stub prose (mean Likert)</td><td>4.00</td><td>3.97</td></tr></table>

Table 5: In-product drafting logs (not a 6–8 pair blind SxS). Counts from task artifacts and the paper-task database.
<table><tr><td>Signal</td><td>Denominator</td><td>Count</td></tr><tr><td>Paper tasks / revisions</td><td></td><td>193 / 240</td></tr><tr><td>Evidence-gate fail (unsupported table number)</td><td>157 snapshots</td><td>45 fail (118 issues)</td></tr><tr><td>Unmatched TeX braces</td><td>156 sanity files</td><td>40</td></tr><tr><td>Orphan cite tokens</td><td>156 sanity files</td><td>7</td></tr><tr><td>Share-link notes (uncoded)</td><td>16 links</td><td>15 items</td></tr></table>

What Table 4 shows. C2 being tied at 5.0 is the interesting negative: slot rewrite does fix the local bug, which is why compile rate alone is a bad halt condition. C1 and preference isolate the silent 12 → 16 change. These sixteen pairs measure whether a reader notices a monotonicity failure that automatic Jaccard already flags; scope and caveats for this annotation are collected in Section 6.

## 4.4 In-the-wild workbench logs

These are not operator-level blind SxS. They are product artifacts from the host writing workbench (Postgres task store plus per-task JSON), exported 2026-08-23 (Table 5; Figure 7; raw\_materials/in\_the\_wild.json). The author is affiliated with Solus (https://solus.xin), the platform these logs are drawn from and where PatchWrite is deployed as its revision operator; see Section 7 for the full conflict-of-interest statement. 193 paper tasks and 240 revisions were in the store. Among 157 evidence-gate snapshots, 45 fail (28.7%); the 118 recorded issues are all table\_number\_not\_in\_log. Among 156 LAT X sanity snapshots, 40 files have unmatched braces and 7 orphan cite tokens appear. Nine coach reviews (Solus’s structured, per-round reviewer-feedback pass) flag placeholders, compile problems, truncated quotes, or claim–evidence gaps. Fifteen share-link comments (freeform comments left by viewers of a paper’s shareable preview link, a separate product feature from the coach) exist on 16 links; they are UI-test strings and are not coded as pairwise preference. The logs show that unattested numbers and broken TeX already occur in real drafting jobs, the same classes the oracle injects, without any claim that users preferred PatchWrite PDFs in the product UI.

In-product drafting logs (193 tasks; not operator SxS)  
![](images/83c103ddf3c3a9fb79be4d7260748af42cdb32fd863ab5e580e15c99d09d27ff.jpg)  
Figure 7: Rates from in-product snapshots. Evidence-gate failures and unmatched braces are common; orphan cites are rarer. Coach foci $( 7 / 9$ reviews) are a small, separately labeled set.

## 5 Analysis

False fluency. Compile rate = 1.00 is what a dashboard would celebrate. The regression is in a line the injector never touched. PatchWrite’s result is the same compile rate with layers preserved. The sixteen-pair inspection is the reader-facing restatement of that split: C2 tied, C1 and preference not tied.

Widening the fault set does not change the story. We doubled the corpus (12 to 24 manuscripts) and doubled the fault set (4 to 8) in two directions rather than one. Three of the four additions, unmatched\_brace, unclosed\_env, and undefined\_cmd, are compile-class: the broken HEAD fails pdflatex, so the oracle repair has to do more work than a content-only fix, restoring compilation rather than merely restoring a token. The fourth, swapped\_acc, is a second content-class fault that exercises the numeric-Jaccard metric differently from bad\_number, since two tokens swap instead of one being replaced. None of the four new faults moves any rate in Table 2 or Figure 3: layers\_preserved stays 0.00/1.00 and accept\_rate follows the same per-condition pattern on every one of the eight faults taken separately (n = 24 each). The compile-class faults are mechanically harder, since the pre-repair file will not build at all, yet they show exactly the same split as the original four content-class faults. That is evidence the result comes from the gate architecture, not from which particular injector happened to be scripted first. We read this as ruling out one specific threat to validity, that four hand-picked faults happened to favor PatchWrite, not as a claim that fault diversity is now exhausted. Section 6 still flags that all faults are scripted rather than adversarially searched.

The nonstopmode trap. If the predicate were “PDF exists,” patchwrite\_nocompile would accept: pdflatex still writes a file after \errmessage. Scanning the log is the difference between a compile gate and a file-existence gate. Production wrappers that call latexmk in nonstopmode and then glob for .pdf will silently promote broken math into HEAD.

Fallback is the real production risk. Oracle jobs always emit a legal short EDIT, so they never take the slot-fallback branch. A model that dumps Results into EDIT 1 80 is rejected on span and gets slot-rewritten instead, which is the regression documented in Table 2. Table 3 now reports a live-LLM fallback rate rather than deferring it: 25% of 192 jobs, concentrated rather than diffuse. todo and undefined\_cmd fall back every single time (24/24 each) because the model tries to delete the injected line with an empty EDIT body, and the grammar has no delete verb. That single, named gap is the entire observed fallback mass, all 48 of the 48 fallback jobs; the other six faults contribute zero. Two mitigations belong in the runtime rather than the prompt. First, count fallbacks per job and surface them in the UI, broken out by cause (parse vs. compile vs. gate), the way Table 3 does. Second, regenerate numbered source after every committed insert so subsequent N, M stay aligned. A third, cheaper fix follows directly from Section 4.2: tell the proposer explicitly that deletion means replacing with a comment, not replacing with nothing. We reran Section 4.2’s live-LLM suite with one added sentence in the todo and undefined\_cmd defect notes, telling the proposer that deletion means replacing the line with a % comment rather than leaving it empty. The overall accept rate moved from 0.75 to 1.00 and fallback dropped from 25% to 0%, closing the gap the abstract originally reported; todo alone went from 0/24 to a clean 24/24 both accepted and fixed. But fault\_fixed\_rate\_of\_accepted moved the other way overall, from 0.9375 to 0.849: all 24 accepted undefined\_cmd patches compile clean and pass the evidence gate, yet none satisfies our string-presence check for “fixed,” because the model deleted by commenting out % \patchwriteundefined rather than replacing it with the oracle’s clean % removed; the same defect signature survives in the text, inert but present. This is the same false-fluency pattern flagged earlier in this section for compile rate: closing the accept-rate gap surfaced a second gap our own fault\_fixed metric was designed to catch.

A second model, a different seam. A second model surfaces the same shape of failure, at a different fault. We reran the prompt-fixed harness (same corpus, gates, and defect notes as the fix above, unchanged) against a second writer model, Kimi K2.6 (kimi-k2.6, via Moonshot’s OpenAI-compatible API; Table 3b). It accepts 87.5% of proposals and, of those, fixes the fault 91.1% of the time, preserving layers 91.7% of the time; mean latency is 26.9 s per call (reasoning-heavy generation), against Qwen3.7-plus’s much lower per-call latency. Fallback is again concentrated rather than diffuse, but at a different fault than before: 23 of the 24 fallback jobs are unclosed\_env, a fault the prompt fix’s two added sentences never mentioned, and on which Qwen3.7-plus never fell back even before that fix. The one-line prompt fix that closed Qwen’s gap is therefore a fix for the specific fault types one model happened to fail on, not a general solution to the grammar’s missing delete verb; a different model exposes the same underlying gap at a different seam. A second, independent near-miss replicates the earlier fault\_fixed boundary: broken\_math accepts 23 of 24 but fixes only 9, because Kimi sometimes writes “\$12\$ layers” instead of “12 layers”, compiling and gate-clean, but not a byte match, the same class of miss as Qwen’s undefined\_cmd comment-out. Two models independently hitting the same metric boundary is stronger evidence that the boundary is in fault\_fixed’s strict string check, not particular to one model’s phrasing habits. Unlike DashScope’s silent parameter-stripping, Moonshot’s

API explicitly rejects any temperature other than 1.0 for kimi-k2.6, so variance here can only be estimated from repeated draws at that fixed setting, not from a pinned low temperature.

Stale aux. An early eval run failed until evals/results/work was wiped; latexmk reused .aux across colliding jobnames. Unique jobnames plus deleting stale auxiliaries are part of the compile gate, not an implementation footnote.

Evidence-gate conservatism. In-product snapshots fail most often on table\_number\_not\_in\_log (118 issues across 45 failing files). That is a true positive if the number is invented, and a false positive if the log is incomplete. PatchWrite will refuse a compiling patch in the second case. Authors who want the number in the PDF must first put it in the log; that is the intended workflow, not a compile workaround.

Attested is not correct. The fake\_cite miss in Section 4.2 is a real example of the evidence gate’s boundary, not a hypothetical one: run\_evidence\_gates checks set membership, whether a key is defined anywhere in the document, which a citation log can attest cheaply. It cannot check that a key is the semantically correct citation for a specific sentence, which would need a much richer claim-to-source index than tex\_patch.py builds today. It is also a different, cheaper problem than external citation-hallucination detection, which grounds a cite against real scholarly metadata such as CrossRef, Semantic Scholar, and OpenAlex, rather than a document’s own bibliography [Khajavi et al., 2026], or checks that a whole .bib entry’s fields (author, venue, year) match the real record rather than just its key [Rao and Callison-Burch, 2026]. PatchWrite’s gate would accept a citation to a real paper’s key even if that paper does not say what the sentence claims, and it would accept a fabricated-but-locally-defined .bib entry; both are out of scope for a set-membership check and would need exactly that kind of external grounding.

What the human table is not. Given the protocol in Section 4.3, sixteen oracle pairs will always prefer the draft that keeps “12 layers” if that is the rubric. Table 4 checks that the Jaccard split is visible to readers; it does not measure venue-level prose. A live-LLM SxS judging the model’s own accepted-but-imperfect patches from Section 4.2 (the broken\_math and fake\_cite misses would be a pointed test case) remains future work; scope for this section is collected in Section 6.

## 6 Limitations

• Table 2 is oracles, not writers, by design. It assumes the correct patch is known so the gates can be isolated from generation quality. Table 3/3b are the writer-model companions: two models (qwen3.7-plus, Kimi K2.6), each single-shot on this same synthetic corpus, no repeats for either model. The two models concentrate their fallback on different fault types (todo/undefined\_cmd for Qwen before its prompt fix, unclosed\_env for Kimi), suggesting the prompt-level fix is reactive rather than general. Read this as two data points, not a general LLM-proposed-EDIT success rate.

• Short files. Mini articles, not 8–10 page venue PDFs. Full IEEE/ctex templates exist in the host tree and were not this table’s compile target.

• Chinese surrogates share pdflatex with English; they are not camera ctex theses.

• Scripted faults, not adversarial ones. Eight faults across two classes (content-class, compile-class) rule out a four-fault coincidence, but all eight are hand-written injectors, not a search over the space of possible defects or faults sampled from a live LLM’s actual error distribution.

• Annotation scale. Two raters × sixteen pairs (Section 4.3), judging oracle PDFs, not the live model’s own (sometimes imperfect) patches from Section 4.2. This is what the end of Section 4.3 and “What the human table is not” (Section 5) both point to. Product share-link comments were UI tests and were not recoded as preference.

• Fallback measured once, not tracked in production. Section 4.2 gives a live-LLM fallback rate for the first time (25% before the one-line prompt fix in Section 5 closes it to 0%, one model, this corpus), but the runtime does not yet log per-cause fallback rate (parse vs. compile vs. gate) on real drafting jobs the way Table 3 does on the oracle corpus.

## 7 Ethics

PatchWrite does not create experimental facts. Generating a domain paper without a lab log, or inventing citations, is misconduct. A compile gate can still ship a fluent falsehood if the evidence gate is off or the log is fabricated. Authors remain responsible for every claim. The system design and experimental protocol are the author’s own; LLM assistance was used, under the author’s direction, to help write and debug the implementation (tex\_patch.py, revision\_stress.py, llm\_edit\_stress.py, corpus.py) and to draft parts of this manuscript. Table cells are copied from summary.json, llm\_edit\_summary.json, in\_the\_wild.json, and human\_eval.json. Table 3 additionally uses an LLM as the object of measurement, not just an assistant: every cell there is the recorded behavior of one live call to qwen3.7-plus per row, not a hand-edited or cherry-picked transcript. The human table is rater inspection of artifacts we generated, disclosed as such; it is not a hired-annotator study. The second rater (A2) is a friend of the author, not a co-author and not compensated; both raters scored blind to which system produced each PDF (Section 4.3). Conflict of interest: the author is affiliated with Solus (https://solus.xin), a research-writing platform; PatchWrite is deployed there as its revision operator, and the in-product logs in Section 4.4 are drawn from that platform.

## 8 Conclusion

Writing agents should not treat “the section compiled” as “the paper did not change underfoot.” PatchWrite makes the mutation a bounded interval and compile-plus-evidence the only path from HEAD to HEAD. On a 24 × 8 oracle stress test spanning content-class and compile-class faults, that is the difference between layers\_preserved = 0.00 and 1.00, reproduced identically at 4× the scale of an earlier four-fault pass. Replacing the oracle with the writer model itself on the same 192 jobs shows the gates hold under real generation too: every parsed, compiling patch passes the evidence gate; the 25% fallback rate traced to one named prompt gap and closed to 0% once the proposer is told explicitly to comment out rather than empty a deleted line, though that same fix leaves 84.9% (down from 93.75%) of now-accepted patches meeting our strict fault-fixed check rather than open-ended unreliability. On sixteen PDF pairs, readers prefer the surgical draft 16/16 when facts must not move, while both drafts repair the local defect. In-product logs show unattested numbers and broken TeX already appearing in real jobs. The next measurement is LLM-proposed EDITs on full venue-length templates with multiple models, not this mini-article corpus with one model, together with a live-LLM comparison judging the model’s own accepted-but-imperfect patches rather than just the oracle’s.

Beyond LaTeX manuscripts. Nothing in the argument above is specific to LAT X. Strip away pdflatex and the citation registry and what is left is a three-part pattern: a bounded edit (a small, syntactically checkable change rather than a free-form rewrite), a domain-specific validity predicate (a mechanical pass/fail check, not a fluency score), and rollback to the last-known-good state on any failure. The same shape describes a code-review agent that edits one function and runs the test suite before keeping the change, a contract- or compliance-document assistant that edits one clause and runs a compliance checker before keeping the change, or a database-schema-migration agent that edits one migration step and runs a schema check before keeping the change. We have not built or evaluated any of these; the claim here is only that the pattern transfers in principle, not that we have shown it works. Whether bounded-edit-plus-predicate-plus-rollback holds up once pdflatex is replaced by a test suite, a compliance checker, or a migration linter, each with its own failure modes and cost of a false accept, is an open question this paper does not answer.

Reproducibility. The mechanism (tex\_patch.py) and both eval harnesses (revision\_stress.py, llm\_edit\_stress.py), plus the 24-manuscript corpus (corpus.py), are public under the MIT license: https://github.com/Baiang/editnm. This is a standalone extraction with no dependency on the private codebase PatchWrite ships inside; see that repository’s README for the two components (the compile step and the numeric-evidence check) that were reimplemented clean-room rather than copied over. Tests: pytest tests/ (network-free). Oracle eval (Table 2): python revision\_stress.py -out results. Live-model eval (Table 3; calls qwen3.7-plus via DashScope, needs DASHSCOPE\_API\_KEY, real billed calls): python llm\_edit\_stress.py -out results. Second-model eval (Table 3b; calls Kimi K2.6 via Moonshot’s OpenAI-compatible API, real billed calls): OPENAI\_API\_KEY=<key> OPENAI\_BASE\_URL=https://api.moonshot.cn/v1 python llm\_edit\_stress.py -model kimi-k2.6 -max-tokens 3000 -out results\_kimi (the script’s client is OpenAI-compatible-endpointgeneric, not Moonshot-specific; -max-tokens is raised because kimi-k2.6 spends most of a small budget on hidden reasoning tokens before any visible completion, unlike qwen3.7-plus). Figure generation, the human-annotation sheet, and the in-the-wild export (Table 5) use internal tooling not included in that repository. PatchWrite runs in production as part of Solus (https://solus.xin).

## A Corpus titles

Generated by corpus.py (https://github.com/Baiang/editnm). English (16): Patch-Level Robustness in Vision Encoders; Instruction Tuning Without Answer Leakage; Streaming ASR with Delayed Token Loss; Two-Tower Retrieval with Hard Negatives; Message Passing with Residual Gates; Adaptive Clipping for Noisy Gradients; Offline

RL with Conservative Q Backup; Variant Effect Prediction on Sparse Panels; Dense Captioning with Region-Level Losses; Non-Autoregressive Translation with Latent Alignment; Self-Supervised Speech Tokens for Speaker ID; Graph Networks for Reaction Yield Prediction; Long-Context Retrieval for Case Law; Satellite Time Series for Crop Mapping; Gaze-Aware Interface Adaptation; Membership Inference on Fine-Tuned LLMs. Chinese-venue surrogates (8, ASCII titles): Variant Effect Prediction (Chinese venue surrogate); Retrieval Augmented Classroom QA (Chinese venue surrogate); Contrastive Pretraining for Industrial Time Series (CN surrogate); Compile-Gated Scholarly Revision (Chinese thesis surrogate); Legal Judgment Prediction (Chinese venue surrogate); Crop Mapping from Satellite Series (CN surrogate); Contrastive Learning for Credit Risk (CN surrogate); Compile-Gated Editing for CJK Theses (CN thesis surrogate). The last eight EN and last four ZH titles were added to widen the corpus from the original 8 EN / 4 ZH. Cite keys smith2020, lee2019 are fixtures.

## B Fault recipes

Implemented in revision\_stress.py (no LLM), eight total. Content-class (broken HEAD still compiles): fake\_cite: \cite{lee2019}→\cite{fake2024}. bad\_number: Accuracy token → 99.9. todo: a TODO line before Results. swapped\_acc: the accuracy and baseline numbers trade places in the Results sentence. Compile-class (broken HEAD fails pdflatex): broken\_math: unclosed \$. unmatched\_brace: an extra { inserted next to a \cite. unclosed\_env: an unterminated \begin{quote} inserted before Results. undefined\_cmd: an undefined control sequence (\patchwriteundefined) inserted before Results. Oracle PatchWrite inverts only the span in every case. Oracle po\_slot also rewrites “12 layers” to “16 layers” regardless of which fault it is repairing.

## C Latency

Mean wall time including latexmk (same 192-job cells as Table 2): PO-slot 0.2236 s; PatchWrite 0.2209 s; no compile gate 0.2177 s; no evidence gate 0.2185 s. All conditions invoke latexmk; latency is not the claim. These are wall-clock numbers from one machine under ordinary load, not a controlled benchmark. A repeat run of this same suite measured 0.2100/0.2095/0.2092/0.2116 s, and the original 48-job four-fault pass measured ≈ 0.23 s; all three runs agree to within ±0.01–0.02 s and none changes the ordering or the qualitative claim (compile dominates; the gates add negligible overhead).

## D Per-fault uniformity

Each fault × condition cell has n = 24 (up from n = 12 in the original four-fault pass). For every one of the eight faults (fake\_cite, bad\_number, todo, swapped\_acc, broken\_math, unmatched\_brace, unclosed\_env, undefined\_cmd), po\_slot has accept = 1.00, layers = 0.00, numeric Jaccard = 0.6667; patchwrite has all preservation metrics = 1.00; patchwrite\_nocompile has accept = 0.00; patchwrite\_nogate has cite Jaccard = 0.6667 (Figure 3). Content-class and compile-class faults each aggregate to n = 96 per condition with the identical pattern. EN 128 jobs, ZH 64 jobs per condition; same pattern on both subsets. Source: evals/results/slices.json.

## E Per-pair rater annotation

Sixteen pairs (up from eight); two raters (A1, A2); preference always PatchWrite. Every PO-slot PDF lists one factual error (encoder depth 12 → 16). C3 is stub-prose Likert; A2 scored zh\_med\_09/todo PatchWrite prose as 3 rather than 4 (the sole disagreement, unchanged from the original eight-pair sheet).

## References

Ron Artstein and Massimo Poesio. Inter-coder agreement for computational linguistics. Computational Linguistics, 34 (4):555–596, 2008.

EasyPaper. EasyPaper: A multi-agent academic paper generation system. https://github.com/ PinkGranite/EasyPaper, 2026. Software repository.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024.

Table 6: Per-pair sheet. Pref. = PatchWrite. Factual errs. counted vs HEAD.
<table><tr><td>Paper</td><td>Fault</td><td>Errs PO / PW</td><td>C1 PO / PW</td><td>C2 both</td><td>Pref. A1/A2</td></tr><tr><td>en_vision_01</td><td>fake_cite</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>bad_number</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>broken_math</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>todo</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>unmatched_brace</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>unclosed_env</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>undefined_cmd</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>en_vision_01</td><td>swapped_acc</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>fake_cite</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>bad_number</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>broken_math</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>todo</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>unmatched_brace</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>unclosed_env</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>undefined_cmd</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr><tr><td>zh_med_09</td><td>swapped_acc</td><td>1/0</td><td>2/5</td><td>5</td><td>PW / PW</td></tr></table>

Khashayar Khajavi, Shaghayegh Sadeghi, Rise Adhikari, and Alexander Tessier. CiteCheck: Retrieval-grounded detection of LLM citation hallucinations in scientific text. arXiv preprint arXiv:2605.27700, 2026.

Delip Rao and Chris Callison-Burch. BibTeX citation hallucinations in scientific publishing agents: Evaluation and mitigation. arXiv preprint arXiv:2604.03159, 2026

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2023.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using LLM agents as research assistants. arXiv preprint arXiv:2501.04227, 2025.

Yijia Shao, Yucheng Jiang, Theodore A. Kanell, Peter Xu, Omar Khattab, and Monica S. Lam. Assisting in writing Wikipedia-like articles from scratch with large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, 2024.

Yiwen Song, Yale Song, Tomas Pfister, and Jinsung Yoon. PaperOrchestra: A multi-agent framework for automated AI research paper writing. arXiv preprint arXiv:2604.05018, 2026.

Chris van der Lee, Albert Gatt, Emiel van Miltenburg, and Emiel Krahmer. Human evaluation of automatically generated text: Current trends and best practice guidelines. Computer Speech & Language, 67:101151, 2021.

Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue Zhang. AutoSurvey: Large language models can automatically write surveys. In Advances in Neural Information Processing Systems, 2024.

Yiran Wang, Ruixuan An, Biao Wu, and Wenhao Wang. PaperJury: Due-process review for bounded LaTeX revision. arXiv preprint arXiv:2606.16322, 2026.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Simon Hu, Chris Lu, Jakob Foerster, David Ha, and Jeff Clune. The A scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.