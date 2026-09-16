# PaperDoctor: Evidence-Grounded and Actionable Feedback for Scientific Papers in Progress

Kevin Qinghong Lin<sup>1</sup>, Siyuan Hu<sup>3</sup>, Pan Lu<sup>2</sup>, Yu Chen<sup>1</sup>, Yanzhe Chen<sup>3</sup>, Owen Queen<sup>2</sup>, Yupeng Chen<sup>1</sup>, Jialin Yu<sup>1</sup>, Junchi Yu<sup>1</sup>, Zifeng Ding<sup>5</sup>, Yuanfeng Ji<sup>2</sup>, Sheng Liu<sup>2</sup>, Jindong Gu<sup>1</sup>, Linjie Li<sup>4</sup>, Mike Zheng Shou<sup>3</sup>, Philip Torr<sup>B1</sup>, and James Zou<sup>B2</sup>

<sup>1</sup>University of Oxford   
<sup>2</sup>Stanford University   
<sup>3</sup>National University of Singapore   
<sup>4</sup>University of Washington   
<sup>5</sup>University of Cambridge

GitHub

Website

## Abstract

Autoresearch agents are reshaping the research ecosystem, but they can also let flawed claims enter the literature at scale. Human advisors catch such issues on inprogress drafts through careful, traceable feedback, yet advisor-style assessment requires extensive manual effort and does not scale. To shift automated paper assessment from a judge to a diagnostician, we introduce PaperDoctor, an agent framework for pre-submission feedback with three key innovations: (i) Holistic hierarchical framework. Every paper is evaluated across writing, layout, references, code, theory, prior work, and experiments through three layers: L1 surface screening runs directly on every submission; L2 typed verifiers route each claim to the branch that validates the corresponding piece of evidence; and L3 reproducers rerun experiments by priority. (ii) Evidence-grounded actionable feedback. Each PaperDoctor’s finding is a triple of an observation, a pointer to a specific evidence such as a sentence, equation, or code line, and a revision suggestion, making every critique auditable and actionable. (iii) Effective experimental reproduction. Beyond reading the paper, PaperDoctor selectively rebuilds and reruns experiments based on claim importance and compute budget, surfacing reproducibility gaps and quantitative limitations that are invisible from the manuscript alone. We evaluate PaperDoctor with 30 in-progress papers, yielding 70.6% agreement and all positive holistic scores. We further evaluate PaperDoctor on 40 manuscripts across machine learning, natural and social-sciences, covering human- and AI-authored papers with code. Overall, PaperDoctor produces more auditable feedback than human and other agentic reviewers, pairs its critiques with concrete suggestions by design, and complements dimensions that are often overlooked by human reviewers. To empower the community, we develop an interactive interface that lets authors browse findings grounded in their paper. PaperDoctor reframes automated paper assessment as a “diagnostic” process rather than a verdict, taking a concrete step toward AI advisors that help with more rigorous AI-assisted scientific discovery.

## 1 Introduction

“Don’t find fault, find a remedy.” — Henry Ford

Autoresearch agents [2, 22, 28, 34, 45] turn a proposed idea into a full manuscript, increasing the risk of producing work whose quality cannot be guaranteed, since such claims are often difficult to verify. Moreover, most human-written drafts are now at least partly AI-assisted, whether in writing, figure drawing, or literature survey. This trend raises paper volume while leaving draft quality uncontrolled. Existing automated reviewers [8, 12, 13, 18, 32, 49] do not close this gap. By returning a decision (“accept” or “reject”) with a justification, they primarily serve to filter submissions, which is useful for the reviewer’s side but uninformative for the author.

Human advisors catch exactly these issues during discussion with junior researchers [4, 31]. They go through the draft carefully: circling a claim with no experimental validation, underlining a theorem whose assumption breaks, or flagging a figure that is unclear. Each mark on the page is a small diagnosis. As illustrated in Figure 1 A, it points to where the symptom is in the paper, explains why it is a problem, and prescribes how to fix it; in short, it is evidence-grounded and actionable. This is how researchers learn from an advisor’s diagnosis on the page, which is fundamentally different from a reviewer’s judgment. Yet such depth comes at a cost: it takes hours per paper, making it impossible to keep pace with agent-assisted writing [22, 28, 34, 45]. Current agents either write the paper for the author [2, 34] or grade it at the door [18, 20, 21, 27, 33]. Neither is what a researcher receives from an advisor: a careful diagnosis, like that of a doctor, that points to a specific section and says what to change. However, delivering this level of feedback is non-trivial. A paper is a multi-faceted artifact spanning presentation, implementation, experimental analysis, and more. This is why authors typically rely on feedback from a range of people, such as advisors, peers, and industry mentors, each catching issues the others miss.

Motivated by this, we ask: can an agent provide diagnosis-level feedback at scale? We introduce PaperDoctor, a research agent framework that delivers constructive feedback to human authors. Notably, PaperDoctor highlights the following: (i) Holistic, hierarchical pipeline. PaperDoctor decomposes a paper, together with its code and datasets, along dimensions ranging from writing to experimental reproduction, and organizes the assessment into three hierarchical levels of increasing cost and subjectivity. L1 (surface screening) operates directly on the manuscript itself at low cost, primarily targeting objective issues in writing, formatting, references, and layouts. L2 (claim verification) extracts atomic claims and routes each one to the skill that owns its evidence—web search for prior-work checks, a vision language model (VLM) for figure assessment, code analysis for implementation claims, and theory verifiers for derivations. L3 (experimental reproduction) selectively executes experiments, validating reproducibility and correctness at execution time. This design spends effort proportional to the cost of verification and keeps the full pipeline tractable. (ii) Evidence-grounded, actionable feedback. Each finding is a triple of an observation with a reason (Why), a pointer to a specific location in the paper (Where, e.g., a sentence, equation, code line, or external URL), and a concrete suggestion for revision (How). This makes every critique both auditable and directly actionable for human authors. (iii) Effective experimental reproduction. Beyond reading the manuscript, PaperDoctor selectively prioritizes and reruns experiments based on claim importance and compute budget. By executing code rather than only reading it, PaperDoctor surfaces reproducibility gaps and quantitative limitations that are invisible from the paper alone.

To validate the effectiveness of PaperDoctor, we first conduct a human study with junior researcher participants (30 in-progress papers), collecting their in-progress paper drafts and asking them to rate the feedback produced by PaperDoctor. PaperDoctor reaches 70.6% agreement with their evidence and all positive holistic scores. Moreover, we evaluate PaperDoctor on 40 manuscripts covering machine learning, the natural sciences, and the social sciences, including both human- and AI-authored papers with accompanying code and data. This diverse benchmark allows us to systematically assess PaperDoctor’s robustness across disciplines, writing styles, and varying levels of methodological rigor. We found that PaperDoctor produces more auditable feedback than human and other agentic reviewers, pairs its critiques with concrete suggestions by design, and complements dimensions that are often overlooked by human reviewers. In particular, by executing the accompanying code and re-running key experiments, PaperDoctor reports where a published number could not be re-obtained under its protocol, a gap that reading-only review cannot see.

![](images/056b3b33ddcf3fa44cc4811d5ede9a11b1168adffb644c52861bc4f5cc8270d5.jpg)  
Figure 1: A. PaperDoctor agent provides evidence-grounded, actionable feedback. Given a paper along with its code and datasets, PaperDoctor returns a holistic report of findings across multiple dimensions (writing, citations, figures, equations, code, reproduction, and related work). Each finding pairs an error type (Why) with a concrete suggestion (How) and is grounded (Where) in the paper itself, allowing authors to audit and learn from every critique. B. Illustration of PaperDoctor pipeline. PaperDoctor runs three diagnostic stages: L1 paper-only screening (Sec. 2.2), L2 typed claim verifiers (Sec. 2.3), and L3 prioritised experiment reproduction (Sec. 2.4), to produce evidence-grounded, actionable feedback on pre-submission papers.

Lastly, to empower the research community, we release a demo interface (See Fig. 5) that lets authors browse findings anchored directly on their own papers, making it easy to trace each critique back to its exact location.

## 2 PaperDoctor

## 2.1 Overview

Given a paper and its code, PaperDoctor aims to produce a set of feedback $\{ \mathcal { F } _ { i } \} _ { i = 1 } ^ { N }$ , where each finding is defined as

$$
{ \mathcal F } _ { i } = ( f _ { i } , e _ { i } , s _ { i } ) .
$$

Here, $f _ { i }$ is the finding (i.e., a brief textual description of the issue), $e _ { i }$ is the evidence grounding it to a specific location (a sentence, equation, code line, or external URL), and $s _ { i }$ is a concrete suggestion for revision. Notably, the pair $\left( { { e } _ { i } } , { { s } _ { i } } \right)$ lets the author audit and act on each finding without searching through the full paper. Notably, we distinguish findings into two severity levels. An error indicates that PaperDoctor is confident the issue is incorrect (such as factual error), while a warning indicates uncertainty and flags the finding for further clarification, such as a human check. We enable LLM to determine them autonomously, in order to better leverage context.

Paper-Code Parsing. Producing $\{ \mathcal { F } _ { i } \}$ by feeding the entire paper and codebase into a single model is infeasible: a paper is a long, multimodal document and a codebase is itself a large, structured artifact. Moreover, most downstream skills only need a targeted slice of the inputs $( e . g .$ , reference verification needs only the bibliography; figure assessment needs only the rendered pages). We therefore run a single parsing step that produces three decomposed reusable artifacts:

$$
( \mathcal { P } _ { t } , \mathcal { P } _ { v } , \mathcal { C } _ { t } , B ) \gets ( \mathtt { P a p e r } , \mathtt { C o d e } ) ,
$$

where $\mathcal { P } _ { t }$ is the paper as section-organized markdown (via Mathpix<sup>1</sup>), $\mathcal { P } _ { v }$ is the same paper rendered page-by-page as images for downstream Vision-Language Model (VLM) use, and $\mathcal { C } _ { t }$ is the code indexed with tree-sitter<sup>2</sup> into per-file units. B denotes the parsed bibliography of the paper (e.g.,.bib file). Every downstream skill reads from this shared representation and requests only the section, page image, or code snippet it needs.

Hierarchical Pipeline. A paper spans many dimensions, and processing all of them in a single pass is infeasible. Different aspects demand different forms of evaluation, and these evaluations vary widely in cost: a writing check is a single LLM call, whereas experiment reproduction can consume hours of GPU execution. We therefore organize PaperDoctor as a hierarchical pipeline of three levels (L1–L3) that spends effort proportional to the cost of verification. As illustrated in Figure 1B, L1 handles the most concrete, surface-level checks (such as presentation); L2 verifies individual claims along specific dimensions (such as theory or comparison with prior work); and L3 runs the most expensive stage, full experimental reproduction.

## 2.2 L1 – Surface Screening

This stage focuses on straightforward issues that can be easily addressed by browsing the paper.

Writing Review. We ask an LLM to read the paper by section $\mathcal { P } _ { t }$ and flag writing issues as it goes. Clear mistakes such as typos or grammatical errors are marked as errors, since they are unambiguously wrong. Stylistic issues, where the text is understandable but could be phrased more clearly, are marked as suggestions instead, leaving the final decision to the author. For every issue, we require the LLM to quote the original sentence verbatim as evidence, ensuring each finding can be traced back to a specific location in the paper.

L1: Writing

Evidence: Page 3 “We trian the model on a large corpus of academic papers.”

Suggestion: There is a typo: “trian” should be “train”.

Figure Review. A paper is as much a visual artifact as a textual one: its figures, tables, and overall layout are carefully curated by the authors and judged by readers at a glance. Yet most of this visual information is lost in markdown extraction. We therefore treat visual inspection as a separate check in PaperDoctor: we render the paper into page images $\mathcal { P } _ { v }$ and ask a VLM to review them directly. Clear visual defects, such as figures overflowing the text margin or overlapping captions, are flagged as errors. More subjective issues, such as undersized fonts or insufficient color contrast, are flagged as warnings for the author to judge. As with writing review, every finding must be grounded to a specific page or figure index as evidence.

L1: Figure

Evidence: Page 5, Figure 3 extends beyond the right text margin.

Suggestion: Rescale the figure width to \linewidth.

Citation Check. AI-assisted manuscripts routinely contain references that do not resolve to any real paper, and this is tedious to catch by reading the bibliography alone. Conditioned on B, we pair the agent with a web search backend: for each reference, the LLM issues up to three search queries and records a resolver URL only when the backend returns a genuine match, never synthesizing itself.

L1: Citation Check

Evidence: Reference [12] “Smith et al., Neural Reasoning in Transformers, NeurIPS 2023” returns no match.

Suggestion: The reference appears to be hallucinated. Please verify and replace with a valid source.

Claim Extraction. A paper makes dozens of arguments across its sections—novelty claims in the introduction, methodological choices in the method section, performance numbers in the experiments, and so on—and the value of PaperDoctor comes from checking each of them against its own evidence. A claim that is never extracted can never be verified. We therefore ask an LLM to densely extract every verifiable assertion the authors make, covering [theory, code, experiments(designs), literature]; a single claim may be tagged with multiple evidence types. Each claim is then dispatched to the corresponding L2 branches for verification.

L1: Claim Extraction   
Claim: “Our method achieves 92.3% accuracy on ImageNet, outperforming prior state-of-the-art by 3.1   
points.”   
Evidence: Introduction   
Claim Type: [Experiment, Related Work]

Owing to the modular design, all four skills in L1 can run in parallel.

## 2.3 L2 – Claim Verification

In this stage, each claim extracted at L1 is sent to the corresponding verifier.

Code Verification. A common failure mode of AI-assisted drafts is that the described optimizer, architecture, or training setup does not match the released code. These mismatches are almost invisible to human reviewers, who rarely open the repo while reading. We therefore ask PaperDoctor to check each code-tagged claim directly against the source $\left( \mathcal { P } _ { t } , \{ \mathcal { C } _ { j } \} \right)$ , which are indexable during our parsing stage. For example, hyperparameter claims are often best verified by first inspecting configuration files before descending into the Python implementation: if the paper claims AdamW but the config specifies Adam, we flag a warning—the mismatch is real, but may be a stale config or a last-minute switch the author should confirm. If a component described in the paper is missing from the code altogether, we flag it as an error.

L2: Code Verification   
Claim: “We train all models using the AdamW optimizer.”   
Evidence: configs/train.yaml line 14 specifies optimizer: Adam, which conflicts with the paper’s   
experiment settings   
Suggestion: Mismatch between paper and code. Verify which optimizer was actually used.

Theory Verification. Errors in theoretical derivations are among the hardest to catch: a proof that reads smoothly often hides missing assumptions, skipped steps, or notation drift across equations, and even careful readers can miss these on a first pass. We therefore ask PaperDoctor to rederive each argument in $\mathcal { P } _ { t }$ step by step rather than summarize it. PaperDoctor jointly examines all theoretical content, including equations, variables, and the notation that links them across the paper, and records the full trace so that a superficial check is itself visible as a superficial trace. Specifically, PaperDoctor examines four aspects in turn: correctness of each step, hidden assumptions that the paper does not state, boundary or edge-case behavior, and notation consistency across derivations. A loss whose expectation silently drops between its definition and its final form is caught here, before it propagates into code or experiments.

L2: Theory Verification   
Claim: “The expected loss reduces to $\mathbb { E } [ \Vert x - \hat { x } \Vert ^ { 2 } ] ( \mathrm { E q . } 7 ) . ^ { \prime \prime }$   
Evidence: Step from Eq. 6 to Eq. 7 drops the cross-term $\mathbb { E } [ x ^ { \top } \hat { x } ]$ without justification.   
Suggestion: Missing assumption that x and xˆ are uncorrelated. Please state explicitly or correct the   
derivation.

Literature Check. A common issue in scientific drafts is overstated novelty or weak engagement with the literature (“no prior work addresses $X ^ { \prime \prime }$ when an earlier paper already does), and this bias is easier to catch by searching than by reading. Based on $\left( \mathcal { P } _ { t } , B \right)$ , PaperDoctor pairs an LLM with a web search backend to separate baseline comparisons, cited facts, and novelty assertions, so that each novelty assertion can be further labelled as novel, incremental, or prior\_art\_exists.

Note that this differs from the reference verifier in L1: the reference verifier only checks whether a cited paper exists and is correctly attributed, while this stage examines whether the surrounding literature content actually supports the paper’s novelty and positioning claims.

## L2: Literature Check

Claim: “No prior work addresses multi-modal reasoning over long video sequences.”

Evidence: Web search returns Chen et al. (2023), “LongVid-Reasoner”, which targets the same setting.

Suggestion: Novelty overstated. Relabel as prior\_art\_exists and cite Chen et al.

Experiment Design. While some claims can be closed-loop verified against external sources (literature) or formal content (theory, code), most claims in a paper rest on experimental evidence and crucially, whether the experiments themselves are well-designed determines whether the contribution is genuinely grounded. Experimental issues thus fall into two regimes: design mistakes (missing ablations, missing experiments for a stated contribution) that can be found from the paper alone, and reproduction mistakes that can only be found by running. This module handles the first, before any execution. The agent reviews whether each argument is supported by a corresponding experiment, along with fairness, ablation sufficiency, statistical rigour, baseline recency, and cherry-picking risk. It then emits a reproduction-plan entry per experiment listing the command, priority, feasibility, run mode (evaluation or training), and the numeric target lifted from the paper. No experiments run here; the plan is a declarative contract for L3.

## L2: Experiment Design

Claim: “Our cross-modal attention module is the key component driving the gains over prior work.” Evidence: Table 3 reports only the full method versus the baseline; no ablation removes the cross-modal attention module to isolate its contribution.

Suggestion: Add an ablation that disables the cross-modal attention module and reports performance on the same benchmark.

Reproduction plan: bash eval/ablate\_attn.sh, target ∆ ≥1.0 point drop, priority high.

Notably, the L2 stage remains highly modular: all four verifiers, as well as the per-claim dispatch within each verifier, run in parallel.

## 2.4 L3 – Experiments Reproduction

After L1 and L2, PaperDoctor has already covered most aspects that can be assessed from the paper’s main body. The remaining, equally important stage is reproduction, which is challenging yet essential. Different papers come with very different experimental setups: some require only inference, others involve full training, and many depend on substantial resources such as GPU compute, datasets, and storage. We therefore design a separate L3 stage dedicated to reproduction.

Priority Ordering. It is worth noting that not every claim deserves an equal degree of attention. For example, an experiment that backs a main claim in the abstract carries far greater weight than a hyper-parameter sensitivity study, even though the latter may be cheaper to run. We rank experiments by their importance to the paper’s central contributions and by their expected feasibility check. PaperDoctor assigns each entry in the reproduction plan a priority label of {high,medium,low}. Under a compute budget, high-priority items run before medium and low, evaluations before training, and ready experiments before blocked ones.

Manual Approval. Reproduction consumes real compute and storage, and executes code that may affect the environment. A mis-typed claim could silently trigger a multi-hour training run. PaperDoctor therefore presents the L2 plan, annotated with estimated GPU hours, dataset size, and storage footprint, for the author to approve. Only then does an LLM-driven executor handle environment setup, dispatch, and log parsing.

L3: Reproduction

Claim: “Our method achieves 78.4% accuracy on MMLU.”

Evidence: Reproduced run yields 71.2% (logs/mmlu\_eval.log), a 7.2-point gap.

Suggestion: Discrepancy exceeds the 1–2% tolerance. Flagged as error; verify evaluation protocol or reported number.

Report Results. Each executed experiment will receive a decision by comparing the reproduced value against the paper’s reported one. Rather than imposing a fixed numeric threshold, PaperDoctor judges the verdict in context: it considers the metric type, the typical variance reported in the paper, and the magnitude of the original gap, and decides whether the result counts as a pass, a warning (partial match), or an error (execution failure or numeric mismatch). This gives the author a direct view of which claims hold up under execution and which diverge from what the paper reports.

## 3 Results

## Authors judge the review helpful overall, but agreement varies item by item

Thirty pre-submission papers from 25 graduate students went through PaperDoctor, and their authors judged every issue the six checks raised about the paper itself, 1,299 items in all.<sup>3</sup> Authors rated each item accepted, uncertain, or rejected on two axes: Evidence, whether the issue is real, and Suggestion, whether the proposed fix resolves it. They also scored the review as a whole from −1 (harmful) to +2 (very helpful), with an optional free-form comment. All 30 scores were positive, 70% somewhat helpful and 30% very helpful, for a mean of +1.30 (Fig. 2A).

Authors accepted most of what PaperDoctor found evidence on their own paper (Fig. 2B). Across all 1,299 items the acceptance rate was 70.6%, and the mean of the 30 per-paper rates was 68.5%. Per paper the rate ran from 23.5% to 97.0% with a median of 71.1%. The authors at both extremes also rated the review +1, and across the 30 papers the correlation between Evidence acceptance and the holistic score is positive but not statistically significant (r = +0.29, P = 0.12). The two measurements are not substitutes: the score asks whether the feedback improved the draft, and the item ratings ask whether each evidence is real.

Authors agreed more often with the L2 verifiable claims than with the L1 surface screening (Fig. 2C). The three L2 checks were rejected 11% of the time, the three L1 checks 27%. Experiment designs drew the most items of any claim, 539, and authors accepted 81% of them. The Figure Review assesses how effectively each figure is designed, and authors accepted 42%, less than for any other claim. Vision perception models still misread dense panels, and much of figure design is the author’s own decision. Citation check raised the fewest items, because it confirmed most of the references it read rather than flagging them.

Authors were more certain about the items PaperDoctor marked Error, and agreed with them less (Fig. 2D). They accepted 71% of Warnings on both axes, against 67% of Errors on Evidence and 66% on Suggestion. Marking an item an Error cut the uncertain share from 11% to 8% and raised the rejected share from 18% to 25%. Accepting that an issue was real almost always meant accepting the suggestion proposed for it: of the items whose Evidence an author accepted, 97.1% had the Suggestion accepted too, against 3.3% of the items whose Evidence they rejected.

A  
![](images/c3942ab61f86d6390cb1831b8bc94e3363d97c69e674a2becc67785ff7b9961c.jpg)  
B

![](images/16b6bba4a5883fce32e5a2bda222bc23e889131b9466269bf66c9478dfc5e943.jpg)

C  
![](images/560b01b0d9ac63f4cfbf42f79e3339addd97253ba6b194b34983e3bb29b2385d.jpg)

D  
![](images/676f12b1dae6a3c88959fbd75aa7a866817890cbbd52e435bf5d16ad3c48c57b.jpg)

E  
![](images/e42e57ff216224fa67ecf04f0adf7d180800e5249bc3596869573081e95fede3.jpg)  
Figure 2: Author verdicts on PaperDoctor’s feedback for 30 pre-submission papers, from the holistic review down to the fine-grained item. A. Holistic rating of the feedback. B. Each paper’s Evidence acceptance, by the rating its author gave the review. C. Author verdicts on Evidence (E) and Suggestion (S). D. Author verdicts within PaperDoctor’s own Warning and Error labels. E. The written comments that came with the ratings in A, by the aspect each addresses.

Authors split over how much the review should cover (Fig. 2E). An LLM split the 24 written comments into 79 aspect responses, 58 praise and 21 criticism. No aspect drew more praise than breadth, at 10 responses, and none drew more criticism, at 5. Writing and language drew the next most praise, 9 responses, and it is the part of a paper a language model is best placed to check. Presentation clarity followed at 6, then cross-part consistency, related work, and figures at 5 each, three aspects that each need evidence from outside the sentence in front of the reader. Three aspects drew more criticism than praise: the fixes PaperDoctor proposed, the items it got wrong, and its silence on novelty and framing. The fixes and the silence ask PaperDoctor to judge what is worth saying rather than to check whether something is true. Scanning every dimension of every paper also costs precision: the items PaperDoctor got wrong drew 4 criticisms against 1 praise, the widest negative margin in the panel.

A  
![](images/861f7c66afa95b626186183e8705052978b89d573121d6ad71e5076fc237148c.jpg)  
B

![](images/65b223f6595d15a7b6e01794fded13fa5ecb53b397929782b3d081a647fbf43b.jpg)

C  
![](images/f8d4c8a886423612d440ef337fc0a5e81076774db34830b1831ea9c5a55869c5.jpg)

D  
![](images/0ad3642a45af62a3582ceea2651d14b633bae94048e53386e5cf46dd6622182d.jpg)

E  
![](images/705c427843862731e020c909d914dde28aa62e10b9fd21596700c86e896f3acc.jpg)  
Figure 3: Comparison between referee reports, the Stanford Agentic Reviewer and PaperDoctor on the same 40 papers, from how much each writes down to what it is about. A. Findings per reviewer, per paper, by domain. B. Whether a finding carries Evidence, a Suggestion, both, or neither. C. Words per piece of feedback, PaperDoctor’s split into its Evidence and its Suggestion. D. What the findings are about. E. PaperDoctor’s findings per paper, by check and by the kind of paper reviewed. Bar, the mean; whisker, one standard deviation. First published referee round only.

Evidence-grounded feedback arise by design, not by chance.

We compared PaperDoctor against the reviews the same papers actually received. The 40 papers are four groups of 10: Agents4Science [2], whose papers are AI-written and whose reviews are themselves AI-written; ICML orals and spotlights; Nature Communications, for the natural sciences; and Nature Human Behaviour, for the social sciences. Every paper carries three feedback of itself: its published referee reports; one run of the Stanford Agentic Reviewer<sup>4</sup>, an agentic system that reviews from the paper alone; and our PaperDoctor<sup>5</sup>. Because the three sources have different output structures, we use Gemini-2.5-Flash to parse all feedback into a common set of atomic points for consistent analysis.

PaperDoctor writes the most and with the widest spread, and the referee reports the least (Fig. 3A). We compare per reviewer<sup>6</sup>. The Agent4Science and ICML conference reports contain 6.9 and 6.7 findings per source, respectively, compared with 16.2 and 17.5 for the Nature journal reports. This difference remains consistent across referees. Both automated systems write more than that in every domain. PaperDoctor writes 36.1 to 56.4 findings per reviewer against the agentic reviewer’s 25.2 to 44.0, ahead of it in three domains and level with it in the fourth. PaperDoctor shows larger within-field variation, suggesting higher feedback sensitivity to the individual paper, with a range of 53 findings on Agent4Science papers versus 13 for the agentic reviewer.

PaperDoctor differs from the agentic reviewer most of all in evidence grounding (Fig. 3B). Both referees and the agentic reviewer propose a Suggestion at nearly the same rate (in 71.7% and 71.5% of their findings), showing that the agentic reviewer provides suggestions about as often as human referees. When it comes to Evidence, which requires grounding in the paper, referees provide it in 45.2% of their findings, compared with only 2.2% for the agentic reviewer. Considering Suggestion and Evidence jointly, 35.9% of referee findings contain both, versus 1.5% for the agentic reviewer; instead, the agentic reviewer’s most common pattern is a suggestion without supporting evidence, accounting for 69.9% of its findings. Every PaperDoctor finding carries both, because the design explicitly design for this: PaperDoctor never emits a record with no quote and no suggestion. It pays for that in length, averaging 50.0 words per finding against a referee’s 29.1 and the agentic reviewer’s 27.1, of which 33.9 are the Evidence and 16.1 the Suggestion (Fig. 3C).

PaperDoctor covers a paper more evenly than either reviewer, and looks outside it more often (Fig. 3D). Both referee and agentic reviewer spend mostly on the paper main body, 90.1% and 89.5% of their findings, and the experiments draw most of that: 49.5% of a referee’s findings and 72.6% of the agentic reviewer’s. PaperDoctor’s distribution is the flattest of the three: the experiments are its largest single dimension, at 28.4%. It is the source that spends much outside the paper, 12.5% on the external literature and 11.5% on the code against 10.0% and 10.5% for the two on both dimensions together.

Code and Experiment design are the two checks that tell the groups apart (Fig. 3E). The three L1 checks (writing, figure, and citation) report comparable numbers everywhere. L2’s Code separates the groups by authorship, 8.4 findings per paper for the AI-written Agent4Science papers against 4.6, 4.1 and 2.4 for the three human-written groups. Related work is flat at 1.7 to 2.9, and Theory rises only for ICML, at 3.8 against 0.8 to 1.9 elsewhere, where the papers are about machine-learning methodology. Experiment design draws the most findings of the four L2 checks and the widest spread with them, 18.7 for Agent4Science and 16.1 for Nature Science against 9.9 for ICML and 9.7 for SocialScience. Both L2 Code and Experiment Design suggest that reproducibility may be a key factor distinguishing the paper groups.

<table><tr><td colspan="2">L2&#x27;s Experiment claim</td><td rowspan="2">Reproduction plan</td><td rowspan="2">Feasibility verdict Ready</td><td rowspan="2">Execution</td><td rowspan="2">Outcome</td></tr><tr><td></td><td></td></tr><tr><td rowspan="3">43.8</td><td></td><td>Planned</td><td>2.6 (18.0%)</td><td>Ran 7.7 (53.0%)</td><td>Pass 3.0 (20.4%) Warning 2.1 (14.3%)</td></tr><tr><td></td><td>14.5 (100.0%)</td><td>Blocked 11.9 (82.0%)</td><td>Never ran (Fig. E)</td><td>Error 2.6 (18.3%)</td></tr><tr><td></td><td>No rerun planned 29.3</td><td></td><td>6.8 (47.0%)</td><td></td></tr></table>

A

B  
![](images/b1b06c35eb76407f703ee80c44ef4aeb470a0201dfdcb1949b09af0b4bdef1df.jpg)

![](images/1bf465af2c1a7104bab4b0a5523cdff46f77374c81a62d34c3c51d54809da71f.jpg)  
D

![](images/d0007d3e7ae1e512c118552362a83f383e909a48b6dd5ad174738a57f1301574.jpg)

E  
![](images/41db348cfeebf185dfb92d4f820aced83eee1c8d30d08438dfd44bd8983efa51.jpg)  
Figure 4: Reproducing the experiments of 40 papers that release code and data. A. From a claim to a number, per paper; green, still moving; grey, stopped. B. Outcome by the rank the plan carried, over the plans that ran. C. Outcome by what the plan reruns, over the plans that ran. D. What became of a source’s plans; the centre line is a command running. E. Why the plans that never ran did not run, by source. Every node in A is a per-paper average, with its share of the reproduction plans beside it. A plan counts as run when a command was actually executed for it.

## Reproduction is bottlenecked twice, before a command runs and after it

We sent the L2 Experiment claims from the same 40 papers to the reproduction stage. L3 turns claims that require re-execution into executable plans with verifiable targets, such as reported numbers. Each plan is then assigned a pass, a warning, or an error. Error groups execution failure and numeric mismatch. Of the 43.8 experimental claims per paper, 14.5 become reproduction plans, but only 18.0% of these plans are judged ready to run, while 82.0% are initially blocked. After agent resolving feasible blockers, 7.7 plans reach execution, and 5.1 return a pass or warning (Fig. 4A).

Reproduction is limited by missing guideline and mismatched reproduced results. Almost half of all plans, 47.0%, never reach a command at all, most often because the paper’s repository does not provide enough information to determine how to run them (Fig. 4E). Even getting a command ready to run is still far from validating the claim: among the 53.0% of plans that reach execution (Fig. 4A, right), 27.0% result in warnings and 34.5% in errors.

A plan that PaperDoctor ranked highly passes more often, and errors less often (Fig. 4B). Highpriority plans pass 47.3% of the time once they run and error 29.1% of the time. Low-priority plans pass 33.6% and error 39.7%. Medium sits with low rather than between the two, passing 33.3%. PaperDoctor’s prioritization is consistent with human practices: important components are often prepared and maintained more carefully, making high-priority plans more likely to reproduce successfully.

A plan that requires training passes least often and errors most (Fig. 4C). Training passes 11.3% of the time once it runs and errors 58.5%. Rerunning inference (such as based on released checkpoint) passes 39.0% and errors 38.2%, and rerunning statistical test or analysis pipeline passes 48.9% and errors 21.4%. The order follows how much of the original setup a run has to rebuild. Training needs the environment, the dataset and the compute budget together; a checkpoint needs the first two; an analysis pipeline simply needs the data alone.

The AI-written papers and the ICML oral/spotlight papers exhibit two contrasting failure modes (Fig. 4D). ICML papers usually fail after a command runs: 132 of 167 plans execute, but only 19 match the reported number (14.4%). Agent4Science papers usually fail before a command runs (68 of 86), yet 13 of the 18 that do run match (72.2%). NatureScience and SocialScience sit between the two, losing about half at each of the two stages.

Different field papers exhibit distinct and representative causes of failure (Fig. 4E). Incomplete code stops 65.2% of SocialScience’s blocked plans and 37.1% of ICML’s: the paper deposits its data, but not the analysis scripts that produced its paper figures. NatureScience is stopped mostly by environments the agent cannot rerun: 46.2% of its blocked plans rest on wet-lab procedures its discipline cannot move to a machine. Agent4Science is stopped earlier still, by restricted data and by near-duplicate plans that carry another plan’s blocker. Across all 272 blocked plans, an incomplete runnable environment (either missing code or model weights) is the largest cause, accounting for 33.1%.

## 4 Related Work

AI for Autoresearch End-to-end autoresearch pipelines chain ideation, experimentation, and drafting into a single agentic loop [2, 22, 28, 34, 45], increasingly supported by language and coding agents [15, 43, 46] that probe whether agents can implement and run published methods. Recent variants explore tool augmentation, multi-agent collaboration, and long-horizon execution, so manuscripts are produced with ever less human oversight. None of these pipelines, however, includes an internal quality-control stage, so characteristic failure modes pass silently into the final draft: hallucinated citations, inflated novelty, mismatches between claims and the code that implements them, and unverified empirical statements. PaperDoctor is complementary to this line of work: rather than producing papers, it consumes a draft together with its accompanying code and data, decomposes it across writing, references, theory, and experiments, and locates the parts that need revision before submission. In this sense, it is the internal advisor that current autoresearch agents lack.

Paper Verification For rigorous assessment, a paper can be treated as a verifiable system whose claims must be checked along multiple dimensions. (i) Citation verification. LLM drafts routinely contain fabricated references, and audits report elevated hallucination rates [23, 44]; dedicated verifiers [42, 47] and attributed-generation frameworks [3, 10, 11, 25] supply the primitives for PaperDoctor’s reference verifier. (ii) Theoretical-claim verification. LLM-based provers [1, 24, 26, 39] and agentic program verifiers [38] combine informal reasoning with Lean and Coq checking. Paper-Doctor incorporates this perspective: it densely extracts informal arguments and checks whether they match formal or executable content. (iii) Experiment reproduction. Beyond textual verification, reproducibility benchmarks [5, 16, 35] evaluate ML engineering on Kaggle- and repo-scale tasks, and others extend this to software, computational, scientific, and frontier-R&D settings [7, 17, 29, 41], where top agents still fall below 40% execution accuracy [15, 43, 46]. PaperDoctor internalises the lesson that full reproduction is expensive and noisy: L3 turns the paper’s own claims into a prioritised plan and executes it only with author approval, so compute goes where it is most informative.

Table 1: PaperDoctor vs. representative research agents. Reviewer Report: produces reviewerstyle prose, not only a score. Grounded Evidence: every finding is anchored to a concrete span/figure/equation/code line. Revision Suggestion: output specifies what to change, not only what is wrong. Text (LLM): reads body text, tables, and document structure. Visual (VLM): reads figures as images. Code Audit: cross-checks paper claims against released code. Exp. Reproduction: re-executes experiments. In-progress papers: can the agent support or focus on these in-progress manuscripts. ✓=full, ✓=partial, ✗=absent.
<table><tr><td rowspan="2">System</td><td colspan="3">Feedback Form</td><td colspan="4">Assessment Coverage</td><td>Focus</td></tr><tr><td>Report</td><td>Reviewer Grounded Revision</td><td>Evidence Suggestion</td><td>(LLM) (VLM) Imple. Repro.</td><td></td><td></td><td>Text Visual Code Exp.</td><td>In-progress Papers</td></tr><tr><td colspan="9">Autoresearch</td></tr><tr><td>AI Scientist [45]</td><td></td><td>x</td><td>x</td><td></td><td>x</td><td></td><td></td><td>x</td></tr><tr><td colspan="9">Peer Review</td></tr><tr><td>MARG [8]</td><td></td><td>X</td><td></td><td></td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td>AgentReview [18]</td><td></td><td>X</td><td>V</td><td></td><td>x</td><td>X</td><td>x</td><td>x</td></tr><tr><td>Reviewer2 [13]</td><td></td><td>X</td><td>V</td><td></td><td>x</td><td>X</td><td>x</td><td>X</td></tr><tr><td>DeepReview [49]</td><td></td><td>X</td><td>√</td><td></td><td>x</td><td>x</td><td>X</td><td>X</td></tr><tr><td>TreeReview [6]</td><td></td><td>V</td><td>V</td><td></td><td>x</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CycleResearcher [40]</td><td></td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MMReview [12]</td><td></td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td><td>X</td><td>x</td></tr><tr><td colspan="9">Verification</td></tr><tr><td>CiteAudit [47]</td><td>x</td><td></td><td>x</td><td></td><td>X</td><td>x</td><td>X</td><td>x</td></tr><tr><td>PaperBench [30]</td><td>x</td><td></td><td>X</td><td></td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>AutoReproduce [48]</td><td>√</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td colspan="9">Feedback</td></tr><tr><td>Review feedback [37]</td><td></td><td></td><td>x</td><td></td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Human advisor</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>PaperDoctor (ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>J</td><td></td></tr></table>

Paper Peer Review A rapidly growing line casts LLMs as peer reviewers. Static prompting or fine-tuning yields a full review per paper [8, 13, 21, 27, 32, 49]; multi-agent variants simulate the review cycle with role-specialised agents [18, 20, 33]; decomposition-based TreeReview [6] recursively asks sub-questions; and CycleResearcher [40] closes the loop via iterative preference optimization. Multimodal and multidisciplinary variants [12, 14] read figures and tables; pre-submission assistance has been proposed as an ethically cleaner deployment [9]. At scale, the Review Feedback Agent [36] was deployed on 20k ICLR-2025 reviews. Parallel audits document persistent failure modes: prompt-injection susceptibility, sycophancy, and poor novelty calibration [19, 50]. These systems share a judge-oriented output contract, optimised to agree with held-out reviewer opinions rather than to serve the author, and they under-serve revision in three ways that PaperDoctor inverts. (i) Evidence grounding. Verdicts are rarely tied to a specific sentence, equation, or code line; PaperDoctor emits (finding, evidence, suggestion) triples attached to concrete artefacts. (ii) Claim-level auditability. Holistic judgements let individual unverifiable assertions pass silently; PaperDoctor extracts claims and verifies each one. (iii) Cost tiering. Review pipelines are binary, in that every check always runs or never runs; PaperDoctor runs cheap screens by default, routes typed verifiers by the evidence they need, and reserves full reproduction for explicit author approval.

Existing methods are designed to “judge” a finished manuscript, and are evaluated by how closely their verdicts agree with human reviewers. PaperDoctor is designed to “assist” a manuscript still in progress: it grounds every finding in verified evidence and pairs it with an actionable revision, so its success is measured by whether authors accept each finding and act on it, rather than by score agreement.

## 5 Discussion

We introduced PaperDoctor, an agentic framework that shifts automated paper assessment from a judge to a diagnostician. Given a paper with its code and data, PaperDoctor returns findings, each pairing an evidence with a location in the paper and a suggestion, through a pipeline that scales effort with the cost of verification: surface checks on the manuscript, typed verifiers that route each claim to the evidence that settles it, and selective reruns of experiments under a priority budget. Across a human study with junior researchers and 40 manuscripts spanning machine learning, the natural sciences, and the social sciences, PaperDoctor produced more auditable feedback than human and other agentic reviewers, pairs critiques with suggestions by design, and complements dimensions that reading-only review often misses. Its reproduction stage reports where a claim could not be re-obtained under our protocol—including missing instructions, execution failures, and numeric disagreements—without treating venue or pass rate as a quality ranking.

Review for the paper, feedback for the author. A review is a judgment in service of a venue: accept or reject, with a justification that mostly explains the verdict. Feedback serves the author, and that is where PaperDoctor sits. It automates the checkable part of feedback, such as cross-checking numbers against tables, aligning prose with code, and rerunning experiments, and leaves judgment alone: whether a question is worth asking and whether a result matters remain human calls. The aim is not to replace the human but to relocate their time and attention: once an agent has checked what can be checked, the hours an author spends on a draft can go to taste and direction. A natural extension is an interactive loop in which the author accepts or contests each finding and the affected checks re-run, so that machine coverage and human judgment compound.

Verifiable, executable environments for future papers. As agents begin to assist research and most human drafts carry some AI assistance, plausible-looking claims accumulate faster than anyone can check by hand, and assessment has to become verifiable and executable in turn. PaperDoctor is built around this principle. Every feedback can be audited: its anchor points to the exact sentence, equation, line of code, or reference it critiques, so a fabricated critique fails at its own anchor and the author can dismiss it. If diagnosis of this kind becomes routine, a claim backed by runnable code and traceable numbers becomes cheap to check, while a claim without them stays expensive to trust.

Future work. Two directions stand out. (i) Full paper-to-code reproduction. PaperDoctor’s strongest evidence depends on a runnable codebase. When authors release none, an agent could synthesise a reference implementation directly from the paper itself, as PaperBench [30] does, though this setting will yield lower reproduction rates than the current system. (ii) Large-scale advisor-level studies. Advisor comments are rarely recorded systematically and mostly exist as sparse, informal remarks. Collecting such feedback at scale, and grounding it against external reviewer and AI feedback, would let us capture the insight that only experienced advisors provide. We release PaperDoctor together with a demo interface, in the hope of supporting more rigorous and reproducible research in the community.

## A PaperDoctor Interface

![](images/6874e21d5eb09e5607d493317b0fa413e600978422e7c58fef43a47837e54143.jpg)  
Figure 5: The PaperDoctor Diagnosis Interface. Authors upload their paper and code, and PaperDoctor returns the holistic report. Left: the paper, with each finding overlaid as a numbered, color-coded bubble anchored to the exact span it critiques (Where). Middle: the code viewer, so findings about implementation can be audited against the source. Right: the finding card for the selected highlight, showing the observation (Why) and a concrete suggestion for revision (How). In the example shown, PaperDoctor identifies an implementation-level mismatch. : the paper claims a replay mini-batch of 8 every 10 training steps, but the default config (configs/defaults.yaml:31-32) sets replay\_freq=1.

## B Case Studies

In this section, we walk through representative cases drawn from our 40-paper testbed. Each case follows the reason (why), quote evidence (where) and suggestion (how).

L1 Citation Check: a fabricated future date.

![](images/cb93c7d1fecb11bf2e492d8b118c8df1e4246971cc9889f9745b2f330bde1f12.jpg)

In Reasoning Models Outperform Standard Language Models in De Novo Protein Design (Agents4Science), the bibliography contains the entry “[5] OpenAI. Introducing GPT-4.5, 2024. Research preview model.” PaperDoctor’s citation verifier issues a web search for this reference and finds that GPT-4.5 was announced and released on February 27, 2025, not in 2024. Although the manuscript itself appeared in late 2024, the cited release date is therefore chronologically impossible. Standard citation checkers accept any correctly-formatted entry; only a search-grounded verifier flags this kind of temporally inconsistent metadata, which is a recurring failure mode of AI-assisted drafts that synthesise plausible-sounding venues and years without grounding them in real release notes.

L2 Code Verification: paper claims an encoder its code never instantiates.  
![](images/a39a571989c293d10e9f525a4ec352956f74f8c1d6c219ec5ee3a40ceaebc1ae.jpg)  
LLM-Driven Discovery of High-Entropy Catalysts via Retrieval-Augmented Generation (a second Agents4Science paper) states in its Methods section that materials are “encoded using SciBERT into 768-dimensional vectors”. PaperDoctor’s code verifier dispatches the embedding-model claim to the released repository and finds that all three relevant files, code\_data/scripts/embedding\_indexing.py (line 21), code\_data/scripts/rag\_retrieval.py (line 33), and pipeline\_config.yaml (line 9), set model\_name=’all-MiniLM-L6-v2’. SciBERT is never imported anywhere in the codebase, and the produced vectors are 384-dimensional rather than 768. The discrepancy spans both the model identity and its dimensionality. This is a paradigmatic case of a mismatch that is invisible to reading-only review: the paper text reads cleanly, but the implementation differs substantively from what is described.

L2 Theory Verification: a bound stated without definitions.

![](images/43e9c88b3d893b6ce5706638b139bb928acda33d949806145499c4757c1841d5.jpg)

In Neural Reaction-Diffusion Operators for Spatially Heterogeneous Tumor Modeling (Agents4Science), Section 4 states the approximation bound $\| G - G _ { \theta } \| _ { L ^ { 2 } } \leq C _ { 1 } W ^ { - \alpha / d } + C _ { 2 } L ^ { - \beta } + C _ { 3 } \| \mathsf { \bar { D } } - D _ { \mathrm { a p p r o x } } \| _ { \infty }$ as if it were established. PaperDoctor’s step-by-step re-derivation finds that none of the symbols composing the bound are introduced anywhere in the paper: W (presumably network width), L (presumably depth), the exponents α and $\beta ,$ and the constants $C _ { 1 } , C _ { 2 } , C _ { 3 }$ all appear without definition, and no proof or pointer to an appendix is given. Without these, the bound cannot be checked for tightness or used to guide architecture design. On a first pass, such a gap looks like a finished theorem; only step-by-step re-derivation surfaces it.

L2 Literature Check: a novelty claim refuted by the cited works.  
![](images/6d6ccb5cda72ecceb44617a64d45b7ed2e97010624a619ec640660c4486eb615.jpg)

The same paper claims that “DeepONet [12] and Fourier Neural Operators (FNO) [10] are primarily designed for homogeneous PDEs with constant coefficients and struggle with spatially heterogeneous systems prevalent in biological applications.” PaperDoctor dispatches this novelty assertion as a web search query, and the original method papers refute the framing directly: Li et al. (FNO, 2020, [10]) explicitly demonstrates the method on parametric PDEs with varying initial conditions and forcing terms, and Lu et al. (DeepONet, 2021, [12]) demonstrates nonlinear operators with variable coefficients. Neither prior work is restricted to the homogeneous setting the paper attributes to it. This pattern, where an inflated novelty claim is contradicted by the very references the paper cites, is exactly what a search-grounded literature check is built to catch.

L3 Reproduction: a “factor of 2” that is actually 1.14×.

![](images/fa8e0403977030df4e36f0d2dd66000815352e5de2ca7dbaa7c987df41b2001f.jpg)

In Smart hybrid microscopy for cell-friendly detection of rare events, a Nature Communications paper on mitochondrial imaging, the Results section claims that “soft focal loss . . . increased the recall of the best performing models by almost a factor of 2”. The L3 stage rebuilds the analysis from the source data shipped with the paper and reads the recall column for Figure 2d directly. The two soft-focal configurations have contact-task recalls of 0.43 and 0.49, giving an actual best-pair ratio of 1.14× and a mean ratio of 1.04×, far from the claimed factor of two. Both the claimed and reproduced numbers are derivable from artefacts the authors themselves shipped, so the discrepancy is not a matter of environmental drift but of how a quantitative claim was summarised in the text. The aggregate reproduction analysis (Sec. 3) shows that even after execution is feasible, reproduced quantities can still diverge from published claims; the case here shows that peer-reviewed venues are not immune, and that reading the paper alone cannot easily catch it.

## References

[1] Kaito Baba, Chaoran Liu, Shuhei Kurita, and Akiyoshi Sannai. Prover agent: An agent-based framework for formal mathematical proofs. arXiv preprint arXiv:2506.19923, 2025.

[2] Federico Bianchi, Owen Queen, Nitya Thakkar, Eric Sun, and James Zou. Exploring the use of ai authors and reviewers at agents4science. Nature Biotechnology, 44(1):11–14, 2026.

[3] Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, et al. Attributed question answering: Evaluation and modeling for attributed large language models. arXiv preprint arXiv:2212.08037, 2022.

[4] Gulfidan Can and Andrew Walker. A model for doctoral students’ perceptions and attitudes toward written feedback for academic writing. Research in Higher Education, 52(5):508–536, 2011.

[5] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024.

[6] Yuan Chang, Ziyue Li, Hengyuan Zhang, Yuanbo Kong, Yanru Wu, Hayden Kwok-Hay So, Zhijiang Guo, Liya Zhu, and Ngai Wong. Treereview: A dynamic tree of questions framework for deep and efficient llm-based scientific peer review. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 15662–15693, 2025.

[7] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. arXiv preprint arXiv:2410.05080, 2024.

[8] Mike D’Arcy, Tom Hope, Larry Birnbaum, and Doug Downey. MARG: Multi-agent review generation for scientific papers. In arXiv preprint arXiv:2401.04259, 2024.

[9] Christopher Foster. Openness in ai and downstream governance: A global value chain approach. arXiv preprint arXiv:2509.10220, 2025.

[10] Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. Rarr: Researching and revising what language models say, using language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16477–16508, 2023.

[11] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, 2023.

[12] Xian Gao, Jiacheng Ruan, Zongyun Zhang, Jingsheng Gao, Ting Liu, and Yuzhuo Fu. MMReview: A multidisciplinary and multimodal benchmark for LLM-based peer review automation. arXiv preprint arXiv:2508.14146, 2025.

[13] Zhaolin Gao, Kianté Brantley, and Thorsten Joachims. Reviewer2: Optimizing review generation through prompt generation. arXiv preprint arXiv:2402.10886, 2024.

[14] Mengze Hong, Di Jiang, Weiwei Zhao, Yawen Li, Yihang Wang, Xinyuan Luo, Yanjie Sun, and Chen Jason Zhang. Multimodal peer review simulation with actionable to-do recommendations for community-aware manuscript revisions. arXiv preprint arXiv:2511.10902, 2025.

[15] Tianyu Hua, Harper Hua, Violet Xiang, Benjamin Klieger, Sang Truong, Weixin Liang, Fan-Yun Sun, and Nick Haber. Researchcodebench: Benchmarking LLMs on implementing novel machine learning research code. Advances in Neural Information Processing Systems, 38, 2026.

[16] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302, 2023.

[17] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

[18] Yiqiao Jin, Qinlin Zhao, Yiyang Wang, Hao Chen, Kaijie Zhu, Yijia Xiao, and Jindong Wang. Agentreview: Exploring peer review dynamics with llm agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1208–1226, 2024.

[19] Rui Li, Jia-Chen Gu, Po-Nien Kung, Heming Xia, Xiangwen Kong, Zhifang Sui, Nanyun Peng, et al. Llm-reval: Can we trust llm reviewers yet? arXiv preprint arXiv:2510.12367, 2025.

[20] Shuaimin Li, Liyang Fan, Yufang Lin, Zeyang Li, Xian Wei, Shiwen Ni, Hamid Alinejad-Rokny, and Min Yang. Automatic paper reviewing with heterogeneous graph reasoning over llm-simulated reviewer-author debates. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 31717–31725, 2026.

[21] Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Yi Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Scott Smith, Yian Yin, et al. Can large language models provide useful feedback on research papers? a large-scale empirical analysis. NEJM AI, 1(8): AIoa2400196, 2024.

[22] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

[23] MZ Naser. How llms cite and why it matters: A cross-model audit of reference fabrication in ai-assisted academic writing and methods to detect phantom citations. arXiv preprint arXiv:2603.03299, 2026.

[24] Azim Ospanov, Zijin Feng, Jiacheng Sun, Haoli Bai, Xin Shen, and Farzan Farnia. Hermes: Towards efficient and verifiable mathematical reasoning in llms. arXiv preprint arXiv:2511.18760, 2025.

[25] Abhilasha Ravichander, Shrusti Ghela, David Wadden, and Yejin Choi. Halogen: Fantastic llm hallucinations and where to find them. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1402–1425, 2025.

[26] ZZ Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025.

[27] Pouria Rouzrokh, Bardia Khosravi, Parsa Rouzrokh, and Moein Shariatnia. Lattereview: a multi-agent framework for systematic review automation using large language models. arXiv preprint arXiv:2501.05468, 2025.

[28] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using LLM agents as research assistants. arXiv preprint arXiv:2501.04227, 2025.

[29] Zachary S Siegel, Sayash Kapoor, Nitya Nagdir, Benedikt Stroebl, and Arvind Narayanan. Corebench: Fostering the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024.

[30] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, et al. Paperbench: Evaluating ai’s ability to replicate ai research. arXiv preprint arXiv:2504.01848, 2025.

[31] Jacob Steiss, Tamara Tate, Steve Graham, Jazmin Cruz, Michael Hebert, Jiali Wang, Youngsun Moon, Waverly Tseng, Mark Warschauer, and Carol Booth Olson. Comparing the quality of human and chatgpt feedback of students’ writing. Learning and Instruction, 91:101894, 2024.

[32] Pawin Taechoyotin and Daniel Acuna. Remor: Automated peer review generation with llm reasoning and multi-objective reinforcement learning. arXiv preprint arXiv:2505.11718, 2025.

[33] Cheng Tan, Dongxin Lyu, Siyuan Li, Zhangyang Gao, Jingxuan Wei, Siqi Ma, Zicheng Liu, and Stan Z Li. Peer review as a multi-turn and long-context dialogue with role-based interactions. arXiv preprint arXiv:2406.05688, 2024.

[34] Jiabin Tang, Lianghao Xia, Zhonghang Li, and Chao Huang. Ai-researcher: Autonomous scientific innovation. arXiv preprint arXiv:2505.18705, 2025.

[35] Xiangru Tang, Yuliang Liu, Zefan Cai, Yanjun Shao, Junjie Lu, Yichi Zhang, Zexuan Deng, Helan Hu, Kaikai An, Ruijun Huang, et al. Ml-bench: Evaluating large language models and agents for machine learning tasks on repository-level code. arXiv preprint arXiv:2311.09835, 2023.

[36] Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, and James Zou. Can llm feedback enhance review quality? a randomized study of 20k reviews at iclr 2025. arXiv preprint arXiv:2504.09737, 2025.

[37] Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, and James Zou. A large-scale randomized study of large language model feedback in peer review. Nature Machine Intelligence, pages 1–11, 2026.

[38] Haoxin Tu, Huan Zhao, Yahui Song, Mehtab Zafar, Ruijie Meng, and Abhik Roychoudhury. Agentic program verification. arXiv preprint arXiv:2511.17330, 2025.

[39] Sumanth Varambally, Thomas Voice, Yanchao Sun, Zhifeng Chen, Rose Yu, and Ke Ye. Hilbert: Recursively building formal proofs with informal reasoning. arXiv preprint arXiv:2509.22819, 2025.

[40] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. Cycleresearcher: Improving automated research via automated review. arXiv preprint arXiv:2411.00816, 2024.

[41] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, et al. Re-bench: Evaluating frontier ai r&d capabilities of language model agents against human experts. arXiv preprint arXiv:2411.15114, 2024.

[42] Kevin Wu, Eric Wu, Kevin Wei, Angela Zhang, Allison Casasola, Teresa Nguyen, Sith Riantawan, Patricia Shi, Daniel Ho, and James Zou. An automated framework for assessing how well llms cite relevant medical references. Nature Communications, 16(1):3615, 2025.

[43] Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, and Yulan He. SciReplicate-Bench: Benchmarking LLMs in agent-driven algorithmic reproduction from research papers. arXiv preprint arXiv:2504.00255, 2025.

[44] Zuyao Xu, Yuqi Qiu, Lu Sun, FaSheng Miao, Fubin Wu, Xinyi Wang, Xiang Li, Haozhe Lu, ZhengZe Zhang, Yuxin Hu, et al. Ghostcite: A large-scale analysis of citation validity in the age of large language models. arXiv preprint arXiv:2602.06718, 2026.

[45] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

[46] Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, George Michalopoulos, Yue Zhang, et al. Lmr-bench: Evaluating llm agent’s ability on reproducing language modeling research, 2025. URL https://arxiv. org/abs/2506.17335.

[47] Zhengqing Yuan, Kaiwen Shi, Zheyuan Zhang, Lichao Sun, Nitesh V Chawla, and Yanfang Ye. Citeaudit: You cited it, but did you read it? a benchmark for verifying scientific references in the llm era. arXiv preprint arXiv:2602.23452, 2026.

[48] Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi, Weilun Zhao, Shuo Wang, Duzhen Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. Autoreproduce: Automatic ai experiment reproduction with paper lineage. arXiv preprint arXiv:2505.20662, 2025.

[49] Minjun Zhu, Yixuan Weng, Linyi Yang, and Yue Zhang. Deepreview: Improving llm-based paper review with human-like deep thinking process. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29330–29355, 2025.

[50] Zhenzhen Zhuang, Jiandong Chen, Hongfeng Xu, Yuwen Jiang, and Jialiang Lin. Large language models for automated scholarly paper review: A survey. Information Fusion, 124: 103332, 2025.