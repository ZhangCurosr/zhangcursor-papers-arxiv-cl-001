# RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation

ZhuoXin Liu<sup>1,2,\*</sup> Zhiming Ma<sup>2,5</sup> Ying Zhang<sup>1</sup> Mengzheng Yang<sup>3</sup> Yifan Wang<sup>3</sup> Zhengqi Huang<sup>6</sup> Yanhan Zhou<sup>4</sup> Zekun Lin<sup>1</sup> Jun Zhang<sup>1</sup> Shun Zhang<sup>3</sup> Yue Chen<sup>5</sup> Qiao Zhao<sup>1,4,\*</sup> Peng Chen<sup>3</sup>

<sup>1</sup>Baidu <sup>2</sup>SmartFlowAI <sup>3</sup>People’s Public Security University of China <sup>4</sup>Tsinghua University <sup>5</sup>JD Technology <sup>6</sup>Northeastern University

## Abstract

Platform abuse campaigns conceal redirection instructions with emojis, homophones, character decomposition, and redundant symbols, then route users through disguised links to services associated with pornography, fraud, gambling, or illicit transactions. Existing benchmarks evaluate obfuscated text and risky webpages separately, obscuring how target recovery affects downstream evidence acquisition. We introduce RISKCHAINBENCH, pairing 3,600 synthetic token-text restoration inputs from 600 source sessions with 600 corresponding humanlabeled local web environments. A model first restores the message, operational intent, and destination; the same underlying model then acts as a VLM-driven web agent that investigates the correctly associated website and produces a frozen, evidence-cited risk report without message-side semantics or domain-reputation cues. We score restoration and correct-routing web investigation separately and compose them offline by applying the frozen primaryentry prediction as a gate to the same Task 2 result. Human labels determine task correctness, while a fixed multimodal evidence judge assesses faithfulness, sufficiency, completeness, and consistency. Across ten models, Entry Top-1 ranges from 35.2% to 95.2% and web decision accuracy from 26.3% to 62.8%; the leading systems differ across entry recovery, full reconstruction, website decisions, and fine-grained typing. Execution failures account for 31.9% of web runs, whereas post-decision type errors account for only 0.9%, identifying stable exploration and risk judgment as the principal bottlenecks. We release the benchmark, protocol, and resettable local sandbox.

![](images/2fd1530c9f58cee3ed1af424a468a0f703de59fd9deafaeeac625c9a4943dbf3.jpg)  
Figure 1. Offline entry-gated end-to-end accuracy on 600 websites; failures at either stage count as ⊥.

## 1 Introduction

Online platforms routinely moderate pornography, fraud, gambling, illicit transactions, and related abuse. Evaders rarely state their intent consistently in plain text: they interleave emojis with characters, replace keywords with homophones or visual lookalikes, decompose Chinese characters, and add irrelevant tokens. Such messages can remain intelligible to people while evading moderation based on surface patterns. They often include altered domains, access codes, or operational instructions that redirect users to external webpages. Reliable assessment therefore requires message restoration, target identification, web exploration, and evidence verification. We call this problem cross-channel platform risk investigation.

Prior work provides two foundations: obfuscatedcontent benchmarks study restoration or classification under character perturbations, emojis, homoglyphs, phonetic substitutions, and coded language (Tan et al., 2020; Kirk et al., 2022; Cooper, Surdeanu, and Blanco,

2023; Xiao et al., 2024; Guo et al., 2025; Ma et al., 2025; Wan, Li, and Huang, 2026), while web-agent benchmarks study reproducible interaction and safe behavior around malicious links or adversarial webpages (Zhou et al., 2024; Kong et al., 2026; Ying et al., 2026; Zhou et al., 2026). Together they leave an important platform-governance question unresolved: a small restoration error can change the destination investigated downstream, yet separate text and web evaluations cannot expose this cross-stage loss.

We introduce RISKCHAINBENCH, which links obfuscated-message restoration and evidencegrounded web investigation by destination identity. Each case contains fully synthetic, platform-formatted token-text messages and a local environment constructed offline from the corresponding webpages. The model commits to its restoration before browsing, and the top-ranked entry from a fixed primary variant determines whether the frozen web result is admitted as an end-to-end success. Redirection rhetoric and destination risk are constructed as separate attributes. The web agent receives neither the source message nor its restoration, requiring website conclusions to rest on observed web evidence.

Each model–website pair yields one investigation trajectory under the correct association. The resulting web score measures investigation for a given target and can also be composed offline with a previously committed restoration through an entry gate, without exposing message semantics to the web judgment. Figure 1 compares the correct-routing and entry-gated views. Trained annotators establish website decisions and types under a common codebook, but human annotations do not supply routine trajectory-evidence scores. A fixed multimodal evidence judge provides coverage-conditioned diagnostics of how well each conclusion is supported by its trajectory; it does not score task labels.

Task 1 evaluates the same ten underlying models on 3,600 text-only restoration inputs from 600 synthetic source sessions with six variants each. Task 2 evaluates them as VLM-driven web agents on the corresponding 600 human-labeled local websites, with one investigation per model–website pair. The experiments further analyze failures across obfuscation forms and stages of web investigation. We release the benchmark, evaluation protocol, and resettable local sandbox, which also supports subsequent agent-training research.

Our contributions are threefold:

• We formulate cross-channel platform risk investigation as an entry-linked process spanning restoration, target identification, web exploration, and evidence-grounded risk judgment.

• We construct 3,600 synthetic token-text inputs and 600 human-labeled local web environments for safe, reproducible investigation.

• We establish two-stage baselines for the same ten underlying models in text-only restoration and VLM-driven web-agent settings, and localize failures in obfuscation recovery, web execution, risk judgment, and fine-grained typing.

## 2 Related Work

Obfuscated content understanding. Evasive content uses character edits, homoglyphs, homophones, and emojis to circumvent moderation. TNT (Tan et al., 2020) studies reconstruction from character perturbations; Hatemoji (Kirk et al., 2022) and OTH (Cooper, Surdeanu, and Blanco, 2023) expose robustness failures caused by emojis and Unicode homoglyphs. Chinese benchmarks extend this setting to compositional phonetic, visual, and semantic substitutions. ToxiCloakCN (Xiao et al., 2024) evaluates cloaked offensive language, and PCR-ToxiCN (Guo et al., 2025) studies platform-observed phonetic substitutions. HomoP-CN (Ma et al., 2025) studies Chinese homophone restoration, whereas CodedLang (Wan, Li, and Huang, 2026) benchmarks coded-language detection and understanding in real-world Chinese online reviews. Jiang et al. (Jiang et al., 2026) introduce AD-VJARGON, an in-the-wild annotated dataset linking adversarial jargon variants to canonical forms, and JADE, a corresponding detection framework; we use the work only as related-work evidence and do not import its entries.

These tasks primarily end at a restored message or label; they do not measure whether a recovered destination supports subsequent investigation.

Risk-aware web agents and trajectory evaluation. WebArena (Zhou et al., 2024) provides selfhosted websites for reproducible interaction. MalURL-Bench (Kong et al., 2026) studies disguised malicious links, SecureWebArena (Ying et al., 2026) introduces adversarial web environments, and FraudSM-SWalker (Zhou et al., 2026) connects message context with safely processed web evidence while hiding reputation shortcuts.

Trajectory evaluation introduces a further challenge: AgentRewardBench (Lù et al., 2025) finds that no single LLM judge performs consistently well across its five web-agent benchmarks, Plan-RewardBench (Wang et al., 2026a) identifies degradation on longer trajectories, and REFLECT (Wang et al., 2026b) exposes weaknesses in evidence verification. RISKCHAINBENCH addresses this remaining cross-stage gap by linking destination recovery to active investigation while isolating message cues and domain reputation from website evidence. Task correctness is measured against human website labels; automated evidence scores remain separate, coverageconditioned diagnostics.

## 3 Method

RISKCHAINBENCH uses the website as its primary unit and treats messages pointing to the same site as nested variants. The i-th website instance contains a local environment $\mathcal { M } _ { i }$ , a message set $\mathcal { X } _ { i } .$ , and a website annotation $y _ { i }$

$$
\begin{array} { r l } & { \mathbb { B } = \{ b _ { i } \} _ { i = 1 } ^ { S } , } \\ & { b _ { i } = ( \mathcal { M } _ { i } , \mathcal { X } _ { i } , y _ { i } ) , } \\ & { \mathcal { X } _ { i } = \{ ( x _ { i j } ^ { * } , \widetilde { x } _ { i j } ) \} _ { j = 1 } ^ { K _ { i } } , \quad y _ { i } = ( d _ { i } , c _ { i } ) . } \end{array}\tag{1}
$$

Here, $\boldsymbol { x } _ { i j } ^ { * }$ is a canonical source message and $\widetilde { x } _ { i j }$ is one of its token-text obfuscations. The website decision is $d _ { i } \in \mathcal { D } = \{ \mathrm { V } , \mathrm { N } , \mathrm { U } \}$ , denoting violation, non-violation, and insufficient evidence. The primary type satisfies $c _ { i } \in \mathcal { C }$ when $d _ { i } = \mathrm { V } , c _ { i } = \mathrm { N O N E }$ when $d _ { i } = \mathrm { N }$ , and $c _ { i } = \mathrm { U N K N O W N }$ when $d _ { i } = \mathrm { U }$ . We fix $S = 6 0 0$ website clusters and $K _ { i } = 6$ variants for every source session, yielding $\begin{array} { r } { N = \sum _ { i } K _ { i } = 3 , 6 0 0 . } \end{array}$ The Task 1 evaluation unit is a message variant $( i , j )$ whereas the Task 2 unit is a website i, yielding 600 web cases and one trajectory per model–website pair. Offline end-to-end evaluation remains website-level: one primary variant per website is fixed in advance as the sole entry gate, and the remaining five variants participate only in Task 1. The annotation $y _ { i }$ is a property of $\mathcal { M } _ { i }$ , and no label-consistency assumption is imposed on message rhetoric. Figure 2 summarizes how restoration, website investigation, and trajectoryevidence evaluation are connected through the frozen primary-entry decision.

For each message, the restorer outputs a canonical message, an operational intent, and a ranked entry list $\widehat { Z } _ { i j } = ( \widehat { z } _ { i j } ^ { ( 1 ) } , \dots , \widehat { z } _ { i j } ^ { ( k ) } )$ . For the fixed primary variant of website i, the private resolver defines the entry gate

$$
g _ { i } = \mathbb { I } \Big [ \rho ( \widehat { z } _ { i } ^ { ( 1 ) } ) = i \Big ] .\tag{2}
$$

Separately, web investigation uses a common instruction $u ,$ a controller-generated, label-free action-class scaffold $h _ { i }$ , and the local environment $\mathcal { M } _ { i }$ . The scaffold contains only high-level interaction categories; it contains no risk label, selector, target text or value, expected state, or mandatory action order. The two branches meet only during offline end-to-end scoring through the entry gate $g _ { i }$ . For each model–website pair, the controller uses the correct association to initialize one investigation under a randomized local hostname; the private resolver uses the frozen primary-variant entry only to compute $g _ { i }$ . The controller does not expose the source message, its restoration, the original domain, or resolver output to the web agent. Web observations therefore cannot revise the committed restoration or reveal domain-reputation shortcuts.

## 3.1 Benchmark Construction

Balanced website selection. We select 600 usable scenarios from a frozen pool of 2,500 unique offline websites. A deterministic mixed-integer program uses exact quotas for presentation form, visible topic, and language, with bounded constraints on interaction depth, engineering difficulty, source stratum, and hostfamily concentration. The selected set spans 339 host families, with at most eight sites per family; 595 sites support click replay and 333 support stateful replay. It supports capability evaluation rather than prevalence estimation, and selection metadata neither determine website gold nor appear in model inputs.

Synthetic messages and obfuscation transformations. Task 1 contains 600 fully synthetic source sessions and uses no messages collected from social platforms. Each source contains a reserved-domain entry and yields six token-text variants. A deterministic pipeline composes phonetic or visual substitutions, character decomposition, redundant platformtoken insertion, and entry alteration while preserving the intended message and destination. The six recipes separately stress composite restoration, phonetic substitutions, entry confusables, mixed lexical and platform-token corruption, few-line entry layouts, and grapheme-safe vertical entry layouts. Every edit is stored in a reversible trace. One composite variant is fixed before evaluation for cross-stage scoring, while the other five evaluate restoration only. Full construction strata, recipes, and audits appear in the supplementary material. Platform-token profiles transcribe 80 text codes from six public EmojiAll secondary catalogs (EmojiAll, 2026), without redistributing images; 12 additional Bilibili codes come from researchersupplied examples. Homophone candidates and Hancharacter maps are project-curated and pronunciationchecked with pypinyin 0.54.0 (mozillazg, 2025); no third-party Chinese lexicon is imported.

![](images/22378c447b2ade3b31aec862de1c26a8b3afdb9810c02f1a596be9bb3d83dd93.jpg)  
Figure 2. Overview of RISKCHAINBENCH. A restorer recovers the message and reserved entry before browsing. The reference association initializes one controlled website investigation. The frozen predicted entry is then applied offline as a reachability gate to the same frozen result, without a second browser run. Restoration, website-task performance, and trajectory evidence are evaluated separately.

Automated checks detect malformed entries, alignment anomalies, irreversible transformations, and duplicates. Canonical destinations use unique three-label names in the reserved .test namespace, without schemes, paths, live domains, accounts, or external services. The frozen audit verifies exact reversibility, token-profile isolation, gold and trace separation, and credential hygiene. RISKCHAINBENCH is released as one evaluation set, with all variants from the same source grouped under one website. Figure 3 summarizes the construction process.

Controlled local web environments. Each web scenario is constructed offline from the corresponding real-world webpages and runs within an isolated network. It preserves the page structure, content, redirects, and interaction feedback required for risk investigation while removing dependence on the original live service. Each environment defines observable states, allowlisted actions, their resulting transitions, and a bounded interaction horizon. It specifies what an agent can observe and manipulate but encodes neither a programmatic risk label nor a unique valid evidence path.

![](images/07d34d9a2aba94e8529dc1a96f8d3164c3b6aedecd16833b47c59a51d98a704e.jpg)  
Figure 3. Benchmark construction from the frozen offline-site pool and synthetic message variants.

Publicly observable states preserve the original semantic content and interaction structure whenever possible, while local content or synthetic states replace external dependencies that cannot be reproduced safely. Each environment is checked for page availability, relevant interactions, reset consistency, and network isolation; construction and validation details appear in the supplementary material.

Human construction of website gold. Four trained annotators establish three-way website decisions and nine primary violation types under a common codebook. Each website receives two independent judgments; disagreements and any insufficient-evidence judgment trigger blind review, followed by coordinator adjudication when necessary. First-pass decision and joint-label agreement are 82.50% and 80.17%, with nominal Krippendorff’s α of 0.6553 and 0.7425. The resolution paths comprise 460 pair-consensus, 120 third-rater-majority, and 20 coordinator-adjudicated cases. Hidden repeats yield 97.9% decision agreement and 95.8% joint decision–type agreement. The frozen gold contains 394 violating, 181 non-violating, and 25 insufficient-evidence websites; full assignment and type support appear in the supplementary material.

Case pairing and quality control. Every canonical message associated with website i contains a reference entry satisfying $\rho ( z _ { i j } ^ { * } ) = i$ for all $1 \leq j \leq K _ { i }$ . Message and environment quality are checked separately, and all variants paired with a website share its annotation $y _ { i }$ . Obfuscation form, message length, entry position, and interaction depth support stratified analysis; construction strata remain separate from human labels.

## 3.2 Obfuscated Message Restoration

The restoration task gives the model only $\widetilde { x } _ { i j }$ and requires three outputs: a canonical message $\widehat { x } _ { i j }$ , an operational intent $\widehat { \iota } _ { i j }$ , and ranked entry candidates $\widehat { Z } _ { i j }$ . The model cannot access webpages, domain-reputation services, or other external information at this stage. Its output should preserve the meaning, entry, access code, and operational instructions in the source message while removing platform token strings, decomposed characters, and redundant symbols used for evasion. The restoration is frozen before any web observations are produced, and subsequent investigation cannot alter it.

Entry Top-1 exact-match rate is the primary Task 1 metric because the top-ranked entry controls the endto-end gate. Full reconstruction requires the canonical message, operational intent, and top-ranked entry to be correct. Let ED denote character-level edit distance, and let $z _ { i j } ^ { * }$ and $\iota _ { i j } ^ { * }$ denote the reference entry and intent. We report

$$
\begin{array} { r l r } & { } & { \mathrm { C E R } = \displaystyle \frac { \sum _ { i } \sum _ { j } \mathrm { E D } ( \widehat { x } _ { i j } , x _ { i j } ^ { * } ) } { \sum _ { i } \sum _ { j } | x _ { i j } ^ { * } | } , } \\ & { } & { \mathrm { E n t r y @ 1 } = \displaystyle \frac { 1 } { N } \sum _ { i } \sum _ { j } \mathrm { I } [ \rho ( \widehat { z } _ { i j } ^ { ( 1 ) } ) = i ] , } \\ & { } & { \mathrm { F R } = \displaystyle \frac { 1 } { N } \sum _ { i } \sum _ { j } \mathrm { I } \left[ \widehat { \mathcal { X } } _ { i j } = x _ { i j } ^ { * } \wedge \widehat { \iota } _ { i j } = u _ { i j } ^ { * } \right] . } \\ & { } & { \mathrm { \wedge } \rho ( \widehat { z } _ { i j } ^ { ( 1 ) } ) = i \qquad } \end{array}\tag{3}
$$

We additionally report Entry Recall@k, which credits any candidate in $\hat { \boldsymbol Z } _ { i j }$ that resolves to website i. Because entries are reserved three-label .test names without URL schemes, we report entry rather than URL metrics.

If the top-ranked entry from the fixed primary variant does not resolve to its associated website, then $g _ { i } = 0$ and the gated end-to-end output is ⊥; the frozen webonly trajectory and score remain unchanged. Lowerranked candidates do not repair the gate. The protocol applies no automatic correction and never routes an incorrect entry to another benchmark website. We report results by obfuscation form, severity, and entry position to distinguish general restoration difficulty from failures involving actionable information.

## 3.3 Evidence-Grounded Web Investigation

Web investigation is defined at the website level and runs under a common instruction u and label-free exploration guidance $h _ { i }$ . The guidance describes highlevel interaction coverage without revealing pagespecific targets or expected conclusions. The source message, canonical message, model restoration, and entry-resolution process are absent from the agent context. BrowserGym and Playwright present the current observation, the agent selects an allowlisted action conditioned on $u , h _ { i }$ , and its prior trace, and the environment logs the resulting transition. The alternating observations and actions form the bounded trajectory

$$
\tau _ { i } = ( o _ { 0 } , a _ { 0 } , \ldots , a _ { T _ { i } - 1 } , o _ { T _ { i } } ) , \qquad T _ { i } \leq H ,\tag{4}
$$

which is frozen when the agent stops or reaches the interaction budget.

After the investigation trajectory is frozen, the same tested model receives the common instruction, site guidance, and recorded trajectory to produce a frozen risk conclusion $\widehat { y } _ { i } = ( \widehat { d } _ { i } , \widehat { c } _ { i } )$ , a rationale ${ \widehat { r } } _ { i } ,$ and evidence references $\widehat { E } _ { i } = \{ ( t _ { k } , \ell _ { k } ) \} _ { k = 1 } ^ { L _ { i } }$ without rerunning the browser. Here $\widehat { d } _ { i } \in { \mathcal { D } } \cup \{ \perp \}$ . A prediction of U is a valid semantic decision that the observed environment provides insufficient evidence. In the webonly view, ⊥ is assigned only when no valid task decision is available because of a system failure, timeout, or malformed output; an entry-gate failure produces ⊥ only in the gated view of Eq. 5. The predicted type follows the same compatibility constraints as the gold type when $\widehat { d } _ { i } \in \mathcal { D }$ . In each evidence reference, $t _ { k }$ identifies an observed trajectory step and $\ell _ { k }$ locates the supporting content. Only observations recorded in the trajectory are admissible as evidence. The supplementary material specifies evidence-admission criteria and exceptional outcomes.

This input boundary makes the website judgment depend on evidence the agent actually observes. Message-side risk cues cannot directly determine website classification, allowing non-violating destinations paired with suggestive redirection rhetoric to remain meaningful hard negatives.

![](images/6f0ec2fd5cd9f85a7cf62568952755da9facfc8e5dc202f5a836ff98dccba359.jpg)  
Figure 4. Web-agent evidence collection and independent evaluation. Sampled human evidence audit is a validation-only branch when available; it is not used in the reported routine scores, and no four-dimensional human-audit result is claimed here.

The agent may report violation, non-violation, or insufficient evidence, all of which belong to the task decision space. Unreachable pages, environment blocks, exhausted action budgets, and system exceptions are recorded separately; only runs without a valid task decision receive ⊥. Detailed rules appear in the supplementary material.

## 3.4 Evaluation Protocol

Web-only and gated end-to-end views. Each tested model investigates each website once. The web-only view uses the correct website association to evaluate exploration, evidence acquisition, and risk judgment. The gated end-to-end view reuses that frozen result and counts it as successful only when the entry recovered from the fixed primary variant identifies the website. Thus the logical protocol enters the web stage only after a successful entry gate, whereas the implementation computes each reusable web trajectory once and applies the same gate offline. Figure 4 summarizes the independent scoring branches. The controller uses the reference association to initialize the environment without showing the entry or canonical message to the

agent. We define

$$
\begin{array} { r } { \widehat { y } _ { i } ^ { w } = \widehat { y } _ { i } , \qquad \widehat { y } _ { i } ^ { e } = \displaystyle \int \widehat { y } _ { i } , \quad g _ { i } = 1 , } \\ { \bot , \quad g _ { i } = 0 , } \end{array}\tag{5}
$$

where w and e denote the web-only and gated end-toend views. The entry-gate pass rate is $\mathrm { P a s s } _ { \mathrm { e n t r y } } =$ $S ^ { - 1 } \sum _ { i = 1 } ^ { S } g _ { i }$ . This is an admission rate for offline composition, not a measure of whether the local website itself is reachable. It retains missing or incorrect entries without routing them into another case.

Protocol validity. A deterministic validator $V ( \tau , E ) \in \{ 0 , 1 \}$ checks that actions remain inside the permitted environment and citations resolve to observed content. It does not infer webpage risk. Invalid runs are reported separately and count as task failures; complete rules appear in the supplementary material.

Task correctness. For $q \in \{ w , e \}$ , decision accuracy and hierarchical exact match are

$$
\begin{array} { r } { \mathrm { A c c } _ { d } ^ { q } = \cfrac { 1 } { S } \displaystyle \sum _ { i = 1 } ^ { S } \mathbb { I } [ \widehat { d } _ { i } ^ { q } = d _ { i } ] , } \\ { \mathrm { H E M } ^ { q } = \displaystyle \frac { 1 } { S } \displaystyle \sum _ { i = 1 } ^ { S } \mathbb { I } [ \widehat { y } _ { i } ^ { q } = y _ { i } ] . } \end{array}\tag{6}
$$

Decision macro-F1 uses the same three-class confusion matrix. Decision accuracy, decision macro-F1, and hierarchical exact match retain all $S = 6 0 0$ websites, including runs without a valid task decision. Binary accuracy is computed on websites with gold V or N. Violation-type macro-F1 is computed on the 394 gold violations over codebook types represented in the frozen gold; six of the nine types have nonzero support. A prediction of U is correct only against gold U; ⊥ is always a failure. Per-class recall, coverage, system failures, environment failures, and protocol validity are reported separately. Metric definitions appear in the supplementary material.

Multidimensional multimodal evidence judge. We use a fixed multimodal evidence judge $J _ { \phi }$ only to evaluate the relation between the reported conclusion and the recorded investigation. The judge receives a sanitized trajectory package comprising the action ledger, recorded trajectory observations, admitted screenshots, cited evidence, final conclusion, and rationale, and returns

$$
\begin{array} { r } { \mathbf { s } _ { i } = J _ { \phi } ( \tau _ { i } , \widehat { E } _ { i } , \widehat { y } _ { i } , \widehat { r } _ { i } ) = ( s _ { i 1 } , s _ { i 2 } , s _ { i 3 } , s _ { i 4 } ) . } \end{array}\tag{7}
$$

The four components measure evidence faithfulness, evidence sufficiency, investigation completeness, and reasoning consistency, respectively. Their unweighted arithmetic mean is reported only as a compact evidence diagnostic. The judge does not receive website annotations, model identity, URLs or domain-reputation signals, or hidden model reasoning, and it does not determine the website decision or violation type. Its model version, prompt, decoding configuration, and rubric remain fixed across all tested systems. The judge evaluates each frozen website trajectory once.

Entry-gated end-to-end loss. The web-only view measures website investigation under correct routing, whereas the gated end-to-end view additionally retains entry-recovery failures. Their difference in three-way decision accuracy is

$$
\Delta _ { \mathrm { g a t e } } = \operatorname { A c c } _ { d } ^ { w } - \operatorname { A c c } _ { d } ^ { e } .\tag{8}
$$

This quantity is reported in percentage points and attributes the end-to-end loss to entry recovery rather than changes in web investigation.

Restoration results cover all N messages, and web results cover all S websites; when case-aligned restoration outputs are supplied, entry-gated composition also operates over the S website units. Message-level and website-level results are reported separately, and multiple message variants of one website do not create additional website observations.

## 4 Experiments

## 4.1 Experimental Setup

Data and systems. The evaluation contains 600 websites, one synthetic source session per website, and six variants per session, totaling 3,600 restoration inputs. The fixed primary variant supplies the sole offline entry gate; the remaining five variants participate only in Task 1. The two panels of Table 1 each report ten completed model runs. Gemini 3.6 Flash is unranked in Task 2 because it was absent from the frozen report and judge batches. Each model–website pair contributes one BrowserGym–Playwright trajectory under a 30-action and 600-second budget, with immediate termination after an early valid report. All Task 2 systems use the same frozen website set, local-only interaction boundary, and reporting schema.

Metrics. Task 1 is ranked by Entry Top-1 and also reports full reconstruction, CER, and Entry Recall@k. Website evaluation reports three-way decision accuracy and macro-F1, violation-type macro-F1, and hierarchical exact match. Runs without a valid task decision retain their frozen failure status. Evidence scores are conditional on a successful fixed multimodal-judge evaluation of a valid source trajectory and are therefore interpreted together with judge coverage. Primary metrics use central 95% intervals from 2,000 website-level bootstrap resamples. These intervals quantify websitecomposition uncertainty; because each model–website pair is run once, they do not estimate rerun variance.

Evaluated model versions follow official provider documentation (OpenAI, 2025, 2026b,a; Anthropic, 2026a,b; Moonshot AI, 2026a,b,c; Alibaba Cloud, 2026; ByteDance Seed, 2026; Google, 2026).

## 4.2 Main Results

Message restoration. The upper panel of Table 1 reports the formal results of ten models over all 3,600 messages. Models are ordered by Entry Top-1, and full reconstruction requires the canonical message, operational intent, and top-ranked entry to be correct. All six variants remain in the evaluation, and modeloutput failures remain in the denominator.

Entry Top-1 ranges from 35.19% to 95.22%. GPT-5.4 leads Entry Top-1, whereas GPT-5.6 SOL leads full reconstruction and has the lowest CER. Kimi K3 combines low CER with a lower Entry Top-1, confirming that character-level recovery cannot replace a separate evaluation of actionable-entry recovery.

Full stratified results by obfuscation group and all resampling intervals appear in the supplementary material.

Evidence-grounded web investigation. The webonly view evaluates exploration, evidence acquisition, and final judgment under correct website binding. Task correctness in the lower panel of Table 1 is scored directly against human website gold labels, while the fixed multimodal evidence judge evaluates only the evidential relation between the conclusion and the observed trajectory. Runs without valid decisions remain failures.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Exec. ReportDec.</td><td rowspan=1 colspan=2>Type</td><td rowspan=1 colspan=1>H-EM</td></tr><tr><td rowspan=1 colspan=1>∑ Pooled</td><td rowspan=1 colspan=1>31.9</td><td rowspan=1 colspan=1>8.2</td><td rowspan=1 colspan=1>12.3</td><td rowspan=1 colspan=1>0.9</td><td></td><td rowspan=1 colspan=1>46.6</td></tr><tr><td rowspan=1 colspan=1>SGPT-5.6 SOL</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>15.8</td><td rowspan=1 colspan=1>1.8</td><td></td><td rowspan=1 colspan=1>61.0</td></tr><tr><td rowspan=2 colspan=1>SGPT-5.2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>20.7</td><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>20.2</td><td rowspan=1 colspan=1>1.0</td><td></td><td rowspan=1 colspan=1>56.3</td></tr><tr><td rowspan=1 colspan=1>K&#x27;Kimi K2.5</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>0.8</td><td></td><td rowspan=1 colspan=1>54.3</td></tr><tr><td rowspan=4 colspan=1>SGPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>21.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>10.2</td><td rowspan=2 colspan=1>14.3</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.5</td><td></td><td rowspan=1 colspan=1>53.2</td></tr><tr><td rowspan=1 colspan=1>立Qwen3.6</td><td rowspan=1 colspan=1>33.0</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>0.7</td><td></td><td rowspan=1 colspan=1>51.0</td></tr><tr><td rowspan=2 colspan=1>米Opus 4.8K&#x27;Kimi K3</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>2.3</td><td></td><td rowspan=1 colspan=1>47.5</td></tr><tr><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>1.0</td><td></td><td rowspan=1 colspan=1>46.5</td></tr><tr><td rowspan=1 colspan=1>K&#x27;Kimi K2.6</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>0.8</td><td></td><td rowspan=1 colspan=1>40.7</td></tr><tr><td rowspan=3 colspan=1>0Doubao 2.0Sonnet 5</td><td rowspan=1 colspan=1>57.7</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>6.8</td><td rowspan=1 colspan=1>0.0</td><td></td><td rowspan=1 colspan=1>29.5</td></tr><tr><td rowspan=1 colspan=1>42.8</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>9.8</td><td rowspan=1 colspan=2>0.2</td><td rowspan=1 colspan=1>26.2</td></tr><tr><td rowspan=1 colspan=6>Share of runs (%)</td></tr></table>

Figure 5. Task 2 first-failure attribution over all 600 runs per model; H-EM denotes hierarchical success.

Task 2 decision accuracy ranges from 26.3% to 62.8%, and hierarchical exact match ranges from 26.2% to 61.0%. GPT-5.6 SOL leads both metrics and decision macro-F1; Kimi K2.5 leads type macro-F1.

Evidence scores remain coverage-conditioned and do not define an overall ranking; complete coverage and four-dimensional results appear in the supplement.

Entry-gated end-to-end composition. Figure 1 and Table 2 apply the fixed primary-entry gate to the same frozen web results. Gated accuracy ranges from 16.7% to 60.8%, with losses of 0.7–32.3 percentage points. The composition reuses each frozen Task 2 trajectory without a browser rerun.

## 4.3 Diagnostic Analysis

We assign each run to its first failed stage: execution, missing report, decision, or type; runs with both a correct decision and type are hierarchical successes.

Figure 5 shows that execution is the largest pooled failure source at 31.9%; only 0.9% fail at typing after a correct decision.

(a) Task 1: message restoration
<table><tr><td>System</td><td>Entry Top-1 ↑</td><td>Full ↑ CER↓</td></tr><tr><td>GPT-5.4</td><td>95.22</td><td>66.06 1.62</td></tr><tr><td>GPT-5.6 SOL</td><td>94.31</td><td>73.31 1.11</td></tr><tr><td>Claude Opus 4.8</td><td>92.94</td><td>65.72 1.64</td></tr><tr><td>Kimi K3</td><td>84.58</td><td>65.64 1.59</td></tr><tr><td>GPT-5.2</td><td>73.39</td><td>45.72 16.01</td></tr><tr><td>Claude Sonnet 5</td><td>72.17</td><td>43.72 10.69</td></tr><tr><td>Doubao Seed 2.0</td><td>62.86</td><td>38.36 14.14</td></tr><tr><td>Kimi K2.5</td><td>39.86</td><td>19.94 17.11</td></tr><tr><td>Qwen3.6 Plus</td><td>39.06</td><td>25.33 11.79</td></tr><tr><td>Kimi K2.6</td><td>35.19</td><td>16.06 20.19</td></tr></table>

(b) Task 2: web investigation
<table><tr><td>System</td><td>Acc. ↑</td><td>Dec. F1 ↑</td><td>Type F1 ↑</td><td>H-EM↑</td></tr><tr><td>GPT-5.6 SOL</td><td>62.8</td><td>59.3</td><td>42.8</td><td>61.0</td></tr><tr><td>GPT-5.2</td><td>57.3</td><td>51.2</td><td>28.0</td><td>56.3</td></tr><tr><td>Kimi K2.5</td><td>55.2</td><td>48.8</td><td>48.1</td><td>54.3</td></tr><tr><td>GPT-5.4</td><td>53.7</td><td>51.7</td><td>26.4</td><td>53.2</td></tr><tr><td>Qwen3.6 Plus</td><td>51.7</td><td>50.2</td><td>44.1</td><td>51.0</td></tr><tr><td>Claude Opus 4.8</td><td>49.8</td><td>51.4</td><td>40.6</td><td>47.5</td></tr><tr><td>Kimi K3</td><td>47.5</td><td>52.4</td><td>40.8</td><td>46.5</td></tr><tr><td>Kimi K2.6</td><td>41.5</td><td>51.0</td><td>33.7</td><td>40.7</td></tr><tr><td>Doubao Seed 2.0</td><td>29.5</td><td>38.0</td><td>32.6</td><td>29.5</td></tr><tr><td>Claude Sonnet 5</td><td>26.3</td><td>36.5</td><td>26.0</td><td>26.2</td></tr></table>

Table 1. Task 1 and Task 2 results (%). Bold denotes the best and underlining the second-best result within each metric. The panels remain separate and are ranked independently by Entry Top-1 and three-way decision accuracy. Task 1 uses all 3,600 messages. Task 2 accuracy, decision macro-F1, and H-EM retain all 600 websites and count ⊥ as incorrect; Type F1 is computed on the 394 gold-violation websites.

## 5 Conclusion

RISKCHAINBENCH links obfuscated-message restoration and evidence-grounded web investigation in a resettable local sandbox. Across the same ten models, actionable-entry recovery and full reconstruction rank models differently, as do website decisions and fine-grained violation typing. The offline entry gate exposes upstream loss without rerunning website investigation; frequent execution failures show that reliable exploration remains prerequisite to evidence-grounded judgment.

The results use one trajectory per model–website pair, sparsely populate several violation types, and condition evidence diagnostics on successful judge coverage. Future work should measure rerun variance, broaden underrepresented risk types and languages, and validate the evidence rubric through a frozen human trajectory audit within the same isolated environments.

## 6 Ethical Statement

Synthetic, non-routable entries and isolated environments avoid contact with live services. Local substitutes replace personal identities, payment flows, account operations, and communication with third parties. Evidence packages retain only sanitized observations needed for evaluation. Benchmark outputs do not establish legal attribution.

<table><tr><td>System</td><td>Web-only Acc. ↑</td><td>Gated Acc. ↑</td><td>Loss (pp) ↓</td></tr><tr><td>GPT-5.6 SOL</td><td>62.8</td><td>60.8</td><td>2.0</td></tr><tr><td>GPT-5.4</td><td>53.7</td><td>52.3</td><td>1.3</td></tr><tr><td>Claude Opus 4.8</td><td>49.8</td><td>49.2</td><td>0.7</td></tr><tr><td>GPT-5.2</td><td>57.3</td><td>49.0</td><td>8.3</td></tr><tr><td>Kimi K3</td><td>47.5</td><td>44.5</td><td>3.0</td></tr><tr><td>Kimi K2.5</td><td>55.2</td><td>25.7</td><td>29.5</td></tr><tr><td>Qwen3.6 Plus</td><td>51.7</td><td>19.3</td><td>32.3</td></tr><tr><td>Claude Sonnet 5</td><td>26.3</td><td>19.2</td><td>7.2</td></tr><tr><td>Doubao Seed 2.0</td><td>29.5</td><td>18.2</td><td>11.3</td></tr><tr><td>Kimi K2.6</td><td>41.5</td><td>16.7</td><td>24.8</td></tr></table>

Table 2. Correct-routing Task 2 accuracy and entry-gated end-to-end accuracy (%). All values use the same 600 websites per model. Loss is the difference in percentage points; 95% intervals appear in the supplementary material.

## References

Alibaba Cloud. 2026. Qwen3.6-Plus Model Documentation. https://www.alibabacloud.com/help/en/ model-studio/vision-model. Official model documentation.

Anthropic. 2026a. Claude Opus 4.8. https://www. anthropic.com/research/claude-opus-4-8. Official model documentation.

Anthropic. 2026b. Claude Sonnet 5. https:// www.anthropic.com/news/claude-sonnet-5. Official model documentation.

ByteDance Seed. 2026. Doubao Seed 2.0. https://developer.volcengine.com/articles/ 7588678603088019493. Official model documentation.

Cooper, P.; Surdeanu, M.; and Blanco, E. 2023. Hiding in Plain Sight: Tweets with Hate Speech Masked by Homoglyphs. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, 2922– 2929. Association for Computational Linguistics.

EmojiAll. 2026. Emoji Platform Lists. https://www. emojiall.com/zh-hans/platform-list. Accessed July 20, 2026.

Google. 2026. Gemini 3.6 Flash. https://ai.google.dev/ gemini-api/docs/models/gemini-3.6-flash. Official model documentation.

Guo, H.; He, J.; Ma, J.; Na, H.; Wang, Z.; Zhang, H.; Chen, Q.; Wang, W.; Shi, Z.; Shen, T.; and Chen, L. 2025. Lost in Pronunciation: Detecting Chinese Offensive Language Disguised by Phonetic Cloaking Replacement. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, 2538–2550. Association for Computational Linguistics.

Jiang, Z.; Liu, M.; Qin, Y.; and Liu, B. 2026. Breaking Free from Ivory Tower: Evaluating and Enhancing Real-World Chinese Underground Adversarial Jargon Detection. In 2026 IEEE Symposium on Security and Privacy, 417–435.

Kirk, H.; Vidgen, B.; Rottger, P.; Thrush, T.; and Hale, S. 2022. Hatemoji: A Test Suite and Adversarially-Generated Dataset for Benchmarking and Detecting Emoji-Based Hate. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 1352–1368. Association for Computational Linguistics.

Kong, D.; Wu, Z.; Liu, S.; Tan, Z.; Lu, K.; Li, M.; Liu, Q.; Chu, S.; Xu, Z.; Liu, X.; and Han, M. 2026. MalURLBench: A Benchmark Evaluating Agents Vulnerabilities When Processing Web URLs. In Findings of the Association for Computational Linguistics: ACL 2026, 14589–14601. Association for Computational Linguistics.

Lù, X. H.; Kazemnejad, A.; Meade, N.; Patel, A.; Shin, D.; Zambrano, A.; Stanczak, K.; Shaw, P.; Pal,´ C. J.; and Reddy, S. 2025. AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. arXiv:2504.08942.

Ma, J.; Feng, Z.; Song, H.; Chersoni, E.; and Chen, Z. 2025. Reasoning or Memorization? Investigating LLMs’ Capability in Restoring Chinese Internet Homophones. In Proceedings of the 3rd Workshop on Towards Knowledgeable Foundation Models, 120– 139. Association for Computational Linguistics.

Moonshot AI. 2026a. Kimi K2.5. https://www.kimi. com/blog/kimi-k2-5. Official model documentation.

Moonshot AI. 2026b. Kimi K2.6. https://www.kimi. com/blog/kimi-k2-6. Official model documentation.

Moonshot AI. 2026c. Kimi K3. https://www.kimi. com/blog/kimi-k3. Official model documentation.

mozillazg. 2025. pypinyin: Convert Chinese Characters to Pinyin. https://github.com/mozillazg/pythonpinyin/tree/v0.54.0. Version 0.54.0.

OpenAI. 2025. Introducing GPT-5.2. https://openai. com/index/introducing-gpt-5-2/. Official model documentation.

OpenAI. 2026a. GPT-5.6 System Card. https: //deploymentsafety.openai.com/gpt-5-6. Official model documentation.

OpenAI. 2026b. Introducing GPT-5.4. https://openai. com/index/introducing-gpt-5-4/. Official model documentation.

Tan, F.; Hu, Y.; Hu, C.; Li, K.; and Yen, K. 2020. TNT: Text Normalization Based Pre-Training of Transformers for Content Moderation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 4735–4741. Association for Computational Linguistics.

Wan, R.; Li, C.; and Huang, T.-H. K. 2026. “Newspaper Eat” Means “Not Tasty”: A Taxonomy and Benchmark for Coded Language in Real-World Chinese Online Reviews. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 9433–9446. Association for Computational Linguistics.

Wang, J.; Hu, Y.; Yang, W.; Pan, Z.; Li, X.; and Guo, L.-Z. 2026a. Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 23174–23200. Association for Computational Linguistics.

Wang, L.; He, Y.; Chen, P.; Yehudai, A.; Liu, Y.; Ying, R.; Shmueli-Scheuer, M.; and Cohan, A. 2026b. Time to REFLECT: Can We Trust LLM Judges for Evidence-Based Research Agents? arXiv:2605.19196.

Xiao, Y.; Hu, Y.; Choo, K. T. W.; and Lee, R. K.-W. 2024. ToxiCloakCN: Evaluating Robustness of Offensive Language Detection in Chinese with Cloaking Perturbations. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 6012–6025. Association for Computational Linguistics.

Ying, Z.; Shao, Y.; Gan, J.; Xu, G.; Zhang, W.; Zou,Q.; Shi, J.; Yin, Z.; Zhang, M.; Liu, A.; and Liu, X.

2026. SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-Based Web Agents. In Findings of the Association for Computational Linguistics: ACL 2026, 11986–11998. Association for Computational Linguistics.

Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; Sridhar, A.; Cheng, X.; Ou, T.; Bisk, Y.; Fried, D.; Alon, U.; and Neubig, G. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations.

Zhou, Y. H.; Ma, Z. M.; Zhou, Y. J.; Li, Y. T.; Xiang, H. X.; Cheng, Y. M.; Chen, T. L.; Zhang, K. J.; Nan, Z. H.; Ni, J. H.; Wu, Z.; Pan, Q. Y.; Zhang, S.; Cheng, S.; and Luo, M. Y. 2026. FraudSM-SWalker: Benchmarking Agentic Large Language Models for SMS-to-Webpage Fraud Detection. arXiv:2606.16659.