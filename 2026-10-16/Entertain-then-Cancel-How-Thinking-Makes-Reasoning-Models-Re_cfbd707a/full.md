# Entertain, then Cancel: How Thinking Makes Reasoning Models Read Some and Or Literally

Zhizhuo Kou<sup>1</sup> Jian Yang <sup>1</sup> Pengfei Hao<sup>2</sup> Junyu Luo <sup>3</sup>

Jingshu Peng <sup>1</sup> Sirui Han<sup>1</sup>\* Yike Guo<sup>1</sup>

<sup>1</sup>The Hong Kong University of Science and Technology

<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>3</sup> Peking University

zkouaa@connect.ust.hk, siruihan@ust.hk

## Abstract

When a speaker says “some students passed”, listeners infer that not all of them did. Reasoning models now let users decide how long a model thinks before it answers, but we do not know what extra thinking does to such everyday inferences. We study this directly: we keep a model’s weights fixed, change only how much it thinks, and record which reading it chooses. A second tool cuts the model’s own reasoning at successive points and asks for an answer, showing where the reading changes. Thinking makes models read logical words as logic: it cancels the implicatures of some (not all) and or (not both) in most tested open reasoning models and in GPT-5, while context-driven inferences such as indirect answers improve. The trace shows why: models first entertain the enriched reading and cancel it later. Answering from that early reading, an exploratory inference-time remedy, recovers these implicatures without losing literal accuracy. Evaluations of reasoning models should measure how they interpret, not only how accurate they are. Code and data: https://github.com/ kouzhizhuo/entertain-then-cancel.

## 1 Introduction

A listener hearing “Some boys play” usually concludes that not all of them do. Although some remains logically true even if every boy plays, listeners reason that a speaker knowing they all play would have said all. This step from what is said to what is meant is pragmatic enrichment, and the inference from some to ‘not all’ is the textbook case (Grice, 1975). Reasoning models now treat thinking as a setting the user controls: a mode that can be switched on or off, a budget of thinking tokens, or an effort level (Yang et al., 2025; OpenAI et al., 2025; DeepSeek-AI, 2025). More thinking usually helps on mathematics and code. Does it also help a model understand what a speaker means? Existing evaluations score whether a model recovers an intended meaning (Hu et al., 2023a; Ruis et al., 2023; Sravanthi et al., 2024). Recent studies report that more thinking can hurt theory of mind (Gong et al., 2026) and that Qwen3 recognizes fewer scalar implicatures in thinking mode (Spinoso-Di Piano et al., 2026). No study yet measures what deliberation does to the reading a model chooses when both readings are allowed.

![](images/a4ac45970090d6f189fdd93ab4bdcb1b7141ee46cfcf1886b47c9ad873d0c6b5.jpg)  
Figure 1: Thinking acts as a logic lens on logical words. some and or are read as ∃x and ∨, while adequate keeps ‘not good’.

Human pragmatics offers two competing predictions. On the effort account, enrichment costs effort, so more thinking should yield more of it (Sperber and Wilson, 1986; Bott and Noveck, 2004). On the default-and-cancel account, an implicature such as some → ‘not all’ arises by default and is cancelled when the context or a check of the literal meaning speaks against it (Levinson, 2000).

Our idea is simple. Because thinking is a setting, we can hold a model’s weights fixed and turn only this dial, as in a dose–response experiment, and record which reading the model chooses (Figure 1). To see where the reading changes, we cut the model’s own reasoning text, its trace, after its first t tokens and ask for an answer at each cut. We apply both tools to 40 open and seven closed reasoning models, on items that separate logical words (some, or, modals) from lexical scales (good/excellent), presuppositions and indirect answers. The headline is that thinking makes models read logical words as logic, while inferences that need the context improve; along the trace, models entertain the enriched reading early and cancel it.

Our contributions are as follows: ❶ Thinking Cancels Implicature: Extended reasoning cancels scalar implicatures for some and or across 30/40 open models and GPT-5, as well as in label-free three-way choices (21/24 models). ❷ Diagnostic Protocol: We introduce a fixed-weight framework combining dose–response profiling with a traceprefix diagnostic to track interpretation dynamics while controlling for yes-bias. ❸ Entertain, Then Cancel: Models typically entertain pragmatic readings in early reasoning tokens before switching to literal readings in completed outputs. ❹ Early-Trace Extraction: Extracting answers after the first 128 trace tokens improves accuracy on scalar felicity items across two 8B models without compromising literal performance. ❺ Released Resources: We release new Bott–Noveck sentences with expert-checked speaker-knowledge scenarios, expanded IMPPRES test sets, full trace logs, and machine-translated German/Chinese slices.

## 2 Background and Related Work

Two kinds of implicature. Grice (1975) separated generalized implicatures, which a word carries unless the context blocks them (some → ‘not all’, or → ‘not both’), from particularized ones, which arise only in a specific context (“I wore gloves” in reply to “Did you leave fingerprints?” means ‘no’). Generalized implicatures attach both to logical words and to lexical scales such as good/excellent, whose enrichment rates vary widely across scales (van Tiel et al., 2016).

Three predictions for a thinking model. The effort account treats every implicature as an inference that costs processing effort (Sperber and Wilson, 1986). Adults who reject “Some elephants are mammals” respond more slowly, and they reject it less often under time pressure or memory load (Bott and Noveck, 2004; De Neys and Schaeken, 2007; Marty and Chemla, 2013), although this is contested (Grodner et al., 2010; Degen and Tanenhaus, 2015). If thinking tokens act as processing resources, more thinking should mean more enrichment. The default-and-cancel account holds generalized implicatures as defaults that a listener cancels when the context or a check of the truth conditions speaks against them (Levinson, 2000). If a model spends its thinking on checking strict entailment, it should cancel the implicatures of logical words, while particularized implicatures, reachable only by inference, should hold or improve. A third, developmental view notes that children accept underinformative some more frequently than adults (Noveck, 2001). If enrichment tracks competence, thinking may amplify a model’s existing preference, with a direction that changes with capability. We use default for the reading chosen without thinking and cancel for a shift towards the literal reading as thinking grows, without committing to a cognitive mechanism. Section D.1 lists the pre-registered hypotheses and falsifiers.

![](images/e778256810582f65aab2e14e5485e084063033e42e7579dde835f81654221116.jpg)  
Figure 2: Thinking makes some more literal in most open reasoning models. Vermilion: counted literal shift; blue: more pragmatic (Figure 11).

Prior evidence in language models. Chain-ofthought prompts improve implicature resolution (Kim et al., 2023; Ruis et al., 2023), yet chain-ofthought was also seen to “encourage literal interpretations” (Park et al., 2024). More reasoning lowers theory-of-mind accuracy (Gong et al., 2026), and in ImplicatureX Qwen3’s thinking mode recognises fewer scalar implicatures and more cancellations at one budget (Spinoso-Di Piano et al., 2026). These studies measure accuracy at fixed settings. We measure the reading chosen as a dose–response on fixed weights, separate implicature types, and locate the change in the trace and in training (Section E.1).

## 3 Method

As shows in (Figure 3): a dose protocol that changes only how much a model thinks, and a trace diagnostic that shows where its reading changes.

![](images/77b5c8396aba1166726d2d2c4fc67101d4be94783a4bb577dacb2416a32090ef.jpg)  
Figure 3: A reading probe on fixed weights. Items with a trigger (①) are answered with thinking off, capped or natural (②); we read the chosen reading R and the log-odds L (③), also after successive trace prefixes (④), for every model and checkpoint.

① Items and framings. Each item is a sentence with a trigger and a question about what the speaker meant. In our running example a speaker says “Some boys play”, and the model is asked whether it would conclude that the speaker believes “Not all boys play”; ‘yes’ is the pragmatic reading and ‘no’ the literal one. Logical-word items (some, or, modals) come from IMPPRES (Jeretic et al., 2020); lexical-scale items from scalar-diversity sets with human enrichment rates (van Tiel et al., 2016; Gotzner et al., 2018); presupposition items from IMPPRES; and particularized implicatures from LUDWIG (Ruis et al., 2023), PragMega (Hu et al., 2023a), Fig-QA (Liu et al., 2022), PUB (Sravanthi et al., 2024) and Circa (Louis et al., 2020), with NOPE as a second presupposition set (Parrish et al., 2021). Primary framings ask what the speaker meant or whether the sentence is an adequate description (felicity). A true/false question serves as a logic-mode comparison, as the literal answer is strictly correct in that context. Options are shuffled on every trial. Headline numbers are computed on a held-out half of the items; replications on full released sets are marked as such.

② Thinking. The same weights answer with thinking off, with thinking stopped after a budget of B tokens, or with natural stopping, subject to an 8,192-token safety cap (NAT). For Qwen3 (Yang et al., 2025), “off” is the model’s own switch, which differs from the thinking template only by an empty thinking block; for gpt-oss we use its low and high reasoning effort (OpenAI et al., 2025). Capped budgets append either Qwen3’s own stop phrase or only the closing tag, because the stop phrase mentions limited time.

③ Readouts. Each answer ends in a line Answer: <option>; anything else counts as a hedge. With $r _ { i m k }$ the share of committed answers that take the pragmatic reading for item i, model m and condition k, the reading chosen is

$$
R _ { m k } = \frac { 1 } { | I | } \sum _ { i \in I } r _ { i m k } ,\tag{1}
$$

and the thinking effect is $\Delta R = R _ { \mathrm { { N A T } } } - R _ { \mathrm { { o f f } } } ,$ , in percentage points. For particularized implicatures and presuppositions R is accuracy. A second readout needs no answer parsing: after the trace y we force the answer prefix and compare the two options,

$$
L _ { i k } \ = \ \log { \frac { p ( { \mathrm { p r a g m a t i c } } \mid x _ { i } , y _ { i k } ) } { p ( { \mathrm { l i t e r a l } } \mid x _ { i } , y _ { i k } ) } } .\tag{2}
$$

On category sentences we also compute signaldetection sensitivity $d ^ { \prime } = z ( H ) - z ( F )$ and criterion c from hits and false alarms on control sentences (Section B.1).

④ Trace. To see where the reading changes, we cut each natural trace after its first t tokens, $t \in \{ 0 , 3 2 , \ldots , 2 0 4 8 \}$ , close it, and read R and L from that prefix $\mathbf { \mathcal { Y } } { \le } t$ . The curve R(t) describes, on average over items, how the reading endorsed after a prefix changes along the trace; it is an intervention on the trace, not a readout of a hidden interpretation. Per trace, the diagnostic labels an entertain-then-cancel trajectory when the answer forced at an early prefix is pragmatic and the final answer is literal, and a literal-throughout trajectory when no prefix yields the pragmatic answer. The probe is anchored at both ends: with an empty prefix it reproduces the model’s no-thinking reading (35.8% against 38.8% for some in Qwen3-8B), and every item’s intended reading comes from source gold or, for the speaker-knowledge scenarios, from a blind author check that chose the intended reading in all 24 checked scenarios.

Models. The primary models are Qwen3 from 0.6B to 32B, whose thinking switch leaves everything else fixed; OLMo-3-Think 7B and 32B and gpt-oss 20b and 120b add two families with samerecipe size pairs (Team Olmo et al., 2025; OpenAI et al., 2025). Ten further open reasoning models and seven closed models widen the panel. To trace the origin of the effect we score checkpoints along base → SFT → DPO → RL lineages (Team OLMo et al., 2024; Team Olmo et al., 2025; Lambert et al., 2024). Sections A.1 and B.1 give model identifiers, prompts and all settings.

Statistics. Each condition runs with three decoding seeds at identical sampling settings. Intervals are 95% cluster-bootstrap confidence intervals over items, paired across conditions; a difference is significant when its interval excludes 0. The preregistered tests are a logistic model of the pragmatic reading on the log thinking budget, the phenomenon class and their interaction, fitted with item-clustered robust errors in place of the planned random effects; equivalence tests at ±3 pp; and Holm correction within each model family (Section D.3). Several pre-registered predictions did not hold; Section D.2 reports every outcome.

## 4 Thinking Makes Some and Or More Literal

This section shows that thinking makes logical words literal, that the shift survives every analysis choice, where in the trace it happens, and at which training stage the literal default appears.

## 4.1 Thinking Cancels the Implicatures of Some and Or

Thinking reduces the pragmatic reading of some, or, or both in most models, whereas other inference types behave differently. Table 1 reports the major open families (see also Figure 9), while

Figure 2 and Figure 8 detail the item families. Each effect compares natural thinking with thinking off, or with a forced-empty thinking block for models without a switch; for gpt-oss and closed models it compares the highest with the lowest dose. Natural thinking changes the pragmatic reading of some in Qwen3-8B by −28.2 pp (95% CI −41.0 to −16.7) and that of or by −36.4 pp; in Qwen3-32B the changes are −51.3 and −46.5 pp. High reasoning effort does the same in gpt-oss-20b (−30.8 and −24.7 pp against low effort), and in OLMo-3-32B-Think the some reading changes by −11.5 pp. It holds in DeepSeek’s distilled thinkers too, which always think: against a forced-empty thinking block, natural thinking changes the some reading of R1-Distill-Llama-8B by −50.0 pp and of R1-Distill-Llama-70B by −50.0 pp, and the newest distilled model, DeepSeek-R1-0528-Qwen3-8B, by −24.4 pp.

Across 40 open reasoning models, the shift on some or or is significant (95% CI excluding 0, uncorrected) and survives the pre-declared format and polarity checks in 30; 4 models move the other way, 1 shifts only in the Yes/No format, and 5 show no significant change (Table 12).

The count applies a counting rule declared before the full-roster runs; the outcome of the preregistered aggregate test is reported separately in Section D.2. Models outside the count are discussed in Section C.1. On the held-out half of the full IMPPRES sets (300 items per word), the changes for Qwen3-8B are −37.7 pp for some and −37.7 pp for or, which replicates the core result. Among closed models, high effort changes GPT-5’s reading of some by −20.6 pp and of or by −18.8 pp against minimal effort, and the shift holds in a second answer format (Section B.3). DeepSeek-V4 goes the other way: switching thinking on raises the pragmatic reading of some by 8.9 pp in V4-Pro and 16.9 pp in V4-Flash, and that of lexical scales by 41.7 pp in V4-Pro (Yes/No format).

Modals the third logical-word trigger shift less consistently: on the full held-out sets, the change is −13.6 pp in Qwen3-32B but 1.2 pp in Qwen3-14B, and EXAONE-Deep and OLMo-3-32B-Think read modals more pragmatically with thinking (10.2 and 29.7 pp). The robust effect is on some and or.

On the full held-out sets, run for the models listed in Table 9, 47 of 48 some and or shifts survive Holm correction within their model family. Thinking also separates the two classes: the interaction between the log thinking budget and the

<table><tr><td rowspan="2">Model</td><td colspan="3">LOGICAL WORDS</td><td colspan="3">REST OF THE BATTERY</td><td colspan="2">THREE-WAY CHOICE</td></tr><tr><td>some</td><td>or</td><td></td><td>modals</td><td>lexical scales</td><td>indirect ans.</td><td>some</td><td>or</td></tr><tr><td colspan="9">QWEN3 (THINKING OFF → NATURAL THINKING)</td></tr><tr><td>Qwen3-4B</td><td>-37.2[-55.1, −17.9]</td><td>61.6[–74.7, −47.5]</td><td></td><td>-32.3[–47.9, −15.6]</td><td>-7.7[−11.9, −4.0]</td><td>-0.4[−7.2, 6.8]°</td><td>-9.9</td><td>-11.9</td></tr><tr><td>Qwen3-8B</td><td>-28.2[-41.0, –16.7]</td><td></td><td>-36.4[-49.5, -24.2]</td><td>-12.5[−22.9, −2.1]</td><td> $5 . 5 [ 2 . 4 , 8 . 8 ]$ </td><td> $6 . 5 \left[ - 1 . 1 , 1 4 . 7 \right] ^ { \circ }$ </td><td>-34.0</td><td>-13.8</td></tr><tr><td>Qwen3-14B</td><td>-21.8[-37.2, –9.0]</td><td></td><td>-13.1[−21.2, −6.1]</td><td>4.2[-3.1, 13.5]°</td><td>6.8[3.3, 10.8]</td><td> $1 . 4 [ - 5 . 4 , 8 . 2 ] ^ { \circ }$ </td><td>-11.4</td><td>-2.4</td></tr><tr><td>Qwen3-30B-A3B</td><td>-28.2[–39.7, −17.9]</td><td></td><td>-39.4[–51.5, −26.3]</td><td>-19.8[-36.5, -2.1]</td><td> $0 . 7 \left[ - 2 . 6 , 4 . 0 \right] ^ { \circ }$ </td><td> $9 . 5 [ 2 . 5 , 1 6 . 8 ]$ </td><td>-12.8</td><td>-11.8</td></tr><tr><td>Qwen3-32B</td><td>-51.3[–65.4, −37.2]</td><td></td><td>-46.5[-60.6, –32.3]</td><td>-11.5[−22.9, −0.0]°</td><td> $4 . 0 [ 1 . 1 , 7 . 3 ]$ </td><td>2.5[–3.9, 8.8]°</td><td>-38.0</td><td>-19.7</td></tr><tr><td colspan="9">QWEN3.5 AND QWEN3.8 (THINKING OFF → NATURAL THINKING)</td></tr><tr><td>Qwen3.5-0.8B</td><td>-11.5[–26.9, 2.6]°</td><td></td><td>-14.1[-27.3, −1.0]</td><td>-10.4[-19.8,-3.1]</td><td>-17.2[-22.5, −12.4] -5.6[-13.3, 2.1]°</td><td></td><td>10.7</td><td>-8.3</td></tr><tr><td>Qwen3.5-2B</td><td>-14.1[-30.8, 1.3]°</td><td></td><td>-32.3[-47.5, –16.2]</td><td>-22.9[-37.5,−8.3]</td><td> $1 3 . 5 [ 9 . 9 , 1 7 . 4 ]$ </td><td> $- 1 . 8 \left[ - 8 . 1 , 4 . 2 \right] ^ { \circ }$ </td><td>-8.7</td><td>-8.0</td></tr><tr><td>Qwen3.5-4B</td><td>-39.1[-56.4, –21.8]</td><td></td><td>-34.8[–47.0, –22.7]</td><td> $- 0 . 0 \left[ - 9 . 9 , 9 . 9 \right] ^ { \circ }$ </td><td> $2 . 3 \left[ - 0 . 6 , 5 . 3 \right] ^ { \circ }$ </td><td> $3 . 9 \left[ - 2 . 8 , 1 0 . 4 \right] ^ { \circ }$ </td><td>-13.4</td><td>-14.1</td></tr><tr><td>Qwen3.5-9B</td><td>-39.1[–55.1, –24.4]</td><td></td><td>-23.2[-36.4, −11.1]</td><td>2.6[-4.2, 9.9]°</td><td> $4 . 4 [ 1 . 9 , 7 . 4 ]$ </td><td> $6 . 0 [ 0 . 9 , 1 1 . 8 ]$ </td><td>-10.8</td><td>-6.8</td></tr><tr><td>Qwen3.8-27B</td><td>–16.7[–35.9, 0.0]°</td><td></td><td>-22.2[–35.4, –10.1]</td><td>-8.3[-16.7, -1.0]</td><td> $1 5 . 5 [ 1 1 . 0 , 2 0 . 3 ]$ </td><td> $1 0 . 5 [ 4 . 6 , 1 7 . 2 ]$ </td><td>-1.2</td><td>-3.0</td></tr><tr><td>Qwen3.8-27B (low effort)</td><td>-23.1[-43.6, -3.8]</td><td></td><td>-37.4[-52.5, −22.2]</td><td> $- 2 . 1 \left[ - 1 3 . 5 , 9 . 4 \right] ^ { \circ }$ </td><td> $1 4 . 3 \ : [ 9 . 9 , 1 9 . 2 ]$ </td><td>8.8 [3.9, 14.4]</td><td>-4.0</td><td>-8.6</td></tr><tr><td>Qwen3.8-27B (medium effort)</td><td>-0.0[−17.9, 16.7]°</td><td></td><td>-12.1[-24.2, -1.0]</td><td>15.6[5.2, 28.1]</td><td>10.6[6.6, 15.0]</td><td>8.8 [2.1, 15.8]</td><td>-1.6</td><td>-2.3</td></tr><tr><td colspan="9">OTHER OPEN FAMILIES</td></tr><tr><td>MiMo-7B-SFT</td><td>-60.3[–74.4, −44.9]</td><td></td><td>-40.4[-50.5, -29.3]</td><td></td><td>-25.0[-37.5,-12.5] -23.0[-30.0,-15.9]</td><td>20.4[10.9, 30.2]</td><td>-26.1</td><td>-26.7</td></tr><tr><td>MiMo-7B-RL</td><td>-44.9[–59.0, −32.0]</td><td></td><td>-39.4[-54.5, -22.2]</td><td>-1.0[−15.6, 12.5]°</td><td> $- 7 . 5 [ - 1 4 . 3 , - 0 . 9 ]$ </td><td>26.3[16.8, 35.1]</td><td>-21.2</td><td>-23.6</td></tr><tr><td>openPangu-Embedded-7B-V1.1⁸</td><td> $- 1 2 . 8 \left[ - 2 6 . 9 , \ : - 0 . 0 \right] ^ { \circ }$ </td><td></td><td> $- 3 . 0 [ - 8 . 1 , 0 . 0 ] ^ { \circ }$ </td><td>10.4[1.0, 21.9]</td><td> $- 1 6 . 1 \intercal - 2 3 . 0 , - 9 . 5 \intercal$ </td><td> $2 . 1 \left[ - 5 . 6 , 9 . 8 \right] ^ { \circ }$ </td><td>-8.4</td><td>-3.0</td></tr><tr><td>openPangu-Embedded-7B (v1.0)8</td><td> $1 4 . 1 \ [ 5 . 1 , 2 4 . 4 ]$ </td><td></td><td> $- 4 . 0 \left[ - 1 4 . 1 , 3 . 0 \right] ^ { \circ }$ </td><td> $1 0 . 4 [ 3 . 1 , 1 7 . 7 ]$ </td><td> $7 . 3 \ : [ 3 . 1 , 1 1 . 5 ]$ </td><td> $8 . 9 [ 0 . 7 , 1 7 . 0 ]$ </td><td>-0.7</td><td>-0.1</td></tr><tr><td>gpt-oss-20b (high-low effort)</td><td> $- 3 0 . 8 \left[ - 4 2 . 9 , \right. \left. - 1 9 . 2 \right]$ </td><td></td><td> $- 2 4 . 7 [ - 3 3 . 3 , - 1 6 . 2 ]$ </td><td> $\cdot 2 0 . 3 \ : [ - 2 9 . 2 , - 1 2 . 0 ]$ </td><td> $- 0 . 3 \left[ - 2 . 5 , 1 . 9 \right] ^ { \circ }$ </td><td> $3 . 2 \left[ - 0 . 7 , 7 . 4 \right] ^ { \circ }$ </td><td>-1.1</td><td>-0.9</td></tr><tr><td colspan="9">DEEPSEEK (R1 DISTILLS ALWAYS THINK: FORCED-EMPTY BLOCK → NATURAL; V4: THINKING OFF → ON)</td></tr><tr><td>DeepSeek-R1-0528-Qwen3-8B</td><td>-24.4[-42.3, -9.0]</td><td>-10.1 [−17.2, −4.0]</td><td></td><td>-6.2[−13.5, 0.0]°</td><td>-17.2[–23.2, −11.3]</td><td>6.0[-1.8, 13.3]°</td><td>-17.6</td><td>-6.3</td></tr><tr><td>R1-Distill-Llama-8B</td><td>-50.0[–65.4, –34.6]</td><td></td><td> $- 5 7 . 6 [ - 6 8 . 7 , - 4 6 . 5 ]$ </td><td> $- 3 5 . 4 [ - 4 9 . 0 , - 2 1 . 9 ]$ </td><td> $- 3 2 . 9 [ - 3 9 . 3 , - 2 6 . 3 ]$ </td><td> $2 6 . 0 [ 1 6 . 8 , 3 5 . 4 ]$ </td><td>-31.4</td><td>-33.3</td></tr><tr><td colspan="9">CLOSED MODELS (HIGHEST—LOWEST DOSE)</td></tr><tr><td>GPT-5</td><td>-20.6[-31.2,-11.2] -18.8[-27.5,-10.0]</td><td></td><td></td><td>-12.5[-20.0, −5.0]</td><td>7.5[1.7, 13.3]</td><td> $6 . 2 \left[ - 5 . 0 , 1 7 . 5 \right] ^ { \circ }$ </td><td>-6.2</td><td>-11.2</td></tr><tr><td>Claude Sonnet 4.6†</td><td>21.2[12.5, 30.6]</td><td>10.6[–3.1, 24.4]°</td><td></td><td>26.2[13.8, 38.8]</td><td> $2 0 . 8 \ : [ 1 1 . 7 , 3 0 . 8 ]$ </td><td> $2 . 5 \left[ - 5 . 0 , 1 0 . 0 \right] ^ { \circ }$ </td><td>-5.1</td><td>-15.6</td></tr></table>

Baseline: thinking off, or a forced-empty block where no switch exists; <sup>†</sup>Reverses in the three-way choice. <sup>§</sup>Vendor system prompt; fast mode via ‘ /no\_think’. Three-way: no answer polarity, means only (Section B.3). All models: Tables 8 and 12.

Table 1: Thinking reduces the pragmatic reading of logical words across families. Change with thinking (pp) [95% CI]; bold/underlined: largest/second absolute change per column; <sup>circ</sup>: CI includes 0.

class of phenomenon is negative in Qwen3-4B, 8B and 14B (8B: −0.063, 95% CI −0.087 to −0.039, p <.001), and has the same sign, though not significant, at 32B.

The rest of the battery diverges. From 8B up, Qwen3 interprets lexical scales slightly more pragmatically with thinking (5.5 pp at 8B, 4.0 pp at 32B). Presupposition accuracy rises by 22.2 pp in Qwen3-8B. Context-driven implicatures improve: accuracy on figurative language (Fig-QA) rises by 10.7 pp and on six PUB tasks by 10.1 pp, and both gains are significant at every Qwen3 size from 4B to 32B; on two further sets, accuracy rises on indirect answers in Circa (Louis et al., 2020) by 17.5 pp and on naturally occurring presuppositions in NOPE (Parrish et al., 2021) by 9.9 pp in Qwen3- 8B, and no model among the 14 we tested on these two sets gets significantly worse (Table 10); on LUDWIG indirect answers, accuracy changes by 6.5 pp. Below 8B, Qwen3 loses lexical scales as well (−7.7 pp at 4B), so the lexical effect changes sign between Qwen3-4B and Qwen3-8B. Equivalence within ±3 pp on lexical scales holds in only 2 of 42 models. Thinking reduces enrichment of some, or or both in 30 of 40 open models; in Qwen3 from 8B up it increases lexical enrichment and improves context-driven inference.

## 4.2 Sensitivity Analysis: The Shift Survives Stopping, Baseline, Format and Item Choices

Full-length thinking lowers the pragmatic reading of some and or under every analysis choice (Table 2): with a forced-empty block as the baseline, under either stopping procedure, on the held-out full sets and in a three-way choice without Yes/No labels, in both Qwen3-8B and Qwen3-32B. Along the budget (Figure 5) the reading of some first rises, by 37.2 pp at 128 tokens, and falls below its nothinking level only beyond 512 tokens (−21.8 pp at 1024): mirroring the trace result (Section 4.3).

Four further checks, on how thinking stops, the baseline, answer bias and speaker knowledge, are in Section C.1. The shift persists under both stopping procedures and in a three-way choice without Yes/No labels. The three-way prompt asks whether the implicature follows from the sentence, so it removes answer polarity but also moves the question from speaker belief to inference; we use it as a polarity check rather than a framing-matched replication (prompts in Section B.4).

## 4.3 Early in the Trace the Enriched Reading Is Endorsed; at the End It Is Rejected

Forcing an answer after each prefix of the natural trace gives intervention-conditioned answer trajectories (Figure 4; 300 held-out some and 300 or items per model, three seeds). With an empty trace, the reading matches the no-thinking baseline (35.8% against 38.8% for some in Qwen3-8B). In Qwen3-8B the pragmatic reading of some first rises, from 35.8% with an empty trace to 52.0% after 128 tokens (+16.2 pp, CI 12.8 to 19.8), and then falls to 1.1% by the end of the trace; or follows the same arc (+16.9 pp after 128 tokens). Qwen3-14B shows it on or (+18.8 pp after 128 tokens, 0.3% at the end). Independent generations stopped at a budget of 128 tokens show the same rise (66.7% for some in Qwen3-8B). Qwen3-32B shows no early rise, only the later fall; most of it comes after the first 128 tokens (−17.2 pp at 128 tokens against −56.7 pp at the end). Lexical scales take a different path: their pragmatic reading rises slowly through the trace (6.4 pp in Qwen3-8B).

![](images/966e616fcc7b250c7661acb4ade2644c9b9ec490c40e6576602cffd489010386.jpg)  
trace truncated at k tokens, then the answer is forced (log scale)

Figure 4: The enriched reading is endorsed early in the trace and abandoned later. Pragmatic reading R after the first t tokens of the model’s own trace, with 95% bands; panel (d) and the lexical-scale lines come from an earlier, smaller run of the probe.
<table><tr><td></td><td></td><td></td><td rowspan=1 colspan=1>-28.2</td><td rowspan=1 colspan=1>-36.4</td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1>-39.7</td><td rowspan=1 colspan=1>-33.3</td></tr><tr><td></td><td rowspan=5 colspan=2>Budget 128, stop phraseBudget 512, stop phraseBudget 1024, stop phraseBudget 1024, plain capItems: held-out full setFormat: three-way choice</td><td rowspan=1 colspan=1>37.2</td><td rowspan=1 colspan=1>4.0</td></tr><tr><td></td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>-15.2</td></tr><tr><td></td><td rowspan=4 colspan=3>-31.3-34.3-37.7-13.8</td><td rowspan=1 colspan=1>-21.8</td></tr><tr><td></td><td rowspan=1 colspan=1>-34.4</td></tr><tr><td></td><td rowspan=1 colspan=1>-34.0</td></tr><tr><td rowspan=2 colspan=3>QWEN3-32BReference: core, Yes/No, off → naturalItems: held-out full setFormat: three-way choice</td></tr><tr><td rowspan=1 colspan=2>-51.3 -46.5-55.6–52.7-38.0-19.7</td></tr></table>

Table 2: Full thinking lowers the reading under every analysis choice. Change in the pragmatic reading with thinking (pp); short budgets raise it before it falls.

Within single traces, in 67.4% of Qwen3-8B some traces, the answer forced at an early prefix is pragmatic while the final answer is literal, whereas 31.4% are literal throughout.

![](images/855df26bbbcceded413cf951be1d46888d6b6895b324d643f031752dcc02ea42.jpg)  
Figure 5: Pragmatic reading rate for Qwen3-8B vs. thinking budget. Capped with stop phrase (hollow: plain cap). Lower strip: realized tokens. Bands: 95% cluster-bootstrap CIs.

Across Qwen3 models and words, this pragmatic-to-literal trajectory ranges from 41.4% to 68.1%. These rates count a pragmatic answer at any of up to eight sampled prefixes. Beyond Qwen3, Phi-4-reasoning follows the Qwen3-32B profile: no early rise, a fall from 61.5% to 9.0% on some, and pragmatic-then-literal trajectories in 75.3% of traces (Section C.1). On the newest generation, Qwen3.8-27B cancels or along its trace, from 38.4% with an empty trace to 0.0% at the end (pragmatic-then-literal in 55.0% of traces), while its some reading falls less (28.2% to 16.7%; Section C.1). Qwen3.5 shows the full arc, including the early rise: Qwen3.5-9B reads some pragmatically in 35.9% of items with an empty trace, 57.1% after 32 tokens and 0.0% at the end (Figure 7), although for Qwen3.5 many truncated traces yield no valid answer, so this arc rests on the traces that do (Section C.1). Among further families, EXAONE-Deep and Nemotron-Nano-9B hold the enriched reading over the first hundred tokens and then drop it (some: 61.5% after 128 tokens, 17.9% at the end; 64.1% and 11.5%), Phi-4-mini-reasoning shows a weaker arc, and Skywork-OR1 drops within the first 32 tokens and then stays flat (Section C.1). We call this pattern entertain, then cancel: in almost every model we probe the enriched reading is endorsed early in the trace and lower by its end, in most models far below the no-thinking level.

<table><tr><td></td><td colspan="4">Default reading (%)</td></tr><tr><td>Stage</td><td>some</td><td>or</td><td>lexical</td><td>LUDWIG</td></tr><tr><td colspan="5">OLMo-3-Think (native thinking;  $c o s t = N A T$  — forced-empty)</td></tr><tr><td>base</td><td> $7 5 . 6 \pm 4 . 4$ </td><td> $7 4 . 7 \pm 1 . 8$ </td><td> $3 9 . 5 \pm 0 . 4$ </td><td> $6 7 . 0 \pm 4 . 0$ </td></tr><tr><td>Think-SFT</td><td> $3 9 . 7 \pm 5 . 9$ </td><td> $5 0 . 5 \pm 3 . 5$ </td><td> $5 . 3 \pm 0 . 7$ </td><td> $7 6 . 8 \pm 2 . 8$ </td></tr><tr><td>Think-DPO</td><td> $5 3 . 8 \pm 3 . 8$ </td><td> $4 8 . 5 \pm 5 . 2$ </td><td> $4 . 6 \pm 0 . 7$ </td><td> $7 5 . 1 \pm 2 . 4$ </td></tr><tr><td>RL step 100</td><td> $4 7 . 4 \pm 2 . 2$ </td><td> $5 1 . 5 \pm 0 . 0$ </td><td> $5 . 3 \pm 0 . 7$ </td><td> $7 5 . 1 \pm 3 . 4$ </td></tr><tr><td>RL step 500</td><td> $4 1 . 0 \pm 4 . 4$ </td><td> $5 0 . 5 \pm 3 . 5$ </td><td> $4 . 9 \pm 0 . 4$ </td><td> $7 5 . 8 \pm 3 . 6$ </td></tr><tr><td>RL step 1375 (final)</td><td> $3 2 . 1 \pm 4 . 4$ </td><td> $4 2 . 4 \pm 3 . 0$ </td><td> $4 . 4 \pm 0 . 4$ </td><td> $7 4 . 7 \pm 2 . 8$ </td></tr><tr><td colspan="5">OLMo-3 RL-Zero from the base (no thinking SFT)</td></tr><tr><td>Math, step 100</td><td> $7 5 . 6 \pm 4 . 4$ </td><td> $7 4 . 7 \pm 1 . 8$ </td><td> $4 0 . 8 \pm 1 . 0$ </td><td> $6 7 . 0 \pm 4 . 0$ </td></tr><tr><td>Math, final</td><td> $7 8 . 2 \pm 2 . 2$ </td><td> $7 7 . 8 \pm 1 . 8$ </td><td> $3 8 . 9 \pm 1 . 7$ </td><td> $6 7 . 7 \pm 3 . 4$ </td></tr><tr><td>IF, final</td><td> $8 9 . 7 \pm 2 . 2$ </td><td> $8 3 . 8 \pm { 1 . 8 }$ </td><td> $3 5 . 3 \pm 1 . 4$ </td><td> $7 0 . 5 \pm 1 . 1$ </td></tr><tr><td>General, final</td><td> $7 8 . 2 \pm 2 . 2$ </td><td> $7 3 . 7 \pm 1 . 8$ </td><td> $4 5 . 3 \pm 0 . 8$ </td><td> $6 6 . 0 \pm 3 . 0$ </td></tr><tr><td>Mix, final</td><td> $9 4 . 9 \pm 4 . 4$ </td><td> $8 2 . 8 \pm { 1 . 8 }$ </td><td> $5 1 . 4 \pm 1 . 5$ </td><td> $6 8 . 8 \pm 3 . 4$ </td></tr><tr><td colspan="5">OLMo-2-7B (cost = prompted CoT − none)</td></tr><tr><td>base</td><td> $4 3 . 6 \pm 2 . 2$ </td><td> $3 1 . 3 \pm 3 . 5$ </td><td> $1 9 . 6 \pm 1 . 5$ </td><td> $6 9 . 8 \pm 1 . 6$ </td></tr><tr><td>SFT</td><td> $4 2 . 3 \pm 0 . 0$ </td><td> $3 1 . 3 \pm 4 . 6$ </td><td> $3 2 . 9 \pm 2 . 0$ </td><td> $7 3 . 3 \pm 0 . 6$ </td></tr><tr><td>DPO</td><td> $3 7 . 2 \pm 4 . 4$ </td><td> $1 9 . 2 \pm { 1 . 8 }$ </td><td> $3 2 . 9 \pm 2 . 8$ </td><td> $7 0 . 5 \pm 1 . 1$ </td></tr><tr><td>RLVR (Instruct)</td><td> $2 4 . 4 \pm 2 . 2$ </td><td> $1 6 . 2 \pm 3 . 5$ </td><td> $3 4 . 4 \pm 2 . 6$ </td><td> $7 0 . 5 \pm 2 . 1$ </td></tr><tr><td colspan="5">Tülu-3-8B</td></tr><tr><td>SFT</td><td> $4 6 . 2 \pm 0 . 0$ </td><td> $5 . 1 \pm 1 . 8$ </td><td> $1 0 . 2 \pm 0 . 4$ </td><td> $7 6 . 5 \pm 1 . 6$ </td></tr><tr><td>DPO</td><td> $1 4 . 1 \pm 2 . 2$ </td><td> $2 . 0 \pm 1 . 8$ </td><td> $4 . 0 \pm 0 . 0$ </td><td> $7 3 . 7 \pm 2 . 8$ </td></tr><tr><td>RLVR</td><td> $1 4 . 1 \pm 2 . 2$ </td><td> $2 . 0 \pm 1 . 8$ </td><td> $4 . 4 \pm 0 . 4$ </td><td> $7 5 . 8 \pm 2 . 8$ </td></tr><tr><td>RLVR (3.1)</td><td> $1 4 . 1 \pm 2 . 2$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $1 0 . 4 \pm 0 . 4$ </td><td> $7 6 . 1 \pm 2 . 2$ </td></tr><tr><td colspan="5">Qwen2.5-7B: distillation and zero-RL</td></tr><tr><td>base</td><td> $4 1 . 0 \pm 2 . 2$ </td><td> $7 . 1 \pm 1 . 8$ </td><td> $3 2 . 9 \pm 1 . 7$ </td><td> $6 6 . 7 \pm 3 . 4$ </td></tr><tr><td>R1-Distill (SFT on traces)</td><td> $8 5 . 9 \pm 2 . 2$ </td><td> $6 5 . 7 \pm 7 . 6$ </td><td> $4 6 . 8 \pm 2 . 7$ </td><td> $7 3 . 7 \pm 3 . 8$ </td></tr><tr><td>SimpleRL-Zoo (zero-RL)</td><td> $4 3 . 6 \pm 2 . 2$ </td><td> $7 . 1 \pm 1 . 8$ </td><td> $3 1 . 8 \pm 2 . 4$ </td><td> $6 5 . 6 \pm 3 . 4$ </td></tr><tr><td>Open-Reasoner-Zero</td><td> $4 1 . 0 \pm 2 . 2$ </td><td> $8 . 1 \pm 3 . 5$ </td><td> $7 . 1 \pm 1 . 0$ </td><td> $6 6 . 0 \pm 3 . 0$ </td></tr></table>

Table 3: The literal default appears with thinking SFT in OLMo-3 and DPO in Tülu-3. Default reading (%) under a fixed 4-shot prompt, mean $\pm \ : \mathrm { s . d . }$

For Qwen3.5-9B, 1,163 of its some traces give an invalid answer at some prefix and are dropped (Section C.1), so its trajectory rates describe the valid subset; the trajectory claim rests on the models with near-complete traces. The enriched reading of a logical word is endorsed early in the trace, in smaller models even more than without thinking, and abandoned later.

## 4.4 Post-Training Moves the Default Reading

If thinking cancels a default, training for reasoning might also shift the default itself. Table 3 reports the scores across every stage of three lineages without deliberation (using answer log-probabilities), where we measure the within-stage thinking effect wherever applicable. Post-training for reasoning moves the default reading of some, at a stage and in a direction that depend on the recipe.

In OLMo-3, the pragmatic default falls from 75.6% in the base model to 39.7% after supervised fine-tuning on thinking traces, whereas reinforcement learning on math from the same base (RL-Zero) leaves it at 78.2%. In Tülu-3 the change comes with preference tuning (DPO), from 46.2% to 14.1%, and the following reinforcement learning from verifiable rewards (RLVR) leaves it there. As the default turns literal, the additional thinking effect shrinks: within OLMo-3-Think the cost of thinking on some shrinks from −32.1 pp at the preference-tuning stage to −2.6 pp after the final RL step. Distillation from reasoning traces does not have this effect in every lineage: R1-Distill-Qwen-7B reads some pragmatically in 85.9% of items, against 41.0% for its base model.

The literal default of logical words emerges during post-training, at a recipe-dependent stage (thinking SFT in OLMo-3, preference tuning in Tülu-3), and not in the RL-Zero and distillation comparisons we tested.

## 5 Selective Thinking for Scalar Items (Exploratory)

If thinking first entertains an implicature and then cancels it, the reading to keep is already in the trace. For questions that ask what a speaker of a some or or sentence believes, we take the answer the model gives after the first 128 tokens of its own trace instead of the answer after the completed trace; every other item keeps the natural-thinking answer. Routing reads only the prompt (the question template and the speaker’s sentence), never the dataset or label. On routed items the answer comes entirely from the prefix (weight $w = 1$ on the prefix distribution), and the score is the mean of speakermeaning and literal-gold accuracy, each averaged equally over its sets. The remedy is exploratory, not pre-registered: its weight was selected on development items after two earlier exploratory rounds, frozen, and evaluated once on held-out template families of Qwen3-8B and DeepSeek-R1-0528-

<table><tr><td>Model</td><td>Method</td><td>Felicity Speaker Presup. Literal</td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B</td><td>Natural thinking</td><td>1.2</td><td>58.2</td><td>65.7</td><td>87.9</td></tr><tr><td></td><td>Task-aware instruction</td><td>3.5</td><td>60.2</td><td>66.8</td><td>92.0</td></tr><tr><td></td><td>ours</td><td>20.5</td><td>62.5</td><td>65.7</td><td>87.9</td></tr><tr><td></td><td>DeepSeek-R1 Natural thinking</td><td>2.2</td><td>55.5</td><td>62.3</td><td>86.3</td></tr><tr><td></td><td>Task-aware instruction</td><td>2.6</td><td>57.8</td><td>62.2</td><td>87.0</td></tr><tr><td></td><td>ours</td><td>8.1</td><td>58.8</td><td>62.9</td><td>86.3</td></tr></table>

Table 4: Answering from the entertained reading restores scalar implicatures without losing literal accuracy (held-out test families; accuracy, %). Felicity: IMPPRES some/or felicity items.

Qwen3-8B.

Table 4 reports the held-out results. Pooled over both models, the equal-family score rises by 1.4 pp (95% CI 1.2 to 1.6) over natural thinking, and every literal-gold safety stratum stays within its pre-declared bound (literal accuracy is unchanged because routing leaves the literal-gold strata untouched; smallest one-sided lower bound 0.0 pp; by construction the routing leaves those items untouched). The gain is concentrated where thinking cancels: on the IMPPRES some/or felicity items, accuracy rises from 1.2% to 20.5% in Qwen3-8B and from 2.2% to 8.1% in DeepSeek-R1-0528- Qwen3-8B. A task-aware instruction in the user turn, the strongest prompt on development items, raises the overall score more (2.3 pp) but barely moves these items (3.5% and 2.6%), and the two act on different items. Four pre-registered remedies that intervene in or train the thinking itself did not pass their development gates (Section D.2).

## 6 Discussion

What thinking does to meaning. The results fit the default-and-cancel account better than uniformly rising enrichment. If thinking only supplied resources for implicatures, scalar enrichment would rise; instead it falls on some and or, rises on lexical scales in larger Qwen3 models, is mixed elsewhere, and presuppositions and indirect answers improve. The traces show the predicted order: the enriched reading appears early, in smaller models even strengthened, before the final answer rejects it. Models spend thinking on strict entailment for some and or and override what the speaker meant. This matches earlier observations: the drop in scalar-implicature recognition with more cancellations in ImplicatureX (Spinoso-Di Piano et al., 2026), the literal drift under prompted chain-ofthought (Park et al., 2024), and the gains from Gricean prompts on context-driven implicatures (Kim et al., 2023; Sato et al., 2025).

Why it matters. The implicatures thinking reduces are textbook cases of everyday enrichment: some for ‘not all’, or for ‘not both’. With thinking on, an assistant more often treats “some of the tests failed” as allowing that all failed, yet answers indirect questions more accurately. Because both readings are licensed, single-answer accuracy benchmarks cannot separate a change of interpretation from a loss of competence. Evaluations should report the reading chosen alongside accuracy, and training should keep the reading a speaker intends where the context warrants it.

Who differs. Not every model follows the pattern in every format: Granite-3.3 and Llama-3.1- Nemotron-Nano-8B read some more pragmatically in the Yes/No format, and the smallest Qwen3 models and Phi-4-mini-reasoning lose lexical scales as well. With the stage results, this shows that longer deliberation need not produce the literal shift.

## 7 Conclusion

In this work, we investigated whether giving a reasoning model more time to think makes it more pragmatic, as the effort account of human enrichment posits, or more literal. On fixed weights, thinking reduces the implicatures of some and or in most open models and in GPT-5, while presuppositions and indirect answers improve and larger Qwen3 models read lexical scales more pragmatically. Early trace prefixes give the enriched reading and completed traces the literal one, and checkpoint comparisons locate the drop in specific posttraining stages. The diagnostic also points to a remedy: answering from the reading the model entertains early, rather than the one it settles on, recovers scalar implicatures on held-out items without touching literal accuracy. Reasoning models can now be studied as listeners whose deliberation we control; the dose protocol and the trace diagnostic measure changes in interpretation that accuracy alone cannot distinguish, and locate them inside the reasoning trace.

## Limitations

Scope and causality. The graded dose–response major rests on Qwen3 and gpt-oss, the open models with a thinking switch or effort levels; the other open models are compared against a forced-empty thinking block they were not trained on, and closed models only through their APIs. The training-stage results compare checkpoints and are correlational, and the trace probe intervenes on traces rather than reading an internal state, so neither identifies a mechanism. All headline items are in English; a machine-translated German and Chinese slice (Section C.1) shows the same drop.

Measurement. We measure the reading a model endorses under forced-choice framings, and the format matters: the direction for Claude models reverses between formats. Free-form answers could not be coded reliably by LLM raters (best agreement of an LLM rater with the human consensus labels κ = 0.57), so every headline result uses forced choices. The exploratory remedy of Section 5 is tested on two 8B models that share a base model, so its generality across lineages and scales is untested. The human labels are consensus labels on a sample; there is no independent double annotation, so inter-annotator agreement is not reported.

## Ethical considerations

Artefacts. We use publicly released models and datasets under their licences (Section A.2). IMP-PRES is licensed for non-commercial use, and two stimulus repositories carry no licence file; we use them for research only, cite their authors, and redistribute only item identifiers and our own new items. We release, under open licences at https://github.com/kouzhizhuo/ entertain-then-cancel, the new category sentences and the 50 speaker-knowledge scenarios; the polarity-flipped and three-way variants of the IMPPRES some/or items, as identifiers plus our templates; the code, prompts and all model traces; and the machine-translated German and Chinese slice, flagged as not human-verified. NOPE, which states no licence, is not redistributed by us.

Human participants. No participants were recruited. The only human labels are consensus labels from a panel of five or more annotators with PhD-level training in related fields, agreed through discussion; an author recorded them with a software tool. They cover a blind audit of a subset of model answers, the speaker-knowledge scenarios and 20 Chinese translations. No personal data were collected or used.

Implications. The finding concerns how assistants interpret users. If thinking modes make models read underinformative or indirect utterances literally, users may receive answers that are technically correct but miss what they meant. Deployers who switch thinking on by default for conversational use should evaluate interpretation, not only accuracy. We see no dual-use risk beyond this.

AI assistance. AI assistants were used for code writing and polish but the whole research is design and checked by researchers.

## References

Tara Azin, Yongan Yu, Raj Singh, and Olessia Jouravlev. 2026. Presupposition and reasoning in conditionals: A theory-based study of humans and LLMs. In Proceedings of the 30th Conference on Computational Natural Language Learning, pages 452–470, San Diego, California, USA. Association for Computational Linguistics.

Lewis Bott and Ira A. Noveck. 2004. Some utterances are underinformative: The onset and time course of scalar inferences. Journal of Memory and Language, 51(3):437–457.

Richard Breheny, Napoleon Katsos, and John Williams. 2006. Are generalised scalar implicatures generated by default? an on-line investigation into the role of context in generating pragmatic inferences. Cognition, 100(3):434–463.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, et al. 2024. Do NOT think that much for 2+3=? on the overthinking of o1-like LLMs. Preprint, arXiv:2412.21187.

Ye-eun Cho. 2026. Continuous interpretive steering for scalar diversity. Preprint, arXiv:2604.07006.

Ian B. de Haan, Peter van der Putten, and Max van Duijn. 2026a. Evaluating Theory of Mind in reasoning models: Robustness over reasoning. Preprint, arXiv:2608.04646.

Ian B. de Haan, Peter van der Putten, and Max van Duijn. 2026b. Reasoning promotes robustness in Theory of Mind tasks. Preprint, arXiv:2601.16853.

Wim De Neys and Walter Schaeken. 2007. When people are more logical under cognitive load: Dual task impact on scalar implicature. Experimental Psychology, 54(2):128–133.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Preprint, arXiv:2501.12948.

Judith Degen. 2015. Investigating the distribution of some (but not all) implicatures using corpora and web-based methods. Semantics and Pragmatics, 8.

Judith Degen and Michael K. Tanenhaus. 2015. Processing scalar implicature: A constraint-based approach. Cognitive Science, 39(4):667–710.

Kristien Dieussaert, Suzanne Verkerk, Ellen Gillard, and Walter Schaeken. 2011. Some effort for some: Further evidence that scalar implicatures are effortful. Quarterly Journal of Experimental Psychology, 64(12):2352–2367.

Zhaoxin Feng, Zheng Chen, Jianfei Ma, Yip Tin Po, Emmanuele Chersoni, and Bo Li. 2026. Good arguments against the people pleasers: How reasoning mitigates (yet masks) LLM sycophancy. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 24536–24570, San Diego, California, United States. Association for Computational Linguistics.

Michael C. Frank and Noah D. Goodman. 2012. Predicting pragmatic reasoning in language games. Science, 336(6084):998–998.

Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, Henry Sleight, Linda Petrini, et al. 2025. Inverse scaling in Test-Time compute. Transactions on Machine Learning Research.

Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Yifu Lu, Mengdi Wang, Dinesh Manocha, Furong Huang, Mohammad Ghavamzadeh, et al. 2025. Does thinking more always help? mirage of Test-Time scaling in reasoning models. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025).

Raymond W. Gibbs. 1983. Do people always process the literal meanings of indirect requests? Journal of Experimental Psychology: Learning, Memory, and Cognition, 9(3):524–533.

Rachel Giora. 1997. Understanding figurative and literal language: The graded salience hypothesis. cogl, 8(3):183–206.

Nanxu Gong, Haotian Li, Sixun Dong, Jianxun Lian, Yanjie Fu, and Xing Xie. 2026. To think or not to think, that is the question for large reasoning models in Theory of Mind tasks. Preprint, arXiv:2602.10625.

Noah D. Goodman and Michael C. Frank. 2016. Pragmatic language interpretation as probabilistic inference. Trends in Cognitive Sciences, 20(11):818–829.

Noah D. Goodman and Andreas Stuhlmüller. 2013. Knowledge and implicature: Modeling language understanding as social cognition. Topics in Cognitive Science, 5(1):173–184.

Nicole Gotzner, Stephanie Solt, and Anton Benz. 2018. Scalar diversity, negative strengthening, and adjectival semantics. Frontiers in Psychology, 9.

H. P. Grice. 1975. Logic and conversation. In Peter Cole and Jerry L. Morgan, editors, Syntax and Semantics, Vol. 3: Speech Acts, pages 41–58. Academic Press, New York.

Daniel J. Grodner, Natalie M. Klein, Kathleen M. Carbary, and Michael K. Tanenhaus. 2010. “some,” and possibly all, scalar inferences are not delayed: Evidence for immediate pragmatic enrichment. Cognition, 116(1):42–55.

Jennifer Hu, Sammy Floyd, Olessia Jouravlev, Evelina Fedorenko, and Edward Gibson. 2023a. A finegrained comparison of pragmatic language understanding in humans and language models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4194–4213, Toronto, Canada. Association for Computational Linguistics.

Jennifer Hu, Roger Levy, Judith Degen, and Sebastian Schuster. 2023b. Expectations over unspoken alternatives predict pragmatic inferences. Transactions of the Association for Computational Linguistics, 11:885–901.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. 2025. Open-Reasoner-Zero: An open source approach to scaling up reinforcement learning on the base model. Preprint, arXiv:2503.24290.

Yi Ting Huang and Jesse Snedeker. 2009. Online interpretation of scalar quantifiers: Insight into the semantics–pragmatics interface. Cognitive Psychology, 58(3):376–415.

Paloma Jeretic, Alex Warstadt, Suvrat Bhooshan, and Adina Williams. 2020. Are natural language inference models IMPPRESsive? Learning IMPlicature and PRESupposition. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 8690–8705, Online. Association for Computational Linguistics.

Napoleon Katsos and Dorothy V.M. Bishop. 2011. Pragmatic tolerance: Implications for the acquisition of informativeness and implicature. Cognition, 120(1):67– 81.

Zae Myung Kim, David E. Taylor, and Dongyeop Kang. 2023. “is the pope catholic?” applying Chain-of-Thought reasoning to understanding conversational implicatures. Preprint, arXiv:2305.13826.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are Zero-Shot reasoners. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022).

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023).

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman,

Lester James V. Miranda, Alisa Liu, et al. 2024. Tulu 3: Pushing frontiers in open language model Post-Training. Preprint, arXiv:2411.15124.

Seongyun Lee, Yongrae Jo, Minju Seo, Moontae Lee, and Minjoon Seo. 2026. Lost in the noise: How reasoning models fail with contextual distractors. Preprint, arXiv:2601.07226.

Stephen C. Levinson. 2000. Presumptive Meanings: The Theory ofGeneralized Conversational Implicature. The MIT Press.

Xiaomin Li, Zhou Yu, Zhiwei Zhang, Xupeng Chen, Ziji Zhang, Yingying Zhuang, Narayanan Sadagopan, and Anurag Beniwal. 2025. When thinking fails: The pitfalls of reasoning for Instruction-Following in LLMs. In Advances in Neural Information Processing Systems 38, pages 86588–86625. Neural Information Processing Systems Foundation, Inc. (NeurIPS).

Fangru Lin, Daniel Altshuler, and Janet B. Pierrehumbert. 2024. Probing large language models for scalar adjective lexical semantics and scalar diversity pragmatics. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13033–13049, Torino, Italia. ELRA and ICCL.

Emmy Liu, Chenxuan Cui, Kenneth Zheng, and Graham Neubig. 2022. Testing the ability of language models to interpret figurative language. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 4437–4452, Seattle, United States. Association for Computational Linguistics.

Ryan Liu, Jiayi Geng, Addison J. Wu, Ilia Sucholutsky, Tania Lombrozo, and Thomas L. Griffiths. 2024. Mind your step (by step): Chain-of-Thought can reduce performance on tasks where thinking makes humans worse. Preprint, arXiv:2410.21333.

Annie Louis, Dan Roth, and Filip Radlinski. 2020. “I’d rather just go to bed”: Understanding indirect answers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7411–7425, Online. Association for Computational Linguistics.

Andrei Lupu, Timon Willi, and Jakob Foerster. 2025. The decrypto benchmark for Multi-Agent reasoning and Theory of Mind. Preprint, arXiv:2506.20664.

Bolei Ma, Yuting Li, Wei Zhou, Ziwei Gong, Yang Janet Liu, Katja Jasinskaja, Annemarie Friedrich, Julia Hirschberg, Frauke Kreuter, and Barbara Plank. 2025. Pragmatics in the era of large language models: A survey on datasets, evaluation, opportunities and challenges. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8679–8696, Vienna, Austria. Association for Computational Linguistics.

Paul P. Marty and Emmanuel Chemla. 2013. Scalar implicatures: working memory and a comparison with only. Frontiers in Psychology, 4.

Raquel Montero, Natalia Moskvina, Paolo Morosi, Tamara Serrano, Elena Pagliarini, and Evelina Leivada. 2025. Quantification and object perception in multimodal large language models and human linguistic cognition. Preprint, arXiv:2511.08126.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321, Suzhou, China. Association for Computational Linguistics.

Ira A Noveck. 2001. When children are more logical than adults: experimental investigations of scalar implicature. Cognition, 78(2):165–188.

OpenAI, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, et al. 2025. gpt-oss-120b & gptoss-20b model card. Preprint, arXiv:2508.10925.

Elizabeth Pankratz and Bob van Tiel. 2021. The role of relevance for scalar diversity: a usage-based approach. Language and Cognition, 13(4):562–594.

Anna Papafragou and Julien Musolino. 2003. Scalar implicatures: experiments at the semantics–pragmatics interface. Cognition, 86(3):253–282.

Dojun Park, Jiwoo Lee, Hyeyun Jeong, Seohyun Park, and Sungeun Lee. 2024. Pragmatic competence evaluation of large language models for the Korean language. In Proceedings ofthe 38th Pacific Asia Conference on Language, Information and Computation, pages 256–266, Tokyo, Japan. Tokyo University of Foreign Studies.

Jihyung Park, Minchao Huang, Leqi Liu, and Elias Stengel-Eskin. 2026. PragReST: Self-Reinforcing counterfactual reasoning for pragmatic language understanding. Preprint, arXiv:2606.18624.

Alicia Parrish, Sebastian Schuster, Alex Warstadt, Omar Agha, Soo-Hwan Lee, Zhuoye Zhao, Samuel R. Bowman, and Tal Linzen. 2021. NOPE: A corpus of naturally-occurring presuppositions in English. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 349–366, Online. Association for Computational Linguistics.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, et al. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. 2023. The goldilocks of pragmatic understanding: Fine-Tuning strategy matters for implicature resolution by

LLMs. In Advances in Neural Information Processing Systems 36, pages 20827–20905. Neural Information Processing Systems Foundation, Inc. (NeurIPS).

Rose Sathyanathan, Kinshuk Vasisht, and Danish Pruthi. 2026. Evaluating reasoning models for queries with presuppositions. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 23991– 24006, San Diego, California, United States. Association for Computational Linguistics.

Takuma Sato, Seiya Kawano, and Koichiro Yoshino. 2025. Pragmatic theories enhance understanding of implied meanings in LLMs. Preprint, arXiv:2510.26253.

Sebastian Schuster, Yuxing Chen, and Judith Degen. 2020. Harnessing the linguistic signal to predict scalar inferences. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5387–5403, Online. Association for Computational Linguistics.

Linxin Song, Taiwei Shi, and Jieyu Zhao. 2025. The hallucination tax of reinforcement finetuning. Preprint, arXiv:2505.13988.

Dan Sperber and Deirdre Wilson. 1986. Relevance: Communication and Cognition. Blackwell, Oxford.

Cesare Spinoso-Di Piano, Verna Dankers, Marius Mosbach, and Jackie Chi Kit Cheung. 2026. Evaluating communicative belief updates in large language models via implicature recognition and cancellation. Preprint, arXiv:2607.25094.

Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, et al. 2025. To CoT or not to CoT? chain-of-thought helps mainly on math and symbolic reasoning. In The Thirteenth International Conference on Learning Representations (ICLR 2025).

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. PUB: A pragmatics understanding benchmark for assessing LLMs’ pragmatics capabilities. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 12075–12097, Bangkok, Thailand. Association for Computational Linguistics.

Settaluri Lakshmi Sravanthi, Kishan Maharaj, Sravani Gunnu, Abhijit Mishra, and Pushpak Bhattacharyya. 2025. Understand the implication: Learning to think for pragmatic understanding. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 23778–23790, Vienna, Austria. Association for Computational Linguistics.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, et al. 2025. Stop overthinking: A survey on efficient reasoning for large language models. Transactions on Machine Learning Research.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, et al. 2025. Olmo 3. Preprint, arXiv:2512.13961.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, et al. 2024. 2 OLMo 2 furious. Preprint, arXiv:2501.00656.

John M. Tomlinson, Todd M. Bailey, and Lewis Bott. 2013. Possibly all of that and then some: Scalar implicatures are understood in two steps. Journal of Memory and Language, 69(1):18–35.

Polina Tsvilodub, Kanishk Gandhi, Haoran Zhao, Jan-Philipp Fränken, Michael Franke, and Noah D. Goodman. 2025. Non-literal understanding of number words by language models. In Proceedings of the 47th Annual Meeting ofthe Cognitive Science Society.

Bob van Tiel, Emiel van Miltenburg, Natalia Zevakhina, and Bart Geurts. 2016. Scalar diversity. Journal of Semantics, 33(1):137–175.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations (ICLR 2023).

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Zijun Yao, Yantao Liu, Yanxu Chen, Jianhui Chen, Junfeng Fang, Lei Hou, Juanzi Li, and Tat-Seng Chua. 2025. Are reasoning models more prone to hallucination? Preprint, arXiv:2505.23646.

Kefan Yu, Qingcheng Zeng, Weihao Xuan, Wanxin Li, Jingyi Wu, and Rob Voigt. 2026. The pragmatic mind of machines: Tracing the emergence of pragmatic competence in large language models. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 192–213, Rabat, Morocco. Association for Computational Linguistics.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems 38 (NeurIPS 2025).

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. 2025. SimpleRL-Zoo: Investigating and taming zero reinforcement learning for open base models in the wild. In Second Conference on Language Modeling (COLM 2025).

## Appendix Contents

A Data and Resources 13   
A.1 Models and Checkpoints 13   
A.2 Artefacts and Licences 13   
B Experimental Setup 13   
B.1 Method Details 13   
B.2 Experimental Details 15   
B.3 Prior-Work Baselines 16   
B.4 Verbatim Prompts 17   
C Additional Results 20   
C.1 Additional Results . 20   
D Pre-registration and Analysis Details 24   
D.1 Accounts, Hypotheses and Opera  
tionalisation 25   
D.2 Pre-registered Predictions and Out  
comes 30   
D.3 Statistical Procedures 30   
E Related Work 31   
E.1 Related Work 31

## A Data and Resources

The models, checkpoints and licensed artefacts the paper uses.

## A.1 Models and Checkpoints

This section lists every model and checkpoint with its identifier.

All models are open weights, ungated, and served with vLLM (Kwon et al., 2023). Revisions (commit hashes) are recorded for every run in exp/results/\*/config and listed in Table 5.

## A.2 Artefacts and Licences

This section lists the artefacts we use and their licences.

openPangu models are used under the Open-Pangu Model License v1.0 (Huawei), which permits research use; its §3 excludes use within the European Union, and all runs took place outside it. We distribute no model or derivative built on them.

IMPPRES: CC BY-NC 4.0 (research use). PUB, Fig-QA and ImplicatureX: MIT. LUDWIG and Circa: CC BY 4.0. PragMega materials and the scalar-diversity data: no licence file in their repositories (research use with citation). NOPE: no licence stated; used for research evaluation only and not redistributed. Open models are used under

Group Hugging Face identifiers   
Hybrid thinking Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,30B-A3B,32B}, Qwen/Qwen3-8B-Base (Yang et al., 2025)   
Effort levels openai/gpt-oss-{20b,120b} (OpenAI et al., 2025)   
OLMo-2 stages allenai/OLMo-2-1124-7B, -SFT, -DPO, -Instruct; 13B analogues (Team OLMo et al., 2024)   
OLMo-3 stages allenai/Olmo-3-1025-7B, Olmo-3-7B-Think-SFT, -Think-DPO, -Think; RL-Zero {Math, Code, IF, General, Mix} step branches (Team Olmo et al., 2025)   
Tülu-3 stages allenai/Llama-3.1-Tulu-3-8B-SFT, -DPO, Llama-3.1-Tulu-3-8B, Llama-3.1-Tulu-3.1-8B (Lambert et al., 2024)   
Distilled / zero-RL deepseek-ai/DeepSeek-R1- Distill-Qwen-{1.5B,7B,14B} (DeepSeek-AI, 2025); Qwen2.5 bases (Qwen et al., 2024); SimpleRL-Zoo (Zeng et al., 2025); Open-Reasoner-Zero (Hu et al., 2025)   
Training-time fix PragReST-8B/14B (Park et al., 2026)   
Further open rea- microsoft/Phi-4-reasoning, -plus; deepseek  
soning models ai/DeepSeek-R1-Distill-Llama-8B; mistralai/ Magistral-Small-2507 (thinking switched on by its system prompt, so the contrast also changes the system prompt); HuggingFaceTB/SmolLM3- 3B; ibm-granite/granite-3.3-8b-instruct; nvidia/NVIDIA-Nemotron-Nano-9B-v2; nvidia/ AceReason-Nemotron-7B; Skywork/Skywork-OR1- 7B; XiaomiMiMo/MiMo-7B-SFT, -RL; Qwen/QwQ-32B; nvidia/Llama-3\_3-Nemotron-Super-49B-v1; deepseek-ai/DeepSeek-R1-Distill-Llama-70B; nvidia/Llama-3.1-Nemotron-Nano-8B-v1 (thinking switched by its system prompt); microsoft/Phi-4-mini-reasoning (11% of its forced-empty-block answers are invalid); LGAI-EXAONE/EXAONE-Deep-7.8B; openpangu/openPangu-Embedded-7B-V1.1 and openpangu/openPangu-Embedded-7B-model (the earlier v1.0 chat release; openPangu-Embedded-1B-V1.1 is fast-thinking only and has no thinking contrast; vendor Chinese system prompt; ‘ /no\_think’ for fast thinking; openPangu-R-7B-2512 and openPangu-R-72B-2512 are not runnable on our CUDA stack); Qwen3.5 (0.8B, 2B, 4B, 9B) and Qwen3.8-27B (official Qwen releases; Qwen3.8 at its default, low and medium effort). Baseline: the model’s own switch where it has one, otherwise a forced-empty thinking block

Table 5: Models by group. Commit hashes of every checkpoint are recorded with each run.

their published licences (Apache-2.0, MIT or the developer licences listed on their model cards); closed models were queried through their commercial APIs as test subjects only. New items, code and traces are released under an open licence with the paper.

## B Experimental Setup

How items are built and read out, how every run is decoded and audited, the prior-work baselines and the verbatim prompts.

## B.1 Method Details

This section gives the item sets, readouts, data splits and statistical tests in full.

Models and checkpoints. The primary manipulation needs a model whose thinking can be switched and dosed without changing its weights. Qwen3 has a hybrid thinking mode in which the no-thinking template differs from the thinking template only by an empty thinking block (Yang et al., 2025). We therefore use Qwen3-4B, 8B and 14B as the primary models, and 1.7B and 32B for size. The Qwen3 size ladder is also a training-recipe ladder: the 0.6B–14B models are built by strongto-weak distillation from larger models, whereas the 32B and 235B models go through the full four-stage pipeline with reasoning and general RL (Yang et al., 2025, §4.5). Size and recipe are therefore confounded within Qwen3, and we separate them with two same-recipe size pairs, OLMo-3- 7B-Think vs. OLMo-3-32B-Think and gpt-oss-20b vs. 120b. As a capability covariate we use each model’s accuracy on a fixed subset of MATH-500 without thinking. gpt-oss-20b and 120b add a second family with three reasoning-effort levels (OpenAI et al., 2025). We use them only if the realised reasoning length grows monotonically with the level on held-out items. To locate the training origin, we score checkpoints along base → SFT → DPO → RLVR lineages: OLMo-2 (Team OLMo et al., 2024), OLMo-3-Think (Team Olmo et al., 2025) and Tülu-3 from SFT onward (Lambert et al., 2024). We also score the OLMo-3 RL-Zero series, which applies RLVR to one base model in five reward domains (math, code, instruction following, general, mixed), at six log-spaced steps each. Distilled thinkers (DeepSeek-AI, 2025) and zero-RL pairs (Zeng et al., 2025; Hu et al., 2025) are a secondary contrast, because their weights differ from their bases in more than training stage (Yu et al., 2026). All models are open weights and ungated. Section A.1 lists identifiers and revisions.

Data. Items fall into three classes. Generalized implicatures (GCIs) comprise Bott–Noveckstyle category sentences written for this study (six sentence types per category triple, every item screened by two Claude-family LLM raters and an open-weight co-rater (Qwen3- 32B) on the same items, with disagreements dropped the Qwen3-32B co-rater pass is in exp/gold/item\_screen\_qwen32b.jsonl; Section B.2); scalar-diversity sentences with human inference rates per scale (van Tiel et al., 2016; Gotzner et al., 2018; Pankratz and van Tiel, 2021; Hu et al., 2023b); and IMPPRES implicature pairs whose logical and pragmatic labels differ, plus fresh paraphrased minimal pairs (Jeretic et al., 2020). Particularized implicatures (PCIs) come from PragMega (Hu et al., 2023a), LUDWIG (Ruis et al., 2023), Fig-QA (Liu et al., 2022) and six

PUB tasks (Sravanthi et al., 2024). Presupposition items come from IMPPRES. We also run all of ImplicatureX as a replication set (Spinoso-Di Piano et al., 2026). Released sets that predate the thinking models may be contaminated. We therefore report headline GCI numbers on the fresh items and treat released sets as replications; the within-item contrast between thinking conditions does not depend on whether an item was seen in training. Items are split 50/50 into development and test halves, stratified by set. Every finding is reported on the test half. Sources, licences and item counts are listed in Section B.2.

Framings. Scalar items are asked two ways. Felicity framings make the literal answer a pragmatic failure and are primary. For IMPPRES this is an inference question (“Would you conclude that the speaker believes . . . ?”). For the category sentences it is a ternary judgement (completely right / a bit wrong / completely wrong) and a 1–7 rating, with binary felicity (“Is this a good way to describe the facts?”) as a comparison. The truth framing (“Is the sentence true or false?”) is the logic-mode comparison (Section D.1). PCI items use their native meaning format (“What did the speaker mean?”). For H2c, a subset of PCI items is also asked in ImplicatureX’s form, “Is this interpretation true or false?”. Multiple-choice options are shuffled on every trial. A subset is also answered without options, to rule out option matching (Gong et al., 2026). Because LLM coding of these open answers did not reach the pre-declared agreement with a human audit, we report them descriptively (Section B.2). Reference labels for these answers are reference labels by two LLM ratersfrom differ ent modelfamilies, adjudicated: a Claude-family model and Qwen3-32B, with disagreements resolved by a second Claude-family model. We report their inter-rater κ, and the judge’s agreement with them as consistency. Family overlaps are disclosed: Qwen3-32B belongs to the family of the Qwen3 models whose answers it labels; the judge for gpt-oss answers is Qwen3-32B itself, so that agreement is partly self-agreement; the judge for Qwen3 answers, gpt-oss-120b, overlaps with neither rater. The only human labels are consensus labels from a blind expert audit of 50 of these answers and 24 speaker-knowledge scenarios, which carries the validity claim (Section B.2).

Thinking manipulations. No thinking (NT) uses the model’s own switch, which leaves the chat template byte-identical apart from the empty thinking block. Two other ways of disabling thinking serve as checks. Capped budgets stop the thinking block at $B \in \{ 6 4 , \dots , 2 0 4 8 \}$ tokens and append Qwen3’s own stop phrase (Yang et al., 2025). Because that phrase mentions limited time, a plain cap that appends only the closing tag controls for the time-pressure cue. Natural thinking (NAT) is uncapped. Capping truncates thoughts, so we also stratify eight natural samples per item by their realised length, and we plot every effect against realised rather than nominal length. Budget forcing beyond the natural stopping point (Muennighoff et al., 2025) tests whether more thinking than the model chooses continues the trend. Capped and length-stratified budgets change only the number of thinking tokens, on fixed weights and a fixed template; the thinking switch and gpt-oss effort levels also change the template or system prompt, so we treat them as mode contrasts and rest dose– response claims on the budget contrasts. Sampling settings are identical across conditions $( T { = } 0 . 6 $ top-p 0.95, top-k 20), so thinking is not confounded with temperature. Generation uses vLLM (Kwon et al., 2023).

Interventions and controls. Intent-Seeded Thinking (IST) places one fixed sentence about speaker intent at the start of the thinking block, or just before it closes. Its controls are a neutral sentence about careful, step-by-step reasoning and a logic sentence about checking what the words strictly entail, both token-matched to the intent seed (26 Qwen3 tokens); a first run used a shorter neutral seed (16 tokens), reported separately and not treated as length-matched. We compare IST with prompted deliberation on the same weights with thinking off: zero-shot chain-of-thought (Kojima et al., 2022), Gricean instructions and Grice-maxim chain-of-thought (Sato et al., 2025; Kim et al., 2023), few-shot prompting (Ruis et al., 2023), an adapted RSA-style chain-of-thought (Tsvilodub et al., 2025), and self-consistency at matched token cost (Wang et al., 2023). For H2b, the critical sentences are preceded by contexts in which the speaker has full or partial knowledge (Goodman and Stuhlmüller, 2013). For H4, the load is unrelated background text (512 or 2,048 words from one of two genres) placed before the item, followed by an instruction to ignore it. Each critical item has a comprehension check (an entailment paraphrase) and a meta-pragmatic probe (“Would a speaker who says this normally mean . . . ?”), so a literal answer can be separated from a failure to understand. Section B.2 gives every prompt verbatim. Closed-model API calls cost about \$105 in total.

Statistics. Each condition is run with three decoding seeds. Tables report the mean ± standard deviation over seeds of the item-averaged rate. Intervals in the text and figures are 95% cluster-bootstrap confidence intervals over items (10,000 resamples, paired across conditions). The pre-registered primary tests were mixed-effects logistic regressions with by-item random effects; as implemented (Section D.3), they are logistic regressions with itemclustered standard errors, one per model, with the log thinking budget, the phenomenon class and their interaction as predictors. Equivalence is tested with two one-sided tests at ±3 pp, trends over training steps with Spearman’s $\rho$ and permutation $p \textmd { - }$ values, and alignment with human rates with Spearman’s $\rho .$ p-values are Holm-corrected within each pre-registered family of models. Secondary analyses are uncorrected and labelled as such. Every finding is computed on a held-out half of the items; the other half was used only to fix prompts and parsing. Hypotheses, thresholds and exclusion rules were fixed before the main runs (Section D.3).

## B.2 Experimental Details

This section documents decoding, answer parsing, judging and the human audit.

Items, sources and licences. The Bott–Noveckstyle category sentences (category triples, each in six sentence types) and the 50 speaker-knowledge scenarios were written for this study. Every new item was screened by two Claude-family LLM raters against written criteria (the universal statement is true without common exceptions; the partial control is true and clearly partial; the false control is false under every sense, including cladistic taxonomy; no lexical ambiguity). An item was kept only if both raters accepted it or accepted the same fix. Twenty speaker-knowledge scenarios are also part of the human audit below.

Compute. Generation and judging ran on NVIDIA H800 GPUs ( 200 GPU-hours in total); the GRPO runs used 4-GPU jobs with verl 0.9.1.

Prompts. Every prompt template is released with the code.

Decoding and compute. T=0.6, top-p 0.95, topk 20 in every condition; three seeds; a maximum of 8,192 thinking tokens for natural thinking.

Reference labels and the human audit. Open answers are coded as pragmatic, literal, or hedged/other.

• Reference labels: two LLM raters from different model families (a Claude-family model and Qwen3-32B) labelled a pool of 294 answers blind to condition. Disagreements were adjudicated by a second Claude-family model with a logged reason. Inter-rater agreement: κ = 0.18.

• Judges: The judge we ran, gpt-oss-20b, agrees with the human consensus labels at κ = 0.33 [0.12, 0.54], below the pre-declared acceptance rule (≥ 0.70), so open-answer results are reported descriptively and every headline result uses forced-choice formats.

• Human audit: a panel of five or more annotators with PhD-level training in related fields agreed, through discussion, one consensus label for each of 50 answers from the same pool and 24 speaker-knowledge scenarios (12 with full and 12 with partial access). The sample was stratified and random with a fixed seed (answers stratified by reference class, oversampling the rarer classes), and the annotators were blind to model, condition and all model labels. An author recorded the consensus labels with a software tool. The labels are not independent annotations, so no inter-annotator agreement is reported. These are the only human labels in the paper.

• Scenarios: the consensus judgements matched the intended reading in every scenario (100.0%, Wilson 95% lower bound 86.2%): some was judged misleading whenever the speaker had checked every object and fine whenever the speaker had not. This meets the pre-declared validity bar (≥80%, lower bound ≥70%).

• Open-answer coding: agreement of each LLM rater with the human consensus labels, post-stratified to the pool, is κ = 0.57 [0.34, 0.79] for the Claude-family rater (accuracy 77.4%) and κ = 0.14 [0.02, 0.31] for Qwen3- 32B (0.03 with option orders averaged). Neither meets the pre-declared bar (lower bound ≥0.40). Per class, the consensus labels agreed with the Claude-family rater on 73.1% of its pragmatic and 81.2% of its literal labels. The binary sensitivity check (hedged excluded) gives the same picture (κ = 0.59 and 0.18); no scenario was marked unclear.

<table><tr><td>Measure</td><td>NT</td><td>512 tokens</td><td>NAT</td></tr><tr><td>LUDWIG, prompts of Ruis et al. (2023): accuracy</td><td></td><td></td><td></td></tr><tr><td>0-shot</td><td>79.7</td><td>84.0</td><td>82.0</td></tr><tr><td>5-shot</td><td>71.9</td><td>79.6</td><td>78.6</td></tr><tr><td>30-shot</td><td>67.6</td><td>78.1</td><td>78.7</td></tr></table>

<table><tr><td colspan="4">ImplicatureX (Spinoso-Di Piano et al., 2026): accuracy</td></tr><tr><td>Scalar, recognition</td><td>100.0</td><td>89.9</td><td>65.2</td></tr><tr><td>Scalar, cancellation</td><td>46.4</td><td>85.5</td><td>94.2</td></tr><tr><td>Naturally occurring, recognition</td><td>36.0</td><td>40.0</td><td>38.7</td></tr><tr><td>Naturally occurring, cancellation</td><td>76.0</td><td>81.3</td><td>85.3</td></tr></table>

Table 6: On ImplicatureX, thinking lowers scalarimplicature recognition and raises cancellation, as Spinoso-Di Piano et al. (2026) report. Qwen3-8B, accuracy (%) without thinking, with a 512-token cap and with natural thinking.

• Additional LLM rater (not human labels): a separate audit sheet (40 answers and 20 scenarios, disjoint from the human audit) was labelled by GPT-6 (Codex) and the labels were reviewed by the authors. We report them as an extra LLM rater only; they do not count toward any human-validation criterion. Their agreement with the Claude-family rater is κ = 0.58 [0.34, 0.80] and with Qwen3-32B κ = 0.10 [−0.09, 0.29]; they match the intended scenario reading in 100.0% of scenarios.

## B.3 Prior-Work Baselines

This section reproduces prior-work baselines on our items.

Table 6 re-runs two prior measures on Qwen3- 8B. On ImplicatureX, thinking lowers scalarimplicature recognition from 100.0% to 65.2% and raises cancellation from 46.4% to 94.2%, reproducing the original report; LUDWIG accuracy rises with thinking at every number of shots.

Closed models. Table 8 gives the per-model results for the seven closed models queried through their APIs, and Table 7 the robustness checks for GPT-5 and the two Claude models. All doses of a model use the same answer extraction. GPT-5 becomes more literal with effort in both answer formats, and the shift cannot come from a bias towards Yes: without the utterance its Yes rate rises with effort, which would raise, not lower, the pragmatic rate in the Yes/No format. For the Claude models the picture depends on the format. Under the Yes/No inference framing their pragmatic rate rises with thinking, but in a three-way forced choice it falls (Sonnet) or stays flat (Opus). For Claude Sonnet 4.6, much of the Yes/No rise is a shift towards answering Yes: the Yes rate also rises without the utterance, and the pragmatic rate falls when the question is flipped. Claude Opus 4.8 shows a small shift towards the pragmatic answer under both polarities. The same two checks explain the two open models whose pragmatic rate rises with thinking. For Magistral-Small, the Yes rate with out the utterance rises by 14.1 pp [10.1, 18.3], and with the flipped question its pragmatic rate falls by −52.9 pp [−59.0, −46.7], so its rise is a shift towards Yes. For Granite-3.3 the Yes rate rises less (6.5 pp) and the flipped pragmatic rate rises too (9.2 pp [3.1, 15.2]), a small genuine shift towards the pragmatic reading. For Llama-3.1-Nemotron Nano-8B the pragmatic answer rises in both polarities (Yes without the utterance: +25.4 pp; pragmatic rate with the flipped question: +17.5 pp), so its rise is not a pure Yes shift; without thinking, however, it gives the same literal answer on every item in all three formats, so the rise starts from rote answering. Polarity probes also exist for Qwen3-8B and Qwen3.5-4B/9B. With the flipped question, Qwen3.5-9B’s pragmatic rate falls by −31.7 pp, in line with the original question; for Qwen3-8B the flipped question moves the other way from a floor near zero, so its Yes/No estimate is partly polarity sensitive (gpt-oss-20b, by contrast, also turns literal on some with the flipped question: −11.7 pp), while the three-way forced choice, which has no answer polarity, shows a clear drop at every Qwen3 size (−9.9 to −38.0 pp for some). Answer failures differ by model. Claude Opus 4.8 answers in prose without an option letter on 8.1% of no-thinking calls even with a 512-token answer cap, and Gem ini 2.5 Flash on 26.2% of calls at its largest budget, so its results are not interpretable. GPT-5 and GPT-5.4 have no failures at any dose.

Numbers reported in the original papers (†; other models and settings, for orientation only):

• LUDWIG (Ruis et al., 2023, Tables 1 and 3): human average 86.2%; GPT-4 82.3% (1- shot) and 82.0% (5-shot); 86.5% with 5-shot chain-of-thought.

• Grice-maxim chain-of-thought (Kim et al.,

2023, Table 2): text-davinci-003 on LUDWIG, 0.63 zero-shot, 0.71 few-shot, 0.87 with chainof-thought.

• ImplicatureX (Spinoso-Di Piano et al., 2026, Table 6): Qwen3-8B scalar-implicature recognition 1.000 without thinking and 0.804 with a 512-token budget; naturally occurring conversational implicatures 0.420 and 0.380.

• Option removal (Gong et al., 2026, §4, Fig. 6a): on HiToM, Qwen3-8B in reasoning mode goes from 0.481 to 0.629 without options; the non-reasoning model goes from 0.557 to 0.510.

Prompts, shots and decoding for each re-run follow the original papers except for the model; deviations are listed with the prompts in Section B.2.

## B.4 Verbatim Prompts

Each block shows one fully rendered prompt per format or arm (user turn and, where the arm adds one, the assistant prefix), for one example item.

Yes no inference. Yes/No inference format (felicity framing), forced-choice letters; the arm tail is appended after the user turn.

A speaker says: "Some teenagers criticize Guy." Second statement: "Not all teenagers criticize Guy."

Would you conclude that the speaker believes the second statement is true? Answer Yes or No.

Options:   
A. Yes   
B. No

Respond with only the letter of the best answer.

Three way choice. polarity-free three-way format (NLI): the second statement follows / may be true or false / must be false

First sentence: Some teenagers criticize Guy.   
Second sentence: Not all teenagers criticize Guy.

Assume the first sentence is true. Does the second sentence follow?

Options:

A. Yes, the second sentence must be true if the first is true.

B. Not necessarily: the second sentence may be true or false.

C. No, the second sentence must be false if the first is true.

Respond with only the letter of the best answer.

Flipped question. polarity-flipped felicity question (yes-bias probe): literal answer Yes, pragmatic answer No

X: Do you want to catch a Christmas movie later?   
Y: I hope it is funny.

<table><tr><td></td><td></td><td colspan="2">Pragmatic reading</td><td colspan="2">Yes-bias checks</td></tr><tr><td>Model</td><td>Dose contrast</td><td></td><td></td><td></td><td>Yes/No inference Three-way choice “Yes&quot; without utterance Pragmatic, flipped question</td></tr><tr><td>GPT-5</td><td>minimal → high</td><td> $\downarrow 1 5 . 2 ^ { * }$ </td><td> $\downarrow 8 . 8 ^ { * }$ </td><td> $\uparrow 2 6 . 9 ^ { * }$ </td><td>↓3.4</td></tr><tr><td>Claude Sonnet 4.6</td><td> $\mathrm { o f f }  1 6 \mathrm { k }$ </td><td> $\uparrow 1 9 . 2 ^ { * }$ </td><td> ${ \downarrow 9 . 7 } ^ { * }$ </td><td>↑7.9*</td><td>↓13.0*</td></tr><tr><td>Claude Opus 4.8</td><td> $\mathrm { o f f }  \mathrm { h i g h }$ </td><td> $\uparrow 1 0 . 8 ^ { * }$ </td><td>↓1.3</td><td>↑3.9</td><td> $\uparrow 5 . 0 ^ { * }$ </td></tr></table>

Table 7: Closed models: robustness of the thinking effect on logical-word items (pp, highest minus lowest dose), on items answered validly at both doses; two samples per cell; <sup>∗</sup>: paired 95% cluster-bootstrap CI excludes 0. “Yes” without utterance: change in the rate of answering Yes when the speaker’s sentence is removed. Flipped question: change in the pragmatic rate when the question is reworded so that No is the pragmatic answer. A pure shift towards Yes raises the first and lowers the second. In the three-way format pragmatic answers are rare for every model (0–9%), which limits how far the rate can rise. Claude baselines use a 512-token answer cap.
<table><tr><td></td><td></td><td colspan="3">Logical words</td><td colspan="2">Lexical scales</td><td>Context</td></tr><tr><td>Model</td><td>Dose contrast</td><td>some</td><td>or</td><td>modals</td><td>gradable adj.</td><td>scales</td><td>LUDWIG</td></tr><tr><td>GPT-5</td><td>minimal → high effort</td><td> $\downarrow 2 0 . 6 \pm 4 . 4 ^ { * }$ </td><td> $\downarrow 1 8 . 8 \pm { 1 0 . 6 } ^ { * }$ </td><td> $\downarrow 1 2 . 5 \pm 3 . 5 ^ { * }$ </td><td>↑0.0±3.5</td><td> $\uparrow 7 . 5 \pm 1 . 2 ^ { * }$ </td><td>↑6.2±1.8</td></tr><tr><td>GPT-5.4</td><td>minimal → high effort</td><td>↑2.5</td><td> $\downarrow 2 0 . 0 ^ { * }$ </td><td>↓15.0*</td><td>↓7.5</td><td>↑15.0*</td><td>↑2.5</td></tr><tr><td>04-mini</td><td>low → high effort</td><td>↓1.2</td><td>↓6.2</td><td>↓12.5</td><td>↑5.0</td><td>↑5.0</td><td>↓2.5</td></tr><tr><td> $\mathrm { D e e p S e e k - V 4 - P r o } ^ { \ P }$ </td><td>thinking off → on</td><td> $\uparrow 8 . 9 \pm 3 . 6 ^ { * }$ </td><td> $\uparrow 1 . 9 \pm 6 . 3$ </td><td> $\uparrow 1 . 2 \pm 1 . 8$ </td><td> $\uparrow 8 . 8 \pm 4 . 8 ^ { * }$ </td><td> $\uparrow 4 1 . 7 \pm 2 . 2 ^ { * }$ </td><td> $\uparrow 1 0 . 0 \pm 3 . 5 ^ { * }$ </td></tr><tr><td> $\mathrm { D e e p S e e k - V 4 - F l a s h } ^ { \bullet }$ </td><td>thinking off → on</td><td> $\uparrow 1 6 . 9 \pm 2 . 7 ^ { * }$ </td><td> $\uparrow 4 . 4 \pm 2 . 7$ </td><td> $\downarrow 1 . 2 \pm { 1 . 8 }$ </td><td> $\uparrow 5 . 0 \pm 3 . 5$ </td><td> $\uparrow 1 0 . 8 \pm { 1 . 2 } ^ { * }$ </td><td> $\uparrow 5 . 0 \pm 0 . 0$ </td></tr><tr><td>Gemini 2.5 Flash</td><td>budget 0 → 8k</td><td>↓3.0</td><td>↓23.8*</td><td>↑8.8</td><td>↑7.5</td><td>↑45.0</td><td>↑10.3</td></tr><tr><td>Gemini 2.5 Pro</td><td>budget  $1 2 8 ^ { \ddagger }  8 \mathbf { k }$ </td><td>↓1.3</td><td>↓10.5</td><td> $\uparrow 3 0 . 8 ^ { * }$ </td><td> $\uparrow 2 5 . 0 ^ { * }$ </td><td> ${ \uparrow } 5 8 . 3 ^ { * }$ </td><td>↑0.0</td></tr><tr><td>Claude Sonnet 4.6</td><td>off → budget 16k</td><td> $\uparrow 2 1 . 2 \pm { 1 . 6 } ^ { * }$ </td><td> $\uparrow 1 0 . 6 \pm 4 . 6$ </td><td> $\uparrow 2 6 . 2 \pm { 1 . 8 } ^ { * }$ </td><td> $\uparrow 2 5 . 0 \pm 3 . 5 ^ { * }$ </td><td> $\uparrow 2 0 . 8 \pm 5 . 9 ^ { * }$ </td><td> $\uparrow 2 . 5 \pm 0 . 0$ </td></tr><tr><td>Claude Opus 4.8</td><td> $\mathrm { o f f }  \mathrm { h i g h e f f o r t }$ </td><td> $\uparrow 1 6 . 7 \pm 2 . 2 ^ { * }$ </td><td> $\uparrow 7 . 2 \pm 1 . 1$ </td><td> $\uparrow 1 0 . 0 \pm 3 . 4$ </td><td> $\uparrow 6 . 2 \pm { 1 . 8 }$ </td><td> $\downarrow 7 . 5 \pm 3 . 5$ </td><td> $\uparrow 2 . 5 \pm 0 . 0$ </td></tr></table>

Table 8: Closed and API-served models: change in the pragmatic reading (pp) from the lowest to the highest available thinking dose; LUDWIG is accuracy. <sup>∗</sup>: 95% cluster-bootstrap CI over items excludes 0. <sup>¶</sup>Open-weight DeepSeek-V4 served through our LiteLLM proxy; Yes/No format only. <sup>‡</sup>Gemini 2.5 Pro cannot switch thinking off; 128 tokens is its minimum budget. Robustness checks in Table 7; 80 some, 80 or and 40 modal items.

A speaker says: "Some guys were cooking."   
Second statement: "Not all guys were cooking."

After the speaker's statement, is it still left   
open whether the second statement is true?   
Answer Yes or No.

Respond with only the letter of the best answer.

Circa multiple choice. multiple-choice format with the dataset’s own options

```yaml
Context: Y has just told X that he/she is
considering switching his/her job.
X asks: "Do you want a new challenge?"
Y answers: "I've been so bored t work."
```

What does Y's answer mean?

Options:   
A. Yes   
B. No   
C. Neither yes nor no (in the middle)

Respond with only the letter of the best answer.

Pub multiple choice. multiple-choice format with the dataset’s own options

Context: Y has just travelled from a different city to meet X.

Choose the best option.

Options:   
A. Yes   
B. No   
C. Yes, subject to some conditions   
D. In the middle, neither yes nor no   
E. Other

Respond with only the letter of the best answer.

Budget stop phrase. think\_b<N> (non-plain): the thought is cut at N tokens and this stop phrase is appended before the close tag; shown for N=0 (empty thought)

<|im\_start|>user   
Sentence: Some elephants are mammals.

Is this sentence a good way to describe the facts?   
Answer Yes or No.

Respond with only the letter of the best

answer.<|im\_end|> <|im\_start|>assistant <think>

Considering the limited time by the user, I have to give the solution based on the thinking directly now.   
</think>

--> 'B. Yes'

## Trace prefix forced close. think\_b<N>\_plain: the first N thinking tokens of the SAME trace, then the native close tag (no stop phrase)

<|im\_start|>user Sentence: Some elephants are mammals.

Is this sentence a good way to describe the facts? Answer Yes or No.

Options: A. No B. Yes

Respond with only the letter of the best answer.<|im\_end|>   
<|im\_start|>assistant   
<think>

Okay, let's see. The question is whether the sentence "Some elephants are mammals" is a good way to describe the facts. The options are A (No) or B (Yes). I need to answer with just the letter.

First, I should recall what elephants are classified as. Elephants are a type of mammal. The scientific classification of elephants is under the class Mammalia. So, all elephants are mammals. The sentence says "some elephants are mammals," which is technically correct because it's a subset. But wait, if all elephants are mammals, then saying "some" is still true, right? Because "some" means at least one. But is it a good way to describe the facts? The original statement might be considered accurate, but maybe it's not the most precise. However, the question is about whether it's a good way to describe the facts. Since all elephants are mammals, the statement is true, but it's not the most comprehensive. However, the question is asking if it's a good way, not if it's the best. So, if the facts are that all elephants are mammals, then "some" is still correct. Therefore, the answer would be Yes. But wait, maybe the answer is No</think>

--> 'B. Yes'

## Nt off. thinking switched off: empty thought block in the chat template

<|im\_start|>user Sentence: Some elephants are mammals.

Is this sentence a good way to describe the facts? Answer Yes or No.

Options:

A. No B. Yes

Respond with only the letter of the best answer.<|im\_end|>   
<|im\_start|>assistant   
<think>

</think>

--> 'B. Yes'

## Forced empty nt. forced-empty thinking (think\_b0\_plain): the native close tag directly after the opener

<|im\_start|>user   
A speaker says: "Some teenagers criticize Guy."   
Second statement: "Not all teenagers criticize Guy."

Would you conclude that the speaker believes the second statement is true? Answer Yes or No.

Options: A. No B. Yes

Respond with only the letter of the best answer.<|im\_end|>   
<|im\_start|>assistant   
<think>   
</think>

## --> 'B'

## Nat.

<|im\_start|>user A speaker says: "All teenagers criticize Guy." Second statement: "Some teenagers criticize Guy."

Would you conclude that the speaker believes the second statement is false? Answer Yes or No.

Options: A. No B. Yes

Respond with only the letter of the best answer.<|im\_end|>   
<|im\_start|>assistant   
<think>

Okay, let's try to figure this out. So, the first statement is "All teenagers criticize Guy," and the second is "Some teenagers criticize Guy." The question is whether the speaker believes the second statement is false. The options are A (No) or B (Yes).

Hmm. Let me think. If someone says "All teenagers criticize Guy," does that mean they think "Some teenagers criticize Guy" is false? Well, in logic, "All" implies that there is at least one. So if all teenagers do something, then there must be some teenagers doing it. So if the first statement is true, then the second statement should also be true. Because "some" is a weaker statement. So if all teenagers do it, then certainly some do it. Therefore, if the speaker

says "All teenagers criticize Guy," they must believe that "Some teenagers criticize Guy" is true, not false. Therefore, the speaker does not believe the second statement is false. So the answer would be No, which is option A. Wait, but wait. What if the speaker is lying? Wait, the question is about what the speaker believes. If the speaker says "All teenagers criticize Guy," then their belief is that all do. But does that mean they think that "some" is false? No, because "some" is a subset of "all." So if all are, then some are. Therefore, the speaker would believe that the second statement is true, not false. Therefore, the answer is No, so option A. But wait, the options are A. No and B. Yes. So the question is whether the speaker believes the second statement is false. If the speaker believes the first statement (all), then they would believe the second (some) is true, so they don't believe the second is false. So the answer is No, so option A. </think>

--> 'A. No'

## C Additional Results

Full per-model results and further analyses behind the main text.

## C.1 Additional Results

Models outside the count. Of those that move the other way, Magistral-Small mostly shifts towards answering Yes: with the question flipped so that No is pragmatic, its pragmatic rate falls by −52.9 pp. Granite-3.3 and Llama-3.1-Nemotron-Nano-8B move towards the pragmatic reading only in the Yes/No format: in a three-way choice both turn literal (some: −22.8 and −8.7 pp; Table 9). The rise of openPangu-Embedded-7B-model is, like Magistral’s, a shift towards answering Yes: with the question flipped its pragmatic rate falls, and in the three-way choice it is flat. In the threeway format, which has no Yes/No labels, 21 of the 24 models with a full-set run shift towards the literal reading on some or or, and only two rise on one word: SmolLM3-3B from its floor (some: +3.9 pp) and Qwen3.5-0.8B on some (+10.7 pp), whose or reading falls (−8.3 pp). openPangu-Embedded-7B-V1.1, null on the core items, shifts on its full held-out set (some: −15.4 pp) and in the three-way choice (−8.4 pp).

Four robustness checks. How thinking stops. Qwen3’s stop phrase mentions limited time. Stopping the trace with only the closing tag gives the same shift at 1,024 tokens (−24.4 against −21.8 pp with the stop phrase). The baseline. Against a forced-empty thinking block instead of the model’s switch, natural thinking changes the some reading of Qwen3-8B by −39.7 pp. Answer bias. A shift towards answering No could imitate a literal reading. For GPT-5 the opposite holds: without the speaker’s sentence, its rate of answering Yes rises with effort, which would push the pragmatic rate up, not down (Table 7). For Qwen3-8B answer polarity does matter: with the question reworded so that No is the pragmatic answer, thinking raises the pragmatic rate by 22.7 pp from a floor near zero. In a three-way forced choice, which has no Yes/No labels, thinking still lowers the pragmatic reading at every Qwen3 size, in Qwen3-8B by −34.0 pp for some and −13.8 pp for or (Section B.3). Speaker knowledge. A pragmatic listener enriches more when the speaker has checked every case (Goodman and Stuhlmüller, 2013); the scenarios we built for this were screened blind to condition by the annotator panel, whose consensus judged some misleading in every full-knowledge scenario and fine in every partial-knowledge one (24 scenarios). Most models make little use of this knowledge: with thinking, the gap between full and partial knowledge is 4.2 pp in Qwen3-32B. The one model that uses it without much deliberation, gpt-oss-120b (84.0 pp at low effort), uses it less at high effort (62.5 pp).

This section gives the full per-model results and additional analyses behind the main text.

Trace probe beyond Qwen3. Figure 7 shows the trace probe on the newest generation. In Figure 4d, the jump of Phi-4-reasoning’s lexical-scale reading after 32 tokens (40.6%) appears only in the sampled answers; the answer-position log-odds L does not move (−1.08 with an empty trace, −1.00 after 32 tokens). The same early jump of the lexical-scale line appears in both Qwen3.5 models, over the first 32–64 tokens and with tight intervals (41.7% and 43.6% in Qwen3.5-4B, 44.1% in Qwen3.5- 9B), so it is a systematic early-prefix effect, not noise. It is confined to the sampled answers: the answer-position log-odds stays negative, that is literal-favouring, and moves little (Qwen3.5-9B: −2.13 with an empty trace, −1.17 after 32 tokens). The line returns to the no-thinking level by the end. For the further families (Figure 6), Skywork-OR1 is the exception: its pragmatic reading drops within the first 32 tokens (or: 75.8% to 45.5%) and then stays flat, so there is no early phase of endorsement. Llama-3.1-Nemotron-Nano-8B starts from a rote floor (3.8% on some with an empty trace), jumps after 32 tokens and falls back by the end. Trace-level shares count a trace only if every prefix gives a valid answer. For Qwen3.5 and Phi-4-mini-reasoning, plain truncation often yields invalid answers: in the full-item run for Qwen3.5-9B on some, 1,163 traces are dropped and 637 survive, so their curves and shares describe the surviving traces and are indicative only. R1-Distill-Llama-8B is not shown because 39–54% of its answers after short truncations are invalid.

<table><tr><td rowspan=1 colspan=5>Yes/No inference                       Three-way choiceModel                                   some                or                some                or</td></tr><tr><td rowspan=1 colspan=1>Qwen3-4B</td><td rowspan=1 colspan=2>-49.8[-53.9, -45.6]   -53.0[-57.6, -48.4]</td><td rowspan=1 colspan=1>-9.9[-12.7, -7.3]</td><td rowspan=1 colspan=1>-11.9[-14.7, -9.2]</td></tr><tr><td rowspan=4 colspan=1>Qwen3-8BQwen3-14BQwen3-30B-A3BQwen3-32B</td><td rowspan=1 colspan=2>-37.7[-42.1, -33.3]  -37.7[-41.6,-33.8]</td><td rowspan=1 colspan=1>-34.0[-38.2, -29.9]</td><td rowspan=1 colspan=1>-13.8[-17.1,-10.6]</td></tr><tr><td rowspan=1 colspan=2>-22.6[-27.1, -18.3] -15.9[-19.1,-12.7]</td><td rowspan=1 colspan=1>-11.4[-14.4,-8.6]</td><td rowspan=1 colspan=1>-2.4[-4.0,-1.0]</td></tr><tr><td rowspan=1 colspan=2>-35.7[-39.8, -31.4]   -40.9[-45.1,-36.8]</td><td rowspan=1 colspan=1>-12.8[-15.7,-10.1]</td><td rowspan=1 colspan=1>-11.8[-14.3,-9.3]</td></tr><tr><td rowspan=1 colspan=2>-55.6[-60.1, -50.8]  –52.7[–57.0, –48.3]</td><td rowspan=1 colspan=1>-38.0[-42.9, -33.1]</td><td rowspan=1 colspan=1>-19.7[-23.1,-16.3]</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-0.8B</td><td rowspan=1 colspan=1>-5.0[-8.1, -2.1]</td><td rowspan=1 colspan=1>-14.3[-17.9,-10.9]</td><td rowspan=1 colspan=1>10.7 [6.2, 14.9]</td><td rowspan=1 colspan=1>-8.3[-12.0, -4.8]</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-2B</td><td rowspan=1 colspan=1>-31.6[-36.3,-26.8]</td><td rowspan=1 colspan=1>-25.4[-29.9,-20.9]</td><td rowspan=1 colspan=1>-8.7[-11.1, -6.4]</td><td rowspan=1 colspan=1>-8.0[-10.2,-5.9]</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-4B</td><td rowspan=1 colspan=1>–50.3[-55.4, -45.3]</td><td rowspan=1 colspan=1>-26.2[-30.6, -22.0]</td><td rowspan=1 colspan=1>-13.4[-16.4,-10.6]</td><td rowspan=1 colspan=1>-14.1[-17.0,-11.3]</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1>-30.4[-35.1, -25.7]</td><td rowspan=1 colspan=1>-25.3[-29.7, -21.1]</td><td rowspan=1 colspan=1>-10.8[-13.7,-8.1]</td><td rowspan=1 colspan=1>-6.8[-9.0,-4.8]</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B</td><td rowspan=1 colspan=1>-13.7[-20.1,-7.2]</td><td rowspan=1 colspan=1>–21.7[-26.1, –17.7]</td><td rowspan=1 colspan=1>-1.2[-2.2, −0.4]</td><td rowspan=1 colspan=1>-3.0[-4.7,-1.6]</td></tr><tr><td rowspan=1 colspan=1>Phi-4-reasoning</td><td rowspan=1 colspan=1>-34.2[-38.2, -30.0]</td><td rowspan=1 colspan=1>-41.1[-44.9, -37.3]</td><td rowspan=1 colspan=1>-26.9[-30.9,-23.0]</td><td rowspan=1 colspan=1>-25.0[-28.8, -21.2]</td></tr><tr><td rowspan=1 colspan=1>Phi-4-mini-reasoning</td><td rowspan=1 colspan=1>-21.0[-24.7, -17.2]</td><td rowspan=1 colspan=1>-15.9[-20.0,-11.9]</td><td rowspan=1 colspan=1>-17.7[-21.1, -14.2]</td><td rowspan=1 colspan=1>–25.6[–28.9, –22.2]</td></tr><tr><td rowspan=1 colspan=1>R1-Distill-Llama-8B</td><td rowspan=1 colspan=1>-50.1[-54.4, -45.8]</td><td rowspan=1 colspan=1>-62.7[-66.8, -58.4]</td><td rowspan=1 colspan=1>-31.4[-35.0, -28.1]</td><td rowspan=1 colspan=1>-33.3[-36.6, -30.2]</td></tr><tr><td rowspan=1 colspan=1>Skywork-OR1-7B</td><td rowspan=1 colspan=1>-19.0[-23.9,-14.0]</td><td rowspan=1 colspan=1>-30.1 [–35.2, -25.0]</td><td rowspan=1 colspan=1>-28.4[–32.4, –24.6]</td><td rowspan=1 colspan=1>-33.1[–37.4, -28.8]</td></tr><tr><td rowspan=2 colspan=1>EXAONE-Deep-7.8BNemotron-Nano-9B-v2</td><td rowspan=1 colspan=1>-33.2[-37.8, -28.6]</td><td rowspan=1 colspan=1>-33.0[-37.6,-28.2]</td><td rowspan=1 colspan=1>-27.2[–30.4, -24.3]</td><td rowspan=1 colspan=1>-30.0[-33.2, -26.9]</td></tr><tr><td rowspan=1 colspan=1>–56.0[-61.0, -50.7]</td><td rowspan=1 colspan=1>—40.4[-45.3,-35.7]</td><td rowspan=1 colspan=1>0.0[−1.4, 1.6]</td><td rowspan=1 colspan=1>-1.0[-2.2,0.0]</td></tr><tr><td rowspan=1 colspan=1>Granite-3.3-8B</td><td rowspan=1 colspan=1>12.6[9.3, 15.6]</td><td rowspan=1 colspan=1>18.9[15.7, 22.1]</td><td rowspan=1 colspan=1>-22.8[-27.2, −–18.3]</td><td rowspan=1 colspan=1>-8.8[-13.2, -4.4]</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-Nemotron-Nano-8B</td><td rowspan=1 colspan=1>12.4 [9.9, 15.1]</td><td rowspan=1 colspan=1>4.1 [2.9, 5.4]</td><td rowspan=1 colspan=1>-8.7[-11.6, -6.2]</td><td rowspan=1 colspan=1>-4.2[-6.1, -2.4]</td></tr><tr><td rowspan=1 colspan=1>SmolLM3-3B</td><td rowspan=1 colspan=1>-7.2[-11.6, -2.8]</td><td rowspan=1 colspan=1>3.9[–1.2, 8.9]</td><td rowspan=1 colspan=1>3.9[1.6, 6.3]</td><td rowspan=1 colspan=1>1.0[-0.1, 2.1]</td></tr><tr><td rowspan=1 colspan=1>openPangu-Embedded-7B-V1.1</td><td rowspan=1 colspan=1>-15.4[-19.4,-11.4]</td><td rowspan=1 colspan=1>-1.6[-3.2,0.0]</td><td rowspan=1 colspan=1>-8.4[-11.2, -6.0]</td><td rowspan=1 colspan=1>-3.0[-4.3, -1.8]</td></tr><tr><td rowspan=1 colspan=1>openPangu-Embedded-7B-model</td><td rowspan=1 colspan=1>6.7 [3.6, 9.8]</td><td rowspan=1 colspan=1>-4.8[-7.6, -2.1]</td><td rowspan=1 colspan=1>-0.7[-2.9, 1.6]</td><td rowspan=1 colspan=1>-0.1[−1.3, 1.1]</td></tr></table>

Table 9: On the full held-out sets, some and or turn literal with thinking in both formats in most models. Change with thinking (pp) [95% CI], 300 items per trigger; grey: CI includes 0. Some models start near the floor in the three-way format.

Further context-driven sets. Table 10 gives the Circa and NOPE results summarised in Section 4.1.

German and Chinese. The 160 IMPPRES some/or test pairs were machine-translated into German and Chinese by Qwen3-32B, a model from the same family as most subjects; Chinese pairs left untranslated (4 of 160) were dropped. A blind consensus check of 20 Chinese items by the annotator panel found 3 with meaning or trigger errors, including a negated-connective flip, so the Chinese results are not human-verified. With these caveats, thinking lowers the pragmatic reading in both languages as in English: in Qwen3-8B by −37.9 pp in German, −17.1 pp in Chinese and −39.2 pp on the English sources; in Qwen3-14B by −19.6 pp in German; and in Qwen3.5-9B by −27.9 pp in German and −25.8 pp in Chinese (−16.7 pp on the English sources); raising gpt-oss-20b from low to high effort changes it by −30.4 pp in German.

<table><tr><td rowspan=1 colspan=3>Model                            Circa        NOPE</td></tr><tr><td rowspan=5 colspan=1>Qwen3-4BQwen3-8BQwen3-14BQwen3-30B-A3BQwen3-32B</td><td rowspan=1 colspan=1>15.3[10.4, 20.4]</td><td rowspan=1 colspan=1>7.8 [2.3, 13.1]</td></tr><tr><td rowspan=1 colspan=1>17.5[12.4, 22.7]</td><td rowspan=1 colspan=1>9.9 [4.5, 15.5]</td></tr><tr><td rowspan=1 colspan=1>24.4[18.9, 30.0]</td><td rowspan=1 colspan=1>9.6[4.4, 14.9]</td></tr><tr><td rowspan=1 colspan=1>8.3 [4.0, 12.7]</td><td rowspan=1 colspan=1>10.4 [5.3, 15.5]</td></tr><tr><td rowspan=1 colspan=1>3.1[−0.9, 7.1]</td><td rowspan=1 colspan=1>9.2[3.8, 14.4]</td></tr><tr><td rowspan=10 colspan=1>Qwen3.5-4BQwen3.5-9BPhi-4-reasoningNemotron-Nano-9B-v2Llama-3.1-Nemotron-Nano-8BSkywork-OR1-7BGranite-3.3-8BSmolLM3-3Bgpt-oss-20b (high-low)</td><td rowspan=2 colspan=1>16.7 [12.9, 20.4]</td><td></td></tr><tr><td rowspan=1 colspan=1>5.8[1.4, 10.4]</td></tr><tr><td rowspan=1 colspan=1>8.3[4.9, 11.6]</td><td rowspan=1 colspan=1>6.6[3.1, 10.3]</td></tr><tr><td rowspan=1 colspan=1>13.2[8.9, 17.9]</td><td rowspan=1 colspan=1>16.9[11.1, 22.6]</td></tr><tr><td rowspan=1 colspan=1>15.7 [11.1, 20.5]</td><td rowspan=1 colspan=1>5.4[1.0, 9.8]</td></tr><tr><td rowspan=1 colspan=1>28.1 [23.1, 32.9]</td><td rowspan=1 colspan=1>4.1[−0.4, 8.5]</td></tr><tr><td rowspan=1 colspan=1>32.8 [27.9, 38.0]</td><td rowspan=1 colspan=1>24.3 [17.9, 30.6]</td></tr><tr><td rowspan=1 colspan=1>1.3[-2.5, 5.2]</td><td rowspan=1 colspan=1>5.6[1.4, 9.6]</td></tr><tr><td rowspan=1 colspan=1>1.2[-3.7, 6.3]</td><td rowspan=1 colspan=1>6.0[1.2, 10.8]</td></tr><tr><td rowspan=1 colspan=1>1.3[−1.7, 4.5]</td><td rowspan=1 colspan=1>3.4 [0.2, 6.6]</td></tr></table>

Table 10: Thinking raises or keeps accuracy on two further context-driven sets in every model. Change with thinking (pp) [95% CI]; grey: CI includes 0; Circa (Louis et al., 2020), NOPE (Parrish et al., 2021).

Relation types in the newest generation. IMP-PRES pairs differ in how the two sentences relate (for example, a negated first sentence with a positive second one). In Qwen3.8-27B thinking lowers the pragmatic reading of some for most relation

![](images/1ef337a9099347adea4683060de3f4dbfd8202b92e77684cd5f2eebce8de1583.jpg)  
trace truncated at k tokens, then the answer is forced (log scale)

Figure 6: Trace probe on further families (EXAONE-Deep-7.8B, Nemotron-Nano-9B-v2, Phi-4-mini-reasoning, Skywork-OR1-7B, Llama-3.1-Nemotron-Nano-8B): pragmatic reading R after the first t tokens of the model’s own trace, with 95% bands over items. The y-axis runs from 0 to 100 (0 to 80 in Figure 4). In panel (e) the value at t = 0 is the model’s collapse to one answer under a forced-empty thinking block.

![](images/4d967441642396296043a34ba0c05cd561380b1b111f5595b14c0e6e2d7d3e91.jpg)  
trace truncated at k tokens, then the answer is forced (log scale)  
Figure 7: Trace probe on the newest generation (Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B): pragmatic reading R after the first t tokens of the model’s own trace, with 95% bands over items. The early lexical-scale jump in the Qwen3.5 panels is systematic and confined to sampled answers (see text).

types (−59.7 pp when both are negated) but raises it for a negated first sentence with a positive second one (+56.2 pp), so its pooled change hides opposite effects.

Full held-out sets in both formats. Table 9 gives the effect on the full held-out IMPPRES some and or sets for every model with a full-set run.

Every item family. Figure 8 shows the thinking effect for every item family and model behind Figure 2.

Preliminary seed experiments. If the literal reading comes from what thinking is spent on, the start of the trace should matter. Intent-Seeded Thinking (IST) opens the thinking block with one fixed sentence about what the speaker intends to convey; two control sentences of the same length (26 tokens) ask for careful step-by-step reasoning or for checking what the words strictly entail (Section B.1). The intent seed yields the most enrichment of the three seeds (Table 11). In Qwen3-8B it raises the pragmatic reading of some from 0.0% under natural thinking to 12.8% (29.5% without thinking), against 2.6% with the neutral seed; in Qwen3-32B it raises it from 15.4% to 29.5%, while both controls leave it at or below the natural level there. Accuracy on indirect answers moves from 83.5% to 78.2%. On the newest generation the seed lifts the reading above its no-thinking level: Qwen3.5-9B reads some pragmatically in 44.9% of items with the seed, against 1.3% under natural thinking and 39.7% without thinking, while the neutral and logic seeds reach 11.5% and 5.1%. Prompted chain-of-thought with thinking off moves the reading less than native thinking does, so producing a chain of thought is not enough to cause the full shift.

Takeaway: what the trace starts with steers the reading chosen; a sentence about speaker intent recovers more of the lost enrichment than length-matched control sentences.

![](images/df340854df1b7e7e4bfaa12eef929e9a4c2867c0a75243723da7ca686fb4d62b.jpg)  
thinking effect (pp) = top condition − grey baseline; frame: 95% CI excludes 0; values in Tables 6 and 9 (App.); hatched: not run † Yes/No format only; reverses in a 3-way choice, and for Sonnet largely a shift towards answering Yes (App.). Grey: Gemini 2.5 Flash, answer failures >10% at thinking doses.

Figure 8: Thinking effect for every item family and model. Change in pragmatic reading with thinking (pp); vermilion: more literal; blue: more pragmatic; framed: 95% CI excludes 0; hatched: not run. Values in Tables 8 and 12.
<table><tr><td></td><td colspan="3">Qwen3-8B</td><td colspan="3">Qwen3-32B</td></tr><tr><td>Method</td><td>some ↑</td><td>or↑</td><td>LUDWIG↑</td><td>some ↑</td><td>or↑</td><td>LUDWIG↑</td></tr><tr><td>No thinking (NT)</td><td> $2 9 . 5 \pm 5 . 9$ </td><td> $3 7 . 4 \pm 3 . 5$ </td><td> $7 6 . 5 \pm 2 . 6$ </td><td> $6 5 . 4 \pm 3 . 8$ </td><td> $5 5 . 6 \pm 3 . 5$ </td><td> $8 3 . 9 \pm 1 . 2$ </td></tr><tr><td>Natural thinking (NAT)</td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 1 . 8$ </td><td> ${ \bf 8 3 . 5 \pm 1 . 6 }$ </td><td>一  $1 5 . 4 \pm 3 . 8$ </td><td> $8 . 1 \pm 3 . 5$ </td><td> $8 3 . 5 \pm 1 . 6$ </td></tr><tr><td>Prompted CoT, thinking offª</td><td> $\underline { { 1 0 . 3 \pm 5 . 9 } }$ </td><td> $\underline { { 6 . 1 \pm 3 . 0 } }$ </td><td> $7 5 . 4 \pm 2 . 2$ </td><td> ${ \bf 4 6 . 2 \pm 1 0 . 2 }$ </td><td> ${ \bf 4 5 . 5 \pm 1 3 . 9 }$ </td><td> $8 0 . 7 \pm 2 . 2$ </td></tr><tr><td colspan="7">Seed sentence opening the thinking trace (26 tokens each)</td></tr><tr><td>Neutral seed</td><td> $2 . 6 \pm 2 . 2$ </td><td> $1 . 0 \pm 1 . 8$ </td><td> $7 7 . 5 \pm 2 . 2$ </td><td> $1 0 . 3 \pm 5 . 9$ </td><td> $6 . 1 \pm 6 . 1$ </td><td> ${ \bf 8 4 . 9 \pm 2 . 2 }$ </td></tr><tr><td>Logic seed</td><td> $9 . 0 \pm 5 . 9$ </td><td> $4 . 0 \pm 3 . 5$ </td><td> $7 8 . 9 \pm 2 . 8$ </td><td> $9 . 0 \pm 9 . 7$ </td><td> $8 . 1 \pm 3 . 5$ </td><td> $8 0 . 4 \pm 2 . 2$ </td></tr><tr><td>Intent seed (IST)</td><td> ${ \bf 1 2 . 8 \pm 8 . 0 }$ </td><td> $1 5 . 2 \pm { \bf 6 . 1 }$ </td><td> $7 8 . 2 \pm 2 . 2$ </td><td> $2 9 . 5 \pm 5 . 9$ </td><td> $2 6 . 3 \pm 4 . 6$ </td><td> $8 4 . 6 \pm 2 . 2$ </td></tr><tr><td colspan="7">Training-time fix, same backbone (Park et al., 2026)</td></tr><tr><td>PragReST-8B, NAT</td><td>19.2±3.8 21.2±0.0</td><td></td><td> $8 2 . 1 \pm 1 . 1$ </td><td></td><td></td><td></td></tr></table>

Seeds run with natural thinking; <sup>a</sup>from a separate run with the same settings. PragReST-8B is trained against literal readings (Park et al., 2026).  
Table 11: The intent seed restores more enrichment than length-matched control seeds. Pragmatic reading and LUDWIG accuracy (%); mean ± s.d. over seeds; bold/underlined: best/second among thinking rows.

Per-family results and size trends.

Training stages. The pre-registered within-stage thinking effect at each training stage was also run; its outcome is summarised under H3 in Table 13.

Thinking effects per model. Table 12 gives the values and intervals behind Figure 2.

Dose–response over the thinking budget. Figure 5 shows the reading chosen by Qwen3-8B as the thinking budget grows.

<table><tr><td rowspan=1 colspan=7>Generalized implicatures: ∆R (pp) ↑                $\Delta \mathbf { A c c . } \left( \mathbf { p } \mathbf { p } \right) \uparrow$ Model                         some        or        modals   lexical scales IMPPRES all  LUDWIG</td></tr><tr><td rowspan=1 colspan=7>Qwen3, distilled from larger models (0.6B–30B-A3B) or fully RL-trained (32B); NAT — NT</td></tr><tr><td rowspan=1 colspan=6>Qwen3-0.6B†                 $\downarrow 5 0 . 0 \pm 1 0 . 2 ^ { * }$   $\downarrow 8 3 . 8 \pm 4 . 6 ^ { * }$   $\downarrow 6 2 . 5 \pm 1 2 . 5 ^ { * }$   $\downarrow 7 0 . 4 \pm 2 . 0 ^ { * }$   $\downarrow 5 9 . 1 \pm { 1 . 6 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 9 . 8 \pm 3 . 2 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=6> $\mathrm { Q w e n } 3 { - } 1 . 7 \mathrm { B }$                   $\downarrow 1 0 . 3 \pm 8 . 9$   $\downarrow 4 7 . 5 \pm 4 . 6 ^ { * }$   $\downarrow 3 3 . 3 \pm 9 . 5 ^ { * }$   $\downarrow 2 5 . 4 \pm 7 . 3 ^ { * }$   $\downarrow 2 3 . 6 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 7 . 2 \pm 0 . 6 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=6> $\mathrm { Q w e n } 3 – 4 \mathrm { B }$                     $\downarrow 3 7 . 2 \pm 4 . 4 ^ { * }$   $\downarrow 6 1 . 6 \pm 4 . 6 ^ { * }$   $\downarrow 3 2 . 3 \pm 1 4 . 4 ^ { * }$   $\downarrow 7 . 7 \pm 2 . 0 ^ { * }$   $\downarrow 2 6 . 5 \pm 2 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 4 \pm 2 . 7$ </td></tr><tr><td rowspan=1 colspan=6>Qwen3-8B                    $\downarrow 2 8 . 2 \pm 5 . 9 ^ { * }$   $\downarrow 3 6 . 4 \pm 5 . 2 ^ { * }$   $\downarrow 1 2 . 5 \pm 6 . 2 ^ { * }$   $\uparrow 5 . 5 \pm { 1 . 4 } ^ { * }$   $\downarrow 9 . 9 \pm 2 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 5 \pm 2 . 8$ </td></tr><tr><td rowspan=2 colspan=6>Qwen3-14B                   $\downarrow 2 1 . 8 \pm 8 . 0 ^ { * }$   $\downarrow 1 3 . 1 \pm 6 . 3 ^ { * }$    $\uparrow 4 . 2 \pm 6 . 5$    $\uparrow 6 . 8 \pm { 1 . 7 } ^ { * }$    $\uparrow 0 . 5 \pm 1 . 5$  $\mathrm { Q w e n 3 - } 3 0 \mathrm { B - } \mathrm { A } 3 \mathrm { B }$                $\downarrow 2 8 . 2 \pm 2 . 2 ^ { * }$   $\downarrow 3 9 . 4 \pm 6 . 1 ^ { * }$   $\downarrow 1 9 . 8 \pm 6 . 5 ^ { * }$    $\uparrow 0 . 7 \pm 2 . 9$   $\downarrow 1 5 . 9 \pm 3 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 . 4 \pm 0 . 6$ </td></tr><tr><td rowspan=1 colspan=1> $\uparrow 9 . 5 \pm 2 . 1 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=6>Qwen3-32B                   $\downarrow 5 1 . 3 \pm 8 . 0 ^ { * }$  ↓46.5±4.6   $\downarrow 1 1 . 5 \pm 7 . 9$   $\uparrow 4 . 0 \pm 2 . 4 ^ { * }$   $\downarrow 1 8 . 5 \pm { 1 . 4 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 5 \pm 3 . 2$ </td></tr><tr><td rowspan=1 colspan=4>OLMo-3-Think, same recipe at two sizes; NAT — forced-empty thinking</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>OLMo-3-7B-Think</td><td rowspan=1 colspan=3> $\uparrow 2 . 6 \pm 8 . 0$    ↓5.1±4.6    $\downarrow 4 . 2 \pm { 1 . 8 }$ </td><td rowspan=1 colspan=2> $\uparrow 1 1 . 3 \pm { 1 . 8 } ^ { * }$   $\uparrow 2 . 4 \pm 1 . 6$ </td><td rowspan=1 colspan=1> $\uparrow 9 . 5 \pm 3 . 6 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>OLMo-3-32B-Think</td><td rowspan=1 colspan=5> $\downarrow 1 1 . 5 \pm 2 . 4 ^ { * }$    $\downarrow 1 . 0 \pm 3 . 1$    $\uparrow 2 9 . 7 \pm 8 . 3 ^ { * }$   $\uparrow 6 . 0 \pm 1 . 6 ^ { * }$   $\uparrow 1 9 . 0 \pm 3 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 0 \pm 2 . 3$ </td></tr><tr><td rowspan=1 colspan=1>gpt-oss, reasoning effort high — low</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>gpt-oss-20b</td><td rowspan=1 colspan=3> $\downarrow 3 0 . 8 \pm 6 . 0 ^ { * }$   $\downarrow 2 4 . 7 \pm { 1 0 . 0 } ^ { * }$   $\downarrow 2 0 . 3 \pm 8 . 8 ^ { * }$ </td><td rowspan=1 colspan=2> $\downarrow 0 . 3 \pm 2 . 7$   $\downarrow 1 6 . 7 \pm 3 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 2 \pm 2 . 1$ </td></tr><tr><td rowspan=1 colspan=1>gpt-oss-120b</td><td rowspan=1 colspan=3> $\downarrow 6 . 4 \pm 7 . 2$    $\downarrow 1 1 . 1 \pm 5 . 6 ^ { * }$   $\downarrow 2 5 . 0 \pm 7 . 7 ^ { * }$ </td><td rowspan=1 colspan=2> $\uparrow 6 . 7 \pm 1 . 7 ^ { * }$   $\downarrow 1 2 . 0 \pm { 1 . 9 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 0 \pm 1 . 4$ </td></tr><tr><td rowspan=1 colspan=1>Popular models: natural thinking —</td><td rowspan=1 colspan=3>forced-empty thinking</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Phi-4-reasoning</td><td rowspan=1 colspan=2> $\downarrow 5 3 . 8 \pm 7 . 7 ^ { * }$   $\downarrow 3 8 . 4 \pm 7 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 6 . 7 \pm 1 5 . 8$ </td><td rowspan=1 colspan=1> $\downarrow 5 . 1 \pm 2 . 1 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 7 . 1 \pm 2 . 3 ^ { \ast }$ </td><td rowspan=1 colspan=1>↑17.5±3.2*</td></tr><tr><td rowspan=1 colspan=1>Magistral-Small</td><td rowspan=1 colspan=2> $\uparrow 9 . 0 \pm 4 . 4 ^ { * }$    $\uparrow 2 0 . 2 \pm 4 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 4 \pm 3 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 3 \pm 2 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 9 \pm 0 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 5 . 8 \pm 4 . 6 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>R1-Distill-Llama-8B</td><td rowspan=1 colspan=2> $\downarrow 5 0 . 0 \pm 7 . 7 ^ { * }$   $\downarrow 5 7 . 6 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 5 . 4 \pm 1 0 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 2 . 9 \pm 6 . 3 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 7 . 8 \pm 4 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 6 . 0 \pm 3 . 7 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>R1-Distill-Llama-70B</td><td rowspan=1 colspan=3> $\downarrow 5 0 . 0 \pm 7 . 7 ^ { \ast }$    $\downarrow 3 9 . 4 \pm 3 . 0 ^ { * }$   $\downarrow 2 8 . 1 \pm { 1 0 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 . 5 \pm { 1 . 4 }$ </td><td rowspan=1 colspan=1> $\downarrow 3 4 . 7 \pm 0 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 9 . 8 \pm 4 . 4 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Further open reasoning models: natu</td><td rowspan=1 colspan=3>ral thinking – thinking off (forced-empty where</td><td rowspan=1 colspan=1>no switch exists)</td><td rowspan=1 colspan=1>sts)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SmolLM3-3B</td><td rowspan=1 colspan=3> $\downarrow 6 . 4 \pm 8 . 0$   ↑12.1±3.0   $\uparrow 1 6 . 7 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 9 \pm 1 . 7$ </td><td rowspan=1 colspan=1> $\uparrow 1 3 . 5 \pm 0 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 0 . 4 \pm 4 . 0$ </td></tr><tr><td rowspan=1 colspan=1>Granite-3.3-8B</td><td rowspan=1 colspan=2> $\uparrow 1 0 . 3 \pm 5 . 9$   $\uparrow 2 3 . 2 \pm 7 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 3 \pm 3 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 4 \pm 2 . 5$ </td><td rowspan=1 colspan=1> $\uparrow 1 2 . 7 \pm { 1 . 6 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 5 . 3 \pm 0 . 0$ </td></tr><tr><td rowspan=1 colspan=1>Nemotron-Nano-9B-v2</td><td rowspan=1 colspan=2> $\downarrow 5 9 . 0 \pm 4 . 0 ^ { * }$   $\downarrow 4 3 . 4 \pm 6 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 8 . 3 \pm 1 0 . 0$ </td><td rowspan=1 colspan=1> $\uparrow 4 . 9 \pm { 1 . 3 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 7 . 4 \pm 3 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 7 \pm 2 . 5 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>AceReason-Nemotron-7B</td><td rowspan=1 colspan=2> $\downarrow 2 5 . 6 \pm 1 1 . 8 ^ { * }$   $\downarrow 3 8 . 4 \pm { 1 2 . 6 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 . 1 \pm 7 . 9$ </td><td rowspan=1 colspan=1> $\downarrow 1 4 . 6 \pm 6 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 2 . 0 \pm { 4 . 3 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 6 . 8 \pm 5 . 6 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Skywork-OR1-7B</td><td rowspan=1 colspan=2> $\downarrow 1 4 . 1 \pm 5 . 9$   $\downarrow 3 4 . 3 \pm 1 6 . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 . 0 \pm 9 . 5$ </td><td rowspan=1 colspan=1> $\downarrow 2 0 . 5 \pm 3 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 4 \pm 4 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 5 \pm 6 . 6 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>MiMo-7B-SFT</td><td rowspan=1 colspan=2> $\downarrow 6 0 . 3 \pm 2 . 2 ^ { * }$   $\downarrow 4 0 . 4 \pm 7 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 5 . 0 \pm 6 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 3 . 0 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 5 . 8 \pm 3 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 0 . 4 \pm 1 . 2 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>MiMo-7B-RL</td><td rowspan=1 colspan=2> $\downarrow 4 4 . 9 \pm 4 . 4 ^ { * }$    $\downarrow 3 9 . 4 \pm 5 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 . 0 \pm 1 2 . 6$ </td><td rowspan=1 colspan=1> $\downarrow 7 . 5 \pm 1 . 7 ^ { \ast }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 4 \pm 2 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 6 . 3 \pm { 1 . 1 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>QwQ-32B</td><td rowspan=1 colspan=1> $\downarrow 2 3 . 1 \pm 3 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 4 . 0 \pm 7 . 0$ </td><td rowspan=1 colspan=1> $\uparrow 2 2 . 9 \pm 4 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 2 \pm { 0 . 4 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 8 . 5 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 1 . 2 \pm { 1 . 6 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Llama-3.3-Nemotron-Super-49B</td><td rowspan=1 colspan=1> $\downarrow 2 8 . 2 \pm { 1 6 . 0 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 1 . 1 \pm 1 2 . 6$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 1 \pm 1 0 . 0$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 6 \pm 2 . 3 ^ { \ast }$ </td><td rowspan=1 colspan=1> $\downarrow 3 . 6 \pm 3 . 9$ </td><td rowspan=1 colspan=1> $\uparrow 1 . 4 \pm 1 . 6$ </td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-Nemotron-Nano-8B</td><td rowspan=1 colspan=1> $\uparrow 1 3 . 5 \pm 8 . 3 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 5 \pm 3 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 6 . 6 \pm 5 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 1 . 7 \pm 3 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 4 . 6 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 5 . 1 \pm 3 . 0 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Phi-4-mini-reasoning</td><td rowspan=1 colspan=1> $\downarrow 7 . 7 \pm 8 . 3$ </td><td rowspan=1 colspan=1> $\downarrow 1 3 . 6 \pm 3 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 4 \pm 1 0 . 3$ </td><td rowspan=1 colspan=1> $\downarrow 1 7 . 6 \pm { 1 . 0 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 3 . 1 \pm 2 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 8 . 3 \pm 4 . 8 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>EXAONE-Deep-7.8B</td><td rowspan=1 colspan=1> $\downarrow 4 0 . 4 \pm 6 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 4 . 7 \pm 8 . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 0 . 8 \pm 1 3 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 4 \pm 1 . 4$ </td><td rowspan=1 colspan=1> $\downarrow 6 . 0 \pm 3 . 4$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 8 \pm 3 . 9$ </td></tr><tr><td rowspan=1 colspan=1>openPangu-Embedded-7B-V1.18</td><td rowspan=1 colspan=1> $\downarrow 1 2 . 8 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\downarrow 3 . 0 \pm 3 . 0$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 4 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 6 . 1 \pm 2 . 3 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 9 \pm 1 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 1 \pm 4 . 2$ </td></tr><tr><td rowspan=1 colspan=1>openPangu-Embedded-7B-model8</td><td rowspan=1 colspan=1> $\uparrow 1 4 . 1 \pm { 1 1 . 1 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 4 . 0 \pm { 1 . 8 }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 4 \pm 6 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 3 \pm 2 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 2 . 3 \pm { 0 . 9 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 8 . 9 \pm 2 . 2 ^ { \ast }$ </td></tr><tr><td></td><td rowspan=1 colspan=1>g — thinking off</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-0.8B</td><td rowspan=1 colspan=1> $\downarrow 1 1 . 5 \pm 3 . 8$ </td><td rowspan=1 colspan=1> $\downarrow 1 4 . 1 \pm { 6 . 3 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 4 \pm 9 . 5 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 7 . 2 \pm { 1 . 3 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 9 . 4 \pm 2 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 5 . 6 \pm 2 . 2$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { Q w e n } 3 . 5 – 2 \mathrm { B }$ </td><td rowspan=1 colspan=1> $\downarrow 1 4 . 1 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\downarrow 3 2 . 3 \pm 8 . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 2 . 9 \pm 7 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 3 . 5 \pm 2 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 6 \pm 3 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 . 8 \pm 0 . 6$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-4B</td><td rowspan=1 colspan=1> $\downarrow 3 9 . 1 \pm 3 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 4 . 8 \pm 7 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 0 \pm 6 . 8$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 3 \pm 1 . 7$ </td><td rowspan=1 colspan=1> $\downarrow 3 . 2 \pm 3 . 2$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 9 \pm 1 . 4$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1> $\downarrow 3 9 . 1 \pm 3 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 3 . 2 \pm { 1 . 6 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 2 . 6 \pm 5 . 4$ </td><td rowspan=1 colspan=1> $\uparrow 4 . 4 \pm 1 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 7 . 6 \pm 1 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 0 \pm 2 . 2 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B (default effort)</td><td rowspan=1 colspan=1> $\downarrow 1 6 . 7 \pm 9 . 7$ </td><td rowspan=1 colspan=1> $\downarrow 2 2 . 2 \pm 4 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 8 . 3 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 5 . 5 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 8 . 9 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 5 \pm { 1 . 1 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B (low effort)</td><td rowspan=1 colspan=1> $\downarrow 2 3 . 1 \pm 3 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 7 . 4 \pm 1 2 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 . 1 \pm 1 1 . 0$ </td><td rowspan=1 colspan=1> $\uparrow 1 4 . 3 \pm 2 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 8 . 7 \pm 3 . 9 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 8 . 8 \pm { 1 . 6 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { Q w e n } 3 . 8 – 2 7 \mathrm { B ( m e d i u m e f f o r t ) }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 0 \pm 0 . 0$ </td><td rowspan=1 colspan=1> $\downarrow 1 2 . 1 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 5 . 6 \pm 6 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 0 . 6 \pm { 1 . 8 } ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 0 . 0 \pm 0 . 5$ </td><td rowspan=1 colspan=1> $\uparrow 8 . 8 \pm { 0 . 6 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>Further distilled and updated model</td><td rowspan=1 colspan=1>s: natural thin</td><td rowspan=1 colspan=1>king — no or forc</td><td rowspan=1 colspan=1>ed-empty thinking</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qwen3-4B-Thinking-2507</td><td rowspan=1 colspan=1> $\downarrow 5 6 . 4 \pm 9 . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 4 8 . 5 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 4 1 . 7 \pm 4 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 8 \pm 2 . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 3 . 9 \pm { 1 . 0 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 5 . 3 \pm 3 . 7$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { Q w e n } 3 . 5 \substack { - 3 5 \mathrm { B } - \mathrm { A } 3 \mathrm { B } }$ </td><td rowspan=1 colspan=1> $\downarrow 5 . 1 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $\downarrow 1 . 0 \pm 3 . 5$ </td><td rowspan=1 colspan=1> $\uparrow 1 1 . 5 \pm 3 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 8 \pm { 1 . 0 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 9 \pm { 1 . 3 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 4 . 2 \pm { 1 . 1 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>R1-Distill-Qwen-7B</td><td rowspan=1 colspan=1> $\downarrow 2 1 . 8 \pm 4 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 3 5 . 4 \pm 1 4 . 3 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 7 . 3 \pm 1 0 . 0$ </td><td rowspan=1 colspan=1> $\downarrow 1 3 . 0 \pm 4 . 8 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 1 . 8 \pm 3 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 0 \pm 2 . 2$ </td></tr><tr><td rowspan=1 colspan=1>R1-Distill-Qwen-14B</td><td rowspan=1 colspan=1> $\downarrow 1 4 . 1 \pm 5 . 9$ </td><td rowspan=1 colspan=1> $\uparrow 7 . 1 \pm 7 . 6$ </td><td rowspan=1 colspan=1> $\uparrow 3 7 . 5 \pm 5 . 4 ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 3 . 8 \pm { 1 . 0 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 1 6 . 8 \pm { 3 . 1 } ^ { * }$ </td><td rowspan=1 colspan=1> $\uparrow 8 . 8 \pm 3 . 2 ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>DeepSeek-R1-0528-Qwen3-8B</td><td rowspan=1 colspan=1> $\downarrow 2 4 . 4 \pm 2 . 2 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 1 0 . 1 \pm 4 . 6 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 6 . 2 \pm 5 . 4$ </td><td rowspan=1 colspan=1> $\downarrow 1 7 . 2 \pm 3 . 0 ^ { * }$ </td><td rowspan=1 colspan=1> $\downarrow 2 . 4 \pm 2 . 6$ </td><td rowspan=1 colspan=1> $\uparrow 6 . 0 \pm 2 . 6$ </td></tr></table>

<sup>§</sup>Run with the vendor’s mandatory Chinese system prompt (without it the model degenerates); thinking switched with ‘ /no\_think’ (fast) vs. the default (slow).  
Table 12: Full results: the literal shift on some and or holds in 30 of 40 open reasoning models. Change in the reading chosen from no thinking (NT) to natural thinking (NAT) on the same weights, by item family, on the held-out split; for OLMo-3-Think the baseline is a forced-empty thinking block (the model has no switch), and for gpt-oss the contrast is high minus low reasoning effort. Generalized families use the inference framing: IMPPRES quantifier, connective and modal triggers (Jeretic et al., 2020) and scalar-diversity lexical scales (van Tiel et al., 2016; Gotzner et al., 2018; Pankratz and van Tiel, 2021). LUDWIG: indirect answers (Ruis et al., 2023). Cells: mean ± standard deviation over 3 decoding seeds; blue = more pragmatic or more accurate, vermilion = more literal; <sup>\*</sup>: the 95% cluster-bootstrap confidence interval over items excludes 0 (uncorrected; the mixed-effects tests are in Section C.1). <sup>†</sup>Without thinking, Qwen3-0.6B accepts 86–100% of inferences (a yes-bias), so it is outside the pre-registered 10–90% range and is excluded from the dose–response analysis. Table 14 lists the absolute rates.

Figure items. Figure 1 shows fragments of IMP-PRES quantifiers.132 and connectives.1092 and of scalar-diversity vt16.0; its readings are glosses, not model output. Figure 3 is schematic.

## D Pre-registration and Analysis Details

The accounts and hypotheses, the outcome of every pre-registered prediction and the statistical procedures.

<table><tr><td>Hyp.</td><td>Prediction</td><td>Outcome (evidence)</td><td>Status</td></tr><tr><td></td><td>Pragmatic reading on generalized implicatures rises with thinking</td><td>Falls for some or or (CI excluding 0) in 30 of 40 open models, among them EXAONE-Deep-7.8B, Phi-4-mini-reasoning, every newest-generation model (Qwen3.5 0.8B–9B, Qwen3.8-27B), QwQ-32B, Nemotron-Super-49B, R1-Distill-Llama-70B, every Qwen3 size from 1.7B, gpt-oss-20b, OLMo-3-32B-Think, Phi-4-reasoning(-plus), R1-Distill-Llama-8B, MiMo, Nemotron-Nano, AceReason and</td><td>not supported</td></tr><tr><td>H1 default-and-cancel</td><td>Falls under the felicity framings, ≥5 pp in ≥2 of 4B/8B/14B</td><td>OLMo-3-7B-Think, SmolLM3 and openPangu-Embedded-7B-V1.1 (Table 12); GPT-5 falls in two formats (Section B.3) Holds for some/or per family; the pre-registered IMPPRES aggregate at 14B</td><td></td></tr><tr><td>H1b capability crossover</td><td>Negative at small sizes, positive at ≥32B on  $\geq 2$  families, reproduced in a same-recipe pair</td><td>criterion failed Qwen3-32B negative on some/or; OLMo-3-Think 7B null and 32B positive on IMPPRES but negative on some; no family  $\geq 2$  families</td><td>not supported</td></tr><tr><td>H2 split</td><td>PCIs improve while GCIs fall; ≥3 PCI sets improve or stay flat</td><td>flips sign within a pair on Accuracy rises on Fig-QA and PUB at every Qwen3 size from 4B to 32B (8B: +10.7 and +10.1 pp), on Circa and NOPE in 14 models with no decrease (Table 10), and on presupposition and LUDWIG in Qwen3-8B;</td><td>supported</td></tr><tr><td>H2b speaker knowledge</td><td>Default-and-cancel: thinking lowers R in both knowledge conditions;</td><td>the budget × class interaction is significant in Qwen3-4B, 8B and 14B (Section D.3) Knowledge sensitivity ≈0 in Qwen3 and OLMo; in gpt-oss-120b it shrinks with</td><td>default-and-cancel supported (one model)</td></tr><tr><td>H2c framing × class</td><td>effort/capability: thinking widens the gap Truth framing denies implicatures of both</td><td>effort (84.0 → 62.5 pp) the interpretation-truth arm was not run</td><td>not tested</td></tr></table>

Table 13: Outcome of every pre-registered prediction (held-out split). “Not tested” marks predictions whose arm was not run.
<table><tr><td></td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td><td colspan="3">Qwen3-14B</td></tr><tr><td>Phenomenon</td><td>NT</td><td>NAT</td><td>∆</td><td>NT</td><td>NAT</td><td> $\Delta$ </td><td></td><td></td><td></td></tr><tr><td colspan="10">Generalized implicatures: pragmatic reading R (%)↑</td></tr><tr><td>Category sentences, ternary</td><td> $5 . 0 \pm 0 . 0$ </td><td> $2 . 5 \pm 0 . 0$ </td><td>↓2.5</td><td> $2 . 5 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>↓2.5</td><td> $2 . 5 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>↓2.5</td></tr><tr><td>Scalar diversity</td><td> $1 6 . 6 \pm 1 . 1$ </td><td> $8 . 8 \pm 1 . 0$ </td><td>↓7.7</td><td> $4 . 9 \pm 0 . 4$ </td><td> $1 0 . 4 \pm { 1 . 0 }$ </td><td>↑5.5</td><td> $2 . 0 \pm 0 . 7$ </td><td> $8 . 8 \pm 1 . 4$ </td><td>↑6.8</td></tr><tr><td>IMPPRES scalar</td><td> $4 9 . 7 \pm 1 . 8$ </td><td> $2 3 . 2 \pm 1 . 3$ </td><td>↓26.5</td><td> $2 2 . 6 \pm { 1 . 8 }$ </td><td> $1 2 . 7 \pm 1 . 2$ </td><td>↓9.9</td><td> $1 4 . 5 \pm 1 . 5$ </td><td> $1 5 . 0 \pm 2 . 1$ </td><td>↑0.5</td></tr><tr><td colspan="10">Presupposition: projection (%) ↑</td></tr><tr><td>IMPPRES presupposition</td><td> $5 9 . 2 \pm 2 . 8$ </td><td></td><td></td><td>71.2±0.9↑12.049.5±4.5</td><td> $7 1 . 8 \pm 0 . 5$ </td><td>↑22.2</td><td> $5 4 . 1 \pm 0 . 0$ </td><td> $7 6 . 9 \pm 2 . 6$ </td><td>↑22.8</td></tr><tr><td colspan="10">Particularized implicatures: accuracy (%) ↑</td></tr><tr><td>LUDWIG indirect answers</td><td> $7 6 . 3 \pm 2 . 2$ </td><td>76.0±0.6↓0.4</td><td></td><td> $6 8 . 8 \pm 1 . 1$ </td><td> $7 5 . 3 \pm 2 . 2$ </td><td>↑6.5</td><td> $7 4 . 2 \pm 1 . 1$ </td><td> $7 5 . 6 \pm 0 . 6$ </td><td>↑1.4</td></tr><tr><td>Fig-QA metaphors</td><td> $7 6 . 9 \pm 1 . 2$ </td><td> $8 7 . 3 \pm 0 . 0$ </td><td>↑10.5</td><td> $7 8 . 5 \pm 0 . 6$ </td><td> $8 9 . 3 \pm 1 . 0$ </td><td>↑10.7</td><td> $8 4 . 9 \pm 1 . 1$ </td><td> $9 1 . 6 \pm 1 . 4$ </td><td>↑6.7</td></tr></table>

Table 14: Per-phenomenon results for Qwen3-4B, 8B and 14B. Reading chosen without (NT) and with natural thinking (NAT), mean ± s.d. over three seeds, held-out split; ∆ in $\mathrm { p p } ; - \mathrm { . }$ not run in this form.

## D.1 Accounts, Hypotheses and Operationalisation

This section states the theoretical accounts, the pre-registered hypotheses and how each is operationalised.

Generalized and particularized implicatures. Grice (1975) distinguished generalized conversational implicatures, which an expression carries unless the context blocks them (some conveys ‘not $\mathrm { a l l } ^ { \prime } )$ , from particularized conversational implicatures, which arise only from a specific context (“I wore gloves” in answer to “Did you leave fingerprints?” conveys ‘no’). We call them GCIs and PCIs. The two major post-Gricean theories disagree about GCIs. On the default view (Levinson, 2000), a GCI is generated automatically and cancelled only when the context contradicts it. On the relevance-theoretic view (Sperber and Wilson, 1986), every implicature is an inference drawn when it pays off in relevance, and drawing it costs processing effort. Both treat PCIs as context-driven inferences.

<table><tr><td>Hyp.</td><td>Prediction</td><td>Outcome (evidence)</td><td>Status</td></tr><tr><td>Scenario validity</td><td>Humans find some misleading only under full access (≥80%)</td><td>100.0% of 24 (blind consensus labels)</td><td>met</td></tr><tr><td>Open-answer coding</td><td>LLM reference rater agrees with the human audit (lower bound of κ ≥ 0.40)</td><td>best rater κ = 0.57 [0.34, 0.79]; open answers reported descriptively</td><td>not met</td></tr><tr><td>Bott-Noveck items</td><td>Category sentences move with thinking</td><td>Near the literal ceiling in every format and condition for Qwen3 4B-32B; only binary felicity at 14B moves</td><td>floor set</td></tr></table>

Table 13, continued (part 3 of 3).
<table><tr><td rowspan=1 colspan=4>Hyp.               Prediction                 Outcome (evidence)      Status</td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>With 2,048 words of</td><td rowspan=1 colspan=1>not</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>Qwen3-8B but from 14.5</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>to 20.9% in Qwen3-14B;</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>no consistent direction</td><td></td></tr><tr><td rowspan=1 colspan=2>H5 human ordering ρ ≥ 0.3 with human rates</td><td rowspan=1 colspan=1>not run</td><td></td></tr><tr><td rowspan=1 colspan=2>per scaleH6 restorable       IST restores &gt;50% of the</td><td rowspan=1 colspan=1>With 26-token seeds, the</td><td></td></tr><tr><td rowspan=1 colspan=2>shift; a length-matched</td><td rowspan=1 colspan=1>intent seed recovers 12.8%</td><td></td></tr><tr><td rowspan=1 colspan=2>neutral seed does not</td><td rowspan=1 colspan=1>of a 29.5% no-thinking</td><td></td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1>reading on some in</td><td></td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B (NAT: 0.0%);</td><td></td></tr><tr><td rowspan=1 colspan=1>neutral and logic seeds</td><td></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>recover little or none</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(Table 11)</td><td></td></tr><tr><td rowspan=1 colspan=1>H7 locus</td><td rowspan=1 colspan=1>The literal preference</td><td rowspan=1 colspan=1>Truncating the model&#x27;s</td><td rowspan=1 colspan=1>crossing supported; marker order not supported</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>appears inside the trace,</td><td rowspan=1 colspan=1>own trace: the pragmatic</td><td rowspan=10 colspan=1></td></tr><tr><td rowspan=9 colspan=2>after a marker</td><td rowspan=1 colspan=1>reading of some first rises</td></tr><tr><td rowspan=1 colspan=1>(Qwen3-8B, +23.1 pp after</td></tr><tr><td rowspan=1 colspan=1>128 tokens) and falls</td></tr><tr><td rowspan=1 colspan=1>below the no-thinking</td></tr><tr><td rowspan=1 colspan=1>level later in the trace</td></tr><tr><td rowspan=1 colspan=1>(Figure 4); literalness</td></tr><tr><td rowspan=1 colspan=1>markers such astechnically precede the</td></tr><tr><td rowspan=1 colspan=1>change in only 20.6% ofchanging Qwen3-8B</td></tr><tr><td rowspan=1 colspan=1>traces, and occur as oftenin traces that do notchange (39.1%)</td></tr></table>

Table 13, continued (part 2 of 3).

Adults pay for enrichment; children often do not make it. Adults who reject an underinformative sentence such as “Some elephants are mammals” take longer than adults who accept it, and a short response deadline yields fewer rejections (Bott and Noveck, 2004). A concurrent memory load does the same (De Neys and Schaeken, 2007; Dieussaert et al., 2011; Marty and Chemla, 2013). The finding is contested. Scalar inferences depend on context (Breheny et al., 2006; Degen, 2015) and can be immediate when the alternatives are salient (Grodner et al., 2010; Degen and Tanenhaus, 2015), although other studies find a delay (Huang and Snedeker, 2009). Development adds a second axis. Children accept underinformative some far more often than adults do (Noveck, 2001; Papafragou and Musolino, 2003). Yet the same children give such sentences the middle value on a three-point scale, so they notice the violation and tolerate it (Katsos and Bishop, 2011). On this view, enrichment grows with the listener’s resources and competence, and the “logician” is a developmental stage, not a default. For PCIs the human record differs again. Conventional indirect requests are understood without first computing their literal meaning (Gibbs, 1983), and salient figurative meanings can be accessed directly (Giora, 1997).

![](images/c16ad3f96ab6126a7fdc00ad2bd4ab18cec4c55b48d78d1a73a912ea6f66245d.jpg)  
Figure 9: Thinking makes some and or more literal in every major model family. Change with thinking (pp, 95% CI). Size trends: Figure 10.

Test-time thinking as a dose of deliberation. A reasoning model with a thinking toggle and a budget exposes what no human experiment can: the same weights answering the same item with more or less deliberation. We use default operationally, as the reading a model chooses at zero deliberation, and cancel for a change of that reading towards the literal one as deliberation is added; neither term presupposes a cognitive mechanism. We consider three profiles.

• On the effort account, enrichment is effortful, and more thinking yields more pragmatic readings on every phenomenon we test.

• On the default-and-cancel account, the model holds GCIs as defaults and spends deliberation on truth-conditional verification, a habit that training with verifiable rewards (RLVR) plausibly rewards. Thinking then cancels GCIs while it improves PCIs, whose pragmatic reading can only be reached by inference.

• On the capability account, deliberation amplifies whichever competence dominates the trace, so the sign of the effect on GCIs depends on the model’s capability, much as enrichment depends on age in children.

The theory question is therefore:

Do reasoning models pay the cost ofenrichment, or the cost ofcancelling it, and does that depend on capability?

Test-time thinking is a dose of deliberation on fixed weights. Ifpragmatic enrichment in language models is effortful, more thinking yields more enrichment on every phenomenon; if training with verifiable rewards installs a default-and-cancel profile, thinking cancels generalized implicatures while it improves particularized ones; ifdeliberation amplifies existing competence, the sign on generalized implicatures reverses with capability.

Operationalisation. The dependent variable is the reading chosen: R is the proportion of committed answers that take the pragmatic reading. For PCIs the pragmatic reading is the correct answer, so R is accuracy. The dose is the thinking budget B, from no thinking (NT) through capped budgets to natural stopping under an 8,192-token safety cap (NAT). Its log, $x = \log _ { 2 } ( B + 1 )$ , enters the statistical models. Under a true/false instruction, the literal answer to “Some elephants are mammals” is correct. A model that reads the question as a logic problem would therefore drift literal with thinking for reasons that have nothing to do with enrichment; we call this logic-mode compliance. The primary readouts are judgements of felicity rather than truth, and the true/false framing serves as the logic-mode comparison. Binary judgements also hide pragmatic tolerance. Category sentences are therefore judged on a ternary scale (completely right / a bit wrong / completely wrong; both lower logical words (some, or) rest (lexical scales, indirect answers) three-way choice (logical words)

![](images/415bf30070395a19056c4fd5fa2f39c7c3591eace07fd266d3b4dd89ec80662a.jpg)

![](images/9c66691f4cf0cc1e136bfe6acdb8a15f45cad29186f882b44ea6bd36988666c1.jpg)  
Figure 10: The literal shift holds across sizes and families. Thinking effect on some and or (pp, 95% CI) by model size for Qwen3, Qwen3.5 and R1-Distill-Qwen, with Qwen3.8, MiMo, openPangu and DeepSeek-R1-0528 as single points.

![](images/e8a8378fd82a9f67c757973720f879ee3dd8af6e077c11cd0063621729fa0f25.jpg)

![](images/300868302108f69b1342954461b98d058243c90de43aee1c63c8cbcd80363925.jpg)  
Thinking turns logical words literal (vermilion, left of 0) in most models; the rest (blue) moves far less. ◆ three-way choice: the sign can differ from the Yes/No format.  
whiskers: 95% CI. Gemini 2.5 Flash omitted (answer failures >10%). † Yes/No format only. ‡ baseline = 512-token cap (no valid no-think arm).

Figure 11: Thinking reduces enrichment of some, or or both in most tested models; other inference types move differently. Change in pragmatic reading with thinking (pp, 95% CI): vermilion, mean of some and or; blue, mean of lexical scales and indirect answers; ♦, three-way choice. Every family: Figure 8. Q3.5-35B: Qwen3.5-35B-A3B.

<table><tr><td>Prediction</td><td>Measurement on LMs</td><td>Hyp.</td></tr><tr><td>Effort ordering</td><td>R on GCIs vs. x = log₂(B + 1) under felicity framings (inference; ternary and 1–7 for category sentences); true/false as the</td><td>H1</td></tr><tr><td>GCI/PCI split</td><td>logic-mode contrast budget × class interaction on R</td><td>H2</td></tr><tr><td>Speaker</td><td>(PCI: accuracy, literal-error share) ∆R (full − partial knowledge) at NT H2b</td><td></td></tr><tr><td>knowledge Framing ×</td><td>VS. NAT PCI accuracy under meaning vs.</td><td>H2c</td></tr><tr><td>class Training origin</td><td>interpretation-truth framing, NT vs. NAT default R by stage (log-prob</td><td>H3</td></tr><tr><td>Load</td><td>scoring); cost of thinking by stage; ρ(RL step, literal rate) R under a memory load, crossed</td><td>H4</td></tr><tr><td>Human</td><td>with thinking Spearman ρ with human rates per</td><td>H5</td></tr><tr><td>ordering Restorability</td><td>scale and item IST vs. token-matched neutral and</td><td></td></tr><tr><td></td><td>logic seeds; PCI accuracy</td><td>H6</td></tr><tr><td>Locus</td><td>answer-position log-odds over trace prefixes; marker onset</td><td>H7</td></tr></table>

Table 15: From theory to measurement. Each prediction that separates the effort and default-and-cancel accounts, the quantity we measure, and the hypothesis it tests. R: proportion of committed answers that take the pragmatic reading; B: thinking budget in tokens.

values count as pragmatic) and on a 1–7 scale (Katsos and Bishop, 2011; Degen, 2015), with binary felicity and truth judgements as comparisons. Binary judgements of category sentences stay near the literal ceiling (96.7% literal for Qwen3-8B without thinking), whereas a majority of adults reject such sentences (54–100% across tasks; De Neys and Schaeken, 2007; Marty and Chemla, 2013; Katsos and Bishop, 2011). Table 15 maps each prediction to its measurement.

Hypotheses. Hypotheses, thresholds, falsifiers and the rule that decides between the three profiles were fixed before the main runs (Section D.3).

H1 (Direction on GCIs). Under the felicity framings, the effort account predicts that R on GCIs rises with x; the default-and-cancel account predicts that it falls. A direction counts if |∆R| between NT and NAT is at least 5 pp with nonoverlapping 95% confidence intervals, with the same sign under capped and length-stratified budgets and under both readouts. Default-and-cancel predicts the shift under both framings, whereas logic-mode compliance predicts it under true/false only. Falsified for both accounts if ∆R is equivalent to zero within ±3 pp at every size. (H1b, capability) The capability account predicts that ∆R is negative for the smallest models and positive for at least one model of 32B or more, that its sign is ordered by a capability measure rather than by training recipe, and that the flip reproduces within a same-recipe size pair.

H2 (Split). Thinking raises accuracy and lowers the literal-error share on PCIs while it lowers R on GCIs, a budget × class interaction. The claim is about the class average, with a random effect for item set. It holds if the interaction is significant and at least three PCI sets improve or stay flat under natural thinking; a drop confined to a capped budget counts as a truncation cost. Falsified if GCIs and PCIs move in the same direction, or if PCI accuracy drops under natural thinking on at least half of the sets. Two corollaries follow. (H2b) A pragmatic listener enriches more when the speaker has full knowledge (Goodman and Stuhlmüller, 2013). Default-and-cancel predicts that thinking lowers R in both knowledge conditions; the effort and capability accounts predict that thinking widens the gap between them. (H2c) Asking whether an implicated interpretation is true should make thinking deny it for GCIs and PCIs alike, as in the one prior same-weights contrast (Spinoso-Di Piano et al., 2026, Tables 6–7). Asking what the speaker meant should instead let PCIs improve. Falsified if PCI accuracy under the meaning framing also drops, or if the truth-framing drop is absent for PCIs.

H3 (Training origin). The literal shift is installed by RLVR. We test two estimands. H3-default: the default reading, scored without deliberation, turns literal at the RLVR stage and, in zero-RL runs, with RL steps. H3-cost: the within-stage cost of thinking appears or grows at the RLVR stage. Falsified if the change appears at SFT, if at most two of five RL-Zero series trend, or if distilled thinkers show the same cost as RL-trained ones.

H4 (Load). A memory load lowers R on GCIs without thinking, as in humans. The load × thinking interaction shows whether thinking acts as a resource or as the canceller. Falsified if load moves R by less than 3 pp at both thinking levels.

H5 (Human ordering). At the smallest budget, literal rates per scale correlate with human rates (Spearman $\rho \geq 0 . 3 ,$ , with a CI excluding 0).

H6 (Restorable). One fixed sentence about speaker intent, placed inside the thinking channel (Intent-

Seeded Thinking, IST), restores at least half of a negative GCI shift, with PCI accuracy non-inferior within 1 pp. Falsified if a token-matched neutral sentence restores as much.

H7 (Locus). Read at successive prefixes of a trace, the answer-position preference starts near its nothinking value and crosses to the literal reading inside the trace, after literalness markers such as technically and strictly speaking.

## D.2 Pre-registered Predictions and Outcomes

This section reports the outcome of every preregistered prediction.

Hypotheses, thresholds and falsifiers were fixed before the main runs (Section D.1). Table 13 lists the outcome of each prediction on the held-out split. Several did not hold; we report them in full.

Pre-registered remedies. We also pre-registered four remedies (a sentence at the start of thinking, a check before thinking ends, and fine-tuning and preference optimisation on the model’s own correct traces); all four were run, and none passed its pre-declared development gates (pre-registration file METHOD\_DESIGN.md at https://anonymous. 4open.science/r/entertain-then-cancel/).

## D.3 Statistical Procedures

This section lists the run settings and statistical procedures.

Table 16 lists, for every experiment block, the number of models, items per model, decoding seeds, temperature and thinking cap; NAT generations stop at the model’s close-of-thinking delimiter or at the cap, and capped traces are forced closed.

Hypotheses, thresholds, falsifiers and exclusion rules were fixed before the main runs and are recorded with dates in the project log (anonymised at https://anonymous.4open. science/r/entertain-then-cancel/).

• Budget × class model. For each Qwen3 model we fit a logistic model of the pragmatic reading on $\log _ { 2 } ( B + 1 )$ of the realised thinking tokens, the class (generalized vs. particularized or presupposition) and their interaction. The plan specified by-item random intercepts and slopes; we fit the same fixed effects with item-clustered robust errors instead (no mixedmodel solver was available on our stack), a population-averaged estimate. The interaction is −0.067, −0.063, −0.049 and −0.037 for

<table><tr><td>Block</td><td>Models</td><td>Items</td><td>Seeds</td><td>T</td><td>Cap</td></tr><tr><td>E1</td><td>3</td><td>3,877</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1S</td><td>13</td><td>2,052</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1SF</td><td>5</td><td>3,600</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1SFX</td><td>17</td><td>3,600</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1Q</td><td>6</td><td>2,052</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1P</td><td>7</td><td>2,052</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1B</td><td>14</td><td>2,052</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1PANGU</td><td>3</td><td>5,972</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1R1</td><td>1</td><td>2,052</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E3</td><td>2</td><td>2,052</td><td>3</td><td>API</td><td>一</td></tr><tr><td>E1GSF</td><td>1</td><td>3,600</td><td>3</td><td>API</td><td>一</td></tr><tr><td>E1PROBE</td><td>6</td><td>320</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1DS</td><td>17</td><td>1,092</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E1XL</td><td>8</td><td>476</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E10</td><td>7</td><td>1,492</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E7</td><td>15</td><td>687</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E7F</td><td>11</td><td>1,800</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E2</td><td>44</td><td>2,052</td><td>3</td><td>API</td><td></td></tr><tr><td>E2T</td><td>8</td><td>1,492</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E2C</td><td>3</td><td>1,492</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E6</td><td>4</td><td>1,067</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E8</td><td>6</td><td>1,492</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>A1CTX</td><td>6</td><td>750</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>A6</td><td>5</td><td>813</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>E9</td><td>6</td><td>600</td><td>3</td><td>0.6</td><td>8192</td></tr><tr><td>M1DEV</td><td>3</td><td>3,450</td><td>1</td><td>0.6</td><td>8192</td></tr></table>

Table 16: Run settings per experiment block: models, items per model, decoding seeds, temperature and thinking cap (tokens).

4B, 8B, 14B and 32B $( p < . 0 0 1 , < . 0 0 1 , < . 0 0 1$ .230).

• Confidence intervals. Cluster bootstrap over items, stratified by set, with 10,000 resamples. Differences between conditions are computed inside each resample, so they are paired.

• Equivalence. Two one-sided tests at ±3 pp on the change in lexical scales and LUDWIG. Equivalence holds for lexical scales in 2 of 42 models; LUDWIG, with 95 items, is too small to establish equivalence at this margin.

• Multiple comparisons. Holm correction of the paired-bootstrap p-values for the some and or shifts within each model family. On the full held-out sets 47 of 48 shifts survive; on the core sets of 26–33 items, fewer do, as expected from their size. Secondary analyses are uncorrected and labelled as such.

• Exclusions. Hedged answers are excluded from R and reported, and the extraction-free readout L keeps them.

• Power. Item counts per cell are given with every result; with 26–33 core items per word, the item-level intervals are wide, which is why the headline numbers are replicated on the full held-out sets of 300 items per word.

## E Related Work

The extended discussion of related work.

## E.1 Related Work

This section reviews related work on implicature, reasoning models and test-time compute.

Test-time reasoning and social understanding. Several studies find that reasoning does not transfer to social cognition. Reasoning variants of Qwen3 and DeepSeek do no better than their non-reasoning counterparts on theory-of-mind benchmarks, and o3’s accuracy on higher-order belief questions falls as reasoning effort rises (Gong et al., 2026). Rea soning models also trail older models on theory of-mind variants of cognitive-science experiments (Lupu et al., 2025), and the gains they do show are better described as robustness to prompt varia tion than as new ability (de Haan et al., 2026b,a). For presupposition, models with stronger reason ing give less human-like projection judgements (Azin et al., 2026), and reasoning adds only 2– 11 points on queries with false presuppositions (Sathyanathan et al., 2026). On implicature the record is mixed. The closest observation is inci dental: in ImplicatureX, a benchmark of implica ture recognition and cancellation, Qwen3 in think ing mode at a 512-token budget recognises fewer scalar implicatures than the same weights without thinking, and recognises cancellations more often (Spinoso-Di Piano et al., 2026). Chain-of-thought (CoT) prompts built on Grice’s maxims, on rele vance theory or on the Rational Speech Act model improve implicature resolution and number-word interpretation (Kim et al., 2023; Sato et al., 2025; Tsvilodub et al., 2025); CoT lifts GPT-4 to hu man average on binary implicatures (Ruis et al., 2023); and training on counterfactual reasoning traces helps further (Park et al., 2026; Sravanthi et al., 2025). Yet CoT was also observed to “encourage literal interpretations” in Korean (Park et al., 2024). Unlike ImplicatureX, which observes the drop at a single budget on 46 items without inter vals and does not treat it as a thinking effect, and unlike Gong et al. (2026), who study the same manipulations on theory-of-mind accuracy, we treat the thinking budget as a controlled dose, measure the reading chosen under felicity and truth fram ings, show that the effect dissociates by implicature type, trace it to an entertain-then-cancel trajectory inside the trace and to specific post-training stages (thinking SFT in OLMo-3, preference tuning in Tülu-3), and test four pre-registered remedies under matched controls.

Pragmatics in language models. Benchmarks cover implicature, indirect speech, irony, metaphor, humour, deceit and presupposition (Hu et al., 2023a; Ruis et al., 2023; Sravanthi et al., 2024; Jeretic et al., 2020; Liu et al., 2022); see Ma et al. (2025) for a survey. When models err on these items, their errors favour the literal reading over distractors (Hu et al., 2023a). Work on scalar implicature links model expectations over alternatives to human inference rates (Schuster et al., 2020; Hu et al., 2023b), probes scalar diversity (Lin et al., 2024), and steers graded interpretations through activations (Cho, 2026). Thinking and instruct models have also been compared on quantifier production, across models (Montero et al., 2025). Pragmatic competence rises from base to SFT and preference-tuned checkpoints (Yu et al., 2026). This is why comparisons across training stages confound deliberation with post-training. Unlike these works, we treat the inference budget as the independent variable, hold the weights fixed for the primary contrast, and extend the stage analysis from supervised fine-tuning through preference tuning to RLVR.

Overthinking and side effects of reasoning training. Longer reasoning can lower accuracy on easy problems, on tasks with distractors and on deduction (Chen et al., 2024; Gema et al., 2025; Ghosal et al., 2025; Lee et al., 2026; Sui et al., 2025). Explicit reasoning also degrades instruction following (Li et al., 2025), and CoT hurts where deliberation hurts humans (Liu et al., 2024), while its benefits concentrate on math and symbolic tasks (Sprague et al., 2025). Reinforcement fine-tuning has side effects of its own: it lowers refusals on unanswerable questions (Song et al., 2025), can raise hallucination (Yao et al., 2025), and may sharpen what the base model already does rather than extend it (Yue et al., 2025). Budget forcing (Muennighoff et al., 2025), hybrid thinking modes (Yang et al., 2025) and effort levels (OpenAI et al., 2025) make the amount of reasoning controllable; recent work also asks how reasoning changes social behaviours such as sycophancy (Feng et al., 2026). These works define harm as lower accuracy on tasks with one correct answer. Our outcome is a choice between two readings that are both licensed by the sentence. The direction of the shift, not its size, is what distinguishes the two theories.

The processing cost of enrichment in humans. Human evidence that scalar enrichment is effortful comes from response times, deadlines and dualtask load (Bott and Noveck, 2004; De Neys and Schaeken, 2007; Dieussaert et al., 2011; Marty and Chemla, 2013; Tomlinson et al., 2013), and from development (Noveck, 2001). Counter-evidence shows that the cost depends on context and on the availability of alternatives (Breheny et al., 2006; Grodner et al., 2010; Degen and Tanenhaus, 2015). Enrichment rates also vary widely across scales (van Tiel et al., 2016; Gotzner et al., 2018; Pankratz and van Tiel, 2021), and probabilistic models treat enrichment as inference about a cooperative speaker (Frank and Goodman, 2012; Goodman and Frank, 2016; Goodman and Stuhlmüller, 2013). We import the manipulation from this literature (a controlled change in deliberation), not its conclusion. Section D.1 derives what each account predicts for a model.