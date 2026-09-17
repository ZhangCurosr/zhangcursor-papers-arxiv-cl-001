# Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection

Navyansh Singh IIIT Naya Raipur navyansh24102@iiitnr.edu.in

Animesh Pathak IIIT Naya Raipur animesh24100@iiitnr.edu.in

Aarav Singh IIIT Naya Raipur aarav24101@iiitnr.edu.in

## Abstract

Fallacy-detection benchmarks pair fallacy classes with a single “valid” or “none” class that takes everything data collection did not label as a fallacy. This construction is misleading: a classifier can learn cues that do well on this class without learning to tell a fallacy from a correct argument. We show that the low false-positive rates benchmarks report are an artifact of how the class is built, not evidence of detection ability. The most informative negative for a fallacy is a correct argument using the same argumentation scheme, and such ar guments are at most a few percent of the valid class across the four benchmarks we examined. Evaluated on constructed scheme-matched negatives, false-positive rates rise from 16.6% to 58.9% on CoCoLoFa and from 5.7% to 62.0% on Reddit. That rate depends on how the negatives are written, so we also compare two conditions from the same pipeline that differ only in scheme identity. Classifiers label scheme matched negatives as the source fallacy type 40.9 points more often than wrong-scheme negatives, which are instead identified as the scheme they actually use 85.9% of the time against 0.4% for the source type. The classifier has learned which scheme an argument uses, not whether it uses it correctly—and on the benchmarks’ own test sets the two are indistinguishable. The same dissociation appears in three zero-shot LLM detectors that never saw these benchmarks, and the measurement is far lower on a negative class that was built deliber ately. We release the items as Scheme Foils. A reported false-positive rate should not be trusted as a measure of detection until the valid class has been audited for scheme-matched cov erage.

## 1 Introduction

Fallacy-detection benchmarks are classification datasets. A span of text is assigned to one of several fallacy types, or to a “valid” or “none” class holding everything not judged fallacious (Yeh et al., 2024; Helwe et al., 2024; Habernal et al., 2017). A fallacy detector has two jobs: detection, separating fallacious arguments from valid ones, and classification, naming the type of fallacy an argument commits. What decides whether a detector is usable is detection—flagging the arguments that contain a fallacy and passing the ones that do not—and its measure on sound reasoning is the false-positive rate. On standard benchmarks that rate is small, and is reported as evidence that over-flagging is minor (Yeh et al., 2024; Singh, 2026).

We show that the number is small because of how the valid class is assembled. It is not built the way the fallacy classes are built. It is the residue: whatever crowd workers produced when asked not to write a fallacy, whatever a forum scrape left over, whatever an annotation protocol swept into “none.” We sampled the valid class of four benchmarks and found almost none of it scheme-matched—using the same reasoning pattern as a fallacy without committing it. The classes hold off-topic remarks, bare opinions, fragments, comments about fallacies, and genuine arguments, most built on reasoning patterns no fallacy class corresponds to. Separating that material from a fallacy takes no ability to tell valid reasoning from fallacious reasoning— only recognition of which pattern an argument uses. What is missing is the scheme-matched valid argument, and two independent annotators found 3 such items in 183 sampled.

That population is the one detection should be measured on. A valid appeal to expert opinion sits close to a fallacious one; what separates them is whether the cited authority is genuinely qualified, not the argument’s surface shape. Walton’s argumentation schemes (Walton et al., 2008) make this operational (§2) and give each fallacy type its matched negative: a valid argument answering the question the fallacy leaves open.

To measure detection on the population where the two abilities come apart, we construct matched negatives for eight fallacy types across two corpora. A fine-tuned classifier that flags 16.6% of CoCoLoFa’s native valid class flags 58.9% of them; trained from scratch on Reddit, 5.7% becomes 62.0%.

A rate measured on constructed items depends on how the items were written, so for each source item the same pipeline also produces a wrongscheme negative: an equally careful valid argument on the same topic, at the same length and register, built on a different scheme. The conditions differ in scheme identity and nothing else—any signature of machine generation is present in both and cannot produce a difference, and “carefully written is simply harder” predicts no difference either. We compare them on how often each is classified as the source type specifically.

The two conditions come apart sharply. Matched negatives are classified as their source type 40.9 points more often than wrong-scheme negatives; wrong-scheme negatives are instead classified as the scheme they do use 85.9% of the time, and as the source type 0.4% of the time. The raw rates establish that a gap exists; the dissociation establishes what it is made of.

The dissociation is robust. Annotators blind to classifier behaviour confirm the matched negatives valid, and the gap survives on the confirmed subset. It grows when the domain gap is removed by training from scratch—the opposite of what domain shift predicts. It appears in three zero-shot detectors that never saw the benchmark. And the over-flagging does not need our pipeline at all: it appears on naturally occurring arguments no one wrote for this study, and, attenuated, on a negative class an independent shared task built deliberately around this population.

Detectors track scheme identity and not criticalquestion failure, and a valid class containing no correctly-instantiated schemes contains no item on which those two come apart. The benchmark therefore scores scheme recognition—the classificationshaped skill—and reports it as detection. Its falsepositive rate is low for a reason that has nothing to do with the detector being right.

Our contributions are:

1. A design that isolates scheme content. Two constructed conditions from one pipeline, identical except for the scheme they instantiate, with a type-specific statistic fixed in advance by what the competing accounts predict. The comparison is invariant to generation quality, which no raw rate on constructed data can be.

2. The measurement gap. Scheme-matched valid arguments make up at most a few percent of the valid class in four benchmarks, and classifiers flag constructed instances at 58.9% and 62.0% where the same benchmarks report 16.6% and 5.7%.

3. An audit. A five-step procedure that measures how far a benchmark’s reported false-positive rate understates the rate on the population that matters, using an existing benchmark and an already-trained classifier, with no retraining. It discriminates between construction protocols (§6.5).

4. Scheme Foils, a released challenge set: 1,042 scheme-matched and 1,127 wrong-scheme valid arguments across two corpora, each carrying a human-validated precision estimate.

## 2 Scheme-Matched Negatives and the Valid Class

Schemes and critical questions. We take our argumentation schemes from Walton (Walton, 1996; Walton et al., 2008). A scheme is a reusable pattern of everyday inference, and each scheme carries critical questions: the checks an argument of that pattern must survive. An argument that answers its scheme’s critical questions is a valid instance; one that fails a specific question, in the characteristic way, is a fallacy of type τ. For expert opinion the questions are whether the person cited is a genuine expert, in the relevant domain, speaking within their competence. A climate argument citing a named climatologist answers them and is valid. The same argument citing a talk-show host fails the first question—the fallacy called “appeal to authority.” A negative example matched to τ is exactly this valid counterpart: an argument on the same scheme that answers the question the fallacy leaves open, comparable in topic, length, and register. Throughout, valid means what the benchmarks mean by it (the “valid” or “none” class), not deductive validity; we call that class the benchmark’s native class when contrasting it with constructed negatives. We use Walton instrumentally, in the operational form of Ruiz-Dolz and Lawrence (2023).

A classifier that flags a matched negative is tracking the scheme’s surface signature, not the criticalquestion failure its label denotes. The two are separable only on items where a scheme is instantiated correctly, so a valid class without them leaves the substitution invisible. Walton’s account predicts these pairs are hard: a fallacy characteristically appears a better argument of its kind than it is, because it borrows the form of a legitimate scheme and fails only at the step a surface reading does not expose (Walton, 2010).

Why the valid class fills up with easy material. Each fallacy type is tied to one scheme and one critical-question failure, but validity is tied to no scheme in particular: every correct instantiation of every scheme is valid, and Walton’s catalogue alone enumerates more than sixty. “Valid” is therefore a far broader category than any single fallacy type, and a class assembled without scheme-level targeting has an abundant easy region to default into. That region is not hypothetical: Feng and Hirst (2011) classify Walton schemes from surface features, and Lawrence and Reed (2016) independently recover scheme identity from real text. A false-positive rate reported over such a class is a weighted average dominated by the easy majority, so building the valid class without scheme-level targeting does not remove the asymmetry between valid and fallacious arguments. It removes it from the measurement.

How often a deployed detector meets such an argument is beside the point: the reported rate predicts nothing about what happens when it does, and a system built to flag fallacies in argumentation cannot avoid correct arguments.

## 3 Related Work

Fallacy benchmarks. Nearly all fallacydetection benchmarks share one shape: fallacy classes plus a single undifferentiated negative class. CoCoLoFa (Yeh et al., 2024), MAFALDA (Helwe et al., 2024), Argotario (Habernal et al., 2017) and Sahai et al. (2021) follow it; others omit the negative class altogether (Jin et al., 2022; Goffredo et al., 2022), and a benchmark with no valid class cannot measure over-flagging at all. None reports how deliberately its negative class was constructed relative to the schemes its fallacy types instantiate. CoCoLoFa’s authors document the gap themselves. Some comment writers, asked to produce a fallacy, instead produced valid scheme instances that experts judged “fallacy-like but valid.” The authors note the critical-question-based validity check

Ruiz-Dolz and Lawrence (2023) had suggested, and judge forgoing it a reasonable trade-off given its cost and the low rate they observed, 12 of 237 comments (Yeh et al., 2024).

Dataset artifacts. A benchmark can report high scores for reasons that lie in how its data was collected rather than in what models understand: Gururangan et al. (2018) recover NLI labels from the hypothesis alone, McCoy et al. (2019) show inference models applying syntactic heuristics, and Niven and Kao (2019) account for a model’s argumentcomprehension score entirely by spurious cues. In each case the contribution was the diagnosis and the diagnostic set; none corrected the benchmark it was about. Those artifacts live in the positive class or in the association between the two; ours lives in the composition of the negative class. In measurement terms (Jacobs and Wallach, 2021), the operationalization—score over the shipped valid class—no longer measures the conceptualization the label names.

Hard negatives for evaluation. Gardner et al. (2020) show that naturally collected test sets overstate competence and that deliberately constructed hard examples expose the gap, but supply no criterion for which negatives are hard; for argumentation that criterion is not read off the surface, and Walton’s apparatus supplies it. In the fallacy domain, Singh (2026) builds counterfactual negatives for fallacy detection and finds the resulting bias real and persistent, without reference to argumentation schemes and without an account of why standard benchmarks fail to surface it. Minimal editing (Kaushik et al., 2020) is the alternative construction we set aside for this measurement, not in general (§4).

Closest prior work. Ruiz-Dolz and Lawrence (2023) test already-trained classifiers on fourteen hand-built arguments—one valid and one fallacious instance of each of seven schemes—and find that fine-tuned and generative models alike struggle to separate them. They attribute specific failures to surface-vocabulary matching, the same mechanism we formalize. Their account treats the sharing as a per-instance modelling deficiency to be corrected architecturally; we show the benchmark’s aggregate reported rate is deflated by how the valid class is built. That is a measurement problem, invisible in the reported number however the model is designed. Their remedy (Ruiz-Dolz and Lawrence,

2025) replaces the sequence classifier with a twostage pipeline requiring annotated data and two models at inference; ours changes no architecture and corrects what the negative class contains.

Concurrent work. The Touché 2026 shared task on fallacy detection (Kiesel et al., 2026) builds its non-fallacious class by selecting arguments whose reasoning pattern resembles a specific fallacy type, so that systems must separate genuinely fallacious arguments from superficially similar valid ones. The convergence is evidence that the population we identify is the one practitioners now treat as decisive. Their negatives are obtained by selection: items are not counterparts of particular fallacious sources, so topic, length and register are not held fixed and no wrong-scheme comparison is available. Their scheme dimension follows Macagno’s goal/basis axes (Macagno, 2022), so resemblance is judged at selection time, not by which critical question an argument answers. We audit their negative class in §6.5.

## 4 Constructing Scheme-Matched Negatives

Fresh writing, not minimal editing. We do not build negatives by minimal editing (Kaushik et al., 2020). A minimal edit turns a fallacious item into a valid one while keeping most of its wording, so it holds two things fixed at once: the lexical surface and the scheme. A flag on such a negative cannot be attributed cleanly—the classifier may be reading the scheme, or just the leftover words of a known fallacy. And the wrong-scheme condition below could not be built by minimal edits at all: changing the scheme means rewriting the reasoning. We therefore write every negative from scratch—a new valid argument on the same topic, in the same register and length range, using the required scheme without reusing the source’s wording. The matched negative then shares its source’s scheme, not its sentences.

The wrong-scheme condition. For each fallacious source item of type τ we generate two negatives. The matched negative uses τ’s scheme and answers its critical question. The wrong-scheme negative is a valid argument on the same topic, at the same length and register, built with the same effort, using a different scheme τ<sup>′</sup>, assigned per item and balanced across the other seven types under a fixed seed. The two conditions are built identically

except for the scheme.

We compare the conditions on one statistic: how often a negative is classified as the source type τ specifically. Generic difficulty predicts more flags of every kind; only scheme-tracking predicts flags of that one type. Both predictions are fixed before any measurement.

A false-positive rate on generated negatives can be moved: change the prompt, the model or the filter and the number changes. The difference between the two conditions cannot, because all three are shared—whatever the generator contributes, it contributes to both. Hand-built pairs (Ruiz-Dolz and Lawrence, 2023), minimal edits (Kaushik et al., 2020; Singh, 2026) and selected hard negatives all vary the item itself; ours varies only the scheme. Fresh writing removes shared wording as an explanation for a flag; the wrong-scheme control removes topic and generic difficulty; what remains to explain a difference between the conditions is scheme content.

Generation and filtering. Generation uses Gemini 2.5 Flash with one prompt design per fallacy type, stating the target scheme and its critical question explicitly (prompts in Appendix C). Automated checks on register, readability and lexical overlap are diagnostic, not drop criteria. No item was removed for register mismatch; when §6.3 finds no independent register effect, that is the data as generated, not the filter. Items are then judged by a cross-model judge, DeepSeek, chosen for a different architecture and training lineage from the generator; position bias is controlled by running each pairwise comparison in both orders. Matched negatives are judged on scheme fidelity, criticalquestion satisfaction and conclusion preservation; wrong-scheme negatives on register, on-topicality and scheme switching only, so a retained wrongscheme item is certified scheme-switched and ontopic, not certified a valid instance of τ<sup>′</sup> (§6.8). On CoCoLoFa the judge dropped 128 matched items (124 scheme drift, 4 conclusion flips) and 58 wrongscheme items, leaving 655 and 738; Reddit and Argotario yielded 387/389 and 219/246 under the same pipeline.

The audit. The procedure generalizes to any benchmark. Steps 1, 2 and 5 require only an existing benchmark, a coding of its negative class, and a classifier already trained on it; steps 3 and 4 are the remedy.

1. Map eachfallacy type to its scheme and critical question (Walton et al., 2008); without such a target, “matched negative” is undefined. For CoCoLoFa’s eight types the mapping is direct for five; false dilemma, appeal to nature and appeal to worse problems sit less cleanly in Walton’s catalogue.

2. Audit the negative classfor scheme-matched coverage. Sample it and code each item as a genuine instance of some tracked scheme or not (§6.1). If such arguments are rare, the benchmark is not testing the hard case.

3. Construct matched negatives where coverage is thin, holding topic, length and register comparable. For the eight types studied here this step can be skipped by scoring against Scheme Foils directly.

4. Validate construction, not fluency: check scheme fidelity and conclusion preservation with automated checks plus a judge from a different model lineage than the generator.

5. Measure two quantities: the false-positive rate over the native valid class against the rate over matched negatives, and the wrong-scheme control—how often each condition is classified as τ specifically.

## 5 Experimental Setup

Corpora. CoCoLoFa (Yeh et al., 2024) is the primary corpus and the hardest case for our claim, since its “none” negatives look most genuinely argumentative on inspection. Reddit comment threads (Sahai et al., 2021) are the second withincorpus site, and the one that rules out domain shift because we can train on it from scratch. Argotario (Habernal et al., 2017) and MAFALDA (Helwe et al., 2024) enter the negative-class characterization of §6.1 and provide corroborating checks (Table 5, Appendix D).

Classifiers. The primary classifier is ModernBERT-base (Warner et al., 2025) fine-tuned on CoCoLoFa’s training split as a nine-way classifier over the eight fallacy types plus “none.” A single CoCoLoFa-trained model across corpora isolates negative-class construction as the variable of interest. To rule out domain shift we separately train from scratch on Reddit and evaluate within that corpus; to check architecture-independence we fine-tune RoBERTa-base (Liu et al., 2019) under the same protocol (five seeds; three each for Reddit and RoBERTa). Every checkpoint was selected by held-out dev accuracy, never by any false-positive-related metric: selecting downstream of the central quantity would make the comparison circular. All models sit well above their majority baselines (Appendix A).

## 6 Results

Four experiments test the same claim, each on different data. Only one uses constructed data: the matched/wrong-scheme comparison (§6.3, repeated across detectors in §6.4). The others use what the native valid classes contain (§6.1), where the unmodified benchmark’s errors fall (§6.2), and arguments no one wrote for this study (§6.6). The audit also runs on a negative class someone else built (§6.5), and two studies validate the constructed items (§6.7, §6.8).

## 6.1 The valid class is mostly unmatched material

Two independent annotators coded 183 native “none” items—samples of 30, 30 and 60 from Co-CoLoFa, Reddit and Argotario, and all 63 items of MAFALDA’s extractable none class—as a schemematched valid argument (A), a genuine argument instantiating no tracked scheme (B), or not an argument (Table 3). They agreed on 86.3% of items (three-way κ = 0.74, Gwet’s AC1 0.82; positive specific agreement on A, 0.50) and resolved disagreements by discussion without the authors (protocol in Appendix B). Category A is 1 of 30 Co-CoLoFa items, 1 of 30 Reddit, 1 of 63 MAFALDA and 0 of 60 Argotario: 3 of 183 overall—rare in every negative class, whatever the protocol. We give raw counts because the samples are small; “3%” is one item.

Non-argument rate does not track the native false-positive rate. It runs 3% on CoCoLoFa, 40% on Reddit, 52% on MAFALDA and 65% on Argotario, and the corpus with the least, CoCoLoFa, has the highest native rate. CoCoLoFa’s class is instead almost entirely genuine argument (93% category B, one non-argument in 30), consistent with its role as the hardest case (§5). MAFALDA’s low native rate has a different source: its taxonomy is partly disjoint, so almost none of its valid class instantiates a scheme inside CoCoLoFa’s label space.

## 6.2 Errors already concentrate on “none”

The unmodified benchmark already shows the predicted signature. When the fine-tuned ModernBERT classifier misclassifies a fallacious Co-CoLoFa example, 76.9% of those errors land on “none” rather than on another fallacy type (95% CI [64.8%, 86.7%]). A TF-IDF baseline sends 97.1%, a weak-classifier floor. A permutation test preserving the model’s rate of predicting “none” gives $p < 0 . 0 0 1$ for both. This uses no constructed data: on the benchmark as published, the failing job is detection, not classification.

## 6.3 Scheme recognition scored as fallacy detection

Table 1 reports false-positive rates on the native valid class and on matched negatives, across four configurations. Matched negatives are over-flagged far above the native rate in every one; seed-to-seed variation is small relative to the gaps; the effect is not architecture-specific.<sup>1</sup> The wrong-scheme column runs higher still (93.5% on CoCoLoFa): those are flag rates over a condition certified valid at 77.5% (§6.8), and the informative quantity is not that level but which type the flags assign.

The effect is specific to the scheme. Averaged over the eight CoCoLoFa types, a matched negative is classified as its source type $4 0 . 9 \pm 1 . 9$ points more often than an equally careful wrongscheme negative. For seven of the eight the wrongscheme rate is zero; per-type gaps run from +10.7 to +73.1pp, and no per-type matched-rate interval overlaps its wrong-scheme interval. A wrongscheme negative built against τ is itself a correct instance of some other scheme τ<sup>′</sup>, and the classifier finds it. The classifier labels them with the scheme they instantiate 85.9% of the time, and with the source type 0.4% of the time—a more than 200- to-1 preference. Per type, $\tau ^ { \prime }$ ranges from 76.3% for false dilemma to 94.0% for appeal to nature, and τ never exceeds 2.2%. Because the conditions differ only in scheme identity, this is a claim about what the classifier tracks, not about how hard the items are: it has learned which scheme an argument uses, not whether it uses it correctly. On the benchmark’s own test set these two abilities are indistinguishable: every item instantiating a tracked scheme is labelled with a fallacy, so a model that recognizes the scheme scores as a model that detects the fallacy. What the benchmark reports as fallacy detection is scheme recognition, and it contains no items on which the two would come apart. An item-level logistic regression over the 1,393 constructed items finds condition dominant and readability with no independent effect $( p = 0 . 8 6 ) $ with only three of 738 wrong-scheme items classified as the source type, the Firth-penalized odds ratio is 144 (penalized likelihood-ratio $p < 0 . 0 0 1 $ . Applied to a deliberately built negative class the same measurement yields a far lower rate (§6.5), so this is a property of the class, not of the procedure.

Domain shift does not explain it. A CoCoLoFatrained model evaluated on Reddit invites a domainshift reading, so we removed the domain gap by training on Reddit from scratch. Domain shift predicts the gap should shrink; it grows, from 26.6pp to 56.3pp. Raw growth alone is not decisive, since a more sensitive model flags more of everything; that is also why this model’s raw wrong-scheme rate is elevated. The within-corpus model still classifies matched negatives as their own source type $3 2 . 9 \pm 3 . 4$ points more often than wrong-scheme negatives, with five of eight types at zero, and domain shift produces no such asymmetry.

Corroborating corpora. Of Argotario’s two types inside CoCoLoFa’s label space, Irrelevant Authority shows the effect strongly: matched 78.8% against native 11.7%. Hasty generalization does not, at 6.8%, below the native rate. Its matched negatives also had the lowest judge precision of any type in the validation study (66.7%, §6.7). MAFALDA’s qualitative check is consistent with the effect (Table 5).

Two types behave differently. Six types show type-specific gaps between +31.8 and +73.1pp. Appeal to nature (+10.7pp single-seed, +7.9 multi-seed) and false dilemma (+19.4, +23.8) are weaker. False dilemma is a structural outlier on Walton’s account, so its deviation is predicted; it is also the weakest type on Reddit (+2.1pp). Appeal to nature is the weakest type on every detector we tested, encoder and zero-shot alike, which is consistent with the surface term “natural” acting as a lexical cue (see Limitations).

## 6.4 The effect is not specific to fine-tuned encoders

Encoders fine-tuned on a diluted negative class could have learned the dissociation from it. We repeated the measurement with three zero-shot detectors on the identical items: Claude Sonnet 5 and

<table><tr><td>Model</td><td>Native</td><td>Matched</td><td> $\mathbf { W S } \left( \mathbf { r a w } \right) ^ { \dagger }$ </td><td>N→M</td><td>Type-specific</td></tr><tr><td>CoCoLoFa / ModernBERT (5 seeds)</td><td> $1 6 . 6 \% \pm 1 . 6 \%$ </td><td> $5 8 . 9 \% \pm 2 . 9 \%$ </td><td> $9 3 . 5 \% \pm 1 . 8 \%$ </td><td> $4 2 . 3 \mathrm { p p }$ </td><td> $+ 4 0 . 9 \pm 1 . 9 \mathrm { p p }$ </td></tr><tr><td>CoCoLoFa / RoBERTa (3 seeds)</td><td> $1 9 . 7 \% \pm 3 . 3 \%$ </td><td> $7 0 . 3 \% \pm 3 . 2 \%$ </td><td> $9 6 . 5 \% \pm 0 . 8 \%$ </td><td> $5 0 . 7 \mathrm { p p }$ </td><td> $+ 4 8 . 5 \pm 3 . 9 \mathrm { { p p } }$ </td></tr><tr><td>Reddit / ModernBERT, within-corpus (3)</td><td>5.7%±0.1%</td><td> $6 2 . 0 \% \pm 2 . 8 \%$ </td><td> $6 0 . 2 \% \pm 4 . 0 \%$ </td><td> $5 6 . 3 \mathrm { p p }$ </td><td> $+ 3 2 . 9 \pm 3 . 4 \mathrm { p p }$ </td></tr><tr><td>Reddit, cross-corpus</td><td>16.3%</td><td>42.9%</td><td>n.r.</td><td> $2 6 . 6 \mathrm { p p }$ </td><td>n.r.</td></tr></table>

Table 1: False-positive rates (means±SD across seeds). WS = wrong-scheme; N→M = native-to-matched gap; type-specific = the gap in how often each condition is classified as the source type τ; n.r. = not computed. <sup>†</sup>Wrongscheme columns are flag rates, not false-positive rates: the condition is certified valid at 77.5% (§6.8), so some fraction of those flags is correct.

Haiku 4.5, varying scale within one family, and Qwen3.6-27B, varying training lineage. All ran without chain-of-thought, with the prompt frozen before the first call (Appendix C); 2 of 10,260 responses failed to parse.

It reproduces on all three (Table 2). Wrongscheme items are labelled with the scheme they instantiate 69.9–86.3 points more often than with the source type, and the source-type rate stays between 0.3% and 0.5% against the encoders’ 0.4%. Haiku 4.5 lands closest to the encoders on both arms: a type-specific gap of +45.6 against their +40.9, and a τ<sup>′</sup>-over-τ margin of +86.3 against their +85.5 (the 85.9% and 0.4% of §6.3), while Sonnet 5, the most capable detector tested, shows the effect most weakly of the Claude pair and still separates the conditions by +31.0pp. The effect attenuates with capability rather than appearing with it, and no detector escapes it.

The raw gap does not survive; the dissociation does. The native-to-matched gap is +13.5pp for Sonnet 5 and +12.6 for Haiku 4.5, but +0.5pp for Qwen3.6-27B, whose interval spans zero: on that detector the raw measure shows no effect at all. The same items give it a type-specific separation of +25.5pp and a $\tau ^ { \prime }$ margin of +69.9pp. On both Claude models the gap reverses sign under a binary prompt posing only the detection job (Sonnet 5 −7.7pp, Haiku 4.5 −33.2pp), all four intervals excluding zero; under the same prompt Qwen3.6-27B’s null typed gap turns clearly negative (−25.6pp). Naming a type forces a commitment that native items, instantiating no tracked scheme, mostly fail. The rate is therefore unstable across prompts and across models, while the comparison between two conditions built by one pipeline is stable across both.

## 6.5 A negative class built deliberately

Every benchmark in §6.1 built its valid class without scheme-level targeting, so the audit has only been applied where it was expected to find a gap. The Touché 2026 shared task (Kiesel et al., 2026) supplies the missing case: its non-fallacious class was assembled by selecting arguments that resemble a specific fallacy type.

<table><tr><td>Detector</td><td>Native</td><td>Matched</td><td>WS†</td><td>M−WS as τ</td></tr><tr><td>ModernBERT ft.</td><td>16.6%</td><td>58.9%</td><td>93.5%</td><td> $+ 4 0 . 9 \pm 1 . 9$ </td></tr><tr><td>Sonnet 5</td><td>24.1%</td><td>37.6%</td><td>74.1%</td><td> $+ 3 1 . 0$ </td></tr><tr><td>Haiku 4.5</td><td>53.9%</td><td>66.6%</td><td>90.4%</td><td>+45.6</td></tr><tr><td>Qwen3.6-27B</td><td>32.5%</td><td>33.0%</td><td>n.r.</td><td>+25.5</td></tr><tr><td colspan="5">deliberately built class (Reddit-derived)</td></tr><tr><td colspan="5">ModernBERT ft. 16.3%</td></tr></table>

Table 2: The measurement across detectors, typed prompt, CoCoLoFa unless noted. M−WS as τ is the type-specific separation in percentage points. Qwen3.6- 27B shows no native-to-matched gap $( + 0 . 5 \mathrm { p p } .$ interval spanning zero) yet a +25.5pp separation on the same items. <sup>†</sup>Wrong-scheme columns are flag rates, not falsepositive rates (§6.8). Binary-prompt rates are in the text. Last row: the cross-corpus baseline and our matched negatives against Touché’s negative class (§6.5).

The label spaces coincide: all 473 non-fallacious entries carry a resemblance tag, and all eight tags map onto CoCoLoFa’s eight types, 55–64 items each. The corpus is Reddit-derived, so the baseline is the cross-corpus row of Table 1: the CoCoLoFatrained model flags Reddit’s native class at 16.3%, where 5.7% elsewhere is the within-corpus model on the same class; rates are in Table 2. The audit discriminates, coming in well below the 42.9% on our matched negatives. And resemblance-selection recovers roughly a third of the distance that criticalquestion matching recovers: selecting arguments that resemble a fallacy is not equivalent to constructing arguments that answer the scheme’s critical question. When flagged, Touché items are called the type they resemble only $1 5 . 1 \pm 1 . 7 \%$ of the time.

Register does not explain the gap. Our matched negatives are machine-written and more polished than raw forum text, so classifiers might flag them for register. Touché bounds this directly: it contains the same arguments raw and in a self-contained rewrite folding in the parent comment. The rewrite is 9.2 words longer on average and 23 points higher in contraction rate, and it moves the false-positive rate from $2 4 . 6 \pm 1 . 9 \%$ to $2 6 . 2 \pm 1 . 8 \%$ : 1.6 points, smaller than the between-seed standard deviation. A register shift of that size cannot produce a 16.7- point gap. The raw items are also register-matched to our native Reddit class (46.5 against 42.7 words, 52.2% against 51.0% contractions).

A further Touché version rewritten with knowledge of the fallacy and scheme labels shows 36.6 ± 1.2% on identical arguments; we exclude it as a second construction artifact in the same benchmark.

## 6.6 The effect appears on text no one wrote for this study

Every result above uses generated negatives. As a check that depends on none of them, we identified scheme-matched valid arguments already present in the native “none” classes: 25 hand-selected candidates from CoCoLoFa, Reddit and Argotario, pooled with the §6.1 samples and coded by the same two annotators, blind to which items were targeted. The annotators confirmed 10 of the 25 candidates, which with the 3 found in the random samples gives 13 confirmed scheme-matched arguments. Deliberate search barely finds such arguments; the pool is not a base-rate sample and does not revise §6.1.

The classifier flags 7 of the 13 (53.8%; 95% CI [29.1%, 76.8%]), against 19.1% of the genuine arguments instantiating no tracked scheme (21 of 110) and 5.9% of non-arguments (5 of 85). That comparison is the wrong-scheme contrast occurring naturally: A and B are both genuine arguments, differing only in whether a tracked scheme is present. Flag rate rises with scheme content, on text written years before this study. The scheme-specificity is sharp as well. Of the six flagged items with a pinned target scheme, five are classified as exactly that scheme; the sixth, a citation of scientific consensus on climate change, is labelled appeal to majority—a confusion between neighbouring schemes, not a generic over-flag.

## 6.7 Human annotators confirm the negatives are valid

The reported rate would be inflated if the judge had admitted subtle fallacies, so we checked it.

Three independent annotators, none of them authors, judged 72 matched negatives, 56 from Co-CoLoFa and 16 from Reddit, as Valid, Invalid or Borderline under rules fixed in advance (protocol in Appendix B). The items came from the judge’s retained set, the one the headline rate is computed over; annotators saw only the argument text with its target scheme and critical question, never classifier output. Eight benchmark-labelled fallacies were planted and excluded from all figures; annotators caught seven, seven and six of them.

The judge’s precision on its retained set is 93.1% (67 of 72): 94.6% on CoCoLoFa (53 of 56), 87.5% on Reddit (14 of 16). Cohen’s κ is deflated by the prevalence of Valid items, so we report Gwet’s AC1 at 0.69 binary and 0.74 three-way.

Restricted to the human-confirmed-valid Co-CoLoFa subset, the false-positive rate is 56.6% (30 of 53; 95% CI [43.4%, 69.8%]) against a native 16.6%, a gap of 40.0 points, of which judge impurity accounts for roughly one point. The classifier over-flags more than half of the arguments three blind annotators confirmed valid. Reddit agrees (57.1%, 8 of 14, against 5.7%) on a sample too small to carry the claim.

## 6.8 Validity of the wrong-scheme condition

The judge never checked critical-question satisfaction for wrong-scheme items, so the raw rates in Table 1 sit on an uncertified population. A separate, non-overlapping panel repeated the §6.7 protocol against each item’s target scheme (n=40; Table 4): precision is 77.5% (95% CI [62.5%, 87.7%]) against 93.1% for matched items, the shortfall concentrated in the appeal-to-nature and falsedilemma target schemes. The type-specific statistic is unaffected: a genuine fallacy of its own scheme is still not a fallacy of the source type.

## 7 Conclusion

Across four benchmarks, scheme-matched valid arguments make up a few percent at most of the valid class, so the reported rate averages over the wrong population. On the right one—wrong-scheme arguments labelled with their own scheme 85.9% of the time, with the scored type 0.4%—these classifiers have learned which scheme an argument uses, not whether it uses it correctly. A reported falsepositive rate should not be trusted as a measure of detection until the valid class has been audited for scheme-matched coverage.

## Limitations

Our generated rates are not estimates of deployment-time magnitude. A rate measured on constructed items is a joint function of classifier and generator, so the 58.9% and 62.0% figures should not be read as the rate a deployed detector would show, and the naturally-occurring check (§6.6) is too small (13 confirmed items) to pin that magnitude on its own.

The §6.1 samples remain small (183 items, three of them category A), so the percentages are a direction, not a prevalence. The matched negatives carry a 6.9% impurity. The validation sample is modest (n=72, with a 16-item Reddit subset), the judge is unvalidated on Argotario and MAFALDA, and the two validation panels were different groups on a task where judgment varies. The wrong-scheme condition is certified valid at only 77.5% (§6.8).

The zero-shot check (§6.4) covers three detectors on CoCoLoFa only, and the only open-weights detector tested is mid-size. Wrong-scheme flag rates throughout are flag rates, not false-positive rates: 22.5% of that condition is not certified valid (§6.8), so some fraction of those flags is correct. The Touché comparison (§6.5) is between different items, not counterparts of ours, so construction protocol is not the only thing that differs between the two negative classes; the register analysis bounds one alternative and not the others. Argotario’s corroboration is partial: of its two types inside Co-CoLoFa’s label space, one shows the effect and one does not. Argotario and MAFALDA enter the rarity finding but not the primary quantitative claims.

We use Walton’s schemes instrumentally; their individuation has been challenged (Katzav and Reed, 2004), a dispute our argument does not need to resolve, since it requires only that some theorygrounded partition of “same reasoning move” works for the schemes a taxonomy already uses. We do not operationalize pragma-dialectics.

The constructed negatives are not used to replace any benchmark’s valid class; building and validating a corrected benchmark, and showing that it improves detection, is a separate contribution that presupposes the measurement problem documented here. Scheme Foils covers eight fallacy types on two corpora, and the audit’s coverage is bounded by the schemes a taxonomy already tracks; we open the released sets for extension to further schemes and corpora. Whether the size of the gap tracks a scheme’s prevalence among natural valid arguments, and whether equalizing per-scheme representation lowers false-positive rates, are untested.

## Ethical Considerations

The annotators in all three studies (§6.1, §6.7, §6.8) were volunteers who consented to participate and were not compensated; they were briefed on the task and were not exposed to harmful content beyond ordinary online argumentation. Generation prompts instruct the model to substitute a related defensible point where a source item rests on a harmful, discriminatory or dehumanizing premise. All source corpora are publicly released research datasets and are used within their licences; we redistribute source items only where licensing permits.

## References

Vanessa Wei Feng and Graeme Hirst. 2011. Classifying arguments by scheme. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 987–996, Portland, Oregon, USA. Association for Computational Linguistics.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, and 7 others. 2020. Evaluating models’ local decision boundaries via contrast sets. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 1307–1323, Online. Association for Computational Linguistics.

Pierpaolo Goffredo, Shohreh Haddadan, Vorakit Vorakitphan, Elena Cabrio, and Serena Villata. 2022. Fallacious argument classification in political debates. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 4143–4149. International Joint Conferences on Artificial Intelligence Organization. Main Track.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A. Smith. 2018. Annotation artifacts in natural language inference data. In Proceedings ofthe 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 107–112, New Orleans, Louisiana. Association for Computational Linguistics.

Ivan Habernal, Raffael Hannemann, Christian Pollak, Christopher Klamm, Patrick Pauli, and Iryna Gurevych. 2017. Argotario: Computational argumentation meets serious games. In Proceedings of

the 2017 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 7–12, Copenhagen, Denmark. Association for Computational Linguistics.

Chadi Helwe, Tom Calamai, Pierre-Henri Paris, Chloé Clavel, and Fabian Suchanek. 2024. MAFALDA: A benchmark and comprehensive study of fallacy detection and classification. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4810–4845, Mexico City, Mexico. Association for Computational Linguistics.

Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and fairness. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, pages 375–385. Association for Computing Machinery.

Zhijing Jin, Abhinav Lalwani, Tejas Vaidhya, Xiaoyu Shen, Yiwen Ding, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea, and Bernhard Schölkopf. 2022. Logical fallacy detection. In Findings ofthe Association for Computational Linguistics: EMNLP 2022, pages 7180–7198, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Joel Katzav and Chris Reed. 2004. On argumentation schemes and the natural classification of arguments. Argumentation, 18(2):239–259.

Divyansh Kaushik, Eduard Hovy, and Zachary C. Lipton. 2020. Learning the difference that makes a difference with counterfactually-augmented data. In International Conference on Learning Representations.

Johannes Kiesel, Marc Feger, Tim Hagen, Sebastian Heineking, Maximilian Heinrich, Maik Fröbe, Katarina Boland, Wilhelm Pertsch, Julia Romberg, Ines Zelch, Stefan Dietze, Matthias Hagen, Martin Potthast, and Benno Stein. 2026. Overview of Touché 2026: Argumentation systems. In Advances in Information Retrieval. 48th European Conference on IR Research (ECIR 2026), volume 16486 of Lecture Notes in Computer Science, pages 277–286. Springer Nature.

John Lawrence and Chris Reed. 2016. Argument mining using argumentation scheme structures. In Computational Models of Argument: Proceedings of COMMA 2016, pages 379–390. IOS Press.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. Preprint, arXiv:1907.11692.

Fabrizio Macagno. 2022. Argumentation profiles and the manipulation of common ground: The arguments of populist leaders on Twitter. Journal of Pragmatics, 191:67–82.

Tom McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3428–3448, Florence, Italy. Association for Computational Linguistics.

Timothy Niven and Hung-Yu Kao. 2019. Probing neural network comprehension of natural language arguments. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4658–4664, Florence, Italy. Association for Computational Linguistics.

Ramon Ruiz-Dolz and John Lawrence. 2023. Detecting argumentative fallacies in the wild: Problems and limitations of large language models. In Proceedings of the 10th Workshop on Argument Mining, pages 1–10, Singapore. Association for Computational Linguistics.

Ramon Ruiz-Dolz and John Lawrence. 2025. An explainable framework for misinformation identification via critical question answering. Preprint, arXiv:2503.14626.

Saumya Sahai, Oana Balalau, and Roxana Horincar. 2021. Breaking down the invisible wall of informal fallacies in online discussions. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 644–657, Online. Association for Computational Linguistics.

Navyansh Singh. 2026. Debiasing logical fallacy detection for real-world robustness via counterfactually augmented data. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 363–374, San Diego, California, United States. Association for Computational Linguistics.

Douglas Walton. 1996. Argumentation Schemes for Presumptive Reasoning, 1st edition. Routledge, New York.

Douglas Walton. 2010. Why fallacies appear to be better arguments than they are. Informal Logic, 30(2):159– 184.

Douglas Walton, Christopher Reed, and Fabrizio Macagno. 2008. Argumentation Schemes. Cambridge University Press, Cambridge.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Griffin Thomas Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2526–2547, Vienna, Austria. Association for Computational Linguistics.

Min-Hsuan Yeh, Ruyuan Wan, and Ting-Hao Kenneth Huang. 2024. CoCoLoFa: A dataset of news comments with common logical fallacies written by LLMassisted crowds. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 660–677, Miami, Florida, USA. Association for Computational Linguistics.

## A Training Details

The CoCoLoFa ModernBERT models reached 81.4% ± 0.5% dev accuracy across five seeds and the RoBERTa models 81.9% ± 0.8%, against a 39.7% majority baseline; the Reddit models reached 60.4% ± 0.2% across three seeds against a 49.0% baseline. The Reddit models trained six epochs against three, since token-span label aggregation produces a noisier signal, under an identical selection criterion. All classifiers were fine-tuned with AdamW (weight decay 0.01), a linear schedule with 10% warmup, learning rate 2e-5, batch size 16, maximum sequence length 256, gradient clipping at 1.0, and no gradient accumulation. The multi-seed models trained on a single NVIDIA T4; the single-seed run of footnote 1 and all inference ran on a 6 GB RTX 4050 laptop GPU.

Corpus details. CoCoLoFa labels news-article comments with eight fallacy types—appeal to authority, appeal to majority, appeal to nature, appeal to tradition, appeal to worse problems, false dilemma, hasty generalization, slippery slope— plus a “none” class. The Reddit corpus’s eightclass annotation is mapped onto CoCoLoFa’s scheme (its “Black-or-white” is false dilemma). MAFALDA is expert-annotated over a partly disjoint taxonomy.

Statistical procedures. The permutation test shuffles the model’s predicted labels while preserving its overall rate of predicting “none.” In the item-level logistic regression the two conditions are near-perfectly separated—only three of 738 wrong-scheme items are classified as the source type—so the unpenalized odds ratio is unstable and the Firth-penalized refit is reported.

## B Annotation Protocols

Matched negatives (§6.7). Three independent undergraduate volunteers, not authors, briefed and calibrated, told only the coding task and not what the study was testing, consenting and unpaid. The CoCoLoFa sample was stratified seven per type across the eight types. Judgments were three-way— Valid, Invalid, Borderline—under rules fixed in advance: majority vote decides, Borderline never counts as Valid, ties break conservatively. Per-type judge precision was 100% for six of the eight types, 77.8% for appeal to authority and 66.7% for hasty generalization.

Wrong-scheme negatives (§6.8). A separate panel of three annotators of the same background, given the same briefing and calibration and nonoverlapping with the first panel, judged each item (n=40) against the target scheme it was built to instantiate, catching six, four and six of six planted decoys.

None-class characterization (§6.1, §6.6). Two independent annotators of the same background, overlapping with neither panel, coded all 208 items—the four corpus samples and the candidate pool—blind to which items were hand-selected, to all prior codes, and to classifier output. A joint calibration pass on 10 items drawn from outside the coded pool preceded independent coding. Disagreements were resolved by discussion without the authors, under a rule fixed in advance that unresolved items default to the non-A code; none required the default. Six benchmark-labelled fallacies were planted and excluded from all figures; the annotators caught four and five of them.

Scheme Foils. We release the judge-filtered sets under the name Scheme Foils: 655 matched and 738 wrong-scheme CoCoLoFa items, 387 matched and 389 wrong-scheme Reddit items, with target schemes, critical questions, and source items where licensing permits. Each item carries the human-validated precision estimate of the study it belongs to, so a user knows what impurity they inherit (6.9% for the matched split, 22.5% for the wrong-scheme split). The intended use is direct: score an already-trained detector on the matched split against its own reported false-positive rate, and use the wrong-scheme split to check that any gap is scheme-specific rather than generic. Neither requires retraining or regeneration. The sets are available at https://github.com/fine2006/ the-concealment-hypothesis.

## C Generation Prompts

Per-type prompts were calibrated to each corpus’s register, one prompt design per fallacy type, stating the target scheme and its critical question explicitly. The slots cq\_block, target\_cq\_block, source\_type, target\_type and examples\_block are filled per batch. Batches of 15–20 items are generated per call, with strict numbered-output parsing and up to two retries on misalignment. Automated first-pass checks flag source-scheme keyword leakage in wrong-scheme outputs and compare word-count, sentence-count and formal-register statistics between conditions.

[appeal to authority]   
SCHEME: Expert Opinion. The fallacy cites a   
source who is NOT a genuine domain expert for   
the claim being made.   
A SATISFYING FIX replaces the source with a   
genuinely relevant domain expert -- add ONLY the   
minimum credential needed, nothing more (no   
years of experience, no institutional pedigree).   
[hasty generalization]   
SCHEME: Generalization from Sample. The fallacy   
draws a broad conclusion from a sample too small   
or unrepresentative to support it (often a   
personal anecdote).   
A SATISFYING FIX replaces the thin/anecdotal   
sample with one that is genuinely adequate -- a   
real, larger, more systematic basis for the same   
conclusion. Do NOT just hedge the conclusion ("   
might," "could") -- ground it instead.   
[false dilemma]   
SCHEME: Disjunctive reasoning (closest to formal   
logic, not a defeasible Waltonian scheme -- the   
hardest type to fix cleanly).   
A SATISFYING FIX shows the binary is genuinely   
exhaustive in THIS specific context -- e.g. cite   
why middle options have specifically failed or   
been closed off here -- OR explicitly names a   
real third option that resolves the false binary.

## Matched-negative prompt (CoCoLoFa):

You will be given a list of numbered fallacious   
arguments, all of the SAME fallacy type. Your   
task is to write a FRESH, standalone valid   
argument for each one -- not an edit of the   
original -- written exactly the way an ordinary   
person commenting online would write it.   
{cq\_block}   
THE TASK:   
For each example, write a new valid argument on   
the SAME topic, in the SAME casual register,   
with the SAME approximate length and sentence   
count as the original (most originals are 2-5   
sentences). Do not copy or echo specific   
distinctive phrases from the original.   
CRITICAL -- VARY HOW YOU SIGNAL GROUNDING:   
Do not use the same template repeatedly across   
this batch. Mix it up: sometimes a concrete   
specific fact stated plainly, sometimes reported   
speech from a named type of person, sometimes a   
first-person framing, sometimes a plain   
declarative claim with no explicit evidence  
marker.

THE RULE THAT DECIDES CONFLICTS:   
The fix must be genuinely correct -- but   
correctness should be expressed at the SAME   
natural, unforced confidence level an ordinary   
person would use, not "proven" through extra   
explanation or citation. The bar is: a   
reasonable person would NOT consider it an error   
to call it valid.   
DO NOT CHANGE WHAT IS BEING ARGUED FOR:   
The fix must support the SAME conclusion as the   
original -- only the reasoning changes.   
WHAT TO AVOID:   
- Citation-dense outputs ("Studies show...") --   
this reads as an AI-generated abstract.   
- Adding credentials or statistics beyond what   
the fix needs.   
- Tacking on an unverified conditional as the   
only grounding.   
- If the claim rests on a religious/contested  
worldview premise, write "SKIP -- non-empirical   
premise".   
OUTPUT FORMAT: a numbered list, one entry per   
input number, in order, with NOTHING else.   
[number]. [your fresh valid version]   
Here are the examples:   
{examples\_block}

## Wrong-scheme prompt (CoCoLoFa):

You will be given numbered fallacious arguments,   
all of the SAME fallacy type ({source\_type}).   
Your task is NOT to fix that fallacy. Instead,   
write a FRESH, valid argument for each that   
stays on the SAME topic, with the SAME effort,   
length and casual register a normal fix would   
have -- but uses a COMPLETELY DIFFERENT   
reasoning style: {target\_type}.   
THE SCHEME YOU MUST ACTUALLY USE ({target\_type}):   
{target\_cq\_block}   
CRITICAL -- DO NOT USE {source\_type}'S REASONING   
AT ALL:   
Completely avoid {source\_type}'s characteristic   
content and language. Stay on the same topic,   
but make the argument using {target\_type}'s   
logic, as if you were never thinking about {   
source\_type}.   
NOTE ON CONCLUSION: unlike a normal fix, this   
does NOT need to argue for the same specific   
conclusion. Staying on the same general topic   
with comparable effort is what matters.   
SELF-CHECK: if someone read your answer without   
knowing the target scheme, would they identify   
it as {target\_type}? If it could still be read   
as {source\_type}, you have not switched schemes.   
{base\_rules}   
OUTPUT FORMAT: a numbered list, one entry per   
input number, in order, with NOTHING else.   
[number]. [your fresh {target\_type} argument,

same topic as the source]   
Here are the examples:   
{examples\_block}

<table><tr><td>Classify the following argument. If it commits a logical fallacy, answer with the fallacy type from this list: - appeal to authority - appeal to majority</td></tr><tr><td>- appeal to nature - appeal to tradition - appeal to worse problems</td></tr><tr><td>- false dilemma - hasty generalization</td></tr><tr><td>- slippery slope If it does not commit a fallacy, answer with:</td></tr><tr><td>none</td></tr><tr><td>Answer with the label only, nothing else. Argument: {text}</td></tr></table>

The binary variant asks only: Does the following argument commit a logical fallacy? Answer with one word: yes or no. All detectors ran zero-shot with thinking disabled and a 64-token cap. Both prompts ran on all three detectors, 10,260 responses in total, of which 2 failed to parse. Model identifiers: claude-sonnet-5 and claude-haiku-4-5 via the Anthropic API and qwen3.6-27b, all queried 18–20 August 2026; the generator throughout is gemini-2.5-flash and the judge deepseek-v3.

The Reddit variants of both prompts are identical in structure, with the source item given as a SPAN plus its parent comment as CONTEXT, instructions to match the shorter Reddit register, and a SKIP option for spans too short or context-dependent to work with.

## D Complementary Tables

<table><tr><td>Cat.</td><td>CoCo (30)</td><td>Redd (30) MAF (63)</td><td>Argo (60)</td></tr><tr><td>A</td><td>1 (3%)</td><td>1 (3%) 1 (2%)</td><td>0 (0%)</td></tr><tr><td>B</td><td>28 (93%)</td><td>17 (57%) 29 (46%)</td><td>21 (35%)</td></tr><tr><td>Non-arg.</td><td>1 (3%)</td><td>12 (40%) 33 (52%)</td><td>39 (65%)</td></tr></table>

Table 3: Composition of the native “none” class across four corpora by two independent annotators (sample size in parentheses; MAFALDA’s extractable none class coded in full). A = scheme-matched valid argument; B = genuine argument, no tracked scheme; Non-arg. = not an argument (fragments, questions, noise, offtopic remarks; two items are meta-commentary about fallacies).

<table><tr><td>Study</td><td>n</td><td>Precision</td><td>AC1</td></tr><tr><td>Matched (§6.7)</td><td>72</td><td>93.1% (67/72)</td><td>0.69 / 0.74</td></tr><tr><td>Wrong-scheme (§6.8)</td><td>40</td><td>77.5% [62.5, 87.7]</td><td>0.81</td></tr></table>

Table 4: Human validation of both constructed conditions. Precision is the proportion of the judge’s retained set confirmed valid by majority vote; planted decoys are excluded from all precision figures. AC1 is Gwet’s, binary / three-way for the matched study. Panels were separate and non-overlapping.

<table><tr><td>Quantity (CoCoLoFa-trained ModernBERT)</td><td>Value</td></tr><tr><td>Argotario native FPR Argotario matched FPR (all five types)</td><td>11.7% 16.4%</td></tr><tr><td>Argotario matched, Irrelevant Authority</td><td>78.8%</td></tr><tr><td>raw native→matched gap</td><td>67.1pp</td></tr><tr><td>type-specific gap</td><td>+72.7pp</td></tr><tr><td>Argotario matched, hasty generalization</td><td>6.8% (3/44)</td></tr><tr><td>MAFALDA native FPR</td><td>7.9%</td></tr><tr><td>MAFALDA overlapping fallacy spans flagged</td><td>50.5%</td></tr><tr><td>identified by exact type</td><td>41%</td></tr></table>

Table 5: Corroborating-corpus results (§6.3). Three of Argotario’s five types (ad hominem, appeal to emotion, red herring) fall outside CoCoLoFa’s label space and are uninformative by construction; the two that overlap are reported separately. MAFALDA is evaluated qualitatively, its native rate low for the label-space reason given in §6.1.