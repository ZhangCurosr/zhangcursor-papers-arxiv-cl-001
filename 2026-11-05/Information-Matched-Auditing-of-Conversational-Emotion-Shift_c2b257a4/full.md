# Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol

Bin Wen<sup>1</sup>, Dai-Qiao Zhang<sup>2</sup> and Tien-Ping Tan<sup>1</sup>

<sup>1</sup>School of Computer Sciences, Universiti Sains Malaysia, 11800 Gelugor, Penang, Malaysia <sup>2</sup>School of Mathematical Sciences, Universiti Sains Malaysia, 11800 Gelugor, Penang, Malaysia

Corresponding author: Tien-Ping Tan (e-mail: tienping@usm.my)

Abstract: Comparisons in conversational emotion-shift forecasting are easily distorted when a learned model receives rich features but its transition baseline is given a noisy or unavailable current-emotion label, when hyperparameter budgets difer, or when target-turn information enters the representation. We present an information-matched, leakage-safe auditing protocol that closes these gaps with an explicit commitment point, executable causality tests, equal four-configuration validation searches, and seed-dialogue-aware inference. Applying it to six causal model families on four text corpora and two feature-source sensitivity configurations shows that a deployable train-only current-emotion classifier feeding the transition baseline yields a diferent verdict than an oracle diagnostic where both sides receive gold current emotion. Under the deployable condition, learned models improve over the predicted-label baseline on all four primary corpora; DailyDialog gains are large and survive a 72-comparison Holm correction (best ROC-AUC gain = +0.167, adjusted $p = . 0 1 4 4 )$ . Under matched gold information, the baseline remains competitive on IEMOCAP and stronger on MELD, while every model retains a large DailyDialog advantage. We additionally report precision-recall AUC, calibration, a strictly utterance-loca feature control, same-person self-shift forecasting, and a future-access dose-response test. An eight-item leakage checklist accompanies the protocol. Code, raw seed predictions, and reporting artifacts are available from the corresponding author upon request.

Keywords: emotion forecasting, conversational emotion recognition, evaluation protocol, data leakage, benchmarking

## 1. Introduction

Consider a call-center assistant that flags, one turn before it happens, that a customer is about to escalate from neutral to angry – giving the agent seconds to change course. This is the promise of forecasting emotion in conversation, as opposed to recognizing it: the target state does not yet exist when the prediction is made. It is also precisely the setting in which a model can look right without being right. If any feature – a pooled dialogue embedding, a bidirectional encoder, a context window one utterance too wide – depends even slightly on the turn being forecast, the model is not anticipating the future; it is reading it of a crib sheet built into its own inputs. The failure is silent because nothing in a standard run signals it happened: loss curves look normal, F1 looks respectable, the leaderboard entry looks like progress.

Emotion recognition in conversation (ERC) has a large, mature literature (Section II). A smaller but growing thread reframes the task as forecasting – predicting an upcoming emotional shift before the triggering utterance is seen, the setting relevant to anticipatory dialogue systems, deescalation tools, and turn-taking agents. Forecasting is target-absent by construction: no feature may depend on utterance n+1. This constraint is easy to violate silently, e.g. via bidirectional encoders, dialogue-level pooling, or context windows not truncated at the decision point – and the architectures ERC has spent a decade refining are, almost without exception, built to violate it, because recognition never had to enforce it in the first place.

This paper asks a measurement question: how do conclusions change when the future is actually hidden and the learned model and transition baseline receive comparable current-state information? The two requirements are distinct. A causal model without gold labels should be compared with a baseline driven by train-only predicted labels; a baseline supplied with gold $y _ { n }$ should be compared with a model supplied with the same $y _ { n }$ . We contribute a measurement framework with six parts:

1) a formal commitment point and target-absent data construction for next-turn emotion shift;

2) an executable eight-item leakage audit plus futureperturbation causality tests;

3) train-only predicted-label and gold-label transition baselines that expose the information available in each comparison;

4) equal four-configuration validation searches and checkpoint selection for six causal model families, evaluated over three seeds;

5) seed–dialogue hierarchical intervals, paired dialogue-cluster permutation tests, and a declared familywise Holm correction; and

6) provenance, future-access, and speaker-target controls that test whether conclusions survive feature re-extraction, target redefinition, and controlled leakage.

Applying this framework yields a picture that is conditional rather than uniformly negative: deployable models consistently improve on the predicted-label baseline, while an information-matched oracle baseline remains hard on several corpora. The final contribution matters because the two targets answer diferent questions. An interaction-level system may legitimately care whether the conversation’s next turn changes emotion regardless of who speaks. A coaching or wellbeing system may instead care whether the same person changes state when they next speak. Reporting one as if it were the other can produce a semantically misleading claim even when the code is temporally leakagefree.

## 2. Related Work

ERC backbones. Our four ERC-derived models are representative of three context-modeling paradigms: recurrent interlocutor-state tracking [7], relational graph convolution [8], [10], and directed relational attention [9]. Knowledge-enriched encoders [11], [5] and multimodal fusion architectures [14], [15], [16], [17], [18] inform our feature choices (COSMIC RoBERTa; MM-DFN [6]-style text+audio+visual). All are, in original form, bidirectional or graph-connected across the full dialogue – appropriate for recognition, but a direct leakage source if repurposed for forecasting without a causal mask; interactive memory [13] and contextual-reasoning networks [12] share this property.

Emotion forecasting (EFC). A parallel line explicitly targets an unobserved future utterance. Altarawneh et al. [20] formalize PEC as predicting turn n+1 from context strictly up to n, decomposing the signal into sequence, self-dependency, and recency terms (the strategy we reimplement as PEC-style, Section V). Shi et al. [21], [22] model bilateral-speaker correlation and word-level cues in text+audio. Hi-EF [19] contributes a dedicated multimodal forecasting benchmark. MREPC [23] pre-trains contrastively over a short-term cross-speaker window. PUGCN [24] substitutes a pseudo-utterance-guided contrastive objective for the absent target (the strategy behind our Pseudofuture-style re-implementation). EDFC [25] targets a nextutterance emotion distribution; ERFC [26] shows feasibility in a deployed call-center setting. All are cross-speaker by design and, to our knowledge, none audits a leakage boundary as strict as ours.

Emotion shift and flip: adjacent, not forecasting. ESD-ERC [27], CFN-ESA [28], and EmoShiftNet [29] use shift labels to sharpen recognition of an observed current utterance – computed over turns the model can already see. Emotion-flip reasoning [30], [31] runs backward: given a flip that has already occurred, it attributes the past trigger. Neither predicts an unobserved future shift under a leakagesafe protocol; we exclude both, noting our transition matrix is close in spirit to the inertia signal they are designed to overcome.

Leakage and baseline validity. Kapoor and Narayanan [32] survey at least 294 papers across 17 fields and find leakage the dominant cause of irreproducible ML-based findings, proposing the taxonomy our 8-item audit is modeled after. Ferrari Dacrema et al. [33] show most neural recommenders fail to beat tuned classical baselines once evaluation is corrected; Lin [34] makes the same point for neural ranking. We view our transition-matrix result as the same phenomenon in emotion forecasting.

## 3. Task Definition and Threat Model

Let a dialogue be $D = \{ ( x _ { t } , y _ { t } , s _ { t } ) \} _ { t = 1 } ^ { T } .$ , where $x _ { t }$ is an utterance-level feature vector, $y _ { t } \in \{ 1 , \cdot \cdot \cdot , K \}$ an emotion label, and $s _ { t }$ a speaker identifier. At commitment point n, a legitimate forecaster may use only

$$
{ \cal \bar { I } } _ { n } = \left\{ x _ { 1 : n } , s _ { 1 : n } \right\} \quad \mathrm { a n d , ~ w h e n ~ d e c l a r e d , } \ : y _ { n } ,\tag{1}
$$

but no quantity computed from $x _ { n + 1 : T } ~ \mathrm { o r } ~ y _ { n + 1 : T }$ . This definition includes an important feature-provenance condition: $x _ { t }$ must itself be utterance-local. Merely slicing a dialoguelevel bidirectional representation at index n does not make it causal.

## A. Two Forecasting Targets

Our primary target is the interaction-level immediate shift

$$
z _ { n } ^ { \mathrm { i n t } } = \mathbb { k } [ y _ { n + 1 } \neq y _ { n } ] , \qquad n < T .\tag{2}
$$

It is operationally appropriate when the question is “will the emotion expressed in the next turn difer from the emotion expressed now $\eta ^ { \ast }$ It is agnostic to speaker identity and therefore mixes two strata: $s _ { n + 1 } = s _ { n }$ and $s _ { n + 1 } \neq s _ { n }$

For semantic clarity we additionally define the next-ownutterance self-shift. Let g(n) = min{m $> n : s _ { m } = s _ { n } \}$ when such an utterance exists. Then

$$
z _ { n } ^ { \mathrm { s e l f } } = \nVdash [ y _ { g ( n ) } \neq y _ { n } ] .\tag{3}
$$

This target asks whether the current speaker changes emotion by the time they next speak, allowing intervening turns by others. It is not “more correct” in the abstract: it answers a diferent product and scientific question, has a variable forecast horizon, and excludes terminal appearances of each speaker. We report both rather than silently substituting one for the other.

## B. What Counts as Leakage?

Table I distinguishes information leakage from other sources of optimistic or misleading evidence. The first three

rows violate Eq. (1); the remaining rows do not necessarily leak the future but can still make comparisons invalid.

## 4. Protocol

Commitment point. For decision point $n ,$ all input features (text/audio/visual) are restricted to utterances $\leq n ;$ the target label is $\mathbb { W } [ y _ { n + 1 } \neq y _ { n } ]$ . All six learned models (Section V) are structurally causal, not causal-by-convention.

Information-matched regimes. In the deployable regime, learned models receive utterance features and causal history but no gold emotion label. The transition baseline instead uses $\hat { y } _ { n }$ from a linear current-emotion classifier fit exclusively on training utterances; its regularization is selected on the validation fold. In the oracle diagnostic, the gold one-hot $y _ { n }$ is appended to every learned model and the transition baseline conditions on the same gold $y _ { n }$ . This is not a deployment claim: it isolates the value of forecasting beyond a shared, perfect observation of current state.

Causal construction and a zero-tolerance check. DialogueRNN’s global/party/emotion states update by chained GRUCell calls that never touch $x _ { t + 1 } .$ . DialogueGCN uses windowed causal edges (j→i if $j \leq i$ and $i - j \leq 6 ) $ ; DAG-ERC uses unbounded causal attention $( j \leq i ,$ any distance) – two families difering sharply in receptive field yet still landing within the same ≲0.04 AUC band (Section V). We verified causality directly rather than trusting architecture alone: perturbing every input at $t ~ \geq ~ n$ to random noise leaves outputs at $t \ < \ n$ changed by exactly 0.0 (floatingpoint identical) for every model.

Automated leakage audit. Every experiment emits an 8-item checklist: last-utterance masking, model causality, per-utterance feature scope, speaker-split disjointness, valonly threshold tuning, dialogue-level bootstrap, train-foldonly baseline fitting, and baseline completeness.

Strong baseline. Beyond base-rate and nochange/inertia, we fit a speaker×emotion transition matrix $P ( y _ { n + 1 } \mid y _ { n } ,$ speaker) on the train fold only (Laplace $\alpha { = } 1 . 0 ; \textbf { a }$ speaker’s row is used only with ≥20 training observations, else global backof). This is the baseline every model must beat.

Equal validation budget. Each architecture receives the same four candidates spanning hidden size {64 128 256}, learning rate $\{ 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ , dropout { 1 3}, and focal versus class-balanced cross-entropy loss. Seed 0 selects the configuration on validation ROC–AUC only; the selected configuration is then trained with seeds {0 1 2} for at most 30 epochs, five-epoch patience after at least five epochs. The test fold is never evaluated during search or checkpoint selection.

Metrics. ROC–AUC is primary because it is thresholdfree. We additionally report PR–AUC for the imbalanced shift class, Brier score and expected calibration error, shift-F1, precision, recall, and balanced accuracy. Predictors assigning more than 98% of decisions to one class are automatically flagged.

Significance. Diferences are estimated with a hierarchical bootstrap [37] that first samples training seeds and then resamples whole test dialogues (1,999 replicates). Twosided paired dialogue-cluster permutation tests use 4,999 sign flips and the plus-one correction, so a reported p-value is never zero. Holm–Bonferroni correction is applied jointly across the two information regimes, six models, and six dataset-configurations (72 declared comparisons).

Reproducibility contract. Random seeds are fixed at 0, 1, and 2. Each dataset is loaded and split before model construction; no fitted current-emotion classifier or cached prediction is shared across seeds. Dataset manifests record SHA-256 hashes, and independent tests verify that perturbing inputs at and after the forecast boundary cannot alter earlier logits. DailyDialog validation/test dialogues exactly duplicated in training are removed before any search.

## 5. Experiments

Data. The primary analysis uses four text corpora with released RoBERTa features (IEMOCAP [2], MELD [1], EmoryNLP [3], DailyDialog [4]; COS-MIC pipeline [5]), plus two multimodal configurations (IEMOCAP, MELD with MM-DFN [6] text+OpenSmileaudio+DenseNet-visual features) reported only as featuresource sensitivity analyses because the available release does not include the complete upstream extraction code. Table II gives size and shift-rate statistics (0.20–0.72) together with an F1-degeneracy analysis referenced below. The six configurations are not six independent corpora: IEMOCAP and MELD each appear once with COSMIC text features and once with MM-DFN multimodal features; inferential claims are made per configuration and are not treated as six independent replications. As an independent control that does not rely on released embeddings, we also fit TF–IDF vocabulary/IDF and truncated SVD on training utterances only and transform every utterance independently. Daily-Dialog contains 134 test and 101 validation dialogues exactly duplicated in training; all revised DailyDialog results remove those dialogues before configuration selection or evaluation.

## A. Compared Models

The benchmark spans recurrence, structured speaker state, graph propagation, directed attention, and two forecasting-specific strategies. Each model maps the same feature sequence to one logit per observed turn and uses the same loss and optimization budget. DialogueRNN maintains global and party states; causal DialogueGCN propagates messages only from earlier nodes in a six-turn window; DAG-ERC uses unbounded causal attention; PEC-style concatenates sequence, self-dependency, and exponentially weighted recency states; and Pseudo-future-style predicts an embedding for $x _ { n + 1 }$ from the causal state before classifying shift, where the actual $x _ { n + 1 }$ is only an auxiliary regression target during training and never a forward-path input.

TABLE I. Threat model for conversational emotion forecasting. The audit treats target-derived features and target-derived model states as zerotolerance failures; the other threats require separate statistical or reporting controls.
<table><tr><td>Threat</td><td>Failure mechanism</td><td>Detection</td><td>Required control</td></tr><tr><td>Target-turn exposure</td><td>xn+1 enters a window, pooled context, or auxiliary input</td><td>inspect window boundaries; sweep visible future turns</td><td>truncate at n before any ag- gregation</td></tr><tr><td>Bidirectional state</td><td>output at n depends on representations from  $n + 1 { : } T$ </td><td>perturb future inputs and compare prefix logits</td><td>unidirectional recurrence or explicit causal mask</td></tr><tr><td>Feature provenance</td><td>nominally per-turn xn was extracted with dialogue-level future context</td><td>trace the feature extractor, not only the classifier</td><td>utterance-local extraction or causal re-extraction</td></tr><tr><td>Split contamination</td><td>duplicate dialogue text or the same global speaker occurs across folds</td><td>hash dialogue text; audit global iden- tities</td><td>group-aware split and dupli- cate removal</td></tr><tr><td>Selection leakage</td><td>test fold determines threshold, epoch, or model family</td><td>inspect tuning log</td><td>validation-only selection with fixed test evaluation</td></tr><tr><td>Metric degeneracy</td><td>high base rate makes always-shift F1 look competitive</td><td>report prediction rate and balanced metrics</td><td>AUC, balanced accuracy, and constant baselines</td></tr><tr><td>Semantic target drift</td><td>cross-speaker next-turn change is described as same-person emotional change</td><td>stratify by  $s _ { n + 1 } = s _ { n } ;$  evaluate Eq. (3)</td><td>name the target and deploy- ment contract</td></tr></table>

TABLE II. Dataset statistics and F1 degeneracy. Always-shift F1 = 2r/(1+r) for shift rate r (analytic). Transition-BA ≈0.50 on MELD/EmoryNLP/IEMOCAP-mm shows standard F1 sits at the degenerate ceiling on exactly the corpora where shift is common; only IEMOCAP and DailyDialog clear it.
<table><tr><td>Dataset</td><td>#Dial</td><td>#Utt</td><td>Shift r</td><td>Always-shift F1</td><td>Transition BA</td></tr><tr><td>IEMOCAP</td><td>151</td><td>7,433</td><td>0.429</td><td>0.600</td><td>0.611</td></tr><tr><td>MELD</td><td>1,432</td><td>13,708</td><td>0.581</td><td>0.735</td><td>0.500*</td></tr><tr><td>EmoryNLP</td><td>827</td><td>9,489</td><td>0.722</td><td>0.839</td><td>0.500*</td></tr><tr><td>DailyDialog</td><td>13,118</td><td>102,979</td><td>0.198</td><td>0.331</td><td>0.681</td></tr><tr><td>IEMOCAP-mm</td><td>151</td><td>7,433</td><td>0.429</td><td>0.600</td><td>0.500*</td></tr><tr><td>MELD-mm</td><td>1,432</td><td>13,708</td><td>0.581</td><td>0.735</td><td>0.500*</td></tr></table>

State-factorized control. Motivated by work that predicts the next-utterance emotion distribution directly [25], we add a causal multi-task control with three heads: current emotion, future emotion, and direct shift. It reaches .628/.628/.546/.747 ROC–AUC on IEMOCAP/MELD/EmoryNLP/DailyDialog and improves over the deployable transition baseline by +.052/+.041/+.043/+.166; MELD and DailyDialog remain significant after four-test Holm correction (p = 0036 and 0008). We also retain a bidirectional GRU solely as a positive leakage control (not a candidate forecaster): if the leaky model improves sharply while the safe model does not, this argues against a failure of the benchmark implementation itself.

## B. Speaker-Relation Audit

Table III exposes a compositional property hidden by aggregate shift rates. In the test folds, only 27.8% of IEMOCAP and 24.2% of MELD adjacent decisions retain the same speaker; EmoryNLP retains the speaker in 9.1%, and DailyDialog alternates speakers by construction. Most labels in Eq. (2) therefore compare the emotions of two diferent people. IEMOCAP’s cross-speaker shift rate is 0.509 versus 0.222 for same-speaker adjacent turns; MELD’s is 0.631 versus 0.423 (Figure 1). A 5,000-sample dialogue-cluster bootstrap confirms the switch-minus-same diference is not an utterance-count artifact on IEMOCAP (+0 287, 95% CI [+0 200 +0 373]) or MELD (+0 208, [+0 151 +0 265]); both two-sided $p ~ < ~ . 0 0 1$ . EmoryNLP shows no supported diference (−0 025, p = 614).

Speaker conditioning contributes exactly zero AUC on every corpus: test-time speaker keys are, by construction, absent from the train-fold speaker table, so the estimator always backs of to the global row. This evaluation provides no evidence that speaker-identity conditioning helps, only that dialogue-local role identities cannot be tested for a within-person efect without a genuinely cross-dialogue speaker identifier. Forecast horizon is another hidden source of heterogeneity (Table V): IEMOCAP and MELD show markedly higher shift rates after gaps of at least three turns than at gap one.

Main result (Table VII). Information matching changes the conclusion. In the deployable regime, all six models exceed the train-only predicted-label transition baseline on every primary text corpus in point estimate. The gain is decisive on duplicate-free DailyDialog (+0 160 to +0 167 ROC–AUC; all adjusted $p = . 0 1 4 4 )$ and smaller elsewhere. Among the primary text corpora, DialogueRNN on MELD is the only other comparison that survives the 72-test familywise correction $( \Delta = + 0 . 0 4 1$ , adjusted $p = . 0 3 5 2 ) \colon$ ; all six MELD-mm gains also survive but sit in a feature-source sensitivity configuration rather than a primary corpus.

The oracle diagnostic (Table VIII) asks a diferent question. When both sides receive gold $y _ { n } { \mathrm { : } }$ all six models remain clearly better on DailyDialog (+0 106 to +0 132; adjusted $p ~ = ~ . 0 1 4 4 )$ , whereas the transition baseline is stronger on MELD (−0 032 to −0 064), with four model diferences surviving correction. IEMOCAP is efectively tied and EmoryNLP is directionally favorable to the baseline. Thus, the transition matrix is neither a universal lower bound nor a straw baseline: its strength depends on whether current emotion is known and on corpus-specific transition structure.

Tables VII and VIII average three independently trained seeds; inference uses all three raw score vectors rather than a single representative run. On DailyDialog every model has a positive delta in every seed, the 95% hierarchical intervals exclude zero, and all paired permutation tests remain significant after the declared joint correction.

TABLE III. Speaker relation and target composition on the four unique test corpora. Same-next is $P ( s _ { n + 1 } = s _ { n } )$ . Self-shift uses the next-ownutterance target in Eq. (3); gap is the mean number of turns from n to g(n).
<table><tr><td>Corpus</td><td>Immediate N</td><td>Same-next</td><td>Shift|same</td><td>Shift|switch</td><td>Self N</td><td>Self shift</td><td>Mean gap</td></tr><tr><td>IEMOCAP</td><td>1,592</td><td>0.278</td><td>0.222</td><td>0.509</td><td>1,561</td><td>0.263</td><td>1.98</td></tr><tr><td>MELD</td><td>2,330</td><td>0.242</td><td>0.423</td><td>0.631</td><td>1,864</td><td>0.538</td><td>2.21</td></tr><tr><td>EmoryNLP</td><td>905</td><td>0.091</td><td>0.744</td><td>0.719</td><td>739</td><td>0.673</td><td>2.51</td></tr><tr><td>DailyDialog</td><td>6,740</td><td>0.000</td><td>一</td><td>0.198</td><td>5,740</td><td>0.180</td><td>2.00</td></tr></table>

![](images/913f6098424d1ce273ab9638fc710f188610f985d414e073104ee1c944541e66.jpg)

![](images/56312d2bb5034a0fbda0b51501b4df9bb6ffedbe2efe543409cfceac6129e244.jpg)  
Figure 1. Speaker composition and emotion-shift rate at adjacent test decision points. Filled bars denote same-speaker transitions; hatched bars denote speaker switches. DailyDialog has no adjacent same-speaker decisions. Exact denominators are given in Table III.

TABLE IV. Train-fold transition baselines under the two targets (test AUC). These are label-only diagnostics, not learned-model results.
<table><tr><td></td><td colspan="2">Immediate</td><td colspan="2">Self-shift</td></tr><tr><td>Corpus</td><td>Global</td><td>Speaker</td><td>Global</td><td>Speaker</td></tr><tr><td>IEMOCAP</td><td>0.642</td><td>0.642</td><td>0.586</td><td>0.586</td></tr><tr><td>MELD</td><td>0.676</td><td>0.676</td><td>0.663</td><td>0.663</td></tr><tr><td>EmoryNLP</td><td>0.608</td><td>0.608</td><td>0.592</td><td>0.592</td></tr><tr><td>DailyDialog</td><td>0.691</td><td>0.691</td><td>0.657</td><td>0.657</td></tr></table>

TABLE V. Self-shift composition by forecast gap. Cells show test N / shift rate; – indicates the released test split contains no decision at that gap.
<table><tr><td>Corpus</td><td>Gap 1</td><td>Gap 2</td><td>Gap ≥ 3</td></tr><tr><td>IEMOCAP</td><td>442 / .222</td><td>864 / .234</td><td>255 / .431</td></tr><tr><td>MELD</td><td>565 / .423</td><td>961 / .580</td><td>338 / .612</td></tr><tr><td>EmoryNLP</td><td>82 / .744</td><td>464 / .619</td><td>193 / .772</td></tr><tr><td>DailyDialog</td><td>1</td><td>5,740 / .180</td><td>1</td></tr></table>

Independent utterance-local features. Table IX removes dependence on released deep embeddings. TF–IDF vocabulary and IDF, followed by rank-128 truncated SVD, are fitted only on training utterances. The local-feature GRU remains substantially above its matched predictedlabel transition baseline on IEMOCAP and DailyDialog and directionally above it on EmoryNLP; MELD is tied. Crucially, the DailyDialog advantage is unchanged in magnitude (+ 164 versus + 166 with RoBERTa), so its headline improvement cannot be attributed solely to unobserved context in an upstream encoder.

Stronger predicted-label baseline. A posteriormarginalized control computes P(shift $\textstyle { \left| \begin{array} { l } { x _ { n } } \end{array} \right) } = \sum _ { k } P ( y _ { n } = k \ |$ $x _ { n } ) P ( y _ { n + 1 } \neq k \ \mid \ y _ { n } = k$ speaker) instead of committing to the classifier’s single most likely label. Marginalization is generally, though not uniformly, stronger: duplicatefree DailyDialog rises from .580 to .662 and MELD-mm from .584 to .605, while IEMOCAP dips slightly (.576 to .569). Every DailyDialog and MELD-mm model gain nevertheless remains corrected-significant in a separately declared 36-comparison Holm family (+ 078 to + 085 and + 111 to + 118, respectively); MELD’s smaller gains and EmoryNLP’s vanish. The control sharpens rather than overturns the conditional conclusion.

Same-architecture dose response (Figure 2). A singlelayer Transformer holds parameters, positional encoding, pooling, optimizer, and checkpoint rule fixed; only its attention mask changes so that position n sees k ∈ {0 1 2 4 all} future utterances. A single layer is required for k to mean what the sweep claims, since a stacked multi-layer encoder with the same window mask gives an efective receptive field of ∼Lk for L layers rather than k. The response is not monotone: MELD rises from .620 at k = 0 to .685 at k = 1, then falls to .651/.637/.632; duplicate-free DailyDialog similarly rises from .736 to .780 and falls to .774/.758/.751. The other four configurations vary by at most .022 without a consistent direction. Exposure to the target utterance can materially inflate performance, but additional future context need not add signal and may dilute optimization.

TABLE VI. Complete next-own-utterance self-shift ROC–AUC (three-seed mean). Trans-Pred is information-matched. DailyDialog gains all survive the separate 24-comparison Holm correction $( p _ { \mathrm { a d j } } = . 0 0 4 8 ) .$
<table><tr><td>Corpus</td><td>Trans-Pred</td><td>GRU</td><td>DRNN</td><td>DGCN</td><td>DAG-ERC</td><td>PEC</td><td>PseudoFut</td></tr><tr><td>IEMOCAP</td><td>.541</td><td>.609</td><td>.602</td><td>.597</td><td>.598</td><td>.593</td><td>.605</td></tr><tr><td>MELD</td><td>.584</td><td>.593</td><td>.600</td><td>.608</td><td>.610</td><td>.593</td><td>.593</td></tr><tr><td>EmoryNLP</td><td>.518</td><td>.509</td><td>.503</td><td>.514</td><td>.515</td><td>.501</td><td>.523</td></tr><tr><td>DailyDialog</td><td>.577</td><td>.699</td><td>.706</td><td>.698</td><td>.691</td><td>.695</td><td>.693</td></tr></table>

TABLE VII. Deployable shift ROC–AUC (three-seed mean). Trans-Pred uses current-emotion predictions from a train-only linear classifier; learned models receive no gold emotion. Bold values exceed the matched baseline. DailyDialog is duplicate-free; MM rows are feature-source sensitivity analyses.
<table><tr><td>Dataset</td><td>Trans-Pred</td><td>GRU</td><td>DRNN</td><td>DGCN</td><td>DAG-ERC</td><td>PEC</td><td>PseudoFut</td></tr><tr><td>IEMOCAP</td><td>.576</td><td>.647</td><td>.627</td><td>.609</td><td>.607</td><td>.632</td><td>.631</td></tr><tr><td>MELD</td><td>.588</td><td>.621</td><td>.629</td><td>.623</td><td>.608</td><td>.620</td><td>.600</td></tr><tr><td>EmoryNLP</td><td>.501</td><td>.521</td><td>.533</td><td>.536</td><td>.554</td><td>.539</td><td>.542</td></tr><tr><td>DailyDialog</td><td>.580</td><td>.746</td><td>.744</td><td>.748</td><td>.744</td><td>.743</td><td>.741</td></tr><tr><td>IEMOCAP-mm</td><td>.566</td><td>.560</td><td>.598</td><td>.593</td><td>.559</td><td>.581</td><td>.578</td></tr><tr><td>MELD-mm</td><td>.584</td><td>.723</td><td>.716</td><td>.724</td><td>.720</td><td>.720</td><td>.723</td></tr></table>

TABLE VIII. Oracle information-matched shift ROC–AUC (three-seed mean). Both the transition baseline and every learned model receive gold current emotion. Bold values exceed Trans-Gold.
<table><tr><td>Dataset</td><td>Trans-Gold</td><td>GRU</td><td>DRNN</td><td>DGCN</td><td>DAG-ERC</td><td>PEC</td><td>PseudoFut</td></tr><tr><td>IEMOCAP</td><td>.642</td><td>.649</td><td>.615</td><td>.617</td><td>.616</td><td>.631</td><td>.610</td></tr><tr><td>MELD</td><td>.676</td><td>.630</td><td>.629</td><td>.628</td><td>.644</td><td>.632</td><td>.612</td></tr><tr><td>EmoryNLP</td><td>.608</td><td>.553</td><td>.561</td><td>.561</td><td>.563</td><td>.591</td><td>.525</td></tr><tr><td>DailyDialog</td><td>.694</td><td>.808</td><td>.820</td><td>.826</td><td>.822</td><td>.804</td><td>.800</td></tr><tr><td>IEMOCAP-mm</td><td>.642</td><td>.566</td><td>.590</td><td>.598</td><td>.577</td><td>.594</td><td>.569</td></tr><tr><td>MELD-mm</td><td>.676</td><td>.802</td><td>.791</td><td>.805</td><td>.807</td><td>.798</td><td>.798</td></tr></table>

TABLE IX. Strict utterance-local feature control (three-seed ROC– AUC). Both the GRU features and the baseline’s current-emotion classifier are refit from the same train-only TF–IDF/SVD rows.
<table><tr><td>Corpus</td><td>GRU</td><td>Trans-Pred</td><td>∆ [95% CI]</td></tr><tr><td>IEMOCAP</td><td>.642</td><td>.552</td><td>+.089 [.010,.154]</td></tr><tr><td>MELD</td><td>.539</td><td>.538</td><td>+.001 [-.027,.028]</td></tr><tr><td>EmoryNLP</td><td>.555</td><td>.494</td><td>+.061 [.014,.112]</td></tr><tr><td>DailyDialog</td><td>.673</td><td>.509</td><td>+.164 [.141,.185]</td></tr></table>

TABLE X. Exact train-fold dialogue duplicates. A duplicate has an identical full ordered text sequence, not merely an overlapping utterance.
<table><tr><td colspan="3">Test</td><td colspan="2">Validation</td></tr><tr><td>Corpus</td><td>N</td><td>Dup.</td><td>N</td><td>Dup.</td></tr><tr><td>IEMOCAP</td><td>31</td><td>0</td><td>12</td><td>0</td></tr><tr><td>MELD</td><td>280</td><td>1</td><td>114</td><td>0</td></tr><tr><td>EmoryNLP</td><td>79</td><td>0</td><td>89</td><td>0</td></tr><tr><td>DailyDialog</td><td>1,000</td><td>134</td><td>1,000</td><td>101</td></tr></table>

## C. Split and Leakage Robustness

We hashed each dialogue by its complete ordered utterance text and compared validation/test hashes against the training fold (Table X). IEMOCAP and EmoryNLP contain no exact cross-fold matches; MELD contains one duplicate test dialogue; DailyDialog contains 134/1,000 duplicate test dialogues and 101/1,000 duplicate validation dialogues. We remove every matching DailyDialog validation/test dialogue before any configuration search, checkpoint selection, baseline fitting, or reported evaluation; test dialogues fall from 1,000 to 866 and validation dialogues from 1,000 to 899. All DailyDialog results in this paper use this same clean split.

## 6. Findings and Discussion

The revised evidence supports six bounded findings: (1) information matching is consequential – no-label models consistently exceed a baseline fed noisy train-only label predictions in point estimate, whereas a gold-label transition matrix is competitive or stronger on several corpora; (2) DailyDialog is the most stable positive case, with every tested model beating the hard predicted-label, posteriormarginalized, and gold-label baselines after duplicate removal, across every seed, with corrected-significant hierarchical inference; (3) MELD is conditional – learned models improve on the deployable baseline but most lose to the oracle baseline, so the current-state observation, not only architecture, changes the answer; (4) ranking and probability quality diverge, since PR–AUC corroborates DailyDialog discrimination while Brier/ECE show that learned scores require calibration before risk interpretation;

![](images/92701be5e7db9c94dc5a411c838d45179d0a22701f871e3a2573f949770059a6.jpg)  
Figure 2. Future-access dose response with one fixed Transformer architecture. Error bars show three-seed standard deviation. MELD and DailyDialog peak when the target utterance alone becomes visible; none of the six configurations supports a general monotonic-more-future claim.

(5) feature provenance remains part of the estimand, as the independently fitted local-feature control preserves large DailyDialog and IEMOCAP gains but not the MELD gain; and (6) adjacent-turn shift is predominantly cross-speaker in all four corpora, so interaction-level next-turn shift and a person’s next-own-utterance self-shift are distinct targets evaluated separately here.

These findings do not establish that one architecture class is generally superior or irrelevant. They establish a narrower measurement result: the sign and magnitude of model-versus-transition diferences depend on the information supplied to both sides, and conclusions that omit this factor can reverse. The multimodal rows reinforce the need for restraint – their point estimates can be large, but the complete upstream producers are unavailable, so they are useful sensitivity evidence rather than a basis for claiming a modality is intrinsically better.

## A. Operational Implications

Temporal validity is necessary but not suficient for deployment. A high-AUC alert can still be poorly calibrated, too late for intervention, or unfairly concentrated on particular speakers. The commitment point must correspond to a real action window: after the current turn is available but before any acoustic, lexical, or metadata trace of the next turn enters the system. Thresholds should follow intervention costs rather than F1 alone: in de-escalation, a false negative may miss harm while repeated false positives create alert fatigue. This distinction is visible in the strongest ranking result: duplicate-free DailyDialog DialogueGCN reaches ROC–AUC .748 and PR–AUC .441, but its uncalibrated

Brier score/ECE are .188/.224, versus .153/.052 for the weaker predicted-label transition baseline. Post-hoc calibration must therefore be fitted on held-out deployment-like data before probabilities are interpreted as risks. This work audits measurement; it does not establish that an automated emotional intervention is safe or appropriate.

## B. Threats to Validity and Limitations

Construct and internal validity. Emotion annotations are discrete expressions, not direct measures of internal afect. Causality tests cover model forward paths, masks, and provided feature arrays, but cannot retroactively certify every upstream feature-extraction step; the train-only TF– IDF/SVD control reduces, but cannot remove, this limitation.

Split validity. We remove exact train–test dialogue duplication from every DailyDialog run, but not subtler writing-style leakage. Only IEMOCAP’s session split provides an unseen-actor test; speaker conditioning elsewhere means local role conditioning, not person-level generalization.

Statistical validity. Three seeds characterize optimization variability but do not make the six configurations independent, since two corpora recur under diferent feature sources; the joint Holm correction is intentionally conservative.

Model scope. Forecasting-specific systems are strategylevel re-implementations because original code was unavailable; the common four-configuration search favors comparability over per-model best-case tuning.

Information matching is imperfect in the oracle direction. In the oracle diagnostic, the gold one-hot label is concatenated to the per-turn feature vector at every position $t ~ \leq ~ n ,$ , so a sequential model’s representation at n is conditioned on the entire observed label history, not on $y _ { n }$ alone; the transition baseline conditions only on the single most recent label. “Information-matched” therefore means matched at current-emotion availability, not in the total label information reaching each side, and the oracle gap likely understates the transition baseline’s competitiveness under a strict single-label match. The symmetric deployable-side concern – that a hard current-label prediction discards classifier uncertainty a stronger baseline could exploit – is addressed by the posterior-marginalized control above.

## 7. Minimal Reporting Checklist

We distill a standalone checklist for future EFCforecasting papers: (1) state the commitment point and verify no feature crosses it; (2) name the semantic target – interaction-level next turn or same-person next own turn – and report speaker-relation composition; (3) verify causality directly (e.g. a future-perturbation test), not by architecture claim alone; (4) trace feature provenance upstream of the classifier; (5) match current-label information between learned models and train-only transition baselines; (6) report ROC–AUC, PR–AUC, calibration, balanced accuracy, and prediction rate rather than F1 alone; (7) use seed–dialogueaware paired inference corrected jointly across the declared comparison family; (8) label leaky ablations and exclude them from headline comparisons; (9) check duplicate dialogues and global speaker overlap; and (10) release masks, split identifiers, raw seed-level predictions, and audit logs.

## 8. Conclusion

Information-matched evaluation changes the interpretation of conversational emotion-shift forecasting. Models that receive no gold current label can meaningfully outperform a transition baseline driven by train-only predicted labels; on duplicate-free DailyDialog this conclusion is large, seed-consistent, and familywise significant. A transition matrix supplied with gold current emotion is nevertheless competitive on IEMOCAP and stronger on MELD, showing why privileged-label diagnostics must not be presented as deployable comparisons. Independent local features preserve the DailyDialog result, while calibration metrics warn that strong ranking is not yet reliable risk estimation. Finally, adjacent interaction shift and sameperson self-shift are both legitimate but diferent targets requiring explicit horizons and matched baselines. The general lesson is methodological rather than architectural: state the temporal, information, provenance, and semantic contracts before interpreting a model comparison [32], [33], [34]. Code, raw seed predictions, tests, and reporting artifacts supporting reproducible auditing are available from the corresponding author upon request.

## Acknowledgment

The authors used a large language model assistant to help with English-language editing, LaTeX formatting, code review, and the documentation and presentation of supplementary analyses. All experimental designs, executions, interpretations, and conclusions were independently reviewed and verified by the authors, who assume full responsibility for the submitted work.

## References

[1] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, “MELD: A multimodal multi-party dataset for emotion recognition in conversations,” in Proc. ACL, 2019, pp. 527–536.

[2] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “IEMOCAP: Interactive emotional dyadic motion capture database,” Language Resources and Evaluation, vol. 42, no. 4, pp. 335–359, 2008.

[3] S. M. Zahiri and J. D. Choi, “Emotion detection on TV show transcripts with sequence-based convolutional neural networks,” in Proc. AAAI Workshop, 2018.

[4] Y. Li, H. Su, X. Shen, W. Li, Z. Cao, and S. Niu, “DailyDialog: A manually labelled multi-turn dialogue dataset,” in Proc. IJCNLP, 2017, pp. 986–995.

[5] D. Ghosal, N. Majumder, A. Gelbukh, R. Mihalcea, and S. Poria, “COSMIC: Commonsense knowledge for emotion identification in conversations,” in Findings of EMNLP, 2020, pp. 2470–2481.

[6] D. Hu, X. Hou, L. Wei, L. Jiang, and Y. Mo, “MM-DFN: Multimodal dynamic fusion network for emotion recognition in conversations,” in Proc. ICASSP, 2022.

[7] N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, A. Gelbukh, and E. Cambria, “DialogueRNN: An attentive RNN for emotion detection in conversations,” in Proc. AAAI, 2019, pp. 6818–6825.

[8] D. Ghosal, N. Majumder, S. Poria, N. Chhaya, and A. Gelbukh, “DialogueGCN: A graph convolutional neural network for emotion recognition in conversation,” in Proc. EMNLP-IJCNLP, 2019, pp. 154–164.

[9] W. Shen, S. Wu, Y. Yang, and X. Quan, “Directed acyclic graph network for conversational emotion recognition,” in Proc. ACL-IJCNLP, 2021.

[10] W. Ai, Y. Shou, T. Meng, and K. Li, “DER-GCN: Dialogue and event relation-aware graph convolutional neural network for multimodal dialogue emotion recognition,” arXiv:2312.10579, 2023.

[11] P. Zhong, D. Wang, and C. Miao, “Knowledge-enriched transformer for emotion detection in textual conversations,” in Proc. EMNLP-IJCNLP, 2019, pp. 165–176.

[12] D. Hu, L. Wei, and X. Huai, “DialogueCRN: Contextual reasoning networks for emotion recognition in conversations,” in Proc. ACL-IJCNLP, 2021.

[13] D. Hazarika, S. Poria, R. Mihalcea, E. Cambria, and R. Zimmermann, “ICON: Interactive conversational memory network for multimodal emotion detection,” in Proc. EMNLP, 2018, pp. 2594– 2604.

[14] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal Transformer for unaligned multimodal language sequences,” in Proc. ACL, 2019, pp. 6558–6569.

[15] J. Hu, Y. Liu, J. Zhao, and Q. Jin, “MMGCN: Multimodal fusion via deep graph convolution network for emotion recognition in conversation,” in Proc. ACL-IJCNLP, 2021, pp. 5666–5675.

[16] D. Hazarika, R. Zimmermann, and S. Poria, “MISA: Modalityinvariant and -specific representations for multimodal sentiment analysis,” in Proc. ACM MM, 2020, pp. 1122–1131.

[17] J. Li, X. Wang, G. Lv, and Z. Zeng, “GA2MIF: Graph and attention based two-stage multi-source information fusion for conversational emotion detection,” IEEE Trans. Afective Computing, vol. 15, no. 1, pp. 130–143, 2023.

[18] T. Meng, F. Zhang, Y. Shou, H. Shao, W. Ai, and K. Li, “Masked graph learning with recurrent alignment for multimodal emotion recognition in conversation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 4298–4312, 2024.

[19] H. Wang, X. Mai, Z. Tao, J. Lin, X. Tong, I. Pan, S. Yan, Y. Wang, and S. Gao, “Hi-EF: Benchmarking emotion forecasting in humaninteraction,” arXiv:2407.16406, 2024.

[20] E. Altarawneh, A. Agrawal, M. Jenkin, and M. Papagelis, “Predicting evoked emotions in conversations,” arXiv:2401.00383, 2023.

[21] X. Shi, X. Li, and T. Toda, “Emotion awareness in multi-utterance turn for improving emotion prediction in multi-speaker conversation,” in Proc. Interspeech, 2023, pp. 765–769.

[22] H. Shi, Z. Liang, and J. Yu, “Emotional cues extraction and fusion for multi-modal emotion prediction and recognition in conversation,” in Proc. Interspeech, 2024, pp. 4074–4078.

[23] X. Ju, D. Zhang, S. Zhu, J. Li, S. Li, and G. Zhou, “Real-time emotion pre-recognition in conversations with contrastive multimodal dialogue pre-training,” in Proc. ACM CIKM, 2023.

[24] Y. Xie, Y. Liu, C. Sun, et al., “Pseudo-utterance-guided contrastive network for emotion forecasting in conversations,” Expert Systems with Applications, vol. 279, p. 127382, 2025.

[25] X. Lu, W. Zhao, Y. Zhao, B. Qin, Z. Zhang, and J. Wen, “A topic-enhanced approach for emotion distribution forecasting in conversations,” in Proc. IEEE ICASSP, 2023.

[26] A. Debsharma, B. Jagyasi, S. Sen, P. Pandey, D. Dovari, V. C. Yuvaraj, R. Parida, and G. Contractor, “ERFC: Happy customers with emotion recognition and forecasting in conversation in call centers,” arXiv:2509.18175, 2025.

[27] Q. Gao, B. Cao, X. Guan, T. Gu, X. Bao, J. Wu, B. Liu, and J. Cao, “Emotion recognition in conversations with emotion shift detection based on multi-task learning,” Knowledge-Based Systems, vol. 248, p. 108861, 2022.

[28] J. Li, Y. Liu, X. Wang, and Z. Zeng, “CFN-ESA: A cross-modal fusion network with emotion-shift awareness for dialogue emotion recognition,” IEEE Trans. Afective Computing, vol. 15, no. 4, pp. 1919–1933, 2024.

[29] H. Nirujan and Y. H. P. P. Priyadarshana, “EmoShiftNet: A shiftaware multi-task learning framework with fusion strategies for emotion recognition in multi-party conversations,” Frontiers in Artificial Intelligence, vol. 8, p. 1618698, 2025.

[30] S. Kumar, S. Dudeja, M. S. Akhtar, and T. Chakraborty, “Emotion flip reasoning in multi-party conversations,” IEEE Trans. Artificial Intelligence, arXiv:2306.13959, 2023.

[31] S. Kumar, M. S. Akhtar, E. Cambria, and T. Chakraborty, “SemEval-2024 Task 10: Emotion discovery and reasoning its flip in conversation (EDiReF),” in Proc. 18th SemEval, 2024, pp. 1933–1946.

[32] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, p. 100804, 2023.

[33] M. Ferrari Dacrema, P. Cremonesi, and D. Jannach, “Are we really making much progress? A worrying analysis of recent neural recommendation approaches,” in Proc. ACM RecSys, 2019, pp. 101– 109.

[34] J. Lin, “The neural hype and comparisons against weak baselines,” ACM SIGIR Forum, vol. 52, no. 2, pp. 40–51, 2019.

[35] T. Saito and M. Rehmsmeier, “The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” PLoS ONE, vol. 10, no. 3, p. e0118432, 2015.

[36] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th Int. Conf. Machine Learning, 2017, pp. 1321–1330.

[37] V. Saravanan, G. J. Berman, and S. J. Sober, “Application of the hierarchical bootstrap to multi-level data in neuroscience,” Neuron, Behavior, Data Analysis, and Theory, vol. 3, no. 5, 2020.