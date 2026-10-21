# Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora

Wonsup Shin Flitto wonsup.shin@flitto.com

Jingu Kim   
Flitto   
jin@flitto.com

## Abstract

When unique pretraining data is scarce, models see the same tokens many times and are thought to memorize. Random-token replacement, an input-corruption augmentation, is one remedy reported to lower validation loss, but a lower loss alone does not reveal which source of error it reduces. On a controlled source whose entropy rate is known in closed form, we show that replacement shrinks predictions toward the corpus average. We also introduce a decomposition that separates validation loss into approximation, estimation, and memorization terms. This shrinkage view predicts that replacement’s bias cost needs no memorization but its benefit does. Memorization raises validation loss after its minimum, so we report the effect at two points, the minimum validation loss and the end of a fixed budget. Where replacement helps, it reduces this rise by about 60%. At the end of the budget, it turns from harmful to helpful as the controlled source becomes less predictable; at the minimum, it helps at no predictability level under the fixed recipe. On Korean and Japanese web text, replacement removes about a third of the loss rise after the minimum by the end of the budget, but its effect at the minimum is much smaller. Across ten real corpora, replacement lowers the final loss on nine. It harms only the census corpus, program-generated JSON that is the most redundant of the ten and much more redundant than any of 64 further prose and code corpora we measure.

## 1 Introduction

Pretraining a language model when the supply of unique text is limited (Villalobos et al., 2024) forces repeated passes over the same tokens. Beyond roughly four epochs the marginal value of repetition falls sharply, and past ∼16 epochs additional repetition adds essentially nothing (Muennighoff et al., 2023). This is usually attributed to memorization, as the model overfits the training data.

A widely used remedy is training-time inputcorruption augmentation. At each step it replaces a fraction of input tokens with random tokens or masks them (Chen et al., 2026; Xu et al., 2026). Yet existing studies leave three questions. First, validation loss alone cannot tell memorization from other sources of error, so an improvement in it does not show whether the corruption acts on memorization or on something else. Second, it is unclear whether the cost and benefit of the augmentation appear at the same point of training, and so whether the effect depends on when training stops. Third, the evidence comes almost entirely from English prose and code (Chen et al., 2026; Xu et al., 2026), leaving unclear how far it carries to other languages and other kinds of text. We address all three for random-token replacement.

Replacement shrinks the model’s predictions toward the corpus average. Shrinkage reduces variance at the cost of a bias (Bishop, 1995; Xie et al., 2017). Under repetition, the natural reading is that this variance comes from memorization. Where text is predictable, the next token is nearly determined, so the shrinkage should cost more and help less (Figure 1(a)). From this shrinkage view we predict that the cost appears immediately, while the benefit emerges only after the model has memorized finite data, and that the cost is larger on more predictable text.

We derive this view on an order-1 Markov source, whose entropy rate is known in closed form (§4). On this source we introduce a decomposition of validation loss into approximation, estimation, and memorization terms. It uses the entropy rate as the ground-truth lower bound on loss, and it identifies which term produces the benefit or harm (§5).

We then test the range of the effect on ten real corpora while holding the model and training recipe fixed.<sup>1</sup> The corpora span four respects: language, modality, structure, and origin (§3). We repeat the core comparison at two model scales.

![](images/9b5b86f7f9fe341b52fd50a15fe64deb06c042fb2df076f767939ca86bbe0193.jpg)

![](images/81f545063caa1d5029af419ab6c8cadef7549854db7b140e4255942e9e82ab0c.jpg)

![](images/9cd505d91c95d252d4893c1fd103b541c5d69e6bef60722842009b20d96e8d61.jpg)  
Figure 1: (a) A schematic of the benefit and the bias as the corruption rate p rises; the bias climbs faster on a more predictable corpus. (b, c) Measured net effect of replacement, positive when it helps. (b) The Markov source at temperatures 0.2 to 2.0, written t02 to t20, under a constant learning rate. (c) Web text under a learning rate decayed within each budget (Table 3). Error bars give the s.d. over three seeds.

On our main training budget (§3), we report each effect at two points of training, since the two can disagree. The minimum loss is the lowest validation loss a run reaches, the point early stopping selects, and the final loss is the validation loss at the end of a fixed budget. Our main budget runs far past the minimum as a deliberate stress test. Runs are often not stopped at the minimum loss, since they follow a fixed token budget (Hoffmann et al., 2022; Muennighoff et al., 2023) and continued training can improve downstream performance after the pretraining loss stops falling (Liu et al., 2023).

## Contributions.

Our decomposition of validation loss shows what an augmentation removes. On a source of known entropy rate, we derive the shrinkage view of replacement (§4) and decompose validation loss into approximation, estimation, and memorization terms (§5). The decomposition needs only validation losses on such a source, and nothing in it is specific to replacement, so it applies to any augmentation.

Replacement helps by reducing the loss rise after the minimum, so its value depends on when training stops. Where replacement helps, it reduces the memorization term by ∼60% while the other two terms offset each other (§5). Its bias cost is present even when nothing is memorized, but the benefit follows only after memorization accumulates. The effect on the controlled source therefore turns from harm to benefit on final loss as predictability falls, while replacement lowers the minimum loss at no predictability level under the fixed recipe (Figure 1(b)). On Korean and Japanese web text, replacement removes a third or more of the overfitting gap, with or without learning-rate decay, but its effect at the minimum is much smaller (Figure 1(c), §6).

Replacement harms only an extremely redundant corpus in our tests. Beyond the ten corpora we train on, we measure a compressionbased redundancy, the gzip ratio, on 64 corpora that are not program-generated. On final loss, replacement helps every trained corpus that is not program-generated and harms only the census corpus, census records serialized to JSON. Mutual information does not set this corpus apart from the helped ones, but by gzip it is more redundant than all 64 surveyed corpora (§7).

## 2 Related Work

Training-time augmentation. Corruption has a long history as a pretraining objective (Devlin et al., 2019; Lewis et al., 2020). We use it differently. The target stays the clean next token and the corruption acts on the input alone. Recent dataconstrained studies use masking, replacement and infilling this way, as regularizers under repeated causal-LM training (Chen et al., 2026; Xu et al., 2026). Chen et al. (2026) report that replacement is the strongest of the three on English.

Measuring what a model has learned. Jumelet and Zuidema (2023) train on artificial languagelike data from a probabilistic grammar, where the true distribution gives an exact lower bound on loss. We use the same access for a different end, splitting loss into terms rather than scoring how close a model gets to the bound. Bahri et al. (2024) also decompose loss, but to explain how it scales with model and data size, whereas ours splits loss at one model size. Our memorization term, a rise in held-out loss after a run’s minimum, is narrower than the exact reproduction of training strings, which Tirumala et al. (2022) show can begin before a model overfits.

Noise as regularization. That input noise acts as a smoothing regularizer is classical. Bishop (1995) shows it is equivalent to Tikhonov regularization, and Xie et al. (2017) give a closedform view of token noising as interpolation toward lower-order distributions. Our mechanism (§4) is an instance of this view. Augmentation is also known to help in moderation and hurt when it erases structure (Lin et al., 2024; Nguyen et al., 2025). We locate the point where the predicted bias starts to outweigh the benefit, on a source of known entropy rate.

Early stopping and fixed budgets. Studies of repeated data read loss at different points of training. Kaplan et al. (2020) stop early, while Muennighoff et al. (2023) deliberately do not, so that overfitting stays visible. The two conventions can disagree, since validation loss need not fall monotonically with training time (Nakkiran et al., 2020). Muennighoff et al. (2023) report that loss rises again with more epochs, expect regularization to prevent that rise, and leave the case outside their scaling law. A decay schedule should also be matched to the training budget (Hoffmann et al., 2022), so we test that expectation under a matched schedule as well as a constant rate, and report both points for every experiment on the main budget.

Non-English, speech, and tabular corpora. Seto et al. (2025) study data-constrained pretraining in a targeted non-English language, and using ASR transcripts for pretraining has precedent (Pelloin et al., 2022; Sundararaman et al., 2021), where spoken and written text differ systematically in register (Biber, 1988). Transformers have also been trained on tabular data with one embedding per field (Padhi et al., 2021), whereas we serialize tables to plain JSON text.

Predicting behavior from corpus statistics. The gzip ratio predicts data-dependent scaling (Pandey, 2024), redundancy predicts memorization (Zhang et al., 2025), and Jiang et al. (2026) predict knowledge retention before training from a size-dependent mutual information score. Language modeling and compression are nearequivalent (Delétang et al., 2024), so such a scalar partly restates model loss. We ask instead whether such scalars predict when replacement helps (§7).

## 3 Experimental Setup

Corpora. We use ten real corpora and one controlled source. Each real corpus is labeled natural language or structured text. Independently, any corpus, real or controlled, is labeled programgenerated when a schema, a template, or a random process produced it.

The natural-language corpora are Korean and Japanese web text from FineWeb-2 (Penedo et al., 2025) and Korean speech transcripts from the open YODAS corpus (Li et al., 2023). Each of the three has exactly 7.4M training tokens. The structured corpora are source code from The Stack (Kocetkov et al., 2023), chess move sequences (Karvonen, 2024), SQL from SQL-create-context (bmc2, 2023) in both a combined context-plus-query form (SQL-comb) and a context-only form (SQLctx), a system-log corpus from LogHub (Zhu et al., 2023), and two tables serialized to JSON. One is a census dataset (Kohavi, 1996), highly templated and repetitive by construction, and the other the public road-safety accident table (Grinsztajn et al., 2022). Code, the system-log corpus and the roadsafety corpus also hold 7.4M tokens; chess, the two SQL forms and the census corpus range from 2.3M to 5.5M, since most of these public datasets are small.

The controlled source is an order-1 Markov source whose true entropy rate H is known in closed form from the transition matrix (Cover and Thomas, 2006), sampled at temperatures from 0.2 to 2.0; the temperature sets H. Each temperature is a separate source. The controlled runs use 2M training tokens and 8,000 steps.

The program-generated label covers the census corpus, the two SQL forms, the Markov source, and the road-safety corpus and system-log corpus. We use the labels to describe corpora, not to explain the sign of the effect.

Model and recipe. All runs train a decoder-only Transformer from scratch for 10,000 steps, our main training budget, under one fixed recipe with a constant learning rate (App. A). That is roughly 22 epochs on each 7.4M-token corpus, well past the point where repetition stops helping (App. C.6). Every baseline reaches its minimum loss within its first eight epochs (Table 1), so each corpus, large or small, spends most of the budget past that point. The main model has 336M parameters and the scale check a 134M variant.

<table><tr><td>Corpus</td><td>Gain (%)</td><td> $\Delta L _ { \mathrm { f i n a l } }$ </td><td> $L _ { \mathrm { m i n } }$ </td><td> $\Delta L _ { \mathrm { m i n } }$ </td><td>epoch at min</td></tr><tr><td>KO web</td><td> $3 5 . 2 { \pm } 1 . 5 $ </td><td> $+ 0 . 9 3 { \pm } 0 . 1 2 ^ { \ast }$ </td><td> $4 . 9 1  4 . 8 5$ </td><td> $+ 0 . 0 6 { \pm } 0 . 0 8$ </td><td> $4 . 4  5 . 3$ </td></tr><tr><td>JA web</td><td> $3 7 . 5 { \pm } 1 . 7 $ </td><td> $+ 1 . 1 4 { \pm } 0 . 0 5 ^ { \ast }$ </td><td> $4 . 7 0  4 . 6 1$ </td><td> $+ 0 . 0 9 { \pm } 0 . 0 1 ^ { \ast }$ </td><td> $4 . 4  5 . 5$ </td></tr><tr><td>KO transcript</td><td> $4 0 . 1 { \pm } 0 . 7 $ </td><td> $+ 0 . 9 3 { \pm } 0 . 0 1 ^ { \ast }$ </td><td> $3 . 3 7  3 . 3 1$ </td><td> $+ 0 . 0 6 { \pm } 0 . 0 0 ^ { \ast }$ </td><td> $4 . 0  5 . 8$ </td></tr><tr><td>code</td><td> $2 6 . 3 { \pm } 1 . 4 $ </td><td> $+ 0 . 2 4 { \pm } 0 . 0 4 ^ { * }$ </td><td> $1 . 8 6  1 . 9 2$ </td><td> $- 0 . 0 6 { \pm } 0 . 0 3$ </td><td> $4 . 4  5 . 8$ </td></tr><tr><td>chess</td><td> $2 5 . 9 { \pm } 4 . 2 $ </td><td> $+ 0 . 2 9 { \pm } 0 . 0 4 ^ { * }$ </td><td> $0 . 8 4  0 . 8 1$ </td><td> $+ 0 . 0 3 { \pm } 0 . 0 1 ^ { \ast }$ </td><td> $3 . 8  7 . 6$ </td></tr><tr><td>road-safety (JSON)†</td><td> $4 1 . 9 { \pm } 1 . 3 $ </td><td> $+ 0 . 0 4 { \pm } 0 . 0 0 ^ { \ast }$ </td><td> $0 . 1 7  0 . 1 6$ </td><td> $+ 0 . 0 0 { \pm } 0 . 0 0$ </td><td> $6 . 0  7 . 7$ </td></tr><tr><td>SQL-comb</td><td> $- 4 . 4 { \pm } 9 . 4 $ </td><td> $+ 0 . 0 4 { \pm } 0 . 0 4$ </td><td> $1 . 4 4  1 . 3 8$ </td><td> $+ 0 . 0 6 { \pm } 0 . 0 5$ </td><td> $6 . 6  7 . 3$ </td></tr><tr><td>SQL-ctx</td><td> $- 1 5 . 2 { \pm } 1 6 . 4$ </td><td> $+ 0 . 0 5 { \pm } 0 . 0 9$ </td><td> $2 . 4 0  2 . 2 7$ </td><td> $+ 0 . 1 3 { \pm } 0 . 0 2 ^ { \ast }$ </td><td> $7 . 1  9 . 3$ </td></tr><tr><td>census (JSON)</td><td>2.0±22.9</td><td> $- 0 . 0 6 { \pm } 0 . 0 2 ^ { \ast }$ </td><td> $0 . 2 6  0 . 3 2$ </td><td> $- 0 . 0 6 { \pm } 0 . 0 2$ </td><td> $2 . 1  1 . 2$ </td></tr><tr><td> $\mathrm { \ l o g h u b ^ { \dagger } }$ </td><td>13.7±7.2</td><td> $+ 0 . 0 7 { \pm } 0 . 0 1$ </td><td> $0 . 9 3  0 . 9 1$ </td><td> $+ 0 . 0 2 { \pm } 0 . 0 2$ </td><td> $6 . 3  4 . 3$ </td></tr></table>

Table 1: Effect of replacement at 336M, mean±s.d. over three seeds. $\Delta$ is $L _ { \mathrm { N O N E } } - L _ { \mathrm { R E P L A C E } }$ in nats, positive when replacement helps; <sup>∗</sup> is $p < 0 . 0 5$ in a one-sample t-test. Gain (%) is the regularization gain of $\ S 3$ , and $L _ { \mathrm { m i n } }$ and its epoch read NONE→REPLACE. Below the line, Gain does not differ from zero at $p < 0 . 0 5$ . <sup>†</sup>Two seeds.

Input-corruption augmentation. We study random-token replacement, which independently replaces each input token with a uniformly random vocabulary token at rate $p ,$ leaving the prediction targets intact. Unless noted, all main experiments use $p { = } 0 . 1 5 ;$ we write REPLACE for this condition and NONE for the no-augmentation baseline with $\scriptstyle p = 0 .$ The corpus marginal, the average token distribution, does not enter the noise. It enters only when the model averages over the clean token that a corrupted context could have come from (§4).

Metrics. We report validation loss in nats throughout. Each corpus reserves a disjoint heldout validation slice, on which every loss is measured (App. A). We evaluate every 300 steps, or every 250 on the Markov source, and read two points from each run. The minimum loss $L _ { \mathrm { m i n } }$ is the lowest validation loss of the run, reported with the epoch at which it occurs. Early stopping with a patience of five evaluations selects this same point on 172 of the 174 runs whose minimum we report (App. C.4). The final loss $L _ { \mathrm { f i n a l } }$ is the last validation loss of the run. At either point, the effect of replacement is $\Delta = L _ { \mathrm { N O N E } } - L _ { \mathrm { R E P L A C E } }$ , so a positive sign means replacement helps. The overfitting gap of a run is $g = L _ { \mathrm { f i n a l } } - L _ { \mathrm { m i n } }$ , and the regularization gain of replacement is (g<sub>NONE</sub> − g<sub>REPLACE</sub>)/g<sub>NONE</sub>, the fraction of the overfitting gap that replacement removes. As redundancy measures we report the gzip compression ratio of the raw text, lower for more redundant text, and the bigram mutual information (MI) normalized by unigram entropy. Both vary with the writing system, so we compare corpora on them only within one writing system.

## 4 Shrinkage and Its Bias Cost

On an order-1 source, each token depends only on the token before it, its context, and replacement swaps the context for a random token with probability $p .$ The noise is uniform over the vocabulary, yet the model is pulled toward the corpus marginal. A model that sees a possibly corrupted context c˜ does best by averaging over the clean context that could have produced it,

$$
Q _ { \tilde { c } } = \alpha _ { \tilde { c } } P _ { \tilde { c } } + \left( 1 - \alpha _ { \tilde { c } } \right) \pi\tag{1}
$$

$$
\alpha _ { \tilde { c } } = \frac { \left( 1 - p \right) \pi ( \tilde { c } ) } { \left( 1 - p \right) \pi ( \tilde { c } ) + p q ( \tilde { c } ) }\tag{2}
$$

where $Q _ { \tilde { c } }$ is the best prediction after seeing c˜, $\alpha _ { \tilde { c } }$ the probability that c˜ is clean, $P _ { \tilde { c } }$ the true conditional distribution, $\pi$ the corpus marginal, and $q$ the noise distribution. The marginal appears in Eq. (1) for any q, because it comes from averaging the clean context under $\pi .$ The noise q only sets $\alpha _ { \tilde { c } }$ in Eq. (2). This averaging is the shrinkage view of input noising (Bishop, 1995; Xie et al., 2017). Shrinkage reduces variance at the cost of a bias. Under repetition we expect the variance it removes to come from memorization. By Eq. (1), the prediction misses the true conditional by $Q _ { { \tilde { c } } } - P _ { { \tilde { c } } } = ( 1 - \alpha _ { \tilde { c } } ) ( \pi - P _ { \tilde { c } } )$ This gap is the bias. It grows with how far the true conditional $P _ { \tilde { c } }$ departs from the marginal π, that is, with how predictable the next token is from its context, which the size of $\pi - P _ { \tilde { c } }$ measures. The other factor, $1 - \alpha _ { \tilde { c } } .$ sets how often the shrinkage lands on a clean context. It depends on whether the noise draws a token that occurs in the corpus. On the Markov source it almost always does, so shrinkage falls on every clean context. On the real corpora it often does not, so the corrupted position is recognizable and much less shrinkage falls on clean contexts (App. C.3). Turning to the benefit, it also varies with the source. The loss rise from memorization should grow with the entropy rate $H$ , so on a highly predictable source replacement removes little. As predictability rises, the benefit shrinks and the bias grows. The sign of the effect depends on which of the two is larger.

The two also act at different times. The bias is present from the start, while the variance reduction can matter only after the minimum loss, once the model starts to overfit. At the minimum the bias is therefore exposed and the net effect is small or negative; by the end of the budget the variance reduction dominates on all but the most predictable sources (Figure 1(c)).

## 5 The Benefit of Replacement Lies in the Memorization Term

§4 expects the variance that replacement removes to come from memorization. A lower validation loss alone cannot test this, because it does not show whether replacement helps the model learn the source or reduces the loss rise that comes with memorizing the training sample. We separate the two on the Markov source, whose entropy rate is known.

Starting from the entropy rate H, the lower bound on loss (Cover and Thomas, 2006), we write the final loss of a run as

$$
\begin{array} { r l } & { L _ { \mathrm { f i n a l } } = H + \underbrace { \left( L _ { \mathrm { a b u n d } } - H \right) } _ { \mathrm { A P P R O X } } } \\ & { \quad + \underbrace { \left( L _ { \mathrm { m i n } } - L _ { \mathrm { a b u n d } } \right) } _ { \mathrm { E S T } } + \underbrace { \left( L _ { \mathrm { f i n a l } } - L _ { \mathrm { m i n } } \right) } _ { \mathrm { M E M } } } \end{array}\tag{3}
$$

where $L _ { \mathrm { a b u n d } }$ is the loss that a model of the same capacity reaches with abundant data (App. A), and $L _ { \mathrm { m i n } }$ is the minimum loss that the same model reaches on the small corpus of §3. These three operational terms split the excess ${ \cal L } _ { \mathrm { f i n a l } } \ : - \ : H$ at these two reference losses. The APPROX term is the loss above H that the model keeps even with abundant data. The EST term is the further loss at the minimum because the data are limited. MEM equals the overfitting gap g of $\ S 3$ measured on this source. We attribute MEM to memorization, and the training loss supports this reading. Every run drives its training loss far below H, the lowest loss any model can reach on held-out sequences from the source (App. C.1). Replacement leaves the training loss about the same or lower and raises the share of 50-token training spans the model reproduces verbatim (Carlini et al., 2023), so it does not stop the model from memorizing but reduces the rise in held-out loss that accompanies it. The abundant-data run, otherwise identical, rises by at most 0.004 nats after its minimum (App. A), so the rise appears only when the data repeat. The bias of replacement needs no memorization, so it does not show in $g .$ Since nothing is memorized in the abundant-data run, the rise in APPROX under replacement, $\mathrm { A P P R O X } _ { \mathrm { R E P L A C E } } - \mathrm { A P P R O X } _ { \mathrm { N O N E } }$ measures this bias. It is 0.02 to 0.05 nats across the three sources (App. A), so without scarcity replacement only costs, though we do not rule out a smaller optimization effect. At the minimum of a data-constrained run, a change in EST adds to it.

![](images/e9e0b588239cbb34483c8b1331dade52d78af70646b3e71b3b75bbb02885c1c0.jpg)

![](images/40d3bad6d2c87d7de907330f40d2d332afaf7dc2dc53d63aef1bb9453ea6b00e.jpg)  
Figure 2: Decomposition of Eq. (3) on the Markov source. The number above each pair in (b) is $\Delta L _ { \mathrm { m i n } }$

Replacement targets MEM. Figure 2 shows the decomposition of Eq. (3) at three temperatures, 0.2, 0.5, and 2.0, written t02, t05, and t20 (App. B.1). In the helpful regime, t05 and t20, replacement reduces MEM by 58–62%. At both sources the minimum loss is unchanged, with the rise in APPROX offset by a fall in EST, so the whole benefit lies in MEM (Figure 2(b)). At the most predictable source t02, by contrast, replacement does not reduce MEM and instead raises the minimum loss by 0.12 nats. Both agree with $\ S 4$ which predicts that on a highly predictable source replacement removes little of the loss rise from memorization and its bias is large. With no reduction in MEM to offset this bias, replacement harms.

![](images/da70cc0a691d81c46319118edb75b54db917120cd58456e2e387d4bf6b84999e.jpg)  
Figure 3: Validation loss over training for one seed, REPLACE in red and NONE in blue, with a dot at each run’s minimum. Each panel has its own scale.

The benefit scales with the entropy rate. Varying the Markov temperature sweeps the entropy rate from near-deterministic to near-uniform. The net effect of replacement on final loss, $L _ { \mathrm { N O N E } } \mathrm { ~ - ~ }$ $L _ { \mathrm { R E P L A C E } }$ , increases overall along this axis (Figure 1(b), with per-seed values in App. B.4). At the most predictable source replacement is harmful; as H rises it becomes helpful and then strongly so, from −0.23 nats to +3.35 nats. A predictability threshold therefore separates help from harm on this source, as the shrinkage view of §4 predicts. At the minimum loss, by contrast, replacement helps at no temperature under the fixed recipe (Figure 1(b)), as §4 predicts for the point where the bias is exposed. A schedule sized to each of five budgets, from 4 to 66 epochs, improves the minimum by at most 0.023 nats (App. C.2).

## 6 Two Points of Training on Real Corpora

On the Markov source, replacement helps on final loss but not at the minimum loss (§5). We ask whether the two points split the same way on real corpora.

Replacement lowers the final loss on web text, transcripts, code, chess, the system-log corpus, and the road-safety corpus (Table 1, per-seed losses in App. B.4). On Korean and Japanese web and on Korean transcripts it removes about a third of the overfitting gap, with every seed positive and a one-sample t-test significant at $p < 0 . 0 5$ for each corpus. On the two SQL forms the effect is within noise, and on the census corpus replacement is harmful, a case $\ S 7$ examines.

<table><tr><td>Corpus</td><td>Schedule</td><td>gNONE</td><td>gREPLACE</td><td>Gain (%)</td><td> $\Delta L _ { \mathrm { m i n } }$ </td></tr><tr><td>KO web</td><td>constant</td><td>2.46</td><td>1.60</td><td>35.1±0.5</td><td> $+ 0 . 0 7 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>KO web</td><td>cosine</td><td>2.18</td><td>1.29</td><td>40.9±1.3</td><td> $+ 0 . 0 6 { \pm } 0 . 0 8$ </td></tr><tr><td>JA web</td><td>constant</td><td>2.81</td><td>1.76</td><td>37.4±0.6</td><td>+0.10±0.02</td></tr><tr><td>JA web</td><td>cosine</td><td>2.62</td><td>1.50</td><td>42.9±1.1</td><td> $+ 0 . 1 0 { \pm } 0 . 0 4$ </td></tr></table>

Table 2: Overfitting gap g with and without replacement, and change in the minimum loss $\Delta L _ { \mathrm { m i n } } ,$ in nats under a constant and a cosine learning rate, 336M, three seeds.

The minimum differs from the Markov source. Unlike on the Markov source, replacement often lowers the minimum loss a little on real corpora, significantly on Japanese web, transcripts, chess, and context-only SQL (Table 1). We noted a reason for this difference in §4. On real corpora the noise often draws a token that never occurs in the corpus, so the average shrinkage weight on clean contexts $( 1 - \alpha _ { \tilde { c } } )$ is at most 0.06, against 0.14 to 0.15 on the Markov source, close to its ceiling at the corruption rate p=0.15 (App. C.3). This smaller bias agrees with replacement lowering the minimum.

The effect at the minimum is still much smaller than on final loss. On web text, transcripts, code, and chess, the NONE and REPLACE runs reach their minima at nearly the same loss and part only after it (Figure 3(a–d)), and replacement delays the minimum by about one to four epochs (Table 1), which suggests that the same training tokens stay useful for longer.

Learning-rate decay shrinks the gap only slightly. A constant learning rate might itself create the late rise that replacement removes. We repeat the web comparison with a cosine schedule that decays over the full budget to one tenth of the peak (Table 2). Learning-rate decay shrinks the overfitting gap by only 7 to 12%, so most of it remains, and replacement removes a larger share of the gap under learning-rate decay than under a constant rate. The effect at the minimum is the same under either schedule, within 0.01 nats. Under stronger weight decay, an explicit regularizer, replacement still removes much of the gap (App. C.5).

<table><tr><td>Budget</td><td>KO web</td><td>JA web</td></tr><tr><td>2 epochs</td><td> $- 0 . 0 6 { \pm } 0 . 0 1 ^ { \ast }$ </td><td> $- 0 . 0 6 { \pm } 0 . 0 1 ^ { \ast }$ </td></tr><tr><td>4 epochs</td><td> $- 0 . 0 2 { \pm } 0 . 0 2$ </td><td> $+ 0 . 0 2 { \pm } 0 . 0 2$ </td></tr><tr><td>5 epochs</td><td> $+ 0 . 0 4 { \pm } 0 . 0 6$ </td><td> $+ 0 . 0 4 { \pm } 0 . 0 2$ </td></tr><tr><td>6 epochs</td><td> $+ 0 . 1 1 { \pm } 0 . 0 7$ </td><td> $+ 0 . 0 9 { \pm } 0 . 0 3$ </td></tr><tr><td>8 epochs</td><td> $+ 0 . 1 7 { \pm } 0 . 1 1$ </td><td> $+ 0 . 1 0 { \pm } 0 . 0 3 ^ { \ast }$ </td></tr><tr><td>best budget</td><td> $+ 0 . 1 6 { \pm } 0 . 1 0$ </td><td> $+ 0 . 0 8 { \pm } 0 . 0 1 ^ { \ast }$ </td></tr></table>

Table 3: Change in the minimum loss, $\Delta L _ { \mathrm { m i n } }$ in nats, with the cosine schedule decayed within each budget, 336M, mean±s.d. over three seeds; <sup>∗</sup> is $p < 0 . 0 5$ . The $L _ { \mathrm { m i n } } ^ { \mathrm { R E P L A C E } } ( b )$ per seed, with b over the five budgets.

<table><tr><td></td><td colspan="2">Gain (%)</td><td colspan="2"> $\Delta L _ { \mathrm { m i n } }$ </td></tr><tr><td>Corpus</td><td>134M</td><td>336M</td><td>134M</td><td>336M</td></tr><tr><td>KO web</td><td>52.8±1.4</td><td>35.2±1.5</td><td> $+ 0 . 1 5 { \pm } 0 . 0 2 ^ { \ast }$ </td><td> $+ 0 . 0 6 { \pm } 0 . 0 8$ </td></tr><tr><td>KO transcript</td><td>61.5±2.1</td><td>40.1±0.7</td><td> $+ 0 . 0 9 { \pm } 0 . 0 2 ^ { \ast }$ </td><td> $+ 0 . 0 6 { \pm } 0 . 0 0 ^ { \ast }$ </td></tr><tr><td>code</td><td>44.4±2.7</td><td>26.3±1.4</td><td> $+ 0 . 0 1 { \pm } 0 . 0 1$ </td><td> $- 0 . 0 6 { \pm } 0 . 0 3$ </td></tr></table>

Table 4: Regularization gain and change in the minimum loss, $\Delta L _ { \mathrm { m i n } }$ in nats, at two model scales, mean±s.d. over three seeds; <sup>∗</sup> is $p < 0 . 0 5$

The best budget still favors replacement. A practitioner would rarely train this far past the minimum, and would instead choose fewer epochs and size the decay schedule to that budget (Hoffmann et al., 2022). We therefore shorten the budget to 2, 4, 5, 6, or 8 epochs and decay the cosine schedule to one tenth of the peak within each (Table 3, Figure 1(c)). Taking the best of the five budgets for each condition, the REPLACE run reaches a lower minimum than the NONE run in every seed. At 8 epochs replacement lowers the minimum loss on both corpora. Within that budget the NONE run reaches its minimum near 4 epochs and then rises, while the REPLACE run keeps improving until about 6 to 7 epochs. The effect at the minimum loss grows with the budget, as §4 predicts. At 2 epochs both runs are still improving at their last step. No loss rise has appeared yet to offset the bias, so replacement raises the minimum loss on both corpora.

![](images/0e619af13683872a758f4933e4e39dddf90dc9a0a930966ee7be06fce485c49d.jpg)

![](images/490bc1c0083507ac767c9b8cba529b9fc361e55548ee85e6d9da67770a2276fa.jpg)  
Figure 4: Net effect of replacement in nats as the corruption rate p varies, (a) on final loss and (b) at the minimum loss, positive when it helps. Error bars give the s.d. over three seeds, two for the census minimum below $p { = } 0 . 5 ;$ the dotted line marks the main rate $\scriptstyle { p = 0 . 1 5 }$

The best corruption rate depends on the point of training. We sweep the corruption rate from 0.05 to 0.50 on two corpora at opposite ends (Figure 4, App. B.4), Korean web, where replacement clearly helps at $\scriptstyle { p = 0 . 1 5 }$ , and the census corpus, where it makes even the final loss worse. On final loss the benefit on Korean web grows with p, so heavier corruption looks better. At the minimum loss the benefit is largest at $\scriptstyle p = 0 . 1 0 ,$ , then shrinks and turns negative on average at $\scriptstyle { p = 0 . 5 0 }$ . Heavier corruption delays the loss rise more but also raises the bias, since $1 - \alpha _ { \tilde { c } }$ grows with p in Eq. (2). On the census corpus the final-loss effect stays slightly negative at every rate instead of growing with p, while the effect at the minimum loss falls steadily as p rises. This is the pattern §4 expects on highly predictable text such as the Markov source t02 (§5), where the bias is large and the benefit small; on the census corpus replacement removes little of the overfitting gap (Table 1).

Robustness across model scale. The effect is not an artifact of one model size. Table 4 repeats the comparison at 134M for three corpora. Replacement removes a larger fraction of the overfitting gap at the smaller 134M scale than at 336M, and at 134M it also lowers the minimum loss more on both natural-language corpora. The census corpus is harmed at both sizes, its final loss rising by 0.05 nats at 134M and 0.06 nats at 336M (Table 14).

## 7 Predicting the Sign on Real Corpora

On the Markov source the entropy rate H measures how predictable the source is, and as H rises the sign of the effect on final loss turns from harm to help (§5). The entropy rate of a real corpus is not known. We therefore ask whether a corpus statistic can stand in for it and predict the sign.

<table><tr><td>Corpus</td><td>Label</td><td>gzip</td><td>MI</td><td> $\Delta L _ { \mathrm { f i n a l } }$ </td><td> $\Delta L _ { \mathrm { m i n } }$ </td></tr><tr><td>chess</td><td>S</td><td>0.386</td><td>0.42</td><td> $+ 0 . 2 9 { \pm } 0 . 0 4 ^ { * }$ </td><td>+0.03±0.01*</td></tr><tr><td>code</td><td>S</td><td>0.199</td><td>0.40</td><td>+0.24±0.04*</td><td>−0.06±0.03</td></tr><tr><td>SQL-comb</td><td>S,P</td><td>0.170</td><td>0.62</td><td>+0.04±0.04</td><td>+0.06±0.05</td></tr><tr><td>SQL-ctx</td><td>S, P</td><td>0.125</td><td>0.69</td><td>+0.05±0.09</td><td>+0.13±0.02*</td></tr><tr><td>road-safety (JSON)†</td><td>S, P</td><td>0.068</td><td>0.74</td><td>+0.04±0.00*</td><td>+0.00±0.00</td></tr><tr><td>loghub†</td><td>S, P</td><td>0.063</td><td>0.72</td><td>+0.07±0.01</td><td> $+ 0 . 0 2 { \pm } 0 . 0 2$ </td></tr><tr><td>census (JSON)</td><td>S, P</td><td>0.049</td><td>0.73</td><td>−0.06±0.02*</td><td>−0.06±0.02</td></tr></table>

Table 5: Change in the final and minimum loss from replacement, in nats, mean±s.d. over three seeds, positive when replacement helps; <sup>∗</sup> is $p < 0 . 0 5$ . S is structured and P program-generated (§3). <sup>†</sup>Two seeds.

Table 5 orders the seven structured corpora by redundancy, how much of the text repeats predictable patterns, from least to most. Replacement lowers the final loss most on chess and code, has smaller effects on the two SQL forms and on the system-log and road-safety corpora, and raises it on the census corpus. On this corpus the baseline overfits, rising from 0.26 to 0.42 nats, yet the replacement run ends higher (Figure 3(f)). MI is the statistic closest to §4, because it averages over contexts the distance between $P _ { \tilde { c } }$ and π that sets the bias (App. C.3). As MI rises the benefit broadly shrinks, which agrees with §4. MI does not separate the sign, however. The census corpus, the system-log corpus and the road-safety corpus lie within 0.02 of each other in MI, yet only the census corpus is harmed, and replacement leaves the minimum loss of the other two nearly unchanged (Table 1).

Both helped corpora have more training tokens than the census corpus, so at the fixed step budget they train for fewer epochs. More epochs let the loss rise more, leaving more for replacement to remove. If repetition decided the sign, replacement would help the census corpus more than these two, yet it harms it. The difference in training tokens therefore does not explain the harm.

Program-generated origin and higher-order conditional entropy do not mark the census corpus either (App. B.3). Under replacement the minimum loss of the census corpus rises by 0.06 nats, though not significantly, whereas on both helped corpora it is nearly unchanged. A low minimum does not mark it either, since the helped roadsafety corpus has a lower one (Table 1). Among the statistics we test, only the gzip ratio sets the census corpus apart.

In gzip ratio the census corpus is the most redundant corpus we train on, and the helped system-log and road-safety corpora sit just above it, at 0.063 and 0.068. One harmed corpus cannot show whether a gzip threshold between these values separates the sign. It can still show at what redundancy the harm occurs, so we ask how often real text is as redundant as the census corpus. We measure raw-text gzip on 64 corpora that are not program-generated, spanning more than 30 languages and 11 domains (Figure 5, App. B.2). Every one sits well above the census corpus, the most redundant being Go source code at 0.092. Only program-generated text falls below that value, and the one harm we observe lies there. Every corpus we train on that is not program-generated is helped by the end of the budget (Table 1).

![](images/356fad4e154aca8bd60bd1f3425252b562424cd5e4217d0d6797d90f1bb179e0.jpg)  
Figure 5: Raw-text gzip ratio of the surveyed corpora and of the ten corpora we train on, by category.

## 8 Conclusion

We decompose validation loss into APPROX, EST, and MEM terms on a source of known entropy, which shows what random-token replacement removes. The decomposition needs only validation losses, so it applies to any augmentation. The benefit of replacement is a smaller loss rise after the minimum, and its cost is a bias that is present from the start. The point at which training stops therefore decides the size and sign of the effect. On final loss this benefit dominates, and learning-rate decay does not replace it. At the minimum loss the bias is exposed and the gain is small, though with a schedule sized to the budget replacement still reaches a lower best minimum. On the controlled source the sign on final loss also turns from harm to help as predictability falls. On real corpora replacement helps every corpus we test that is not program-generated, and harms only the census corpus, which is more redundant than all of them.

## Limitations

• Model scale. Our study spans two model scales, 134M and 336M; the benefit is present at both but diminishes as scale grows, and we do not claim it persists at substantially larger scale.

• Harm on real corpora. On final loss, replacement harms one real corpus we test, so we observe the predictability threshold only on the controlled source. Locating it on real corpora needs more corpora that cross it, and building such a set is our next step.

• Cross-family comparisons and controls. Several structured corpora hold fewer training tokens than the web and transcript corpora, so under the fixed step budget they train for more epochs. Cross-family sign comparisons therefore mix predictability with repetition dose; the equal-token web and transcript corpora and the controlled Markov source are the clean comparisons. An English-prose control under the same recipe would tie our runs to the published English effect, and it is the next control we would add.

• Corruption objective and rate. We sweep the corruption rate on two corpora, and the remaining experiments use p=0.15. We size the decayed schedule to five budgets from 2 to 8 epochs on web text, and to five from 4 to 66 on the controlled source. Dropout and other explicit regularizers, the masking and infilling variants, rates beyond p=0.5, and decayed budgets longer than 8 epochs on real corpora are the sweeps we would run next.

• Languages. Both languages we test use non-Latin scripts but have abundant data overall. Under-resourced languages and writing systems with higher tokenizer fertility are the setting we would test next.

• Controlled source and metric. The controlled analysis uses an order-1 Markov source; higher-order sources would test the mechanism further. Reported effects are based on validation loss, which suits the pretraining regime we study, and they may not transfer to task performance.

## Ethical Considerations

This work studies a training-time regularization method for small language models trained from scratch. All corpora are publicly available (§3, App. A) and are used rather than re-released; the YODAS speech transcripts are public automatic captions, and we perform no extraction of personally identifying information. The computational footprint is modest (about three GPU-hours per run on a single GPU, App. A), limiting environmental cost. The method targets pretraining loss with no deployment or downstream claims, so we identify no significant additional ethical, societal, or dual-use risks.

## References

b-mc2. 2023. SQL-create-context dataset. https: / / huggingface . co / datasets / b - mc2 / sql - create- context. Derived from WikiSQL and Spider.

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. 2024. Explaining neural scaling laws. Proceedings of the National Academy ofSciences, 121(27):e2311878121.

Douglas Biber. 1988. Variation Across Speech and Writing. Cambridge University Press.

Chris M Bishop. 1995. Training with noise is equivalent to tikhonov regularization. Neural Computation, 7(1):108–116.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, and Chiyuan Zhang. 2023. Quantifying memorization across neural language models. In International Conference on Learning Representations.

Michael K Chen, Xikun Zhang, Fan Bai, Zhengding Hu, and Zhen Wang. 2026. Demystifying training-time augmentation for data-constrained language model pretraining. arXiv preprint arXiv:2606.16246.

Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information Theory, 2nd edition. Wiley-Interscience.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. 2024. Language modeling is compression. In International Conference on Learning Representations (ICLR). ArXiv:2309.10668.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.

Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. 2022. Why do tree-based models still outperform deep learning on typical tabular data? In

Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, and 3 others. 2022. Training compute-optimal large language models. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2203.15556.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Code-SearchNet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.

Changhao Jiang, Ming Zhang, Yifei Cao, Junjie Ye, Xiaoran Fan, Shihan Dou, Zhiheng Xi, Jiajun Sun, Yi Dong, Yujiong Shen, Jingqi Tong, Baoyu Fan, Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. Beyond scaling: Measuring and predicting the upper bound of knowledge retention in language model pre-training. In Association for Computational Linguistics (ACL), pages 26377–26396.

Jaap Jumelet and Willem Zuidema. 2023. Transparency at the source: Evaluating and interpreting language models with access to the true distribution. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4354–4369.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Adam Karvonen. 2024. Chess games dataset. https: //huggingface.co/datasets/adamkarvonen/ chess\_games. PGN game transcripts.

Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. 2023. The stack: 3 TB of permissively licensed source code. Transactions on Machine Learning Research.

Ron Kohavi. 1996. Scaling up the accuracy of naive-Bayes classifiers: a decision-tree hybrid. In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD).

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. In Association for Computational Linguistics (ACL), pages 7871–7880.

Xinjian Li, Shinnosuke Takamichi, Takaaki Saeki, William Chen, Sayaka Shiota, and Shinji Watanabe. 2023. YODAS: Youtube-oriented dataset for audio and speech. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU).

Chi-Heng Lin, Chiraag Kaushik, Eva L Dyer, and Vidya Muthukumar. 2024. The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective. Journal of Machine Learning Research (arXiv:2210.05021).

Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma. 2023. Same pre-training loss, better downstream: Implicit bias matters for language models. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of PMLR, pages 22188–22214.

Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. 2023. Scaling data-constrained language models. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2305.16264.

Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. 2020. Deep double descent: Where bigger models and more data hurt. In International Conference on Learning Representations (ICLR).

Alex Nguyen, David J Schwab, and Vudtiwat Ngampruetikorn. 2025. Data coarse graining can improve model performance. arXiv preprint arXiv:2509.14498.

Inkit Padhi, Yair Schiff, Igor Melnyk, Mattia Rigotti, Youssef Mroueh, Pierre Dognin, Jerret Ross, Ravi Nair, and Erik Altman. 2021. Tabular transformers for modeling multivariate time series. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

Rohan Pandey. 2024. gzip predicts data-dependent scaling laws. arXiv preprint arXiv:2405.16684.

Valentin Pelloin, Franck Dary, Nicolas Herve, Benoit Favre, Nathalie Camelin, Antoine Laurent, and Laurent Besacier. 2022. Asr-generated text for language model pre-training applied to speech tasks. In Interspeech. ArXiv:2207.01893.

Guilherme Penedo, Hynek Kydlícek, Vinko Sabolˇ cec,ˇ Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. 2025. FineWeb2: One pipeline to scale them all – adapting pretraining data processing to every language. Preprint, arXiv:2506.20920.

Skyler Seto, Maartje ter Hoeve, Richard He Bai, Natalie Schluter, and David Grangier. 2025. Training bilingual LMs with data constraints in the targeted language. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19096– 19122.

Mukuntha Narayanan Sundararaman, Ayush Kumar, and Jithendra Vepa. 2021. Phoneme-BERT: Joint language modelling of phoneme sequence and ASR transcript. In Proceedings ofInterspeech 2021.

Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. 2022. Memorization without overfitting: Analyzing the training dynamics of large language models. In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2205.10770.

Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. 2024. Position: Will we run out of data? Limits of LLM scaling based on human-generated data. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of PMLR.

Ziang Xie, Sida I Wang, Jiwei Li, Daniel Lévy, Aiming Nie, Dan Jurafsky, and Andrew Y Ng. 2017. Data noising as smoothing in neural network language models. In International Conference on Learning Representations (ICLR). ArXiv:1703.02573.

Zhiwei Xu, Shihao Wu, Hanseul Cho, Wei Hu, and Yixin Wang. 2026. Data-constrained language model pretraining: Improved regularization and scaling laws. In Proceedings of the 43rd International Conference on Machine Learning (ICML).

Jie Zhang, Qinghua Zhao, Chi-ho Lin, Zhongfeng Kang, and Lei Li. 2025. Beyond frequency: The role of redundancy in large language model memorization. arXiv preprint arXiv:2506.12321.

Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. 2023. Loghub: A large collection of system log datasets for ai-driven log analytics. In IEEE International Symposium on Software Reliability Engineering (ISSRE).

## A Training and Reproducibility Details

Model. The 336M model is a decoder-only Transformer with 24 layers, hidden size 1024, 16 attention heads in 4 key-value groups, and feedforward width 2816; the 134M variant has 14 layers, hidden size 768, 12 heads in 2 key-value groups, and feed-forward width 2048. On the Markov source the vocabulary is 1025 rather than 64k, which lowers the parameter count to 271.6M at the 336M configuration. All runs use bfloat16 and gradient checkpointing.

Recipe. Constant learning rate $3 \times 1 0 ^ { - 4 }$ with 300 warmup steps, AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ and weight decay 0.1, gradient clipping at 1.0, batch size 8, sequence length 2048, 10,000 steps, 8,000 on the Markov source, and a shared 64k SentencePiece tokenizer. Replacement independently replaces each input token with a uniform random vocabulary token at rate p=0.15, leaving prediction targets intact. The learning-rate comparison of §6 uses a cosine schedule from the same peak to $3 \times 1 0 ^ { - 5 }$ over the 10,000 steps, with the same warmup. These runs and the constant-rate runs they are compared with were trained on an H100 GPU. The H100 constant-rate runs reproduce the main-experiment runs within 0.04 nats at both points of training. The budget-sized schedules of Table 3 run for 900 to 3,600 steps, which is 2, 4, 5, 6, or 8 epochs of a 7.4M-token corpus, with the same warmup, cosine decay to $3 \times 1 0 ^ { - 5 }$ at the last step, and an evaluation every 150 steps. The abundant-data runs at t02 were repeated as a pair on an H100 GPU, where replacement raises $L _ { \mathrm { a b u n d } }$ by 0.048 nats.

Data split. For every corpus we draw the training tokens and a disjoint validation slice from the same source. The 7.4M-token corpora have exactly 7,383,561 training tokens and a separate 0.5M-token validation slice. All $L _ { \mathrm { m i n } }$ and $L _ { \mathrm { f i n a l } }$ values in the paper are computed on this held-out slice, which the model never sees in training; the Markov source’s split is given below.

Compute and availability. Every run trains a single 134M or 336M model on one L4 or L40S GPU in about three hours, or on one H100 GPU in under an hour, with no multi-GPU or multi-node parallelism. All corpora in Table 6 are public, and our training code and configuration files are available at https: //github.com/wonsupshin-flitto/randomtoken-replacement-repeated-data, so every result can be reproduced.

Markov source. The controlled source is an order-1 Markov chain over V=1024 states. The model vocabulary has V+1 ids. The extra id is reserved for masking and never occurs in the data, and replacement draws uniformly from all $V { + 1 }$ ids. Each row $P _ { i }$ of the transition matrix $P ,$ with entries $P _ { i j } ,$ is softmax $( g _ { i } / T )$ with independent standard-normal logits $g _ { i }$ and a fixed seed, so the chain is reproducible, and the temperature $T$ sets predictability: small $T$ gives neardeterministic rows, while large $T$ makes them uniform with H→ log V . Writing π for the stationary distribution, $\pi P = \pi$ , the entropy rate has the closed form $\begin{array} { r } { H = - \sum _ { i } \pi _ { i } \sum _ { j } P _ { i j } } \end{array}$ ln $P _ { i j }$ (Cover and Thomas, 2006), which serves as the lower bound in $\ S 5$ . Training and validation sequences are disjoint samples from the same fixed chain, 2M and 0.2M tokens, and the Markov runs train for 8,000 steps, about 65 epochs.

Abundant-data reference. The approximation term $\mathrm { A P P R O X } { = } L _ { \mathrm { a b u n d } } { - } H$ uses an abundantdata run, with the same model and recipe but a much larger draw from the same Markov source $( 1 . 4 \times 1 0 ^ { 8 }$ tokens versus the $2 \times 1 0 ^ { 6 }$ -token dataconstrained corpus), so tokens repeat little and memorization does not arise. Its loss $L _ { \mathrm { a b u n d } }$ is 1.686, 5.085, and 6.878 nats at t02, t05, and t20 for the baseline, and 1.727, 5.139, and 6.901 under replacement. In every abundant-data run, validation loss rises by at most 0.004 nats after its minimum, against 2.0 to 5.4 nats on average in the data-constrained runs at the same temperatures.

Corpora. Table 6 lists sources and trainingtoken counts. The two SQL forms are labeled program-generated because their CREATE TA-BLE contexts were generated with SQLGlot from Spider and WikiSQL queries.

<table><tr><td>Corpus</td><td>Src</td><td>Train tok.</td></tr><tr><td>KO web</td><td>FineWeb-2</td><td>7.4M</td></tr><tr><td>JA web</td><td>FineWeb-2</td><td>7.4M</td></tr><tr><td>KO transcript</td><td>YODAS</td><td>7.4M</td></tr><tr><td>code</td><td>The Stack</td><td>7.4M</td></tr><tr><td>chess</td><td>move sequences</td><td>5.2M</td></tr><tr><td>SQL-comb</td><td>SQL-create-context</td><td>5.0M</td></tr><tr><td>SQL-ctx</td><td>SQL-create-context</td><td>2.3M</td></tr><tr><td>census (JSON)</td><td>Adult census</td><td>5.5M</td></tr><tr><td>road-safety (JSON)</td><td>road-safety table</td><td>7.4M</td></tr><tr><td>loghub</td><td>LogHub</td><td>7.4M</td></tr><tr><td>Markov (constr.)</td><td>order-1 chain</td><td>2.0M</td></tr><tr><td>Markov (abund.)</td><td>order-1 chain</td><td>140M</td></tr></table>

Table 6: Corpora, sources, and training-token counts.

## B Detailed Results

## B.1 Loss Decomposition on the Markov Source

Table 7 lists the exact three-seed decomposition values plotted in Figure 2: the approximation (APPROX), estimation (EST), and memorization (MEM) terms of Eq. (3) at three source temperatures.

## B.2 Redundancy Survey

Table 8 gives the full redundancy survey behind Figure 5: 64 corpora that are not programgenerated, across more than 30 languages and 11 domains, plus program-generated text. No corpus that is not program-generated falls below rawtext gzip 0.092, the value of Go source code from CodeSearchNet (Husain et al., 2019), the most redundant one; among the surveyed corpora only the system logs do. The two JSON serializations we train on, at gzip 0.049 and 0.068, also lie below it but were not part of the survey; the survey’s nearby program-generated entry is a distinct serialized dump at gzip 0.098.

<table><tr><td>Src</td><td>Cond</td><td>H</td><td>APPROX</td><td>EST</td><td>MEM</td></tr><tr><td>t02</td><td>none</td><td>1.646</td><td>0.040</td><td>0.102</td><td>2.29±0.01</td></tr><tr><td>t02</td><td>replace</td><td>1.646</td><td>0.081</td><td>0.177</td><td> $2 . 4 1 { \pm } 0 . 0 8$ </td></tr><tr><td>t05</td><td>none</td><td>5.019</td><td>0.066</td><td>0.212</td><td> $4 . 7 8 { \pm } 0 . 0 4$ </td></tr><tr><td>t05</td><td>replace</td><td>5.019</td><td>0.120</td><td>0.157</td><td> $\mathbf { 2 . 0 2 { \pm } 0 . 0 0 }$ </td></tr><tr><td>t20</td><td>none</td><td>6.806</td><td>0.072</td><td>0.057</td><td> $5 . 3 9 { \pm } 0 . 1 5 $ </td></tr><tr><td>t20</td><td>replace</td><td>6.806</td><td>0.095</td><td>0.033</td><td> $\mathbf { 2 . 0 5 { \pm } 0 . 3 9 }$ </td></tr></table>

Table 7: Loss decomposition on the Markov source in nats, three-seed means, with MEM as mean±s.d.; APPROX and EST vary by less than 0.01 nats across seeds.

<table><tr><td>Category</td><td>n</td><td>gzip</td><td>MI/H1</td></tr><tr><td>web (FineWeb-2)</td><td>32</td><td>0.186-0.468</td><td>0.32-0.52</td></tr><tr><td>Wikipedia</td><td>10</td><td>0.209-0.441</td><td>0.33-0.48</td></tr><tr><td>legal</td><td>6</td><td>0.262–0.322</td><td>0.42-0.49</td></tr><tr><td>code</td><td>6</td><td>0.092-0.248</td><td>0.42-0.64</td></tr><tr><td>reviews</td><td>2</td><td>0.282-0.384</td><td>0.41-0.45</td></tr><tr><td>scientific</td><td>1</td><td>0.331</td><td>0.42</td></tr><tr><td>medical</td><td>1</td><td>0.332</td><td>0.44</td></tr><tr><td>math</td><td>1</td><td>0.333</td><td>0.42</td></tr><tr><td>dialogue</td><td>1</td><td>0.334</td><td>0.42</td></tr><tr><td>news</td><td>2</td><td>0.375-0.376</td><td>0.40-0.47</td></tr><tr><td>social</td><td>2</td><td>0.411-0.441</td><td>0.50-0.53</td></tr><tr><td>serialized dump</td><td>1</td><td>0.098</td><td>0.66</td></tr><tr><td>machine logs</td><td>2</td><td>0.014–0.063</td><td>0.71-0.81</td></tr></table>

Table 8: Redundancy survey by category, ranges of raw-text gzip ratio and normalized bigram MI.

## B.3 Predictability Scalars Do Not Separate the Census Corpus

Table 9 reports higher-order conditional entropies, measured on 2M-token samples with the shared tokenizer. No conditional-entropy scalar separates the census corpus from the helped corpora: a helped web corpus even has a lower order-3 conditional entropy than the census corpus, an artifact of data sparsity in low-repetition text. Among the corpora in Table 9, the census corpus also has the lowest baseline minimum loss, at 0.26 nats against 0.93 for loghub. This does not separate it either. The helped road-safety corpus, not in the table, has a lower minimum still, at 0.17 nats (Table 1).

<table><tr><td>Corpus</td><td> $H _ { 1 }$ </td><td> $H _ { c 1 }$ </td><td> $H _ { c 2 }$ </td><td> $H _ { c 3 }$ </td><td> $\mathbf { M I } _ { 1 } / H _ { 1 }$ </td><td> $L _ { \mathrm { m i n } }$ </td></tr><tr><td>census (harm)</td><td>5.76</td><td>1.56</td><td>0.85</td><td>0.57</td><td>0.73</td><td>0.26</td></tr><tr><td>loghub (help)</td><td>6.58</td><td>1.89</td><td>1.00</td><td>0.69</td><td>0.72</td><td>0.93</td></tr><tr><td>SQL-ctx (noise)</td><td>6.12</td><td>1.88</td><td>1.26</td><td>1.07</td><td>0.69</td><td>2.40</td></tr><tr><td>code (help)</td><td>8.36</td><td>5.01</td><td>2.65</td><td>1.43</td><td>0.40</td><td>1.86</td></tr><tr><td>KO web (help)</td><td>11.65</td><td>6.23</td><td>1.86</td><td>0.50</td><td>0.47</td><td>4.91</td></tr></table>

Table 9: Unigram entropy $H _ { 1 }$ and order-1 to order-3 conditional entropies $H _ { c 1 } , H _ { c 2 } , H _ { c 3 }$ in bits on 2Mtoken samples, normalized bigram MI, $\mathbf { M I } _ { 1 } / H _ { 1 }$ , and the baseline’s minimum loss $L _ { \mathrm { m i n } }$ in nats. The row label gives the sign of the final-loss effect; $L _ { \mathrm { m i n } }$ uses three seeds, two for loghub.

## B.4 Per-Seed Results

Table 10 reports the per-seed minimum and final losses behind the web-text results, and Table 11 the per-seed effects behind the Markov sweep of Figure 1(b). Tables 12 and 13 report the per-seed net effects behind the two panels of the corruptionrate sweep of Figure 4, Tables 14 and 15 report those behind the two-scale comparison and the best-budget comparison of $\ S 6 .$ . We assess significance with a one-sample t-test on the three-seed effects, two-sided with df=2, and mark $p < 0 . 0 5$ with <sup>∗</sup> in the tables. The two SQL forms, described as within noise in $\ S 6 ,$ have $p > 0 . 2$

<table><tr><td>Corpus</td><td>Cond</td><td></td><td> $\mathbf { s } 1$ </td><td> $\mathbf { s } 2$ </td><td> ${ \bf s } 3$ </td></tr><tr><td>KO web</td><td>none</td><td> $L _ { \mathrm { m i n } }$ </td><td>4.86</td><td>5.01</td><td>4.87</td></tr><tr><td rowspan="3">KO web</td><td rowspan="3">replace</td><td> $L _ { \mathrm { f i n a l } }$ </td><td>7.29</td><td>7.53</td><td>7.32</td></tr><tr><td> $L _ { \mathrm { m i n } }$ </td><td>4.87</td><td>4.85</td><td>4.84</td></tr><tr><td> $L _ { \mathrm { f i n a l } }$ </td><td>6.44</td><td>6.43</td><td>6.47</td></tr><tr><td rowspan="2">JA web</td><td rowspan="2">none</td><td> $L _ { \mathrm { m i n } }$ </td><td>4.75</td><td>4.65</td><td>4.71</td></tr><tr><td> $L _ { \mathrm { f i n a l } }$ </td><td>7.47</td><td>7.55</td><td>7.48</td></tr><tr><td rowspan="2">JA web</td><td rowspan="2">replace</td><td> $L _ { \mathrm { m i n } }$ </td><td>4.65</td><td>4.57</td><td>4.62</td></tr><tr><td> $L _ { \mathrm { f i n a l } }$ </td><td>6.40</td><td>6.40</td><td>6.28</td></tr></table>

Table 10: Per-seed minimum loss and final loss in nats for the two web corpora at 336M.

<table><tr><td>Src</td><td>net s1</td><td>net s2</td><td>net s3</td><td>mean±s.d.</td></tr><tr><td>t02</td><td>-0.35</td><td>-0.19</td><td>-0.17</td><td> $- 0 . 2 3 { \pm } 0 . 0 8$ </td></tr><tr><td>t03</td><td>+1.00</td><td>+1.14</td><td>+0.95</td><td> $+ 1 . 0 3 { \pm } 0 . 0 8 $ </td></tr><tr><td>t05</td><td> $+ 2 . 7 2$ </td><td> $+ 2 . 7 5$ </td><td> $+ 2 . 8 2$ </td><td> $+ 2 . 7 6 { \pm } 0 . 0 4$ </td></tr><tr><td>t08</td><td>+2.69</td><td> $+ 2 . 3 7$ </td><td> $+ 1 . 8 3$ </td><td> $+ 2 . 2 9 { \pm } 0 . 3 6 $ </td></tr><tr><td>t20</td><td>+3.02</td><td>+3.62</td><td>+3.40</td><td> $+ 3 . 3 5 { \pm } 0 . 2 5 $ </td></tr></table>

Table 11: Per-seed change in the final loss, $\Delta L _ { \mathrm { f i n a l } }$ in nats, across the Markov temperatures, with the threeseed mean±s.d.

<table><tr><td>p</td><td>s1</td><td> $s 2$ </td><td> ${ \mathrm { s } } 3$ </td><td>mean±s.d.</td></tr><tr><td colspan="5">KO web</td></tr><tr><td>0.05</td><td>0.51</td><td>0.52</td><td>0.50</td><td>0.51±0.01</td></tr><tr><td>0.10</td><td>0.74</td><td>0.87</td><td>0.73</td><td>0.78±0.08</td></tr><tr><td>0.20</td><td>1.04</td><td>1.19</td><td>1.03</td><td>1.09±0.09</td></tr><tr><td>0.30</td><td>1.29</td><td>1.45</td><td>1.34</td><td>1.36±0.08</td></tr><tr><td>0.50</td><td>1.83</td><td>1.92</td><td>1.82</td><td>1.85±0.06</td></tr><tr><td colspan="5">census corpus</td></tr><tr><td>0.05 0.10</td><td>-0.03 -0.05</td><td>-0.06 -0.06</td><td>-0.04 -0.08</td><td> $- 0 . 0 4 { \pm } 0 . 0 1$   $- 0 . 0 6 { \pm } 0 . 0 1$ </td></tr><tr><td>0.20</td><td>-0.09</td><td>-0.07</td><td>-0.05</td><td> $- 0 . 0 7 { \pm } 0 . 0 2$ </td></tr><tr><td>0.30</td><td>-0.06</td><td>-0.05</td><td>-0.03</td><td> $- 0 . 0 4 { \pm } 0 . 0 2$ </td></tr><tr><td>0.50</td><td>-0.01</td><td>0.00</td><td>-0.03</td><td> $- 0 . 0 1 { \pm } 0 . 0 2$ </td></tr></table>

Table 12: Per-seed change in the final loss, $\Delta L _ { \mathrm { f i n a l } }$ in nats, across the corruption rates of Figure 4(a) on Korean web and the census corpus, with the three-seed mean±s.d.
<table><tr><td> $p$ </td><td>s1</td><td> $s 2$ </td><td> ${ \mathrm { s } } 3$ </td><td>mean±s.d.</td></tr><tr><td>KO web 0.05 0.10</td><td>+0.051 +0.063</td><td>+0.091 +0.161 +0.119</td><td>+0.041 +0.042 +0.031</td><td>+0.061±0.022  $+ 0 . 0 8 9 { \pm } 0 . 0 5 2$   $+ 0 . 0 4 4 { \pm } 0 . 0 5 7$ </td></tr><tr><td>0.30 0.50 census corpus 0.05 0.10</td><td>+0.006 -0.105 -0.039 -0.056</td><td>+0.067 -0.002 -0.038 -0.027 -0.046</td><td>-0.006 -0.200</td><td> $+ 0 . 0 2 2 { \scriptstyle \pm 0 . 0 3 2 }$   $- 0 . 1 0 2 { \pm } 0 . 0 8 1$  −0.038±0.001 -0.041±0.015 -0.062±0.015</td></tr></table>

Table 13: Per-seed change in the minimum loss, $\Delta L _ { \mathrm { m i n } }$ in nats, across the corruption rates of Figure 4(b). Two seeds are available for the census corpus below $\mathrm { \ p { = } 0 . 5 }$
<table><tr><td>Scale</td><td>s1</td><td> $\mathbf { s } 2$ </td><td> ${ \bf s } 3$ </td><td>mean±s.d.</td></tr><tr><td>134M</td><td>-0.06</td><td>-0.03</td><td>-0.06</td><td> $- 0 . 0 5 { \pm } 0 . 0 1$ </td></tr><tr><td>336M</td><td>-0.05</td><td>-0.04</td><td>-0.08</td><td> $- 0 . 0 6 { \pm } 0 . 0 2$ </td></tr></table>

Table 14: Per-seed change in the final loss from replacement on the census corpus, $\Delta L _ { \mathrm { f i n a l } }$ in nats, at both model scales; negative means replacement harms. The 336M row is the census (JSON) row of Table 5.

## C Supplementary Experiments

## C.1 Memorization and Shrinkage Probes on the Markov Source

To check that MEM arises from memorization, we measure each run’s loss on clean training and validation windows of the same length. Every run fits its training windows to a loss far below the entropy rate H, the lowest loss any model can reach on held-out windows from the source (Table 16). All runs therefore memorize their training sequences. Replacement leaves the training loss about the same or lower, and lowers the validation loss where it helps. We also measure verbatim extraction (Carlini et al., 2023), the share of 50-token spans after at least 1,024 tokens of context that greedy decoding reproduces exactly. Replacement raises this share on training windows at every temperature, while it lowers MEM at t05 and t20 (§5), and the share is zero on validation windows in every run. MEM therefore does not measure how much the model memorizes. It measures the rise in held-out loss that accompanies memorization.

<table><tr><td>Corpus</td><td>s1</td><td> $^ { \mathrm { s 2 } }$ </td><td>s3</td><td>mean±s.d.</td></tr><tr><td>KO web</td><td>+0.052</td><td>+0.288</td><td>+0.148</td><td> $+ 0 . 1 6 3 { \scriptstyle \pm 0 . 0 9 7 }$ </td></tr><tr><td>JA web</td><td>+0.089</td><td>+0.065</td><td>+0.082</td><td> $+ 0 . 0 7 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr></table>

Table 15: Per-seed change in the minimum loss, $\Delta L _ { \mathrm { m i n } }$ in nats, behind the best-budget row of Table 3. Each condition uses its minimum loss at its own best budget within each seed.

<table><tr><td>Source</td><td>Cond</td><td>H</td><td>Train</td><td>Val</td><td>Extr.</td><td> $\mathrm { K L } ( Q _ { \theta } \| P _ { \mathrm { u r e } } )$ </td><td>KL(Q∥π)</td></tr><tr><td>t02</td><td>none</td><td>1.65</td><td>0.14</td><td>4.06</td><td>0.69</td><td>0.15</td><td>5.31</td></tr><tr><td>t02</td><td>replace</td><td></td><td>0.04</td><td>4.40</td><td>0.90</td><td>2.55</td><td>3.57</td></tr><tr><td>t05</td><td>none</td><td>5.02</td><td>0.30</td><td>10.02</td><td>0.55</td><td>0.56</td><td>2.52</td></tr><tr><td>t05</td><td>replace</td><td></td><td>0.30</td><td>7.32</td><td>0.79</td><td>0.66</td><td>1.24</td></tr><tr><td>t20</td><td>none</td><td>6.81</td><td>0.83</td><td>12.47</td><td>0.40</td><td>0.60</td><td>0.49</td></tr><tr><td>t20</td><td>replace</td><td></td><td>0.70</td><td>9.48</td><td>0.56</td><td>0.30</td><td>0.27</td></tr></table>

Table 16: Clean training and validation loss in nats, the share of 50-token training spans reproduced verbatim (Extr.), and π-weighted mean $\operatorname { K L } ( Q _ { \theta } | | \cdot )$ to the true transition $P _ { \mathrm { t r u e } }$ and the marginal π, on the Markov source. Each loss averages 160 windows of 2,048 tokens.

We also compare each model’s learned nexttoken distribution $Q _ { \theta } ( \cdot \mid c )$ , aggregated over validation positions by the preceding context c, with the true transition $P _ { \mathrm { t r u e } }$ and the corpus marginal π. The table reports, for each reference R, the π- weighted mean $\begin{array} { r } { \sum _ { c } \pi ( c ) \operatorname { K L } ( Q _ { \theta } ( \cdot \ : | \ : c ) | | R ( \cdot \ : | \ : c ) ) } \end{array}$ , with $R ( \cdot \mid c ) = \pi$ for the marginal. Replacement moves $Q _ { \theta }$ toward π at every temperature, the shrinkage of §4. Its effect on the distance to $P _ { \mathrm { t r u e } }$ follows predictability. The distance rises sharply at the most predictable source t02, barely changes at t05, and falls at t20, which agrees with the bias that §4 predicts.

## C.2 Budget-Sized Schedules on the Markov Source

Table 3 varies the training budget on web text with the cosine schedule decayed within each budget. We repeat that design on the Markov source, using the same chains as §5 at three temperatures.

Budgets run from 500 to 8,000 steps, which is 4 to 66 epochs, with three seeds per cell, for 90 runs in total.

At t02 replacement raises the minimum at every budget, which agrees with the shrinkage bias of §4 (Table 17). At t20 it leaves the minimum unchanged at every budget. The largest improvement in any cell is 0.023 nats, at t05 under a 16-epoch budget.

<table><tr><td>Budget</td><td>t02</td><td>t05</td><td>t20</td></tr><tr><td>4 epochs</td><td> $- 0 . 1 4 9 { \pm } 0 . 0 0 1 ^ { \ast }$ </td><td> $- 0 . 0 4 2 { \pm } 0 . 0 0 1 ^ { \ast }$ </td><td>+0.001±0.003</td></tr><tr><td>8 epochs</td><td> $- 0 . 1 0 6 { \pm } 0 . 0 0 5 ^ { \ast }$ </td><td> $- 0 . 0 2 8 { \pm } 0 . 0 0 0 ^ { \ast }$ </td><td>+0.000±0.000</td></tr><tr><td>16 epochs</td><td> $- 0 . 1 1 3 { \pm } 0 . 0 0 7 ^ { \ast }$ </td><td> $+ 0 . 0 2 3 { \scriptstyle \pm 0 . 0 0 2 ^ { * } }$ </td><td>-0.001±0.003</td></tr><tr><td>33 epochs</td><td> $- 0 . 1 1 7 { \scriptstyle \pm 0 . 0 0 9 ^ { \ast } }$ </td><td> $+ 0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>+0.002±0.001</td></tr><tr><td>66 epochs</td><td> $- 0 . 1 1 9 { \pm } 0 . 0 0 9 ^ { \ast }$ </td><td> $- 0 . 0 0 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>+0.001±0.001</td></tr></table>

Table 17: Change in the minimum loss on the Markov source, $\Delta L _ { \mathrm { m i n } }$ in nats, with the cosine schedule decayed within each budget, mean±s.d. over three seeds; ∗ is $p < 0 . 0 5$

## C.3 Shrinkage Weight on Real Corpora

Eq. (1) in §4 writes the optimal prediction after an observed context token c˜ as $\alpha _ { \tilde { c } } P _ { \tilde { c } } + ( 1 - \alpha _ { \tilde { c } } ) \pi$ where $P _ { \tilde { c } }$ is the true conditional, π the corpus marginal, and $\alpha _ { \tilde { c } }$ the posterior probability that c˜ is clean. Averaged over observed positions, $1 - \alpha$ equals the corruption rate p on every corpus, because by Eq. (2), $( 1 - \alpha _ { \tilde { c } } )$ times the probability of observing c˜ is $p q ( \tilde { c } )$ , where q is the noise distribution and sums to one. What differs between corpora is where that weight falls. We report two quantities, both computed from the training tokens with no model involved.

The first, out-of-corpus in Table 18, is how often the uniform noise draws a token that never occurs in the corpus. Such a token is recognizable as noise, and the model can fall back to the marginal at that position alone, so no clean context is shrunk. The second is the shrinkage that falls on clean contexts, $\begin{array} { r } { \begin{array} { r } { w = \sum _ { c } \pi ( c ) ( 1 - \alpha _ { c } ) } \end{array} } \end{array}$ That is the part that costs accuracy, since it dilutes a context that was in fact clean. The bias also grows with how far the true conditional departs from the marginal. The expectation of that distance over contexts is the bigram mutual information, MI, used here in nats rather than normalized as in §3. We therefore report the product w MI as a predicted bias. Table 18 gives all three at $\scriptstyle { p = 0 . 1 5 }$

On the Markov source almost no noise is recognizable, so shrinkage falls on every prediction and w is 0.137 to 0.150, close to $p .$ On the real corpora 34 to 99.7% of noise draws are out-of-corpus and w is 0.001 to 0.058. Across the three Markov temperatures w MI tracks the rise in the minimum loss, from 0.69 at t02, where the minimum rises by 0.12 nats, to 0.06 at t20, where it is unchanged. Across the real corpora w MI does not separate the harmed corpus from the helped ones. The harmed census corpus and the helped loghub and roadsafety corpora sit within 0.001 of each other on w MI. This matches §7, where mutual information does not set the census corpus apart either.

<table><tr><td>Corpus</td><td>out-of-corpus (%)</td><td>w</td><td>MI</td><td>wMI</td><td> $\Delta L _ { \mathrm { m i n } }$ </td></tr><tr><td>KO web</td><td>33.7</td><td>0.058</td><td>3.34</td><td>0.194</td><td>+0.06</td></tr><tr><td>JA web</td><td>63.6</td><td>0.036</td><td>2.80</td><td>0.101</td><td>+0.09</td></tr><tr><td>KO transcript</td><td>55.6</td><td>0.038</td><td>3.35</td><td>0.128</td><td>+0.06</td></tr><tr><td>code</td><td>81.8</td><td>0.016</td><td>2.11</td><td>0.033</td><td>-0.06</td></tr><tr><td>chess</td><td>99.7</td><td>0.000</td><td>1.64</td><td>0.001</td><td>+0.03</td></tr><tr><td>SQL-comb</td><td>87.8</td><td>0.010</td><td>3.09</td><td>0.031</td><td>+0.06</td></tr><tr><td>SQL-ctx</td><td>93.9</td><td>0.006</td><td>2.94</td><td>0.018</td><td>+0.13</td></tr><tr><td>census</td><td>99.0</td><td>0.001</td><td>2.91</td><td>0.004</td><td>-0.06</td></tr><tr><td>loghub</td><td>98.6</td><td>0.002</td><td>2.89</td><td>0.005</td><td>+0.02</td></tr><tr><td>road-safety</td><td>99.2</td><td>0.001</td><td>3.18</td><td>0.004</td><td>+0.00</td></tr><tr><td>Markov t02</td><td>0.1</td><td>0.137</td><td>5.01</td><td>0.689</td><td>-0.12</td></tr><tr><td>Markov t05</td><td>0.1</td><td>0.149</td><td>2.08</td><td>0.311</td><td>+0.00</td></tr><tr><td>Markov t20</td><td>0.1</td><td>0.150</td><td>0.42</td><td>0.063</td><td>+0.00</td></tr></table>

Table 18: Shrinkage measured from the training tokens at $\scriptstyle { p = 0 . 1 5 }$ . out-of-corpus is the chance that the uniform noise draws a token that never occurs in the corpus; w is the shrinkage falling on clean contexts; MI is the bigram mutual information in nats. $\Delta L _ { \mathrm { m i n } }$ repeats the measured effect at the minimum loss from Table 1 and §5.

## C.4 Minimum Loss and Early Stopping

Our minimum loss is read from the full curve after training, whereas early stopping halts once validation loss stops improving for a set number of evaluations and keeps the best checkpoint seen so far. The two can differ when the loss rises for a few evaluations and then falls again before its true minimum. We simulate early stopping on the 174 runs whose minimum loss the paper reports, evaluated every 250 or 300 steps. With a patience of one evaluation it misses the minimum on 79 runs, by up to 0.65 nats. With a patience of three it misses on 10, by up to 0.09, and with five on 2, by at most 0.01. The runs it misses at low patience are mostly REPLACE runs, 58 of the 79 at a patience of one, because their loss descends more slowly. The comparison in §6 therefore needs a patience of about five evaluations, which is 1,500 steps or about three epochs on the 7.4M-token corpora.

## C.5 Weight Decay

To check that replacement is not a proxy for stronger explicit regularization, we raise the weight decay from 0.1 to 0.5 and compare with and without replacement. On Korean web and code, two seeds each, replacement still removes 46% and 39% of the overfitting gap, and at the minimum loss it lowers the loss on Korean web and leaves it unchanged on code. It therefore acts on a part of the gap that weight decay does not remove.

## C.6 Epoch Sweep on Repeated Data

To confirm that our corpora sit in the repeated-data regime, we extend one corpus (Korean web, 7.4M tokens) to 32 epochs under the fixed recipe and read validation loss as a function of epoch (Figure 6). On the baseline (Table 19) the validation loss reaches its minimum within about five epochs and then rises steeply, a 2.7–2.8 nat increase by 32 epochs. It places the 22-epoch main budget well past the minimum. Replacement reaches an equal or slightly lower minimum and reduces the subsequent rise at both scales, from 2.73 to 2.04 nats at 336M and from 2.83 to 1.58 at 134M, the overfitting suppression of §6 seen along the epoch axis.

<table><tr><td>Scale</td><td>Cond</td><td> $L _ { \mathrm { m i n } }$ </td><td>epoch at min</td><td>final (32ep)</td></tr><tr><td>134M</td><td>none</td><td>4.97</td><td>5.3</td><td>7.80</td></tr><tr><td>134M</td><td>replace</td><td>4.84</td><td>8.0</td><td>6.42</td></tr><tr><td>336M</td><td>none</td><td>4.89</td><td>4.7</td><td>7.61</td></tr><tr><td>336M</td><td>replace</td><td>4.83</td><td>4.7</td><td>6.87</td></tr></table>

Table 19: Epoch sweep on Korean web, 32 epochs at the fixed recipe, one seed;final is the loss at 32 epochs. With one seed, the epoch of the minimum can differ from the three-seed means of Table 1.

![](images/900f77c6e746f9d5fd3599618d3e87487d9828abb2238f54426e9639f6d12104.jpg)  
Figure 6: Validation loss in nats over epochs on Korean web, one seed. Solid lines are NONE and dashed RE-PLACE; blue is 134M and red 336M. The dotted line marks the 22-epoch main budget.