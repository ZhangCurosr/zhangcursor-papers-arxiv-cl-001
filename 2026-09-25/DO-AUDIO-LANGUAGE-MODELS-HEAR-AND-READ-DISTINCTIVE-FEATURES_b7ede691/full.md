# DO AUDIO LANGUAGE MODELS HEAR AND READ DISTINCTIVE FEATURES ALIKE?

Yuanhao Chen Peter Chin

Thayer School of Engineering, Dartmouth College, Hanover, NH, USA yc.th@dartmouth.edu pc@dartmouth.edu

## ABSTRACT

Audio language models pass speech and text through a single decoder. We ask whether that decoder represents a distinctive feature in the same direction when a phoneme is heard and when it is read. For minimal pairs of phonemes differing in one feature, we take the offset between the two members mean representations. Averaging those offsets gives a direction for each stream, and we measure the cosine between the two. Because the two streams already agree about arbitrary phoneme pairs, we compare every measure against a reference built from random pairings rather than against zero. We apply this to 6 models, 7 features and 15 languages from 11 families. Only voicing in the two Qwen2.5-Omni models exceeds that reference after correction for multiple testing, and the reference varies by a factor of seven between models. In three of the six models, voicing has one direction in audio across the 14 languages with enough minimal pairs to measure it, and every language pair agrees in two of them. The model family, not the model size, predicts which stream represents a feature.

Index Terms— audio language models, phonological features, cross-modal representation, multilingual speech

## 1. INTRODUCTION

An audio language model (ALM) receives speech through an audio encoder and text through a tokeniser, and both streams pass through one decoder. We study whether that decoder represents a distinctive feature as one direction in both streams, or keeps a separate representation for each.

Work on self-supervised speech models suggests that distinctive features are encoded as directions. Choi et al. [1] show that distinctive features behave additively in such models, so that the offset between [d] and [t], added to [p], gives [b]. Phonetic information in these models also has a characteristic depth profile [2], and a multilingual one learns latent units shared across languages [3]. ALAS [4] runs an ALM on audio and on the transcript of the same audio, scoring per layer how well an audio frame’s hidden state identifies the text token it corresponds to. That measures the temporal binding between the two streams rather than the geometry of any one feature.

We take minimal pairs of phonemes differing in one distinctive feature, in 15 languages from 11 families, and compare the direction of their offsets when the phonemes are heard and when they are read. Every measure has a reference drawn from random pairings rather than a comparison against zero. After correction for multiple testing, only voicing in the two Qwen2.5-Omni models exceeds that reference, which itself differs more between models than any feature effect does. Three of the six models use one voicing direction in audio for every language in which voicing can be measured. Which stream represents a feature is predicted by the model family and not by the model size.

## 2. METHOD

## 2.1. Corpus, languages and models

We draw phonemes from the Language Documentation Reference Corpus [5]. It provides time-aligned segments of spoken narrative, labelled with a broad transcription in the International Phonetic Alphabet (IPA). We keep the rows that annotate speech and discard pauses and disfluency markers. Its transcription needs some canonicalisation before a feature table reads it (e.g., an ASCII g appears across 26,272 occurrences where the IPA voiced velar stop is expected).

A language enters the study on three conditions: (1) Its audio must be distributed with the corpus. (2) Its segments must be linked to recordings. (3) At least one of the 7 features in Table 1 must have three or more minimal pairs among phoneme types occurring at least 100 times, though two pairs suffice to measure a feature. Forty of the 47 languages qualify. We rank those by how many features meet that threshold, breaking ties in favour of a less represented family, and take the first 15. Table 1 lists them, across 11 families.

Phonemes are grouped into the utterance containing them. A group’s span runs from its first phoneme’s start to its last phoneme’s end, so the audio a model receives contains every phoneme we then locate within it.

We study Qwen2-Audio [6], Qwen2.5-Omni at two sizes [7], Gemma 4 at two sizes [8], and Audio-Flamingo 3 [9].

Table 1. The language sample. Features are those with at least two minimal pairs among phoneme types occurring at least 100 times: voi voicing, hi height, long length, back backness, sg spread glottis, nas nasality, cg constricted glottis.
<table><tr><td>Language</td><td>Family</td><td>Features</td></tr><tr><td>Baïnounk Gubëeher [10]</td><td>Atlantic-Congo Atlantic-Congo</td><td>back, long, voi back, hi, long, voi</td></tr><tr><td>Ruuli [11] Bora [12]</td><td>Boran</td><td>back, hi, long, sg</td></tr><tr><td>Cabécar [13]</td><td>Chibchan</td><td>back, hi, nas, voi</td></tr><tr><td>French (Swiss) [14]</td><td>Indo-European</td><td>back, nas, voi</td></tr><tr><td>Svan [15]</td><td>Kartvelian</td><td>cg, hi, long, voi</td></tr><tr><td>Texistepec Popoluca [16]</td><td>Mixe-Zoque</td><td>back, long, nas, voi</td></tr><tr><td>Sanzhi Dargwa [17]</td><td>Nakh-Daghestanian cg, hi, long, voi</td><td></td></tr><tr><td>Tabasaran [18] Anal [19] Sadu [20]</td><td>Nakh-Daghestanian Sino-Tibetan</td><td>cg, hi, voi long, sg, voi</td></tr><tr><td>Sümi [21]</td><td>Sino-Tibetan Sino-Tibetan</td><td>back, hi, sg, voi hi, sg, voi</td></tr><tr><td>Evenki [22]</td><td>Tungusic</td><td>hi, long, voi</td></tr><tr><td>Dolgan [23]</td><td>Turkic</td><td>back, hi, long, voi</td></tr><tr><td>Kamas [24]</td><td>Uralic</td><td>hi, long, voi</td></tr></table>

## 2.2. Locating a phoneme in audio and in text

To map decoder position to time we encode clips of several durations and count the positions holding the model’s audio placeholder. The count grows in proportion to duration, and the ratio gives milliseconds per position r.

For the audio stream we cut each recording to a group’s span and run one forward pass, keeping every layer. Measured from the group’s start, an occurrence running from $t _ { 1 }$ to t<sub>2</sub> occupies positions $[ \left\lfloor t _ { 1 } / r \right\rfloor , \lceil t _ { 2 } / r \rceil )$ , or the first position when that range is empty. We average the hidden states over that range.

For the text stream we write the same group as an IPA string between slashes, as broad transcription conventionally is written and as dictionaries give it, and pass it through the same decoder. A Sanzhi Dargwa phrase reads /muX:raj daxul/. We locate each occurrence by the character(s) it occupies, then average over every token whose span overlaps them, since a token may cover several symbols or only part of one.

## 2.3. Feature directions from minimal pairs

Each phoneme is described by a vector of binary distinctive features, read from a feature table [25]. A minimal pair is two phoneme types whose vectors differ in exactly one position. We always subtract the member $\boldsymbol { x } _ { i } ^ { - }$ that lacks the feature from the member $x _ { i } ^ { + }$ that has it, so that every pair of a feature is oriented alike.

Write $v ( \cdot )$ for a phoneme type’s mean representation at one layer. For pair $\{ x _ { i } ^ { + } , x _ { i } ^ { - } \}$ of a feature with n pairs, the offset is $d _ { i } = v ( x _ { i } ^ { + } ) - v ( x _ { i } ^ { - } )$ and its unit vector is $\hat { d } _ { i }$ . Subtraction removes whatever the two members share, including the mean of the whole space, which is large in these anisotropic representations [26, 27].

Their agreement A is the mean cosine between those unit offsets,

$$
A = \frac { 2 } { n ( n - 1 ) } \sum _ { i < j } \langle \hat { d } _ { i } , \hat { d } _ { j } \rangle .\tag{1}
$$

$A = 1$ means every pair points the same way. The feature’s direction is $u = \bar { d } / \| \bar { d } \|$ for $\begin{array} { r } { \bar { d } = \frac { 1 } { n } \sum _ { i } \hat { d } _ { i } } \end{array}$ . We compute A and u separately for the audio and the text stream.

Our primary measure is the cosine similarity between a feature’s directions in the two streams,

$$
c = \langle u _ { \mathrm { a u d i o } } , u _ { \mathrm { t e x t } } \rangle .\tag{2}
$$

Both are taken at the same layer, and no alignment step is needed, since both directions lie in the same decoder’s space.

## 2.4. What each measure is compared against

For arbitrary pairs, the distribution of A is wider when there are fewer pairs, and that of c is not centred on zero, because one pair’s offsets in audio and in text both reflect which phonemes the pair contains. Neither distribution can be computed, so we build a reference by pairing at random, in the manner of a control task [28]. For each of $B = 2 0 0 0$ repetitions we draw, uniformly and without replacement, as many pairs of distinct phoneme types as the feature itself has, and recompute the quantities above. Each pair’s members are taken in the order drawn, so its orientation, like the pairing, does not come from a feature. Within a repetition the same pairing is used on both streams, since pairing them independently would compare directions built from different phonemes and lower the reference. With k of the B repetitions reaching the observed value, we report $p = ( k + 1 ) / ( B + 1 )$ [29].

We also take the cosine of a feature’s audio direction with each other feature’s text direction. The largest of those is its cross-feature value, and comparing c against it asks whether the agreement is specific to that feature.

## 2.5. Testing across languages

The language is our unit of replication. For a feature we take the median c over languages at each layer, then the largest of those medians. The layer is not fixed in advance, since the depth at which the two streams agree most differs between models.

The reference therefore has to account for that choice. The bth repetition from §2.4 uses its own pairing in every language and keeps it at every layer. We then take the median over languages at each layer, and the largest of those medians. The repetitions are independent across languages, so how they are paired does not matter. We test every model against every feature, so we report a Benjamini–Hochberg q [30] across those 42 tests alongside p. The fractions in Table 3 describe where a comparison holds across depth rather than adding hypotheses, so we leave their p uncorrected.

Table 2. Median c per model and feature, at the layer where the median over languages is largest. Cells give that median (c¯), the 95th percentile of its reference $( c _ { 9 5 } ) ,$ , the number of languages with a positive c (+), and the number of languages whose c exceeds its largest value against another feature (×). An asterisk mark $q \leq 0 . 0 5$ over the 42 tests.
<table><tr><td></td><td colspan="4">Voicing (14 lang.)</td><td colspan="4">Height (11 lang.)</td><td colspan="4">Length (10 lang.)</td><td colspan="4">Backness (8 lang.)</td></tr><tr><td>Model</td><td>č</td><td>C95</td><td>+ </td><td>X</td><td>c</td><td>C95</td><td>十</td><td>X</td><td>č</td><td>C95</td><td>十</td><td>X</td><td>é</td><td>C95</td><td>+ ×</td><td></td></tr><tr><td>Qwen2-Audio 7B</td><td>+0.54</td><td>+0.75</td><td>12</td><td>7</td><td>+0.68</td><td>+0.80</td><td>8</td><td>5</td><td>+0.36</td><td>+0.78</td><td>8</td><td>3</td><td>+0.10</td><td>+0.78</td><td>7</td><td>1</td></tr><tr><td>Qwen2.5-Omni 7B</td><td> $+ 0 . 4 0 ^ { * }$ </td><td>+0.34</td><td>14</td><td>14</td><td>+0.27</td><td>+0.35</td><td>11</td><td>11</td><td>+0.17</td><td>+0.35</td><td>10</td><td>6</td><td>+0.27</td><td>+0.34</td><td>6</td><td>5</td></tr><tr><td>Qwen2.5-Omni 3B</td><td> $+ 0 . 4 5 ^ { * }$ </td><td>+0.35</td><td>14</td><td>14</td><td>+0.20</td><td>+0.36</td><td>11</td><td>11</td><td>+0.10</td><td>+0.35</td><td>8</td><td>5</td><td>+0.27</td><td>+0.35</td><td>7</td><td>7</td></tr><tr><td>Gemma 4 E4B</td><td>+0.07</td><td>+0.11</td><td>9</td><td>7</td><td>+0.08</td><td>+0.12</td><td>8</td><td>4</td><td>+0.08</td><td>+0.12</td><td>8</td><td>3</td><td>+0.10</td><td>+0.13</td><td>7</td><td>5</td></tr><tr><td>Gemma 4 E2B</td><td>+0.11</td><td>+0.12</td><td>12</td><td>7</td><td> $+ 0 . 0 8 + 0 . 1 3$ </td><td></td><td>8</td><td>2</td><td>+0.11</td><td>+0.13</td><td>7</td><td>4</td><td>+0.07</td><td>+0.14</td><td>7</td><td>0</td></tr><tr><td>Audio-Flamingo 3</td><td>+0.18</td><td>+0.20</td><td>13</td><td>13</td><td>+0.15</td><td>+0.22</td><td>11</td><td>10</td><td>+0.08</td><td>+0.21</td><td>7</td><td>2</td><td>+0.16</td><td>+0.22</td><td>7</td><td>5</td></tr></table>

We also ask whether a feature is one direction in a model or one per language, since a direction separating voiced from voiceless phonemes in one language might reflect that inventory rather than voicing in general. For a feature and a stream, the across-language agreement is the mean cosine between the languages’ directions, which is Eq. (1) applied across languages rather than across the pairs of one feature. Its reference shuffles, within each language, which feature each of that language’s directions belongs to. Every language therefore keeps the directions it had, and only the correspondence of features between languages is destroyed.

## 3. RESULTS

## 3.1. Features with matching directions in audio and in text

Voicing has matching directions in the two Qwen2.5-Omni models, $\mathrm { a t + 0 . 4 0 }$ and +0.45 with $q = 0 . 0 1 1$ (Table 2). All 14 languages are positive in both (Figure 1), and every language also exceeds its cross-feature value. No other combination of model and feature reaches $q \leq 0 . 0 5$ . Four of the 42 combinations reach $p \leq 0 . 0 5$ against 2.1 expected, and the other two have only three and four languages, exceeding their reference by 0.03 or less.

Qwen2-Audio’s voicing median is the largest in the table $\mathrm { a t + 0 . 5 4 }$ , and its reference is +0.75. Its two streams already agree about arbitrary phoneme pairs more than they agree about voicing. The reference differs more between models than any feature effect does, from +0.11 in Gemma 4 E4B to +0.75 in Qwen2-Audio. No median exceeds its own reference by more than 0.10.

The reference and the cross-feature value in Table 2 answer different questions. Qwen2.5-Omni-7B has all 11 languages positive for height and all 11 exceeding their cross-feature value, yet its +0.27 falls below a reference of +0.36. A feature can be the best-matching one and still be unremarkable against arbitrary phoneme pairs.

## 3.2. Which stream represents the features better

The models fall into two patterns (Table 3). A stream counts as representing a feature at a layer when A there exceeds its reference, and the Wthn columns give the median over languages of that count, as a fraction of the model’s depth. The two Qwen2.5-Omni models represent voicing when they hear it and not when they read it, at 0.84 and 0.85 of their layers versus no layers. The Gemma models show the reverse for height and length, representing them at 0.39 to 1.00 of their layers when reading and at 0.00 to 0.08 when hearing.

Table 3. Fraction of a model’s layers at which each comparison holds, taking $p \leq 0 . 0 5$ as the criterion. Wthn: A stream’s own offsets agree more than arbitrary pairings of the same phonemes, median over languages. Acrs: The languages directions agree more with each other than with other features directions. In each pair of rows the larger of the two values is set in bold.
<table><tr><td colspan="2"></td><td colspan="2">Voicing</td><td colspan="2">Height</td><td colspan="2">Length</td></tr><tr><td>Model</td><td>Stream</td><td>Wthn Acrs</td><td></td><td>Wthn Acrs</td><td></td><td>Wthn Acrs</td><td></td></tr><tr><td rowspan="2">Qwen2-Audio 7B</td><td>heard</td><td>0.09</td><td>0.30</td><td>0.00</td><td>0.00</td><td>0.21</td><td>0.97</td></tr><tr><td>read</td><td>0.00</td><td>0.21</td><td>0.09</td><td>0.06</td><td>0.70</td><td>1.00</td></tr><tr><td rowspan="2">Qwen2.5-Omni 7B</td><td>heard</td><td>0.84</td><td>1.00</td><td>0.52</td><td>1.00</td><td>1.00</td><td>0.83</td></tr><tr><td>read</td><td>0.00</td><td>0.17</td><td>0.14</td><td>0.14</td><td>0.19</td><td>1.00</td></tr><tr><td rowspan="2">Qwen2.5-Omni 3B</td><td>heard</td><td>0.85</td><td>1.00</td><td>0.51</td><td>1.00</td><td>1.00</td><td>0.97</td></tr><tr><td>read</td><td>0.00</td><td>0.24</td><td>0.22</td><td>0.11</td><td>0.26</td><td>1.00</td></tr><tr><td rowspan="2">Gemma 4 E4B</td><td>heard</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.08</td><td>0.51</td></tr><tr><td>read</td><td>0.05</td><td>0.84</td><td>0.42</td><td>0.42</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">Gemma 4 E2B</td><td>heard</td><td>0.00</td><td>0.03</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.83</td></tr><tr><td>read</td><td>0.07</td><td>0.83</td><td>0.39</td><td>0.33</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">Audio-Flamingo 3</td><td>heard</td><td>0.00</td><td>1.00</td><td>0.03</td><td>0.07</td><td>0.98</td><td>0.97</td></tr><tr><td>read</td><td>0.00</td><td>0.17</td><td>0.14</td><td>0.14</td><td>0.17</td><td>1.00</td></tr></table>

Each family includes two model sizes, so the family and not the size predicts which pattern a model shows. The Qwen models initialise their audio encoder from Whisper [31] and train it against the decoder [6, 7], whereas Gemma uses a Conformer encoder that stays frozen throughout pre-training [8]. With two families we cannot separate the effect of the encoder’s architecture from whether it was trained.

Model size does not predict the median c either. For voicing the smaller model of each pair has the larger median $c , + 0 . 4 5$ against +0.40 in Qwen2.5-Omni and +0.11 against +0.07 in Gemma, while for height and length the larger Qwen2.5-Omni model has the larger median (Table 2).

The Acrs columns ask something else: whether a model uses one direction for a feature in every language, or a separate direction in each. A single value exists per layer here, so those columns give the fraction of layers directly rather than a median over languages. The two questions have different answers. Gemma 4 E2B represents length when it hears it at only 0.01 of its layers, yet at 0.83 of them the languages agree on one length direction. A language’s direction for a feature is the mean of several offsets, and a mean can point consistently even when the offsets it averages disagree with each other. Representing a feature within a single language is therefore not a precondition for sharing a direction across languages.

![](images/9bfe5a1187963e3e1064721c25e21f27c93a38b8aa559e08b20e41302ddfa345.jpg)  
Fig. 1. c for each language, at the layer where the median over languages is largest. Bars mark the median and an asterisk marks $q \leq 0 . 0 5$ . Models are Qwen2-Audio 7B (Q2A), Qwen2.5-Omni 7B and 3B (O7B, O3B), Gemma 4 E4B and E2B, and Audio-Flamingo 3 (AF3).

## 3.3. One direction across languages, or one per language

In three of the six models the languages’ voicing directions in audio agree more than the reference at every layer (Table 3). They agree at +0.54 and +0.57 in the two Qwen2.5-Omni models and at +0.52 in Audio-Flamingo 3, with that reference within 0.01 of zero. In the two Qwen2.5-Omni models every one of the 91 language pairs agrees, the weakest at +0.31, whereas two of Audio-Flamingo 3’s pairs point opposite ways. Qwen2-Audio reaches +0.49, but at only 10 of its 33 layers, and the two Gemma models reach +0.19 and +0.15 at one layer each.

In Audio-Flamingo 3 the median language has no layer at which its own voicing offsets exceed their reference, yet the languages agree with each other at every layer. Across all models and features, a feature’s agreement within a language and its across-language agreement correlate at +0.67 in audio and +0.73 in text. A model that represents a feature within languages usually shares a direction across them as well.

Length in text has the highest across-language agreement of any feature, +0.88 or higher in every model, and it peaks at the first or second layer in all six. Length is the one feature represented by adding the length mark (:) rather than by changing a symbol, so the agreement is present in the token embeddings rather than built by the decoder. A shared written mark can give a trivially shared direction, without any phonological knowledge.

## 4. DISCUSSION

A feature’s offset uses every occurrence of one phoneme against every occurrence of another, so any systematic difference in where the two occur enters the offset, and the design cannot remove that. We measure that difference as the distance between the two members’ distributions over neighbouring phonemes, and its correlation with c is −0.01 for voicing, +0.05 for height and +0.06 for length. At the embedding layer c never exceeds +0.10 for any model or feature, so the voicing agreement in Qwen2.5-Omni is built by the decoder rather than inherited from the tokeniser. Splitting every phoneme’s occurrences in two gives two estimates of each direction, agreeing at a median of 0.82 in audio and 0.96 in text, so most estimates are well determined even though the weakest are not.

A model that mapped audio into a rotated copy of its text space would preserve every distance between phonemes while giving a generic direction zero cosine with its counterpart. That is not what we observe for voicing in the two Qwen2.5- Omni models, though we cannot rule out a rotation that leaves some directions fixed. A c near zero elsewhere need not mean the two streams share no structure at all. We check with representational similarity analysis (RSA) [32], correlating the two streams’ phoneme-similarity matrices by rank against a reference that shuffles which phoneme corresponds to which. This comparison needs no minimal pairs, so all 15 languages contribute, and every model has a layer at which they all exceed that reference.

Both directions in c are taken at the same layer, and in 12 of the 18 combinations in Table 3 no layer has a majority of languages exceeding the reference in both streams. A feature encoded early in audio and late in text would therefore read as absent, and ruling that out needs a search over pairs of layers.

After correction, agreement between the two streams holds only for voicing and only in the two Qwen2.5-Omni models, where comparing against zero would have found it in 15 of the 42 combinations instead. Agreement across languages is wider. In three of the six models the languages agree on one voicing direction in audio.

## 5. REFERENCES

[1] Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, et al., “[b] = [d] - [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic,” in Findings of ACL, 2026, pp. 11048–11069.

[2] Ankita Pasad, Ju-Chieh Chou, and Karen Livescu, “Layer-Wise Analysis of a Self-Supervised Speech Representation Model,” in IEEE ASRU Workshop, 2021, pp. 914–921.

[3] Alexis Conneau, Alexei Baevski, Ronan Collobert, et al., “Unsupervised Cross-lingual Representation Learning for Speech Recognition,” in Proc. Interspeech, 2021.

[4] Pooneh Mousavi, Yingzhi Wang, Mirco Ravanelli, and Cem Subakan, “ALAS: An Automatic Latent Alignment Score for Audio Language Models,” 2026.

[5] Ludger Paschen, François Delafontaine, Christoph Draxler, et al., “Building a Time-Aligned Cross-Linguistic Reference Corpus from Language Documentation Data (DoReCo),” in Proc. LREC, 2020, pp. 2657– 2666.

[6] Yunfei Chu, Jin Xu, Qian Yang, et al., “Qwen2-Audio Technical Report,” 2024.

[7] Jin Xu, Zhifang Guo, Jinzheng He, et al., “Qwen2.5- Omni Technical Report,” 2025.

[8] Gemma Team, “Gemma 4 Technical Report,” 2026.

[9] Sreyan Ghosh, Arushi Goel, Jaehyeon Kim, et al., “Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models,” in Proc. NeurIPS, 2025.

[10] Alexander Yao Cobbinah, “Baïnounk gubëeher doreco dataset,” 2024.

[11] Alena Witzlack-Makarevich, Saudah Namyalo, Anatol Kiriggwajjo, and Zarina Molochieva, “Ruuli doreco dataset,” 2024.

[12] Frank Seifart, “Bora doreco dataset,” 2024.

[13] Juan Diego Quesada, Stavros Skopeteas, Carolina Pasamonik, Carolin Brokmann, and Florian Fischer, “Cabécar doreco dataset,” 2024.

[14] Mathieu Avanzi, Marie-José Béguelin, Gilles Corminboeuf, Federica Diémoz, and Laure Anne Johnsen, “French (swiss) doreco dataset,” 2024.

[15] Jost Gippert, “Svan doreco dataset,” 2024.

[16] Søren Wichmann, “Texistepec popoluca doreco dataset,” 2024.

[17] Diana Forker and Nils Norman Schiborr, “Sanzhi dargwa doreco dataset,” 2024.

[18] Natalia Bogomolova, Dmitry Ganenkov, and Nils Norman Schiborr, “Tabasaran doreco dataset,” 2024.

[19] Pavel Ozerov, “Anal doreco dataset,” 2024.

[20] Xianming Xu and Bibo Bai, “Sadu doreco dataset,” 2024.

[21] Amos Teo, “Sümi doreco dataset,” 2024.

[22] Olga Kazakevich and Elena Klyachko, “Evenki doreco dataset,” 2024.

[23] Chris Lasse Däbritz, Nina Kudryakova, Eugénie Stapert, and Alexandre Arkhipov, “Dolgan doreco dataset,” 2024.

[24] Valentin Gusev, Tiina Klooster, Beáta Wagner-Nagy, and Alexandre Arkhipov, “Kamas doreco dataset,” 2024.

[25] David R. Mortensen, Patrick Littell, Akash Bharadwaj, et al., “PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors,” in Proc. COLING, 2016, pp. 3475–3484.

[26] Kawin Ethayarajh, “How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings,” in Proc. EMNLP-IJCNLP, 2019, pp. 55–65.

[27] William Timkey and Marten van Schijndel, “All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality,” in Proc. EMNLP, 2021, pp. 4527–4546.

[28] John Hewitt and Percy Liang, “Designing and Interpreting Probes with Control Tasks,” in Proc. EMNLP-IJCNLP, 2019, pp. 2733–2743.

[29] Belinda Phipson and Gordon K. Smyth, “Permutation Pvalues should never be zero: Calculating exact P-values when permutations are randomly drawn,” Stat. Appl. Genet. Mol. Biol., vol. 9, pp. Article39, 2010.

[30] Yoav Benjamini and Yosef Hochberg, “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” J. R. Stat. Soc. B, vol. 57, no. 1, pp. 289–300, 1995.

[31] Alec Radford, Jong Wook Kim, Tao Xu, et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” in Proc. ICML, 2023, pp. 28492–28518.

[32] Nikolaus Kriegeskorte, Marieke Mur, and Peter A. Bandettini, “Representational similarity analysis - connecting the branches of systems neuroscience,” Front. Syst. Neurosci., vol. 2, 2008.