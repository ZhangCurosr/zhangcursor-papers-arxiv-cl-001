# On the information rate of phones, syllables and words

Herman Kamper

Electrical and Electronic Engineering, Stellenbosch University, South Africa

## Abstract

Which linguistic units minimise the bitrate required to encode speech content: phones, syllables, words, maybe larger units, or something in between? This question has implications for the development of speech and language systems and theories of human language processing. Using a transcribed corpus of 960 hours of English audiobooks, I compare the information rates of phones, syllables, words, multi-word combinations and intermediate tokenisations. While all units are equivalent under a perfect language model, it makes sense to compare these under limited modelling budgets; I use n-gram and transformer language models. The lowest bitrates are achieved using words, with slightly longer units sometimes performing best. The results show that, when modelling capacity is limited, units larger than syllables but smaller than long multi-word phrases provide the most efficient representation in terms of information rate.

## 1. Introduction

Suppose I have a spoken utterance and I want to send you the content of that utterance. I write the utterance down phonetically, I encode the phones in bits, you receive the bits, and (because we agreed beforehand on an encoding scheme) you can decode the bits and know what speech sounds were in the utterance. For example, if the utterance is the staircase groaned, I would encode DH AH S T EH R K EY S G R OW N D, as shown in the phone tier of Figure 1. For our encoding, we assign bit sequences to each phone unit based on the probability of the unit, meaning our scheme uses the minimum number of bits on average [1–3]. We measure how many bits this optimal phone-based encoding uses.

What if, rather than phones, we decide to tokenise the utterance into syllables? The goal is still to transmit the phonetic content, but we just do it with a different set of units, e.g. DH-AH S-T-EH-R K-EY-S G-R-OW-N-D. Using the optimal syllable-tobits encoding scheme, I again encode the units, send the bits, you decode the bits into syllables, and then you decode the syllables into phones. A different tokenisation, but we’ve communicated exactly the same content. Is this more efficient than the phone encoding? Rather than syllables, why not words? Or maybe we can communicate even more efficiently with multi-word units like DH-AH-S-T-EH-R-K-EY-S (the staircase); if in my data these two words often co-occur, I could save bits by encoding these together (Figure 1 top). By now you probably get the picture.

This paper asks which units – phones, syllables, words, units longer than words, something in between – are the most efficient for communicating linguistic content. Experiments are done on transcribed phonetic sequences (from forced alignments) on English data (audiobooks from the LibriSpeech [4] corpus).

Why do this comparison? First, the literature does not report the relative efficiencies of these different units. Studies report bitrates like 50 bits per second (bps) for phones [5, 6] or 39 bps for syllables [7]. But these are not directly comparable since information rate is defined differently in each study and is not estimated on the same data.

A second motivation for such a comparison is its relevance to human language processing, which is likely shaped by pressures for efficient communication [8]. I do not want to state this too strongly, because this paper presents a purely informationtheoretic analysis ignoring human mental effort, how the mental message is converted into a phonetic sequence, production and perception effort, and channel noise. Nevertheless, characterising relative efficiency is relevant to theories of linguistic representation, such as how the mental lexicon is organised [9].

The third motivation is for the zero-resource speech processing community. Over more than a decade, this community has tried to learn linguistically meaningful units from unlabelled speech without any transcriptions [10]. Studies have looked at the discovery of phone-like units [11, 12], syllabic units [13–15], words [16, 17], and even phrase-like units [18]. But why should we be interested in particular units rather than others? None of this paper deals with speech (all experiments are on transcribed phonetic sequences). But still, as guidance for this community, it is worth knowing whether some units are more efficient than others in an idealised communication setting.

After defining information rate, I first show that all units, in fact, have the same bitrate under a perfect model: different tokenisations of the same input are just different encodings of the same underlying events. But this is only true with a perfect model – something we never have. I therefore refine the research question to consider the different information rates when using real models to estimate the probabilities. The models are n-gram language models [19] and transformer language models [20].

Apart from phones, syllables and words, I also apply byte pair encoding (BPE) [21] at the word-token level, generating multi-word units, like the one for the staircase in the top tier of Figure 1. Larger units are always expressed in terms of their phones to ensure a one-to-one correspondence between tokenisations, as shown in the figure. Intermediate sanity checks, like applying subword BPE on phone units, are also done.

![](images/63159b104384292410b0ffe87a4435ebc510652368e8af8f620ea7afbe1fae96.jpg)  
Figure 1. This work compares the information rates of different linguistic units when used to encode the same phonetic content.

I find that the best bitrates are achieved with a 20Mparameter transformer using word units (28.07 bps). With ngrams, adding a small number of word-BPE merges improves results over isolated words. For both model classes, if too many word merges are added, bitrate gets worse. Under a constrained modelling budget, it therefore seems that wordlike units strike a good balance, sitting between a small set of shorter-duration units (e.g. phones; best bitrate 29.56 bps) vs having a larger set of longer units (e.g. word-BPE combinations; 29.27 bps when adding 5,000 word merges). Code is released at https://github.com/kamperh/inforate/.

## 2. Information rate

Information rate, measured in bits per second, is defined as [7]:

$$
{ \mathrm { b i t r a t e } } = { \frac { \mathrm { b i t s } } { \mathrm { d u r a t i o n } } } = { \frac { \# { \mathrm { u n i t s } } } { \mathrm { d u r a t i o n } } } \times { \mathrm { b i t s ~ p e r ~ u n i t } }
$$

To know the average bits per unit, we need to decide on an encoding scheme to assign bit sequences to unit types. If we use entropy coding, where the number of bits is chosen based on the probability of the unit, then the bits per unit would be given by the entropy of the distribution over the units [1–3]. But, to do this, we would need to know the true unit probabilities $P ( w _ { t } | w _ { 1 : t - 1 } )$ . We don’t have these. Instead we have a model $P _ { \theta } ( w _ { t } | w _ { 1 : t - 1 } ) .$ Under this model, we can then estimate the number of bits on held-out data (consisting of $_ T$ tokens):

$$
{ \mathrm { b i t s ~ p e r ~ u n i t } } = - { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \log _ { 2 } P _ { \theta } ( w _ { t } | w _ { 1 : t - 1 } )
$$

Formally, this is the estimated cross-entropy rate when we use model $P _ { \theta }$ to encode units coming from the true distribution $P .$ We can now measure the bitrate under the model:

$$
\mathrm { b i t r a t e } = \frac { \mathrm { b i t s } } { \mathrm { d u r a t i o n } } = - \frac { 1 } { D } \sum _ { t = 1 } ^ { T } \log _ { 2 } P _ { \theta } ( w _ { t } | w _ { 1 : t - 1 } )\tag{1}
$$

where D is the data duration in seconds.

Research question. What are the relative coding efficiencies of different linguistic units (phones, syllables, words, etc.)? Essentially we are asking what w should be to give low bitrates.

Other bitrate calculations. Bitrate has been calculated differently elsewhere. The Zero Resource Speech Challenges, for instance, use [22]:

$$
{ \mathrm { b i t r a t e } } = - { \frac { T } { D } } \sum _ { w } P _ { \theta } ( w ) \log _ { 2 } P _ { \theta } ( w )
$$

Here $P _ { \theta } ( w )$ are unigram probabilities estimated from counts on the evaluation data; bitrate is taken as the expected value under this distribution. Coupe et al. [´ 7] do something similar, but with a second-order model that takes the preceding token into account. Both these match (1) if we (a) make a first-/second-order ngram modelling assumption, and (b) calculate bitrate on the same data used for the unigram/bigram model estimates. This is reasonable for coarse comparisons, but (a) units can presumably be encoded much more efficiently with larger contexts using a higher capacity $P _ { \theta } ( w )$ , and (b) these measurements do not account for generalisation to unseen data.

## 3. But units don’t matter: they all have the same ideal bitrate

We now have an appropriate way to measure information rate. But there is still a question of what the model $P _ { \theta }$ should look like. I now briefly digress to show that, under an ideal model, all units will have the same bitrate. But a comparison is still meaningful if the model $P _ { \theta }$ is constrained.

For illustration, let us assume we are using orthographic words as units, and are wondering whether we should not rather use larger units. For instance, maybe in our data the word York is always preceded by New. Can’t we save bits by having one unit New York instead of two separate units New and York, each separately taking up bits? No, it doesn’t matter how you tokenise the input; the following should hold:

$$
\begin{array} { r l } & { P ( N e w ~ Y o r k ~ i s ~ b e a u t i f u l ) } \\ & { \quad = P ( N e w ) \cdot P ( Y o r k | N e w ) \cdot P ( i s | N e w ~ Y o r k ) \cdot \cdot \cdot } \\ & { \quad = P ( N e w ~ Y o r k ) \cdot P ( i s | N e w ~ Y o r k ) \cdot \cdot \cdot } \end{array}
$$

But how does this make sense, given that we should have saved bits? Turns out that if York always followed New (and no other word did), then $P ( Y o r k | N e w ) = 1$ . And so $P ( N e w Y o r k ) =$ $P ( N e w )$ . In information-theoretic terms: it takes zero extra bits to transmit York. More formally, different tokenisations correspond to different random variables defined on the same underlying sample space.

So, apparently, the question of the relative information rates of different units is silly – they are all the same. But this relies on a strong assumption: that we know the true probability distribution ${ \dot { P } } ,$ the actual distribution governing the world. But no one (except the Creator) actually knows these probabilities. So we always have a model $P _ { \theta }$ . This model will be finite. And this model would be tied to a particular type of unit. And now, under this limited capacity, it matters how you tokenise your input.

Research question refined. What are the relative coding efficiencies of different linguistic units under a limited modelling budget?

## 4. Models

I consider two classes of finite models: n-gram language models [19, 23] and transformer-based neural language models [20].

## 4.1. Classic n-gram language models

For the n-gram models, I use absolute discounting in which higher-order n-gram language models are interpolated with lower-order models [19]. Discounted mass is used to deal with unit types that are unseen at training time.

Formally, for the highest order n, we have [19, §2.6]:

$$
P _ { n } ( w _ { t } | h ) = \frac { \operatorname* { m a x } \{ c ( h , w _ { t } ) - d , 0 \} } { c ( h ) } + \frac { d N _ { 1 + } ( h \cdot ) } { c ( h ) } P _ { n - 1 } ( w _ { t } | h ^ { \prime } )
$$

where d is the discount value, $h \ : = \ : w _ { t - n + 1 : t - 1 }$ denotes the preceding units, $h ^ { \prime } = w _ { t - n + 2 : t - 1 }$ drops the first symbol from $h ,$ , and $\bar { N _ { 1 + } } ( h \cdot ) = | \{ w : c ( h , w ) > 0 \} |$ is the number of distinct continuations after h. This recurses down to the unigram case:

$$
P _ { 1 } ( w _ { t } ) = \frac { \mathrm { m a x } \{ c ( w _ { t } ) - d , 0 \} } { T } + \frac { V d } { T } P _ { 0 } , \quad P _ { 0 } = \frac { 1 } { V + 1 }
$$

The +1 in the denominator of $P _ { 0 }$ reserves probability mass for unseen types.

Instead of a single discount value d for all orders, these are set per order [24]: $d = n _ { 1 } / ( n _ { 1 } + 2 n _ { 2 } )$ , with $n _ { 1 }$ and $n _ { 2 }$ the number of distinct n-grams occurring exactly once and exactly twice in the training data (at this order). If these are zero, $d = 0 . 7 5$ is used as a fall-back (I did not tune this value). The discount is also clamped to the range $d \in [ 0 . 1 , 0 . 9 ]$ to avoid strange values.

When dealing with syllables and words, we have to deal with units that are not seen during training. For these, I back off to a spelling model, paying bits to spell out the unit token into its constituent phones. So if we have a token G-R-OW-N-D (groaned) that has never been seen, it will be decomposed into five tokens G R OW N D (with an end-of-word symbol). This spelling model is itself an absolute discounted n-gram using the interpolation as above, but trained on spelled-out tokens.

Implementation. In development experiments, I varied the ngram order (from 1 to 5) for all the different units. For most of the units, trigrams worked the best, and I therefore use this throughout. I also always use a third-order spelling model. This is then our fixed n-gram modelling budget.

## 4.2. Transformer language models

I use a decoder-only autoregressive transformer language model based on a modification of the OPT-125M architecture [25]. I reduced the model size so that the body (excluding the output unit-predicting layer) has approximately 20M parameters. This is kept constant regardless of the input unit. Each model is a standard causal transformer using stacked blocks with masked multi-head self-attention and a feedforward layer, pre-layernorm, ReLU activations, learned positional embeddings, and tied input and output embeddings. I use 6 transformer layers with 512 hidden dimensions, 8 heads, feedforward networks with 2048 dimensions, and a context window of 2048 tokens. The output layer is different depending on the input unit. More detail is given below, but the phone neural language model only has 39 outputs (together with sentence-boundary and pad tokens) while the word model has roughly 56k outputs.<sup>3</sup>

Implementation. The neural models are trained with an effective batch size of 40 on a single NVidia RTX3070 (8 GB). I use AdamW optimisation [27] with a linear warmup for 8% of the steps to a maximum learning rate of $1 \cdot 1 0 ^ { - 3 }$ followed by a cosine decay to $2 \cdot 1 0 ^ { - 5 }$ . Models are trained for approximately 20 epochs; the step sizes are changed based on the unit to ensure that the equivalent of roughly 20 passes is made over the encoded data. All these values are based on the recipe from [28].

## 5. Data and units

I perform experiments on the multi-speaker English Libri-Speech [4] corpus of read audiobooks. Models are trained on forced alignments of the 960-hour training set and evaluated on dev-clean (5.4 hours from 40 speakers). On this data, I compare the bitrates of phones, syllables, words, multi-word combinations obtained from BPE at the word-token level, as well as intermediate units. The number of tokens and types for phones, syllables and words in the training data are shown in Table 1.

Table 1. Number of tokens and types for each unit on the Libri-Speech 960h training set.
<table><tr><td>Units</td><td>Tokens</td><td>Types</td><td>Non-singleton types</td></tr><tr><td>Phones</td><td>33,464,416</td><td>39</td><td>39</td></tr><tr><td>Syllables</td><td>13,025,947</td><td>13,656</td><td>11,386</td></tr><tr><td>Words</td><td>9,403,427</td><td>84,163</td><td>56,255</td></tr></table>

The phone-level forced alignments [29] use an ARPABet phoneset without stress markers, giving a set of 39 phones. To obtain syllable units from the phone-level forced alignments, a rule-based method is used [30], yielding within-word syllables, as was also done in [28].

Word types are defined based on their unique phonetic realisations, e.g. two and too are both represented as the type T-UW. This is needed so that the exact phone sequence is recoverable from a word tokenisation – a one-to-one correspondence necessary for proper bitrate comparisons. For the word transformer models, only types occurring more than once in the training data are used in the output vocabulary, giving an output layer of 56k types (last column and row in Table 1). Singletons are treated as unseen and are spelled out as described above.

Maybe the most efficient units are something larger than words, like phrases or common multi-word combinations? To test this, I use byte pair encoding (BPE) [21, 31], but rather than applying standard BPE where co-occurring characters are combined, I apply BPE at the word-token level to find new multiword combinations. New types like AH-V-DH-AH (of the) are therefore added to the vocabulary. I again constrain the starting vocabulary to words occurring at least twice (56k types), and then add the new merged units. I consider 200, 1,000 and 5,000 merges. For the transformer models, this increases the output vocabulary by the merge count. Unseen units are again spelled out as phones. With 5,000 merges, the longest unit is AA-N-DH-IY-AH-DH-ER-S-AY-D-AH-V-DH-AH (on the other side ofthe).

## 6. Results

## 6.1. Main results

The goal of this paper is to see which unit has the best information rate under a constrained modelling budget. For the main comparison, we look at phones, syllables, words and different word-BPE tokenisations (200, 1,000 and 5,000 merges). We consider two model classes: third-order n-gram language models and transformer language models. The results are respectively given in Figures 2a and 2b. Error bars for the transformer model are standard deviations across five training seeds.

Under both the n-gram and transformer models, we see that a unit larger than syllables but smaller than long multi-word combinations gives the lowest bitrates. For the n-gram model, this is the word-BPE 200 units, while for the transformer, this is word units. The latter achieves the best overall bitrate of 28.07 bps among models trained on phonetic realisations.

Although the differences for the n-gram models are small, bootstrap confidence estimates [32] reveal that all pairwise differences are significant at the 95% level, except for the word vs the word-BPE 1,000 models (which makes sense given that their results are both 31.05 bps). Bootstrap confidence interval estimation also shows that pairwise differences between adjacent transformer models are statistically significant in all cases. (This is despite some overlap in the error bars over different seeds, but these bars do not constitute a paired comparison.)

![](images/f0a8c80b05e538c9ec6b1bb55723eafe6e5ea4839b95b1a0fe6eb35ccdfbada3.jpg)

![](images/c25b43320cfc51ef9cfa1e29e10aba60d847de654bacfa40c416a6a57e9fbcc2.jpg)

![](images/54f754d7b567c2cc070b715a2311b9aa5224e201808f137b55a3a5a72bc45f90.jpg)  
Figure 2. Bitrates for different linguistic units. In the main experiments, (a) n-gram and (b) transformer models are used to determine the bitrates of different tokenisations using phones as the atomic unit. In sanity-check experiments, (c) a transformer is used on different units using characters as the atomic unit. The hatched bar is the best performance in each panel and error bars for the transformers are standard deviations over five seeds. Vertical axes differ between panels, as the bitrates are not directly comparable.

## 6.2. Additional experiments and sanity checks

My goal was to compare the information rates of the linguistic units above, but other (intermediate) units and comparisons are useful for supporting the above results and their interpretation.

The first of these uses the more standard form of BPE, where atomic units (normally characters but here I use phones) are merged to form new subword units. Concretely, I apply BPE to the phone units to obtain new types like AH-N (an) and DH-AH (the). I merge to get a vocabulary of 2,000 and disallow merges across word boundaries (as is standard in BPE). The hypothesis would be that the bitrate of this tokenisation would lie somewhere between phones and words (maybe close to syllables), and this is indeed the case: 31.59 bps under the n-gram and 28.98 ± 0.279 bps under the transformer.

The results so far are obtained using phones as the smallest atomic unit. As a sanity check, I repeat the analysis on characters. Words are now represented in their orthographic form so that two and too are now separate entries in the vocabulary. Results are given in Figure 2c when using transformers (20M parameters) to measure bitrate for characters, subword-BPE-merged characters (2,000 types), orthographic words, and word tokens merged with BPE (200, 1,000 and 5,000 merges). Bitrates are lower than those for phone-based models in Figure 2b, but these are not directly comparable: phones and characters do not have a direct one-to-one correspondence. Nevertheless, we see the same trend as with the phone models, with orthographic words giving significantly lower bitrates than the other units.

Finally, I wanted to gauge how close we are to the true bitrate – the lowest achievable. Given the discussion in Section 3, this clearly isn’t possible since we will never have the true model. Nevertheless, we can consider bitrates when training data and model size are dramatically scaled. For this I use the 1.5B- and 3B-parameter Qwen2.5 models [26]. These models are trained on roughly 18T tokens of text. They use a bytelevel BPE tokeniser with an output vocabulary of about 150k types. Because Qwen’s training domain and punctuation are different from the normalised LibriSpeech text, I condition the

Qwen models on a lowercase unpunctuated primer drawn from train-clean-100, prepended to every evaluation utterance. With this setup, Qwen2.5-1.5B achieves a bitrate of 25.90 bps, while Qwen2.5-3B achieves 25.26 bps. We can compare these to the character-based result in Figure 2c, where the best overall bitrate is 27.48 bps. This shows that substantially larger models can achieve even better bitrates. But the aim of this paper was a systematic comparison of different units under the same modelling budget. In this light, the bitrates of the Qwen models serve as a reasonable sanity check on the bitrates achieved.

## 7. Discussion, limitations and future work

The goal of this paper was to compare different linguistic units to see whether an optimal information rate exists when measured on the same data under the same constrained modelling budgets. Results when using phones as the atomic unit suggest that words (or potentially units just slightly longer than words) give a good trade-off, yielding better bitrates than syllables or very long multi-word combinations. These results are supported by experiments with characters as the atomic unit, where orthographic words (now spelled out as characters rather than phones) also achieve the lowest bitrates when using a 20M-parameter transformer language model.

The above seem to suggest that words are more informationtheoretically efficient than smaller or much larger units. However, the concept of a “word” is controversial and not consistent across languages [33, 34]. This is a limitation of this study, and future work should repeat a similar analysis on other languages. But my intention here is not to claim that “words” are the most efficient units. What I do claim is that the most efficient unit (under a limited budget) is something bigger than phones or syllables and smaller than long multi-word combinations. And I suspect that this will also hold on other languages.

In the introduction I noted that this work could have relevance to human language processing. A problem is that this paper measures the information rate of phonetic sequences rather than that of the message that a speaker intends to convey. Measuring the latter is far less straightforward. One possible route would be to adopt a psycholinguistic perspective, examining how different tokenisations relate to human language processing; [35] is one example in this direction. Ultimately, it remains an interesting open question whether and how an informationtheoretic analysis like the one here can tell us something about the linguistic representations employed by humans.

## Acknowledgements

I would like to thank Nicol Visser, Benjamin van Niekerk, Simon Malan, Camryn Abrahamson, Daanshil Ramgutty, Kyle Janse van Rensburg, and Danel Slabbert for helpful discussions.

## AI disclosure

The code for this project was written using AI (Claude Opus). I conceived of all the experiments and carefully laid out a plan; AI then implemented it. I reviewed the code, found and fixed bugs, edited the code to improve clarity, and wrote additional documentation. I wrote a first full draft of this paper before using any AI. I then used Claude, ChatGPT and Grammarly to find factual mistakes, fix spelling and grammar mistakes, highlight unclear sentences and propose alternatives. I then reviewed the entire paper again.

## 8. References

[1] Claude E. Shannon, “A mathematical theory of communication,” Bell System Technical Journal, 1948.

[2] Thomas M. Cover and Joy A. Thomas, Elements of Information Theory, 2nd ed. Wiley-Interscience, 2006.

[3] Grant Sanderson, “Reinventing entropy: Compression is intelligence part 1,” 2026. [Online]. Available: https://www. 3blue1brown.com/lessons/entropy/

[4] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, “LibriSpeech: An ASR corpus based on public domain audio books,” in ICASSP, 2015.

[5] Cristina Videira Lopes and Pedro M. Q. Aguiar, “Alternatives to speech in low bit rate communication systems,” arXiv preprint arXiv:1010.3951, 2010.

[6] James L. Flanagan, Jont B. Allen, and Mark A. Hasegawa-Johnson, Speech Analysis, Synthesis and Perception, 3rd ed., 2014.

[7] Christophe Coupe, Yoon Mi Oh, Dan Dediu, and Fran´ c¸ois Pellegrino, “Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche,” Science Advances, 2019.

[8] Edward Gibson, Richard Futrell, Steven P. Piantadosi, Isabelle Dautriche, Kyle Mahowald, Leon Bergen, and Roger Levy, “How efficiency shapes human language,” Trends in Cognitive Sciences, 2019.

[9] Sami Virpioja, Minna Lehtonen, Annika Hulten, Henna Kivikari,´ Riitta Salmelin, and Krista Lagus, “Using statistical models of morphology in the search for optimal units of representation in the human mental lexicon,” Cognitive Science, 2018.

[10] Ewan Dunbar, Mathieu Bernard, Nicolas Hamilakis, Tu Anh Nguyen, Maureen de Seyssel, Patricia Roze, Morgane Rivi´ ere,\` Eugene Kharitonov, and Emmanuel Dupoux, “The Zero Resource Speech Challenge 2021: Spoken language modelling,” in Interspeech, 2021.

[11] Chia-ying Lee, Timothy J. O’Donnell, and James R. Glass, “Unsupervised lexicon discovery from acoustic input,” Trans. ACL, vol. 3, pp. 389–403, 2015.

[12] Santiago Cuervo, Maciej Grabias, Jan Chorowski, Grzegorz Ciesielski, Adrian Łancucki, Paweł Rychlikowski, and Ricard´ Marxer, “Contrastive prediction strategies for unsupervised segmentation and categorization of phonemes and words,” in ICASSP, 2022.

[13] Okko Ras¨ anen, Gabriel Doyle, and Michael C. Frank, “Pre-¨ linguistic segmentation of speech into syllable-like units,” Cognition, 2018.

[14] Cheol Jun Cho, Nicholas Lee, Akshat Gupta, Dhruv Agarwal, Ethan Chen, Alan W. Black, and Gopala K. Anumanchipalli, “Sylber: Syllabic embedding representation of speech from raw audio,” in ICLR, 2025.

[15] Alan Baade, Puyuan Peng, and David Harwath, “SyllableLM: Learning coarse semantic units for speech language models,” in ICLR, 2025.

[16] Alex S. Park and James R. Glass, “Unsupervised pattern discovery in speech,” IEEE Transactions on Audio, Speech, and Language Processing, 2008.

[17] Simon Malan, Benjamin van Niekerk, and Herman Kamper, “Should top-down clustering affect boundaries in unsupervised word discovery?” IEEE Transactions on Audio, Speech and Language Processing, 2026.

[18] Benjamin van Niekerk, Julian Za¨ıdi, Marc-Andre Carbonneau, and´ Herman Kamper, “Spoken-term discovery using discrete speech units,” in Interspeech 2024, 2024.

[19] Stanley F. Chen and Joshua Goodman, “An empirical study of smoothing techniques for language modeling,” Computer Speech and Language, 1999.

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” NeurIPS, 2017.

[21] Philip Gage, “A new algorithm for data compression,” C Users Journal, 1994.

[22] Ewan Dunbar, Robin Algayres, Julien Karadayi, Mathieu Bernard, Juan Benjumea, Xuan-Nga Cao, Lucie Miskic, Charlotte Dugrain, Lucas Ondel, Alan W. Black, Laurent Besacier, Sakriani Sakti, and Emmanuel Dupoux, “The Zero Resource Speech Challenge 2019: TTS without T,” in Interspeech, 2019.

[23] Daniel Jurafsky and James H. Martin, Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models, 3rd ed., 2026. [Online]. Available: https: //web.stanford.edu/∼jurafsky/slp3/

[24] Hermann Ney, Ute Essen, and Reinhard Kneser, “On structuring probabilistic dependences in stochastic language modelling,” Computer Speech and Language, 1994.

[25] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer, “OPT: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.

[26] Qwen Team, “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[27] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.

[28] Nicol Visser, Simon Malan, Danel Slabbert, and Herman Kamper, “ZeroSyl: Simple zero-resource syllable tokenization for spoken language modeling,” in Interspeech, 2026.

[29] Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger, “Montreal Forced Aligner: Trainable text-speech alignment using Kaldi,” in Interspeech, 2017.

[30] Kyle Gorman, “Python module for syllabifying English ARPABET transcriptions,” GitHub repository, 2013. [Online]. Available: https://github.com/kylebgorman/syllabify/

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch, “Neural machine translation of rare words with subword units,” in ACL, 2016.

[32] Maximilian Bisani and Hermann Ney, “Bootstrap estimates for confidence intervals in ASR performance evaluation,” in ICASSP, 2004.

[33] Martin Haspelmath, “The indeterminacy of word segmentation and the nature of morphology and syntax,” Folia Linguistica, 2011.

[34] ——, “Defining the word,” WORD, 2023.

[35] Byung-Doh Oh and William Schuler, “The impact of token granularity on the predictive power of language model surprisal,” in ACL, 2025.