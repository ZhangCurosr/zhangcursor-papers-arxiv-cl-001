# Letters hide the truth from our eyes: English homophones have meaningfully diferent phonetic realizations

Yu-Hsiang Tseng<sup>1</sup>, Mirjam T. C. Ernestus<sup>2</sup>, Louis F. M. ten Bosch<sup>2</sup>, R. Harald Baayen<sup>1</sup>

<sup>1</sup>University of Tübingen, Germany <sup>2</sup>Radboud University, the Netherlands

## Abstract

The distribution of spoken word duration of English homophones is known to co-vary with frequency of use. This study investigates whether other aspects of the phonetic realization of homophones also difer. A series of quantitative investigations of 14,000 homophone tokens in American television news broadcasts revealed that the tokens of homophone pairs such as weight and wait have diferent phonetic realizations, and that these can be predicted from their meanings in utterance context. These systematic diferences remain even when taking duration-related variation into account. Timenormalized spectrograms emerged as an excellent tool for probing the fine details of phonetic realization, and obviate the need for phonetic transcriptions, which inevitably hide the phonetic truth from our eyes.

## 1 Introduction

Do homophones sound the same? Gahl (2008) and Lohmann (2018) provided evidence that this is not the case. A survey of the spoken word durations of English homophones in the Switchboard corpus (Godfrey et al., 1992) revealed that higher-frequency homophones (e.g. time) tend to be shorter than the corresponding lower-frequency homophones (e.g., thyme). One possible reason that frequency-related durational diferences arise is that their entries in the mental lexicon have their own frequency-based resting activation level. Other possible reasons are that higher-frequency words are more predictable in context and more likely to benefit from articulatory routinization.

A diferent explanation was ofered by Gahl & Baayen (2024). This study showed that the spoken word duration of homophones could be predicted to a considerable extent from the meanings of these homophones. They showed that the greater the support a word form receives from its meaning, the longer the spoken word duration.

This finding indicates that meaning and form are tightly linked, challenging formal phonology (following in the footsteps of Port & Leary, 2005), as well as derivative models in psychology, according to which abstract phonological units shield articulation from semantics. This tight coupling between meaning and form also aligns with the extensive literature on phonesthemes, recurring sound–meaning pairings that are not readily analyzable as morphemes (Bergen, 2004). For example, English words beginning with “gl-” (e.g., glimmer, glisten, glitter) are often associated with light or vision. Such non-arbitrary pairings have been consistently identified in quantitative studies across languages (e.g. Monaghan et al., 2014; Blasi et al., 2016; Pimentel et al., 2019), and similar efects have also been observed in iconic correspondences between English spoken word forms and the sounds of their referents, such as the word frog and frog croaks (de Varda & Marelli, 2025). These systematicities suggest that members of a homophone pair, carrying diferent meanings, will not sound identical.

The present study asks whether, within each English heterographic homophone pair (e.g. wait and weight), the two homophone members difer not only with respect to the distributions of their spoken word durations, but also with respect to their phonetic realization. A further question considered in the present study is whether diferences in phonetic realization can be traced back to diferences in meaning. In what follows, we show that the answer to both questions is yes.

In section 2, we introduce key findings in the research in psychology on English homophones, and section 3 zooms in on the literature on phonetic realization diferences between segmentally similar or identical words. The theoretical framework adopted in the present study, the Discriminative Lexicon Model (Heitmeier et al., 2026), is introduced in section 4. Section 5 presents a series of quantitative investigations of the spectrograms of 14,000 tokens of 35 homophone pairs extracted from the Redhen resource (Joo et al., 2017). These investigations provide strong support for the possibility that how a homophone token is articulated depends on its meaning in context. Section 6 shows that the same patterns are observed after controlling for word duration. The implications of these findings for formal phonology and cognitive models of the mental lexicon are presented in the general discussion (Section 7).

## 2 Homophones

Homophones are words that are said to sound the same but that have diferent meanings. Examples from English are heterographic homophones such as wait and weight, and homographic homophones such as financial bank and river bank.

Homophones have been studied extensively as they ofer the analyst the possibility to study meaning while controlling for form. In psychology, there are informed theories of the organization of the mental lexicon. Classical models of the mental lexicon typically posit multiple layers of representations, distinguishing between layers with syntactic, morphological, and phonological units. Nodes in these layers pass activation to one another, either in a cascading manner or through interactive feedback (e.g., Dell, 1986; Roelofs, 1997; Levelt et al., 1999). As homophones are words that have diferent meanings but the same pronunciation, they are assumed to share the same phonological representations. For example, Levelt (2001) distinguished three strata: a conceptual stratum that accounts for conceptual semantics, a lemma stratum that accounts for words’ morpho-syntactic properties, and a form stratum that contains units for stems and exponents. According to this model, a speaker first starts with a concept (e.g., weight or wait), then selects the corresponding lemma (which provides pointers to information about word category, N for weight and V for wait). The lemma in turn passes activation on to the shared phonological form /we<sup>i</sup>t/. The phonological form activates the corresponding phones, which are then bundled into syllables, which are passed on to the articulators.

According to this model by Levelt (2001), the word frequency efect is located at the level of lexemes.<sup>1</sup> If homophones share the same lexeme, then the frequency efect for high- and low-frequency homophonic words should be the same. Jescheniak & Levelt (1994) tested this prediction using experiments with Dutch-English bilinguals. They compared participants’ reaction times in a translation task with those in a semantic judgment task, and found that low-frequency homophones with high-frequency counterparts behaved like non-homophonic high-frequency words. They argued that a less frequent word (e.g., nun) “inherits” the frequency of its more frequent homophone twin (e.g., none) through their shared lexeme representation.

Dell (1990) observed that not only do high-frequency words elicit fewer speech errors, but low-frequency words with high-frequency homophone twins are less prone to speech errors as well. This study explained this observation by again assuming that homophone twins have distinct lemma nodes encoding word identity, but share the same lexeme node encoding phonological form. In Dell’s model, interactive activation flowing between lemma and lexeme nodes leads to greater lexeme activation for low-frequency homophones than would be expected on the basis of their frequency alone.

However, not all empirical findings align with the hypothesis that homophones share the same form representations. Caramazza et al. (2001), using a picture naming task, compared three types of words: English low-frequency words (e.g. nun) with high-frequency homophones (e.g. none), controls matched for word-specific frequency (e.g., owl), and controls matched with cumulative-homophone frequency (e.g., tooth, matching the combined frequency of none and nun). After further controlling for potential confounds between visual and articulatory complexity, they observed reaction times for homophones to be more similar to the reaction times to word-specific matched controls. They did not find any evidence for frequency inheritance. Further studies investigating diferent languages, diferent kinds of homophones, and using diferent tasks also failed to find consistent evidence for shared form representations for homophones (Miozzo & Caramazza, 2005, 2003; Jescheniak et al., 2003).

A further problem for the hypothesis of homophones having a shared form representation was reported by Ferreira & Grifin (2003). They suggested that lemma selection may not be entirely encapsulated from phonological processes. In their experiments, participants were asked to name pictures (e.g., of a priest). Before the picture was presented, participants first read a cloze sentence that was completed either with a semantic competitor of the target object name (e.g., nun), a homophone of the competitor (e.g., none), or an unrelated control (e.g., match). Interestingly, picture naming showed interference not only from semantically related words, but also from semantically unrelated homophones. This interference is unexpected in models with a strict top-down architecture, but may be compatible with interactive activation architectures. Ferreira & Grifin (2003) and Miozzo & Caramazza (2005) argue for partially independent form representations and feedback between semantics and phonological information, suggesting a closer interaction between form and meaning in homophones.

## 3 The phonetic realization of homophones

If homophones share the same form representation driving articulation, then the prediction is that there are no systematic diferences in the phonetic realization of homophones (Kiparsky, 1982; Dell, 1986; Levelt et al., 1999).

However, Gahl (2008) observed for heterographic homophones in English that more frequent homophones (e.g., time) in a pair tend to have shorter spoken word duration in spontaneous speech compared to their lower-frequency homophone twins (e.g., thyme), after controlling for variables such as speech rate, predictability, syntactic category and orthographic regularity. Lohmann (2018) reported that duration also varies as a function of syntactic category: words such as cut have diferent spoken durations when used as nouns as compared to verbs.

Another example of words that were originally supposed to be homophones but, upon closer inspection, turned out to difer in their phonetic realization are words with syllablefinal voiced and unvoiced obstruents in German and Dutch. Syllable-final obstruents have been described as being realized as voiceless (Kenstowicz, 1994), resulting in homophone pairs such as German Rad “wheel” and Rat “council”, or Dutch ik verwijd “I widen” and ik verwijt “I reproach” (Port & O’Dell, 1985; Warner et al., 2004). Although these pairs should be phonologically identical under the rule of final devoicing, a range of studies nevertheless found that the voice neutralization in these word pairs is incomplete. The word-final devoiced obstruents are realized diferently from their unvoiced counterparts with respect to vowel duration, burst duration, and the number of glottal pulses. While the efect of the incomplete neutralization is small and may involve knowledge of the written word forms (Fourakis & Iverson, 1984; Warner et al., 2006), empirical findings show that not only do speakers produce the obstruent diferently, but that listeners can also perceive the subtly diferent duration cues (Port & Crawford, 1989; Ernestus & Baayen, 2006; Roettger et al., 2014).

Afixes have also been reported to be realized with diferent pronunciation durations, depending on their morphological status. For English, Plag et al. (2017) observed that the duration of word-final $\mathrm { ^ 6 s } ^ { \prime }$ is longer when it is non-morphemic as compared to when it is an inflectional exponent or a clitic. Furthermore, the $/ \mathrm { s } /$ as plural exponent tends to be longer than the clitic $/ \mathrm { s } /$ . These findings dovetail well with the study of Drager (2011), which reported for conversational English that the grammatical and discourse functions of like co-determine the realization of pitch and the degree of diphthongization.

Finally, the investigation of Seyfarth et al. (2017) indicated that morphologically distinct homophones difer phonetically. Because they have diferent morphological relatives, fricative-final inflected words (e.g. laps) are argued to have longer stem and sufix duration than their morphologically simple counterparts (e.g., lapse), even though both share the same phones. These results suggest that the phonetic implementation of segmentally identical morphologically complex words is shaped by their morphological structure (Song et al., 2013; Plag et al., 2017; Celata et al., 2022; Schmitz & Baer-Henney, 2024), but see Saito et al. (2024) for an alternative explanation. A discussion of the role of re-syllabification and concomitant prosodic restructuring of otherwise segmentally identical stems in Dutch and English is provided by Kemps et al. (2005a,b).

In summary, there is more systematic variation in the phonetic realization of segmentally similar or identical words. This systematicity has been argued to be due to diferences in frequency of use, diferences in word category, diferences in morphological structure, and semantic transparency. The abovementioned study by Drager (2011) on English “like” adds word meaning to the set of explanatory factors.

Using methods from distributional semantics and computational modeling (which we discuss in more detail below), Gahl & Baayen (2024) showed for English homophones that word meaning is a strong predictor of spoken word duration. For Mandarin single-syllable homophones, Jin et al. (2026) observed diferences in the details of the phonetic realization of their tones, indicating that pitch is a further dimension in which diferences in meaning between words sharing the same segments can be expressed. Similar form-meaning alignment has also been reported for Mandarin tone at the word token level (see, e.g., Chuang et al., 2026; Lu et al., 2026) and also for the realization of pitch in English left-stressed two-syllable words (Chuang et al., 2023).

The goal of the present study is to clarify whether fine-grained high-dimensional contextspecific representations of the meanings of word tokens of homophones are not only aligned with spoken word duration, but also with other aspects of the fine details of phonetic realization of homophones. Building on the observation of Drager (2011) that how segments are realized phonetically depends on words’ meanings, in what follows we investigate whether consistent diferences in the phonetic realization of homophones exist, and whether these diferences can be predicted, at least in part, from their meanings.

## 4 Theoretical and methodological framework

The present research is theoretically anchored in the Discriminative Lexicon Model (DLM; Baayen et al., 2019; Heitmeier et al., 2026), a computational model of the mental lexicon that implements mappings between forms and meanings to model language comprehension and production. These mappings predict that the details of words’ meanings unavoidably give rise to diferences in phonetic realization, not only for spoken word duration and the realization of F0 contours, but also for how words are articulated (for an articulography study, see Saito et al., 2024). The mappings are typically implemented with networks that constitute both the model’s memory and its processing algorithm. The model does not work with underlying forms, and it does not have entries for words’ forms or meanings.

In the DLM, a form-and-meaning mapping is learned by continually adjusting the association weights between inputs and outputs. For discrete inputs (for comprehension, following Danks (2003), referred to as cues) and for discrete outputs (referred to as outcomes), the mechanism by which the associations are updated can be elegantly described by a simple learning rule, in which the weights are strengthened when a cue and an outcome co-occur, and weakened when a cue is present but an outcome is absent (Rescorla & Wagner, 1972). For example, seeing a large flat surface (cue) of a grand piano (outcome) will strengthen their association, but weaken the association between a large surface and a table. Experimental evidence shows that seeing a table followed by a grand piano in a naming experiment slows naming responses (Marsolek, 2008). This suggests that the association weights of discriminative cues are continually recalibrated by real-life exposure to co-occurring events in the environment. For continuous inputs and outputs, the learning rule of Widrow & Hof (1960) follows the same underlying learning mechanism, with an update rule that iteratively reduces prediction error.

There are no static representations of forms and meanings in DLM; only the association weights are stored for comprehension and production, and these are continuously updated and recalibrated with experience. For comprehension, a word’s form is an ephemeral input to the network. The network subsequently calculates a corresponding meaning, which is also not stored. After a mapping is completed, the network’s weights are updated. For production, the process of conceptualization is assumed to result in an ephemeral semantic representation that is input to the network mapping meaning to form. Once the form has been estimated, the weights of the production network are updated.

These form-meaning mappings of the DLM are mostly implemented as simple linear transformations, not unlike multivariate linear regression. If so required, non-linearity can be added to increase prediction accuracy (Chuang et al., 2026; Heitmeier et al., 2026). Using simple linear mappings has the advantage that no intermediate layers of hidden nodes are involved that filter the information flowing between form and meaning, and that may not be well interpretable linguistically. Moreover, by using linear mappings, it becomes possible to investigate to what extent semantic detail and phonetic realization co-vary systematically.

The form and meaning representations that the DLM works with consist of lists of numbers. In the discrete binary case, each number indicates whether a cue or outcome is present; in the continuous real-valued case, it indicates how strongly a cue or outcome is activated. When many cues and outcomes are involved, such lists can be treated as high-dimensional vectors. A vector is commonly described as an object with a direction and magnitude, and most intuitively visualized as an oriented arrow in a 2-D or 3-D space. The high dimensionality of embeddings makes it possible to capture wide arrays of highly nuanced similarities and diferences. In what follows, we introduce the form and meaning vectors that are central to the present study.

## 4.1 Representing Meanings

Vectors in the meaning space of the DLM are referred to as semantic vectors. One of the widely adopted semantic vectors in computational linguistics is the so-called word vectors, also known as word embeddings, which have their roots in distributional semantics (Harris, 1954; Firth, 1957). Although word embeddings are often associated with natural language processing (Mikolov et al., 2013; Pennington et al., 2014; Bojanowski et al., 2017), the idea of representing meanings as vectors has a long tradition also in the psycholinguistic literature (Lund & Burgess, 1996; Landauer & Dumais, 1997; Shaoul & Westbury, 2010; Grifiths et al., 2007). The embeddings referenced thus far are all type-based, that is, one word — in the sense of a letter string — is represented by one vector, regardless of the context the word occurs in. For example, the orthographic form bank in “financial bank” and “river bank” is associated with the very same embedding.

To avoid the limitations of type-specific embeddings, in the present study, we make use of contextualized embeddings (CEs). These token-based embeddings represent the meaning of a word diferently depending on the context in which it occurs. CEs usually are vectors with greater dimensionality as they require more capacity to diferentially represent lexical meanings across contexts. Early studies on CEs trained a simple recurrent network and showed that the same word types have diferent CEs if they are embedded in diferent sentence structures (Elman, 1991, 2004). Later studies and models have pursued many diferent architectures (e.g., Bengio et al., 2003; Jones & Mewhort, 2007; Sutskever et al., 2011; Radford et al., 2018; Peters et al., 2018; Peng et al., 2023; Dao & Gu, 2024).

For the present study, we extracted contextualized embeddings at the word token level from a large language model (LLM; here, GPT-2). This model has 137M parameters and a tokenizer with 50,257 unique string and substring tokens. Both model and tokenizer are available at https://huggingface.co/openai-community/gpt2.<sup>2</sup>

The homophones used in the present study are all tokenized into one ‘model token’, capitol being the only exception. For this word, we averaged the embeddings of both ‘model tokens’ to obtain the CE. For each model token, the input text consisted of a five-word window immediately preceding (and including) the target word. The context-specific meaning of a word token is represented as a 768-dimensional vector. For studies assessing the quality and potential of CEs, see Manning et al. (2020); Linzen & Baroni (2021); Tenney et al. (2019); Pavlick (2022); Schrimpf et al. (2021); Caucheteux et al. (2022); Goldstein et al. (2024); Chuang et al. (2026); Lu et al. (2026).

## 4.2 Representing Forms

Previous research investigating the phonetic properties of spoken words has studied, for example, spoken word duration (Hay, 2004; Gahl, 2008; Seyfarth et al., 2017), spectral-temporal measurements (Port & Crawford, 1989), and time-series of f0 measurements (Chuang et al., 2026; Lu et al., 2026). In the present study, instead of extracting partial information from the speech signal, we focus on the spectrogram of that speech signal.

Our point of departure is a word token’s Mel-scale spectrogram, which is an eficient compressed representation that is informed by the human cochlear frequency-place map (Stevens et al., 1937; Greenwood, 1990). Current deep learning-based speech models extract feature representations from MEL spectrograms (e.g., Baevski et al., 2020; Radford et al., 2023). These feature representations have been successfully applied in linguistic studies modeling speech and accent variation across dialects (e.g., Bartelds et al., 2022; Yang et al., 2023). However, the caveats of these deep learning-based features are that they integrate information across time spans and thus mix information from linguistic units of very diferent grain sizes. As a consequence, the feature vector for a single spoken word token also typically encodes information about surrounding words or phrases. Such a mix of information can be advantageous for optimizing task performance (e.g., distinguishing accents), but makes interpretation harder when studying form–meaning relations at the word token level.

We therefore opted for constructing speech-derived form vectors directly using MEL spectrograms. At this point, the problem arises that speech tokens in spontaneous speech vary in duration. Creating spectrograms from words with diferent durations will lead to spectrograms with very diferent sizes, and will mask similarities in the shapes of spectrograms.

One possible solution is to derive (continuous) frequency band summaries and zero-pad them into fixed-length vectors (Arnold et al., 2017; Shafaei-Bajestan et al., 2021). However, instead of zero-padding, when constructing a MEL spectrogram, we dynamically adjust the step size, a parameter determining how far the analysis window advances before the spectral values of the next frame are calculated. Across word tokens of diferent lengths, we adjust the step size such that spectrograms with a fixed number of timesteps are obtained. Similarities in the shape of spectrograms emerge straightforwardly in such time-normalized spectrograms. These time-normalized spectrograms incorporate information about word duration. In section 6, we show that when duration is taken out of the time-normalized spectrograms using Partial Least Squares, results remain basically the same.

## 4.2.1 Constructing time-normalized spectrograms

Time-normalized spectrograms are constructed as follows. For a given audio token, we (1) determine an optimal step size producing 50 time steps, (2) transform the speech signal into a spectrogram with the found step size, (3) convert the spectral information of each time step into 21 mel-frequency banks, and (4) vectorize the time-normalized spectrogram into a 1,050-dimensional speech vector.

Processing begins with the speech signals, which are time-domain digital waveforms representing physically continuous audio streams. From each speech signal, we derive a standard spectrogram (see the upper panel of Figure 1 for an example), a matrix that represents both temporal information by column (along the x-axis when visualized as a rectangular bitmap) and spectral information by row (along the y-axis). Each column represents the spectrum computed on a stretch (window) of the speech signal. The ofset between consecutive windows, the step size, determines the number of columns in a spectrogram. If the step size is small, the window moves slowly along the speech signal and produces many columns; by contrast, if the step size is large, the window strides quickly and the spectrogram has fewer columns. We control the number of columns in a time-normalized spectrogram by determining an optimal step size, using the following equation:

$$
{ \mathrm { s t e p ~ s i z e } } = \left\lfloor { \frac { N - w } { T - 1 } } \right\rfloor
$$

The operator ⌊x⌋ indicates the floor function, finding the largest integer smaller than x. N is the number of samples in the speech signal, T refers to the target number of frames, which is set to 50, and w is the window size set to 10ms (i.e., 160 samples under a sampling rate of 16,000 Hz). Concretely, for a token with 7,520 samples (470ms), the equation gives us a step size of $( 7 5 2 0 - 1 6 0 ) / ( 5 0 - 1 ) \approx 1 5 0$ samples (9.4ms). Using the found step size, we convert the speech signal into a spectrogram of 50 columns (the temporal axis). On the spectral axis side, the number of rows is determined by the window size, which is set to 10 ms and yields 80 rows at a sampling rate of 16,000 Hz.<sup>3</sup> However, the frequency sensitivity of the human cochlea varies across the frequency range. By compressing 80 rows into 21 rows using MEL-frequency filters, we derive a more compact spectrogram that better reflects the frequency sensitivity of the human cochlea (Greenwood, 1990). After these steps, the speech signal is now transformed into a fixed-size MEL-spectrogram: 50 columns in the temporal axis, and 21 rows in the spectral axis (see the bottom panel of Figure 1). Finally, we flatten the spectrogram by reshaping it into a 1,050-dimensional speech vector through row-wise concatenation. <sup>4</sup>

Spectrogram  
![](images/837cdc4327d2462eb57cb860412299bd87dbe463a6c09b98716d58c4799769bf.jpg)

Mel-spectrogram (ada-hop)  
![](images/10c3124251ef3a9d7d2fce9b550250214c4092306e330517c3dc23dc1795ee94.jpg)  
Figure 1: Spectrogram of a 470ms “wait” token (upper panel) and the corresponding fixed-size spectrogram with adaptive hop length (lower panel). The horizontal dashed lines indicate the frequency steps for the 21 Mel-frequency banks. The vertical dashed lines represent the 50 timesteps that are obtained with a step size of 9.4ms.

## 4.3 Linear mappings between form and meaning

The alignment between the speech vectors, in form space, and CEs, in the meaning space, is investigated in the present study using the linear mappings of the DLM (Baayen et al., 2019; Heitmeier et al., 2026). Let n denote the number of word tokens under investigation. For comprehension, a linear mapping $\pmb { F }$ transforms the $n \times 1 0 5 0$ form matrix C into a corresponding $n \times 7 6 8$ semantic matrix S. Conversely, a production mapping $G$ starts with the semantic matrix and transforms it into the form matrix. As the mappings are matrices and no intermediate representation or non-linear functions are involved, they can be readily estimated using linear algebra <sup>5</sup>:

$$
\begin{array} { r } { \pmb { F } = ( \pmb { C } ^ { \top } \pmb { C } ) ^ { - 1 } \pmb { C } ^ { \top } \pmb { S } } \\ { \pmb { G } = ( \pmb { S } ^ { \top } \pmb { S } ) ^ { - 1 } \pmb { S } ^ { \top } \pmb { C } . } \end{array}
$$

where $C ^ { \top }$ indicates the transpose of C and $C ^ { - 1 }$ is the inverse of C. The learned F and G matrices are used to compute the predicted semantic matrix, ${ \hat { S } } ,$ and the predicted form matrix, $\hat { C } .$ , respectively:

$$
\begin{array} { l } { { C \cdot F = \hat { S } } } \\ { { S \cdot G = \hat { C } . } } \end{array}
$$

The quality of these Linear Discriminative Learning (LDL) mappings can be evaluated by measuring how close the $\hat { S }$ or $\hat { C }$ matrices are to the actual S or C matrices. One such measure is by-word nearest neighbor accuracy (Heitmeier et al., 2026). For instance, when the comprehension mapping places a token’s speech vector in the semantic space as a predicted or estimated CE, then, if the nearest neighbor in the semantic space belongs to the same word type, we count it as a correct prediction. Similarly, for production, spoken word tokens are represented as speech vectors derived from time-normalized spectrograms in form space, and a prediction is counted as correct if the nearest neighbor of a predicted speech vector belongs to the same word type. If the proportion of such correct predictions is higher than chance in the held-out data, this indicates the linear mapping has succeeded in capturing some of the similarity between the form and meaning spaces.

## 5 Modeling the phonetic realization of homophones

In this section, we carry out a series of experiments investigating whether homophone members can be distinguished by their phonetic realization, represented as speech vectors, and how speech vectors and CEs are aligned in form and meaning space. Section 5.1 introduces the dataset that we investigated. Section 5.2 investigates whether the identity of a homophone pair can be predicted from the time-normalized spectrograms of its tokens, while Section 5.3 asks whether the individual members within each pair can also be predicted. Section 5.4 makes use of the generalized additive model to clarify how the time-normalized spectrograms of homophones difer. Next, in Section 5.5, we implemented the LDL mappings between the speech vectors (flattened time-normalized spectrograms) and contextualized embeddings (CEs) to investigate the potential role of meanings in these word-specific realizations. We compared the alignments with permutation baselines (Section 5.5.1 and 5.5.2), and show that the prototypical realization of a homophonic word can be constructed from the prototypical meanings of that word.

## 5.1 Data

The spoken word tokens are selected from the Redhen 2016 Dataset(Joo et al., 2017), which includes videos of United States television news broadcasts, corresponding subtitles, and the word-level timestamps obtained from a forced aligner (Ochshorn & Hawkins, 2017). Next, we identified the heterographic homophone pairs where the two word types met the following criteria: (1) each word type occurs at least 200 times; word types with more than 200 tokens were downsampled to 200 tokens, and (2) each token appears in a unique context, defined by up to 10 preceding words. Requiring each token to appear in a unique context removes artifacts caused by repeated advertisement phrases that occur across programs. These selection criteria led to a dataset with 35 homophone pairs, with 70 orthographic word types and 14,000 tokens.

The 14,000-word tokens originated from 215 unique news programs. On average, a word type occurs in 63.2 programs. The mean duration of these tokens is 330ms $( S D = 1 1 7 \mathrm { m s } )$ The distribution of spoken word durations has a slight positive skew: the first quartile (Q1) is 250ms, the median is 320ms, and the third quartile (Q3) is 400ms. The step size selected for time normalization of the tokens has a mean of 6.5ms $( S D = 2 . 4 \mathrm { m s } )$ , with $Q 1 = 4 . 9 \mathrm { m s }$ median= 6.3ms, and $Q 3 = 7 . 9 \mathrm { m s }$

A minority of homophone pairs in this data set (8) are words with diferent morphological structure, such as knows and nose, band and banned, or passed and past. Three comprise irregular verb forms (seen, knew, blew). However, most of the pairs of homophones (24) consist of monomorphemic words, such as wait and weight, mail and male. Some diferences in spoken word duration may be expected due to diferences in morphological structure and word category, as suggested by studies such as Seyfarth et al. (2017) and Lohmann (2018). Spoken word duration, however, is also co-determined by word meaning, as Gahl & Baayen (2024) showed using context-independent embeddings. In what follows, we first move beyond word duration to investigate systematic diferences in how homophones are articulated, and how these diferences covary with context-specific word meanings. The role of word duration will be revisited later, when we examine whether the observed diferences remain after accounting for durations.

## 5.2 Experiment 1: Predicting homophone pairs from speech vectors

Experiment 1 addresses the question of whether homophone pairs can be properly distinguished on the basis of their speech vectors. This is a minimal requirement for the speech vectors that we constructed: if a homophone pair such as wait and weight cannot be distinguished from a homophone pair such as sell and cell, then the speech vectors that we constructed cannot be expected to be revealing about potential diferences in the phonetic detail between the members within a homophone pair.

We used Linear Discriminant Analysis (LDA, Fisher, 1936; Rao, 1948) with homophone pairs as a 35-level response factor, and the speech vectors as predictors. Given the correlational structure in spectrograms, we used Principal Components Analysis (PCA) to reduce the dimensionality of the speech vectors to 50, as the first 50 principal components account for 90% of the variance in the original 1,050-dimensional speech vectors. These more compact speech vectors alleviate the computational burden of handling high-dimensional vectors, while at the same time suppressing potential noise in the data (Jollife, 2010).

We performed 10-fold cross-validation to assess how well homophone pairs can be predicted from the 50-dimensional speech vectors. Mean accuracy was 79.30% $( S E = 0 . 4 2 \% )$ A permutation baseline, where we perform 10-fold cross-validation on a randomly permuted dataset, is 2.82% $( S E = 0 . 1 0 \% )$ . Most misclassifications involve pairs with highly similar pronunciations, such as seen and sea, wait and way, plane and planes. These results clearly show that the speech vectors that we constructed, perhaps unsurprisingly, contain rich information for discriminating between pronunciations.

## 5.3 Experiment 2: Predicting homophones from speech vectors

Using the same procedure and method as for Experiment 1, we next ask whether a speech vector can distinguish homophone members across all pairs, using an LDA classifier. Each speech vector is classified into one of the 70 possible orthographic words. This is a much harder task, not only because the number of categories to be distinguished doubles, but also because if the members of a homophone pair are truly homophonous, the classifier should not be able to distinguish them (e.g., between wait and weight). Mean accuracy for 10-fold crossvalidation was 51.13% $( S E = 0 . 4 6 \% )$ . The corresponding permutation baseline was 1.43% $( S E = 0 . 1 1 \% )$ . To rule out the possibility of random guessing within homophone pairs, we conducted additional LDA analyses, one for each homophone pair. Classification results are shown in Figure 2. All homophone pairs are at least two standard errors above the 50%- chance level, indicating spoken word tokens, despite being homophonous, can be classified into their word types significantly better than chance. These findings are consistent with previous studies documenting that heterographic homophones have, on average, diferent spoken word durations (Gahl & Baayen, 2024; Lohmann, 2018). This experiment clarifies that the speech signals of members within homophone pairs difer not only in duration but also in their spectrograms: even a straightforward linear classifier can tell wait and weight apart with far above-chance accuracy. Given the immense variability in speech, this is a surprising result.

This result cannot be simply explained by a potential diference between the members of the pairs in their utterance position. For each homophone pair, we computed proportion test statistics for tokens preceded by and followed by a pause, and correlated these statistics with LDA classification accuracies. Neither correlation was significant: $r = . 2 6 , p = . 1 4$ for preceding pauses, and $r = . 1 0 , p = . 5 8$ for following pauses. These results suggest that the classification accuracy cannot be simply attributed to diferences in utterance positions.

While the LDA analyses show that diferences in the pronunciation of homophones exist, they are uninformative about how members of a given homophone pair difer from each other. To address the question of how the spectrograms of homophones difer, we made use of the Generalized Additive Model (GAM, Wood, 2017).

![](images/01a8aa508b5eff29db79dcc630d241ea22b44315477df7ea9cc6cc61c4753884.jpg)  
Figure 2: LDA classification accuracies for the speech tokens of the members of homophones within homophone pairs. The black vertical dashed line is the 50% baseline, the red vertical dashed line represents the overall mean accuracy with a 95% confidence interval in red, and the horizontal blue lines indicate 2 standard errors calculated with 10 cross-validation runs. Mean accuracies are represented by blue dots.

## 5.4 Experiment 3: Diference spectrograms

In order to clarify in what way the spectrograms of the members of the homophone pairs difer, we made use of the Generalized Additive Model (GAM, Wood, 2017). In general, GAMs make it possible to statistically evaluate whether the estimated spectrogram surfaces for the two members in a homophone pair difer. For the present data, for any given homophone pair, the question is whether there is suficient evidence to argue that the spectrogram predicted on the basis of the tokens of the first homophone (e.g., weight) is significantly different from the spectrogram predicted on the basis of the tokens of the second homophone (e.g., wait).

The dataset we constructed for the GAM takes, for each homophone, its 1,050-dimensional speech vector as 1,050 observations, each of which is tied to a timestep (position on the temporal axis) and a frequency band (position on the spectral axis). For each observation, information was added about the duration of the token, the presence or absence of a preceding or following pause, and the speaking rate. For example, the datasets for weight and wait were combined into one dataframe with 420,000 rows and 7 columns <sup>6</sup>.

The analysis that we implemented sets up a smooth regression surface (predicted spectrogram) for one homophone, in combination with a diference surface (the predicted diference spectrogram) for the second homophone. When the diference spectrogram is added to the spectrogram of the first homophone, the predicted spectrogram of the second homophone is obtained. We modeled the response variable, the spectrogram value spec, as a smooth interaction of timestep and filter\_bank, using a tensor-product smooth (te).

$$
\begin{array} { l l l } { { \mathrm { s p e c } } } & { { \sim } } & { { \mathrm { t e ( t i m e s t e p , ~ f l t e r ~ b a n k ) + } \nonumber } } \\ { { } } & { { } } & { { \mathrm { t e ( t i m e s t e p , ~ f l t e r ~ b a n k , ~ b y { = } I ) + } \nonumber } } \\ { { } } & { { } } & { { \mathrm { s ( d u r a t i o n ) + s ( s p e a k i n g ~ r a t e ) + } \nonumber } } \\ { { } } & { { } } & { { \mathrm { p r e c e d i n g ~ p a u s e + s u b s e q u e n t ~ p a u s e } } } \end{array}\tag{1}
$$

As can be seen in this model specification, two tensor product smooths are requested, one for the first homophone, and a second one for the diference surface between the second homophone and the first. The variable I in the second smooth is an indicator variable that is 0 for the first homophone, and 1 for the second homophone. If a diference surface is statistically supported (in the present study, by a low empirical Bayes p-value), then the diference surface can be used to clarify which regions in the spectrogram are significantly diferent between the homophones of a homophone pair. The above model also includes the abovementioned covariates (spoken word duration, speaking rate) and factorial controls for the presence of a preceding or following pause.

Figure 3 illustrates key results for the homophone pair wait and weight. The two panels on the left present the mean spectrograms of the two homophones. The two dark bands in the middle of these average spectrograms reflect the formants of the diphthong eI. The lower band around fbank 3 (centered at 416Hz) presents F1, and the upper band around fbank 12 (centered at 1952Hz) presents F2. The coda t is not clearly observable in the mean spectrogram, suggesting it may not be fully articulated in the spontaneous speech under investigation here.

Although, as expected, the two spectrograms are similar in many ways, some diferences are already accessible to a visual scan. These diferences are brought out in a precise way by the diference spectrogram (i.e., diference surface) estimated by the GAM for wait as

Figure 3: An example of a GAM for a homophone pair: wait vs. weight. (a) The averaged spectrogram of weight; (b) the averaged spectrogram of wait; (c) the phone probabilities of wait/weight in normalized time; (d) the diference spectrogram.  
![](images/cba808fa9bb4975ba587e621deff73215661e43ad4b587fcbbfdd6de99596348.jpg)  
compared to weight. Below, we return to the details of the color coding. For now, the black contour lines and their credible confidence regions (red dots: 1-SE down, green dots: 1-SE up) are of interest. If the 1-SE confidence contour regions of two neighboring contour lines overlap, it indicates that the corresponding contour lines are not reliably distinguishable, and hence the gradient between them is not statistically well supported. Here, however, we first note these confidence contour regions hardly overlap, from which we can conclude there are reliable gradients in the diference surface. Second, we see that during the first 10 timesteps, there is some more energy in the spectrogram of wait, that around timestep 20, there is substantially less energy in the spectrum of wait, and that around timestep 35, there is much more energy in the spectrogram of wait. There is also a diference in timing of the vowel clearly visible in the diference surface: the energy of the nucleus is first reduced (shown in blue) and subsequently increased (shown in red), indicating a delayed onset of the vowel in wait relative to weight.

In order to assess whether the diference surface is statistically well-supported, we compared the GAM specified in equation 1 with a simpler GAM,

$$
\begin{array} { l l l } { { \mathrm { s p e c } } } & { { \sim } } & { { \mathrm { I + t e ( t i m e s t e p , ~ f l t e r ~ b a n k ) + } \nonumber } } \\ { { } } & { { } } & { { \mathrm { s ( d u r a t i o n ) + s ( s p e a k i n g ~ r a t e ) + } \nonumber } } \\ { { } } & { { } } & { { \mathrm { p r e c e d i n g ~ p a u s e + s u b s e q u e n t ~ p a u s e } } } \end{array}\tag{2}
$$

in which both words have the same smooth surface for time by filter bank, but in which the indicator variable (I) allows for a diference in average intensity between the two words. The diference in AIC of the two models was ∆ AIC = 5208.73 in favor of model (1), which implies that the model that includes a diference surface is substantially more likely to have generated the data.

We carried out the same GAM analysis for all 35 pairs of homophones. The full set of 35 figures, showing averaged time-normalized spectrograms and their diference surfaces, is provided in Appendix A. Qualitative diferences between homophone members are visible there; here we focus on quantitative diferences. Across all 35 pairs, substantial reductions in AIC were present for the models that included a diference surface, indicating a strong word efect. The average AIC reduction of the word models across homophone pairs was 3413.92 (SD = 1881.87). The homophone pair with the largest diference is mail/male (∆AIC = 8909.03). This pair also ranks among the highest in the LDA analysis (Section 5.2). The homophone pair with the smallest yet still highly significant AIC diference is sight/site (∆AIC = 638.13). This pair also has low accuracy in the LDA classification.

These findings clearly indicate that these homophones have word-specific phonetic realizations, and LDA and GAM provide converging yet complementary evidence for this conclusion. The LDA demonstrates the contrast under a cross-validation perspective: this classifier generalizes from training to test data, which is possible only if there are systematic diferences in the spectrograms of homophone pairs. The results obtained with LDAs are strengthened by the results obtained with the GAMs, which provide converging evidence from a statistical model-comparison perspective. Homophones must have word-specific realizations, as a model that includes a by-smooth for homophone outperforms a model that does not distinguish between homophones. In addition to identifying the spectral contrast surface, the GAM analysis also controlled for covariates such as duration and speech rate, showing that the word efect persists even when these covariates are included. Most importantly, the tensor-product smooth term provides a clear picture of how these words difer in terms of their mel-spectrograms.

Thus far, we have presented evidence that the spectrograms of homophones have homophone-specific properties. In the next section, we address the question of where these wordspecific realizations originate from.

## 5.5 Experiment 4: alignment between speech vectors and semantic vectors

Are the observed homophone-specific spectrograms aligned with their homophone-specific and utterance-specific meanings? Whereas in the preceding sections we represented word meaning categorically, by means of discrete class labels indicating homophone pairs or homophone members, we now turn to contextualized embeddings, which we expect to provide much more refined approximations of what words mean in their utterance contexts. Following the dimension reduction applied to the speech vectors, we also used PCA to reduce the dimensionality of the CEs to 50, with the first 50 principal components accounting for 96% of the variance in the original 768-dimensional CEs.

Figure 4 visualizes the locations in the semantic space of the CEs of the 14,000 tokens of the 70 homophones in our dataset, using t-SNE (Maaten & Hinton, 2008). The by-word clusters are well-separated, which is consistent with the high word classification accuracy in LDA. In addition, the relative distances between words loosely reflect more general semantic similarities. For example, words in the same inflectional paradigm are close to each other, such as see and seen, knows and knew.<sup>7</sup> Other pairs also show interesting proximities in the t-SNE map, such as counsel and aid, both relating to helping people; and sea and tide, both related to the ocean. Furthermore, there is a clear tendency for nouns to have more negative values on the first t-SNE dimension, and for verbs to have more positive values on this dimension.

Having clarified that the CEs indeed capture important aspects of meaning, we next discuss three computational experiments using these CEs.

## 5.5.1 Experiment 4.1: LDL with full permutation

We first investigated, for each homophone pair separately, the alignment between the speech vectors, in the form space, and the CEs, in the meaning space, using Linear Discriminative Learning (introduced above in section 4.3). We computed and evaluated both comprehension and production mappings for each homophone pair.

Each homophone pair has 400 tokens, comprising 400 rows in the respective form and semantic matrices. Each row in the form and semantic matrices is either a speech vector or a CE, both of which have 50 numbers. Thus, both the C and S matrices are of dimension $4 0 0 \times 5 0$ . We used 10-fold cross-validation, evaluating a mapping F (or G) with one fold of the data and learning from the remaining folds. As a baseline, we randomly permuted the rows of the 400 × 50 semantic matrix S, and followed the same evaluation process. Actual mapping performance can now be compared straightforwardly with a baseline in which the relation between form and meaning is broken. Results are presented in Figure 5. Across comprehension (left panel) and production (right panel), LDL mappings perform significantly better for the actual pairing of form and meaning than the permutation baseline for most of the homophone pairs. Performance above the baseline indicates that the mappings, albeit simple linear ones, learn systematic regularities that are generalizable to unseen homophone tokens. The comprehension and production results both point to the same conclusion: form and meaning do not constitute two independent spaces.

## 5.5.2 Experiment 4.2: LDL with within-word permutation

In the preceding section, we observed that LDL accuracy is far above a full permutation baseline. What we do not know at this point is to what extent the tokens of a homophone are aligned with their own CEs. Imagine two perfectly separable word clouds, each corresponding to one homophone of a pair of homophones (e.g., weight and wait). If a mapping perfectly follows the token-level structure and places all predicted tokens exactly at their counterparts, classification accuracy at the word level will be perfect. However, if the mapping fully respects word structure without capturing any token-level correspondence, then each predicted token will still fall in the correct word cluster, also resulting in perfect accuracy. Yet, in the second case, the mapping reflects only broader word-level structure, whereas the first provides stronger support for alignment between form and meaning at the individual token level.

For assessing the quality of form-meaning mappings between the spectrograms and CEs of homophones’ individual word tokens, we proceeded as follows. First, we carried out withinword permutation, where the 200 semantic vectors of one homophone, e.g., weight were

Figure 4: Contextualized embeddings of homophone tokens. Each point stands for one word token’s CE and is color-coded by word identity. All CEs are first transformed into word space using LDA and reduced into a two-dimensional space using t-SNE for visualization. Words are well-separated in the figure, consistent with the 95% word classification accuracy based on CEs.

Contextualized embeddings visualized with tSNE  
![](images/87d33d6525e9a2ff566c320475f77f0ed08049b965f535c8b52aeef6620b8e66.jpg)

Figure 5: Alignments of predicted meaning vectors (left panel) and form vectors (right panel) compared to a full permutation baseline (in red). The vertical dashed lines represent mean accuracies. Confidence intervals are based on 10-fold cross-validation.  
![](images/9d59c3c3ca41faada6b6ecb0c9f43e7c8240cbaffd8a27e1d1db1e6dc61770cb.jpg)

randomly re-ordered, and those of the other homophone, wait were randomly re-ordered separately. As a consequence, the permuted semantic vectors of the tokens of weight will always correspond to a speech token of weight, albeit to a randomly selected token.

Second, we use a more informative index to evaluate the token-level alignment. The nearest neighbor word accuracy used above, defined as the proportion of predicted vectors whose nearest neighbor belongs to the same word type, is insensitive to the token-level structure, as it can be perfect even when the mapping captures only word-type distinctions. Therefore, we use the ratio $R _ { \mathrm { L 2 } }$ , which assesses how much of the original vectors (speech vectors or CEs) is recovered by the LDL predictions. $R _ { \mathrm { L 2 } }$ is defined as the ratio between the sum of squared L2-norms of the predicted vectors vˆ and those of of the original vectors v across n (here, n = 400) observations: <sup>8</sup>

$$
R _ { \mathrm { L 2 } } = \frac { \sum _ { i = 1 } ^ { n } \lVert \hat { v _ { i } } \rVert ^ { 2 } } { \sum _ { i = 1 } ^ { n } \lVert v _ { i } \rVert ^ { 2 } } = \frac { \sum _ { i = 1 } ^ { n } \hat { { v _ { i } } } \cdot \hat { { v _ { i } } } } { \sum _ { i = 1 } ^ { n } { { v _ { i } } \cdot { v _ { i } } } } .\tag{3}
$$

To assess whether the $R _ { \mathrm { L 2 } }$ scores from actual data are significantly higher than those of within-word permuted data, we repeated the permutation process thirty times and constructed a 95% confidence interval. An $R _ { \mathrm { L 2 } }$ score above the upper bound of this interval provides evidence for the existence of form-meaning alignment at the token level.

![](images/194b4ba3224184a8f2e64b4f69bbaedb0650cb71dffdf9b912f685907f94c929.jpg)  
Figure 6: $R _ { \mathrm { L 2 } }$ scores for comprehension (left panel) and production (right panel). Blue dots represent empirical $R _ { \mathrm { L 2 } }$ scores, red dots and red horizontal lines represent 95% credible intervals and means based on 30 within-word permutation cross-validation runs.

Results are shown in Figure 6 for comprehension (left panel) and production (right panel). The blue dots represent the empirical $R _ { \mathrm { L 2 } }$ scores, the red dots and red horizontal lines represent the 95% confidence intervals and means based on the 30 within-word permutation cross-validation runs. For production, almost all homophone pairs show above-chance formmeaning alignment, with only two exceptions. For comprehension, only 12 homophone pairs have observed values that lie outside the cross-validation interval. Nevertheless, there is only one pair where the empirical score is less than the cross-validation mean (blue, blew).

Comparing the present findings with those reported in the previous section (5.5.1), it is clear that token-level alignment is weaker than the type-level alignment. This is unsurprising as the diferences in the meanings of the homophones within homophone pairs are much larger than the more subtle diferences in the contextualized semantics of individual tokens of these homophones.

The efects in comprehension are less pronounced, possibly because the comprehension models start from speech vectors that reflect individual speakers’ own speech habits and emotions at the time of speaking. By contrast, the production models take CEs derived from a large language model that is trained on a vast amount of data and hence likely provides more stable and less variable representations. As a consequence, the richer speech vectors contain information that is not matched in the semantic vectors, rendering prediction more dificult compared to the production model. With simpler ‘average’ vectors as the starting point for production, the rich person-specific information in the speech vectors simply becomes just measurement noise.

The present results clarify that meaning variation below the level of word type plays a role in shaping phonetic realization. The subtle yet statistically significant token-level alignments shown in Figure 6 demonstrate that phonetic variation of homophones is not only conveying and realizing information about word type identities (e.g., weight vs wait), but also reflects above chance level subtle diferences in the meanings of these word tokens as captured in the CE meaning space.

## 5.5.3 Experiment 4.3: mapping from centroids

We have shown that there is at least some alignment between the speech vectors of individual homophone tokens and their corresponding CEs. Next, we investigate whether the learned alignment gives rise to informative spectrotemporal structure by applying the learned production mapping to semantic centroids.

LDL learns the mappings through token-level semantic and speech vectors. Two possible outcomes may arise from these learned mappings: they either rely on subtle regularities that are generalizable without forming a clearly structured spectrotemporal pattern, or give rise to systematic patterns that, when contrasted between homophone members within a pair, are broadly consistent with the diference surfaces estimated from time-normalized spectrograms using GAM in Section 5.4. In this section, we examine the learned production mapping by applying the transformation matrix to prototypical semantic vectors, obtained by averaging semantic vectors within each homophone member, and ask to what extent the resulting predicted speech vectors correspond to the prototypical speech vectors.

Applying prototypical semantic vectors, i.e. centroids, to the learned mapping matrix, e.g., G for the production model, follows naturally from a geometric point of view. The mapping matrix fully describes a schema with which the form space should be rotated, skewed, scaled, or optionally reflected, to best align with the semantic space. The prototypical vectors are valid inputs to the mapping matrix because they also live in the same semantic space on which the mapping matrix was estimated. Importantly, the G matrix works on both the observed and not-yet-observed vectors in the form space: the mapping is productive.

As a consequence, we can, for a given homophone (e.g., wait) first calculate the centroid of its contextualized embeddings and then calculate its corresponding form vector predicted by the G mapping for that centroid. The mapping has never seen this centroid during training, yet it can predict for the prototypical meaning of wait what its spectrogram most likely looks like. Similarly, we can calculate the centroid of the spectrograms of wait, and map this prototypical spectrogram into the semantic space. The expectation is that the semantic centroid and the form centroid align.

If form and meaning centroids are indeed aligned (for empirical evidence with respect to pitch contours in Mandarin Chinese, see Chuang et al., 2026; Lu et al., 2026; Jin et al., 2026), then the prediction follows that the empirical spectrogram diference surface of homophone pairs such as weight and wait detected by a GAM should be very similar to the diference between the predicted spectrograms of weight and wait obtained by applying the mapping G to the semantic centroids of these two words.

We tested this prediction as follows. We first computed, for a given word type w, its corresponding semantic centroid $s _ { w }$ by averaging the CEs of all tokens of w. The resulting centroid represents the prototypical, context-independent, meaning of w. We then applied the G mapping to $s _ { w } ,$ which resulted in a predicted speech vector $\hat { c } _ { w }$ . As the mapping G is estimated using 50-dimensional speech vectors, in order to reconstruct the MEL spectrogram, we multiplied the 50-dimensional predicted speech vectors by the PCA basis vectors to obtain 1,050-dimensional vectors, which were subsequently reshaped into 50×21 matrices to recover the predicted MEL spectrograms.

It turns out that contrast spectrograms obtained by taking the diferences between two predicted MEL spectrograms generated from the semantic centroids are remarkably similar to the diference surfaces estimated with GAMs. Figure 7 presents four homophone pairs as examples: wait and weight, sell and cell, son and sun, and council and counsel. The contour lines and their 1SE confidence regions are taken from the GAM models. The color coding reflects the diferences in the spectrograms predicted from the homophones’ CE centroids. The color coding and the GAM-based contour lines are well aligned. The correspondence is not perfect, as stripes in the colored background resulting from reconstruction errors are clearly visible. However, maxima and minima all align across figures, indicating that the diference surfaces estimated using empirical speech vectors alone can also be well approximated from the semantic space.

The similarity between the contrast of predicted spectrogram centroids and the GAMestimated diference surface is surprising, and shows that the linear mapping from embeddings to spectrograms is of high quality.<sup>9</sup> The contrast between the LDL-predicted spectrograms and the diference spectrograms estimated by the GAMs converges to a surprising extent. These results would not be possible without the presence of the structures within each homophone pair that were observed in Section 5.5.1 and 5.5.2.

The examples shown in Figure 7 illustrate a further point, namely, that the diferences in homophones’ spectrograms are unlikely to reflect only diferences in the timing of articulatory gestures within homophones. Figure 7a revisits the wait and weight pair discussed in Section 5.4: the contrast spectrogram closely aligns with the GAM-estimated diference spectrogram, and the phone alignment profile, provided as a visual reference, consistently shows that listeners have to wait longer for the vowel onset in wait. Additional examples are provided in Figure 7. Figure 7b displays the diference spectrogram for sell and cell, together with its phone alignment profile. This phone alignment profile shows that there are hardly any diferences in the timing of the segments of sell and cell. Nevertheless, for this pair, there are substantial diferences within the segments. In Figure 7b, both the contrast spectrogram (colored background) and diference surface (contour lines) suggest more energy in the $/ \mathrm { s } /$ of cell during the first five timesteps (darker red colors), after which this $/ \mathrm { s } /$ becomes softer (darker blue colors at frequencies above frequency band 18, which is around 4,576Hz). The vowel of cell is also articulated with more energy in the first and second formants, and the final liquid is overall realized with less energy, especially for lower frequency bands. These efects appear only in the spectral domain, as the durations of all three segments are wellmatched in the temporal alignment profile.

A similar pattern is also observed for the sun–son pair (Figure 7c). Here, the $/ \mathrm { s } /$ of sun shows not only a longer duration, but is also realized with stronger high-frequency energy, as indicated by the red area in the interval of 10–20 normalized time units. It is also clear that the first and second formants of the vowel carry more energy for $s u n ,$ and in the case of the first formant, it continues to do so for the final nasal. Within-segment variation can also be seen for counsel and counsil (Figure 7d). For instance, the $/ \mathrm { s } /$ carries more fricative energy for counsel than for counsil, and the vowel of the second syllable of counsel has more energy in the lower frequency bands.

In summary, Figure 7 illustrates that the diferences in phonetic realization of homophone pairs align with their contextualized embeddings. The learned alignment of embeddings and spectrograms in LDL captures informative structures, as shown by the structured contrasts obtained from semantic centroids, and by the similarity between these contrasts and the GAM-estimated diference surface derived from empirical speech vectors.

## 6 Accounting for word duration

Time-normalized spectrograms provide fixed-size representations for tokens that difer in spoken word duration. However, the duration may still leave traces in a time-normalized spectrogram, partly because vowels and consonants may be shortened diferently in a spoken word (Gay, 1978; Max & Caruso, 1997; Janse et al., 2003). This raises the question of whether the diferences in the time-normalized spectrograms within homophone pairs are the consequence of spoken word duration itself, rather than the internal shape of the spectrogram. To address this issue, we carried out two further analyses: one investigates the predictability of word duration from speech vectors derived from time-normalized spectrograms and semantic vectors; the other assesses whether the efects observed in earlier analyses still hold after removing duration from the speech vectors.

Figure 8 summarizes the results of the first analyses, which consider to what extent spoken word durations are predictable from spectrograms. This figure presents the distributions of $R ^ { 2 }$ values (across the 35 homophone pairs) obtained with 5 diferent regression models predicting spoken word duration. The mean $R ^ { 2 }$ for a regression model predicting duration from the GPT2 embeddings is 24.15% (permutation baseline: 12%), but when duration is instead predicted from the speech vectors (Spec, orange), $R ^ { 2 }$ increases to 52.90%. This indicates that the speech vectors contain rich information about duration. Unsurprisingly, the speech vectors predicted from the embeddings contain as much information about duration as the embeddings themselves, as shown by the green boxplot.

![](images/e6b1dd3c35133910a68fe09b240187b2f5f9267a87da49a6dcec12a5ba65a28e.jpg)

![](images/9c5a51a050298934b9fcefc579109901de41699d1ed12401a3ddcd51386e7af6.jpg)

![](images/0d9ab44d0acef80c53a68220580c392508a8e775e3d3bf768782382264abc7db.jpg)

![](images/b2849b5c6475979935cc133bca3f7b335845c4e3e1aabbea3a2783955a3cb02d.jpg)

![](images/b798b72efbdf296e4c5e1b20c1a9a9b0b51853812bcdb4257980e4d552fb5cf4.jpg)

![](images/f2a53c21ca45d9f7270ceb7ec7fc966f92d597b4ec998aff1e3bcb6f529a8061.jpg)  
(a) wait and weight.

![](images/ebe7a6e083b24d81170ca675ceabcd128ec46b37565d76f17174e7aa2e3a9092.jpg)  
(b) sell and cell.  
(c) son and sun.  
(d) council and counsel.  
Figure 7: Selected diference spectrograms using GAMs applied to the observed spectra (contour lines) and using linear mappings from CE centroids (color coding). The close match between the predictions from LDL and the diference surfaces from GAM indicates that homophones’ prototypical meanings co-determine their prototypical spectra. Upper panels show phone alignment profiles as references. Four homophone twins are shown: (a) sell and cell, (b) son and sun, and (c) council and counsel.

We used Partial Least Squares (PLS) to remove the information about duration from the speech vectors (Wold, 1966; Wold et al., 2001, a brief introduction about PLS is also available in the appendix.). PLS iteratively identifies linear combinations of the predictors that are most strongly related to duration, and then removes these components, or deflates them following the terminology in Wold et al. (2001), so that the remaining predictors no longer contain enough information to predict duration beyond chance. The result of the PLS-deflated speech vector is shown as the red boxplot in Figure 8, and its $R ^ { 2 }$ has dropped to 13.00%, a value that is highly similar to that obtained with the permuted speech vectors that provide the baseline accuracy (12.74%).

$R ^ { 2 }$ of Duration Regression  
![](images/9ceae7b8762e1a83a2b7fb9ceba5603f1b640369f0d7ebd7438af9070a75bd11.jpg)  
Figure 8: $R ^ { 2 }$ distributions across homophone pairs of regression models for duration. Predicting from embeddings (GPT2, blue) has a much lower $R ^ { 2 }$ compared to predicting from the speech vectors derived from time-normalized spectrum (Spec, orange). Unsurprisingly, predicting from the fitted speech vectors (Predicted Spec, green) has the same overall $R ^ { 2 }$ as predicting from embeddings. $R ^ { 2 }$ is lowest when predicting from deflated speech vectors (Deflated Spec, red) and similar to the $R ^ { 2 }$ obtained with permuted spectrograms (Perm Spec, brown). Deflating the speech vectors successfully removes meaningful duration efects.

To assess the contribution of word duration in diferentiating homophone members, the second set of analyses compares how well homophone members can be classified within homophone pairs (1) when the original time-normalized spectrogram is used, (2) when the classification is based on the PLS-deflated spectrogram from which information about duration is removed, and (3) when the LDA is given access only to word duration. Figure 9 shows that predicting based on the PLS-deflated spectrogram decreases prediction accuracy $( M = 6 2 . 5 0 \% , S E = 1 . 2 4 )$ , an expected result given that duration is predictive for homophone discrimination (see also Gahl, 2008). However, the decrease in accuracy is relatively small, and accuracy remains substantially higher than when homophones have to be classified on the basis of duration only $( M = 5 8 . 1 \% , S E = 0 . 0 1 \% )$ . Therefore, we can conclude that, consistent with the results shown in Section 5.3, the time-normalized spectrogram can still distinguish homophone members even when the word duration is controlled for.

Consistent results are also observed when mapping from speech vectors to semantic vectors using PLS-deflated speech vectors (Figure 9, left panel). Compared to the original speech vectors from Section 5.5.1, the PLS-deflated spectrograms (orange) still align with the semantic vectors and can predict the semantic vector only slightly worse than the original speech vectors (green boxplot; $M = 5 8 . 3 9 \% , S E = 1 . 2 3 \% )$ . In the case of LDL mappings, the spoken word durations by themselves can only predict the semantic vectors at the chance level $( M = 5 1 . 7 9 \% , S E = 1 . 1 2 \% )$ . These results consistently show that word duration contributes to form-meaning alignment, but does not fully account for the alignment between speech and semantics.

In these analyses, we have considered whether the alignment between form and meaning can be traced back completely to spoken word duration. Our analyses do not support this argument, as time-normalized spectrograms from which duration has been removed remain highly predictive for the members of homophone pairs, indicating that the shape of their spectrotemporal realization still plays an important role in form-meaning alignment.

![](images/81535d2ac92a76e20d84ff1f7b2eecca8d54d717c12465fc81e57fed627f5dbb.jpg)  
Figure 9: Distributions (across the 35 homophone pairs) of LDA held-out accuracy (left panel) and nearest neighbor held-out accuracy (right panel) for models predicting from the observed spectrograms (orange), the deflated spectrograms (green), and from duration only (red). Removing duration from the spectrogram decreases prediction accuracy somewhat, but accuracy nevertheless remains high.

## 7 General Discussion

This study investigates the phonetic realization of English heterographic homophones and the role of word meaning in shaping the fine details of their pronunciation. To this end, we collected 200 tokens for 70 homophones of 35 homophone pairs from the Redhen 2016 dataset of US television news broadcasts. For each of the 14,000 tokens, we calculated the time-normalized spectrogram, together with a contextualized embedding using GPT-2.

Experiment 1 showed that the 35 homophone pairs can be teased apart with high accuracy (79.3% vs. a 2.8% permutation baseline) based on their vectorized time-normalized spectrograms. Experiment 2 clarified that LDA can discriminate with good accuracy (51.1%, vs. a 1.4% permutation baseline) between the 70 individual words (wait, weight, . . . ). Furthermore, LDA analyses carried out for the 400 tokens of individual homophone pairs consistently predicted homophone identity (e.g., weight vs. wait) far above the 50% baseline. This shows that there are systematic diferences in the speech signals of homophones within homophone pairs.

In order to visualize in what way the spectrograms of homophones within a homophone pair difer, Experiment 3 made use of the generalized additive model (GAM), using tensorproduct smooths. The 35 GAM models consistently pointed to substantial diferences between the spectrograms of homophone twins (the average decrease in AIC obtained by including a diference spectrogram in the model specification was 3114). The GAMs highlight clear diferences in the fine detail of phonetic realization between members in a homophone pair. For instance, for the homophone pair weight and wait, the listener has to wait longer for the vowel to arrive in wait as compared to weight. In another case, for the sun and son pair, the /s/ in sun is realized with stronger high-frequency energy than that in son.

Experiment 4 addressed the hypothesis that the diferences in the phonetic realization of members within a homophone pair can be traced back to their meanings using LDL. For each of the 35 homophone pairs, we constructed a comprehension mapping from spectrograms to CEs and a production mapping from CEs to spectrograms. These mappings were evaluated and examined in three sub-experiments. First, Experiment 4.1 evaluated the mappings using by-word nearest neighbor accuracy. Both mappings performed remarkably well, outperforming mappings trained on fully permuted embedding matrices by a wide margin. Next, in Experiment 4.2, we provided evidence supporting the alignment of form and meaning at the individual token level. For production, all homophone pairs yielded higher scores than the permutation baseline, and with only two exceptions, the empirical scores are well outside the 95% permutation baseline confidence interval. This constitutes clear evidence for the CE space and the phonetic space being aligned to some extent even at the token level.

In Experiment 4.3, we showed that the learned alignment has an informative structure linking the most prototypical meaning to the most prototypical speech realization. By using the semantic centroid as input, the learned alignment predicted the corresponding spectrogram centroid, and the contrast between the two prototypical spectrograms closely resembled the diference surface estimated by GAMs. Taken together, the meanings of homophone members are aligned with their phonetic realizations both at the type level, where word identity can be predicted above chance and semantic centroids map onto spectrogram centroids, and at the token level, where variation in meaning is reflected in phonetic variation across individual tokens.

The potential confound of word duration was addressed in Section 6. We showed that, even after controlling for duration using partial least squares, the deflated representations, which no longer predict duration above chance level, still diferentiate the members of a homophone pair and align with the semantic vectors, further supporting the results reported in Section 5.3 and 5.5. Importantly, these findings indicate that the realization diferences observed in the speech representations cannot be fully attributed to diferences in spoken word durations, the importance of which has been documented in previous studies (Drager, 2011; Gahl & Baayen, 2024). Instead, they suggest that other fine-grained aspects of phonetic realization vary systematically with word meaning.

The present set of 35 homophone pairs is too small for teasing apart the efects of morphological structure (Seyfarth et al., 2017) or word category (Lohmann, 2018). In all likelihood, these efects of morphological structure and word category are part and parcel of the semantic efects that we have documented. For instance, the diferentiation of the contextualized embeddings of homophones in a t-SNE map (see Figure 4) shows some diferentiation by word category, which suggests that our results are to some extent co-determined by the diferent semantics associated with nouns, verbs, and adjectives.

Two aspects of the present results are particularly surprising. First, contextualized embeddings provide only a rough approximation of the full richness of what individual speakers had in mind when uttering a homophone token, but they nevertheless align with the spoken word tokens. Second, the alignment can be described simply as a linear mapping, without requiring a more complex non-linear one. We will return to the linearity issue later in the discussion, but it already shows that the underlying isomorphism between form and meaning is significant.

More importantly, the diferences in realization reported in this series of experiments are not merely descriptive patterns in the observed samples, but systematic regularities that generalize to unseen data. In other words, the point is not simply that the homophone members such as wait and weight difer, but that the time-normalized spectrograms contain systematic cues that make homophone members distinguishable. This regularity is also found in the alignment between form and meaning. The precise source of these regularities remains unclear, particularly given the richness of information carried by the speech signal. However, Experiment 5.2 shows that the efect cannot be attributed only to utterance position, and an exclusive efect of word duration is ruled out in Section 6. Although other factors may potentially influence speech, such as speaker identity, emotional states, stress, focus, prominence, and even force-aligner errors, they are unlikely to systematically contribute to a generalizable diference in 400 random samples per homophone pair, across 35 homophone pairs drawn from diverse contexts and broadcast sources.

One possible factor underlying the generalizable diference of homophone members is the information content of diferent words. Van son & Pols (2003) reported that the higher the segmental information content of a segment, the less likely the segment is to be reduced. Segmental information content was operationalized by means of a corpus-based estimate of how distinctive a word is in context. Specifically, Van son & Pols (2003) computed the conditional probabilities of words occurring within the context surrounding the target word; the more distinctive the context of a target word, the lower its segmental information content and, in turn, the greater the tendency toward reduction, including slightly shorter duration.

Current results are compatible with this account: homophone members difer phonetically, possibly because each member’s contextual distinctiveness difers. This diference can be observed throughout the speech signal (i.e., the time-normalized spectrogram) and is not limited to duration. Furthermore, the current study approaches the efect of context diferently, namely, by using CEs, which can be viewed as a token-level semantic content representation computed from the preceding words rather than a scalar estimate of information content. These two notions are nevertheless related under the view of a large language model (LLM) as a compressor: information content corresponds to code length, whereas CEs can be understood as the internal predictive state from which such code is produced (for literature of viewing LLM as a compressor, see Deletang et al., 2024; Tseng et al., 2024). Our analyses provide evidence that words’ meaning in context, as captured by contextualized embeddings, predicts word tokens’ spectrograms not only at the level of word types but also above chance level for individual tokens. In other words, both token-level measures of informativity and surprisal, and token-level contextual meaning capture aspects of how the cognitive system responds to speech as it unfolds.

It is possible that prosodic factors such as rhythm, loudness, intonation, prominence, and stress contribute to the segmental diferences, but such an account is not very compelling, for two reasons. First, we found that the realization diferences align with contextualized embeddings. Contextualized embeddings are based on text and cannot directly encode intonation, stress, rhythm, or loudness, unless these are already co-determined by the grammar and lexical choice inherent in the text. In that case, those features will be grounded in grammar and lexis (for the reflection of grammar in contextualized embeddings, see Linzen & Baroni, 2021; Manning et al., 2020). The relation between contextualized embeddings and lexis is clearly visible in Figure 4: the primary factor shaping the t-SNE map is homophone semantics. Those prosodic factors that are not themselves systematically grounded in grammar and lexis are unlikely to be captured by the contextualized embeddings, and hence cannot explain the observed isomorphy between the form and meaning spaces.

Second, prosody can influence speech in a way independent from CEs, and we do not claim that CEs can completely reconstruct the spectrogram. Prosodic factors such as speech rate, the speaker’s emotional state, and the mechanical constraints on articulation that underlie many sandhi processes are all contributing factors but are outside the scope of what can be captured by current CEs. What the present study provides is a detailed computational/statistical model for the extent to which articulation is co-determined by meaning in context. The residuals of this model are non-negligible, and thus present a challenge for follow-up research, namely, to provide equally explicit and equally falsifiable complementary predictions for these residuals that are derived from factors such as the speaker’s emotional state, speech rate, register, focus, and the mechanical constraints on articulation.

We modeled form-meaning mapping with simple linear transformations, which appears to be all we need for predicting tokens’ time-normalized spectrograms from the corresponding contextualized embeddings and vice versa; a similar conclusion with respect to the phonetic realization of tone in Mandarin Chinese was reached by Chuang et al. (2026). Although the true form-meaning isomorphism underlying basic linguistic cognition (Huettig & Hulstijn, 2025) may involve both linear and non-linear components, the success of linear transformations in the current study suggests that, at least within the local form and meaning space examined here, the contribution of non-linear components is limited. The relatively strong linearity that we observed in the present study may also result in part from the speech representations used, which are derived from mel-spectrograms and thus reflect some of the non-linear transformation performed by the human cochlea. The resulting outputs from the cochlea thus ofer to the cognitive system representations that can be eficiently handled by linear mappings, which have broader cognitive advantages. They are simple and energy-eficient,<sup>10</sup> and the learning rules for linear mappings are well-established across plant cognition (Gagliano et al., 2016), animal cognition (Rescorla & Holland, 1982; Rescorla, 1988), and human language learning (Ellis, 2006; Heitmeier et al., 2023, 2026). Thus, linear mappings may ofer an evolutionary advantage for cognitive processes for which eficient routinized execution is essential, as they are less computationally intensive and require less energy.

The alignment of phonetic form and distributed meaning at the token level challenges a foundational assumption in linguistic phonology and psychology. First consider the classical models of Dell (1986) and Levelt et al. (1999). The finding that homophone twins have diferent pronunciations could perhaps be accommodated by assigning them diferent form nodes, although further ancillary measures would be required to ensure that their segments are realized with appropriate homophone-specific detail. However, the alignment of form and meaning at the token level argues against abstract word form units blocking meaning-in-context shaping segmental phonetic realization. Likewise, models of auditory word recognition (e.g., McQueen et al., 2006; Norris & McQueen, 2008; ten Bosch et al., 2022) in which abstract phones play a pivotal role, are severely challenged by the isomorphy that we have shown to exist between the phonetic space and the semantic embedding space.

In their criticism of formal phonology, Port & Leary (2005) stated that “Linguistics has mistakenly presumed that speech can always be spelled with letter-like tokens.” (p. 927) They argued against the common assumption that phonetic segments are formal symbols, and their study provides ample evidence for their case. Our results likewise argue against abstract phones, not only in psychological models of lexical processing, but also in linguistic theory. In the present study, we have used the term “segment” in a pre-theoretical and descriptive sense. The fact that the spectrograms of homophone tokens can be predicted to a considerable extent from their contextualized embeddings provides a clear indication that the theoretical construct of the phone may be redundant. Likewise, the fact that pitch contours on Mandarin words are aligned to a considerable extent with contextualized embeddings (Chuang et al., 2026; Jin et al., 2026; Lu et al., 2026) obviates the need for discrete tonal units. Furthermore, spoken word duration is also co-determined by semantics (Gahl & Baayen, 2024; Jin et al., 2026).

However, for modeling speech production, mappings from embeddings to time-normalized spectrograms, time-normalized pitch contours, or spoken word durations are no more (but also no less) than a promise of a proof of concept. Given that homophone duration can also be predicted from word embeddings as shown in Section 6 (in combination with other factors such as speech rate and contextual probabilities, see Gahl & Baayen, 2024), a word token’s spectrogram in real time can in principle be predicted by combining mappings from embeddings to spectrogram shape and mappings from embeddings to duration. A more realistic production model can directly model articulators by predicting sets of time series of control parameters for the vocal tract that jointly give rise to the speech signal, instead of predicting the speech signal itself. Whether this is possible with simple linear mappings currently is an open question. Considerable headway in this general direction has been made with a deep learning architecture trained on spectrograms and embeddings in conjunction with a physical model of the vocal tract (Sering, 2023; Sering & Baayen, 2024).

The distinction between Basic Language Cognition (BLC) on the one hand and Extended Language Cognition (ELC) Hulstijn (2024) (renamed as the Enhanced Literary Mind (ELM) by Huettig & Hulstijn (2025)) is useful for understanding the present results in a broader context. Hulstijn defines BLC as “a person’s ability to comprehend and produce spoken language in situations of everyday life, common to all adult native speakers in a given language community” (p. 3). BLC is viewed as implicit, unconscious, experience-driven cognition, acquired through massive exposure that enables fast and automatic speech processing. The Discriminative Lexicon Model (Heitmeier et al., 2026) was conceptualized as a computational proposal for basic lexical cognition. The DLM’s production and comprehension mappings are subliminal, not open to introspection, and represent automatized processes that are continuously updated with experience (cf. Heitmeier et al., 2023). As a consequence, we predict that in meta-linguistic tasks, language users cannot tell homophones apart from just their audio signals.

It remains an open question to what extent basic lexical cognition is afected by the cognitive processes that build on the knowledge and skills that come with training in literacy. We cannot rule out that the heterographic homophones are articulated diferently precisely because they have diferent spellings. If the tokens of homographic homophones (e.g., bank and bank) are found to have discriminable contextualized embeddings but indistinguishable pronunciations, this would provide evidence against the Discriminative Lexicon Model.

In summary, heterographic homophones are only approximately homophonous and difer not only with respect to their spoken word duration, but also with respect to a range of other aspects of their phonetic realization. The exact realization of a homophone token in normalized time is co-determined by its meaning in context, and can be approximated by a linear mapping from its contextualized embedding. Time-normalized spectrograms are a promising tool for probing the fine details of phonetic realization and obviate the need for phonetic transcriptions. The abstract sound symbols of formal phonology hide the truth of the remarkable alignment of phonetics with semantics from our eyes.

## Appendix A GAM-estimated diference surfaces of 35 pairs

![](images/0bb0ac09e4c6144e5aac92e4d0c288a1f84d25d0f5eb7401f60aca250d8057ce.jpg)

Figure A1: Average time-normalized spectrograms and the diference surface of aide and aid. æd: add, ad  
![](images/e3066ec0135b48827d041f198f30134d9b01d8cb4d0a707fd88c66d80543af35.jpg)  
Figure A2: Average time-normalized spectrograms and the diference surface of ad and add. bænd: banned, band

![](images/64ddf88553a2b6f6a8c6413b1715be2526ab55c9799c11ba1ee5030367cbc3b4.jpg)  
Figure A3: Average time-normalized spectrograms and the diference surface of band and banned. blu: blew, blue

![](images/205d93c4f93fcff4f9065379adcbaea82a6d9e4b57f7ea40a541835ef2c9b9b9.jpg)  
Figure A4: Average time-normalized spectrograms and the diference surface of blew and blue.

![](images/f59cf2e8d5bac81654897f2961ffd7fd405e1f5c7ced1ac3fc2d5e1ec58b2584.jpg)

Figure A5: Average time-normalized spectrograms and the diference surface of phil and fill. haia: higher, hire  
![](images/191ae3a18695ad348adcc8750811e42817fcf05dabee34067d533fbac0e2cc94.jpg)  
Figure A6: Average time-normalized spectrograms and the diference surface of hire and higher. houl: whole, hole

![](images/bd859da6085097c9915a6c58d0cb1c455b91fbc202456b201a13a8e4e67421df.jpg)  
Figure A7: Average time-normalized spectrograms and the diference surface of hole and whole. hia: here, hear

![](images/4a1b48fe2fdd0b0e29dcaad31853233bdc46c4bd84b2a88a5d15e9176d1a26bf.jpg)  
Figure A8: Average time-normalized spectrograms and the diference surface of here and hear.

![](images/3dd690c9a5bab772fa4a2ce15f0a314248dfc85bf5f1ee05dbf12f79ae7f564c.jpg)  
Figure A9: Average time-normalized spectrograms and the diference surface of counsel and council. kæpitəl: capitol, capital

![](images/7fa7f165dbcdf915da383d41cb84d9b2b855dbb0d800b1a5f20e856fe5eb5dcd.jpg)  
Figure A10: Average time-normalized spectrograms and the diference surface of capitol and capital. kur: corps, core

![](images/a287a8c76205bd88f42b3faf16d352305ba3839aaf7c4da2c06fa13b6375dc00.jpg)  
Figure A11: Average time-normalized spectrograms and the diference surface of corps and core. kruz: crews, cruise

![](images/775179db09af276c0bfc41be8bcaeb1483081cc4f739161eccea510a2f28016c.jpg)  
Figure A12: Average time-normalized spectrograms and the diference surface of crews and cruise.

![](images/f4f8d239f74e365fa7cbb17736393cd7f7ea350bdc17cefb97a2d94ba421e1be.jpg)  
Figure A13: Average time-normalized spectrograms and the diference surface of loan and lone. meil: male, mail

![](images/7e2d787169e64cd6a05260ee917a4435b1a88e9feee86bdf630dd6ad0aa5d3a9.jpg)  
Figure A14: Average time-normalized spectrograms and the diference surface of mail and male. murnın: mourning, morning

![](images/83e96686cd2b592414c302cd2ebd5336dd83393ff8091463412d6af60c423d39.jpg)  
Figure A15: Average time-normalized spectrograms and the diference surface of mourning and morning. mit: meet, meat

![](images/36cc7003f71bbfcb37d8ed325cea531161d6237d7520f538947ba4b02470d4d6.jpg)  
Figure A16: Average time-normalized spectrograms and the diference surface of meat and meet.

![](images/95170519d8cbd6a0bdf305edc67f1b79c8d69c341e29cd1c5d6a76009dc1a6f7.jpg)  
Figure A17: Average time-normalized spectrograms and the diference surface of nose and knows. nu: knew, new

![](images/bed4140ef53be626d53bcc0f3a2521e6339a154d4a0e440c838c64982574cb21.jpg)  
Figure A18: Average time-normalized spectrograms and the diference surface of new and knew. poul: pole, poll

![](images/78a17c176e2348f2032b94d57edb63d3b5f548d085f195d058a30aa0dfdd6857.jpg)

Figure A19: Average time-normalized spectrograms and the diference surface of pole and poll. pæst: passed, past  
![](images/e55b40af1a8b2674f6a3ae596bfd8232f105fb3c452e8fbcda65ae16b8304df2.jpg)  
Figure A20: Average time-normalized spectrograms and the diference surface of past and passed.

![](images/798f6b963db49997307754c02ef5e9a4e082bd60986668db15c7427a1327f95c.jpg)

Figure A21: Average time-normalized spectrograms and the diference surface of piece and peace. plein: plane, plain  
![](images/a97a2d1771a49bbd3058ed42a12bb37aeac6150e4ff2fa85a6fdffcf18504228.jpg)  
Figure A22: Average time-normalized spectrograms and the diference surface of plane and plain. pleınz: planes, plains

![](images/8bee43bf6a244ef079ac1d1c30d105b7a1f8bbee2b8292458a58fa0435372250.jpg)

Figure A23: Average time-normalized spectrograms and the diference surface of planes and plains. rait: write, right  
![](images/b5c76b614406440474552c23857c788266cd01b9b35b79262fd0389f14ff2366.jpg)  
Figure A24: Average time-normalized spectrograms and the diference surface of write and right.

![](images/bba33e1779093021fa6ebf902ae75b9a2ba288c9c5cf943b5e86f908cda7169c.jpg)  
Figure A25: Average time-normalized spectrograms and the diference surface of writes and rights. sait: sight, site

![](images/6631d514fe8279329a670218376f244b65dd9793ba80cb6717b13fde850ebbe1.jpg)  
Figure A26: Average time-normalized spectrograms and the diference surface of site and sight. sel: sell, cell

![](images/71aa1c8a4f32ee42ab0a8791c0546e4830a67ec017a4505b1290b6cbeaf0b364.jpg)  
Figure A27: Average time-normalized spectrograms and the diference surface of sell and cell. sΛn: son, sun

![](images/ab6900714eb1a935e457d4467677d7633647b6471267456b54b408cf69171805.jpg)  
Figure A28: Average time-normalized spectrograms and the diference surface of sun and son.

![](images/119a10264d6ac202c00b8412f674b4babdc1f03271e357a383483d06554a8676.jpg)  
Figure A29: Average time-normalized spectrograms and the diference surface of see and sea. sin: scene, seen

![](images/f351db3503123a6cb1e4c0cb616da34edced76df2871fc842337c7b25d935d0c.jpg)  
Figure A30: Average time-normalized spectrograms and the diference surface of seen and scene. stil: steal, steel

![](images/497af6c58afbb04f2d37a72bdcd06da7af51a57720f2207832e1f10423de5171.jpg)  
Figure A31: Average time-normalized spectrograms and the diference surface of steal and steel. taıd: tied, tide

![](images/eb03062d911d8f0d2f1ce398bf33a5bd8807b06041d1cc4ff451e7a2a2c65b80.jpg)  
Figure A32: Average time-normalized spectrograms and the diference surface of tide and tied.

![](images/e1f96ed43687e10f8042b0723dfdec94e2c0974c12812fcbda3c722bd86de141.jpg)

Figure A33: Average time-normalized spectrograms and the diference surface of weigh and way. weit: weight, wait  
![](images/57c62eb823671447190d739f125dd9e66558f1b66d9c9a627a54981a4ea51344.jpg)

Figure A34: Average time-normalized spectrograms and the diference surface of weight and wait. wik: weak, week  
![](images/fbb82346b0237419a3a93dc606f045283c8280a84abf0c5c30cbd25f5d154957.jpg)  
Figure A35: Average time-normalized spectrograms and the diference surface of weak and week.

## Appendix B Partial Least Squares

Partial least squares (PLS) is an iterative process. We write the original cue matrix $C$ as $C _ { 1 }$ , where the subscript indicates the first iteration. First, PLS finds a direction $w _ { 1 }$ that maximizes the covariance between the spectrogram features $C$ and the durations. The score $t _ { 1 }$ is the linear combination of the features with the found weights, $w _ { 1 }$ :

$$
t _ { 1 } = C _ { 1 } w _ { 1 }
$$

After finding the $t _ { 1 }$ , we can find the directions, or loadings, $p _ { 1 }$ , so that the corresponding rank-1 matrix, $t _ { 1 } p _ { 1 } ^ { \top }$ , best approximates $C { \mathrm { : } }$

$$
p _ { 1 } ^ { \top } = \frac { t _ { 1 } ^ { \top } C _ { 1 } } { t _ { 1 } ^ { \top } t _ { 1 } }
$$

Finally, we can remove the rank-1 matrix from $C .$ . The scores that maximally covary with the durations are thus removed from the $C$

$$
C _ { 2 } = C _ { 1 } - t _ { 1 } p _ { 1 } ^ { \intercal }
$$

We repeat this process for 3 times, and obtain three weights, $w _ { 1 } , \ w _ { 2 }$ , and $w _ { 3 }$ , and three scores, $t _ { 1 } , t _ { 2 } .$ , and $t _ { 3 } .$ , along with the loadings, $p _ { 1 } , p _ { 2 }$ and $p _ { 3 }$ . Using these components, we can construct $C _ { 1 } , C _ { 2 } , C _ { 3 }$ and $C _ { 4 }$ :

$$
{ \begin{array} { r l } & { C _ { 4 } = C _ { 3 } - t _ { 3 } p _ { 3 } ^ { \top } } \\ & { \quad = C _ { 2 } - t _ { 2 } p _ { 2 } ^ { \top } - t _ { 3 } p _ { 3 } ^ { \top } } \\ & { \quad = C _ { 1 } - t _ { 1 } p _ { 1 } ^ { \top } - t _ { 2 } p _ { 2 } ^ { \top } - t _ { 3 } p _ { 3 } ^ { \top } } \\ & { \quad = C _ { 1 } - \left[ t _ { 1 } \quad t _ { 2 } \quad t _ { 3 } \right] \left[ { \frac { p _ { 1 } ^ { \top } } { p _ { 2 } ^ { \top } } } \right] } \\ & { \quad = C _ { 1 } - T P ^ { \top } } \end{array} }
$$

One important property of the scores is that they are orthogonal to the deflated matrices obtained in later iterations. To see that, we can use $t _ { 1 }$ and $C _ { 2 }$ as an example:

$$
\begin{array} { r l } & { t _ { 1 } ^ { \top } C _ { 2 } = t _ { 1 } ^ { \top } ( C _ { 1 } - t _ { 1 } p _ { 1 } ^ { \top } ) } \\ & { \quad \quad \quad = t _ { 1 } ^ { \top } C _ { 1 } - t _ { 1 } ^ { \top } t _ { 1 } ( \frac { t _ { 1 } ^ { \top } C } { t _ { 1 } ^ { \top } t _ { 1 } } ) } \\ & { \quad \quad = 0 } \end{array}
$$

Using this result, we can further show that $t _ { 1 }$ is also orthogonal to $C _ { 3 } .$

$$
t _ { 1 } ^ { \top } C _ { 3 } = t _ { 1 } ^ { \top } ( C _ { 2 } - t _ { 2 } p _ { 2 } ^ { \top } ) = t _ { 1 } ^ { \top } C _ { 2 } - t _ { 1 } ^ { \top } ( C _ { 2 } w _ { 2 } ) p _ { 2 } ^ { \top } = 0
$$

Following the same reasoning, all extracted scores are orthogonal to the final deflated matrix $C _ { 4 }$ , that is,

$$
T ^ { \top } C _ { 4 } = \left[ \overline { { { - \_ } } } \ t _ { 2 } ^ { \top } \ \overline { { { - \_ } } } \right] C _ { 4 } = 0
$$

We denote $C _ { 4 }$ as $C _ { \mathrm { d e f } }$ , as this is the deflated speech representations which can no longer predict durations beyond the permutation baseline. Using the orthogonality derived above, the Frobenius norm of the original matrix $C _ { 1 }$ can be decomposed into the sum of the Frobenius norms of $C _ { \mathrm { d e f } }$ and the components extracted by PLS, $T P ^ { \top }$

$$
\begin{array} { r l } & { \| C _ { 1 } \| _ { F } ^ { 2 } = \| T P ^ { \top } + C _ { \mathrm { d e f f } } \| _ { F } ^ { 2 } } \\ & { \qquad = \| T P ^ { \top } \| _ { F } ^ { 2 } + \| C _ { \mathrm { d e f f } } \| _ { F } ^ { 2 } + 2 \operatorname { t r } \left( ( T P ^ { \top } ) ^ { \top } C _ { \mathrm { d e f f } } \right) } \\ & { \qquad = \| T P ^ { \top } \| _ { F } ^ { 2 } + \| C _ { \mathrm { d e f f } } \| _ { F } ^ { 2 } } \end{array}
$$

The cross-product term is zero because we have shown that $T ^ { \top } C _ { \mathrm { d e f l } } = 0$ , and a rearrangement of the cross-product term becomes

$$
( T P ^ { \top } ) ^ { \top } C _ { \mathrm { d e f l } } = P ( T ^ { \top } C _ { \mathrm { d e f l } } ) = 0
$$

This derivation shows that although PLS does not provide orthogonal bases, as the loadings $P$ are generally not orthogonal, the fitted component $T P ^ { \top }$ is nevertheless orthogonal to the deflated matrix $C _ { \mathrm { d e f } }$ . This still allows us to think of the decomposition in linear terms. For more background on the Frobenius norm, see Appendix C.

Using this decomposition property, we can further quantify the role of the duration component in the two components found in the LDL production model: the prediction $\hat { C }$ and the residual E. The proportion of variance, measured by the Frobenius norm, accounted for by $\hat { C }$ is .43, and that of $E$ is .56. Consistent with Figure 8, the $R ^ { 2 }$ of duration regression for $\check { C }$ is .24, and that for E is .38. We can further decompose $\hat { C }$ using PLS into a duration component $\hat { C } _ { d u r }$ and $\hat { C } _ { o t h e r }$ . The proportion of variance of $\hat { C } _ { d u r }$ is .08, and that of $\hat { C } _ { o t h e r }$ is .35, already showing that duration is not a major component in the form-meaning alignment. Most of ${ \hat { C } } ,$ therefore, reflects information beyond duration. Looking further at the $R ^ { 2 }$ of duration regression clarifies where the duration efect comes from: using $\hat { C } _ { d u r }$ as predictors, the $R ^ { 2 }$ is .16, while that for $\hat { C } _ { o t h e r }$ is .08. Therefore, part of the duration component is reflected in the semantic embeddings and is also mapped into the predicted spectrograms. This is consistent with the fact that duration is codetermined by semantics. However, a substantial part of the information unrelated to duration also contributes to the alignment.

Not all duration information can be captured by semantics, at least not by the contextualized embeddings derived from text. We again decompose the residual E into duration and other components. The proportion of variance of $E _ { d u r }$ is .31, and that of $E _ { o t h e r }$ is .25, indicating that a relatively large amount of information in the residual still concerns duration. The $R ^ { 2 }$ of duration regression using $E _ { d u r }$ is .28, and that for $E _ { o t h e r }$ is .10. These numbers are consistent with prosodic factors remaining important in the spectrogram features.

## Appendix C Frobenius norm decomposition under LDL

We use the Frobenius norm to evaluate alignment in Linear Discriminative Learning (LDL), and interpret it as variance explained, in a way comparable to $R ^ { 2 }$ . This interpretation requires clarification, since variance is not formally defined for matrices and because the corresponding decomposition into explained and residual components is not explicit in LDL. This section provides the justification for using the Frobenius norm under this interpretation.

In what follows, we first show how the Frobenius norm is related to the matrix trace and why it is equivalent to the sum of squared matrix elements. Secondly, we show that under LDL, the Frobenius norms of the prediction and the residual can be linearly decomposed.

To begin, we write $C$ as vertically stacked row vectors, and following the convention in LDL, we denote each row vector by $c _ { i }$ :

$$
C = \left( \begin{array} { c } { { - c _ { 1 } - } } \\ { { - c _ { 2 } - } } \\ { { \vdots } } \\ { { - c _ { n } - } } \end{array} \right) .
$$

Then

$$
C C ^ { \top } = \left( { \begin{array} { c c c c } { c _ { 1 } c _ { 1 } ^ { \top } } & { c _ { 1 } c _ { 2 } ^ { \top } } & { \cdot \cdot \cdot } & { c _ { 1 } c _ { n } ^ { \top } } \\ { c _ { 2 } c _ { 1 } ^ { \top } } & { c _ { 2 } c _ { 2 } ^ { \top } } & { \cdot \cdot \cdot } & { c _ { 2 } c _ { n } ^ { \top } } \\ { \vdots } & { \vdots } & { \cdot } & { \vdots } \\ { c _ { n } c _ { 1 } ^ { \top } } & { c _ { n } c _ { 2 } ^ { \top } } & { \cdot \cdot \cdot } & { c _ { n } c _ { n } ^ { \top } } \end{array} } \right) .
$$

The trace picks out the diagonal entries:

$$
\operatorname { t r } ( C C ^ { \top } ) = c _ { 1 } c _ { 1 } ^ { \top } + c _ { 2 } c _ { 2 } ^ { \top } + \cdot \cdot \cdot + c _ { n } c _ { n } ^ { \top } .
$$

Therefore,

$$
\| C \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { n } { \pmb { c } } _ { i } { \pmb { c } } _ { i } ^ { \top } = \operatorname { t r } ( C C ^ { \top } ) = \operatorname { t r } ( C ^ { \top } C ) .
$$

The last equality follows from the cyclic property of trace.

In the LDL production route, and similarly in the comprehension route, we estimate the G matrix by least squares, where speech vectors are predicted from their corresponding semantic embeddings. That is,

$$
\begin{array} { l } { { C = S G + ( C - S G ) } } \\ { { \ } } \\ { { = \hat { C } + E . } } \end{array}
$$

The Frobenius norm of C becomes,

$$
\begin{array} { r l } & { \| C \| _ { F } ^ { 2 } = \| \hat { C } + E \| _ { F } ^ { 2 } } \\ & { \qquad = \| \hat { C } \| _ { F } ^ { 2 } + \| E \| _ { F } ^ { 2 } + 2 \operatorname { t r } ( \hat { C } ^ { \top } E ) . } \end{array}
$$

There is a cross-product term in the equation, which equals zero because the least-squares solution decomposes $C$ into two orthogonal components. To see this, we can write $\hat { C }$ in terms of $C$ and a projection matrix $P _ { S }$

$$
\begin{array} { c } { \hat { C } = S ( S ^ { \top } S ) ^ { - 1 } S ^ { \top } C } \\ { = P s C . } \end{array}
$$

The derivation becomes clearer if we write S using the singular value decomposition as $U _ { S } \Sigma _ { S } V _ { S } ^ { \top }$ , and the predicted $\hat { C }$ can be written as

$$
\begin{array} { r l } & { \hat { C } = P _ { S } C } \\ & { \quad = U _ { S } \Sigma _ { S } { V _ { S } ^ { \top } } ( { V _ { S } \Sigma _ { S } ^ { \top } \Sigma _ { S } V _ { S } ^ { \top } } ) ^ { - 1 } V _ { S } \Sigma _ { S } ^ { \top } U _ { S } ^ { \top } C } \\ & { \quad = U _ { S } U _ { S } ^ { \top } C . } \end{array}
$$

Here, since $U _ { S }$ has orthonormal columns, $U _ { S } ^ { \top } U _ { S } = I$ , which allows us to derive the orthogonality between $\hat { C }$ and $E = C - { \hat { C } } ;$

$$
\begin{array} { r l } & { \hat { C } ^ { \top } E = \hat { C } ^ { \top } ( C - \hat { C } ) } \\ & { \quad \quad = ( U _ { S } U _ { S } ^ { \top } C ) ^ { \top } ( C - U _ { S } U _ { S } ^ { \top } C ) } \\ & { \quad \quad = C ^ { \top } U _ { S } U _ { S } ^ { \top } ( I - U _ { S } U _ { S } ^ { \top } ) C } \\ & { \quad \quad = C ^ { \top } ( U _ { S } U _ { S } ^ { \top } - U _ { S } U _ { S } ^ { \top } U _ { S } U _ { S } ^ { \top } ) C } \\ & { \quad \quad = C ^ { \top } ( U _ { S } U _ { S } ^ { \top } - U _ { S } U _ { S } ^ { \top } ) C } \\ & { \quad \quad = 0 . } \end{array}
$$

Since $\hat { C }$ and $E$ are orthogonal, $\mathrm { t r } ( \hat { C } ^ { \top } E )$ equals zero. The Frobenius norm decomposition of $C$ can therefore be written as

$$
\| C \| _ { F } ^ { 2 } = \| \hat { C } \| _ { F } ^ { 2 } + \| E \| _ { F } ^ { 2 } ,
$$

which means that the Frobenius norm of $C$ can be expressed as the sum of the Frobenius norms of the prediction $\hat { C }$ and the residual E.

## References

Arnold, D., Tomaschek, F., Sering, K., Lopez, F., & Baayen, R. H. (2017). Words from spontaneous conversational speech can be recognized with human-like accuracy by an error-driven learning algorithm that discriminates between meanings straight from smart acoustic features, bypassing the phoneme as recognition unit. PLOS ONE, 12(4), e0174623.

Baayen, R. H., Chuang, Y.-Y., Shafaei-Bajestan, E., & Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de) composition but in linear discriminative learning. Complexity, 2019(1), 4895891.

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for selfsupervised learning of speech representations. Advances in neural information processing systems, 33, 12449–12460.

Bartelds, M., de Vries, W., Sanal, F., Richter, C., Liberman, M., & Wieling, M. (2022). Neural representations for modeling variation in speech. Journal of Phonetics, 92, 101137.

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of machine learning research, 3(Feb), 1137–1155.

Bergen, B. K. (2004). The psychological reality of phonaesthemes. Language, 80(2), 290–311.

Blasi, D. E., Wichmann, S., Hammarström, H., Stadler, P. F., & Christiansen, M. H. (2016). Sound–meaning association biases evidenced across thousands of languages. Proceedings of the National Academy of Sciences, 113(39), 10818–10823.

Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, 135–146.

Caramazza, A., Costa, A., Miozzo, M., & Bi, Y. (2001). The specific-word frequency effect: implications for the representation of homophones in speech production. Journal of Experimental Psychology: Learning, Memory, and Cognition, 27(6), 1430.

Caucheteux, C., Gramfort, A., & King, J.-R. (2022). Deep language algorithms predict semantic comprehension from brain activity. Scientific Reports, 12(1).

Celata, C., Bissiri, M. P., & Schmid, C. (2022). Does morphology impact the pronunciation of consonant clusters? evidence from German. Studi e Saggi Linguistici, 60, 51–79.

Chuang, Y.-Y., Baayen, R. H., & Bell, M. J. (2023). Do words sing their own tunes? wordspecific pitch realizations in Mandarin and English. In Proceedings of ICPhS 2023.

Chuang, Y.-Y., Bell, M. J., Tseng, Y.-H., & Baayen, R. H. (2026). Word-specific tonal realizations in Mandarin. Language. arXiv:2405.07006v2.

Danks, D. (2003). Equilibria of the rescorla–wagner model. Journal of Mathematical Psychology, 47(2), 109–121.

Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML).

de Varda, A. G. & Marelli, M. (2025). Cracking arbitrariness: A data-driven study of auditory iconicity in spoken English. Psychonomic Bulletin & Review, 32(3), 1425–1442.

Deletang, G., Ruoss, A., Duquenne, P.-A., Catt, E., Genewein, T., Mattern, C., Grau-Moya, J., Wenliang, L. K., Aitchison, M., Orseau, L., Hutter, M., & Veness, J. (2024). Language modeling is compression. In The Twelfth International Conference on Learning Representations.

Dell, G. S. (1986). A spreading-activation theory of retrieval in sentence production. Psychological review, 93(3), 283.

Dell, G. S. (1990). Efects of frequency and vocabulary type on phonological speech errors. Language and cognitive processes, 5(4), 313–349.

Drager, K. K. (2011). Sociophonetic variation and the lemma. Journal of Phonetics, 39(4), 694–707.

Ellis, N. C. (2006). Selective attention and transfer phenomena in l2 acquisition: Contingency, cue competition, salience, interference, overshadowing, blocking, and perceptual learning. Applied linguistics, 27(2), 164–194.

Elman, J. L. (1991). Distributed representations, simple recurrent networks, and grammatical structure. Machine Learning, 7(2–3), 195–225.

Elman, J. L. (2004). An alternative view of the mental lexicon. Trends in Cognitive Sciences, 8(7), 301–306.

Ernestus, M. & Baayen, H. (2006). The functionality of incomplete neutralization in Dutch: The case of the past-tense formation. In L. Goldstein, D. H. Whalen, & C. T. Best (Eds.), Laboratory Phonology 8 (pp. 27–50). Berlin, New York: De Gruyter Mouton.

Ferreira, V. S. & Grifin, Z. M. (2003). Phonological influences on lexical (mis)selection. Psychological Science, 14(1), 86–90.

Firth, J. (1957). A synopsis of linguistic theory, 1930-1955. Studies in linguistic analysis, (pp. 10–32).

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2), 179–188.

Fourakis, M. & Iverson, G. K. (1984). On the ‘incomplete neutralization’ of German final obstruents. Phonetica, 41(3), 140–149.

Gagliano, M., Vyazovskiy, V. V., Borbély, A. A., Grimonprez, M., & Depczynski, M. (2016). Learning by association in plants. Scientific reports, 6(1), 38427.

Gahl, S. (2008). Time and thyme are not homophones: The efect of lemma frequency on word durations in spontaneous speech. Language, 84(3), 474–496.

Gahl, S. & Baayen, R. H. (2024). Time and thyme again: Connecting English spoken word duration to models of the mental lexicon. Language, 100(4), 623–670.

Gay, T. (1978). Efect of speaking rate on vowel formant movements. The Journal of the Acoustical Society of America, 63(1), 223–230.

Godfrey, J. J., Holliman, E. C., & McDaniel, J. (1992). SWITCHBOARD: Telephone speech corpus for research and development. In Acoustics, Speech, and Signal Processing, IEEE International Conference on, volume 1 (pp. 517–520).: IEEE Computer Society.

Goldstein, A., Grinstein-Dabush, A., Schain, M., Wang, H., Hong, Z., Aubrey, B., Nastase, S. A., Zada, Z., Ham, E., Feder, A., Gazula, H., Buchnik, E., Doyle, W., Devore, S., Dugan, P., Reichart, R., Friedman, D., Brenner, M., Hassidim, A., Devinsky, O., Flinker, A., & Hasson, U. (2024). Alignment of brain embeddings and artificial contextual embeddings in natural language points to common geometric patterns. Nature Communications, 15(1).

Greenwood, D. D. (1990). A cochlear frequency-position function for several species—29 years later. The Journal of the Acoustical Society of America, 87(6), 2592–2605.

Grifiths, T. L., Steyvers, M., & Tenenbaum, J. B. (2007). Topics in semantic representation. Psychological review, 114(2), 211.

Harris, Z. S. (1954). Distributional structure. Word, 10(2-3), 146–162.

Hay, J. (2004). Causes and consequences of word structure. Routledge.

Heitmeier, M., Chuang, Y.-Y., & Baayen, R. H. (2023). How trial-to-trial learning shapes mappings in the mental lexicon: Modelling lexical decision with linear discriminative learning. Cognitive Psychology, 146, 101598.

Heitmeier, M., Chuang, Y.-Y., & Baayen, R. H. (2026). The Discriminative Lexicon: Theory and implementation in the Julia package JudiLing. Cambridge: Cambridge University Press.

Huettig, F. & Hulstijn, J. (2025). The enhanced literate mind hypothesis. Topics in Cognitive Science, 17(4), 909–918.

Hulstijn, J. (2024). Predictions of individual diferences in the acquisition of native and non-native languages: An update of BLC theory. Languages, 9(5), 173.

Janse, E., Nooteboom, S., & Quené, H. (2003). Word-level intelligibility of time-compressed speech: prosodic and segmental factors. Speech Communication, 41(2-3), 287–301.

Jescheniak, J. D. & Levelt, W. J. M. (1994). Word frequency efects in speech production: Retrieval of syntactic information and of phonological form. Journal of Experimental Psychology: Learning, Memory, and Cognition, 20(4), 824–843.

Jescheniak, J. D., Meyer, A. S., & Levelt, W. J. M. (2003). Specific-word frequency is not all that counts in speech production: Comments on Caramazza, Costa, et al. (2001) and new experimental data. Journal of Experimental Psychology: Learning, Memory, and Cognition, 29(3), 432–438.

Jin, X., Ernestus, M., & Baayen, R. H. (2026). A new kid on the block: Distributional semantics predicts the word-specific tone signatures of monosyllabic words in conversational Taiwan Mandarin. Journal of Phonetics. arXiv preprint arXiv:2511.17337.

Jollife, I. (2010). Principal Component Analysis. Springer Series in Statistics. Springer, 2nd ed. edition.

Jones, M. N. & Mewhort, D. J. K. (2007). Representing word meaning and order information in a composite holographic lexicon. Psychological Review, 114(1), 1–37.

Joo, J., Steen, F. F., & Turner, M. (2017). Red hen lab: Dataset and tools for multimodal human communication research. KI-Künstliche Intelligenz, 31, 357–361.

Kemps, R. J. J. K., Ernestus, M., Schreuder, R., & Harald Baayen, R. (2005a). Prosodic cues for morphological complexity: The case of Dutch plural nouns. Memory and Cognition, 33(3), 430–446.

Kemps, R. J. J. K., Wurm, L. H., Ernestus, M., Schreuder, R., & Baayen, H. (2005b). Prosodic cues for morphological complexity in Dutch and English. Language and Cognitive Processes, 20(1–2), 43–73.

Kenstowicz, M. J. (1994). Phonology in generative grammar, volume 7. Blackwell Cambridge, MA.

Kiparsky, P. (1982). Lexical morphology and phonology. In I.-S. Yang (Ed.), Linguistics in the Morning Calm: Selected Papers From SICOL (pp. 3––91).: Seoul:Hanshin.

Landauer, T. K. & Dumais, S. T. (1997). A solution to plato’s problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge. Psychological review, 104(2), 211.

Levelt, W. J., Roelofs, A., & Meyer, A. S. (1999). A theory of lexical access in speech production. Behavioral and brain sciences, 22(1), 1–38.

Levelt, W. J. M. (2001). Spoken word production: A theory of lexical access. Proceedings of the National Academy of Sciences, 98(23), 13464–13471.

Linzen, T. & Baroni, M. (2021). Syntactic structure from deep learning. Annual Review of Linguistics, 7(1), 195–212.

Lohmann, A. (2018). Cut (n) and cut (v) are not homophones: Lemma frequency afects the duration of noun–verb conversion pairs. Journal of Linguistics, 54(4), 753–777.

Lu, Y., Chuang, Y.-Y., & Baayen, R. H. (2026). The realization of tones in spontaneous spoken Taiwan Mandarin: a corpus-based survey and theory-driven computational modeling. Corpus Linguistics and Linguistic Theory. arXiv preprint arXiv:2503.23163.

Lund, K. & Burgess, C. (1996). Producing high-dimensional semantic spaces from lexical co-occurrence. Behavior research methods, instruments, & computers, 28(2), 203–208.

Maaten, L. v. d. & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579–2605.

Manning, C. D., Clark, K., Hewitt, J., Khandelwal, U., & Levy, O. (2020). Emergent linguistic structure in artificial neural networks trained by self-supervision. Proceedings of the National Academy of Sciences, 117(48), 30046–30054.

Marsolek, C. J. (2008). What antipriming reveals about priming. Trends in Cognitive Sciences, 12(5), 176–181.

Max, L. & Caruso, A. J. (1997). Acoustic measures of temporal intervals across speaking rates: Variability of syllable- and phrase-level relative timing. Journal of Speech, Language, and Hearing Research, 40(5), 1097–1100.

McFee, B., Rafel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference, volume 8.

McQueen, J. M., Cutler, A., & Norris, D. (2006). Phonological abstraction in the mental lexicon. Cognitive Science, 30(6), 1113–1126.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.

Miozzo, M. & Caramazza, A. (2003). When more is less: a counterintuitive efect of distractor frequency in the picture-word interference paradigm. Journal of Experimental Psychology: General, 132(2), 228.

Miozzo, M. & Caramazza, A. (2005). The representation of homophones: evidence from the distractor-frequency efect. Journal of Experimental Psychology: Learning, Memory, and Cognition, 31(6), 1360.

Monaghan, P., Shillcock, R. C., Christiansen, M. H., & Kirby, S. (2014). How arbitrary is language? Philosophical Transactions of the Royal Society B: Biological Sciences, 369(1651), 20130299.

Norris, D. & McQueen, J. (2008). Shortlist B: A Bayesian model of continuous speech recognition. Psychological Review, 115(2), 357–395.

Ochshorn, R. & Hawkins, M. (2017). Gentle forced aligner. github. com/lowerquality/gentle.

Pavlick, E. (2022). Semantic structure in deep learning. Annual Review of Linguistics, 8(1), 447–471.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Derczynski, L., Du, X., Grella, M., Gv, K., He, X., Hou, H., Kazienko, P., Kocon, J., Kong, J., Koptyra, B., Lau, H., Lin, J., Mantri, K. S. I., Mom, F., Saito, A., Song, G., Tang, X., Wind, J., Woźniak, S., Zhang, Z., Zhou, Q., Zhu, J., & Zhu, R.-J. (2023). RWKV: Reinventing RNNs for the transformer era. In H. Bouamor, J. Pino, & K. Bali (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2023 (pp. 14048–14077). Singapore: Association for Computational Linguistics.

Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP) (pp. 1532–1543).

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. In M. Walker, H. Ji, & A. Stent (Eds.), Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (pp. 2227–2237). New Orleans, Louisiana: Association for Computational Linguistics.

Pimentel, T., McCarthy, A. D., Blasi, D., Roark, B., & Cotterell, R. (2019). Meaning to form: Measuring systematicity as information. In Proceedings of the 57th annual meeting of the association for computational linguistics (pp. 1751–1764).

Plag, I., Homann, J., & Kunter, G. (2017). Homophony and morphology: the acoustics of word-final s in English. Journal of Linguistics, 53(1), 181–216.

Port, R. & Crawford, P. (1989). Incomplete neutralization and pragmatics in German. Journal of Phonetics, 17(4), 257–282.

Port, R. F. & Leary, A. P. (2005). Against formal phonology. Language, 81, 927–964.

Port, R. F. & O’Dell, M. L. (1985). Neutralization of syllable-final voicing in German. Journal of Phonetics, 13(4), 455–471.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via largescale weak supervision. In International conference on machine learning (pp. 28492–28518).: PMLR.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. (2018). Improving language understanding by generative pre-training. OpenAI Blog.

Rao, C. R. (1948). The utilization of multiple measurements in problems of biological classification. Journal of the Royal Statistical Society. Series B (Methodological), 10(2), 159–203.

Rescorla, R. A. (1988). Pavlovian conditioning. It’s not what you think it is. American Psychologist, 43(3), 151–160.

Rescorla, R. A. & Holland, P. C. (1982). Behavioral studies of associative learning in animals. Annual Review of Psychology, 33(1), 265–308.

Rescorla, R. A. & Wagner, A. R. (1972). A theory of pavlovian conditioning: Variations in the efectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), Classical Conditioning II: Current Research and Theory (pp. 64–99). New York: Appleton-Century-Crofts.

Roelofs, A. (1997). The weaver model of word-form encoding in speech production. Cognition, 64(3), 249–284.

Roettger, T., Winter, B., Grawunder, S., Kirby, J., & Grice, M. (2014). Assessing incomplete neutralization of final devoicing in German. Journal of Phonetics, 43, 11–25.

Saito, M., Tomaschek, F., Sun, C.-C., & Baayen, R. H. (2024). Articulatory efects of frequency modulated by inflectional meanings. In M. Schlechtweg (Ed.), Interfaces of Phonetics (pp. 125–154). De Gruyter.

Schmitz, D. & Baer-Henney, D. (2024). Morphology renders homophonous segments phonetically diferent: Word-final /s/ in German. In Speech Prosody 2024: ISCA.

Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. Proceedings of the National Academy of Sciences, 118(45).

Sering, K. (2023). Predictive Articulatory speech synthesis Utilizing Lexical Embeddings (PAULE). University of Tübingen. PhD thisis.

Sering, K. & Baayen, H. (2024). Articulatory speech synthesis without phones? In ISSP 2024 13th International Seminar on Speech Production, issp\_2024 (pp. 5–7).: ISCA.

Seyfarth, S., Garellek, M., Gillingham, G., Ackerman, F., & Malouf, R. (2017). Acoustic differences in morphologically-distinct homophones. Language, Cognition and Neuroscience, 33(1), 32–49.

Shafaei-Bajestan, E., Moradipour-Tari, M., Uhrig, P., & Baayen, R. H. (2021). Ldl-auris: a computational model, grounded in error-driven learning, for the comprehension of single spoken words. Language, Cognition and Neuroscience, 38(4), 509–536.

Shaoul, C. & Westbury, C. (2010). Exploring lexical co-occurrence space using hidex. Behavior Research Methods, 42(2), 393–413.

Song, J. Y., Demuth, K., Evans, K., & Shattuck-Hufnagel, S. (2013). Durational cues to fricative codas in 2-year-olds’ American English: Voicing and morphemic factors. The Journal of the Acoustical Society of America, 133(5), 2931–2946.

Stevens, S. S., Volkmann, J., & Newman, E. B. (1937). A scale for the measurement of the psychological magnitude pitch. The journal of the acoustical society of america, 8(3), 185–190.

Sutskever, I., Martens, J., & Hinton, G. E. (2011). Generating text with recurrent neural networks. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 1017–1024).

ten Bosch, L., Boves, L., & Ernestus, M. (2022). Diana, a process-oriented model of human auditory word recognition. Brain Sciences, 12(5), 681.

Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In A. Korhonen, D. Traum, & L. Màrquez (Eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 4593–4601). Florence, Italy: Association for Computational Linguistics.

Tseng, Y.-H., Chen, P.-E., Lian, D.-C., & Hsieh, S.-K. (2024). The semantic relations in LLMs: An information-theoretic compression approach. In Proceedings of the Workshop: Bridging Neurons and Symbols for Natural Language Processing and Knowledge Graphs Reasoning (NeusymBridge) @ LREC-COLING-2024 (pp. 8–21). Torino, Italia: ELRA and ICCL.

Van son, R. & Pols, L. C. (2003). Information structure and eficiency in speech production. In 8th European Conference on Speech Communication and Technology (Eurospeech 2003) (pp. 769–772).

Warner, N., Good, E., Jongman, A., & Sereno, J. (2006). Orthographic vs. morphological incomplete neutralization efects. Journal of Phonetics, 34(2), 285–293.

Warner, N., Jongman, A., Sereno, J., & Kemps, R. (2004). Incomplete neutralization and other sub-phonemic durational diferences in production and perception: evidence from Dutch. Journal of Phonetics, 32(2), 251–276.

Widrow, B. & Hof, M. E. (1960). Adaptive switching circuits. In 1960 IRE WESCON Convention Record, Part 4 (pp. 96–104).: Institute of Radio Engineers Institute of Radio Engineers.

Wold, H. (1966). Estimation of principal components and related models by iterative least squares. Multivariate analysis, (pp. 391–420).

Wold, S., Sjöström, M., & Eriksson, L. (2001). Pls-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109–130.

Yang, M., Shekar, R. C. M. C., Kang, O., & Hansen, J. H. L. (2023). What can an accent identifier learn? Probing phonetic and prosodic information in a wav2vec2-based accent identification model. In Interspeech 2023 (pp. 1923–1927).

Wood, S. N. (2017). Generalized Additive Models. New York: Chapman & Hall/CRC.