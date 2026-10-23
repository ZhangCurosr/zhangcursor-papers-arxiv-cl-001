# Direct cueing of word identity through word duration: evidence from Mandarin homophones

Xiaomeng Zhu,<sup>1</sup> Megan Niu,<sup>1</sup> Jason Shaw,<sup>1</sup> and Kevin Tang<sup>2,</sup> <sup>a)</sup>

<sup>1)</sup>Department of Linguistics, Yale University, New Haven,

USA

<sup>2)</sup>Department of English Language and Linguistics, Institute of

English and American Studies, Faculty of Arts and Humanities,

Heinrich Heine University Düsseldorf, Düsseldorf, 40225,

Germany

cueing of word identity through word duration: evidence from Mandarin homophones While there is increasing evidence that words are stored in the mental lexicon with some phonetic detail, there are open questions about exactly which phonetic details are stored and whether they are functionally relevant for spoken word recognition. Word-level phonetic cues, such as word duration, provide an appropriate test case. Across languages, including Mandarin Chinese, high frequency words are shorter than low frequency words. Using a homophone ordering task, we investigate whether differences in word duration impact word recognition. Mandarin Chinese was examined since it has abundant disyllabic homophones. Our results indicate that participants were able to use word duration to recognize the order of homophone pairs, selected to vary in lexical frequency. Both with high noise levels and when no other cues were available, participants mapped the shorter duration auditory stimulus to the higher frequency word. The results provide evidence that word duration can directly cue the identity of words, suggesting that it may be a phonetic detail incorporated into the lexical representation. We discuss this direct cueing account alongside alternative accounts of how word duration may modulate word recognition.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## I. INTRODUCTION

There is ample evidence that the temporal duration of linguistic units in the speech signal may function to cue the identity of those units. This is true for languages with phonological length, such as Japanese, where the duration of segments (vowels and consonants) alone can diferentiate meaning: e.g., kite ‘come (imp.)’ vs. kiite ‘listen (imp.)’ vs. kitte ‘postage stamp’. It is also true for languages without phonological length, such as English, where vowel duration is a robust cue to certain contrasts, including tense/lax pairs, such as [i] vs. [I], vowel height pairs, such as [E] vs. [ae], and fricative voicing, [s] vs. [z]. In these cases, temporal duration facilitates lexical recognition because it cues the identity of the phonological units that make up a word.

It is less clear whether temporal duration also cues higher-level linguistic units such as words. In speech production, word duration varies with several factors, including frequency, local predictability, and prosody. Whether these diferences in word duration facilitate word recognition, however, remains an open question. Existing perceptual evidence is mixed and depends on the task and the available context. In a cloze task, manipulations of duration had little efect on word recognition compared with manipulations of pitch and spectral properties (Bernhard and Tucker, 2015; Podlubny et al., 2018). Other work suggests that duration matters by shaping how listeners interpret the surrounding signal. For example, duration may afect the processing of reduced speech by modulating rhythmic expectations, specifically altering the speech rate of a context can make words appear or disappear (Dilley and Pitt, 2010). When context is absent, duration manipulations have been reported to

Direct cueing of word identity through word duration: evidence from Mandarin homophones degrade word recognition accuracy (Vitevitch, 2002). Evidence that listeners may use durational cues in word retrieval can also be found in speech misperception studies (Felty et al., 2013; Tang, 2015; Tang and Nevins, 2014). These studies have found a positive correlation between the lexical frequency of the target word and that of the perceived word, suggesting that durational information may influence which lexical candidates listeners retrieve. Taken together, these findings establish that duration can influence word recognition, but they leave open whether it does so only indirectly (e.g., by giving listeners more time to use context or to process other cues) or whether it can directly cue word identity.

Homophones ofer a way to address this question while controlling for segmental influences on word duration. Because homophones share the same segmental (and, in Mandarin, tonal) makeup, any systematic diference between them in the signal cannot be attributed to their segments. A growing body of corpus-based studies of spontaneous speech (on at least English, Mandarin, and German) suggests that such diferences may exist, because words carry word-specific phonetic properties. In English, Seyfarth (2014) found that word duration is potentially lexicalized and is related to a word’s informativity, that is, its average contextual predictability. Tang and Shaw (2021b) extended this finding to the three major prosodic cues (duration, intensity, and pitch) in Mandarin Chinese. Further evidence of word-specific duration and pitch realizations in English and Mandarin show that these phonetic values are partly predictable from the words’ meanings in utterance context (Chuang et al., 2023, 2026; Jin et al., 2026; Lu et al., 2025). Word-specific phonetics is not limited to single words: there is also evidence for the lexicalization of intonation (f0 contours) in English and German collocations (Calhoun and Schweitzer, 2013; Schweitzer et al., 2015). These findings

Direct cueing of word identity through word duration: evidence from Mandarin homophones suggest that homophones may difer phonetically as well. In English, higher-frequency (more predictable) words tend to be shorter, so homophones that difer in frequency are expected to difer in duration even when their segments are identical (Gahl, 2008; Lohmann, 2018b). In Mandarin Chinese, Tang and Shaw (2021a) found that the duration of homophones is related to their informativity, such that a word of a homophonic set tends to be shorter than its other homophonic member if the former has a lower informativity (on average less predictable). Beyond duration, Tseng et al. (2026) showed that English homophones also difer in other phonetic details and that these diferences are predictable from the words meanings in utterance context.

Whether listeners use such diferences to recognize words and to distinguish homophones is less well understood. Sanker (2019) found that listeners’ ability to distinguish homophones depended on the production context: listeners performed above chance when discriminating homophone pairs taken from sentences but not when the words were produced in isolation. This suggests that listeners can exploit acoustic details of homophones. Furthermore, English-speaking listeners were able to discriminate homophones when they difered in vowel duration, but not when they difered in mean F0 (Sanker, 2020). Sanker (2022) found that listeners can discriminate homophones at above-chance accuracy after prior exposure to the same test words, an efect that persisted across speakers. These studies highlight that listeners do not rely on just any acoustic detail.

Evidence that duration is informative for homophones can be also found in computational work. Schuppler et al. (2022) found that function words with homophonous realizations (ah, ach, auch, eine, and er can also be reduced to [a]) could be disambiguated better by

Direct cueing of word identity through word duration: evidence from Mandarin homophones a machine classifier when durational information was available. Similarly, Schuppler and Schrank (2018) studied the German forms der, die, das and their inflections des, dem, den, each of which can function as a determiner, relative pronoun, or demonstrative pronoun, and examined 193 acoustic features for automatic disambiguation. They found that durational features were particularly robust as nearly all of them survived feature selection. However, just because a cue is robust for machine learning does not necessarily mean that humans will be sensitive to it. A similar line of computational work on vowel distinctions in English shows that machine classification shows a higher sensitivity to duration cues than native speakers, unless the machine learning is explicitly directed towards spectral cues (Jurov et al., 2023). These findings leave a gap. That duration is useful to a machine classifier does not show that human listeners use it, and it is not clear whether the perceptual system is sensitive to word-level durational diferences of the size found in production. Mandarin provides a good test case, since it has abundant homophones, especially among disyllabic words.

In this paper we ask whether listeners can use word duration alone to identify homophones that difer in lexical frequency. Our hypothesis is that, if listeners have learned the association between frequency and duration, they should map the shorter token of a homophone pair to the higher-frequency member and the longer token to the lower-frequency member. We situate this hypothesis against three ways in which duration could facilitate word recognition: (i) by giving more time for sentential context to support prediction, i.e., “context integration”; (ii) by giving more time to process other cues to word identity, i.e., “spectral integration” and (iii) by directly cuing word identity, i.e., “direct cueing”.

Direct cueing of word identity through word duration: evidence from Mandarin homophones Our study is designed primarily to distinguish account (iii), “direct cueing”, from accounts (i) and (ii). We present our stimuli in a neutral context. Because of this, the “context inte gration” account predicts that listeners will not reliably associate the shorter/longer duration audio stimulus with the higher/lower frequency homophone. We also control our stimulus items to difer only in duration, holding other acoustic cues constant across homophone pairs. The “spectral integration” account predicts that, in the general case, associations between word duration and frequency will facilitate word recognition only when non-durational cues are also present. That is, if duration primarily facilitates word recognition by giving more time to process other cues, then masking cues will eliminate otherwise facilitatory efects of longer duration. In the case of Mandarin homophones, there may be other non-durational cues to word identity, such as pitch (Chuang et al., 2026). On the spectral integration account, duration could help to make subtle acoustic cues more salient to the listener. Our design rules out this possibility in this particular experiment. To focus on duration alone as a cue to word identity, we control all other cues diferentiating homophone pairs. Under these conditions, the “spectral integration” account predicts the same outcome as the “con text integration” account, listeners should not be able to associate word duration diferences with homophone frequency. Direct cueing, in contrast, predicts an efect of word duration on word recognition that persists even when no other cues are available. Notably, our design is narrowly focused on testing the direct cueing account while controlling for factors that would be relevant for “context integration” and “spectral integration” accounts, which may still be relevant in other contexts.

Direct cueing of word identity through word duration: evidence from Mandarin homophones Account (i) Context integration (ii) Spectral integration (iii) Direct cueing Prediction At chance At chance Above chance

TABLE I. Predictions of the three accounts for our experiments testing homophone identification based on word duration.

We report two experiments. In Experiment 1 (Section II), we manipulated the duration of Mandarin homophone pairs embedded in a carrier phrase and masked with noise, so that duration was the only cue distinguishing the members of each pair. In Experiment 2 (Section III), we removed the speech signal altogether, leaving only noise of the same duration, to test whether the efect persists in the absence of an intelligible signal. Of the three accounts discussed above, only direct cueing predicts above-chance behavior in these experiments.

## II. EXPERIMENT 1: SOME SIGNAL FOR HOMOPHONE PAIRS

## A. Methods

## 1. Design

In this experiment, we test whether participants are sensitive to duration as a cue for identifying homophones that difer in lexical frequency. More specifically, given a pair of homophones that difer in frequency, we ask whether they were able to associate the highfrequency word with shorter duration and its low-frequency counterpart with longer duration.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

To achieve this goal, we present participants with audio stimuli that contain homophone pairs that were masked with noise in carrier phrases. The only diference between the auditory signals for the target words in the pair is duration. In some conditions, the duration of the first target noise is shorter than the second (henceforth short-long). In others, the first is longer than the second (henceforth long-short). Participants were then asked to pick the order of the homophone they heard. They were presented with two response buttons that difer in the order of the homophone pair. One response button shows the high frequency word first, and then the low frequency one (henceforth high-low), while the other button shows the reverse order (i.e., low-high). For example, one button is 句子 (‘sentence’) / 锯子 (‘saw’), where the first word is more frequent than the second one. In this case, the other button would be 锯子 (‘saw’) / 句子 (‘sentence’), where the first word is less frequent than the second. An example of our experimental interface is shown in Figure 1.

If word duration alone can serve as a cue for homophone perception, then participants will be more likely to pick the word order whose frequency corresponds to the duration cue. Specifically, if they hear long-short, they should be more likely to pick the button with the frequency low-high, since low-frequency words tend to be longer than high-frequency ones. On the other hand, if the audio is short-long, they should be more likely to pick the button with the frequency high-low than low-high.

## 2. Materials

The homophone identification task consisted of 48 trials in total, with 16 target trials and 32 filler trials. All words were produced in the carrier phrase in (1). For target trials,

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## 你听到的单词顺序是哪一个？

## 锯子／句子

## 句子/锯子

FIG. 1. Experimental interface in Gorilla showing an example test item (homophone pairs). The instruction at the top says “Which order of words did you hear?”. The left button has a homophone pair of low-high order (frequency), and the right button shows the reverse order.

the words in the blanks are homophones. For filler trials, the words constitute phoneticallysimilar but non-homophonous pairs. For example, one word is 道路 (dào lù; ‘road’) and the other is 大路 (dà lù; ‘avenue’). The words in the pair also difer in frequency.

(1) diyige ci shi \_\_\_\_\_ dierge ci shi \_\_\_\_\_ xing ma?

first word be second word be correct sfp

The first word is \_, the second word is \_, is that correct?

a. Stimuli selection. Target and filler stimuli were selected from the Chinese Lexical Database (CLD) (Sun et al., 2018).

The target stimuli were identified using six criteria. First, each item belonged to a homophone set with exactly two members (i.e., a pair of homophones rather than a set of

Direct cueing of word identity through word duration: evidence from Mandarin homophones three or more homophones). Second, both items in the pair were disyllabic. This is because disyllabic words were the most frequent word length in Mandarin (Neergaard et al., 2022) as well as the word type that includes the most homophones. Third, both items (i.e., high and low) in the pair had the same part of speech. This controls for possible confounding efects for diferent parts of speech, such as noun-verb duration efects (Lohmann, 2018a, 2020; Lohmann and Conwell, 2020). Fourth, the dominant part of speech accounted for at least 90% of occurrences for each member of the pair. Fifth, the prevalence of both items was greater than 80%<sup>1</sup> to ensure familiarity to speakers, especially the low-frequency words (Keuleers et al., 2015). Sixth, the pair has a cosine similarity less than 0.5 to ensure that the environments in which the words occur are suficiently diferent, so that their meanings are suficiently diferent and not near-synonyms (Mikolov et al., 2013).

After applying the criteria listed above, we obtained 143 eligible homophone pairs. From these candidate pairs, we selected 16 target pairs using the Zipf scale of word frequency (Brysbaert et al., 2018) (with 1 being least frequent, and 7 being most frequent). The resulting set of target stimuli contained high-frequency words with Zipf values ranging from 3.8 to 4.8, as well as low-frequency words with Zipf values ranging from 2.0 to 3.0. The mean diference in Zipf frequency between the high- and low-frequency words was 1.58.

In terms of filler items, we applied criteria 2-6 from the target stimuli selection process. Additionally, we restricted to words that began with voiceless stops (either aspirated or unaspirated) that did not belong to any homophone set. For each target homophone pair, we selected two filler items that were minimal pairs with similar lexical variables.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

b. Stimuli recording and preprocessing. After selecting the target and filler stimulus items, the stimuli were recorded by a male native speaker of Mandarin Chinese in his 20’s at the time of recording. Six repetitions of each pair were recorded. We excluded the repetitions whose low-frequency member had a duration greater than 2.5 standard deviations from the mean duration across items. We then selected the token whose low-frequency member had the longest duration. Correspondingly, the high-frequency token of the pair was manipulated such that its duration was 25% shorter than that of its low-frequency counterpart. This was to approximate the expected efect of lexical frequency on word duration. Duration manipulation was performed using a custom Python script, which interfaced with Praat’s (Boersma and Weenink, 2021) Manipulation Object through the parselmouth library (Jadoul et al., 2018). Importantly, the duration of the high-frequency token was manipulated, while all other properties of the recorded signal was preserved.

After duration manipulation, we imposed speech-shaped noise on the test items in the sentence. The duration of the noise matched the manipulated duration of the words. In other words, since high frequency words were shorter than that for low frequency words in the audio, the noise was also correspondingly shorter. There were two diferent noise levels: low (0 dB) and medium (-8 dB). Under the low noise condition, the words were expected to remain readily identifiable. For the medium-noise condition, the identification was expected to be more dificult.

In the current experiment, the homophone pairs in the target trials were masked with the low noise level (0 dB). Since the homophone pairs were segmentally identical and difered

Direct cueing of word identity through word duration: evidence from Mandarin homophones only in their manipulated durations, this design left duration as the only available cue for distinguishing them.

## 3. Procedure

200 native Mandarin speakers participated in the experiment. Half were recruited on Prolific, and the remaining half were recruited on the Chinese social media platform Xiaohongshu. All participants were directed to the experiment-building platform Gorilla to complete the experiment. There were three separate tasks in the experiment: (i) homophone identification, (ii) LexTale, and (iii) a demographic questionnaire.

First, homophone identification is the main task. On each trial, participants were first briefly (for 500 ms) presented with the two response options before being presented with the auditory stimulus. This was intended to encourage the participants to attend to both response alternatives before hearing the stimulus. We adopted this priming procedure following Sanker (2019), who found that simultaneous presentation of the response options and auditory stimulus increased a positional bias that facilitates the choice of the first ortho graphic form that is compatible with the speech signal. An example test trial was shown in Figure 1.

After the preview, the auditory stimulus was played while both response options remained visible on the screen. The order of the response options were randomized across trials (i.e., the left button could have high-low first or low-high first). Participants could not click on the buttons to advance to the next trial until the audio had finished playing and the

Direct cueing of word identity through word duration: evidence from Mandarin homophones question appeared on the screen. The audio was only played once with no replay option. The inter-trial interval was set to 500 ms.

(2) ni tingdao de danci shunxu shi na yige?

you hear de word order be which one?

Which order of words did you hear?

Participants received two practice trials, one from the low noise level and one from the medium noise level, and the words in the practice trials were minimal pairs of each other. As described in the previous section, there were 48 experimental trials (16 tests and 32 fillers), and the homophones in the target trials were masked with the low noise level. The words in the filler trials were randomly assigned to either the low or medium noise level. Given the dificulty of the test trials, we wanted the easier filler trials to serve as encouragement through the task. Each of the 32 filler trials had eight possible conditions—for each condition below, there was both a low and medium noise level option. The condition chosen for each trial was randomized across participants.

In the second task, participants completed the LexTALE task as a measure of their lexical knowledge. Participants were asked whether or not the character on the screen was a real simplified Mandarin character using a web-compatible version of LexTALE\_CH (Chan and Chang, 2018) by Tang (2024). The LexTALE task consisted of 60 lexical items (i.e., real Chinese characters) and 30 nonce items (i.e., non-words). Participants were visually presented with 90 items total.

Direct cueing of word identity through word duration: evidence from Mandarin homophones In the last part of the experiment, participants completed a demographics questionnaire that collected information about age, gender, and language background.

## B. Analysis

The dependent variable was participants’ choice of homophone order, coded as either high-low (i.e., choosing the order where the high-frequency item is followed by the lowfrequency item) or low-high (the low-frequency item followed by the high-frequency item). We fitted the binary responses using logistic mixed-efects regression implemented with the glmer function in the lme4 package (Bates et al., 2015) in R (R Core Team, 2023, version 4.3.1). In the regression model, low-high was treated as the reference level.

The main fixed-efect predictor of interest was the duration pattern of the auditory stimulus (long-short vs. short-long). The prediction is that if the auditory stimulus was short-long, participants should be more likely to pick the response of high-low. On the other hand, if the auditory stimulus was long-short, participants should be more likely to pick the response of low-high.

In the analysis, the variable auditory stimulus type was sum-coded, with long-short coded <sup>as</sup> −<sup>0.5</sup> <sup>and</sup> <sup>short-long</sup> <sup>as</sup> <sup>0.5.</sup> <sup>Trial</sup> <sup>number</sup> <sup>(i.e.,</sup> <sup>the</sup> <sup>order</sup> <sup>of</sup> <sup>the</sup> <sup>trial</sup> <sup>in</sup> <sup>the</sup> <sup>presentation</sup> sequence) was standardized and included as a fixed-efect predictor to account for changes in responses over the course of the experiment. The model also included random intercepts for participant and target item. Only target trials from participants who passed the lexical proficiency screening were included in the analysis.

Direct cueing of word identity through word duration: evidence from Mandarin homophones More explicitly, the structure of the logistic mixed-efects model is as follows: Frequency.Chosen ∼ <sup>Auditory.Stimuli</sup> <sup>+</sup> <sup>Trial\_z</sup> <sup>+</sup> <sup>(1</sup> <sup>|</sup> <sup>Participant.ID)</sup> <sup>+</sup> <sup>(1</sup> <sup>|</sup> <sup>Trial.ID).</sup>

## C. Results

We first analyzed the performance of participants on the supplementary LexTALE task in order to exclude participants whose Mandarin proficiency was insuficient for the purposes of our experiment. 164 out of 200 participants passed the LexTale task using a cutof of 0.73 in raw accuracy and 28 in corrected accuracy.

For the remaining participants, we analyzed their accuracy in the filler items. First, the filler accuracy in the low noise condition is higher than the high noise condition, which suggests that the noise manipulation afected participants’ perception of non-homophone filler pairs. This diference in noise conditions is confirmed by a logistic mixed-efect model predicting correctness on the filler trial with noise level and trial order as fixed-efect predictors as well as participant and item as random intercepts $( \beta = - 2 . 0 1 , S E = 0 . 1 1 , z = - 1 7 . 5 3 .$ $p < . 0 0 1 )$ . Importantly, participants recruited from both Prolific and Xiaohongshu achieved comparable filler accuracy across noise conditions. In the low-noise condition, accuracy was high for both Prolific (96.7%) and Xiaohongshu (95.3%) participants. In the high-noise condition, accuracy decreased for both groups as expected, reaching 82.7% for Prolific participants and 75.2% for Xiaohongshu participants. This suggests that participants recruited through both sources performed similarly on the filler trials, providing no evidence of a systematic diference in participant quality between the two samples.

Direct cueing of word identity through word duration: evidence from Mandarin homophones We next examined whether the duration can serve as a cue for homophone perception. We fitted the logistic regression model described in the previous section, with auditory stimulus type and standardized trial number as fixed-efect predictors and random intercepts for item and participant.<sup>2</sup> Results are presented in Table II.

<table><tr><td>Parameter Log-Odds SE 95% CI z</td></tr><tr><td></td></tr><tr><td>Intercept 0.11 0.04 [0.03, 0.19] 2.81 0.005</td></tr><tr><td>Auditory Stimuli [Short-Long] 0.24 0.08 [0.08, 0.39] 3.01 0.003</td></tr><tr><td>Trial z 0.01 0.04 [-0.07, 0.09] 0.26 0.793</td></tr></table>

TABLE II. Results of the logistic mixed-efects regression model predicting homophone-order choice as a function of auditory stimulus type and standardized trial order in Experiment 1.

Crucially, auditory stimulus type is a significant main efect predictor of word order selection $( \beta = 0 . 2 4 , S E = 0 . 0 8 , z = 3 . 0 1 , p = . 0 0 3 )$ . This suggests that participants were more likely to select the high-low ordering when the duration of the first homophone item is shorter than the second. This pattern is consistent with the predicted mapping between frequency and duration. Additionally, trial order did not significantly predict response choice $\left( p = . 7 9 3 \right)$ , which suggests that the response patterns did not change throughout the course of the experiment.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## III. EXPERIMENT 2: NO SIGNAL FOR HOMOPHONE PAIRS

In Experiment 1, we found that participants’ choice of homophone order was influenced by the duration pattern of the auditory stimulus: participants were more likely to choose the high-low ordering when the auditory stimuli were short-long. This suggests that listeners were able to use duration as a cue for identifying homophones that difer in lexical frequency. This is especially striking given that both items in the pair are segmentally identical and only difer in (manipulated) duration. In Experiment 2, we took this one step further and tested whether the efect would persist in the absence of an intelligible speech signal. Specifically, the items in the homophone pair for target trials were presented under the highest noise condition. In this condition, the speech signal was unintelligible, and listeners could only perceive the noise with an intensity level of 65dB with no signal present. The diference in duration between the high-frequency item and the low-frequency item in the pair remains. This allowed us to test in a more strict condition whether duration alone could serve as a cue for identifying homophones that difer in frequency.

## A. Methods

Experiment 2 is the same as Experiment 1 except for the following two modifications. First, as described above, the speech-shaped noise imposed on the target items was of the highest noise condition (noise only with an intensity value of 65dB). Second, we used the platform Prolific to recruit an additional 200 participants. Participants were recruited in two rounds (50 in the first, 150 in the second).

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## B. Results

Out of the 200 participants who participated in Experiment 2, 174 participants passed the LexTale task. In terms of fillers, they achieved an accuracy of 0.78 for high noise conditions and an accuracy of 0.95 for low noise conditions. This is comparable with the filler accuracies reported in Experiment 1.

We used the same mixed-efect model structure as Experiment 1. Regression results were shown in Table III. Auditory stimulus type remains as a significant predictor of the choice of homophone order $( \beta = 0 . 2 0 , \ S E = 0 . 0 8 , \ z = 2 . 6 0 , \ p = 0 . 0 0 9 )$ . As in Experiment 1, participants were more likely to select the high-low ordering when the auditory stimulus was short-long. Trial order also did not surface as a significant main efect $\left( p = . 8 0 5 \right)$
<table><tr><td>Parameter</td><td>Log-Odds SE</td><td>95% CI z</td></tr><tr><td>Intercept</td><td>0.01 0.05 [−0.08, 0.10] 0.25 0.804</td><td></td></tr><tr><td>Auditory Stimuli [Short-Long]</td><td>0.20 0.08 [0.05, 0.35] 2.60 0.009</td><td></td></tr><tr><td>Trial z</td><td>0.00946 0.04 [-0.07, 0.08] 0.25 0.805</td><td></td></tr></table>

TABLE III. Results of the logistic mixed-efects regression model predicting homophone-order choice as a function of auditory stimulus type and standardized trial order in Experiment 2.

To test whether noise level on the target items modulated listeners’ use of duration as a cue to homophone identity, we fit a logistic mixed-efects model that includes noise level as a fixed-efect predictor in addition to the existing model structure. Crucially, noise level

Direct cueing of word identity through word duration: evidence from Mandarin homophones was allowed to interact with auditory stimulus type in order to test whether the efect of the short-long versus long-short duration pattern difered across noise conditions. The structure of the interactive model is as follows: Frequency.Chosen  Auditory.Stimuli \* ExpVersion + Trial\_z + (1 | Participant.ID) + (1 | Trial.ID).

The regression results are shown in Table IV. The interaction between auditory stimulus type and noise level was not significant (p = .700), indicating that the efect of duration patterns on homophone-order choice did not reliably difer across noise conditions. Together with the significant efect of auditory stimulus type, this pattern suggests that the duration efect was detectable even in the absence of any will signal.

<table><tr><td>Parameter</td><td>Log-Odds SE</td><td>95% CI z</td></tr><tr><td>Intercept</td><td></td><td>0.06 0.03 [0.00, 0.12] 2.05 0.040</td></tr><tr><td>Auditory Stimuli [Short-Long]</td><td></td><td>0.22 0.05 [0.11, 0.32] 3.94 &lt; .001</td></tr><tr><td>ExpVersionBinary [easyNoise]</td><td></td><td>0.10 0.06 [-0.01, 0.21] 1.82 0.069</td></tr><tr><td>Trial z</td><td></td><td>0.010 0.03 [-0.04, 0.06] 0.36 0.715</td></tr><tr><td>Auditory Stimuli [Short-Long] × ExpVersionBinary [easyNoise]</td><td>0.04 0.11 [-0.17, 0.26] 0.39 0.700</td><td></td></tr></table>

TABLE IV. Results of the logistic mixed-efects regression model predicting homophone-order choice from auditory stimulus type, experiment, their interaction, and standardized trial order. <sup>Categorical</sup> <sup>predictors</sup> <sup>were</sup> <sup>coded</sup> <sup>using</sup> −<sup>0.5</sup> <sup>and</sup> <sup>+0.5</sup> <sup>contrasts.</sup>

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## IV. DISCUSSION

The efect of word predictability on word duration is robust across languages, such as English (Aylett and Turk, 2004; Bell et al., 2003), Dutch (Pluymaekers et al., 2005), Japanese (Hashimoto, 2021), Mandarin Chinese (Tang and Bennett, 2018) and Kaqchikel (Tang and Bennett, 2018). Words that are more predictable tend to be shorter in duration. Less is known about the relevance of such duration diferences for speech perception.

Generally speaking, increasing the duration of an interval could facilitate word recognition for a couple of diferent reasons, which we described in the introduction and summarize in TABLE I along with predictions. Word duration could, as past work has suggested, facilitate word recognition by allowing more time for sentential context to facilitate prediction (Podlubny et al., 2018). We refer to this account as “context integration”. A second way is by providing more time to process other (non-durational) cues to word identity, which we refer to as “spectral integration”. Finally, a third way that duration could influence perception is by providing a direct cue to word identity, which we refer to as “direct cueing”. Our study was designed primarily to test this last possibility.

Results indicate that word duration facilitates word recognition and, moreover, that differences in word duration are associated with word frequency. Listeners were able to use word duration to distinguish between homophone pairs of difering frequency, mapping the higher frequency member of the pair to the stimulus of shorter duration.

In our experiments, listeners had no context to facilitate word recognition. They were forced to rely on duration as a cue, either in tandem with degraded spectral cues (Experiment

Direct cueing of word identity through word duration: evidence from Mandarin homophones 1) or with no spectral cues at all (Experiment 2). In both experiments, participants were able to use duration as a cue, identifying the word order with greater than chance accuracy. This indicates that there is more to word duration efects in perception than just allowing more time to process other cues (spectral integration) or for sentential context to facilitate prediction (context integration). The results support direct cueing.

The other two accounts we considered, context integration and spectral integration, predict at-chance behavior in both of our experiments. This is because, in our design, we deprived participants of both contextual cues and other (non-durational) phonetic cues to word identity. Since duration was the only available cue to word identity, the spectral integration account predicts that more time to process other cues would be fruitless, leading to at-chance behavior. This should be the same whether there are some uninformative spectral cues (Experiment 1) or none at all (Experiment 2). The same is true for the context integration account. The neutral context of target word presentation meant that additional time to integrate the sentential context would not lead to increased performance. Thus, for our experiments, both alternative accounts predict at-chance performance.

Since we found a similar level of above chance performance in both experiments, our design provides positive evidence that word duration can directly cue lexical items. Duration has a role in lexical access above and beyond providing more time for processing. That said, our results do not rule out these other possibilities. In naturalistic contexts, increased duration may, in addition to directly cuing lexical items, provide additional time to process other cues, both contextual and spectral, to word identity.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

The totality of our results are most consistent with the direct cueing account– word duration directly cues word identity. Participants were able to distinguish homophone pairs, even when no other cues were present.

Our results have implications for how lexical items are represented. Word duration is a type of phonetic detail that is not represented in strings of segments, a form of lexical representation in classic models of spoken word recognition, e.g., Cohort model (Marslen-Wilson and Welsh, 1978), ShortList (Norris, 1994; Norris and McQueen, 2008), Trace (McClelland and Elman, 1986). There is a consensus that the facts of speech perception necessitate a hybrid model of speech representation, combining both abstract categories and phonetic detail (Baese-Berk et al., 2022; Cutler, 2010; Goldinger, 2007; Hawkins, 2003; McMurray, 2022).The precise mechanism for encoding phonetic detail, e.g., exemplars (Pierrehumbert, 2002), memory bufers (Muegge et al., 2026), dynamic neural fields (Kirov and Gafos, 2007; Shaw and Tang, 2022), neural networks (Beguš, 2021; Beguš et al., 2023; Magnuson et al., 2020; Scharenborg, 2010), and the precise dimensions of phonetic detail that are encoded remain frontier issues in speech perception. The current work contributes to the latter issue. Evidence that the phonetic duration of words can directly cue their identity in speech suggests that this aspect of phonetic detail is encoded in the lexicon.

## V. CONCLUSION

This study asked whether listeners can use word duration alone to identify homophones that difer in lexical frequency in Mandarin Chinese, a language known for having a high number of homophones. In two experiments, Mandarin listeners heard homophone pairs in

Direct cueing of word identity through word duration: evidence from Mandarin homophones a neutral carrier phrase, with duration as the only diference between the two words. When the first word was shorter than the second, listeners were more likely to choose the response order in which the higher-frequency homophone came first. The efect held when the words were partially masked by noise (Experiment 1) and when no speech signal was present at all, only noise of matching duration (Experiment 2). Moreover, the magnitude of the efect did not difer reliably between the two noise conditions.

Because the design removed sentential context and held all non-durational cues constant, the results are dificult to explain through context integration or spectral integration alone. They are most consistent with a direct-cueing account, in which word duration is itself associated with word identity. This implies that listeners’lexical representations encode word-level temporal detail, and that this detail is available for lexical access. The finding supports hybrid models of speech perception that combine abstract phonological categories with fine phonetic detail, and it identifies word duration as one dimension of that detail.

More broadly, these findings suggest that the temporal properties of words are part of what listeners know about them. If duration can cue word identity when no other information is available, then models of spoken word recognition need a way to represent word-level timing alongside segmental and tonal form, and the frequency-duration relationship documented in production appears to be something listeners can draw on in perception. This extends the case for phonetic detail in lexical representation beyond segment-level cues to the duration of whole words, and it raises the possibility that other word-specific properties documented in corpus work (Tang and Shaw, 2021b), such as pitch and intensity, are similarly available to listeners. The efects here were modest, and the stimuli came from a single speaker, 16

Direct cueing of word identity through word duration: evidence from Mandarin homophones item pairs, and one fixed duration ratio, so the generality of the mechanism remains to be established. Future work should test graded duration diferences, multiple speakers, and other languages, and should ask how direct cueing by duration combines with sentential context and spectral cues in natural speech, where, as the three accounts considered here are not mutually exclusive, all of these sources of information are likely to contribute to word recognition.

## ACKNOWLEDGMENTS

This work was partly conducted as part of the Bachelor of Science thesis of Megan Niu. We thank Zhenrui Zhang for assistance with data collection in China for Experiment 1. All errors remain ours.

AUTHOR DECLARATIONS

Conflict of Interest

There are no conflicts to disclose.

Ethics Approval

The data collection was in accordance with Yale University’s IRB No. 2000036833.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

## DATA AVAILABILITY

The datasets generated during the current study are available in this published article and its supplementary information files; and in the Zenodo repository (the repository is accessible via a hyperlink; it will be made public upon publication).

## AUTHORS’ CONTRIBUTIONS

Corresponding author: KT. We follow the CRediT taxonomy (https://credit.niso. org/). Conceptualization: XZ, MN, JS, KT; Methodology: XZ, MN, JS, KT; Investigation: XZ, MN; Data curation: XZ, MN; Formal analysis: XZ, JS, KT; Software: XZ, MN; Validation: XZ, JS, KT; Visualization: XZ; Writing –original draft: XZ, JS, KT; Writing –review & editing: XZ, JS, KT; Supervision: JS, KT; Project administration: JS, KT; Funding acquisition: JS; Resources: JS.

## APPENDIX A: STIMULUS SELECTION

From these candidate pairs, we selected 16 target pairs using the Zipf scale of word frequency. The frequency ranges from 1 (lowest frequency) to 7 (highest frequency). Specifically, we conducted an exhaustive search and computed the following measures for each pair:

1. Minimization of the range between the upper and lower Zipf scores for our highfrequency words.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

2. Minimization of the range between the upper and lower Zipf scores for our lowfrequency words.

3. Maximization of the mean frequency diference across all items (the mean frequency of all high-frequency words minus the mean frequency of all low-frequency words).

4. Maximization of the minimum frequency diference (the minimum frequency of all high-frequency words minus the maximum frequency of all low-frequency words).

Direct cueing of word identity through word duration: evidence from Mandarin homophones v067.i01.

Beguš, G. (2021). “CiwGAN and fiwGAN: Encoding information in acoustic data to model lexical learning with Generative Adversarial Networks,” Neural Networks 139, 305–325, doi: mu10.1016/j.neunet.2021.03.017.

Beguš, G., Zhou, A., and Zhao, T. C. (2023). “Encoding of speech in convolutional layers and the brain stem based on language experience,” Scientific Reports 13(1), 6480, doi: mu10.1038/s41598-023-33384-9.

Bell, A., Jurafsky, D., Fosler-Lussier, E., Girand, C., Gregory, M., and Gildea, D. (2003). “Efects of disfluencies, predictability, and utterance position on word form variation in English conversation,” The Journal of the Acoustical Society of America 113(2), 1001– 1024, muhttps://doi.org/10.1121/1.1534836, doi: mu10.1121/1.1534836.

Bernhard, D., and Tucker, B. V. (2015). “The efects of duration on human processing of reduced speech,” Canadian Acoustics 43(3), 120–121, muhttps://jcaa.caa-aca.ca/ index.php/jcaa/article/view/2784.

Boersma, P., and Weenink, D. (2021). “Praat: doing phonetics by computer [Computer program]” Version 6.1.38, retrieved 2 January 2021 muhttp://www.praat.org/.

Brysbaert, M., Mandera, P., and Keuleers, E. (2018). “The word frequency efect in word processing: An updated review,” Current Directions in Psychological Science 27(1), 45–50, muhttps://doi.org/10.1177/0963721417727521, doi: mu10.1177/0963721417727521.

Calhoun, S., and Schweitzer, A. (2013). Can intonation contours be lexicalised? Implications for discourse meanings, 271–328 (De Gruyter Mouton, Berlin, Boston), muhttps://doi. org/10.1515/9783110261790.271, doi: mu10.1515/9783110261790.271.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

Chan, I. L., and Chang, C. (2018). “Lextale\_ch: A quick, character-based proficiency test for mandarin chinese,” in Proceedings of the 42nd Annual Boston University Conference on Language Development, pp. 114–130.

Chuang, Y.-Y., Baayen, R. H., and Bell, M. J. (2023). “Do words sing their own tunes? Word-specific pitch realizations in Mandarin and English,” in Proceedings of the 20th International Congress of Phonetic Sciences, Prague 2023, edited by R. Skarnitzl and J. Volín (Guarant International), pp. 1603–1607.

Chuang, Y.-Y., Bell, M. J., Tseng, Y.-H., and Baayen, R. H. (2026). “Word-specific tonal realizations in Mandarin,” Language 102(1), 1–45, doi: mu10.1017/S0097850725000001. Cutler, A. (2010). “Abstraction-based eficiency in the lexicon,” Laboratory Phonology 1(2), 301–318, doi: mu10.1515/labphon.2010.016.

Dilley, L. C., and Pitt, M. A. (2010). “Altering context speech rate can cause words to appear or disappear,” Psychological Science 21(11), 1664–1670, doi: mu10.1177/ 0956797610384743.

Felty, R. A., Buchwald, A., Gruenenfelder, T. M., and Pisoni, D. B. (2013). “Misperceptions of spoken words: Data from a random sample of american english words,” The Journal of the Acoustical Society of America 134(1), 572–585, doi: mu10.1121/1.4809540.

Gahl, S. (2008). “Time and Thyme are not homophones: The efect of lemma frequency on word durations in spontaneous speech,” Language 84(3), 474–496, muhttps://dx.doi. org/10.1353/lan.0.0035, doi: mu10.1353/lan.0.0035.

Goldinger, S. (2007). “A complementary-systems approach to abstract and episodic speech perception,” in Proceedings of the 16th International Congress of Phonetic Sciences, pp.

Direct cueing of word identity through word duration: evidence from Mandarin homophones 49–54.

Hashimoto, D. (2021). “Probabilistic reduction and mental accumulation in Japanese: Frequency, contextual predictability, and average predictability,” Journal of Phonetics 87, 101061, doi: mu10.1016/j.wocn.2021.101061.

Hawkins, S. (2003). “Roles and representations of systematic fine phonetic detail in speech understanding,” Journal of Phonetics 31(3-4), 373–405, doi: mu10.1016/j.wocn.2003. 09.006.

Jadoul, Y., Thompson, B., and de Boer, B. (2018). “Introducing parselmouth: A python interface to praat,” Journal of Phonetics 71, 1–15, muhttps://www.sciencedirect.com/ science/article/pii/S0095447017301389, doi: mu/10.1016/j.wocn.2018.07.001.

Jin, X., Ernestus, M., and Baayen, R. H. (2026). “Using embeddings to predict spoken word duration and pitch in Mandarin monosyllabic words” muhttps://arxiv.org/abs/2607. 02002.

Jurov, N., Wolf, G., Idsardi, W., and Feldman, N. H. (2023). “Speech features are weighted by selective attention,” in Proceedings of the Conference on Cognitive Computational Neuroscience.

Keuleers, E., Stevens, M., Mandera, P., and Brysbaert, M. (2015). “Word knowledge in the crowd: Measuring vocabulary size and word prevalence in a massive online experiment,” Quarterly Journal of Experimental Psychology 68(8), 1665–1692, muhttps://doi. org/10.1080/17470218.2015.1022560, doi: mu10.1080/17470218.2015.1022560 pMID: 25715025.

Direct cueing of word identity through word duration: evidence from Mandarin homophones Kirov, C., and Gafos, A. (2007). “Dynamic phonetic detail in lexical representations,” in Proceedings of the 16th International Congress of Phonetic Sciences, University of Saarbrücken, pp. 637–640.

Lohmann, A. (2018a). “Cut (n) and cut (v) are not homophones: Lemma frequency afects the duration of noun–verb conversion pairs,” Journal of Linguistics 54(4), 753–777, doi: mu10.1017/S0022226717000378.

Lohmann, A. (2018b). “Time and thyme are not homophones: A closer look at Gahl’s work on the lemma-frequency efect, including a reanalysis,” Language 94(2), e180–e190, doi: mu10.1353/lan.2018.0032.

Lohmann, A. (2020). “Nouns and verbs in the speech signal: Are there phonetic correlates of grammatical category?,” Linguistics 58(6), 1877–1911, muhttps://doi.org/10.1515/ ling-2020-0249, doi: mu10.1515/ling-2020-0249.

Lohmann, A., and Conwell, E. (2020). “Phonetic efects of grammatical category: How category-specific prosodic phrasing and lexical frequency impact the duration of nouns and verbs,” Journal of Phonetics 78, 100939, muhttps://www.sciencedirect.com/science/ article/pii/S0095447018300627, doi: mu10.1016/j.wocn.2019.100939.

Lu, Y., Chuang, Y.-Y., and Baayen, R. H. (2025). “The realization of tones in spontaneous spoken Taiwan Mandarin: a corpus-based survey and theory-driven computational modeling” muhttps://arxiv.org/abs/2503.23163.

Magnuson, J. S., You, H., Luthra, S., Li, M., Nam, H., Escabi, M., Brown, K., Allopenna, P. D., Theodore, R. M., Monto, N. et al. (2020). “EARSHOT: A minimal neural network model of incremental human speech recognition,” Cognitive Science 44(4), e12823, doi:

Direct cueing of word identity through word duration: evidence from Mandarin homophones mu10.1111/cogs.12823.

Marslen-Wilson, W. D., and Welsh, A. (1978). “Processing interactions and lexical access during word recognition in continuous speech,” Cognitive Psychology 10(1), 29–63, muhttps://doi.org/10.1016/0010-0285(78)90018-X, doi: mu10.1016/0010-0285(78) 90018-X.

McClelland, J. L., and Elman, J. L. (1986). “The TRACE model of speech perception,” Cognitive Psychology 18(1), 1–86, muhttps://doi.org/10.1016/0010-0285(86)90015-0, doi: mu10.1016/0010-0285(86)90015-0.

McMurray, B. (2022). “The myth of categorical perception,” The Journal of the Acoustical Society of America 152(6), 3819–3842, doi: mu10.1121/10.0016614.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. (2013). “Distributed representations of words and phrases and their compositionality” arXiv preprint, muhttps: //arxiv.org/abs/1310.4546, arXiv:1310.4546.

Muegge, J. B., Kim, H., and McMurray, B. (2026). “Decoupling speech processing from time,” Trends in Cognitive Sciences 30(1), 66–79, doi: mu10.1016/j.tics.2025.05.017.

Neergaard, K. D., Xu, H., German, J. S., and Huang, C.-R. (2022). “Database of word-level statistics for Mandarin Chinese (DoWLS-MAN),” Behavior Research Methods 54(2), 987–1009, muhttps://doi.org/10.3758/s13428-021-01620-7, doi: mu10.3758/ s13428-021-01620-7.

Norris, D. (1994). “Shortlist: A connectionist model of continuous speech recognition,” Cognition 52(3), 189–234, doi: mu10.1016/0010-0277(94)90043-4.

Direct cueing of word identity through word duration: evidence from Mandarin homophones

Norris, D., and McQueen, J. M. (2008). “Shortlist B: A Bayesian model of continuous speech recognition,” Psychological Review 115(2), 357–395, muhttps://doi.org/10. 1037/0033-295X.115.2.357, doi: mu10.1037/0033-295X.115.2.357.

Pierrehumbert, J. (2002). “Word-specific phonetics,” in Laboratory Phonology 7, edited by C. Gussenhoven and N. Warner (Mouton de Gruyter), pp. 101–140, muhttps://doi.org/ 10.1515/9783110197105.1.101, doi: mu10.1515/9783110197105.1.101.

Pluymaekers, M., Ernestus, M., and Baayen, R. (2005). “Articulatory planning is continuous and sensitive to informational redundancy,” Phonetica 62(2-4), 146–159, muhttps://doi. org/10.1159/000090095, doi: mu10.1159/000090095.

Podlubny, R. G., Nearey, T. M., Kondrak, G., and Tucker, B. V. (2018). “Assessing the importance of several acoustic properties to the perception of spontaneous speech,” The Journal of the Acoustical Society of America 143(4), 2255–2268, muhttps://doi.org/10. 1121/1.5031123, doi: mu10.1121/1.5031123.

R Core Team (2023). R: A Language and Environment for Statistical Computing, R Foundation for Statistical Computing, Vienna, Austria, muhttps://www.R-project.org/.

Sanker, C. (2019). “Efects of lexical ambiguity, frequency, and acoustic details in auditory perception,” Attention, Perception, & Psychophysics 81, 323–343, muhttps://doi.org/ 10.3758/s13414-018-1604-x, doi: mu10.3758/s13414-018-1604-x.

Sanker, C. (2020). “Lexical ambiguity and acoustic distance in discrimination,” Proceedings of the Linguistic Society of America 5(1), 431–445, muhttps://journals. linguisticsociety.org/proceedings/index.php/PLSA/article/view/4719, doi: mu10.3765/plsa.v5i1.4719.

Sanker, C. (2022). “Homophone discrimination based on prior exposure,” Journal of Phonetics 95, 101182, doi: mu10.1016/j.wocn.2022.101182.

Scharenborg, O. (2010). “Modeling the use of durational information in human spokenword recognition,” The Journal of the Acoustical Society of America 127(6), 3758–3770, doi: mu10.1121/1.3377050.

Schuppler, B., Berger, E., Kogler, X., and Pernkopf, F. (2022). “Homophone disambiguation profits from durational information,” in Interspeech 2022, pp. 3198–3202, doi: mu10.21437/ Interspeech.2022-10109.

Schuppler, B., and Schrank, T. (2018). “On the use of acoustic features for automatic disambiguation of homophones in spontaneous German,” Computer Speech & Language 52, 209–224, muhttps://www.sciencedirect.com/science/article/pii/ S0885230817300232, doi: mu10.1016/j.csl.2017.12.011.

Schweitzer, K., Walsh, M., Calhoun, S., Schütze, H., Möbius, B., Schweitzer, A., and Dogil, G. (2015). “Exploring the relationship between intonation and the lexicon: Evidence for lexicalised storage of intonation,” Speech Communication 66, 65–81, muhttps:// www.sciencedirect.com/science/article/pii/S0167639314000727, doi: mu10.1016/ j.specom.2014.09.006.

Seyfarth, S. (2014). “Word informativity influences acoustic duration: Efects of contextual predictability on lexical representation,” Cognition 133(1), 140–155, muhttps://doi.org/ 10.1016/j.cognition.2014.06.013, doi: mu10.1016/j.cognition.2014.06.013.

Shaw, J., and Tang, K. (2022). “A dynamic neural field model of leaky prosody: proof of concept,” in Proceedings of the Annual Meetings on Phonology, edited by N. Elkins,

Direct cueing of word identity through word duration: evidence from Mandarin homophones B. Hayes, J. Jo, and J.-L. Siah, Linguistic Society of America, doi: mu10.3765/amp.v10i0. 5442.

Sun, C. C., Hendrix, P., Ma, J., and Baayen, R. H. (2018). “Chinese lexical database (CLD): A large-scale lexical database for simplified Mandarin Chinese,” Behavior Research Methods 50, 2606–2629, muhttps://doi.org/10.3758/s13428-018-1038-3, doi: mu10. 3758/s13428-018-1038-3.

Tang, K. (2015). “Naturalistic speech misperception,” Ph.D. thesis, University College London.

Tang, K. (2024). “Upscaled image files for LEXTALE\_CH” muhttps://doi.org/10. 17605/OSF.IO/Z2V35, doi: mu10.17605/OSF.IO/Z2V35, derived from Chan, I. L., & Chang, C. B. (2023), Image files for LEXTALE\_CH.

Tang, K., and Bennett, R. (2018). “Contextual predictability influences word and morpheme duration in a morphologically complex language (Kaqchikel Mayan),” The Journal of the Acoustical Society of America 144(2), 997–1017, doi: mu10.1121/1.5046095.

Tang, K., and Nevins, A. (2014). “Measuring segmental and lexical trends in a corpus of naturalistic speech,” in Proceedings of the 43rd Meeting of the North East Linguistic Society, edited by H.-L. Huang, E. Poole, and A. Rysling, GLSA (Graduate Linguistics Student Association), Vol. 2, pp. 153–166.

Tang, K., and Shaw, J. A. (2021a). “Non-homophonic homophones in Mandarin Chinese: evidence for phonetic detail in the lexicon,” 95th Annual Meeting of Linguistic Society of America 2021, San Francisco, USA.

Direct cueing of word identity through word duration: evidence from Mandarin homophones Tang, K., and Shaw, J. A. (2021b). “Prosody leaks into the memories of words,” Cogni tion 210, 104601, muhttps://doi.org/10.1016/j.cognition.2021.104601, doi: mu10. 1016/j.cognition.2021.104601.

Tseng, Y.-H., Ernestus, M. T. C., ten Bosch, L. F. M., and Baayen, R. H. (2026). “Letters hide the truth from our eyes: English homophones have meaningfully diferent phonetic realizations” muhttps://arxiv.org/abs/2608.26749.

Vitevitch, M. S. (2002). “Naturalistic and experimental analyses of word frequency and neighborhood density efects in slips of the ear,” Language and Speech 45(4), 407–434, doi: mu10.1177/00238309020450040501.