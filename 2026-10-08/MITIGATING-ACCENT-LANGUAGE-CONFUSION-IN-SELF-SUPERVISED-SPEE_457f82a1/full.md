# MITIGATING ACCENT–LANGUAGE CONFUSION IN SELF-SUPERVISED SPEECH REPRESENTATIONS FOR LANGUAGE IDENTIFICATION

Minu Kim <sup>1</sup>, Jihwan Lee <sup>1</sup>, David R. Mortensen <sup>2</sup>, Shrikanth Narayanan <sup>1</sup>

<sup>1</sup>Signal Analysis and Interpretation Lab (SAIL), University of Southern California, USA <sup>2</sup>Language Technologies Institute, Carnegie Mellon University, USA

minukim@usc.edu

## ABSTRACT

Spoken language identification (LID) aims to recognize the target language regardless of accent. In practice, however, LID models fine-tuned from self-supervised speech representations frequently confuse accents with languages, misclassifying non-native (L2) speech as the speaker’s first language (L1). We show that non-native speech representations lie between native target-language and native L1 poles, causing systematic misclassification. To address this, we introduce a geometric projection that estimates an L1-bias direction solely from native speech and removes it before the frozen LID head. Across five MMS-LID models and non-native corpora, this projection substantially improves target language identification for L2-accented speech while preserving predictions for native speech. These results show that accentinduced L1 bias can be corrected directly within the representation space without L2 training data or model adaptation.

Index Terms— self-supervised speech models, language identification, representation geometry, non-native speech.

## 1. INTRODUCTION

What should a language identification (LID) system output for non-native (L2) accented speech: accent or language? It should recognize the language being spoken rather than the accent origin. However, its prediction accuracy degrades with accent strength [1], with errors often shifting toward the speaker’s first language (L1) [2], leading to accent–language confusion. This confusion derives from interlanguage phonology transfer [3], where L2 speakers often realize unfamiliar L2 sounds through L1’s phonological categories [4,5], pulling LID predictions toward the L1. LID systems thus identify a speaker’s linguistic background rather than the language being spoken.

This issue is particularly evident in LID systems built on self-supervised speech models (S3Ms) [6–8]. To mitigate such L1 bias, existing approaches combine acoustic predictions with lexicon-free automatic speech recognition (ASR) outputs [1], aggregate short chunks, incorporate phonetic sequences [2], or fine-tune models on code-switched speech [9]. However, these methods require extra processing, components, or parameter adaptation.

In this work, we propose a method to mitigate the accent– language confusion directly within the representation geometry without any additional data or training. We draw on prior work showing that phonetic information in S3Ms is organized along linear directions [10, 11] and that language bias can be removed via residualization [12, 13]. Building on this, we estimate the L1 bias in non-native speech by fitting a linear separator solely between native speech in the target language and native speech in the speakers’ first languages (L1s). We then project out this rank-one component before classification.

We investigate five MMS-LID models [14] on three L2- accented English corpora (Svarah [15]; L2-ARCTIC [16]; ALLSSTAR [17]) spanning diverse L1 backgrounds. Across these datasets, misclassified English utterances disproportionately shift toward the speaker’s L1 or a language genealogically related to or in contact with it. This pattern is consistent with contact-driven phonetic convergence [18] and the language relationships captured by MMS-LID [19]. Without using L2-accented speech for direction estimation, our projection selectively restores English identification under strong L1 bias, whereas removing a random direction yields no such gain. Crucially, predictions for native speech and correctly classified L2 utterances remain fully intact.

Our contributions are threefold: (1) We demonstrate that L2-accented English errors in MMS-LID reflect the speaker’s L1 and its related languages. (2) We propose a closed-form projection to eliminate the L1-biased direction using solely native reference speech, requiring no retraining or non-native data for axis estimation. (3) Across five MMS-LID models and three corpora, our approach selectively restores misclassified non-native English utterances while leaving native and correctly classified L2 speech fully intact.

## 2. METHOD

We propose a geometric projection method to remove L1 bias directly within the representation space without retraining.

![](images/7dca1b119a054ac53e4d7887245cb453fdf0c73b26cb87166fe550b41e962efa.jpg)

![](images/dde41aa48111ed4e5575e035ceeaa74f4f7e9a3c842d51d79096960be165fb1d.jpg)

![](images/db5119450c6c9f7d878f36307e02cd7b1811b19b3caf651b411ce6bfb7b1c57b.jpg)  
Fig. 1. Geometry of L1-biased representations illustrated for native and non-native English speech. Top-left (Poles): Native reference distributions for English and non-English L1s. Top-right (X): Axis uˆ estimated from native speech (English+non-English). Bottom-left (Y): Decision margin from the frozen LID head. Bottom-right (Combined): Projections $( t = x \cdot { \hat { u } } )$ show L2-accented English positioned between native English and L1s, causing L1-related misclassifications.

Let a be a speech utterance and

$$
\begin{array} { r } { x = f _ { \theta } ( a ) \in \mathbb { R } ^ { d } , \qquad \hat { y } = g _ { \phi } ( x ) , } \end{array}\tag{1}
$$

where $f _ { \theta }$ is a pretrained speech encoder, d is the representation dimension, and $g _ { \phi }$ is its original LID head. For nonnative English, $g _ { \phi }$ can predict the speaker’s first language (L1) rather than English. Our goal is to correct this behavior without updating either θ or ϕ.

## 2.1. Native-only direction estimation

Our key hypothesis is that non-native English features are displaced along the geometric axis separating native English and native L1 speech. Consequently, this L1-biased direction can be estimated entirely without non-native samples. As conceptualized in Figure 1, evaluating the decision margin along this estimated axis illustrates how non-native speech can occupy an intermediate space.

We construct a native reference set

$$
\mathcal { D } _ { \mathrm { n a t } } = \{ ( x _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{2}
$$

containing a total of N reference representations $x _ { i } = f _ { \theta } ( a _ { i } )$ where $z _ { i } ~ = ~ 0$ denotes native English and $z _ { i } ~ = ~ 1$ denotes native speech in non-English L1s. No non-native (L2) speech is included.

Table 1. Per-L1 clip counts in the evaluation corpora. Only non-native English clips from speakers whose L1 is represented in FLEURS are retained.
<table><tr><td>Svarah</td><td></td><td colspan="2">L2-ARCTIC</td><td colspan="2">ALLSSTAR</td></tr><tr><td>Nepali</td><td>473</td><td>Spanish</td><td>600</td><td>Cantonese</td><td>551</td></tr><tr><td>Kannada</td><td>347</td><td>Hindi</td><td>450</td><td>Mandarin</td><td>458</td></tr><tr><td>Malayalam</td><td>311</td><td>Mandarin</td><td>450</td><td>Turkish</td><td>382</td></tr><tr><td>Urdu</td><td>292</td><td>Korean</td><td>300</td><td>Spanish</td><td>377</td></tr><tr><td>Odia</td><td>276</td><td>Vietnamese</td><td>300</td><td>Korean</td><td>370</td></tr><tr><td>Telugu</td><td>214</td><td>Arabic</td><td>150</td><td>Portuguese</td><td>179</td></tr><tr><td>Tamil</td><td>200</td><td></td><td></td><td>Russian</td><td>143</td></tr><tr><td>Hindi</td><td>173</td><td></td><td></td><td>Hindi</td><td>140</td></tr><tr><td>Gujarati</td><td>164</td><td></td><td></td><td>Vietnamese</td><td>136</td></tr><tr><td>Bengali</td><td>141</td><td></td><td></td><td>Hebrew</td><td>124</td></tr><tr><td>Marathi</td><td>138</td><td></td><td></td><td>Japanese</td><td>102</td></tr><tr><td>Assamese</td><td>111</td><td></td><td></td><td>Farsi</td><td>84</td></tr><tr><td>Punjabi</td><td>111</td><td></td><td></td><td>German</td><td>54</td></tr><tr><td>Sindhi</td><td>84</td><td></td><td></td><td>French</td><td>31</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td>Gujarati</td><td>30</td></tr><tr><td></td><td></td><td></td><td>Greek</td><td>27</td></tr><tr><td></td><td></td><td></td><td>Indonesian</td><td>26</td></tr><tr><td>3,035 Total</td><td></td><td>2,250 Total</td><td></td><td>3,214</td></tr></table>

We estimate the native English–L1 axis using an $\ell _ { 2 } \cdot$ regularized linear separator:

$$
\begin{array} { r l } { \displaystyle ( \boldsymbol { w } ^ { * } , \boldsymbol { b } ^ { * } ) = \arg \operatorname* { m i n } _ { \boldsymbol { w } , \boldsymbol { b } } } & { \frac { 1 } { 2 } \| \boldsymbol { w } \| _ { 2 } ^ { 2 } } \\ { + \displaystyle \sum _ { ( \boldsymbol { x } _ { i } , \boldsymbol { z } _ { i } ) \in \mathcal { D } _ { \mathrm { n a t } } } \mathcal { L } _ { \mathrm { C E } } \left( \boldsymbol { z } _ { i } , \boldsymbol { \sigma } ( \boldsymbol { w } ^ { \top } \boldsymbol { x } _ { i } + \boldsymbol { b } ) \right) , } \end{array}\tag{3}
$$

where $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ . The normalized classifier weight

$$
\hat { u } = \frac { w ^ { * } } { \Vert w ^ { * } \Vert _ { 2 } }\tag{4}
$$

defines the L1-biased direction. Once uˆ is estimated, this auxiliary separator is discarded and no longer used.

## 2.2. L1 bias removal via geometric projection

To remove L1 bias, we project each representation vector x away from the estimated direction uˆ:

$$
\boldsymbol { x } ^ { \prime } = \left( \boldsymbol { I } - \boldsymbol { \hat { u } } \boldsymbol { \hat { u } } ^ { \top } \right) \boldsymbol { x } = \boldsymbol { x } - \left( \boldsymbol { \hat { u } } ^ { \top } \boldsymbol { x } \right) \boldsymbol { \hat { u } } ,\tag{5}
$$

where $I \in \mathbb { R } ^ { d \times d }$ is the identity matrix. The scalar $\hat { u } ^ { \top } x$ is the coordinate (or magnitude) of x along the L1-biased direction. As $\hat { u } \hat { u } ^ { \top }$ has rank one, Eq. (5) removes only this onedimensional component.

The corrected representation is passed directly to the frozen LID head:

$$
\boxed { \boldsymbol { x } \xrightarrow { \mathrm { p r o j e c t o u t } \hat { \boldsymbol { u } } } \boldsymbol { x } ^ { \prime } = \left( \boldsymbol { I } - \hat { \boldsymbol { u } } \hat { \boldsymbol { u } } ^ { \top } \right) \boldsymbol { x } \xrightarrow { \mathrm { f r o z e n L I D h e a d } g _ { \phi } } \hat { \boldsymbol { y } } ^ { \prime } }
$$

Thus, the correction requires no L2-accented speech and no model retraining.

```html
<sup>1</sup>https://huggingface.co/facebook/mms-lid-256
<sup>2</sup>https://huggingface.co/facebook/mms-lid-512
<sup>3</sup>https://huggingface.co/facebook/mms-lid-1024
<sup>4</sup>https://huggingface.co/facebook/mms-lid-2048
<sup>5</sup>https://huggingface.co/facebook/mms-lid-4017
```

![](images/cdeec132e6ff8c2d4f7fbd5e3ebc4f19c0eed93f2e90736c50db2459410d9ea0.jpg)  
Fig. 2. Accent-induced confusion in MMS-LID-4017. (a) Prediction breakdown for non-native English speech, categorized into English, speaker’s own L1 (Own L1), languages related to or in contact with L1 (Near L1), and remaining languages (True Miss). (b) Mean Whisper token-normalized negative log-likelihood (NLL) per speaker (ordered by NLL), with horizontal lines indicating corpus averages and dashed lines denoting the FLEURS native English baseline.

## 3. EXPERIMENTAL SETUP

## 3.1. Data

We conduct experiments on three English corpora spanning diverse speaker L1 backgrounds: Svarah [15], L2- ARCTIC [16], and ALLSSTAR [17]. We retain utterances whose speaker L1 is represented in FLEURS [20], yielding 3,035, 2,250, and 3,214 clips from 14, 6, and 17 L1 groups, respectively (Table 1). We use native English and the corresponding native L1 speech from the FLEURS dev split to estimate the direction in Eq. (4). Its effect on native-language predictions is evaluated separately on the FLEURS test split. No accented-English speech is used for direction estimation.

## 3.2. Models and evaluation

We test five model variants sharing the Massively Multilingual Speech (MMS) backbone [14]: MMS-LID-256<sup>1</sup>, MMS-LID-512<sup>2</sup>, MMS-LID-1024<sup>3</sup>, MMS-LID-2048<sup>4</sup>, and MMS-LID-4017<sup>5</sup>. For each model, we extract its representation x from the last encoder layer before the classifier, apply Eq. (5), and pass $x ^ { \prime }$ to the original frozen LID head. We evaluate the English identification rate (% of clips predicted as English) on native English, non-native English, and native L1 (non-English) speech. To validate that the estimated axis captures meaningful L1 directions, we compare our approach against 20 random control directions.

## 4. RESULTS

## 4.1. L1-dependent accent-language confusion

We examine the misclassification patterns of LID models on non-native speech. Figure 2(a) shows that MMS-LID-4017 predicts 14%, 57%, and 93% of non-native English clips in Svarah, L2-ARCTIC, and ALLSSTAR as English, respectively. Most misclassifications fall into the speaker’s own L1 or a language genealogically related to or in contact with it (e.g., Hindi-accented English → Punjabi/Telugu; Spanishaccented English → Catalan; Cantonese-accented English → Mandarin). This structured pattern is consistent with contactdriven phonetic convergence [18] and the language relationships captured by MMS-LID [19]. Similar trends hold across all model scales (Section 4.2).

Following prior work that estimates accentedness from posterior mismatch in automatic speech recognition (ASR) [21], we use Whisper [22]’s token-normalized negative log-likelihood (NLL) as a proxy for accent strength. Figure 2(b) shows that mean NLL decreases from 1.80 on Svarah to 1.51 on L2- ARCTIC and 1.03 on ALLSSTAR (versus 0.90 for native English on FLEURS), inversely mirroring the English identification rate. Thus, greater accent strength corresponds to higher L1-biased confusion.

## 4.2. Results across models and corpora

Figure 3 shows that the proposed projection generalizes across all MMS-LID variants. English identification rates for non-native English improve by 51–71%p on Svarah and 17–24%p on L2-ARCTIC, whereas the effect is minimal on ALLSSTAR where non-native English is already correctly identified. This corpus-level pattern reflects baseline L2-accent strength (Section 4.1); indeed, the performance gain across L1 groups strongly correlates with Whisper NLL (r = +0.77). Meanwhile, identification rates for native English and native L1 speech change negligibly, confirming that the projection corrects L1-biased misclassifications without indiscriminately shifting speech toward English.

To examine this mechanism, we visualize MMS-LID-4017 on Svarah in Figure 4. The direction uˆ, estimated solely from the FLEURS dev set, separates the two native poles on test data. As non-native English representations lie between these poles, they are often misclassified as L1 or related languages to L1. Removing this single component redirects most non-native English representations to English while keeping native English and native L1 (non-English) speech on their respective sides, isolating accent-induced L1 bias without collapsing the native English–L1 distinction.

## 4.3. Validity of the estimated direction

The observed gains depend on isolating the specific L1-bias axis rather than arbitrary subspace modification. Figure 5(a)

![](images/cf7862a6bf312b4fe328f40e275de85fa4d1f4d1312618f9a894ebd1fee14734.jpg)

Fig. 3. English identification performance across models and corpora. Open and filled markers show English identification rates before and after projection, respectively. Blue numbers indicate percentage-point changes for non-native English, while red annotations at the bottom show model-averaged native L1 identification rates before and after projections. The projection substantially improves English identification on Svarah and L2-ARCTIC while preserving predictions for both native groups.  
![](images/46aca89dcce648c068bcc3c08e6fd219aa29283507a02a3b1c2a8a2762792a21.jpg)

![](images/72e05f501b3f8b537af02309c7f5b076eea91fc4d47f115ab6f17e0789e88838.jpg)

![](images/ed9f18aef5f777a98dccf5aad997f1d3389b2c3efb7110cdd5b52ae0ac9dc545.jpg)  
Fig. 4. Effect of removing the native English–L1 direction. For MMS-LID-4017, uˆ is estimated solely from the FLEURS dev set and visualized on test representations from FLEURS and Svarah. Projecting uˆ out moves non-native English representations toward the English side of the frozen LID decision boundary while preserving representations for both native groups.

(a)  
![](images/24e72279872393ac1a5fe4eba1572fab0e3067860e85397873a3b7d13da3f753.jpg)

(b)  
![](images/27e327c785709d26047e693743d7e76644409515e066ae4a32cb579f7afc2417.jpg)  
Fig. 5. Validity of the L1-bias direction. (a) Alignment of uˆ versus 20 random directions with the native English–L1 axis (MMS-LID-4017). (b) English identification accuracy before projection and after removing uˆ or random directions.

shows that uˆ aligns with the separation between native English and native L1 speech, unlike random directions. Consequently, removing 20 random control directions yields no improvement over the baseline, whereas removing uˆ consistently improves English identification on Svarah and L2- ARCTIC (Figure 5(b)). This confirms that performance gains arise directly from targeting the identified L1-bias direction.

## 5. CONCLUSION

We show that language identification errors for non-native English cluster around the speaker’s L1 and related languages. We correct this L1 bias by estimating a single native English–L1 direction solely from native speech and projecting it out before the frozen LID head. Across models and corpora, this projection restores L2-accented English identification while leaving native English and native L1 speech unaffected; removing random directions yields no gains. Accent-induced L1 bias can thus be corrected in the representation space without non-native training data or model adaptation.

## 6. REFERENCES

[1] Kunnar Kukk and Tanel Alumae, “Improving Language Iden-¨ tification of Accented Speech,” in Interspeech 2022, 2022, pp. 1288–1292.

[2] Niyati Bafna and Matthew Wiesner, “LID Models are Actually Accent Classifiers: Implications and Solutions for LID on Accented Speech,” in Interspeech 2025, 2025, pp. 1488–1492.

[3] Roy C Major, Foreign accent: The ontogeny and phylogeny of second language phonology, Routledge, 2001.

[4] Jisang Park, Minu Kim, DaYoung Hong, and Jongha Lee, “Compositional phoneme approximation for l1-grounded l2 pronunciation training,” in Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics, 2025, pp. 434–443.

[5] Joseph Tepperman, Erik Bresch, Yoon-Chul Kim, Sungbok Lee, Louis Goldstein, and Shrikanth S Narayanan, “An articulatory analysis of phonological transfer using real-time mri.,” in INTERSPEECH, 2009, pp. 700–703.

[6] Zhiyun Fan, Meng Li, Shiyu Zhou, and Bo Xu, “Exploring wav2vec 2.0 on Speaker Verification and Language Identification,” in Interspeech 2021, 2021, pp. 1509–1513.

[7] Andros Tjandra, Diptanu Gon Choudhury, Frank Zhang, Kritika Singh, Alexis Conneau, Alexei Baevski, Assaf Sela, Yatharth Saraf, and Michael Auli, “Improved language identification through cross-lingual self-supervised learning,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 6877–6881.

[8] Yoonjeong Lee, Jihwan Lee, and Shrikanth Narayanan, “L1 influence on stability in speech foundation model-based articulatory mapping of l2 english speech,” The Journal of the Acoustical Society ofAmerica, vol. 158, no. 4 Supplement, pp. A194–A195, 2025.

[9] Adyasha Patra, Dhiraj Kumar Sah, and Preethi Jyothi, “Improving language identification for code-switched speech: The pivotal role of accented english,” in Findings of the Association for Computational Linguistics: EACL 2026, 2026, pp. 4643–4656.

[10] Mukhtar Mohamed, Oli Danyi Liu, Hao Tang, and Sharon Goldwater, “Orthogonality and isotropy of speaker and phonetic information in self-supervised speech representations,” in Interspeech 2024, 2024, pp. 3625–3629.

[11] Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, David R Mortensen, and David Harwath, “Self-supervised speech models encode phonetic context via position-dependent orthogonal subspaces,” arXiv preprint arXiv:2603.12642, 2026.

[12] Minu Kim, Eunjung Yeo, Kwanghee Choi, and June-Woo Kim, “Language orthogonalization of self-supervised speech representations for cross-lingual parkinson’s detection,” arXiv preprint arXiv:2609.09499, 2026.

[13] Minu Kim, Ji Sub Um, and Hoirin Kim, “Language orthogonalization for zero-shot cross-lingual audio deepfake detection,” arXiv preprint arXiv:2609.16458, 2026.

[14] Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv

Vyas, Maryam Fazel-Zarandi, et al., “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research, vol. 25, no. 97, pp. 1–52, 2024.

[15] Tahir Javed, Sakshi Joshi, Vignesh Nagarajan, Sai Sundaresan, Janki Nawale, Abhigyan Raman, Kaushal Bhogale, Pratyush Kumar, and Mitesh M. Khapra, “Svarah: Evaluating English ASR Systems on Indian Accents,” in Interspeech 2023, 2023, pp. 5087–5091.

[16] Guanlong Zhao, Evgeny Chukharev-Hudilainen, Sinem Sonsaat, Alif Silpachai, Ivana Lucic, Ricardo Gutierrez-Osuna, and John Levis, “L2-arctic: A non-native english speech corpus,” 2018.

[17] Ann R Bradlow, Lauren Ackerman, L Ann Burchfield, Lisa Hesterberg, Jenna Luque, and Kelsey Mok, “Language-and talker-dependent variation in global features of native and nonnative speech,” in Proceedings of the... International Congress of Phonetic Sciences. International Congress of Phonetic Sciences, 2011, p. 356.

[18] Minu Kim, Kangwook Jang, and Hoirin Kim, “Improving cross-lingual phonetic representation of low-resource languages through language similarity analysis,” in ICASSP 2025- 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[19] Minu Kim, Hoirin Kim, and David R. Mortensen, “Scaling Self-Supervised Speech Models Uncovers Deep Linguistic Relationships: Evidence from the Pacific Cluster,” in Interspeech 2026, 2026, pp. 1671–1676.

[20] Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna, “Fleurs: Few-shot learning evaluation of universal representations of speech,” in 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2023, pp. 798–805.

[21] Ramya Rasipuram, Milos Cernak, Alexandre Nachen, and Mathew Magimai-Doss, “Automatic accentedness evaluation of non-native speech using phonetic and sub-phonetic posterior probabilities,” in Interspeech 2015, 2015, pp. 648–652.

[22] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28492– 28518.