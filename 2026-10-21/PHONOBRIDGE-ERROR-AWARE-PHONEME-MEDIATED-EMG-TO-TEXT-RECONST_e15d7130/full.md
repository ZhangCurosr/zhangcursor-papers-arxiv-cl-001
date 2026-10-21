# PHONOBRIDGE: ERROR-AWARE PHONEME-MEDIATED EMG-TO-TEXT RECONSTRUCTION

Wen Qiu<sup>1,∗</sup>, Jiacheng Zhang<sup>1,2,∗</sup>, Xuanda Chen<sup>1</sup>, Siqi Cai<sup>2,3,†</sup>, and Haizhou Li<sup>1,2,†</sup>

<sup>1</sup>School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, China <sup>2</sup>Shenzhen Loop Area Institute, Shenzhen, China <sup>3</sup>Harbin Institute of Technology, Shenzhen, China

## ABSTRACT

Electromyography (EMG) provides a non-invasive means of decoding silently articulated speech into text, but accurate decoding remains challenging due to limited availability of paired EMG–text data. We introduce PhonoBridge, a two-stage framework comprising a channel-aware EMGto-phoneme frontend and a large language model (LLM) backend for text reconstruction. The frontend is trained on paired EMG–text data that consists of channel-wise feature extraction, cross-channel fusion, and electrode embeddings. With phonemes as the intermediate representation, we expand the backend’s training data by converting LibriSpeech transcripts into phoneme–text pairs without additional EMG recordings. To adapt the LLM to frontend noise, we introduce Error-Profile-Guided Augmentation (EPGA), which perturbs transcript-derived phoneme sequences according to error patterns estimated from frontend predictions, generating additional training pairs for the LLM. On the Open Vocabulary silent-speech dataset, PhonoBridge achieves 24.77% WER, outperformingthe previously reported best of 28.27% on the same test split.

Index Terms— Silent speech recognition, surface electromyography

## 1. INTRODUCTION

Silent-speech interfaces aim to recover linguistic content without requiring an audible utterance. Electromyography (EMG) offers a non-invasive means of recording muscle activity associated with articulation through electrodes placed on the face and neck [1, 2, 3, 4]. This has motivated research on the decoding task, which seeks to recover the intended text or audio from EMG signals.

Existing studies have adopted modular architectures for EMG-to-text decoding. For example, Gaddy and Klein [5, 6] developed a character-level recognition model trained with connectionist temporal classification (CTC) [7], incorporating an external language model during decoding. Gowda and Miller [8] used geometric signal representations and CTC training to predict phonemes, followed by weighted finitestate transducer (WFST) decoding with an n-gram language model. These methods share a common division of roles: the frontend extracts linguistic information from EMG signals, while the decoder reconstructs the intended text from the frontend’s predictions. Recent advances in large language models (LLMs) offer new opportunities to strengthen the reconstruction stage [9] within the same modular framework.

![](images/458912ade7b70e2e225691eba8d618c1da921ef507b83a2c1d45e9a932ee1ed1.jpg)  
Fig. 1. Electrode placement and qualitative comparison of decoding errors. Left: Electrode placement for EMG acquisition. Right: A comparison of EMG-decoded text and human transcription of the corresponding voiced utterance against the reference text.

However, an LLM alone does not guarantee accurate reconstruction from EMG-derived predictions. The backend must learn to map noisy frontend outputs to the intended texts, but limited paired EMG-text data provide insufficient supervision for this mapping. External speech transcripts [10] offer abundant training data, but the phoneme sequences derived from them are clean and therefore do not reflect the errors encountered at inference time. This raises a central question: “How can transcript-derived phoneme-text pairs be used to train an LLM to correct EMG-specific recognition errors?”

To address this question, we propose PhonoBridge, a two-stage EMG-to-text framework comprising a channelaware EMG-to-phoneme frontend and an error-aware LLMbased text reconstruction backend. The frontend learns to predict phoneme sequences from paired EMG–text data using a channel-aware encoder. It first extracts temporal features from each EMG channel separately and then integrates them across channels, with electrode embeddings providing explicit channel identity.

We use phonemes as both the frontend supervision target and the intermediate representation because EMG and acoustic speech provide different observations of the same underlying articulatory process. During speech production, coordinated muscle activation drives movements of the lips, jaw, and tongue, while EMG records electrical activity associated with these articulatory movements [11]. Although distinct at the signal level, both modalities reflect the same underlying phonemic structure and can therefore exhibit similar phoneme-level confusions. As shown in Fig. 1, both the EMG-decoded output and the human transcription of the parallel voiced utterance replace “shooting” with “shouting.” This shared substitution provides a qualitative example of such cross-modal phonetic similarity and further motivates the use of phonemes as the intermediate representation.

The phoneme interface also allows us to expand the backend’s training data by converting LibriSpeech transcripts into phoneme-text pairs without collecting additional EMG recordings. Since these transcript-derived sequences do not contain the errors produced by the EMG frontend, we introduce Error-Profile-Guided Augmentation (EPGA), which estimates substitution, deletion, and insertion patterns from a held-out subset of the training set and applies them to clean phoneme sequences. We fine-tune the LLM on a mixture of clean and EPGA-augmented phoneme-text pairs, enabling the backend to reconstruct text from noisy EMGderived predictions. On the Open Vocabulary dataset, Phono-Bridge achieves 24.77% WER, compared with 28.27% for the strongest prior system on the same test split.

## 2. METHOD

## 2.1. Channel-Aware EMG-to-Phoneme Frontend

Given a multichannel EMG sequence $\mathbf { x } \in \mathbb { R } ^ { T _ { 0 } \times n }$ with $T _ { 0 }$ time samples and n electrode channels, the frontend predicts a phoneme sequence $\hat { \mathbf { y } }$ . Since each electrode captures muscle activity from a different facial location, we interleave channel-specific temporal encoding with cross-electrode fusion to preserve electrode-specific information while modeling coordinated muscle activity.

For the i-th electrode, let $\mathbf { \bar { x } } ^ { ( i ) } = \mathbf { x } _ { : , i } \in \mathbb { R } ^ { T _ { 0 } }$ . A channelspecific residual convolutional block first extracts and downsamples its local temporal features:

$$
\mathbf { h } ^ { ( i ) } = f _ { i } ^ { \mathrm { p r e } } ( \mathbf { x } ^ { ( i ) } ) \in \mathbb { R } ^ { T _ { 1 } \times d } ,\tag{1}
$$

where $T _ { 1 }$ is the downsampled sequence length, d is the feature dimension, and the branches $f _ { i } ^ { \mathrm { p r e } }$ have independent parameters. Before fusion, each channel is associated with a learn-

able electrode embedding $\mathbf { e } _ { i } \in \mathbb { R } ^ { d _ { e } }$ . The electrode embedding is concatenated with each channel feature and mapped back to the original feature dimension:

$$
\mathbf { u } _ { t } ^ { ( i ) } = \mathbf { W } _ { e } [ \mathbf { h } _ { t } ^ { ( i ) } ; \mathbf { e } _ { i } ] + \mathbf { b } _ { e } .\tag{2}
$$

The resulting channel representations are concatenated across electrodes and passed to a residual temporal fusion module, which jointly models cross-electrode interactions and local temporal context. The fused representation is then sliced into n channel-specific streams, each of which is further processed by two residual convolutional blocks with independent parameters. The refined channel representations are concatenated and projected into the Transformer hidden space.

The resulting sequence is passed through a Transformer encoder [12, 13] with learned relative positional representations to model long-range temporal dependencies. Let U denote the target vocabulary. A linear prediction head produces frame-level logits over the CTC vocabulary $\tilde { \mathcal { U } } = \mathcal { U } \cup$ {⟨blank⟩}, and the frontend is trained with the standard CTC objective [7]. During inference, greedy CTC decoding collapses consecutive repetitions and removes the blank symbol, yielding yˆ as input to the LLM backend.

## 2.2. LLM-Based Text Reconstruction

Given the phoneme sequence $\hat { \mathbf { y } }$ decoded by the frontend, we serialize its phoneme units as the LLM input. The LLM then autoregressively reconstructs the normalized text sequence zˆ.

To adapt the backend, we convert transcripts from LibriSpeech [10] into phoneme sequences y using graphemeto-phoneme conversion, producing clean phoneme–text pairs $( \mathbf { y } , \mathbf { z } )$ without additional EMG recordings. We adapt the LLM with LoRA [14] in two phases: the clean adaptation phase first learns the phoneme-to-text mapping from clean pairs, after which the error-aware adaptation phase continues training on a mixture of EPGA-augmented pairs (y˜, z) and replayed clean pairs. This exposes the backend to frontendlike errors while retaining its clean reconstruction ability. The construction of y˜ is described in Sec. 2.3.

## 2.3. Error-Profile-Guided Augmentation

EPGA transfers error patterns observed from the frontend to clean transcript-derived phoneme sequences while preserving their text targets.

Error-profile construction. We freeze the trained frontend, decode a held-out EMG subset using greedy CTC decoding, and align each prediction with its reference by minimum edit distance [15]. For each aligned utterance $j ,$ we record a sentence-level edit-count template $\tau _ { j } = ( n _ { S } ^ { ( j ) } , n _ { D } ^ { ( j ) } , n _ { I } ^ { ( j ) } )$ which contains its numbers of substitutions, deletions, and insertions. These templates are grouped into sets $\mathcal { T } _ { \ell }$ according to reference length.

![](images/0a70fd9a9961c725eedc099afc55e7dda2e15ab6ee7441f271a5be4d1add57e3.jpg)  
Fig. 2. Overall architecture of PhonoBridge. The framework comprises two stages: (a) a channel-aware frontend that maps multichannel EMG to phoneme sequences and is trained with CTC loss; and (b) an LLM backend that reconstructs text from the predicted phoneme sequence and is fine-tuned on clean and EPGA-augmented phoneme–text pairs

We additionally estimate phoneme-conditioned edit statistics. Let $C _ { \mathrm { s u b } } ( a , b )$ count substitutions from reference unit a to predicted unit b, $C _ { \mathrm { d e l } } ( a )$ count deletions of $a , C _ { \mathrm { r e f } } ( a )$ count its occurrences in the reference sequences, and $C _ { \mathrm { i n s } } ( a , b )$ count insertions of b associated with reference context a. We define

$$
\begin{array} { r l } & { P _ { \mathrm { s u b } } ( b \mid a ) = \cfrac { C _ { \mathrm { s u b } } ( a , b ) } { \sum _ { v \ne a } C _ { \mathrm { s u b } } ( a , v ) } , \quad r _ { \mathrm { d e l } } ( a ) = \cfrac { C _ { \mathrm { d e l } } ( a ) } { C _ { \mathrm { r e f } } ( a ) } , } \\ & { P _ { \mathrm { i n s } } ( b \mid a ) = \cfrac { C _ { \mathrm { i n s } } ( a , b ) } { \sum _ { v } C _ { \mathrm { i n s } } ( a , v ) } . } \end{array}\tag{3}
$$

These statistics characterize replacement content, deletion tendency, and insertion content, respectively.

Profile-guided augmentation. Given a clean sequence $\mathbf { y } =$ $\left( y _ { 1 } , \dots , y _ { L } \right)$ , EPGA identifies its length bucket ℓ and samples an edit-count template $\tau = ( n _ { S } , n _ { D } , n _ { I } )$ from $\tau _ { \ell } .$ . The template specifies the numbers of substitutions, deletions, and insertions, preserving the sentence-level edit-count structure observed in frontend predictions of similar length.

To instantiate the template, EPGA first samples $n _ { D }$ deletion positions without replacement according to $r _ { \mathrm { d e l } } ( y _ { i } )$ . It then selects $n _ { S }$ substitution positions from the remaining reference units and draws a replacement b for each selected position from $P _ { \mathrm { s u b } } ( b \mid y _ { i } )$ . For each of the $n _ { I }$ insertions, an anchor is sampled from the sentence start and the reference positions. Given its reference context $^ { a , }$ the inserted unit b is drawn from $P _ { \mathrm { i n s } } ( b \ | \ a )$ . The resulting sequence $\tilde { \textbf { y } } =$ $\mathcal { A } _ { \mathrm { E P G A } } ( \mathbf { y } )$ is paired with the original text target.

## 3. EXPERIMENTS

## 3.1. Experimental Setting

Dataset. We evaluate on the eight-channel Open Vocabulary dataset introduced by Gaddy and Klein [5], using the official split and text normalization protocol of [6]. We reserve a held-out subset of the official training split exclusively for estimating the EPGA error profile and use the remaining data to train the frontend. We report phoneme error rate (PER) for frontend recognition and word and character error rates (WER/CER) for reconstruction and end-to-end evaluation on the test set. Lower error rates are preferred.

Baselines. We compare PhonoBridge with Gaddy and Klein [6], EMG2Vec [16], EMG-LLM [17], and two wordlevel CTC [7] baselines using CNN-Transformer [12] and CNN-BiLSTM [18] encoders. All systems are evaluated on the same test split.

## 3.2. Implementation Details

EMG-to-phoneme frontend. The 1-kHz EMG signals are notch-filtered at 60 Hz and resampled to 800 Hz. The frontend is initialized from a character-target warm-start checkpoint and trained with CTC using AdamW [19] $( 3 \times 1 0 ^ { - 4 } )$ ; the checkpoint with the lowest development-set PER is used for subsequent experiments.

LLM backend adaptation. We adapt Qwen3-8B [9] with LoRA (rank 16) in two phases. The clean adaptation phase uses 1M LibriSpeech-derived phoneme–text pairs, followed by an error-aware adaptation phase using 1M EPGA-augmented pairs mixed with 200k clean replay pairs. Each phase runs for one epoch with learning rates of $2 \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 4 }$ respectively. EPGA estimates length-conditioned, phonemedependent edit statistics from held-out reference–prediction alignments, with the sampled error rate scaled by 1.17 and capped at 0.30. We use greedy decoding for inference.

Table 1. End-to-end WER (%) on the Open Vocabulary test split. All systems are evaluated on the same split.
<table><tr><td>Method</td><td>WER (%)</td></tr><tr><td>Gaddy and Klein [6]</td><td>28.27</td></tr><tr><td>EMG2Vec [16]</td><td>29.45</td></tr><tr><td>EMG-LLM [17]</td><td>108.29</td></tr><tr><td>CNN-Transformer</td><td>51.29</td></tr><tr><td>CNN-BiLSTM</td><td>45.31</td></tr><tr><td>PhonoBridge (Ours)</td><td>24.77</td></tr></table>

## 4. RESULTS AND DISCUSSION

## 4.1. Main Results

Table 1 compares PhonoBridge against baseline systems. PhonoBridge achieves 24.77% WER, outperforming all baselines, including the previous state-of-the-art Gaddy and Klein system (28.27%) and the self-supervised EMG2Vec approach by Hou et al. (29.45%). The substantial improvement over EMG-LLM by Mohapatra et al. (108.29% WER) supports the effectiveness of phoneme-mediated decoupling over direct EMG-to-text approaches. Our method also substantially outperforms the direct CNN-based architectures (CNN-Transformer and CNN-BiLSTM), providing further evidence for the two-stage design that combines signal-level encoding with linguistic reconstruction.

## 4.2. Channel-Aware Encoder Ablation

We ablate the channel-aware design in the frontend encoder to validate the effectiveness of channel-wise processing and electrode embeddings. Table 2 compares different frontend configurations on PER and collapse WER (cWER), computed by splitting predicted and reference phoneme sequences at word-boundary tokens (<SEP>) and measuring edit distance [15] between the resulting phoneme chunks.

Channel-wise processing provides the largest improvement: without electrode embeddings, it reduces PER from 33.43% to 14.17% and cWER from 67.35% to 37.62%. Electrode embeddings provide additional gains, with the complete encoder achieving the lowest PER (13.65%) and cWER (35.53%), slightly outperforming Gaddy and Klein on both metrics. These results indicate that preserving channelspecific patterns before cross-channel fusion is critical, while explicitly encoding electrode identity provides a further improvement.

Table 2. Channel-aware encoder ablation on the test split. Results are reported in PER and collapse WER (cWER), with Gaddy and Klein included for reference.
<table><tr><td>Method</td><td>Channel-wise</td><td>Elec. emb.</td><td>PER (%)</td><td>cWER (%)</td></tr><tr><td>Gaddy and Klein [6]</td><td>x</td><td>x</td><td>14.12</td><td>36.35</td></tr><tr><td>PhonoBridge</td><td>x</td><td>x</td><td>33.43</td><td>67.35</td></tr><tr><td>PhonoBridge</td><td>x</td><td>√</td><td>32.39</td><td>63.76</td></tr><tr><td>PhonoBridge</td><td>√</td><td>x</td><td>14.17</td><td>37.62</td></tr><tr><td>PhonoBridge</td><td>√</td><td>√</td><td>13.65</td><td>35.53</td></tr></table>

Table 3. Ablation of LLM backend adaptation on the test set. Mixed training combines EPGA-augmented pairs with replayed clean pairs, and → denotes sequential adaptation.
<table><tr><td>Adaptation setting</td><td>WER (%)</td><td>CER (%)</td></tr><tr><td>Qwen3-8B (zero-shot)</td><td>188.22</td><td>116.77</td></tr><tr><td>Clean only</td><td>66.53</td><td>34.53</td></tr><tr><td>EPGA-augmented only</td><td>29.41</td><td>17.23</td></tr><tr><td>Mixed, single phase</td><td>25.89</td><td>22.39</td></tr><tr><td>Clean → mixed (ours)</td><td>24.77</td><td>14.58</td></tr></table>

## 4.3. Backend Training Strategy Ablation

We ablate the backend adaptation strategy to examine the roles of clean and EPGA-augmented phoneme–text pairs and the benefit of sequential training. Table 3 compares zeroshot inference, single-source adaptation, single-phase mixed adaptation, and our two-phase procedure.

Without task-specific adaptation, Qwen3-8B performs poorly on phoneme-to-text reconstruction. Clean-only adaptation learns the basic phoneme-to-text mapping but remains sensitive to frontend errors, whereas training only on EPGAaugmented pairs improves robustness but is less effective than combining the two data sources. Direct single-phase training on the mixed data achieves 25.89% WER, while our two-phase strategy further reduces WER to 24.77% and CER to 14.58%. This improvement indicates that both the training-data composition and adaptation order contribute to the final performance.

## 5. CONCLUSION

This paper presents PhonoBridge, a phoneme-mediated framework for open-vocabulary silent-speech decoding from multichannel EMG. It separates phoneme recognition from text reconstruction and uses EPGA to expose the language model to realistic frontend errors. Experiments on the Open Vocabulary dataset demonstrated its effectiveness, with ablations confirming the contributions of channel-aware processing, electrode embeddings, EPGA, and sequential backend adaptation. Overall, the results support phoneme-level mediation for connecting noisy EMG recognition with openvocabulary text generation.

## 6. REFERENCES

[1] Jose A. Gonzalez-Lopez, Alejandro Gomez-Alanis, Juan M. Mart´ın Donas, Jos˜ e L. P´ erez-C´ ordoba, and An-´ gel M. Gomez, “Silent speech interfaces for speech restoration: A review,” IEEE Access, vol. 8, pp. 177995–178021, 2020.

[2] Xinyi Yang, Jing Yan, Hanting Zhang, and Liang Wu, “Noninvasive and quantitative measurement of intrinsic laryngeal muscle activity using surface electromyography,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–10, 2025.

[3] Bradley Denby, Tanja Schultz, Kiyoshi Honda, Thomas Hueber, Juan M. Gilbert, and Jonathan S. Brumberg, “Silent speech interfaces,” Speech Communication, vol. 52, no. 4, pp. 270–287, 2010.

[4] Cara E. Stepp, “Surface electromyography for speech and swallowing systems: Measurement, analysis, and interpretation,” Journal ofSpeech, Language, and Hearing Research, vol. 55, no. 4, pp. 1232–1246, 2012.

[5] David Gaddy and Dan Klein, “Digital voicing of silent speech,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. 2020, pp. 5521–5530, Association for Computational Linguistics.

[6] David Gaddy and Dan Klein, “An improved model for voicing silent speech,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2021, pp. 175–181, Association for Computational Linguistics.

[7] Alex Graves, Santiago Fernandez, Faustino Gomez, and´ Jurgen Schmidhuber, “Connectionist temporal classifi-¨ cation: labelling unsegmented sequence data with recurrent neural networks,” in Proceedings ofthe 23rd International Conference on Machine Learning, New York, NY, USA, 2006, ICML ’06, pp. 369–376, Association for Computing Machinery.

[8] Harshavardhana T Gowda and Lee M. Miller, “Noninvasive electromyographic speech neuroprosthesis: a geometric perspective,” in Findings of the Association for Computational Linguistics: ACL 2026, Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, Eds., San Diego, California, United States, July 2026, pp. 11636–11650, Association for Computational Linguistics.

[9] An Yang, Anfeng Li, Baosong Yang, et al., “Qwen3 technical report,” 2025.

[10] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, “Librispeech: An asr corpus based on public domain audio books,” in 2015 IEEE International Conference on Acoustics, Speech and Signal Processing, 2015, pp. 5206–5210.

[11] Szu-Chen Jou, Tanja Schultz, Matthias Walliczek, Florian Kraft, and Alex Waibel, “Towards continuous speech recognition using surface electromyography,” in Proceedings of Interspeech 2006, 2006, pp. paper 1592– Mon3WeS.3.

[12] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30, 2017, pp. 5998–6008.

[13] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani, “Self-attention with relative position representations,” in Proceedings of the 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). 2018, pp. 464–468, Association for Computational Linguistics.

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “Lora: Low-rank adaptation of large language models,” 2021.

[15] Vladimir I. Levenshtein, “Binary codes capable of correcting deletions, insertions, and reversals,” Soviet Physics Doklady, vol. 10, pp. 707–710, 1966.

[16] Qinhan Hou, Stefano van Gogh, Kevin Scheck, Zhao Ren, Tanja Schultz, Michael Wand, and Jurgen Schmid-¨ huber, “emg2vec: Self-supervised pretraining in electromyography-based silent speech interfaces,” in 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2024, pp. 1–5.

[17] Payal Mohapatra, Akash Pandey, Xiaoyuan Zhang, and Qi Zhu, “Can LLMs understand unvoiced speech? exploring EMG-to-text conversion with LLMs,” in Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics. 2025, pp. 703–712, Association for Computational Linguistics.

[18] Sepp Hochreiter and Jurgen Schmidhuber, “Long short-¨ term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[19] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.