# CHIME-9 ECHI: A MACHINE LEARNING CHALLENGE FOR ENHANCING CONVERSATIONS TO ADDRESS HEARING IMPAIRMENT

Robert Sutherland<sup>1</sup>, Thomas Kuebert<sup>2</sup>, Marko Lugger<sup>2</sup>, Stefan Petrausch<sup>2</sup>, Eline Borch Petersen<sup>3</sup>, Juan Azcarreta Ortiz<sup>4</sup>, Buye Xu<sup>5</sup>, Stefan Goetze<sup>1,6</sup>, Jon Barker<sup>1</sup>

<sup>1</sup>School of Computer Science, University of Sheffield, Sheffield, United Kingdom <sup>2</sup>WS Audiology, Erlangen, Germany <sup>3</sup>ORCA Labs, WS Audiology, Lynge, Denmark <sup>4</sup>Meta Reality Labs, Cambridge, United Kingdom <sup>5</sup>Meta Reality Labs, Redmond, United States of America <sup>6</sup>South Westphalia University of Applied Sciences, Iserlohn, Germany

## ABSTRACT

This work presents the task and results of the CHiME-9 challenge for Enhancing Conversations to address Hearing Impairment. The challenge considers the scenario of four-party conversations in a noisy, cafeteria-style environment with interfering speech sources and sound effects. Participants are provided with audio recordings made with Meta Aria glasses and hearing aid microphones, and clean speech samples of the conversation participants. The task is to extract the speech of the conversation partners from the noisy multichannel recordings with the goal of improving the intelligibility and quality of the speech, evaluated using objective metrics and subjective listening tests. This paper reviews submissions from seven teams and ranks them on a combination of subjective intelligibility and quality. Results show that while the objective metrics do not reflect listener performance, the top systems were able to make substantial improvements over the challenge baseline in both intelligibility and quality ratings.

Index Terms— speech intelligibility, speech quality, conversational speech enhancement, machine learning

## 1. INTRODUCTION

While existing assistive hearing devices (AHDs) provide a substantial benefit in some scenarios, there are still many scenarios in which they face serious challenges. In particular, this issue is found in conversations in dynamic, noisy environments such as cafes, restaurants and pubs. Their limited effectiveness makes it harder for those with hearing impairments to comfortably socialise, leading to a poorer quality of life.

In recent years, research into using neural networks (NNs) to enhance speech for AHDs has shown great potential for improving the experience of hearing-impaired individuals. This, coupled with recent advances in low-power NN hardware, has ushered in a new era for AHDs, with AI-enabled hearing aids (HAs) now available to the general public.

Enhancing speech in scenarios that are both acoustically and socially complex presents a variety of challenges for NN-based techniques. Real-world situations often contain dynamic background noise, including interfering speech, making it especially difficult for algorithms to effectively enhance the speech of a desired speaker. Further, multi-party conversations tend to have very rapid turn-taking and overlapping speech [1, 2], where the active speaker is frequently changing, and there can also be multiple active speakers at any one time.

The CHiME-9 Enhancing Conversations to address Hearing Impairment (ECHI) challenge focuses on this scenario, providing researchers with the ECHI dataset [3] (cf. Section 2), which captures real conversations in a cafeteria-style noisy environment. Teams participating in the challenge developed NN systems to enhance speech in conversations specifically for AHDs (Aria glasses or HAs). Systems submitted to the challenge were evaluated using objective metrics, but also by a set of subjective listening tests designed to assess the speech intelligibility and quality of their systems. This paper presents the challenge task and the final results from both objective and subjective evaluation.

While the challenge is now complete, the ECHI dataset is available for future research<sup>1</sup>, and the data obtained through the listening tests will be made public in the near future.

The rest of the paper is structured as follows: a brief overview of the provided data is given in Section 2, the challenge task, rules and baseline are described in Section 3, the submitted systems are covered in Section 4, the challenge results for objective and subjective assessment are presented in Section 5 and discussed in Section 6 and conclusions from the challenge are presented in Section 7.

## 2. DATA

Participants in the challenge were provided with the CHiME-9 ECHI dataset [3], which contains real recordings of four-party conversations in a noisy environment. The dataset consists of 29 hours of audio, recorded over 48 sessions, with a total of 190 unique participants, and contains noisy, multi-channel audio, close-talk (CT) microphone audio, reference audio (for NN training), and clean speech samples for each participant.

## 2.1. Recording Scenario

![](images/942464e00a9b41b33dfe314efd7d3a26db2c42080a548ea2970428d53b1c3f67.jpg)  
Fig. 1. The recording setup for the ECHI dataset. Note that this diagram is not drawn to scale.

Groups of four participants would be seated around a table in the centre of the room, surrounded by 18 loudspeakers playing background noise, as illustrated by Figure 1 (not to scale). Recording sessions lasted 36 minutes, with background noise playing at different levels throughout the session. In each session, one person would be wearing Aria glasses and another (always different from the Aria wearer) would be wearing a pair of HA shells.

The four loudspeakers in the corners of the room played back ambient noise sourced from the WHAM! dataset [4]. The remaining 14 loudspeakers surrounded the participants, and played back dynamic background noise constructed using speech from the LibriSpeech dataset [5] and the EARS dataset [6], and sound effects from FSD50K [7]. Only sound effects which could reasonably be heard in a cafeteria were included, e.g. coughing, typing and cutlery noises.

## 2.2. Materials

The noisy audio to be used as input for the algorithms is audio from Aria glasses and HA shells. The Aria glasses record 7-channel audio on the device, and the HAs record 4-channel audio through an external soundcard; the HAs are worn with one on each ear, so the 4-channel audio is made up of two channels on the left ear and two channels on the right. The Aria glasses and HAs are always worn by different participants in each session.

All conversation participants also wore a CT microphone, which captured speech close to the mouth of the wearer at a very high signal-to-noise ratio (SNR), recorded through the same soundcard as the HAs (i.e. sample-synchronous). Reference signals were generated by first applying a NN speech denoiser [8] to the CT audio, as this still contained some background noise and speech from other conversation partners, followed by a delay-compensation algorithm to account for delays due to time-of-flight and clock drift [9] between the Aria clock and the soundcard for the CT recordings.

Each conversation participant also recorded a reading of the first paragraph of the rainbow passage [10] in a quiet environment, as speech enhancement techniques can use a sample of the target speaker’s voice to help extract them from a noisy mixture, e.g. by creating speaker embeddings [11, 12].

Motion tracking data was also recorded, which gives the position and orientation of each participant in the conversation. This was used in the generation of the reference signals to estimate time-of-flight, but challenge participants were also able to use it during training of their systems.

Finally, voice activity detection (VAD) labels were provided for each conversation participant by applying a NN VAD algorithm to the CT microphones [13].

## 3. CHALLENGE TASK AND BASELINE

The challenge task was defined as a multi-speaker extraction (MSX) task, where the goal is to extract the speech from the conversation partners using the Aria/HA recordings and rainbow passages of the conversation partners. Two separate tracks were defined for the Aria glasses and HAs; submitted systems should only process the Aria audio or HA audio at once. Challenge participants were also expected to suppress the wearer’s speech, and could use the rainbow passage of the wearer to this end.

## 3.1. Data Subsets

The dataset was split into three subsets at the session level: the training subset contains 30 sessions (18 hours), the development subset contains 10 sessions (6 hours), and the evaluation set contains 8 sessions (4.8 hours). Each subset was disjoint with respect to the conversation participants and background noise, so no speakers in the training subset would appear in the development or evaluation subsets.

For the training and development sets, all materials described in Section 2.2 were provided to the challenge participants. For the evaluation set, only the noisy Aria and HA audio and the rainbow passages were made public, as these were the only materials which were supposed to be used by the model.

## 3.2. Challenge Rules

The full rules to the challenge can be found on the challenge website<sup>2</sup>, but a description of the key rules is given below.

## 3.2.1. Latency Requirements

Given that the fundamental goal of the challenge is to enhance speech using AHDs, there was a strict requirement for submitted systems to operate with a 20 ms theoretical latency. This means that any submitted system may not use any part of the signal > 20 ms in the future from the current time step, for example, when normalising the audio.

This latency is theoretical, as it does not account for any computation time. There is significant variation between the computational power of AHDs, so an accurate quantification of how quickly a model can run was not feasible. That said, teams were instructed to report model size and computational complexity in their reports.

## 3.2.2. External Resources

A whitelist of open-source external resources was provided to the challenge participants, ensuring that teams did not use private resources unavailable to other teams, potentially providing an advantage. Teams were able to request additions to the whitelist until 5 months before the challenge submission deadline.

## 3.3. Baseline

The baseline system for the challenge was a target speaker extraction (TSX) model with a TF-GridNet backbone [12, 11, 14]. This model only extracts one speaker at a time, so it is applied iteratively with each of the target rainbow passages to extract each speaker.

A system diagram for the TSX model is given in Figure 2. To ensure compliance with the latency rules, components were modified from the original TF-GridNet model [14, 12] to make them causal [11]. This primarily affected the short-time Fourier transform (STFT) and GridNet blocks, which use long-short term memory (LSTM) networks and self-attention in the time dimension. For the STFT, the window size was fixed to 8 ms, with a hop size of 4 ms. The LSTM blocks were fixed to be unidirectional in the time dimension, and masked self-attention was used to prevent look-ahead.

![](images/172f986905e65806594716bf1ee50fbedebcaa9623e3ab6ac5f447b970cf2b4a.jpg)  
Fig. 2. System diagram of the baseline TSX network.

The baseline system was trained only using the training subset of the challenge data, with no external resources. The model hyperparameters were tuned using the short-time objective intelligibility (STOI) scores [15] on the development subset. The full training parameters can be found in the challenge GitHub repository<sup>3</sup>.

## 4. SUBMITTED SYSTEMS

In total, the challenge received submissions from 7 different teams, with teams participating in the Aria track, HA track or both; all systems are listed in Table 1. A wide variety of techniques was employed, with some teams extending the provided baseline and others submitting solutions based on entirely different model architectures.

## 4.1. System Design

All but one team proposed TSX models to solve the task, with AHU-IEI [16] opting for an MSX system instead. Eleven [17] and Waseda-NTT [18] both introduced an own-voice suppression (OVS) block to the beginning of the system to remove the speech of the device wearer, with Eleven using the wearer’s rainbow passage for NNbased OVS, and Waseda-NTT estimating null-beamformer coefficients. AHU-IEI only used the target speaker’s rainbow passage as a training target, not using it as input to the model at all. Eleven also used the rainbow passages of the non-target conversation partners to better disambiguate the conversation partners from one another.

## 4.2. Reference Signals

While reference signals for each conversation partner were provided in the dataset [3], the AHU-IEI and Waseda-NTT teams decided to produce their own reference signals to better remove leaked background noise/cross-talk and model transmission effects more accurately. AHU-IEI achieved this by training a speech denoiser, using data constructed from the CHiME-9 ECHI CT recordings, while Waseda-NTT used a close-to-distant microphone projection, which uses beamforming to estimate the transmission effects from the CT microphone to the worn device.

## 4.3. Exclusion

During post-submission processing, one team reported generating extra training data with the CHiME-3 dataset [19]. Under challenge rules, this team could not be included in the final rankings as using non-whitelisted data may confer an advantage relative to compliant systems. However, the challenge organisers felt that the system was scientifically relevant in the wider context of speech enhancement for AHDs, and so retained the system in the subjective listening tests.

## 5. RESULTS

Systems were evaluated by objective measures of speech intelligibility and speech quality as well as subjective listening tests, presented in Table 1. The objective measures considered here are the commonly used frequency-weighted segmental SNR (fwSegSNR) [24], perceptual evaluation of speech quality (PESQ) [25], and the composite speech quality metrics (CSig, CBak and COvl) [26] to assess speech quality, as well as the STOI [15] to assess intelligibility. These metrics are computed with the Versa toolkit [27].

For subjective scoring, listening tests were conducted to assess speech intelligibility and speech quality. For each test, a listening panel of 32 native English speakers with self-reported normal hearing was recruited. Due to the limited capacity of listening tests, teams that submitted multiple systems to the challenge were asked for their preferred system to be evaluated subjectively. These teams were provided with their objective scores on the evaluation set to inform their decision. Systems not included in the subjective evaluation have empty rows in the corresponding columns of Table 1.

For speech intelligibility, listeners were played a short passage (∼ 8 s) of the conversation and asked to transcribe the speech of a target speaker from 5 s onwards (cued visually) under one-shot listening conditions. Systems were scored on correctness by computing the longest common subsequence between the listener responses and the ground truth, giving percentages in the range [0, 100] [28]. The longest common subsequence rating was chosen over word error rate as it does not penalise insertions; listeners could hear multiple talkers in the audio segments, and so they should not be penalised if they write down speech from the non-target speaker. Each system received 512 listener responses.

Speech quality was evaluated according to the ITU P.835 standard [29]. This involved listeners hearing 5 s segments of audio and rating on three scales: how natural the speech sounded (Sig), how intrusive the background noise was (Bak), and overall quality defined in the context of everyday speech communication (Ovr). These were rated on continuous scales (with range 1-5) with Likert-style verbal anchors. Each system received a minimum of 340 ratings.

Included in the subjective evaluation was an oracle system, which consisted of the summed CT microphone channels for each session. This was included to indicate what might be achievable under near-ideal speaker extraction and to help contextualise the performance of the submitted systems. For the objective evaluation of the CT microphones, metrics were computed using the reference signals without the delay-compensation applied, as this was only applied to account for the time-of-flight from the speaker to the device (Aria glasses or HAs), as indicated by italics in Table 1.

Table 1. The full results table for the CHiME-9 ECHI challenge. The <sup>†</sup> indicates the team excluded from the final challenge rankings (Section 4.3), and \* indicates the oracle system from the CT microphones, with the italic objective scores indicating the slightly different objective evaluation. Bold scores indicate the best score of the systems in the official ranking. The Subjective Score is the average of the Subjective Correctness and Subjective Ovr scores, with the latter linearly mapped to the same range as the correctness. Empty rows in the subjective columns indicate teams which were not evaluated in the listening tests. Standard error is provided for the Subjective Score.
<table><tr><td rowspan="2">Team</td><td rowspan="2"></td><td rowspan="2">System</td><td colspan="5">Objective Metrics</td><td rowspan="2">Subjective Correctness</td><td colspan="3">Subjective Quality</td><td rowspan="2">Subjective Score</td></tr><tr><td>STOI</td><td>FW-SegSNR</td><td>PESQ</td><td>CSig</td><td>CBak COvr</td><td>Sig</td><td>Bak</td><td>Ovr</td></tr><tr><td colspan="2">ECHI</td><td>Baseline</td><td>0.50</td><td>4.73</td><td>1.11</td><td>1.74</td><td>1.08</td><td>1.32</td><td>55.28</td><td>2.31 2.75</td><td>2.16</td><td>42.13±0.89</td></tr><tr><td rowspan="7"></td><td rowspan="3">AHU-IEI [16]</td><td>CloseTalk*</td><td>0.85</td><td>15.41</td><td>2.23</td><td>3.82</td><td>2.94 3.05</td><td>86.93</td><td rowspan="3"></td><td>4.57 3.82</td><td>4.31</td><td>84.8±0.61</td></tr><tr><td>v1-011</td><td>0.55</td><td>5.97</td><td>1.23 2.22</td><td>1.67</td><td>1.64</td><td></td><td></td><td></td><td></td></tr><tr><td>v2-001</td><td>0.54</td><td>5.62</td><td>1.22 2.22</td><td>1.61</td><td>1.63</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>v3-003</td><td>0.52</td><td>5.58</td><td>1.23 2.15</td><td>1.62</td><td>1.60</td><td>52.68</td><td>2.50</td><td>3.67</td><td>2.44</td><td>44.38±1.02</td></tr><tr><td>ASNTU [20]</td><td></td><td>0.50</td><td>5.48</td><td>1.18</td><td>2.20</td><td>1.56 1.60</td><td></td><td>51.86 48.34</td><td>1.61 4.11</td><td>1.84</td><td>36.41±0.90</td></tr><tr><td>Eleven [17]</td><td></td><td>0.45</td><td>0.80</td><td>1.08</td><td>1.61</td><td>1.06 1.22</td><td></td><td></td><td>3.56 1.81</td><td>2.28</td><td>40.17±0.95</td></tr><tr><td>MTEC [21]</td><td>Approach1</td><td>0.49 0.51</td><td>1.49</td><td>1.13</td><td>1.67</td><td>1.31</td><td>1.28</td><td></td><td></td><td></td><td></td></tr><tr><td>SFU-SpeechEnhancer [22]</td><td>Approach2</td><td>2.29</td><td>1.13</td><td>1.79</td><td>1.29</td><td>1.34</td><td>57.45 60.31</td><td>3.69</td><td>2.71</td><td>2.95</td><td>53.14±0.95</td></tr><tr><td rowspan="3"></td><td>RBF1_2x</td><td>0.53</td><td>2.88</td><td>1.17</td><td>1.81</td><td>1.12</td><td>1.38</td><td rowspan="3"></td><td>3.27</td><td>2.42</td><td>2.68</td><td>51.11±0.95</td></tr><tr><td rowspan="2">Waseda-NTT† [18]</td><td>0.58 0.58</td><td>1.47 4.37</td><td>1.30 1.24</td><td>1.76</td><td>1.31 1.23</td><td>1.43</td><td>63.54</td><td>3.74</td><td>2.85</td><td></td></tr><tr><td>RBF1_nw_3x RBF1_w_3x 0.57</td><td>2.96</td><td>1.25</td><td>2.09 2.00</td><td>1.22</td><td>1.55 1.51</td><td></td><td></td><td>3.14</td><td>58.49±0.89</td></tr><tr><td rowspan="10">HA</td><td>ECHI</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43.94</td><td></td><td></td><td></td></tr><tr><td>CloseTalk*</td><td>Baseline 0.50 0.87</td><td>4.42 17.77</td><td>1.11 2.32</td><td>1.90 3.92</td><td>1.08 3.01</td><td>1.39 3.15</td><td>85.34</td><td>2.37 4.54</td><td>2.53 3.82</td><td>2.21 4.26</td><td>37.15±0.90</td></tr><tr><td>v1-011</td><td></td><td>6.73</td><td>1.30</td><td></td><td></td><td></td><td></td><td>2.77</td><td></td><td></td><td>83.43±0.62</td></tr><tr><td>AHU-IEI [16] v2-001</td><td>0.61</td><td></td><td></td><td>2.53</td><td>1.70</td><td>1.84</td><td>59.02</td><td></td><td>3.58</td><td>2.75</td><td>51.35±0.98</td></tr><tr><td>v3-003</td><td>0.50</td><td>5.37 5.14</td><td>1.19</td><td>2.06</td><td>1.58</td><td>1.53</td><td>42.56</td><td>2.14</td><td></td><td></td><td></td></tr><tr><td></td><td>0.46</td><td>5.17</td><td>1.19 1.13</td><td>1.95 2.18</td><td>1.57 1.47</td><td>1.48 1.56</td><td></td><td>33.72 1.57</td><td>3.61</td><td>2.20</td><td>36.27±1.01</td></tr><tr><td>ASNTU [20]</td><td></td><td>0.47 5.12</td><td>1.17</td><td>2.11</td><td>1.39</td><td>1.53</td><td></td><td>45.24 2.09</td><td>3.57 2.95</td><td>1.70 2.16</td><td>25.57±0.86</td></tr><tr><td>CITISIN [23]</td><td>Approach1</td><td>0.53 0.52</td><td>1.88</td><td>1.11</td><td>1.80</td><td>1.11</td><td>1.33</td><td>44.54 4.30</td><td>1.96</td><td>2.79</td><td>37.10±0.95</td></tr><tr><td>MTEC [21]</td><td>Approach2</td><td>0.52</td><td>1.37</td><td>1.11</td><td>1.74</td><td>1.11</td><td>1.30</td><td></td><td></td><td></td><td>44.61±0.97</td></tr><tr><td>SFU-SpeechEnhancer [22]</td><td></td><td>0.52</td><td>2.64</td><td>1.13</td><td>1.80</td><td>1.13</td><td>1.34</td><td>43.18</td><td>3.48 2.15</td><td>2.62</td><td>41.89±0.93</td></tr></table>

## 6. DISCUSSION

All systems were able to make substantial improvements over the baseline systems in at least one of the subjective metrics, with different teams choosing to focus on different approaches to achieve the task. The Subjective Score is the average of the correctness score and the overall quality score (which was linearly mapped from its native range, [1, 5], to a percentage in the range [0, 100]).

For the HA track, the winner by Subjective Score was AHU-IEI v1-011, with a Wilcoxon test confirming their improvement over the next best system with strong evidence $( p < 0 . 0 1 )$ . The subjective metrics suggest that this system operated primarily as a denoiser; it scored strongly on the Bak ratings, and while this caused some distortion to the naturalness of the speech, it made a huge difference to the intelligibility compared to other teams, with the next best system 13.78% behind in the correctness score.

The winner of the Aria track is more tightly contested, with both MTEC-Approach2 and SFU-SpeechEnhancer performing well on the speech quality and speech intelligibility scores, respectively. A Wilcoxon test only shows weak evidence that MTEC-Approach2 outperforms SFU-SpeechEnhancer $( p < 0 . 1 0 )$ according to the Subjective Score, so the ranking cannot be concluded only from this score. However, for the correctness ratings, SFU-SpeechEnhancer’s improvements are also shown to have weak significance $( p < 0 . 1 0 )$ while on the Ovr quality score, there is strong evidence to suggest that MTEC-Approach2 produces a greater improvement than SFU-SpeechEnhancer $( p < 0 . 0 1 )$ . With this in mind, MTEC-Approach2 is considered the winner of the Aria track.

Interestingly, MTEC-Approach2 took the converse approach to AHU-IEI-v1-011 (the HA winners); their subjective ratings suggest that the background noise remained at an intrusive level, but the naturalness of the speech was preserved and intelligibility improved.

In general, there is very little agreement between the objective metrics and subjective results, with all objective metrics offering vastly different rankings from what is seen in the subjective results. However, it should be noted that reference signals for live recordings, such as the CHiME-9 ECHI data, can be difficult to perfect, and so the mismatch may be related to the reference signals, as well as the metrics not quite being appropriate for conversational speech. These factors were the core of the motivation for evaluating the systems with subjective listening tests.

It can also be seen that systems typically performed better on the Aria glasses than on the HAs. This is most likely due to the inherent differences between the form factors of the devices. The Aria glasses have a comparatively fixed array geometry (subject to the arms of the glasses flexing), whereas the HAs are worn on the ears, meaning the microphones are separated by human heads, which can vary greatly in size and shape. This variation can make it more difficult for multichannel NN-techniques to learn inter-channel differences, leading to degraded NN performance.

## 7. CONCLUSIONS

The CHiME-9 ECHI Challenge presents the task of enhancing the speech of conversation partners in noisy environments using recordings of real conversations in a simulated cafeteria-style environment. The challenge also evaluates submissions using both objective metrics and subjective listening tests of speech intelligibility and speech quality. Submissions varied greatly in how they approached the task, both from a technical and methodological approach, yielding improvements in all metrics over the challenge baseline, with the top systems making substantial gains in both the intelligibility and quality of the speech. These results are very encouraging for research into improving the experience of hearing-impaired listeners in social situations, and will hopefully promote future work in this field.

## 8. REFERENCES

[1] L. V. Hadley, W. M. Whitmer, W. O. Brimijoin, and G. Naylor, “Conversation in small groups: Speaking and listening strategies depend on the complexities of the environment and group,” Psychonomic bulletin & review, vol. 28, no. 2, pp. 632–640, 2021.

[2] L. V. Hadley and J. F. Culling, “Timing of head turns to upcoming talkers in triadic conversation: Evidence for prediction of turn ends and interruptions,” Frontiers in Psychology, vol. 13, pp. 1061582, 2022.

[3] R. Sutherland, J. Clarke, H. Elghazaly, T. Kuebert, M. Lugger, S. Petrausch, J. A. Ortiz, B. Xu, S. Goetze, and J. Barker, “Descriptor: Enhancing Conversations for the Hearing Impaired in the 9th Computational Hearing in Multisource Environments Challenge (CHiME9 ECHI),” IEEE Data Descriptions, vol. 3, pp. 73–81, 2026.

[4] G. Wichern, J. Antognini, M. Flynn, L. R. Zhu, E. McQuinn, D. Crow, E. Manilow, and J. Le Roux, “Wham!: Extending speech separation to noisy environments,” arXiv preprint arXiv:1907.01160, 2019.

[5] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an ASR corpus based on public domain audio books,” in 2015 IEEE ICASSP. IEEE, 2015, pp. 5206–5210.

[6] J. Richter, Y. C. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, and T. Gerkmann, “Ears: An anechoic fullband speech dataset benchmarked for speech enhancement and dereverberation,” arXiv preprint arXiv:2406.06185, 2024.

[7] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “FSD50k: an open dataset of human-labeled sound events,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829–852, 2021.

[8] A. Defossez, G. Synnaeve, and Y. Adi, “Real time speech enhancement in the waveform domain,” in Interspeech, 2020.

[9] A. Tessmer and N. Aschenbruck, “Characterization of an acoustic sensor network to improve clock synchronization strategies,” in 2025 IEEE LCN, 2025, pp. 1–9.

[10] G. Fairbanks, Voice and articulation drillbook, Harper & Row, New York, 2nd edition, 1960.

[11] S. Cornell, Z. Q. Wang, Y. Masuyama, S. Watanabe, M. Pariente, and N. Ono, “Multi-channel target speaker extraction with refinement: The WavLab submission to the Second Clarity Enhancement Challenge,” arXiv preprint arXiv:2302.07928, 2023.

[12] F. Hao, X. Li, and C. Zheng, “X-TF-GridNet: A time–frequency domain target speaker extraction network with adaptive speaker embedding fusion,” Information Fusion, vol. 112, pp. 102550, 2024.

[13] Silero Team, “Silero VAD: pre-trained enterprise-grade voice activity detector (VAD), number detector and language classifier,” https://github.com/snakers4/ silero-vad, 2024.

[14] Z. Q. Wang, S. Cornell, S. Choi, Y. Lee, B. Y. Kim, and S. Watanabe, “TF-GridNet: Integrating full-and sub-band modeling for speech separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 3221– 3236, 2023.

[15] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A short-time objective intelligibility measure for time-frequency weighted noisy speech,” in 2010 IEEE ICASSP. IEEE, 2010, pp. 4214–4217.

[16] Y. Tu, R. He, L. Xu, X. Wang, C. Sun, and Y. Fang, “A lowlatency multi-stage mimo system with cross-beam interaction for chime-9 task 2 (echi),” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[17] F. Zhao, C. Zhao, Z. Guo, W. Zhang, Y. Yan, and X. Zhang, “A three-stage system for chime-9 echi: Self-interference suppression, target speaker extraction, and post-processing,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[18] D. Hu, T. Nakatani, N. Kamo, M. Delcroix, T. Ochiai, and S. Makino, “Training low-latency target speech extraction for wearable devices using phase- and amplitude-aligned data in the chime-9 echi task,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[19] J. Barker, R. Marxer, E. Vincent, and S. Watanabe, “The third ‘chime’ speech separation and recognition challenge: Dataset, task and baselines,” in 2015 IEEE ASRU, 2015, pp. 504–511.

[20] R. Chao, Z. Jhou, Y. J. Li, and Y. Tsao, “A multichannel bandsplit recurrent architecture for real-time speaker-conditioned speech enhancement,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[21] P. Sharma, F. Campe, and D. Kolossa, “Perceptually motivated low-latency target speaker extraction for chime-9 echi,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[22] A. Haghbin and R. Vaughan, “Reinforcement learning for multi-channel speech enhancement,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[23] R. Chao, Z. Jhou, Y. J. Li, S. F. Huang, M. La Quatra, S. M. Siniscalchi, W. H. Cheng, X. W. Fu, and Y. Tsao, “A unified latency-flexible framework with a causal mamba core for multichannel speech enhancement,” in The Joint Workshop on HSCMA and CHiME 2026, 2026.

[24] J. Tribolet, P. Noll, B. McDermott, and R. Crochiere, “A study of complexity and quality of speech waveform coders,” in ICASSP ’78. IEEE ICASSP, 1978, vol. 3, pp. 586–590.

[25] A.W. Rix, J.G. Beerends, M.P. Hollier, and A.P. Hekstra, “Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE ICASSP, 2001, vol. 2, pp. 749–752 vol.2.

[26] Y. Hu and P. C. Loizou, “Evaluation of objective quality measures for speech enhancement,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 16, pp. 229–238, 2008.

[27] J. Shi, H. J. Shim, J. Tian, S. Arora, H. Wu, D. Petermann, J. Q. Yip, Y. Zhang, Y. Tang, W. Zhang, D. S. Alharthi, Y. Huang, K. Saito, J. Han, Y. Zhao, C. Donahue, and S. Watanabe, “VERSA: A versatile evaluation toolkit for speech, audio, and music,” in 2025 NAACL – System Demonstration Track, 2025.

[28] G. Roa-Dabike, T. J. Cox, J. Barker, B. M. Fazenda, S. Graetzer, R. R. Vos, M. A. Akeroyd, J. Firth, W. M. Whitmer, S. Bannister, et al., “The Cadenza Lyric Intelligibility Prediction (CLIP) Dataset.,” Data in Brief, p. 112466, 2026.

[29] ITU-T, “P.835: Subjective test methodology for evaluating speech communication systems that include noise suppression algorithms,” ITU-T recommendation, 2003.