# Fine-Tuning a Transducer for Impaired Speech: Where the Gains Appear and What They Mean for Evaluation

Kien Nguyen

Tuan Dinh

Bao Phan

kien@eloky.ai

Eloky tuanad121@gmail.com

bao@eloky.ai

## Abstract

Speech recognition still works poorly for many people whose speech is affected by conditions such as ALS, Parkinson’s disease, cerebral palsy, Down syndrome or stroke. We took part in the Speech Accessibility Project Challenge 2 (SAPC2) with a public speech recogniser that we fine-tuned on the challenge data. Our best entry scored 7.81% character error rate on Test1, the first of two hidden test sets (official baseline 15.26%). Our live entries ran on CPUs with a combined median latency of about 0.9 seconds at 7.96–8.07% error. On new speakers, fine-tuning reduced errors most for the speakers that an off-the-shelf recogniser finds hardest to understand. Sentences already seen in training had far lower error rates than new ones (2.58% vs 6.42%), a larger difference than being a familiar speaker made. We close with lessons for evaluating such systems.

## 1 Introduction

Voice interfaces are now everywhere, yet they make far more mistakes for people with speech impairments, who may need them most. A leaderboard tells us which system won, but not what a team building a product for these users needs to know. We use our SAPC2 entry to study four questions:

1. How much does fine-tuning a public model reduce errors, for which conditions and at which severity?

2. How do familiar speakers and familiar sentences relate to accuracy?

3. Can models of this kind serve both offline and live use on ordinary computers?

4. What general lessons did we learn about evaluating improvements?

We report character error rate (CER): the share of characters that are wrong after the official text normalisation. Lower is better. Tables also give word error rate (WER), the same measure counted in words.

Background. Early collections of impaired speech, such as UASpeech [Kim et al., 2008] and TORGO [Rudzicz et al., 2012], are small and mostly contain isolated words. Project Euphonia [Mac-Donald et al., 2021] and the Speech Accessibility Project (SAP) [Hasegawa-Johnson et al., 2024] collected speech from hundreds to thousands of people. SAP’s recordings come from people with Parkinson’s disease, ALS, cerebral palsy, Down syndrome and stroke, and its challenges [Zheng et al., 2025, SAPC2 organisers, 2026] measure accuracy on speakers who are not in the training data. SAPC2 has two tracks: Track 1 scores offline transcription, and Track 2 scores live (streaming) transcription on both accuracy and speed.

Table 1: Official Track 1 results on the hidden Test1 set (%). Results for the second test set were not released when this version was written.
<table><tr><td>System</td><td>Trained on selection speakers</td><td>CER</td><td>WER</td></tr><tr><td>Official challenge baseline (organisers)</td><td></td><td>15.26</td><td>21.53</td></tr><tr><td>Earlier entry</td><td>no</td><td>8.30</td><td>11.95</td></tr><tr><td>Best entry trained without the selection speakers</td><td>no</td><td>7.87</td><td>11.78</td></tr><tr><td>Best scored entry, trained on all labelled speakers</td><td>yes</td><td>7.81</td><td>11.69</td></tr></table>

## 2 Data and how we measured

Data. Each SAP recording is one prompt: a short command, a read sentence or an answer to a question. As the challenge allows [SAPC2 organisers, 2026], we trained on the challenge training set together with the training and development parts of the SAP research data release: about 950 hours from over 1,100 speakers. Our later entries also trained on the challenge’s labelled development speakers. We used no speech from outside SAP and no synthetic speech. The organisers’ published speaker counts for the hidden test sets (both combined) are ALS 66, cerebral palsy 58, Down syndrome 32, stroke 24 and Parkinson’s disease 13 [SAPC2 organisers, 2026].

Evaluation sets. We split the challenge’s labelled development set by speaker and used four sets:

• Selection panel: 41 speakers, 4,510 recordings, examined only through aggregate scores.

• Former sealed group: 83 speakers, kept sealed at first and later added to training.

• Held-back recordings of 300 training speakers, used only for the familiarity study (Section 4.3).

• Test1: the official hidden test set. Entries trained on all labelled speakers can only be judged here.

The severity and familiarity analyses use all labelled recordings of the 124 new speakers (47,929 recordings), including the many extra recordings of repeated prompts in the challenge’s development data, so most of these recordings are sentences that also appear in training. All scores use the official scoring tool. The ranges in our tables are 95% intervals from resampling speakers 2,000 times (“speaker intervals”), because a product meets new speakers, not just new recordings. We never accessed the second hidden test set and never adapted to test audio.

## 3 Our system in brief

We kept the system simple and built it from public parts.

• Starting model. We compared several public recognisers and chose Parakeet TDT 1.1B [NVIDIA, 2024, Rekesh et al., 2023, Xu et al., 2023], a 1.1-billion-parameter model whose decoder also predicts how long each token lasts (“TDT”).

• Fine-tuning. We fine-tuned the whole model on the impaired-speech data with NeMo [Kuchaiev et al., 2019].

• Averaging. We averaged the weights of fine-tuned models [Wortsman et al., 2022]. The result is one model, so this adds no cost when the system runs.

• Live use. Our live entries run fine-tuned models of the same kind on CPUs.

## 4 Results

## 4.1 Overall accuracy and accuracy by condition

Hidden test set. Table 1 shows our official Track 1 scores. Our best entry trained without the selection speakers has 7.87% CER, 48% fewer character errors than the official baseline. Our best scored entry, trained on all labelled speakers, has 7.81%.

By condition. Table 2 compares our averaged model with two recognisers that were not fine-tuned on this data: the original Parakeet model, and Whisper large-v3 [Radford et al., 2023, SYSTRAN,

Table 2: CER (%) on the selection panel (41 new speakers) by condition, with 95% speaker intervals. “Ours” is the model of our best entry trained without the selection speakers. <sup>†</sup>Only four speakers.
<table><tr><td>Condition</td><td></td><td></td><td>Speakers Recordings Ours (averaged) fine-tuned</td><td>Parakeet TDT 1.1B Whisper large-v3 before fine-tuning</td><td>off the shelf</td></tr><tr><td>Parkinson&#x27;s disease</td><td>11</td><td>1,356</td><td>4.76 [2.37, 7.47]</td><td>7.51 [3.21, 13.29]</td><td>8.00 [3.72, 13.18]</td></tr><tr><td>ALS</td><td>10</td><td>522</td><td>4.95 [2.02, 11.67]</td><td>8.60 [3.82, 19.38]</td><td>11.60 [6.32, 21.39]</td></tr><tr><td>Cerebral palsy</td><td>9</td><td>1,058</td><td>8.49 [5.01, 14.34]</td><td>20.37 [10.07, 40.11]</td><td>19.59 [11.88, 30.79]</td></tr><tr><td>Down syndrome</td><td>7</td><td>1,066</td><td>7.00 [5.01, 9.72]</td><td>14.73 [10.12, 19.74]</td><td>15.12 [10.46, 20.65]</td></tr><tr><td>Stroke†</td><td>4</td><td>508</td><td>8.56 [3.00, 17.21]</td><td>20.59 [7.44, 50.68]</td><td>20.76 [6.83, 41.66]</td></tr><tr><td>All</td><td>41</td><td>4,510</td><td>6.09 [4.52, 8.00]</td><td>12.16 [8.44, 16.75]</td><td>12.73 [9.44, 16.48]</td></tr></table>

Table 3: CER (%) of the 124 new speakers grouped by their CER under Whisper large-v3, with 95% speaker intervals. Groups are Whisper CER ranges in %. “Fine-tuned” is the analysis model; “Original” is Parakeet before fine-tuning; “Removed” is the share of its errors that fine-tuning removed. “New sentences only” keeps recordings whose text is not in the analysis model’s training data; groups are recomputed on those recordings, so speaker counts differ.
<table><tr><td></td><td></td><td colspan="2">New sentences only</td><td colspan="2"></td><td colspan="2">All recordings</td></tr><tr><td>Group (Whisper CER)</td><td></td><td>Spk. Fine-tuned</td><td>Original</td><td>Removed</td><td>Spk.</td><td>Fine-tuned</td><td>Original</td></tr><tr><td>Mild (&lt;5)</td><td></td><td>37 1.91 [1.53, 2.30]</td><td>2.38 [1.99, 2.77]</td><td>20%</td><td>41</td><td>1.22</td><td>2.76</td></tr><tr><td>Moderate (5–15)</td><td></td><td>384.04 [3.15, 5.05]</td><td>5.57 [4.46, 6.85]</td><td>27%</td><td>34</td><td>2.55</td><td>6.49</td></tr><tr><td>Severe (15–30)</td><td></td><td>23 10.89 [8.93, 13.37]</td><td>17.00 [11.96, 23.86]</td><td>36%</td><td>22</td><td>7.09</td><td>18.08</td></tr><tr><td>Very severe (≥30)</td><td></td><td>2630.62 [24.98, 37.46]</td><td>51.88 [41.62, 62.15]</td><td>41%</td><td>27</td><td>16.38</td><td>56.44</td></tr></table>

2023] as used in the challenge starter kit. Our column is the model of our best entry trained without the selection speakers. On the selection panel, it has about half the errors of either, and lower point estimates for every condition, although the intervals overlap for Parkinson’s disease, ALS and stroke. Parkinson’s disease and ALS have the lowest error rates; cerebral palsy, Down syndrome and stroke remain harder.

Test1 versus the selection panel. For that entry, Test1 CER (7.87%) was about 1.29 times its selection-panel CER. Reweighting the panel’s conditions by the published hidden-set speaker counts raises its CER from 6.09% to 6.79%, suggesting that the mix of conditions may explain part of the difference.

## 4.2 Accuracy by severity

We grouped the 124 new speakers by how well an off-the-shelf recogniser (Whisper) understands them, and scored every model on the same speakers and recordings. This grouping is not a clinical severity rating. We compare against the other recogniser, the original Parakeet model before finetuning, since grouping by Whisper’s own errors would bias Whisper’s numbers. “Fine-tuned” is the single fine-tuned analysis model of Section 4.3, not our best entry.

Table 3 leads with new sentences, the cleaner comparison. There, the share of errors removed by fine-tuning grows with severity: 20% in the mildest group, 27% and 36% in the middle groups, and 41% in the most severe, where CER falls from 51.9% to 30.6%. In the mildest group the intervals overlap. Over all recordings, which are mostly known sentences, the original model’s error rate rises from 2.8% to 56% across the groups, while the fine-tuned model’s rises from 1.2% to 16%.

## 4.3 Familiar speakers and familiar sentences

SAP prompts repeat across speakers, so a test recording may share its speaker, its sentence, or both with training data. To study this, we trained one analysis model with a single fine-tuning run while holding back some recordings of 300 training speakers. This gives four groups: (1) known speaker, known sentence; (2) known speaker, new sentence; (3) new speaker, known sentence; (4) new speaker, new sentence. Group 1 is further split into (1a) the same person said this sentence in training and (1b) only other people did. The new speakers are the 124 from the selection panel and the former sealed group. A sentence counts as known if its text, ignoring case and punctuation, appears in training. Each speaker can appear in several groups.

Table 4: CER (%) by group, with 95% speaker intervals. Groups are labelled speaker / sentence. Group 1 combines 1a (the same speaker said the sentence in training) and 1b (only other speakers did). “Fine-tuned” is the analysis model; “Parakeet” is Parakeet TDT 1.1B before fine-tuning; “Whisper” is Whisper large-v3 off the shelf. “Known” refers to the analysis model’s training data; for the other two models these are simply the same recordings.
<table><tr><td>Group (speaker / sentence)</td><td></td><td>Rec. Spk. Fine-tuned</td><td>(analysis)</td><td>Parakeet original</td><td>Whisper off the shelf</td></tr><tr><td>1: known / known</td><td>2,362</td><td></td><td>300 1.45 [1.05, 1.87]</td><td>13.87 [11.84, 16.11]</td><td>13.07 [11.07, 15.19]</td></tr><tr><td>1a: same speaker</td><td>1,105</td><td></td><td>2930.75 [0.47, 1.08]</td><td></td><td>12.90 [10.79, 15.29]12.29 [10.07, 14.72]</td></tr><tr><td>1b: other speakers only</td><td>1,257</td><td></td><td>3001.86 [1.38, 2.44]</td><td>14.45 [12.35, 16.77]</td><td>13.54 [11.51, 15.74]</td></tr><tr><td>2: known / new</td><td>1,500</td><td></td><td></td><td>3005.02 [4.34, 5.78] 9.32 [7.84, 11.01]</td><td>8.67 [7.50, 9.95]</td></tr><tr><td>3: new / known</td><td>37,281</td><td></td><td></td><td>124 2.58 [1.71, 3.60] 16.18 [12.68, 20.03]</td><td>15.20 [11.90, 18.86]</td></tr><tr><td>4: new / new</td><td>10,648</td><td></td><td></td><td>124 6.42 [5.26, 7.96] 9.89 [7.90, 12.52]</td><td>11.88 [9.77, 14.49]</td></tr></table>

Table 5: Our six scored Track 2 entries on Test1, as reported by the organisers: CER and the two median latencies in seconds. “Sum” adds the two medians.
<table><tr><td>Track 2 entry</td><td>CER (%)</td><td>First stable word (s)</td><td>Final text (s)</td><td>Sum (s)</td></tr><tr><td>Entry 1</td><td>8.05</td><td>7.35</td><td>1.20</td><td>8.56</td></tr><tr><td>Entry 2</td><td>7.98</td><td>0.82</td><td>1.51</td><td>2.34</td></tr><tr><td>Entry 3</td><td>8.02</td><td>0.41</td><td>0.53</td><td>0.94</td></tr><tr><td>Entry 4</td><td>7.96</td><td>0.41</td><td>0.52</td><td>0.93</td></tr><tr><td>Entry 5</td><td>7.97</td><td>0.42</td><td>0.50</td><td>0.92</td></tr><tr><td>Entry 6</td><td>8.07</td><td>0.42</td><td>0.44</td><td>0.85</td></tr></table>

Table 4 shows three observations.

• Familiar sentences were linked to a larger difference than familiar speakers. For new speakers, CER is 2.58% on known sentences against 6.42% on new ones. A known speaker saying a new sentence has 5.02%. Known sentences are not easy in themselves: the original Parakeet model does worse on them (16.18%) than on new ones (9.89%), so the low error comes from training on them. The groups also differ in speakers, prompt types and length, so this is an association, not a causal effect.

• A person’s own sentences had the lowest error rate. CER was 0.75% on held-back recordings of sentences the same speaker had said in training. For the 30 held-back speakers with the highest analysis-model CER, it was 5.84% on such sentences against 29.03% on new sentences; because these speakers were chosen by this model’s own errors, the contrast is exaggerated.

• Part of the gap is length. Known prompts are often short commands. After matching recording lengths, known sentences still have about 1.7 times fewer errors than new ones (4.89% vs 8.32%), down from about 2.5 times.

Evaluations that reuse training prompts may therefore show lower error rates than a system achieves on new text.

## 4.4 Live transcription on CPUs

Track 2 reports two latencies: the time until the first word appears and stays, and the time from the end of speech to the final text. Our live entries run fine-tuned models of the same kind as our offline entries, on CPUs without a GPU.

In the challenge’s CPU evaluation (Table 5), the sum of the two median latencies fell from 8.6 seconds for our first entry to between 0.85 and 0.93 seconds for the later ones. Live CER was 7.96–8.07%, close to our offline entries (7.81–8.30%). The fastest entry was also the least accurate, so speed and accuracy still trade off slightly.

## 4.5 Lessons for evaluating improvements

We tested many other changes. Most did not show consistent improvements, and four general lessons stand out.

• Count speakers, not recordings. On a 41-speaker panel, many measured gains were smaller than the speaker-level uncertainty, and those rarely held on other speakers or on Test1.

• Check the hardest speakers separately. Several changes that were neutral overall made results worse for slow or severely affected speakers.

• Fix the rules before looking. Setting pass/fail rules in advance, on panels separate from the selection panel, stopped us from adopting changes that only fit one panel.

• Development panels are optimistic. Test1 error was about 1.29 times selection-panel error, partly because a small panel has few of the hardest speakers.

## 5 Implications

• Fine-tuning substantially reduced character errors, most of all for the hardest-tounderstand speakers. Whether the remaining error rates are usable in practice is untested.

• Personal recordings may help. The low error rate on people’s own repeated sentences suggests that recordings from the user could be valuable, in line with earlier work on personalised models [Shor et al., 2019, Green et al., 2021, Tobin and Tomanek, 2022]. This is a hypothesis, not a result of this paper.

• Report accuracy in parts. Results by condition, by severity and by whether the sentence was known, with speaker-level uncertainty, are more informative than a single number.

## 6 Limitations and ethics

The selection panel has only 41 speakers (four with stroke), so per-condition intervals are wide. The severity and familiarity analyses use a single fine-tuned model, not our best entry. Severity is measured by an off-the-shelf recogniser’s errors, not by a clinical rating. Known-speaker test recordings come from the same sessions as that speaker’s training recordings, which may overstate the speaker effect. The hidden test sets report only overall scores, so we cannot break them down by condition.

All speech comes from the SAP corpus under its data use terms. We report only aggregate numbers: no audio, transcripts or individual recordings appear here. Recognition is still much less accurate for some people, especially with severe cerebral palsy, Down syndrome or stroke, so such systems should support, not replace, human communication partners, and should be evaluated with the people who will use them.

## 7 Conclusion

Fine-tuning a public speech recogniser and averaging fine-tuned models roughly halved errors on our 41-speaker selection panel and gave 7.81% CER on Test1, against 15.26% for the official baseline. In a separate single-model analysis, the share of errors removed was largest for the hardest-to-understand speakers. Live entries of the same kind ran on CPUs with about 0.9 seconds of combined median latency at similar accuracy. Having the sentence in training was linked to a larger drop in errors than having the speaker in training, which matters for how such systems are evaluated.

## References

Jordan R. Green, Robert L. MacDonald, Pan-Pan Jiang, Julie Cattiau, Rus Heywood, Richard Cave, Katie Seaver, Marilyn A. Ladewig, Jimmy Tobin, Michael P. Brenner, Philip C. Nelson, and Katrin Tomanek. Automatic speech recognition of disordered speech: Personalized models outperforming human listeners on short phrases. In Proc. Interspeech, 2021.

Mark Hasegawa-Johnson, Xiuwen Zheng, Heejin Kim, Clarion Mendes, et al. Community-supported shared infrastructure in support of speech accessibility. Journal of Speech, Language, and Hearing Research, 67(11): 4162–4175, 2024.

Heejin Kim, Mark Hasegawa-Johnson, Adrienne Perlman, Jon Gunderson, Thomas Huang, Kenneth Watkin, and Simone Frame. Dysarthric speech database for universal access research. In Proc. Interspeech, 2008.

Oleksii Kuchaiev, Jason Li, Huyen Nguyen, Oleksii Hrinchuk, Ryan Leary, Boris Ginsburg, Samuel Kriman, Stanislav Beliaev, Vitaly Lavrukhin, Jack Cook, Patrice Castonguay, Mariya Popova, Jocelyn Huang, and Jonathan M. Cohen. NeMo: a toolkit for building AI applications using neural modules. arXiv preprint arXiv:1909.09577, 2019.

Robert L. MacDonald, Pan-Pan Jiang, Julie Cattiau, Rus Heywood, Richard Cave, Katie Seaver, Marilyn A. Ladewig, Jimmy Tobin, Michael P. Brenner, Philip C. Nelson, Jordan R. Green, and Katrin Tomanek. Disordered speech data collection: Lessons learned at 1 million utterances from Project Euphonia. In Proc. Interspeech, 2021.

NVIDIA. parakeet-tdt-1.1b (model card). https://huggingface.co/nvidia/parakeet-tdt-1.1b, 2024.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proc. International Conference on Machine Learning (ICML), volume 202 of PMLR, 2023.

Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh Balam, and Boris Ginsburg. Fast conformer with linearly scalable attention for efficient speech recognition. In Proc. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2023.

Frank Rudzicz, Aravind Kumar Namasivayam, and Talya Wolff. The TORGO database of acoustic and articulatory speech from speakers with dysarthria. Language Resources and Evaluation, 46(4):523–541, 2012.

SAPC2 organisers. Speech accessibility project challenge 2 (SAPC2): challenge website, rules and leaderboard reports. https://xiuwenz2.github.io/SAPC2-website/, 2026. Accessed October 2026.

Joel Shor, Dotan Emanuel, Oran Lang, Omry Tuval, Michael Brenner, Julie Cattiau, Fernando Vieira, Maeve McNally, Taylor Charbonneau, Melissa Nollstadt, Avinatan Hassidim, and Yossi Matias. Personalizing ASR for dysarthric and accented speech with limited data. In Proc. Interspeech, pages 784–788, 2019.

SYSTRAN. faster-whisper: Faster Whisper transcription with CTranslate2. https://github.com/SYSTRAN/ faster-whisper, 2023.

Jimmy Tobin and Katrin Tomanek. Personalized automatic speech recognition trained on small disordered speech datasets. In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Proc. International Conference on Machine Learning (ICML), volume 162 of PMLR, 2022.

Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, and Boris Ginsburg. Efficient sequence transduction by jointly predicting tokens and durations. In Proc. International Conference on Machine Learning (ICML), volume 202 of PMLR, 2023.

Xiuwen Zheng et al. The Interspeech 2025 speech accessibility project challenge. In Proc. Interspeech, 2025. arXiv:2507.22047.