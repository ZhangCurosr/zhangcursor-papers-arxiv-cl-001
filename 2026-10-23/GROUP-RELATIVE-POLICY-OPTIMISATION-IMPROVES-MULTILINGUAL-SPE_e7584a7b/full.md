# GROUP RELATIVE POLICY OPTIMISATION IMPROVES MULTILINGUAL SPEECH RECOGNITION IN LOW-RESOURCE LANGUAGES

Tim Wenjie Hu, Evelyn Nafula Ouma, Benjamin Akera, John A. Quinn

Sunbird AI, Uganda

## ABSTRACT

Autoregressive encoder-decoder speech recognition models trained with teacher forcing can suffer from exposure bias, limiting their ability to recover from their own prediction errors at inference time. Recent work has shown that reinforcement learning with Group Relative Policy Optimisation (GRPO) improves speech recognition in major languages with LLM-based models. We extend this line of work to smaller, non-LLM encoder-decoder ASR models, using Whisper-large-v3 as a case study across 51 African languages. The benefits of this approach are most pronounced when training data is scarce or when input audio is noisy or degraded. Our resulting model, SunflowerASR, achieves the best published macro-averaged WER on the AfriVox-v2 (0.227) and SimbaBench (0.372) benchmarks, and is publicly available.

Index Terms— speech recognition, low-resource languages, reinforcement learning.

## 1. INTRODUCTION

Autoregressive speech recognition models such as Whisper [1] are typically trained only with supervised fine-tuning (SFT). This creates a risk of exposure bias: the model receives a training signal only for the case where it has been exactly correct about every preceding token, leaving it poorly prepared to recover from mistakes during inference.

Reinforcement learning methods provide an alternative, more representative of conditions at deployment time, where we generate entire rollouts from input audio sequences and then score each one as a whole. Recent work has shown that Group Relative Policy Optimisation (GRPO) [2], applied to a speech encoder attached to an Llama3 LLM decoder, can improve robustness on major European languages [3], and is effective at reducing insertion errors such as hallucinations and repetition glitches with another LLM-based model, Qwen2- Audio-7B, in English [4]. It has also been demonstrated that GRPO outperforms SFT in English when adapting a Llamabased ASR decoder using synthetic speech [5].

In this work, we investigate whether these benefits extend to smaller, non-LLM encoder-decoder architectures, and to languages with scarce training data. We take the case of Whisper-large-v3 [1] (1.55B parameters) and set up GRPO training with the objective of minimising character error rate (CER). We train a single model to transcribe 51 African languages, with per-language training data ranging from 5.2 hours (Ruruuli) to 1,392 hours (Kinyarwanda), and observe consistent improvements, which are most pronounced for noisy test audio and for languages with scarce training data. Our resulting model, SunflowerASR, achieves the best published average WER (per-language rankings vary) on the AfriVox-v2 and SimbaBench benchmarks, and can be accessed at https://hf.co/Sunbird/ SunflowerASR-51-african-languages.

## 2. DATASETS AND BASE MODEL

Training corpus The training corpus combines 110 dataset subsets covering 51 African languages, a mean of 2.2 independent sources per language. Sources include Common Voice [6], African Voices [7], Waxal [8] and other publicly available datasets. Full source and language level details are available in our Hugging Face release. After filtering it holds 3.82 M clips totalling 7 412 hours of speech. Training rows are filtered to remove suspected noisy labels, using an intermediate ASR model checkpoint, and then sample packed.

GRPO improvement in CER across augmentation levels  
![](images/01138aa74ec7d2fd26a1458f5e935b60428da7daac80f945c651808af69a1402.jpg)  
Fig. 1. CER before (supervised fine-tuning only) and after GRPO for Whisper-large-v3 on 51 African languages. The improvement becomes more pronounced at increasing levels of background noise.

Supervised fine-tuning We carry out full parameter fine tuning of the 1.55 B parameters of the Whisper-large-v3 model from OpenAI [1]. Training runs for 5 epochs over the packed corpus with a per-device batch of 64 packed samples and 4 gradient accumulation steps, an effective batch of 256 packed samples, or roughly 1.7 hours of audio per optimiser step, giving approximately 4 300 optimiser steps per epoch. Audio augmentation is applied on the fly, including additive noise drawn from a corpus of Ugandan ambient sound recordings, and modifications to speed and bandwidth. We use a peak learning rate of $2 \times 1 0 ^ { - 5 }$ with 100 warmup steps and cosine decay, and label smoothing of 0.1.

## 3. REINFORCEMENT LEARNING WITH GRPO

Supervised fine-tuning optimises the per-token likelihood of one reference transcript, so the model only ever receives a training signal for the case in which it has been exactly right about every preceding token. Group relative policy optimisation (GRPO) instead optimises the sequence-level metric we report. For a clip x with reference transcript $y ,$ the policy π<sub>θ</sub> samples a group of $G$ complete transcripts $\{ \hat { y } _ { 1 } , \dotsc , \hat { y } _ { G } \}$ , each is scored by a reward, and the group’s own mean is used as the baseline against which its members are ranked:

$$
r _ { i } = - \operatorname* { m i n } \big ( \mathrm { C E R } ( \hat { y } _ { i } , y ) , 1 \big ) , \qquad A _ { i } = \frac { r _ { i } - \mathrm { m e a n } ( \mathbf { r } ) } { \mathrm { s t d } ( \mathbf { r } ) } .\tag{1}
$$

Probability mass is then moved towards the rollouts with positive advantage, under a token-level normalised objective with a KL penalty of $\beta = 0 . 0 1$ against the initial policy.

Selecting examples with headroom In Equation 1, advantages are defined relative to the group, so a clip whose $G$ rollouts all receive the same reward produces $A _ { i } = 0$ for every member and contributes no gradient at all, however hard that clip is in absolute terms. Both failure modes are common in a 51-language corpus: clips the model already transcribes perfectly, and clips it fails on so completely that every rollout is equally wrong. GRPO can learn to make improvements for the clips on which the model is capable of a good transcript but does not reliably produce one.

To measure this directly, we use the SFT checkpoint in sampling mode over a random sample of the training data, generating 10 transcripts per clip. For each clip we then compute the headroom,

$$
g = \mathrm { m e d i a n } _ { i } \ : \mathrm { C E R } ( \hat { y } _ { i } , y ) \ : - \ : \underset { i } { \mathrm { m i n } } \ : \mathrm { C E R } ( \hat { y } _ { i } , y ) \ : ,\tag{2}
$$

the gap between the CER of a typical sample and that of the best sample in the group. A large g means a better transcript is already inside the model’s sampling distribution, and only needs to be made more probable. We keep the 200 clips with the largest $g$ from each training data subset, totalling 15, 285 clips. Section 4.1 ablates this choice against drawing the same number of clips per subset at random.

Training GRPO training is applied to the decoder of the model only, 906 M of the model’s 1.55B parameters. We generate $G = 1 0$ rollouts per training example, with 64 distinct clips per update for a batch size of 640 at peak learning rate $1 0 ^ { - 5 }$ . The batch is deliberately large for a run of this length. Because the training set is shuffled across all 51 languages, a batch of 64 clips draws on many languages at once, and a single update is therefore an average over them rather than a step that improves whichever language a small batch happened to contain. This is the main practical difference we found between the multilingual and the single-language case: with small batches, updates tend to improve one language at the expense of others. Section 4.1 ablates the batch size directly. The Whisper encoder is frozen and only the decoder is trained, on the hypothesis is that GRPO is most effective at correcting decoding errors such as spelling, word choice and orthographic convention.

## 4. RESULTS

We evaluate SunflowerASR, our final GRPO model, on three benchmarks, shown per language in Fig. 2 and Tables 1 and 2. The first is our own Sunbird Speech Benchmark [9], which we assemble from the held-out test splits of the 110 dataset subsets of the training corpus (Section 2), capped at 200 clips each. It scores 20, 213 clips over 51 languages drawn from 28 source corpora, and is the only one of the three that covers the full language range of the model. The second is AfriVox-$\mathbf { v } 2$ [10], an in-the-wild benchmark of 251 h aggregating the Africa Next Voices, Waxal and AfriVox v1 corpora over a 10-domain taxonomy spanning health, finance, government and agriculture; it is largely spontaneous rather than read, and we score 111, 051 clips in the 14 languages that have metrics reported in their paper. The third is SimbaBench [11], which aggregates read and conversational speech from Common Voice, BembaSpeech, Lwazi, NCHLT and regional collections. On all three benchmarks, our final model has superior accuracy to all published alternatives: see Figure 2 and Tables 1 and 2.

To evaluate the benefits of GRPO specifically, compared to the SFT-only initial checkpoint, we create three additional versions of 51-language evaluation set, with noise drawn from a corpus of Ugandan street recordings at three mean relative amplitudes, 0.5, 0.9 and 1.3, and also score the undegraded audio as a control. CER is consistently better after GRPO, and the improvement grows monotonically with the severity of the degradation, as shown in Fig. 1.

GRPO mainly repairs failures under acoustic stress, and the low-resource languages with the less supervised data gain the most. Fig. 3 illustrates each language’s CER improvement against the hours of supervised audio it contributed to the training corpus, for which there is a correlation of $r = - 0 . 4 1$ on $\log _ { 1 0 }$ hours, with 35 of 40 languages improving.

![](images/5bf2846805d4e842fdd61c7bb175fd8aeda111e7363f83e8dd769c02c51dca56.jpg)  
Fig. 2. WER by language on the Sunbird Speech Benchmark, comparing SunflowerASR against external baselines. Languages are sorted by the WER of SunflowerASR.

Table 1. WER by language on AfriVox-v2 for the 14 languages with published baselines, which report WER only. Simba-S is our own evaluation; it does not cover three of the languages, and its macro is over the other eleven.
<table><tr><td rowspan="2">ASR (ours)</td><td rowspan="2">Sunflower Sahara</td><td rowspan="2">v2</td><td rowspan="2">Gemini 3 Flash</td><td colspan="3">Omni-CTC</td><td rowspan="2">Simba S</td></tr><tr><td>7B</td><td>1B</td><td>300M</td></tr><tr><td>Language Swahili</td><td>0.108</td><td>0.071</td><td>0.076</td><td>0.077</td><td>0.097</td><td>0.152</td><td>0.150</td></tr><tr><td>Tswana</td><td>0.116</td><td>0.140</td><td>0.275</td><td>0.142</td><td>0.210</td><td>0.277</td><td>0.604</td></tr><tr><td>Kinyarwanda</td><td>0.133</td><td>0.066</td><td>0.165</td><td>0.104</td><td>0.138</td><td>0.215</td><td>0.591</td></tr><tr><td>Sotho</td><td>0.152</td><td>0.195</td><td>0.346</td><td>0.421</td><td>0.473</td><td>0.533</td><td>0.596</td></tr><tr><td>Luganda</td><td>0.160</td><td>0.393</td><td>0.315</td><td>0.422</td><td>0.428</td><td>0.483</td><td>0.341</td></tr><tr><td>Zulu</td><td>0.161</td><td>0.256</td><td>0.237</td><td>0.202</td><td>0.256</td><td>0.346</td><td>0.396</td></tr><tr><td>Xhosa</td><td>0.166</td><td>0.206</td><td>0.292</td><td>0.200</td><td>0.256</td><td>0.344</td><td>0.501</td></tr><tr><td>Shona</td><td>0.209</td><td>0.249</td><td>0.368</td><td>0.286</td><td>0.322</td><td>0.425</td><td></td></tr><tr><td>Amharic</td><td>0.251</td><td>0.253</td><td>0.249</td><td>0.327</td><td>0.374</td><td>0.490</td><td>0.716</td></tr><tr><td>Hausa</td><td>0.288</td><td>0.285</td><td>0.269</td><td>0.502</td><td>0.366</td><td>0.401</td><td>0.835</td></tr><tr><td>Yoruba</td><td>0.316</td><td>0.271</td><td>0.296</td><td>0.361</td><td>0.365</td><td>0.438</td><td>0.723</td></tr><tr><td>Fulani</td><td>0.343</td><td>0.352</td><td>0.729</td><td>0.557</td><td>0.569</td><td>0.588</td><td></td></tr><tr><td>Akan</td><td>0.347</td><td>0.307</td><td>0.456</td><td>0.447</td><td>0.497</td><td>0.549</td><td></td></tr><tr><td>Igbo</td><td>0.425</td><td>0.287</td><td>0.425</td><td>0.459</td><td>0.395</td><td>0.446</td><td>0.998</td></tr><tr><td>Overall</td><td>0.227</td><td>0.238</td><td>0.321</td><td>0.322</td><td>0.339</td><td>0.406</td><td>0.586</td></tr></table>

Table 2. WER by language on SimbaBench over the 19 languages in common. Baselines from the public leaderboard.
<table><tr><td>Language</td><td>SunflowerASR (Ours)</td><td>OmniASR LLM-7B</td><td>Simba S</td><td>Simba M</td><td>Gemini 2.5 Pro</td></tr><tr><td>Hausa</td><td>0.167</td><td>0.169</td><td>0.641</td><td>0.291</td><td>0.219</td></tr><tr><td>Luganda</td><td>0.220</td><td>0.171</td><td>0.227</td><td>0.357</td><td>0.421</td></tr><tr><td>Swahili</td><td>0.246</td><td>0.156</td><td>0.163</td><td>0.252</td><td>0.168</td></tr><tr><td>Kinyarwanda</td><td>0.251</td><td>0.192</td><td>0.531</td><td>0.367</td><td>0.463</td></tr><tr><td>Luo</td><td>0.297</td><td>0.271</td><td>0.391</td><td>0.442</td><td>0.494</td></tr><tr><td>Igbo</td><td>0.304</td><td>0.362</td><td>0.777</td><td>0.850</td><td>0.428</td></tr><tr><td>Afrikaans</td><td>0.312</td><td>0.279</td><td>0.145</td><td>0.313</td><td>0.158</td></tr><tr><td>Wolof</td><td>0.324</td><td>0.583</td><td>0.581</td><td>0.586</td><td>1.830</td></tr><tr><td>Yoruba</td><td>0.335</td><td>0.232</td><td>0.201</td><td>0.415</td><td>0.524</td></tr><tr><td>Kabyle</td><td>0.349</td><td>0.287</td><td>0.604</td><td>0.643</td><td>0.933</td></tr><tr><td>Tswana</td><td>0.353</td><td>0.489</td><td>0.187</td><td>0.306</td><td>0.464</td></tr><tr><td>Bemba</td><td>0.374</td><td>0.392</td><td>0.391</td><td>0.437</td><td>0.635</td></tr><tr><td>Sotho</td><td>0.441</td><td>0.634</td><td>0.210</td><td>0.327</td><td>0.495</td></tr><tr><td>Ndebele</td><td>0.444</td><td>0.588</td><td>0.246</td><td>0.358</td><td>0.413</td></tr><tr><td>Zulu</td><td>0.459</td><td>0.401</td><td>0.585</td><td>0.333</td><td>0.359</td></tr><tr><td>Xhosa</td><td>0.466</td><td>0.448</td><td>0.272</td><td>0.383</td><td>0.466</td></tr><tr><td>Chichewa</td><td>0.545</td><td>0.609</td><td>0.223</td><td>0.468</td><td>0.428</td></tr><tr><td>Kalenjin</td><td>0.580</td><td>0.700</td><td>0.713</td><td>0.734</td><td>0.959</td></tr><tr><td>Amharic</td><td>0.600</td><td>0.234</td><td>0.330</td><td>0.525</td><td>0.457</td></tr><tr><td>Overall</td><td>0.372</td><td>0.379</td><td>0.390</td><td>0.442</td><td>0.543</td></tr></table>

Practical deployment audio Table 3 reports the effect of GRPO on Sunbird’s salt-practical-eval set, a 60-clip validation set assembled to reflect real deployment. The recordings were collected in Uganda and cover Ugandan English (28 clips) and Luganda (32 clips) across three deployment settings: citizen feedback gathered outside hospitals and government offices, conversations recorded at the call centre of a bank, and phone-in contributions to an interactive radio programme. The audio is spontaneous and conversational, captured in noisy public spaces, with overlapping turns, hesitation and backchannel markers, and frequent code-switching from Luganda into English for domain vocabulary and numerals. GRPO improves both metrics in both languages more significantly than for benchmark test data: the aggregate reduction is 6.2% relative on WER and 9.7% relative on CER.

## 4.1. Ablations

Three ablations isolate the contribution of the reinforcement objective itself, of the clip selection of Section 3, and of the batch size of Section 3; results are in Table 4.

GRPO against further supervised training on the same clips. GRPO adds an extra epoch of training over a curated 15, 285-clip subset, so its gain could in principle come from that extra pass rather than from the reinforcement signal. We test this by continuing supervised fine-tuning from the same checkpoint for one further epoch over exactly the clips GRPO was given. This regresses the model on both WER and CER (Table 4). The same data is thus useful under a reinforcement objective and harmful under a supervised one.

Selecting examples with headroom. Replacing the headroom ranking of Equation 2 with 200 clips drawn uniformly at random from each subset, everything else held fixed, gives WER 0.4220 and CER 0.1714. Random selection still improves on the supervised checkpoint, but it recovers only about two thirds of the CER gain, −3.9% against −5.8% relative.

![](images/ad94439119f64984c103af81ae0a539e6ee06d7b4d48e2fe36a5e95e4f7d9efe.jpg)  
Fig. 3. Per-language CER improvement from GRPO against the number of hours of supervised training data for that language.

Table 3. Effect of GRPO on the Sunbird salt-practical-eval set, by deployment scenario and language.
<table><tr><td></td><td></td><td colspan="3">WER</td><td colspan="3">CER</td></tr><tr><td>Scenario</td><td></td><td></td><td>n Before After</td><td></td><td>∆ Before After</td><td></td><td> $\Delta$ </td></tr><tr><td>Ugandan English</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Citizen feedback 12</td><td></td><td></td><td>0.3240.282</td><td>-13.0%</td><td></td><td></td><td>0.223 0.182 -18.0%</td></tr><tr><td>Call centre</td><td>16</td><td></td><td>0.302 0.273</td><td>-9.4%</td><td></td><td></td><td>0.213 0.182 -14.8%</td></tr><tr><td>Luganda</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Citizen feedback</td><td>8</td><td></td><td>0.705 0.683</td><td>-3.0%</td><td>0.3240.317</td><td></td><td>-2.2%</td></tr><tr><td>Radio phone-in</td><td>15</td><td></td><td>0.535 0.528</td><td>-1.4%</td><td></td><td>0.1400.132</td><td>-5.4%</td></tr><tr><td>Call centre</td><td>9</td><td></td><td>0.602 0.565</td><td>-6.2%</td><td></td><td>0.272 0.265</td><td>-2.5%</td></tr><tr><td>All</td><td>60</td><td></td><td>0.4230.397</td><td>-6.2%</td><td>0.215 0.194</td><td></td><td>-9.7%</td></tr></table>

Table 4. Ablations on the noise-1.3 set (20, 347 rows), with all variants starting from the same SFT checkpoint.
<table><tr><td>Variant</td><td>WER</td><td> $\Delta$ </td><td>CER</td><td> $\Delta$ </td></tr><tr><td>SFT checkpoint (initialisation)</td><td>0.4279</td><td></td><td>0.1783</td><td></td></tr><tr><td>+ 1 further SFT epoch</td><td>0.4794</td><td>+12.0%</td><td>0.2089</td><td>+17.2%</td></tr><tr><td>GRPO, random clips per subset</td><td>0.4220</td><td>-1.4%</td><td>0.1714</td><td>-3.9%</td></tr><tr><td>GRPO, 16 clips per batch</td><td>0.4243</td><td>-0.8%</td><td>0.1689</td><td>-5.3%</td></tr><tr><td>GRPO, full (SunflowerASR)</td><td>0.4201</td><td>-1.8%</td><td>0.1680</td><td>-5.8%</td></tr></table>

Batch size. Reducing the batch from 64 distinct clips to 16, at the same G = 10 and over the same selected clips, gives WER 0.4243 and CER 0.1689. The regression is smaller than for the other two ablations and is clearer on WER (−0.8% against −1.8% relative) than on CER, but its direction matches the argument of Section 3: with 51 languages shuffled together, 16 clips is a thin sample of the language set, and an update is correspondingly likely to benefit only the languages that batch happened to contain.

Table 5. Substitution quality on the 20, 124 rows where neither model loops.
<table><tr><td>Measure</td><td>SFT</td><td>GRPO</td></tr><tr><td>Word substitutions</td><td>114,755</td><td>115,272</td></tr><tr><td>Character substitutions</td><td>165,096</td><td>160,220</td></tr><tr><td>Mean char. edit distance per substitution</td><td>3.437</td><td>3.385</td></tr><tr><td>Substitutions within ≤ 1 edit</td><td>26.7%</td><td>27.4%</td></tr><tr><td>Substitutions within  $\leq :$  2 edits</td><td>44.9%</td><td>45.8%</td></tr><tr><td>Total char. cost of substitutions</td><td>394,368</td><td>390,231</td></tr><tr><td>Long words cut to  $\leq$  half their length 1</td><td>18.8%</td><td>18.0%</td></tr></table>

Table 6. Cases where the supervised model heard a fragment or an unrelated word and GRPO heard a near-miss spelling.
<table><tr><td>Lang.</td><td>Reference</td><td>SFT heard</td><td>GRPO heard</td><td>Edits</td></tr><tr><td>Oromo</td><td>libiraal</td><td>liibiraaldimookiraasiitti</td><td>liibiraal</td><td>17 →1</td></tr><tr><td>Rukonjo</td><td>ngerikulhakulhania</td><td>ebyo</td><td>erikulhakulhania</td><td>17 → 2</td></tr><tr><td>Shona</td><td>vobva</td><td>rwakachenerukira</td><td>wobva</td><td>15→1</td></tr><tr><td>Luganda</td><td>bibili</td><td>ebitakyalabirwaako</td><td>bibiri</td><td>14→ 1</td></tr><tr><td>Ruruuli</td><td>ebyankulakulana</td><td>mbe</td><td>ebyangulagulana</td><td>14→ 2</td></tr><tr><td>Kanuri</td><td>kənasartiyende</td><td>də</td><td>kənasartiyande</td><td>13 → 1</td></tr><tr><td>Zulu</td><td>amabhayisikili</td><td>i</td><td>mubhayisikili</td><td>13 → 2</td></tr><tr><td>Rutooro</td><td>baali</td><td>batarukwegesaana</td><td>baaba</td><td>13 → 2</td></tr><tr><td>Somali</td><td>dheellitirka</td><td>wax</td><td>dheelitirka</td><td>12→1</td></tr><tr><td>Kwamba</td><td>ndia</td><td>ndibhakazwisikilangu</td><td>ndiabho</td><td>16→ 3</td></tr></table>

## 4.2. Characterisation of improvements

To understand the nature of the improvements, we aligned the 20, 347 hypotheses of the noise-1.3 set against the references, and observed two different mechanisms. The first is the suppression of degenerate decoding. Under acoustic stress the supervised model sometimes falls into a repetition loop and emits hundreds of words for a short utterance. There are 223 rows (1.1% of the evaluation) for which the SFT-only model becomes stuck in a loop. GRPO repairs these decoding failures, and this accounts for the WER gain.

On the remaining 20, 124 rows the two models are wrong about equally often; word substitutions are flat at +0.5%, yet character substitutions fall by 3.0%. The characteristic supervised failure is to replace a word either with a fragment or with an unrelated long compound, while GRPO returns a near-miss spelling that tracks the acoustics through to the end of the word (Table 6).

## 5. CONCLUSION

We have shown that GRPO transfers from LLM-decoder systems to a smaller non-LLM encoder-decoder model, in a multilingual, low-resource and noisy setting. The benefits increase for languages with fewer data resources (e.g. tens of hours of audio), and in test time conditions with challenging audio. Selection of training examples with headroom and appropriate batch sizing are helpful to attain improvements in accuracy. This therefore appears to be a widely generalisable post-training method for autoregressive ASR models.

## 6. REFERENCES

[1] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning. 2023, vol. 202 of Proceedings of Machine Learning Research, pp. 28492–28518, PMLR.

[2] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo, “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” 2024.

[3] Prashanth Gurunath Shivakumar, Yile Gu, Ankur Gandhe, and Ivan Bulyko, “Group relative policy optimization for speech recognition,” 2025.

[4] Tingzhen Xiong, Rilin Chen, Weiwei Li, Wentao Zhang, and Qicong Xie, “Grounded in sound: Reinforcement learning with a frozen acoustic judge to curb ASR insertion hallucinations,” 2026.

[5] Shashi Kumar, Yanis Labrak, Hasindri Watawana, Sergio Burdisso, Esau Villatoro-Tello, Kadri Hacio´ glu, Petr˘ Motlicek, and Andreas Stolcke, “When synthetic speech is all you have: Better call GRPO,” arXiv preprint arXiv:2607.08409, 2026.

[6] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis M. Tyers, and Gregor Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the 12th Language Resources and Evaluation Conference (LREC), 2020, pp. 4218– 4222.

[7] Perez Ogayo, Graham Neubig, and Alan W Black, “Building African voices,” in Proc. Interspeech, 2022, pp. 1263–1267.

[8] Abdoulaye Diack, Perry Nelson, Kwaku Agbesi, Angela Nakalembe, MohamedElfatih MohamedKhair, Vusumuzi Dube, et al., “WAXAL: A large-scale multilingual African language speech corpus,” 2026.

[9] Sunbird AI, “Sunbird speech benchmark: a multilingual asr test set for 51 african languages,” https://huggingface.co/datasets/ Sunbird/speech-benchmark, 2026.

[10] Busayo Awobade, Gabrial Zencha Ashungafac, and Tobi Olatunji, “Afrivox-v2: A domain-verticalized benchmark for in-the-wild african speech recognition,” 2026.

[11] AbdelRahim A. Elmadany, Sang Yun Kwon, Hawau Olamide Toyin, Alcides Alcoba Inciarte, Hanan Aldarmaki, and Muhammad Abdul-Mageed, “Voice of a continent: Mapping Africa’s speech technology frontier,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, Eds., Suzhou, China, Nov. 2025, pp. 11039–11061, Association for Computational Linguistics.