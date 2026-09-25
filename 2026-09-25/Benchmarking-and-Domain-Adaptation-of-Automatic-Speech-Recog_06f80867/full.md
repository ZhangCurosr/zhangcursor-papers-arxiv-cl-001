# Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages

Stephen E. Moore<sup>\*,1,2</sup>, Akwasi Asare<sup>2</sup>, Mich-Seth Owusu<sup>2</sup>, Paul Azunre<sup>2</sup>, Joel Budu<sup>2</sup>, Lawrence A. Adu-Gyamfi<sup>2</sup>

<sup>1</sup>Department of Mathematics, University of Cape Coast, Ghana

<sup>2</sup>Ghana Natural Language Processing, Cape Coast, Ghana

Corresponding Author: stephen.moore@ucc.edu.gh

## ABSTRACT

This paper presents an end-to-end study of automatic speech recognition (ASR) for adolescent health communication in three Ghanaian languages (Twi, Dagbani, and Ewe). The work proceeds in three connected stages; First, we benchmark five ASR systems (three language-specific Wav2Vec2 models and two multimodal LLMs, Gemma 3n and Gemma 4) on a general-domain Bible corpus and a Youth Adolescent Sexual and Reproductive Health (ASRH) Domain ASR dataset, using Character and Word Error Rate (CER, WER). Second, guided by the benchmark, we perform supervised domain adaptation: although Gemma 4 was the strongest zero-shot candidate, fine-tuning it proved computationally infeasible, so we pivoted to the compact Qwen3-ASR-0.6B, fine-tuned on a large Ghana Bible corpus (\~90k samples) and evaluated strictly on held-out human-collected in-domain audio. Fine-tuning reduced WER on every language, most dramatically for Ewe (WER from 109.3% to 64.8%, a drop of 44.5 pp; CER from 65.1% to 24.9%). Third, we validate the work through KasaHealth, a live voice-first ASRH application deployed in all three languages, complemented by Senti-Check, a technical evaluation harness. KasaHealth was tested by 50 community respondents and achieved a 100% chat-approval rate, a 72% Good-or-Excellent translation rating, and a 92% would-recommend rate, while surfacing the domain gaps that most constrain real-world use. Across all three stages the evidence converges: for these languages the binding constraint is validated in-domain data, not model capability or computation.

Keywords: ASR; low-resource languages; Twi; Dagbani; Ewe; CER; WER; ASRH; Wav2Vec2;   
Gemma; Qwen3-ASR; domain adaptation; user acceptance testing.

## 1. INTRODUCTION

Access to health information in one’s native language is foundational to meaningful participation in health services, particularly for adolescents in multilingual communities [1]. In Ghana, where over 80 languages are spoken, digital services have historically targeted English-speaking, literate populations, leaving millions of indigenous-language speakers without adequate access [2,3]. Three languages anchor this study: Twi, an Akan dialect cluster and the most widely spoken first language and national lingua franca; Dagbani, the principal language of the Northern Region, spoken by several million people; and Ewe, spoken across the Volta Region and into Togo and Benin [4–6]. Despite their importance, these languages remain severely underrepresented in NLP and commercial speech technology, so voice interfaces, especially valuable where literacy is lower or stigma limits text-based communication, are not reliably available to their speakers. ASRH is an especially sensitive setting: accurate transcription of health terminology directly affects whether young people and health workers can trust a voice interface. This paper reports a complete arc from measurement to deployment. We benchmark available models across a general domain (Bible speech) and the target ASRH domain; use that benchmark to select and execute a domain-adaptation strategy, documenting a pivot from a multimodal LLM to a compact deployable ASR model; and validate the result with real users through the KasaHealth application, characterising the remaining gaps. All datasets, model weights, and tooling are released open-source. Our contributions are: (i) a five-model, two-domain ASR benchmark for Twi, Dagbani, and Ewe introducing Gemma 4 as a newly competitive zero-shot baseline; (ii) a documented adaptation pathway from an infeasible Gemma fine-tune to a tractable Qwen3-ASR-0.6B fine-tune on modest (T4) hardware with strict train/evaluation domain separation; (iii) a user-validated deployment (KasaHealth) plus a technical harness (Senti-Check); and (iv) a fully open-source, reusable pipeline that other West and Central African language communities can adopt.

## 2. BACKGROUND AND RELATED WORK

ASR for low-resource languages: Progress has historically been constrained by scarce annotated audio, complex tonal phonology and rich morphology, and the absence of standardised orthographies [7]; traditional systems needed hundreds to thousands of hours of transcribed speech [8,9]. Self-supervised learning changed this: wav2vec 2.0 [10] learns acoustic representations from unlabeled audio and can be fine-tuned on as little as ten minutes of labeled speech, and its multilingual XLSR variant [11] enables cross-lingual transfer. In health settings the bar is higher. Afonja et al. found that although general WER was often low, errors on medical entities were substantially higher, with domain fine-tuning improving medical WER by 25–34% relative [12], while Blocker et al. reported isiXhosa CER of 43–51% in primary care even where English approached human transcription [13]. General benchmarks are therefore insufficient proxies for health-domain performance. Ghanaian-language NLP and zero-shot LLMs: GhanaNLP has produced publicly available Wav2Vec2 ASR models and datasets for Twi, Dagbani, and Ewe [2,4,5,14–16] exposed through the Khaya AI initiative. Multimodal LLMs offer zero-shot transcription without task-specific training: Gemma 3n performed poorly across all three languages [17], whereas Gemma 4 (evaluated here for the first time) is substantially upgraded. As the adaptation results below show, however, competitiveness as a zero-shot baseline does not imply tractability as a fine-tuning target under realistic compute budgets.

The ASRH domain gap: Domain mismatch raises error rates through unfamiliar vocabulary, speaking styles, and prosody [18]. ASRH is a particularly demanding domain: reproductive anatomy, contraception, STIs, and adolescent development require precise terminology rarely present in Bible-derived corpora, and local-language discussion involves socially constructed euphemisms that diverge from formal medical vocabulary. This structural gap motivates the adaptation and deployment stages that follow.

## 3. METHODOLOGY

## 3.1 Datasets

Two benchmarking domains were used, both public on the Hugging Face Hub under the Ghana NLP organization: Bible datasets [14–16] and Youth ASRH Domain ASR datasets [19–21]. The Youth ASRH

ASR datasets (ghananlpcommunity/UNICEF-Ghana-{Twi,Dagbani,Ewe}-ASR) pair audio with transcriptions drawn from ASRH-relevant materials; the Bible datasets (asante-twi-bible-speech-text, dagbani- and ewe-bible-audio-text-tts) provide broader general-domain vocabulary and a reference domain closer to the training distribution. A synthetic ASRH-domain text dataset grounded in UNICEF Ghana’s adolescent counselling manual, covering ASRH, Child Protection, Education, and Environmental Concerns in English and all three languages, was also generated and released [22]. Benchmarking uses 50-sample subsets per dataset per language, up from a preliminary 30-sample evaluation, improving statistical reliability.

## 3.2 Models and Benchmarking Methodology

Five models were benchmarked. dagbani\_wav2vec2 and ewe\_wav2vec2 are Wav2Vec2-BERT (w2v-bert-2.0) models fine-tuned on their languages [10,23,24] ; twi\_w2v\_bert adds a BERT language-model decoding component for improved word-level accuracy [23,25]. gemma-3n-E2B-it [17] and gemma-4-E2B-it [26] are instruction-tuned multimodal LLMs evaluated fully zero-shot. CER and WER [27–29] were computed with jiwer as the mean of per-sample scores; a WER above 100% indicates hallucination or poorly calibrated decoding. Audio was resampled to 16 kHz; references and hypotheses were lowercased, and whitespace stripped. Wav2Vec2-BERT models used greedy CTC decoding (no beam search, no external LM, batch 16); Gemma models ran via Unsloth FastModel in bfloat16 with greedy decoding and up to 200 new tokens, using a minimal transcription prompt with no language specification or few-shot examples. Wav2Vec2 models were evaluated on native and both non-native languages; Gemma on all six conditions.

![](images/c516bd017f1c4bed6457eecc8b0035f091401b1f3e00dc388cd94f980c66d5fe.jpg)  
3.3 Audio Collection and the Pivot from Gemma to Qwen3-ASR

A Python offline GUI recorder [30], designed for low-connectivity settings, collected transcribed audio from community participants: \~30 minutes per language in Phase 1 (the 50-sample benchmarking set) and up to two hours per language in Phase 2, reserved strictly as the held-out evaluation set. Gemma 4 E2B was initially selected as the fine-tuning target based on its cross-dataset stability (Section 4.1), but its multimodal architecture and parameter count exceeded the available GPU resources. The work therefore adopted Qwen3-ASR-0.6B instead, a compact multilingual model whose smaller footprint made supervised fine-tuning tractable on T4 hardware and whose language-prefix output format enables simultaneous transcription and language identification. Training data and evaluation data were deliberately separated: rather than mixing synthetic TTS audio with human recordings, the model was trained on a large out-of-domain Ghana Bible corpus, with the human-collected Phase-2 audio reserved strictly for evaluation, so that reported metrics reflect generalisation to the adolescent domain rather than training overlap. Table 1 summarises the fine-tuning configuration; training loss fell from approximately 150 to below 1.0 and evaluation loss from 0.32 to 0.09, indicating convergence without overfitting.

Table 1. Qwen3-ASR-0.6B fine-tuning configuration.
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Training dataset</td><td rowspan=1 colspan=1>ghananlpcommunity/ghana-bible-combined-90k-twi-ewe-dagbani(~90k sâmples; 98/2 train/eval)</td></tr><tr><td rowspan=1 colspan=1>Base / released model</td><td rowspan=1 colspan=1>Qwen/Qwen3-ASR-0.6B →ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani</td></tr><tr><td rowspan=1 colspan=1>Epochs / batch / LR</td><td rowspan=1 colspan=1>~3.2 (to convergence) / effective 32 (4×16) / 2e-5, warmup +cosine decay</td></tr><tr><td rowspan=1 colspan=1>Final eval losshardware</td><td rowspan=1 colspan=1>~0.09 / T4 GPU (Google Colab)</td></tr><tr><td rowspan=1 colspan=1>Evaluation set</td><td rowspan=1 colspan=1>Human-collected in-domain audio (up to 2 h/language, Phase 2)</td></tr></table>

## 4. RESULTS AND DISCUSSION

This section reports the three stages of evaluation in turn: the cross-model benchmark (Section 4.1), the domain-adaptation fine-tuning (Section 4.2), and the user acceptance testing (Section 4.3). Section 4.4 synthesises the implications, and Section 4.5 lists the open-source resources released with the work.

## 4.1 Benchmarking

Table 2 reports CER and WER for all five models across both datasets, and Figure 1 and Figure 2 summarise the WER and CER results respectively.

Table 2. CER and WER for all benchmarked models (50 samples per dataset). Wav2Vec2 rows report native-language evaluation; Gemma rows are per language. Lower is better.
<table><tr><td colspan="1" rowspan="1">Model</td><td colspan="1" rowspan="1">Lang</td><td colspan="1" rowspan="1">Dataset</td><td colspan="1" rowspan="1">CER%</td><td colspan="1" rowspan="1">WER%</td><td colspan="1" rowspan="1">Model</td><td colspan="1" rowspan="1">Lang</td><td colspan="1" rowspan="1">Dataset</td><td colspan="1" rowspan="1">CER%</td><td colspan="1" rowspan="1">WER%</td></tr><tr><td colspan="1" rowspan="1">dagbani w2v</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">25.2</td><td colspan="1" rowspan="1">74.6</td><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">199.1</td><td colspan="1" rowspan="1">215.6</td></tr><tr><td colspan="1" rowspan="1">dagbani_w2v</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">19.5</td><td colspan="1" rowspan="1">56.4</td><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">59.7</td><td colspan="1" rowspan="1">114.4</td></tr><tr><td colspan="1" rowspan="1">ewe_w2v</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">17.9</td><td colspan="1" rowspan="1">60.8</td><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Twi</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">326.2</td><td colspan="1" rowspan="1">337.4</td></tr><tr><td colspan="1" rowspan="1">ewe_w2v</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">18.9</td><td colspan="1" rowspan="1">64.0</td><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Twi</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">75.6</td><td colspan="1" rowspan="1">132.3</td></tr><tr><td colspan="1" rowspan="1">twi_w2v_bert</td><td colspan="1" rowspan="1">Twi</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">17.0</td><td colspan="1" rowspan="1">53.6</td><td colspan="1" rowspan="1">gemma-4</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">51.2</td><td colspan="1" rowspan="1">98.4</td></tr><tr><td colspan="1" rowspan="1">twi_w2v_bert</td><td colspan="1" rowspan="1">Twi</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">20.9</td><td colspan="1" rowspan="1">62.2</td><td colspan="1" rowspan="1">gemma-4</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">55.4</td><td colspan="1" rowspan="1">96.8</td></tr><tr><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">240.3</td><td colspan="1" rowspan="1">344.0</td><td colspan="1" rowspan="1">gemma-4</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">Bible</td><td colspan="1" rowspan="1">47.2</td><td colspan="1" rowspan="1">98.8</td></tr><tr><td colspan="1" rowspan="1">gemma-3n</td><td colspan="1" rowspan="1">Dag</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">93.0</td><td colspan="1" rowspan="1">141.1</td><td colspan="1" rowspan="1">gemma-4</td><td colspan="1" rowspan="1">Ewe</td><td colspan="1" rowspan="1">YouthASRHASR</td><td colspan="1" rowspan="1">50.1</td><td colspan="1" rowspan="1">98.3</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">gemma-4</td><td colspan="1" rowspan="1">Twi</td><td colspan="1" rowspan="1">B/ASRH</td><td colspan="1" rowspan="1">49.2/44.3</td><td colspan="1" rowspan="1">92.5/92.6</td></tr></table>

Figure 1. WER by model: Bible vs Youth ASRHASR.

WER — Bible vs Youth ASRH ASR per Model (Wav2Vec2 averaged over native language only)  
![](images/e975d9a490ba5faf81994dcca6375b52eeb5ec9cb53939efba4b9e9dbb953994.jpg)  
CER — Bible vs Youth ASRH ASR per Model (Wav2Vec2 averaged over native language only)

![](images/fe3b641b7961511e7dc38733f2931e9e4e2631145ceea15fb519552fb2a2621c.jpg)  
Figure 2. CER by model: Bible vs Youth ASRH ASR. (Wav2Vec2 values averaged over native conditions; Gemma across all three languages.)

## 4.1.1 Model-by-Model Analysis

Wav2Vec2 language-specific models: The three fine-tuned models perform strongly on their native languages. twi\_w2v\_bert records the best CER overall (17.0% Bible, 20.9% Youth ASRH); ewe\_wav2vec2 achieves 17.9%/18.9% CER, a notably small \~1 pp domain gap although its WER gap is larger at 3.2 pp; and dagbani\_wav2vec2 records 25.2%/19.5% CER, with the Youth ASRH set producing a lower CER than Bible, a reversal suggesting the Dagbani model is better calibrated to the acoustic characteristics of the health recordings. On non-native languages all three degrade sharply (CER 45–65%, WER consistently >95%), confirming that fine-tuning confers language-specific rather than broadly transferable capability.

Gemma 3n: persistent limitations: Gemma 3n exhibits very high error rates across all conditions, with CER on Bible from 199.1% (Ewe) to 326.2% (Twi) and WER from 215.6% (Ewe) to 344.0% (Dagbani), systematic hallucination in which the model generates text unrelated to the reference. Performance is markedly better on Youth ASRH data (CER 59.7% Ewe, 75.6% Twi, 93.0% Dagbani), consistent with the preliminary 30-sample results and likely reflecting the constrained vocabulary and topic distribution of the health recordings relative to the diverse register of biblical text [17], [25]. Given the extreme absolute error rates and the hallucination observed on Bible data, these lower values do not indicate genuine ASR competence.

Gemma 4: a competitive zero-shot baseline: Gemma 4 is the most notable result of the benchmark. Across all six conditions it achieves CER 44.3–55.4% and WER 92.5–98.8%, substantially and consistently better than Gemma 3n (a CER reduction exceeding 100 pp on Bible and 15–35 pp on Youth ASRH). Its WER is directly competitive with the fine-tuned Wav2Vec2 models (which reach 53.6–74.6% on Bible and 56.4–64.0% on Youth ASRH) despite receiving no language-specific training; the Wav2Vec2 models retain a clear CER advantage, but the WER convergence indicates Gemma 4 produces more coherent word-level output than its CER alone would imply. A notable feature is its stability: consistent CER and WER across languages and datasets with a domain gap of at most a few percentage points, reflecting multilingual pretraining. This cross-language and cross-domain stability motivated its initial selection as the fine-tuning target. Figure 3 presents the full CER matrix across all model and dataset combinations, including cross-lingual conditions.

![](images/ac35b35a8a51812f508949a1584d1650e70a9ab47fe3e035c381ae991c0e896c.jpg)  
Figure 3. CER heatmap across all model–dataset combinations. Blue-bordered cells are native pairings; Gemma 3n values >120% are capped for legibility.

![](images/47ea01cfd2c5663f41272fc7661bf749375194baa2494c064ba6661f2be7d2b6.jpg)  
Qwen3-ASR-0.6B: Base vs Fine-tuned — Ghana Adolescent Health Domain

The domain gap as a structural challenge: Degradation on Youth ASRH data relative to Bible data is inherent to the current state of low-resource ASR here. The Bible corpus is the largest transcribed source for these languages, so Wav2Vec2 models inherit its vocabulary and acoustic profile and encounter distribution shifts elsewhere. The gap extends even to Gemma 4, indicating general multimodal pretraining has not closed the distributional gap between religious speech and health communication, motivating the adaptation stage.

## 4.2 Domain Adaptation: Fine-Tuning

The fine-tuned Qwen3-ASR-0.6B model was evaluated against its base version on held-out human-collected Youth ASRH-domain audio. Because the Bible training corpus and the in-domain evaluation set are disjoint in domain, these results measure generalization rather than memorization. Figure 4 compares base and fine-tuned WER and CER, and Table 3 reports the corresponding values.

Figure 4. WER (left) and CER (right) for the base and fine-tuned Qwen3-ASR-0.6B on held-out human-collected adolescent-domain audio.  
Table 3. Base and fine-tuned (FT) Qwen3-ASR-0.6B on held-out in-domain audio. A negative Δ indicates improvement.
<table><tr><td rowspan=1 colspan=1>Language</td><td rowspan=1 colspan=1>WERBase</td><td rowspan=1 colspan=1>WER FT</td><td rowspan=1 colspan=1>WERΔ</td><td rowspan=1 colspan=1>CERBase</td><td rowspan=1 colspan=1>CER FT</td><td rowspan=1 colspan=1>CERΔ</td></tr><tr><td rowspan=1 colspan=1>Twi</td><td rowspan=1 colspan=1>106.6%</td><td rowspan=1 colspan=1>95.1%</td><td rowspan=1 colspan=1>-11.5 pp</td><td rowspan=1 colspan=1>73.0%</td><td rowspan=1 colspan=1>60.2%</td><td rowspan=1 colspan=1>-12.8 pp</td></tr><tr><td rowspan=1 colspan=1>Dagbani</td><td rowspan=1 colspan=1>99.1%</td><td rowspan=1 colspan=1>88.1%</td><td rowspan=1 colspan=1>-11.0 pp</td><td rowspan=1 colspan=1>64.8%</td><td rowspan=1 colspan=1>51.9%</td><td rowspan=1 colspan=1>-12.9 pp</td></tr><tr><td rowspan=1 colspan=1>Ewe</td><td rowspan=1 colspan=1>109.3%</td><td rowspan=1 colspan=1>64.8%</td><td rowspan=1 colspan=1>-44.5 pp</td><td rowspan=1 colspan=1>65.1%</td><td rowspan=1 colspan=1>24.9%</td><td rowspan=1 colspan=1>-40.2 pp</td></tr></table>

Fine-tuning improves every language on both metrics. Ewe shows the most dramatic gain, with WER dropping 44.5 pp (from 109.3% to 64.8%) and CER dropping 40.2 pp (from 65.1% to 24.9%), while Twi and Dagbani show consistent \~11 pp WER improvements. The largest gains accrue to Ewe, the language with the weakest base performance, indicating that adaptation on out-of-domain data yields the greatest benefit for the languages furthest from the base model’s training distribution. Residual error reflects both the inherent difficulty of low-resource ASR and the domain gap between the Bible training corpus and the adolescent-health evaluation set. Because the model improves despite training only on out-of-domain data, its in-domain performance is bounded chiefly by the quantity of in-domain material seen during training, which here was none. These results therefore represent a conservative lower bound, and a modest volume of in-domain training audio would be expected to improve them substantially.

## 4.3 User Acceptance Testing

Beyond error-rate improvement, a deployed system must be usable, trusted, and helpful to its intended users. Validation therefore combined two complementary applications. KasaHealth is a production, user-facing application evaluated by community respondents, and additionally serves as a reference for how a strong production pipeline performs in ASRH without domain-specific adaptation. Senti-Check is a technical harness that isolates the effect of the fine-tune. Because KasaHealth runs Ghana NLP’s production Khaya pipeline rather than the fine-tuned model, the two are complementary rather than directly comparable.

## 4.3.1 KasaHealth: Voice-First ASRHApplication

KasaHealth [31] is a voice-first health information application by GhanaNLP (Khaya AI model powered) delivering ASRH information in Twi, Dagbani, and Ewe through an ASR, LLM, and TTS pipeline built on production Khaya models. It most directly embodies the project’s core objective of putting voice-based health information into young people’s hands in their own language and is the flagship demonstration of the work. Before entry, every user was shown a disclaimer identifying the platform as an AI testing tool rather than a medical service, directing serious concerns to SHEplus Ghana, and confirming that sharing personal information was voluntary. Figure 5 shows the application interface.

![](images/d6c247ff9939fc4999b2776dabe6d6d7c264302216a764426eb452b40e982333.jpg)  
Figure 5. The KasaHealth interface (Twi). Users select language and voice; the main panel offers voice/text input ("Hold to speak"), quick health topics (puberty, contraception, STIs & HIV, pregnancy, menstruation, consent & safety), and a chat with transcription,framed as "Safe · Private · Free".

Testing ran 22–26 May 2026 with 50 respondents (24M, 26F), generating 107 items of correspondence (50 feedback-form submissions, 57 in-app ratings); 26 female respondents represent the primary intended user group. Of original testers, 92% would recommend the application and 75% rated answers Helpful or Very Helpful, with none rating them unhelpful. Figure 6 summarises component-level quality ratings across all 50 respondents.

KasaHealth: AI Component Quality Ratings (n = 50)  
![](images/40b6b68338556f08ed12250a14592313b4097a91315365cddf736e7f5d09c4ac.jpg)  
Figure 6. KasaHealth AI component quality (n = 50). Translation received zero Poor ratings.

a) ASR/transcript: 70% Good/Excellent (35/50). Translation: 72% Good/Excellent (36/50), zero Poor, the strongest component. TTS: 58% Good/Excellent (29/50), main issue Twi vowel-elongation artefacts.

b) Female respondents (n=26): 88% rated translation and 80% rated ASR Good/Excellent, the strongest of any subgroup, and encouraging for the primary audience.

c) In-app (57 interactions): all 26 health-question interactions received a thumbs-up (100% chat approval; Twi 11/11, Dagbani 14/14, Ewe 1/1). Of 29 flagged translation-review interactions, reason codes were dominated by bad translation (18, 62%), then incomplete answer (5), wrong language (2), other (2), wrong information (1). Ewe drew the most negative feedback (1 up/16 down), consistent with its residual benchmarking difficulty.

d) Key issues: Twi TTS vowel elongation ("anaa"→"anaaaaa"); a language-switching session-state bug; and ASRH vocabulary gaps where English health terms lack local equivalents, the signature of a domain gap, directly motivating targeted in-domain data collection.

## 4.3.2 Senti-Check: Technical Validation

Senti-Check [32] evaluates base and fine-tuned Qwen3-ASR-0.6B via a blind emotion-classification proxy: if a transcription preserves meaning, evaluators should identify the correct emotional tone from the transcript alone. Both models transcribe the same audio; transcripts receive one of 24 emotion labels; and six native-speaker testers (two per language, 20 samples each = 120 judgements/model) select the perceived emotion blind to model identity. As shown in Figure 7, the fine-tuned model doubled base accuracy (16/120, 13.3% vs 8/120, 6.7%). Absolute accuracy is low, as expected for a 24-class blind task, but the relative gain is consistent: strongest in Twi (from 3 to 9 of 40), with Dagbani improving (from 2 to 4 of 40) and Ewe stable (3 of 40), pending an upstream word-boundary fix. Table 4 contrasts the design and outcomes of the two applications.

Senti-Check: Blind Emotion-Labelling Accuracy  
![](images/44027cbf1bff95d27f0b0e371663ef2a87e49296e5560c5b49352d73a04ad41e.jpg)  
Figure 7. Senti-Check blind emotion-labelling accuracy: fine-tuned 16/120 vs base 8/120 (2×).

Table 4. KasaHealth and Senti-Check: two complementary evaluation applications.
<table><tr><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1>KasaHealth (primary)</td><td rowspan=1 colspan=1>Senti-Check (validation)</td></tr><tr><td rowspan=1 colspan=1>Tests</td><td rowspan=1 colspan=1>Is the voice health appusable/helpful?</td><td rowspan=1 colspan=1>Does fine-tuning improvetranscription quality?</td></tr><tr><td rowspan=1 colspan=1>Method / users</td><td rowspan=1 colspan=1>Survey + thumbs / 50 community(24M,26F)</td><td rowspan=1 colspan=1>Blind emotion (24 labels) / 6testers</td></tr><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>Production Khaya pipeline</td><td rowspan=1 colspan=1>Base vs fine-tunedQwen3-ASR-0.6B</td></tr><tr><td rowspan=1 colspan=1>Key result</td><td rowspan=1 colspan=1>100% chat approval; 72%translation Good+</td><td rowspan=1 colspan=1>Fine-tuned 2× base (16/120 vs8/120)</td></tr></table>

## 4.4 Discussion

The benchmarking, adaptation, and validation results converge on a common implication: the principal opportunity for improvement lies in data, followed by models, and only then compute. The limiting factor throughout was not model capability or compute but the availability of validated in-domain training data. The model was trained on a large but out-of-domain corpus of approximately 90,000 Bible samples because it was the largest high-quality transcribed resource available, and the recurring KasaHealth issues, namely ASRH vocabulary gaps and domain-specific translation errors, are characteristic of a domain gap rather than a fundamental model limitation.

Data requirements: Three kinds are most valuable. First, in-domain spoken audio for ASR: conversational health questions and counselling-style dialogue paired with human-verified transcriptions. Second, parallel health-terminology text for the translation and LLM components: English-to-local pairs covering ASRH terms, many of which have no standard local equivalent and which are the chief source of the translation errors observed in KasaHealth. Third, targeted natural-speech recordings for TTS, to address artefacts such as the Twi vowel elongation. In all cases the data should be speaker-diverse across gender, age, dialect, and recording conditions, mirroring the recorder methodology already validated here. As an indicative target, moving in-domain transcribed audio from the current \~2 hours per language (evaluation-only) toward the order of tens of hours per language available for training is a reasonable next milestone, though this is a direction, not a fixed prescription. Critically, well-validated in-domain data delivers gains out of proportion to its size relative to large out-of-domain corpora: a smaller, carefully validated in-domain set will usually outperform a larger, noisier one.

On models, KasaHealth shows that strong production-grade local-language models already deliver a usable, trusted end-to-end experience; the path is to domain-adapt efficient, deployable models (the Qwen3-ASR route) and apply targeted engineering fixes, not to switch to larger architectures. Compute is secondary and non-binding: the fine-tune was tractable on modest T4 hardware and would become a constraint only if substantially larger or multimodal architectures were pursued, which the evidence does not warrant. The principal practical constraint is time: transcription and, above all, native-speaker validation take considerably longer than recording. Allocating a dedicated window for collection and validation is therefore essential, as under-resourcing validation tends to yield a larger but noisier dataset. The reusable open-source pipeline released here, comprising the recorder, datasets, fine-tuning recipe, evaluation harness (Senti-Check), and reference application (KasaHealth), establishes that voice-first ASRH delivery in local languages is feasible and well received, de-risks future community-facing investment, and provides a lasting public good for African NLP.

## 4.5 Open-Source Release

All datasets, model weights, and tooling produced in this work are publicly released; Table 5 lists the resources and their locations.

Table 5. Open-source release inventory.
<table><tr><td rowspan=1 colspan=1>Resource</td><td rowspan=1 colspan=1>Link</td></tr><tr><td rowspan=1 colspan=1>Youth-domain text; YouthASRH ASR audio datasets</td><td rowspan=1 colspan=1>huggingface.co/collections/ghananlpcommunity/{youth-domain-datasets, unicef-asr-datasets}</td></tr><tr><td rowspan=1 colspan=1>Ghana Bible corpus (~90k)fine-tuned model weights</td><td rowspan=1 colspan=1>huggingface.co/{datasets/ghananlpcommunity/ghana-bible-combined-90k.ghananlpcommunity/qwen3-asr-0.6b-ghana...}</td></tr><tr><td rowspan=1 colspan=1>Recorder / Senti-Check /KasaHealth (live)</td><td rowspan=1 colspan=1>github.com/GhanaNLP/{recorder, senti-check}kasa-health-frontend.onrender.com</td></tr></table>

## 5. CONCLUSION

This study reports an end-to-end investigation of ASR for adolescent health communication in Twi, Dagbani, and Ewe, spanning benchmarking, domain adaptation, and user validation. The benchmark confirms a clear character-level advantage for language-specific Wav2Vec2 models (CER 17–25%) and identifies Gemma 4 as a competitive zero-shot baseline at the word level, while documenting a persistent domain gap across all models. Following an infeasible Gemma fine-tune, the compact Qwen3-ASR-0.6B was fine-tuned on a large out-of-domain corpus and evaluated on strictly held-out in-domain audio, reducing WER for every language and by 44.5 pp for Ewe. KasaHealth demonstrates that a production local-language system is usable and trusted by community users (100% chat approval, 72% Good-or-Excellent translation, 92% would recommend), and Senti-Check shows that the fine-tune doubles downstream semantic accuracy. The results consistently identify validated in-domain data as the binding constraint. The open-source pipeline released with this work provides a foundation for that investment and for adoption by other African language communities.

Funding. The project was funded by UNICEF Office of Innovation (WCARO) with Contract number: 43456330 having duration from 19.March.2026 to 18.June.2026.

Acknowledgements. Carried out by the Ghana NLP Community with the UNICEF Office of Innovation (WCARO) and UNICEF Ghana. We thank the community volunteers and native-speaker testers who contributed recordings, transcriptions, and evaluations; Jonathan Asiamah, Issah Abdul Haqq Niendoo, Humphrey Donkor, Emmanuel Adu Saah, Priscilla N. Lartey, Livingstone Eli Ayivor, Datsomor Gerhardt Kwame, Lucas Woedem Kpatah, Mukson Ibrahim, Ibrahim Abdul-Halim, Sualey Naporo Alhassan, Alhassan Amin Naporo, Jida Asare, Franklyn Armah, Agartha Enyonam Aziedor, Benedicta Tutu, Alhassan Fuseini Naporo

## REFERENCES

1. Kwame, A. (2025). Achieving universal healthcare coverage in a multilingual care setting: Linguistic diversity and language use barriers as social determinants of care in Ghana. Qualitative Health Research, 35(10–11), 1219–1230. https://doi.org/10.1177/10497323241298886

2. Gyamfi, L. A., Azunre, P., Moore, S. E., Budu, J., Asare, A., Owusu, M.-S., et al. (2026). GhanaNLP parallel corpora: Comprehensive multilingual resourcesfor low-resource Ghanaian languages. arXiv. https://doi.org/10.48550/arXiv.2603.13793

3. Nekoto, W., Marivate, V., Matsila, T., Fasubaa, T., Fagbohungbe, T., Akinola, S. O., et al. (2020). Participatory research for low-resourced machine translation: A case study in African languages. In Findings of the Association for Computational Linguistics: EMNLP 2020 (pp. 2144–2160). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.195

4. Azunre, P., Osei, S., Addo, S., Adu-Gyamfi, L. A., Moore, S. E., Adabankah, B., et al. (2021). NLPfor Ghanaian languages. https://arxiv.org/abs/2103.15475v2

5. Azunre, P., Osei, S., Addo, S., Adu-Gyamfi, L. A., Moore, S. E., Adabankah, B., et al. (2021, April 1). English-Twi parallel corpus for machine translation.

6. Agyekum, K. (2010). Language shift: A case study of Ghana. Sociolinguistic Studies, 3(3), 381–403. https://doi.org/10.1558/sols.v3i3.381

7. Hafiz, I. S., Sani, B., Ketema, G. D., Yimam, A. B., Said, A. I., Abdulmumin, I., et al. (2025). Automatic speech recognitionfor African low-resource languages: Challenges andfuture directions. https://arxiv.org/abs/2505.11690v1

8. Besacier, L., Barnard, E., Karpov, A., & Schultz, T. (2014). Automatic speech recognition for under-resourced languages: A survey. Speech Communication, 56, 85–100. https://doi.org/10.1016/j.specom.2013.07.008

9. Akera, B., Nafula, E., Walukagga, P., Yiga, G., Quinn, J., & Mwebaze, E. (2025, October 8). How much speech data is necessaryfor ASR in African languages? An evaluation ofdata scaling in Kinyarwanda and Kikuyu. arXiv. http://arxiv.org/abs/2510.07221

10. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020, October 22). wav2vec 2.0: Aframework for self-supervised learning ofspeech representations. arXiv. http://arxiv.org/abs/2006.11477

11. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., et al. (2020). Unsupervised cross-lingual representation learning at scale. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics (pp. 8440–8451). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.747

12. Afonja, T., Olatunji, T., Ogun, S., Etori, N. A., Owodunni, A., & Yekini, M. (2024). Performant ASR models for medical entities in accented speech. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH) (pp. 2315–2319). International Speech Communication Association. https://doi.org/10.21437/Interspeech.2024-2261

13. Blocker, A., Meyer, F., Biyabani, A., Mwangama, J., Datay, M. I., & Malila, B. (2025). Benchmarking IsiXhosa automatic speech recognition and machine translation for digital health provision. In S. Ananiadou, D. Demner-Fushman, D. Gupta, & P. Thompson (Eds.), Proceedings of the Second Workshop on Patient-Oriented Language Processing (CL4Health) (pp. 169–179). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.cl4health-1.14

14. GhanaNLP Community (2026). Asante Twi Bible audio-text TTS. Hugging Face. Retrieved April 9, 2026, https://huggingface.co/datasets/ghananlpcommunity/asante-twi-bible-speech-text

15. GhanaNLP Community. (2026). Ewe Bible audio-text TTS [Data set]. Hugging Face. Retrieved April 9, 2026, from https://huggingface.co/datasets/ghananlpcommunity/ewe-bible-audio-text-tts

16. GhanaNLP Community. (2026). Dagbani Bible audio-text TTS [Data set]. Hugging Face. https://huggingface.co/datasets/ghananlpcommunity/dagbani-bible-audio-text-tts

17. Gemma Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., et al. (2025). Gemma 3 technical report. arXiv. http://arxiv.org/abs/2503.19786

18. Hsu, W.-N., Sriram, A., Baevski, A., Likhomanenko, T., Xu, Q., Pratap, V., et al. (2021). Robus wav2vec 2.0: Analyzing domain shift in self-supervisedpre-training. arXiv. http://arxiv.org/abs/2104.01027

19. GhanaNLP Community (2026). Youth ASRHASR-Ghana-Dagbani-ASR [Data set]. Hugging Face. Retrieved April 9, 2026, https://huggingface.co/datasets/ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani

20. GhanaNLP Community (2026). Youth ASRHASR-Ghana-Twi-ASR [Data set]. Hugging Face. Retrieved April 9, 2026, https://huggingface.co/datasets/ghananlpcommunity/ghana-nlp-health-UNICEF-asr-twi

21. GhanaNLP Community (2026). Youth ASRHASR-Ghana-Ewe-ASR [Data set]. Hugging Face. Retrieved April 9, 2026, https://huggingface.co/datasets/ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe

22. GhanaNLP Community (2026). Youth domain datasets (Twi, Dagbani, Ewe, English). Hugging Face. https://huggingface.co/collections/ghananlpcommunity/ghana-adolescent-conversations-datasets

23. Chung, Y.-A., Zhang, Y., Han, W., Chiu, C.-C., Qin, J., Pang, R., et al. (2021). W2v-BERT: Combining contrastive learning and masked language modelingfor self-supervised speech pre-training. arXiv. http://arxiv.org/abs/2108.06209

24. Ridoy, M. S. I., Akter, S., & Rahman, M. A. (2025, July 2). Adaptability ofASR models on low-resource language: A comparative study of Whisper and Wav2Vec-BERT on Bangla. arXiv. http://arxiv.org/abs/2507.01931

25. Yi, C., Zhou, S., & Xu, B. (2021). Efficiently fusing pretrained acoustic and linguistic encoders for low-resource speech recognition. IEEE Signal Processing Letters, 28, 888–892. https://doi.org/10.1109/LSP.2021.3071668

26. Google AI for Developers. (2026). Get started with Gemma models. Google. https://ai.google.dev/gemma/docs/get\_started#gemma-4-family

27. von Neumann, T., Boeddeker, C., Delcroix, M., & Haeb-Umbach, R. (2025). Word error rate definitions and algorithms for long-form multi-talker speech recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing. https://doi.org/10.1109/TASLPRO.2025.3589862

28. Morris, A. C., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: Improved evaluation measures for connected speech recognition. In Interspeech 2004 (pp. 2765–2768). ISCA. https://doi.org/10.21437/Interspeech.2004-668

29. K, T. D., James, J., Gopinath, D. P., & Ashraf, M. K. Advocating character error rate for multilingual ASR evaluation. GitHub. https://github.com/

30. GhanaNLP. (2026). Offline Audio Recorder. GitHub. Retrieved July 31, 2026, from https://github.com/GhanaNLP/recorder

31. GhanaNLP, & Khaya AI. (2026). KasaHealth: Voice-first ASRH application [Web application]. Render. Retrieved July 31, 2026, from https://kasa-health-frontend.onrender.com

32. GhanaNLP. (2026). Senti-Check: ASR evaluation harness [Computer software]. GitHub. Retrieved July 31, 2026, from https://github.com/GhanaNLP/senti-check