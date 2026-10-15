# SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets

Yixian Tan, Aulia Adila, Jonathan Heng, Kunat Pipatanakul, Donghang Wu, Anocha Sutaveephamochanon

AI Singapore, Singapore

yixian,aulia.adila,jonathan,kunat.p,donghang,anocha@aisingapore.org

## Abstract

We introduce SEA-HELM Audio, a Southeast Asian (SEA) multilingual evaluation framework that extends the text-based SEA-HELM benchmark to assess performance degradation in cascaded automatic speech recognition and large language model (ASR–LLM) pipelines. Using synthesized speech, we compare speech-input and text-input performance across five languages—Chinese, English, Malay, Indonesian, and Tamil—on multiple-choice reasoning and instruction-following tasks. We evaluate open-weight and commercial ASR–LLM systems using identical verifiers and quantify the speech gap as the difference between text and speech accuracy. Results show substantial degradation, with accuracy drops of 9.00–18.50 percentage points for multiple-choice reasoning and 20.00–42.86 points for instruction following. Tamil exhibits the largest degradation, exceeding 40 percentage points. Our findings highlight the limitations of text-only benchmarks in predicting voice-agent reliability and demonstrate the value of SEA-HELM Audio for evaluating multilingual voice-agent systems. Index Terms: automatic speech recognition, multilingual evaluation, spoken language understanding, SEA-HELM Audio

## 1. Introduction

As artificial intelligence systems transition from text-based interfaces to conversational voice-enabled agents, evaluating the end-to-end accuracy of cascaded speech-language stacks has become a critical research priority. While text-based Large Language Models (LLMs) continue to achieve high benchmark scores, deploying these models within voice pipelines introduces acoustic transcription errors, phonetic ambiguities, and domain-specific degradation. Quantifying this degradation, specifically measuring the accuracy penalty a user incurs by speaking rather than typing, is essential for understanding realworld agent reliability across diverse tasks.

Existing spoken language model benchmarks, such as VoiceBench [1], evaluate capabilities including speech recognition, reasoning, and instruction following through spoken inputs. However, their predominantly English-language evaluation settings provide limited insight into how voice-agent systems perform on Southeast Asian languages and tasks. In particular, it remains unclear to what extent transcription errors alter task-critical information and how these errors affect downstream task performance, especially in the context of Southeast Asian languages.

To address the need for localized evaluation, AI Singapore developed SEA-HELM (Southeast Asian Holistic Evaluation of Language Models) [2]. SEA-HELM provides authentic perlanguage prompts and tailored programmatic verifiers across regional languages. However, SEA-HELM was engineered natively as a text-only benchmark, consisting exclusively of written prompts without paired audio recordings.

This paper addresses the central research question: How can a text-based, regionally grounded benchmark be adapted into a speech-based evaluation dataset to systematically measure the impact of speech input on downstream LLM task performance across Southeast Asian languages? By bridging SEA-HELM’s text prompts with synthesized audio pipelines, we establish a framework to quantify the Speech gap, the difference between Score<sub>text</sub> and Score<sub>speech</sub>, across multiple Southeast Asian languages, providing empirical baselines for regional speech agent deployment.

## 2. Methodology

This section outlines the experimental framework, dataset selection, and evaluation metrics employed to systematically isolate and quantify the performance degradation introduced by the speech modality across our five target languages.

## 2.1. Experimental Design

Prior to finalizing the evaluation pipeline, we conducted an exploratory phase to establish a robust and regionally relevant testing environment. A primary distinction between our setup and existing voice evaluation frameworks lies in audio corpus generation: VoiceBench uses pre-recorded human speech, whereas SEA-HELM provides only text prompts. Consequently, our cascade architecture synthesizes its own input speech before evaluating downstream models.

To ensure audio fidelity across regional languages, we generated synthetic speech with OmniVoice [3] Text-to-Speech (TTS) engines, selected for acoustic naturalness, prosodic accuracy, and regional accent representation.

For task design, we selected multiple-choice and instruction-following formats, which are established in voice evaluation, and instantiated them with SEA-HELM text prompts. This approach preserves regional nuances, codeswitching patterns, and local cultural contexts across Chinese, Malay, Tamil, Indonesian, and English. By evaluating each sample twice, once through the synthesized voice pipeline (Score<sub>speech</sub>) and once via direct text prompts (Score<sub>text</sub>), we establish a controlled baseline to isolate and measure the precise performance penalty imposed by the speech-to-text layer.

## 2.2. Datasets and Task Selection

To evaluate the voice cascade cost-effectively and deterministically, we selected tasks from SEA-HELM that rely on rulebased scoring and closely align with established VoiceBench

task types.

The evaluation suite draws from four primary SEA-HELM task subsets, summarized in Table 1.

## 2.2.1. Benchmark Alignment and Active Subset Scope

For the initial empirical runs detailed in this study, benchmarking focuses on the SEA-HELM IFEval and Global-MMLU-Lite subsets across our five focus languages (Chinese, Malay, Tamil, Indonesian, and English).

The selection criteria prioritized two key attributes:

1. Speakable Prompts: Prompts can be read aloud as natural speech without removing formatting or template markup first.

2. Verifiable Evaluation: All scoring is executed via programmatic, rule-based verifiers, so the text path and the speech path are scored in exactly the same way.

## 2.3. Evaluation Metrics

To formalize the measurement of the speech modality penalty, the following metrics were captured across every benchmark run:

• Text accuracy $\scriptstyle ( \mathrm { S c o r e } _ { \mathrm { t e x t } } ) :$ The accuracy of the target LLM when provided with the original, ground-truth text input. This represents the theoretical performance ceiling for the task.

• Speech accuracy $\scriptstyle ( \mathrm { S c o r e } _ { \mathrm { s p e e c h } } ) \colon$ The end-to-end accuracy of the target LLM when the prompt is spoken, synthesized by TTS and transcribed by ASR.

• ∆ (The Gap): $\Delta = \mathrm { S c o r e } _ { \mathrm { t e x t } } - \mathrm { S c o r e } _ { \mathrm { s p e e c h } }$ . This absolute difference represents the modality penalty, quantifying how much task performance degrades due to transcription errors, phonetic ambiguities, or missing acoustic signals.

## 3. SEA-HELM Audio

SEA-HELM contains only written prompts, so SEA-HELM Audio creates the spoken input itself: each prompt is read aloud by a speech synthesizer, transcribed by the ASR system under test, and answered by the LLM. Comparing this against the same LLM answering the original text isolates the cost of the speech front-end.

## 3.1. Evaluation Flow

Each SEA-HELM prompt is evaluated in two ways (Figure 1). In the speech path, the prompt is converted to speech with TTS, transcribed back to text by ASR, and the transcript is answered by the LLM. In the text path, the LLM answers the original prompt directly. Both answers are scored by the same rulebased verifier, giving $\mathrm { S c o r e } _ { \mathrm { s p e e c h } }$ and $\mathrm { S c o r e } _ { \mathrm { t e x t } } .$ Because the two paths share the same LLM and the same verifier, the difference between them $( \Delta = \mathrm { S c o r e _ { t e x t } \ - \ S c o r e _ { s p e e c h } } )$ is the accuracy lost by speaking the prompt instead of typing it.

## 3.2. Spoken Prompt Construction

The TTS system plays the role of the user. Its voice is chosen independently of the system being evaluated, so every ASR–LLM configuration hears the same spoken prompts and differences in $\mathrm { S c o r e } _ { \mathrm { s p e e c h } }$ reflect the system rather than the input.

Only the content a user would actually say is synthesized. For multiple-choice tasks, this is a short instruction followed by the question and its four options; for instruction-following tasks, it is the prompt exactly as written. Written conventions that do not read naturally aloud are adapted per language: in Chinese, the option labels A–D and the answer marker are written as Chinese words. The answer key (the correct option, or the constraints an IFEval response must satisfy) is never spoken and is used only at scoring time.

## 3.3. Speech Recognition

The ASR system turns the synthesized speech back into text. We evaluate three systems chosen for their coverage of the target languages (Table 2): Qwen3-ASR-1.7B [7] as an openweight multilingual baseline, MERaLiON3-ASR-3B [8] as a regionally adapted open-weight model for Tamil, and ElevenLabs Scribe Realtime as a commercial service. Open-weight models use greedy decoding so that transcripts are reproducible.

Transcription quality is also measured directly, as Word Error Rate (WER) and Character Error Rate (CER) between the transcript and the original prompt. This separates how much the ASR changed the prompt from how much those changes affected the task.

## 3.4. Response Generation

The LLM answers the ASR transcript (speech path) and the original prompt (text path). We evaluate SEA-LION-4.5- 27B [9], a regional open-weight model, and Gemini 3.5 Flash, a commercial model. Both use deterministic decoding (temperature 0), so any difference between the two paths comes from the input rather than from sampling.

## 3.5. Scoring

All responses are scored with rule-based verifiers taken from SEA-HELM, which keeps scoring deterministic, cheap to repeat, and identical across the two paths. For multiple-choice questions, the chosen option letter (A–D) is extracted from the response and compared with the answer key. For instruction following, each response is checked against the constraints stated in its prompt and is counted as correct only if it also uses the expected language.

## 4. Results

This section presents the empirical findings from our multilingual voice cascade evaluations. Performance is reported across two primary axes: task type (multiple-choice questions via Global-MMLU-Lite vs. instruction following via SEA-HELM IFEval) and pipeline configuration (Regional Open-Weight Baseline vs. Commercial Stack).

## 4.1. Multiple Choice Reasoning (Global-MMLU-Lite)

Evaluations on the SEA-HELM Global-MMLU-Lite subset were conducted for Chinese and English, assessing general domain knowledge under a multiple-choice format (400 samples per run). Across both languages, the commercial stack (Eleven-Labs Scribe Realtime paired with Gemini 3.5 Flash) outperformed the regional open-weight baseline in both text accuracy and speech accuracy.

However, Chinese demonstrated a distinct resilience to the modality shift compared to English. For Chinese, ∆ remained between 9.00 and 10.75 points, whereas English incurred a much steeper penalty of 15.50 to 18.50 points.

Table 1: SEA-HELM tasks consideredfor SEA-HELM Audio and their evaluation scope.
<table><tr><td>Task</td><td>Task Type</td><td>SEA- HELM Languages</td><td>Evaluated Here</td><td>Selection Rationale</td></tr><tr><td>SEA-HELM IFEval</td><td>Instruction following [4]</td><td>10</td><td>English, Malay, Indonesian, Tamil</td><td>Prompts are plain instructions that can be spoken as written. Scored by SEA-HELM&#x27;s multilingual verifier, which also verifies the</td></tr><tr><td>Global-MMLU-Lite</td><td>Multiple choice [5] 7</td><td></td><td>Chinese, English</td><td>response language. Short questions with four options; the answer is a single letter, so</td></tr><tr><td>SEA-HELM NLU QA</td><td>Question answering</td><td>5</td><td>Future work</td><td>scoring is exact. Scored with SEA-HELM&#x27;s rule-based token-F1 matching.</td></tr><tr><td>SEA-HELM Safeguard</td><td>Harmful-request refusal</td><td>8</td><td>Future work</td><td>Harmful prompts used as a refusal probe (analogous to AdvBench [6] in VoiceBench), testing whether the LLM still refuses when the request is spoken.</td></tr></table>

![](images/6b0f528f725f3b8bd343071c70366aeedaa92555b1b5fc1fdb64cf0d7766abb1.jpg)  
Figure 1: SEA-HELM Audio evaluation: speech path (TTS  Speech  LLM) vs. text path (LLM only).

Table 2: ASR systems evaluated.
<table><tr><td>ASR System</td><td>Type</td><td>Languages</td><td>Decoding</td></tr><tr><td>Qwen3-ASR-1.7B</td><td>Open-weight</td><td>en, zh, ms, id</td><td>Greedy</td></tr><tr><td>MERaLiON3-ASR-3B</td><td>Open-weight</td><td>ta</td><td>Greedy</td></tr><tr><td>ElevenLabs Scribe</td><td>Commercial</td><td>en, zh, ms, id, ta</td><td>Provider</td></tr><tr><td>Realtime</td><td>API</td><td></td><td>default</td></tr></table>

Table 3: Global-MMLU-Lite results (400 samples per run; TTS: OmniVoice). Accuracy (Speech, Text) and ∆ = Text Speech, in percentage points.
<table><tr><td>Language</td><td>ASR Model</td><td>LLM</td><td>Speech</td><td>Text</td><td>∆</td></tr><tr><td>Chinese</td><td>Qwen3-ASR-1.7B</td><td>SEA-LION- 4.5-27B</td><td>71.50</td><td>80.50</td><td>9.00</td></tr><tr><td>Chinese</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>79.75</td><td>90.50</td><td>10.75</td></tr><tr><td>English</td><td>Qwen3-ASR-1.7B</td><td>SEA-LION- 4.5-27B</td><td>69.00</td><td>87.50</td><td>18.50</td></tr><tr><td>English</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>76.75</td><td>92.25</td><td>15.50</td></tr></table>

## 4.2. Instruction Following (SEA-HELM IFEval)

SEA-HELM IFEval provided multiple test of deterministic instruction adherence across English, Malay, Indonesian, and Tamil (105 samples per run). The text-only baselines achieved exceptionally high scores, peaking at 97.14% for English, but introducing the speech modality resulted in severe performance degradation across all tested languages.

The modality gap (∆) expanded significantly in instructionfollowing tasks compared to the MCQ evaluations. Tamil suffered the steepest performance drops (exceeding 40 points) for both the specialized MERaLiON3-ASR-3B pipeline and the commercial ElevenLabs equivalent, indicating severe structural vulnerabilities in Tamil ASR transcriptions.

While the commercial stack generally outperformed the baseline, SEA-LION-4.5-27B proved highly competitive in text accuracy for Southeast Asian languages, notably surpassing Gemini 3.5 Flash in Indonesian IFEval (93.33% vs 89.52%). Despite this strong text foundation, the open-weight pipeline struggled to recover from ASR-induced errors, reflecting ∆ of over 25 points in Malay and Indonesian.

Table 4: SEA-HELM IFEval results (105 samples per run; TTS: ElevenLabs v3). Accuracy (Speech, Text) and ∆ = Text Speech, in percentage points.
<table><tr><td>Language</td><td>ASR Model</td><td>LLM</td><td>Speech</td><td>Text</td><td>∆</td></tr><tr><td>English</td><td>Qwen3-ASR-1.7B</td><td>SEA-LION- 4.5-27B</td><td>69.52</td><td>97.14</td><td>27.62</td></tr><tr><td>English</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>76.19</td><td>97.14</td><td>20.95</td></tr><tr><td>Indonesian</td><td>Qwen3-ASR-1.7B</td><td>SEA-LION- 4.5-27B</td><td>60.95</td><td>94.29</td><td>33.33</td></tr><tr><td>Indonesian</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>58.10</td><td>89.52</td><td>31.43</td></tr><tr><td>Malay</td><td>Qwen3-ASR-1.7B</td><td>SEA-LION- 4.5-27B</td><td>52.38</td><td>84.76</td><td>32.38</td></tr><tr><td>Malay</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>60.00</td><td>80.00</td><td>20.00</td></tr><tr><td>Tamil</td><td>MERaLiON3- ASR-3B</td><td>SEA-LION- 4.5-27B</td><td>47.62</td><td>87.62</td><td>40.00</td></tr><tr><td>Tamil</td><td>ElevenLabs Scribe Realtime</td><td>Gemini 3.5 Flash</td><td>41.90</td><td>81.90</td><td>40.00</td></tr></table>

## 5. Discussion & Analysis

The empirical results underscore a critical reality in voiceagent deployment: strong text-based LLM performance does not guarantee equivalent voice capabilities. The modality gap (∆) observed across our evaluations highlights significant disparities driven by language, task structure, and pipeline composition.

## 5.1. Linguistic Resilience vs. Vulnerability

The most striking finding is the sheer variance in the gap across the five evaluated languages.

• Resilience in Chinese: Chinese exhibited the lowest performance degradation across all configurations, maintaining a Delta of approximately 9.00 to 10.75 points. This suggests two compounding advantages: first, modern ASR systems (both Qwen and ElevenLabs) possess highly robust acoustic models for Chinese; second, the downstream LLMs demonstrate strong semantic recovery, able to infer correct answers even in the presence of minor phonetic transcription errors.

• The Tamil Bottleneck: Conversely, Tamil suffered the most severe modality penalty, shedding over 40 points in accuracy when transitioning from text to voice. Despite SEA-LION-4.5-27B achieving a respectable 85.71% baseline accuracy on perfect text, the end-to-end performance collapsed to 42.86% utilizing the specialized MERaLiON3-ASR-3B model. This indicates a severe bottleneck at the acoustic processing layer, highlighting a critical data scarcity issue in Tamil ASR training corpora compared to high-resource languages.

## 5.2. Task Format Sensitivity: MCQ vs. Instruction Following

The data reveals that the modality gap is heavily influenced by the nature of the evaluation task.

On Global-MMLU-Lite (MCQ), LLMs could often recover from noisy transcripts because multiple-choice reasoning primarily requires identifying key semantic concepts to select an option (A, B, C, or D). If an ASR model slightly misheard a noun, the LLM could still deduce the correct multiple-choice letter from contextual clues.

IFEval, in contrast, scores each response against explicit lexical, length, and format constraints stated in the prompt. When ASR drops or substitutes a word that carries one of these constraints, the LLM answers a different instruction, and the rule-based verifier marks the response as a failure even if it is otherwise correct. This mechanism accounts for the larger IFEval gaps of 20 to 43 points.

## 5.3. Commercial vs. Open-Weight

While the commercial stack (ElevenLabs Scribe Realtime + Gemini 3.5 Flash) consistently delivered higher speech accuracy, analyzing text accuracy reveals a nuanced picture of regional model capabilities.

The SEA-LION-4.5-27B model proved highly competitive, and occasionally superior, in pure-text regional evaluations. Most notably, it outperformed Gemini in the Indonesian IFEval text baseline (94.29% vs. 89.52%). The fact that SEA-LION’s end-to-end voice performance drops so sharply (a 33.33 point penalty) confirms that the limiting factor in the open-weight pipeline is not the LLM’s regional reasoning capability, but rather the fragility of the open-weight ASR models (Qwen3-ASR-1.7B and MERaLiON3-ASR-3B) handling Southeast Asian accents and code-switching.

For organizations deploying regional voice agents, these findings suggest that heavily investing in localized ASR finetuning will yield a much higher return on investment than further scaling the LLM parameters.

## 6. Conclusion and Future Work

This study shows that a text-native benchmark, such as SEA-HELM, can be adapted to benchmark end-to-end voice pipelines, provided the evaluation tasks are highly deterministic. By measuring the modality gap (∆) between pure text accuracy and speech accuracy across five languages, we have isolated the precise performance penalty introduced when a user speaks to a conversational agent rather than typing.

Our findings demonstrate that voice pipeline reliability is heavily dependent on the target language and the structural rigidity of the task. While Chinese multiple-choice reasoning exhibited a robust resilience to the modality shift $( \Delta \ < \ 1 1$ points), instruction following in Tamil suffered catastrophic degradation $( \Delta > 4 0$ points). Furthermore, while the commercial stack (ElevenLabs Scribe Realtime + Gemini 3.5 Flash) provided the most stable end-to-end performance, the strong pure-text baseline of SEA-LION-4.5-27B suggests that the primary bottleneck for regional open-weight deployments lies in the acoustic transcription layer, not the language model’s reasoning capabilities.

## 7. References

[1] Y. Chen, X. Yue, C. Zhang, X. Gao, R. T. Tan, and H. Li, “VoiceBench: Benchmarking LLM-based voice assistants,” arXiv preprint arXiv:2410.17196, 2024.

[2] Y. Susanto, A. V. Hulagadri, J. R. Montalan, J. G. Ngui, X. Yong, W. Q. Leong, H. Rengarajan, P. Limkonchotiwat, Y. Mai, and W. C. Tjhi, “SEA-HELM: Southeast Asian holistic evaluation of language models,” in Findings of the Association for Computational Linguistics: ACL 2025. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 12 308–12 336.

[3] H. Zhu, L. Ye, W. Kang, Z. Yao, L. Guo, F. Kuang, Z. Han, W. Zhuang, L. Lin, and D. Povey, “OmniVoice: Towards omnilingual zero-shot text-to-speech with diffusion language models,” arXiv preprint arXiv:2604.00688, 2026.

[4] J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou, “Instruction-following evaluation for large language models,” arXiv preprint arXiv:2311.07911, 2023.

[5] S. Singh, A. Romanou, C. Fourrier, D. I. Adelani, J. G. Ngui, D. Vila-Suero, P. Limkonchotiwat, K. Marchisio, W. Q. Leong, Y. Susanto, R. Ng, S. Longpre, S. Ruder, W.-Y. Ko, A. Bosselut, A. Oh, A. Martins, L. Choshen, D. Ippolito, E. Ferrante, M. Fadaee, B. Ermis, and S. Hooker, “Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation,” in Proc. 63rd Annual Meeting ofthe Associationfor Computational Linguistics (ACL). Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 18 761–18 799.

[6] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint arXiv:2307.15043, 2023.

[7] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Zhang, Z. Guo, H. Hao, Y. Xi, B. Yang, J. Xu, J. Zhou, and J. Lin, “Qwen3-ASR technical report,” arXiv preprint arXiv:2601.21337, 2026.

[8] Y. He, Z. Liu, S. Sun, B. Wang, W. Zhang, X. Zou, N. F. Chen, and A. T. Aw, “MERaLiON-AudioLLM: Bridging audio and language with large language models,” arXiv preprint arXiv:2412.09818, 2024.

[9] R. Ng, T. N. Nguyen, Y. Huang, N. C. Tai, W. Y. Leong, W. Q. Leong, X. Yong, J. G. Ngui, Y. Susanto, N. Cheng et al., “SEA-LION: Southeast Asian languages in one network,” arXiv preprint arXiv:2504.05747, 2025.