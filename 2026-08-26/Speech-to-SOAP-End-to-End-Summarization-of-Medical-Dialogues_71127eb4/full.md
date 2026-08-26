# Speech-to-SOAP: End-to-End Summarization of Medical Dialogues: KIT@BeTraC 2026

1<sup>st</sup> Enes Yavuz Ugan

ISL, Karlsruhe Institute of Technology enes.ugan@kit.edu

2<sup>nd</sup> Fabian Retkowski

ISL, Karlsruhe Institute of Technology

3<sup>rd</sup> Yuka Ko

AI4LT, Karlsruhe Institute of Technology

4<sup>th</sup> Thai-Binh Nguyen

5<sup>th</sup> Maike Zufle¨

6<sup>th</sup> Jan Niehues

ISL, Karlsruhe Institute of Technology AI4LT, Karlsruhe Institute of Technology AI4LT, Karlsruhe Institute of Technology

7<sup>th</sup> Alexander Waibel

InterACT, Carnegie Mellon University

ISL, Karlsruhe Institute of Technology

Abstract—With the advent of Large Language Models and its instruction following capabilities a promising application is the task of summarization. Within this domain of task the extractive sub-task of clinical protocolling has emerged as a topic of particular interest as it can significantly reduce the downtime and protocolling burden of health-care workers thus enabling them to focus on their core work helping humans. A further step towards automation is the direct generation of clinical notes from speech without intermediate transcripts, reducing processing time while preserving information such as coughing or other paralinguistic cues that may be lost in transcript-based systems. To this end, we present KIT’s submission to this years BeTraC challenge in the lightweight track. Our main contribution is a scalable data augmentation pipeline that unifies heterogeneous medical dialogue datasets through synthetic speech generation and automatically generated SOAP supervision, enabling robust adaptation of a speech foundation model for end-to-end speechto-SOAP generation.

Index Terms—medical dialogue summarization, SOAP note generation, speech foundation models

## I. EXPERIMENTAL SETUP

a) Architecture: We use Qwen2.5-Omni-3B [1], an endto-end speech-language model that has demonstrated instruction following [2] and speech summarization [3] capabilities. The model is adapted to medical dialogue understanding and SOAP note generation using LoRA [4], [5] in LLaMA-Factory [6].

b) Data and Data Augmentation: We use Synth-DoPaCo [7], ACI-Bench [8], MTS-Dialog [9], PriMock57 [10], and OMI [11] datasets. Synth-DoPaCo and OMI are fully synthetic datasets, ACI-Bench contains role-played encounters, Pri-Mock57 consists of simulated consultations with real recordings, and MTS-Dialog provides text-only medical conversations. For datasets without audio, we synthesize speech using Kokoro-82M [12]. All datasets are converted into a unified Audio→SOAP, Transcript→SOAP, and Audio→Diarized Transcript format when applicable. The resulting data consists of 18795 dialogues and a total of 1653,067 hours of audio.

The final dataset and code are publicly available. <sup>1</sup> <sup>2</sup>

For datasets without SOAP-style target notes, we generate additional SOAP supervision from dialogue transcripts using GPT-3.5-27B in a non-thinking setting. Each prompt consists of a general task instruction and optional SOAP formatting guidance. For downstream training, we use a SOAP template + concept statistics prompt, which combines an annotated SOAP structure and representative examples with corpus-derived clinical concept frequencies, charting-style expressions, and lay-to-clinical terminology guidance. This setting aims to normalize heterogeneous note styles and terminology into a consistent SOAP format.

## A. Experiments

We conduct several experiments to understand the impact of pretraining, data modality, and reasoning strategies on medical SOAP note generation. All results are collated in Tables I-III.

a) Prompts: We tried 3 different system prompts in order to determine their effect on the base models performance and to choose which one to use. Additionally we applied two of the more complex prompts in the actual user instruction of the model.

b) Duration and Cleaning.: During manual inspection, we observed several severe TTS hallucinations where the generated audio no longer matched the reference transcript. We therefore developed an alignment-based script that removes non-aligned audio segments. To study the effect of long conversations, we additionally filtered training samples longer than 15, 21, and 25 minutes in the DoPaCo dataset, motivated by the reported average dialogue duration of 9 minutes.

c) Multi-stage Methods.: We investigate different initialization and multi-stage adaptation strategies for adapting Qwen2.5-Omni to medical dialogue understanding. In particular, we study whether intermediate tasks such as Transcript→SOAP and Audio→ASR improve subsequent Audio→SOAP generation. We additionally evaluate whether explicit speaker diarization benefits downstream SOAP generation. However, after Audio→ASR adaptation, the model already achieves a speaker-attributed WER of approximately 3%, indicating strong inherent diarization capabilities. Consequently, we do not pursue explicit speaker diarization in subsequent experiments.

d) Training on Speech versus Text and Speech.: A central question of this work is whether transcript supervision improves end-to-end speech-to-SOAP generation. We therefore compare models trained exclusively on Audio→SOAP examples with models jointly trained on Audio→SOAP and Transcript→SOAP data. Joint training may provide additional supervision by decoupling speech recognition from the downstream summarization task while exposing the model to a larger number of semantic SOAP generation examples.

e) Chain-of-Thought Generation.: We investigate chainof-thought (CoT) supervision by introducing intermediate reasoning targets before SOAP note generation. Specifically, we derive several reasoning targets tailored to the evaluation metrics, including medical concepts, entities, and terminology extraction. We further compare explicit reasoning traces enclosed in <think>...</think> tags with natural-language reasoning prompts preceding the final SOAP note generation.

## B. Training Details

Unless otherwise stated, all experiments employ the official Qwen2-Omni chat template, FlashAttention 2, bfloat16 precision, and gradient checkpointing. We apply LoRA with rank r = 32 to all target modules while keeping the multimodal projector frozen throughout training.

Models are optimized using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ , cosine learning rate decay, and a warmup ratio of 10%. Following [14], we use a small effective batch size of 4, which also allows training under our computational constraints. Model selection is performed using the checkpoint with the lowest development-set perplexity.

## C. Evaluation

We follow the official BeTraC evaluation protocol for medical SOAP note generation. The primary evaluation metrics are based on lexical overlap measures, including ROUGE, which quantify the similarity between generated and reference SOAP notes. However, lexical metrics alone may not fully capture clinical correctness, as semantically equivalent notes can differ substantially in wording.

## II. RESULTS

a) Prompt Ablation.: We investigate the impact of prompt design and prompt placement on SOAP note generation. Starting from a simple baseline prompt, we evaluate more detailed prompts both as system prompts (System) and as instruction prompts (Instruction), with and without additional examples. As shown in Table I, increasing prompt complexity in the system prompt consistently degrades performance compared to the simple baseline. In contrast, placing the detailed prompts to the instruction position substantially improves summarization quality, achieving the best ROUGE-2 and ROUGE-3 scores. These results suggest that Qwen2.5- Omni is sensitive to instruction placement and benefits more from detailed task guidance as explicit instructions rather than as persistent system-level behavior.

TABLE I  
DEVELOPMENT-SET PROMPT ABLATION. DETAILED PROMPTS WERE EVALUATED AS SYSTEM OR INSTRUCTION PROMPTS, WITH AND WITHOUT AN EXAMPLE.
<table><tr><td rowspan=1 colspan=1>System</td><td rowspan=1 colspan=1>Concept-F1</td><td rowspan=1 colspan=1>R-2</td><td rowspan=1 colspan=1>R-3</td></tr><tr><td rowspan=1 colspan=1>Baselines</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Reported</td><td rowspan=1 colspan=1>0.2604</td><td rowspan=1 colspan=1>0.0920</td><td rowspan=1 colspan=1>0.0344</td></tr><tr><td rowspan=1 colspan=1>Base Prompt</td><td rowspan=1 colspan=1>0.3276</td><td rowspan=1 colspan=1>0.1315</td><td rowspan=1 colspan=1>0.0589</td></tr><tr><td rowspan=1 colspan=4>System Prompt Ablation</td></tr><tr><td rowspan=1 colspan=1>System-Detailed</td><td rowspan=1 colspan=1>0.2348</td><td rowspan=1 colspan=1>0.1232</td><td rowspan=1 colspan=1>0.0610</td></tr><tr><td rowspan=1 colspan=1>System-Detailed+Example</td><td rowspan=1 colspan=1>0.2332</td><td rowspan=1 colspan=1>0.1171</td><td rowspan=1 colspan=1>0.0569</td></tr><tr><td rowspan=1 colspan=4>Instruction Prompt Ablation</td></tr><tr><td rowspan=1 colspan=1>Instruction-Detailed</td><td rowspan=1 colspan=1>0.2666</td><td rowspan=1 colspan=1>0.1671</td><td rowspan=1 colspan=1>0.0919</td></tr><tr><td rowspan=1 colspan=1>Instruction-Detailed+Example</td><td rowspan=1 colspan=1>0.2521</td><td rowspan=1 colspan=1>0.1482</td><td rowspan=1 colspan=1>0.0780</td></tr></table>

b) Training on Speech versus Text and Speech.: We investigate whether incorporating transcript-conditioned supervision improves downstream SOAP note generation. Table II compares models trained solely on Audio→SOAP examples with models jointly trained on Audio→SOAP and Transcript→SOAP data.

Joint audio-text training consistently improves clinical concept extraction, increasing Concept-F1 from 0.4780 to 0.4902, while yielding nearly identical ROUGE scores. This suggests that transcript supervision primarily benefits the model’s ability to identify and represent medically relevant information rather than substantially changing the lexical overlap with reference notes. Consequently, we adopt joint Audio+Text→SOAP training in our subsequent experiments.

c) Multi-Stage Adaptation.: We further investigate whether intermediate adaptation tasks can improve downstream Audio→SOAP generation. In particular, we consider intermediate objectives including Audio→ASR, Transcript→SOAP, joint Audio/Text→SOAP training, and chain-of-thought (CoT) supervision.

As shown in Table II, all intermediate adaptation strategies substantially outperform the Audio→ASR baseline. Initializing from an Audio→ASR model and subsequently finetuning on Audio→SOAP achieves the strongest ROUGE-2 and ROUGE-3 scores, indicating that explicit transcript generation provides a useful intermediate representation for medical note generation. In contrast, CoT supervision achieves the highest Concept-F1 score, suggesting that explicitly modeling intermediate reasoning steps can improve the extraction of clinically relevant concepts. In general, no single adaptation strategy dominates across all metrics, but intermediate adaptation consistently provides large improvements over direct Audio→ASR training.

d) Duration and Cleaning Ablation.: During manual inspection of the synthetic DoPaCo audio, we observed several severe TTS hallucinations in which the generated audio diverged substantially from the reference transcript. To mitigate this issue, we developed an alignment-based cleaning procedure that removes audio segments that cannot be aligned with the corresponding transcript.

TABLE II  
DEVELOPMENT-SET ABLATIONS FOR MULTI-STAGE ADAPTATION, JOINT AUDIO/TEXT TRAINING, AND DURATION/AUDIO CLEANING. A, T, AND AT DENOTE AUDIO, TRANSCRIPT, AND AUDIO/TRANSCRIPT INPUTS, RESPECTIVELY. ”CLEAN” INDICATES TRANSCRIPT-ALIGNED AUDIO AFTER REMOVING HALLUCINATED TTS SEGMENTS.
<table><tr><td rowspan=1 colspan=1>#</td><td rowspan=1 colspan=1>Strategy / Dataset</td><td rowspan=1 colspan=1>C-F1</td><td rowspan=1 colspan=1>R-2</td><td rowspan=1 colspan=1>R-3</td></tr><tr><td rowspan=1 colspan=1>Audi</td><td rowspan=1 colspan=1>o → SOAP</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>A-SOAP</td><td rowspan=1 colspan=1>0.4780</td><td rowspan=1 colspan=1>0.3366</td><td rowspan=1 colspan=1>0.2283</td></tr><tr><td rowspan=1 colspan=1>Audi</td><td rowspan=1 colspan=1>o + Text → SOAP</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>AT-SOAP</td><td rowspan=1 colspan=1>0.4902</td><td rowspan=1 colspan=1>0.3366</td><td rowspan=1 colspan=1>0.2261</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>-Stage</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>A-ASR</td><td rowspan=1 colspan=1>0.3233</td><td rowspan=1 colspan=1>0.0940</td><td rowspan=1 colspan=1>0.0407</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>T-SOAP → A-SOAP</td><td rowspan=1 colspan=1>0.4834</td><td rowspan=1 colspan=1>0.3261</td><td rowspan=1 colspan=1>0.2188</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>A-ASR → A-SOAP</td><td rowspan=1 colspan=1>0.4871</td><td rowspan=1 colspan=1>0.3430</td><td rowspan=1 colspan=1>0.2338</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>A-ASR → AT-SOAP</td><td rowspan=1 colspan=1>0.4906</td><td rowspan=1 colspan=1>0.3378</td><td rowspan=1 colspan=1>0.2275</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>A-ASR + AT-SOAP</td><td rowspan=1 colspan=1>0.4842</td><td rowspan=1 colspan=1>0.3256</td><td rowspan=1 colspan=1>0.2175</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>AT-SOAP → A-SOAP</td><td rowspan=1 colspan=1>0.4809</td><td rowspan=1 colspan=1>0.3234</td><td rowspan=1 colspan=1>0.2151</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>AT-CoT → AT-SOAP</td><td rowspan=1 colspan=1>0.4908</td><td rowspan=1 colspan=1>0.3391</td><td rowspan=1 colspan=1>0.2275</td></tr><tr><td rowspan=1 colspan=1>Durat</td><td rowspan=1 colspan=1>ion and Audio Cleaning</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Clean (21 min)</td><td rowspan=1 colspan=1>0.4898</td><td rowspan=1 colspan=1>0.3368</td><td rowspan=1 colspan=1>0.2278</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>Clean (25 min)</td><td rowspan=1 colspan=1>0.4791</td><td rowspan=1 colspan=1>0.3319</td><td rowspan=1 colspan=1>0.2218</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>Unclean (15 min)</td><td rowspan=1 colspan=1>0.4781</td><td rowspan=1 colspan=1>0.3287</td><td rowspan=1 colspan=1>0.2210</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>Unclean (21 min)</td><td rowspan=1 colspan=1>0.4965</td><td rowspan=1 colspan=1>0.3386</td><td rowspan=1 colspan=1>0.2287</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>Unclean (25 min)</td><td rowspan=1 colspan=1>0.4780</td><td rowspan=1 colspan=1>0.3366</td><td rowspan=1 colspan=1>0.2283</td></tr></table>

We further investigate the effect of long conversations by filtering training examples exceeding 15, 21, and 25 minutes in duration. As shown in Table II, cleaning does not improve performance, with the best results obtained using the uncleaned dataset filtered to a maximum duration of 21 minutes. Increasing the duration threshold beyond 21 minutes yields no further improvements and slightly degrades performance, suggesting that very long conversations introduce additional noise that outweighs the benefit of increased training data.

e) Chain-of-Thought Generation.: We investigated several chain-of-thought (CoT) strategies based on intermediate medical concepts, entities, terminology extraction, and explicit reasoning traces. Although natural-language reasoning performed better than explicit <think> tags, none of the explored CoT variants improved over direct end-to-end SOAP generation. Consequently, all final systems use direct SOAP generation.

f) Final Systems.: After selecting the most promising training strategies through the preceding ablations, we train our final systems using all available datasets and generated supervision. Table III reports two representative models together with our final submission.

Motivated by [15]–[17], we further investigate checkpoint averaging using different combinations of the best-performing development-set checkpoints. Among the evaluated combinations, averaging the checkpoints corresponding to rows 13, 16, and 17 achieved the best development-set performance and was therefore selected for our final submission.

## III. DISCUSSION

a) Official Evaluation: The official evaluation, shown in Table III, comprises three test sets: the in-domain DoPaCo test set, the Mock Dialogue dataset [18], and the Realistic dialogue recordings collected for the shared task. Although both submissions achieved similar development-set performance, our primary merged model, using the checkpoint averaging strategy of [16], consistently outperformed the contrastive submission on all official test sets, with the largest gains under increasing domain shift. These results indicate improved robustness and reduced overfitting to synthetic TTS data.

TABLE III  
REPRESENTATIVE FINAL SYSTEMS, MERGED DEVELOPMENT-SET SUBMISSION MODEL, AND OFFICIAL SUBMISSION RESULTS.
<table><tr><td rowspan=1 colspan=1>#</td><td rowspan=1 colspan=1>System / Split</td><td rowspan=1 colspan=1>C-F1</td><td rowspan=1 colspan=1>R-2</td><td rowspan=1 colspan=1>R-3</td></tr><tr><td rowspan=1 colspan=1>Devel</td><td rowspan=1 colspan=1>opment-set model selectio</td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>A-ASR → AT-SOAP</td><td rowspan=1 colspan=1>0.4918</td><td rowspan=1 colspan=1>0.3345</td><td rowspan=1 colspan=1>0.2256</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>AT-CoT → AT-SOAP</td><td rowspan=1 colspan=1>0.4894</td><td rowspan=1 colspan=1>0.3403</td><td rowspan=1 colspan=1>0.2300</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>AT-SOAP</td><td rowspan=1 colspan=1>0.4924</td><td rowspan=1 colspan=1>0.3430</td><td rowspan=1 colspan=1>0.2307</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>Merged Submission</td><td rowspan=1 colspan=1>0.4986</td><td rowspan=1 colspan=1>0.3537</td><td rowspan=1 colspan=1>0.2417</td></tr><tr><td rowspan=1 colspan=1>Offic</td><td rowspan=1 colspan=1>ial primary submission (R</td><td rowspan=1 colspan=1>ow 18)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>DoPaCo test</td><td rowspan=1 colspan=1>0.4949</td><td rowspan=1 colspan=1>0.3601</td><td rowspan=1 colspan=1>0.2499</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Mock dialogue</td><td rowspan=1 colspan=1>0.4618</td><td rowspan=1 colspan=1>0.3186</td><td rowspan=1 colspan=1>0.2011</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Realistic</td><td rowspan=1 colspan=1>0.4855</td><td rowspan=1 colspan=1>0.3430</td><td rowspan=1 colspan=1>0.2326</td></tr><tr><td rowspan=1 colspan=1>Offic</td><td rowspan=1 colspan=1>ial contrastive submission</td><td rowspan=1 colspan=1>(Row 13)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>DoPaCo test</td><td rowspan=1 colspan=1>0.3889</td><td rowspan=1 colspan=1>0.2161</td><td rowspan=1 colspan=1>0.1289</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Mock dialogue</td><td rowspan=1 colspan=1>0.3372</td><td rowspan=1 colspan=1>0.1771</td><td rowspan=1 colspan=1>0.0940</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Realistic</td><td rowspan=1 colspan=1>0.2814</td><td rowspan=1 colspan=1>0.1377</td><td rowspan=1 colspan=1>0.0733</td></tr></table>

b) SOAP Data Generation Prompt Variants: We further analyze simpler prompt variants on the Synth-DoPaCo development set, including a SOAP template prompt and a few-shot prompt with two SOAP examples. As shown in Table IV, the few-shot SOAP examples prompt obtains the best development scores, suggesting that in-domain examples are particularly useful for direct SOAP generation. Since the downstream training experiments in this work are based on the SOAP template + concept statistics prompt, we leave incorporating few-shot-generated SOAP supervision into the full training pipeline for future work.

TABLE IV  
DEVELOPMENT-SET COMPARISON OF SOAP GENERATION PROMPT VARIANTS USING GPT-3.5-27B.
<table><tr><td rowspan=1 colspan=1>Prompt variant</td><td rowspan=1 colspan=1>Concept-F1</td><td rowspan=1 colspan=1>R-2</td><td rowspan=1 colspan=1>R-3</td></tr><tr><td rowspan=1 colspan=1>SOAP template + concept statistics</td><td rowspan=1 colspan=1>0.4191</td><td rowspan=1 colspan=1>0.2481</td><td rowspan=1 colspan=1>0.1481</td></tr><tr><td rowspan=1 colspan=1>SOAP template</td><td rowspan=1 colspan=1>0.4409</td><td rowspan=1 colspan=1>0.2616</td><td rowspan=1 colspan=1>0.1562</td></tr><tr><td rowspan=1 colspan=1>Few-shot SOAP examples</td><td rowspan=1 colspan=1>0.4791</td><td rowspan=1 colspan=1>0.2916</td><td rowspan=1 colspan=1>0.1751</td></tr></table>

## IV. CONCLUSION

We presented KIT’s submission to the BeTraC 2026 Lightweight Track for end-to-end speech-to-SOAP generation. Our main contribution is a scalable data augmentation pipeline that unifies heterogeneous medical dialogue datasets through synthetic speech generation and automatically generated SOAP supervision, enabling effective adaptation of Qwen2.5-Omni.

Our final merged system achieved the best performance among our submitted systems across the official test sets, indicating that averaging diversely trained checkpoints can improve robustness under domain shift.

## ACKNOWLEDGMENT

Generative AI tools were used for grammar correction, readability improvements, and LAT X formatting and structuring. All scientific content was produced and verified by the authors. This work was supported by the project “How is AI Changing Science? Research in the Era of Learning Algorithms” (Hi-AICS), funded by the Volkswagen Foundation, and partially by the European Union’s Horizon research and innovation programme under grant agreement No. 101135798, project Meetween (My Personal AI Mediator for Virtual MEETtings BetWEEN People) and European Union’s Horizon Europe programme grant agreement No. 101213369 (DVPS). The authors gratefully acknowledge computing time provided on HoreKa at the National High-Performance Computing Center at KIT (NHR@KIT), supported by the Federal Ministry of Education and Research, the Ministry of Science, Research and the Arts of Baden-Wurttemberg, and the DFG.¨

## REFERENCES

[1] Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y. & Lin, J. Qwen2.5-Omni Technical Report. (2025), https://arxiv.org/abs/2503.20215

[2] Papi, S., Zufle, M., Gaido, M., Savoldi, B., Liu, D., Douros,¨ I., Bentivogli, L. & Niehues, J. Mcif: Multimodal crosslingual instruction-following benchmark from scientific talks. ArXiv Preprint ArXiv:2507.19634. (2025)

[3] Retkowski, F., Zufle, M., Sudmann, A., Pfau, D., Watanabe, S., Niehues,¨ J. & Waibel, A. Summarizing speech: A comprehensive survey. Proceedings Of The 2025 Conference On Empirical Methods In Natural Language Processing. pp. 27263-27294 (2025)

[4] Pham, N., Nguyen, T., Stuker, S. & Waibel, A. Efficient weight¨ factorization for multilingual speech recognition. ArXiv Preprint ArXiv:2105.03010. (2021)

[5] Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. & Others Lora: Low-rank adaptation of large language models.. Iclr. 1, 3 (2022)

[6] Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z. & Ma, Y. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. Proceedings Of The 62nd Annual Meeting Of The Association For Computational Linguistics (Volume 3: System Demonstrations). (2024), http://arxiv.org/abs/2403.13372

[7] Labrak, Y., Grunert, D., Baroudi, S., Chun, J., Cyrta, P., Burdisso, S.,¨ Hassoon, A., Liu, D., Rothschild, A., Van Deusen, R., Motlicek, P., Perrault, A., Marxer, R. & Schaaf, T. Generating Synthetic Doctor-Patient Conversations for Long-form Audio Summarization. Proc. Interspeech 2026. (2026)

[8] Yim, W., Fu, Y., Ben Abacha, A., Snider, N., Lin, T. & Yetisgen, M. ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation. Nature Scientific Data. (2023)

[9] Ben Abacha, A., Yim, W., Fan, Y. & Lin, T. An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters. Proceedings Of The 17th Conference Of The European Chapter Of The Association For Computational Linguistics. pp. 2291-2302 (2023,5), https://aclanthology.org/2023.eacl-main.168

[10] Papadopoulos Korfiatis, A., Moramarco, F., Sarac, R. & Savkov, A. (in press): PriMock57: A Dataset Of Primary Care Mock Consultations. Proceedings Of The 60th Annual Meeting Of The Association For Computational Linguistics. (2022)

[11] Wang, J., Yao, Z., Yang, Z., Zhou, H., Li, R., Wang, X., Xu, Y. & Yu, H. Notechat: a dataset of synthetic patient-physician conversations conditioned on clinical notes. Findings OfThe Association For Computational Linguistics: ACL 2024. pp. 15183-15201 (2024)

[12] Li, Y., Han, C., Raghavan, V., Mischler, G. & Mesgarani, N. Styletts 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. Advances In Neural Information Processing Systems. 36 pp. 19594-19621 (2023)

[13] Li, Y., Han, C., Raghavan, V., Mischler, G. & Mesgarani, N. StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models. (2023), https://arxiv.org/abs/2306.07691

[14] Ugan, E., Zufle, M., Ko, Y., Sinhamahapatra, S., Retkowski, F., Akti, S.,¨ Niehues, J. & Waibel, A. Multilingual Long-Form Speech Instruction Following: KIT’s Submission to IWSLT 2026. Proceedings Of The 23rd International Conference On Spoken Language Translation (IWSLT 2026). pp. 132-149 (2026)

[15] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. Averaging weights leads to wider optima and better generalization. ArXiv Preprint ArXiv:1803.05407. (2018)

[16] Ugan, E., Pham, N. & Waibel, A. Weight factorization and centralization for continual learning in speech recognition. ArXiv Preprint ArXiv:2506.16574. (2025)

[17] Vander Eeckt, S. & Others Rehearsal-free online continual learning for automatic speech recognition. ArXiv E-prints. pp. arXiv-2306 (2023)

[18] Fareez, F., Parikh, T., Wavell, C., Shahab, S., Chevalier, M., Good, S., De Blasi, I., Rhouma, R., McMahon, C., Lam, J., Lo, T. & Smith, C. A dataset of simulated patient-physician medical interviews with a focus on respiratory cases. Scientific Data. 9, 313 (2022,6), https://doi.org/10.1038/s41597-022-01423-1