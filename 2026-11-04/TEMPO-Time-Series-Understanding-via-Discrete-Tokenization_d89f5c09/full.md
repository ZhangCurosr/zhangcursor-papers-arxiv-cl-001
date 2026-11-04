# TEMPO: Time Series Understanding via Discrete Tokenization

Riccardo Maggioni <sup>1</sup>

## Abstract

Time-series interpretation often requires both numerical observations and textual context. Integrating these inputs into a pretrained language model raises a representation question: how should a signal enter the model’s token sequence? We present TEMPO, which extends an LLM’s vocabulary with discrete codes from a context-aware Transformer tokenizer using finite scalar quantization. A frozen tokenizer maps four-step patches to a 625-code vocabulary; signal tokens, scale statistics, and text are then processed by the LLM’s existing self-attention layers. Adaptation updates the new token embeddings and DoRA adapters. With a Qwen3-4B backbone and taskspecific fine-tuning, TEMPO achieves macro-F1 scores of 89.0% on activity recognition, 85.0% on sleep staging, and 86.0% on bearing-fault diagnosis. Time-series question answering and qualitative bearing-diagnosis examples further illustrate how this interface combines signal representations with textual context. Together, these results demonstrate the utility of discrete vocabulary extension for time-series understanding.

## 1. Introduction

Time-series data record how a system evolves, while text describes the conditions under which those observations should be interpreted. A vibration trace, for example, may have different implications during commissioning and during routine operation. An interpretation therefore often needs to combine temporal patterns with operating conditions, maintenance history, and a user’s question.

Task-specific classifiers provide useful predictions, but their output interface is usually a fixed label set. Language models offer a complementary interface: users can ask questions and receive descriptions or explanations. The central challenge is to give a pretrained LLM access to the signal itself.

Numerical text serialization, continuous encoder embeddings, and learned discrete tokens make different trade-offs in sequence length, representation, and integration with the language model.

TEMPO studies a discrete interface. A separately pretrained tokenizer converts a normalized signal into contextual codes. These codes become additional vocabulary entries, so text and time-series tokens occupy one input sequence and use the same self-attention layers. Per-channel mean and standard deviation accompany the codes as text. This separates waveform shape from scale information while retaining both as model inputs.

Our contribution is the combination of a context-aware FSQ tokenizer, direct vocabulary extension, and a two-stage adaptation recipe for time-series tasks. The tokenizer attends over the full observed window before quantization, and the LLM is adapted through new embeddings and DoRA adapters. We evaluate this design on activity recognition, sleep staging, bearing-fault diagnosis, and time-series QA, and analyze the tokenizer encoder representations on UCR datasets.

## 2. Related Work

Time-series foundation models. Chronos quantizes scalar values for token-based forecasting (Ansari et al., 2024); TimesFM and Moirai develop pretrained forecasting models (Das et al., 2024; Woo et al., 2024). Mantis learns representations for classification (Feofanov et al., 2026). These systems primarily produce forecasts, predictions, or embeddings. TEMPO focuses on integrating observed signals with natural-language questions and responses.

Signal-to-LLM integration. ChatTS, OpenTSLM, TsLLM, and ITFormer connect time-series encoders to language models through continuous representations, including soft prompts or cross-attention (Xie et al., 2025; Langer et al., 2025; Parker et al., 2025; Wang et al., 2025). Soft prompts can enter the existing self-attention stream without requiring a separate cross-attention module.

Discrete approaches include scalar binning in Chat-TS (Quinlan et al., 2025) and learned vector-quantized representations in TokenCast (Tao et al., 2025). TempoGPT is a closely related approach that quantizes temporal embeddings and shares an embedding layer between temporal and textual tokens (Zhang et al., 2025). Shared discrete vocabularies also support multimodal models for vision and audio (Chameleon Team, 2024; Nguyen et al., 2025).

TEMPO’s distinguishing design choice is a Transformer encoder with global attention over the input window followed by finite scalar quantization (FSQ) (Mentzer et al., 2024). Global attention makes the representation of each patch depend on its surrounding signal context, while FSQ maps these representations onto a fixed scalar grid. TEMPO combines this contextual tokenization with direct vocabulary extension and staged LLM adaptation.

Reasoning over time series. FactoryBench evaluates LLM reasoning over industrial robotic telemetry through questions about system state, interventions, counterfactuals, and decisions (Merzouki et al., 2026). In its zero-shot evaluation of six frontier LLMs, no model exceeds 50% on structured levels or 18% on decision-making. These results highlight substantial difficulties in reasoning over industrial time series and motivate interfaces that integrate signal representations with textual context.

## 3. Method

TEMPO comprises a signal tokenizer, an extended LLM vocabulary, and two adaptation stages (Figure 1). The tokenizer is trained independently of the language model and remains frozen during LLM adaptation.

Preprocessing. For channel $c ,$ we compute its mean $m _ { c }$ and standard deviation $s _ { c }$ over the observed input window and normalize

$$
\hat { x } _ { t , c } = \frac { x _ { t , c } - m _ { c } } { \operatorname* { m a x } ( s _ { c } , \epsilon ) } ,\tag{1}
$$

where ϵ prevents division by zero. Each channel is tokenized separately. Its statistics are included in the text context, preserving scale information alongside the normalized shape representation.

Context-aware tokenization. The tokenizer divides a normalized channel into non-overlapping patches of $L = 4$ time steps and projects them to d = 128 dimensions. A four-layer Transformer encoder applies global self-attention before a projection to four latent dimensions. FSQ quantizes each dimension to one of five levels, giving an implicit vocabulary of $5 ^ { 4 } = 6 2 5$ codes:

$$
z _ { i } = P ( \mathrm { E n c } ( \hat { x } _ { 1 : T } ) _ { i } ) ,\tag{2}
$$

$$
c _ { i } = \mathrm { F S Q } ( z _ { i } ) \in \{ 0 , \ldots , 6 2 4 \} .\tag{3}
$$

Here i indexes patches and P is the latent projection. A symmetric Transformer decoder reconstructs the normalized signal from quantized representations. The tokenizer is pretrained on approximately 500K windows from nine source groups using reconstruction, multi-scale spectral, and temporal-smoothness losses. Training takes approximately four hours on one A10G GPU. Appendix B gives the training configuration.

The encoder can represent a patch differently depending on the surrounding signal. This differs from independent scalar binning and from convolutional encoders whose context is limited by their receptive field. The tokenizer compresses four time steps into each code and uses all 625 codes in the vocabulary-usage analysis.

Vocabulary extension. We add 627 entries: 625 signal codes and two delimiters, <ts start> and <ts end>. New embeddings are initialized as random vectors scaled to the mean norm of existing embeddings. A gradient mask restricts embedding updates to the new entries. A typical input has the form

[user] Signal: (mean=0.12,   
std=0.45)   
<ts start> <ts 207> <ts 14> ...   
<ts end> What fault is present?   
[assistant]

The numbers and token IDs above illustrate the input format. Signal and text tokens use the existing self-attention blocks. The embedding and vocabulary dimensions are expanded, but the Transformer backbone does not require new attention or gating modules.

Stage 0: embedding alignment. Only the 627 new embeddings are updated; the tokenizer, backbone, and adapters are frozen. Approximately 200K examples combine discriminative signal tasks (50%), masked-span code prediction (20%), and M4 captions (30%). The discriminative component covers 22 synthetic signal types and 19 task types, with 20% noise augmentation. This stage associates the discrete signal vocabulary with the language model’s existing representation space.

Stage 1: multi-task adaptation. We update the new embeddings and DoRA adapters (Liu et al., 2024), using rank r = 32 on the attention and MLP projections. The mixture contains approximately 0.26M examples spanning classification, anomaly detection, temporal reasoning, diagnosis, and description. Around 50K text-only instruction examples are included to mitigate language forgetting. Base-model weights and original text embeddings remain frozen.

Task-specific fine-tuning. For the in-distribution evaluations, the Stage 1 checkpoint is further fine-tuned on each domain’s training split. Each domain is evaluated with its corresponding adapted checkpoint. The tokenizer remains fixed throughout.

![](images/787bb94113ecd4d59c25782a36b52569add17716ad42083a6da1249ddf8f15b1.jpg)  
Figure 1. TEMPO architecture and training stages. Tokenizer pretraining learns a discrete signal representation through reconstruction. Stage 0 aligns the new signal embeddings with the LLM; Stage 1 adds DoRA adaptation for time-series tasks. Signal codes, scale statistics, and text form a shared input sequence.

## 4. Experiments

## 4.1. In-Distribution Evaluation

We fine-tune the Stage 1 checkpoint on domain-specific training splits and evaluate on held-out test sets. All TEMPO results in Table 1 use Qwen3-4B. HAR, sleep staging, and bearing diagnosis are evaluated with macro-F1; TSAQA (Jing et al., 2026) classification and anomaly detection use accuracy.

TEMPO reaches 89.0% on HAR, 85.0% on sleep staging, and 86.0% on bearing diagnosis. OpenTSLM with Llama3.2-3B is the closest comparison in model size: Soft-Prompt reaches 64.87% on HAR and 54.40% on Sleep; Flamingo reaches 62.77% and 45.45%. TEMPO exceeds both OpenTSLM configurations on these tasks. Specialized models outperform TEMPO on two of the three classification tasks: random forest reaches 96.0% on HAR, and

WDCNN reaches 99.2% on bearing diagnosis. Closing this gap remains a challenge for the general-purpose signal-tolanguage interface.

On TSAQA, TEMPO achieves 80.5% classification accuracy and 91.3% anomaly-detection accuracy. The tuned LLaMA3.1-8B baseline reaches 91.3% and 91.0%, respectively. With a 4B backbone, half the size of the 8B baseline, TEMPO achieves comparable anomaly-detection accuracy but trails it on classification.

## 4.2. External-Benchmark Evaluation

We evaluate transfer to Time-MQA (Kong et al., 2025) using a smaller Qwen3-1.7B TEMPO model with one-shot prompting on 600 questions. Table 2 compares these results with published baselines.

TEMPO reaches 62% classification accuracy, compared with 52% for Qwen-2.5-7B and 32% for GPT-4o. Its anomaly-detection and judgment scores are lower than both baselines. The lower judgment score is consistent with the capacity gap between TEMPO’s 1.7B backbone and the 7B Qwen baseline. Classification nevertheless remains a strength despite the smaller model size.

Table 1. Downstream task results. HAR, Sleep, and Bearing: macro-F1 (%); TSAQA: accuracy (%). Models use their respective training protocols. TEMPO results are highlighted in bold.
<table><tr><td>Interface / family</td><td>Model</td><td>HAR</td><td>Sleep</td><td>Bearing</td><td>TSAQA cls.</td><td>TSAQA anom.</td></tr><tr><td>Specialized classifier Specialized classifier</td><td>Random forest AttnSleep</td><td>96.0</td><td>63.0 84.4</td><td>96.1</td><td></td><td></td></tr><tr><td>Specialized classifier</td><td>WDCNN</td><td></td><td></td><td>99.2</td><td></td><td></td></tr><tr><td>Text serialization (zero-shot)</td><td>GPT-40</td><td></td><td>15.5†</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>2.95†</td><td></td><td>36.6</td><td>28.8</td><td>55.7</td></tr><tr><td>Text serialization (tuned)</td><td>LLaMA3.1-8B</td><td></td><td></td><td></td><td>91.3</td><td>91.0</td></tr><tr><td>Soft prompt</td><td>Llama3.2-3B†</td><td>64.87</td><td>54.40</td><td></td><td></td><td></td></tr><tr><td>Cross-attention (Flamingo)</td><td>Llama3.2-3B†</td><td>62.77</td><td>45.45</td><td></td><td></td><td></td></tr><tr><td>Discrete vocabulary (TEMPO)</td><td>Qwen3-4B</td><td>89.0</td><td>85.0</td><td>86.0</td><td>80.5</td><td>91.3</td></tr></table>

† OpenTSLM macro-F1 values are from Table 2 of Langer et al. (2025), arXiv v3. GPT-4o HAR/Sleep values are also from OpenTSLM; its remaining scores are from local evaluations.

Table 2. Time-MQA accuracy (%). TEMPO uses one-shot prompting; GPT-4o is zero-shot and Qwen-2.5-7B is a published tuned baseline.
<table><tr><td>Model</td><td>Anom.</td><td>Class.</td><td>Judg.</td></tr><tr><td>GPT-4o†</td><td>64</td><td>32</td><td>72</td></tr><tr><td> $\mathrm { Q w e n } { - 2 . 5 { - 7 } { \bf B } ^ { \dagger } }$ </td><td>68</td><td>52</td><td>82</td></tr><tr><td>TEMPO (1.7B, one-shot)</td><td>56</td><td>62</td><td>52</td></tr></table>

† Baselines reported in Kong et al. (2025).

## 4.3. Cross-Domain Code Sharing

We analyze code activations on held-out windows from six source groups: ECG, financial, forecasting, synthetic, UCR, and UEA data. The tokenizer uses all 625 codes, with normalized entropy $H / H _ { \mathrm { m a x } } = 0 . 8 8 9$ . The cosine similarities between domain-level activation vectors are 0.78 for ECG/UCR and 0.73 for financial/forecasting signals.

The activation statistics show overlapping code usage across signal families, with particularly similar distributions for ECG/UCR and financial/forecasting signals.

## 4.4. Qualitative: Context-Dependent Responses

We examine a Qwen3-4B checkpoint adapted to bearingfault diagnosis using 20K examples. The examples use CWRU signals from an SKF 6205-2RS bearing at 1800 RPM. The following examples illustrate how operational context influences the generated responses.

Context-dependent interpretation. For the same vibration trace, the model diagnoses an outer-race fault when told that the bearing is aged. When told that the part is newly installed and non-OEM, it questions whether the observed features reflect a fault or manufacturing variation. The response thus changes with the supplied operating context while the signal remains fixed.

Table 3. Bearing-fault examples. Diameter is in inches; RMS is in g. ISO zones are supplied textual context, and actions are model-output summaries.
<table><tr><td>Diameter</td><td>RMS</td><td>Zone</td><td>Action summary</td></tr><tr><td>0.007</td><td>0.29</td><td>A</td><td>No immediate action</td></tr><tr><td>0.014</td><td>0.20</td><td>B</td><td>Monitor</td></tr><tr><td>0.021</td><td>0.49</td><td>C</td><td>Plan replacement</td></tr><tr><td>0.028</td><td>0.87</td><td>D</td><td>Replace now</td></tr></table>

Severity-dependent recommendations. Table 3 presents four outer-race fault examples. The recommended action escalates with the ISO zone supplied in the prompt, from monitoring to immediate replacement.

Response to an incorrect prior. In a further example, GPT-4o receives a deliberately incorrect inner-race label together with a numerically serialized signal and produces an explanation consistent with that label. TEMPO instead identifies an outer-race fault in this example.

## 5. Discussion and Conclusion

TEMPO integrates observed time series and text through a shared discrete vocabulary and the LLM’s existing selfattention. Its context-aware tokenizer maps four-step patches to a 625-code vocabulary, while textual statistics preserve scale information. This provides a compact signal interface that can be adapted through new embeddings and DoRA adapters while keeping the tokenizer and base LLM weights frozen.

With task-specific adaptation, TEMPO achieves 89.0%, 85.0%, and 86.0% macro-F1 on activity recognition, sleep staging, and bearing diagnosis. The QA results extend this interface to natural-language questions, and the bearing examples illustrate responses conditioned on operating context. The linear-probe analysis in Appendix A shows that the tokenizer encoder representations retain information useful for classification.

Limitations and future work. Quantization introduces a trade-off between compression and temporal detail, while separate channel tokenization makes sequence length grow with the number of channels. Future directions include channel-aware compression and adaptation to additional signal domains. FactoryBench (Merzouki et al., 2026) offers a natural extension of the evaluation to industrial questions about interventions, counterfactuals, and decisions.

## References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., and Wang, Y. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. URL https: //openreview.net/forum?id=gerNCVqqtR.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. URL https://arxiv.org/abs/2405. 09818.

Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 10148–10167, 2024. URL https://proceedings.mlr.press/ v235/das24c.html.

Feofanov, V., Wen, S., Xie, S., Roschmann, S., Alonso, M., Guo, H., Ilbert, R., Tiomoko, M., Bouniot, Q., Akata, Z., Pan, L., Zhang, J., and Redko, I. Mantis: Lightweight foundation model for time series classification. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pp. 30692–30719, 2026. URL https://proceedings.mlr.press/ v306/feofanov26a.html.

Jing, B., Chen, S., Zheng, L., Liu, B., Li, Z., Zou, J., Wei, T., Liu, Z., Zeng, Z., Qiu, R., Lin, X., Yan, Y., Fu, D., Ni, J., He, J., and Tong, H. TSAQA: Time series analysis question and answering benchmark. arXiv preprint arXiv:2601.23204, 2026. URL https:// arxiv.org/abs/2601.23204.

Kong, Y., Yang, Y., Hwang, Y., Du, W., Zohren, S., Wang, Z., Jin, M., and Wen, Q. Time-MQA: Time series multitask question answering with context enhancement. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29736–29753, 2025. doi: 10.18653/v1/ 2025.acl-long.1437. URL https://aclanthology. org/2025.acl-long.1437/.

Langer, P., Kaar, T., Rosenblattl, M., Xu, M. A., Chow, W., Maritsch, M., Jakob, R., Wang, N., Liu, J., Verma, A., Han, B., Kim, D. S., Chubb, H., Ceresnak, S., Zahedivash, A., Sandhu, A. T. S., Rodriguez, F., McDuff, D., Fleisch, E., Aalami, O., Barata, F., and Schmiedmayer, P. OpenTSLM: Time-series language models for reasoning over multivariate medical text- and timeseries data. arXiv preprint arXiv:2510.02410, 2025. URL https://arxiv.org/abs/2510.02410v3.

Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., and Chen, M.-H. DoRA: Weight-decomposed low-rank adaptation. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32100–32121, 2024. URL https://proceedings.mlr.press/ v235/liu24bn.html.

Mentzer, F., Minnen, D., Agustsson, E., and Tschannen, M. Finite scalar quantization: VQ-VAE made simple. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/ 2309.15505.

Merzouki, Y., Izquierdo, C., Ignuta-Ciuncanu, M., Gomez-Bracamonte, M., Maggioni, R., Lombardi, A., Mazzoleni, C., Martelli, F., Gunther, B., Petersen, J., and Petersen, P. FactoryBench: Evaluating industrial machine understanding. arXiv preprint arXiv:2605.07675, 2026. URL https://arxiv.org/abs/2605.07675.

Nguyen, T. A., Muller, B., Yu, B., Costa-jussa, M. R., El-\` bayad, M., Popuri, S., Ropers, C., Duquenne, P.-A., Algayres, R., Mavlyutov, R., Gat, I., Williamson, M., Synnaeve, G., Pino, J., Sagot, B., and Dupoux, E. SpiRit-LM: Interleaved spoken and written language model. Transactions ofthe Associationfor Computational Linguistics, 13: 30–52, 2025. doi: 10.1162/tacl a 00728. URL https: //aclanthology.org/2025.tacl-1.2/.

Parker, F., Chan, N., Zhang, C., and Ghobadi, K. TsLLM: Augmenting LLMs for general time series understanding and prediction. arXiv preprint arXiv:2510.01111, 2025. URL https://arxiv.org/abs/2510.01111.

Quinlan, P., Li, Q., and Zhu, X. Chat-TS: Enhancing multimodal reasoning over time-series and natural language

data. arXiv preprint arXiv:2503.10883, 2025. URL https://arxiv.org/abs/2503.10883.

Tao, X., Zhang, S., Cheng, M., Wang, D., Pan, T., Pan, B., Zhang, C., and Wang, S. From values to tokens: An LLM-driven framework for context-aware time series forecasting via symbolic discretization. arXiv preprint arXiv:2508.09191, 2025. URL https:// arxiv.org/abs/2508.09191.

Wang, Y., Lei, P., Song, J., Hao, Y., Chen, T., Zhang, Y., Jia, L., Li, Y., and Wei, Z. ITFormer: Bridging time series and natural language for multi-modal QA with large-scale multitask dataset. arXiv preprint arXiv:2506.20093, 2025. URL https://arxiv.org/abs/2506.20093.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. Unified training of universal time series forecasting transformers. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 53140– 53164, 2024. URL https://proceedings.mlr. press/v235/woo24a.html.

Xie, Z., Li, Z., He, X., Xu, L., Wen, X., Zhang, T., Chen, J., Shi, R., and Pei, D. ChatTS: Aligning time series with LLMs via synthetic data for enhanced understanding and reasoning. Proceedings of the VLDB Endowment, 18(8):2385–2398, 2025. doi: 10.14778/3742728. 3742735. URL https://www.vldb.org/pvldb/ vol18/p2385-xie.pdf.

Zhang, H., Yang, C., Han, J., Qin, L., and Wang, X. TempoGPT: Enhancing time series reasoning via quantizing embedding. arXiv preprint arXiv:2501.07335, 2025. URL https://arxiv.org/abs/2501.07335.

## A. Encoder Representation Quality: Linear Probe

We evaluate the tokenizer encoders by mean-pooling their time-series representations and training logistic-regression classifiers on 15 UCR datasets. The encoders remain frozen during probe training. We compare FSQ-T with absolute positional embeddings, CNN+VQ, and FSQ-T with rotary positional embeddings (FSQ-T-RoPE). All 15 probe datasets occur in the tokenizer pretraining corpus, so the comparison measures representation quality under shared dataset exposure. Sample-level overlap is not documented.

Table 4. Encoder linear-probe accuracy and validation loss.
<table><tr><td>Tokenizer</td><td>Mean accuracy (%)</td><td>Validation loss</td></tr><tr><td>FSQ-T (absolute positions)</td><td>73.5</td><td>1.42</td></tr><tr><td>CNN+VQ</td><td>71.4</td><td>0.94</td></tr><tr><td>FSQ-T-RoPE</td><td>69.6</td><td>1.97</td></tr></table>

FSQ-T has the highest mean probe accuracy, exceeding CNN+VQ by 2.1 percentage points and FSQ-T-RoPE by 3.9 points.   
Its validation loss is not the lowest, showing that the ordering by validation loss differs from the ordering by probe accuracy.

![](images/58f23748cc28ac27ead0b44dd042d28c55528197b3b8915c35ddd03d3b6058fe.jpg)  
Figure 2. Per-dataset probe accuracy and FSQ-T’s advantage over each comparator. Values correspond to Table 5. The signed heatmap is centered at zero; orange indicates a positive FSQ-T difference. All 15 datasets were exposed during tokenizer pretraining.

Table 5. Per-dataset linear-probe accuracy (%). Bold marks the highest accuracy in each row; means are unweighted across datasets.
<table><tr><td>Dataset</td><td>FSQ-T</td><td>CNN+VQ</td><td>FSQ-T-RoPE</td></tr><tr><td>ArrowHead</td><td>57.7</td><td>66.9</td><td>55.4</td></tr><tr><td>CBF</td><td>54.9</td><td>60.2</td><td>51.7</td></tr><tr><td>ECG200</td><td>75.0</td><td>69.0</td><td>66.0</td></tr><tr><td>FaceFour</td><td>61.4</td><td>51.1</td><td>53.4</td></tr><tr><td>GunPoint</td><td>92.7</td><td>75.3</td><td>86.7</td></tr><tr><td>Trace</td><td>100.0</td><td>94.0</td><td>92.0</td></tr><tr><td>Wafer</td><td>93.7</td><td>92.4</td><td>94.7</td></tr><tr><td>SyntheticControl</td><td>55.3</td><td>51.7</td><td>39.0</td></tr><tr><td>TwoPatterns</td><td>61.7</td><td>59.5</td><td>63.2</td></tr><tr><td>FaceAll</td><td>42.8</td><td>41.1</td><td>39.2</td></tr><tr><td>StarLightCurves</td><td>92.1</td><td>85.9</td><td>92.8</td></tr><tr><td>Chinatown</td><td>87.5</td><td>86.6</td><td>81.1</td></tr><tr><td>ItalyPowerDemand</td><td>83.1</td><td>74.6</td><td>84.0</td></tr><tr><td>SmoothSubspace</td><td>53.3</td><td>69.3</td><td>58.7</td></tr><tr><td>UMD</td><td>91.0</td><td>93.1</td><td>86.1</td></tr><tr><td>Mean</td><td>73.5</td><td>71.4</td><td>69.6</td></tr><tr><td>Dataset wins</td><td>7</td><td>4</td><td>4</td></tr></table>

The row-wise maxima give seven wins for FSQ-T, four for CNN+VQ, and four for FSQ-T-RoPE, with no ties at the reported precision. FSQ-T’s largest advantage over CNN+VQ is on GunPoint (+17.4 points); its largest disadvantage is on SmoothSubspace (-16.0 points).

## B. Training Details

The following tables summarize the training settings. The tokenizer is frozen during both LLM adaptation stages. Stage 0 updates 627 new embeddings; Stage 1 additionally updates DoRA adapters. The expanded vocabulary contains 625 signa codes and two boundary tokens.

Table 6. Model configurations. The 627 new vocabulary entries comprise 625 signal codes and two delimiters.
<table><tr><td>Parameter</td><td>Qwen3-1.7B</td><td>Qwen3-4B</td></tr><tr><td>Hidden dimension</td><td>2048</td><td>3584</td></tr><tr><td>Transformer layers</td><td>28</td><td>36</td></tr><tr><td>Attention heads</td><td>16</td><td>28</td></tr><tr><td>New vocabulary entries</td><td>627</td><td>627</td></tr><tr><td>DoRA rank / alpha / dropout</td><td>32 / 64 / 0.05</td><td>32 / 64 / 0.05</td></tr><tr><td>DoRA target projections</td><td>q, k, v, o, gate, up, down</td><td></td></tr><tr><td>Signal-code vocabulary</td><td>625</td><td>625</td></tr><tr><td>FSQ levels</td><td>[5, 5, 5, 5]</td><td>[5, 5, 5, 5]</td></tr><tr><td>Patch size (time steps)</td><td>4</td><td>4</td></tr><tr><td>Maximum signal tokens</td><td>750</td><td>750</td></tr><tr><td>Maximum sequence length</td><td>2048</td><td>2048</td></tr></table>

## B.1. Training Phases

Table 7. Hyperparameters for the main LLM-adaptation runs.
<table><tr><td>Parameter</td><td>Stage 0: alignment</td><td>Stage 1: multi-task tuning</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Epochs</td><td>5</td><td>3</td></tr><tr><td>Scheduler</td><td>Linear warmup and linear decay to 0.1 × initial rate</td><td></td></tr><tr><td>Warmup fraction</td><td>10%</td><td>10%</td></tr><tr><td>Batch size per GPU</td><td>8</td><td>8</td></tr><tr><td>Gradient accumulation</td><td>2</td><td>2</td></tr><tr><td>Effective batch size</td><td>128</td><td>128</td></tr><tr><td>Optimizer</td><td>AdamW,  $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ </td><td></td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td></tr><tr><td>Precision</td><td>BF16 mixed precision</td><td>BF16 mixed precision</td></tr><tr><td>Parallelism</td><td>DDP, 8×H100 80GB</td><td>DDP, 8×H100 80GB</td></tr></table>

Stage 0 data. The 200K-example mixture contains 50% discriminative tasks, 20% code imputation, and 30% M4 captions. Discriminative tasks use 22 signal types and 19 question types, with 20% noise augmentation.

Stage 1 data. The mixture lists TSAQA-Benchmark (139K), HAR-CoT (20K), Sleep-CoT (9K), CWRU Bearing-CoT (20K), ECG-QA-CoT (20K), and SlimOrca/Alpaca instructions (50K). The listed components total approximately 0.26M examples.

## B.2. Tokenizer Training

Table 8. Tokenizer configuration and pretraining source categories.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Architecture</td><td>4-layer Transformer encoder, FSQ, 4-layer Transformer decoder</td></tr><tr><td>Model dimension / attention heads</td><td>128 / 4</td></tr><tr><td>FSQ grid</td><td>[5, 5, 5, 5], 625 codes</td></tr><tr><td>Patch size / input length</td><td>4 / 256 time steps</td></tr><tr><td>Training corpus</td><td>Approximately 500K windows from nine source groups</td></tr><tr><td>Source categories</td><td>Forecasting, UCR, bearing, ECG, financial, IMU, synthetic</td></tr><tr><td>Objective</td><td>MSE + multi-scale spectral + temporal-smoothness losses</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW /  $1 0 ^ { - 3 }$ </td></tr><tr><td>Hardware / duration</td><td>One A10G (24GB) / approximately 4 hours</td></tr><tr><td>Observed code utilization</td><td>625/625 (100%)</td></tr><tr><td>Normalized code entropy</td><td> $H / H _ { \mathrm { m a x } } = 0 . 8 8 9$ </td></tr></table>

## B.3. Infrastructure and Cost

Reported costs are approximately \$3 for tokenizer training (four A10G hours), \$49 for Stage 0 (five hours on eight H100s), and \$34 for Stage 1 (3.5 hours on eight H100s). The total is approximately \$86 for the 1.7B pipeline, excluding repeated experiments and downstream specialization.

## B.4. DoRA Rank Observations

In an exploratory bearing-diagnosis study with Qwen3-4B and 186K training examples, rank 16 produced 9 errors in 30 manually inspected examples. Rank 64 reportedly achieved perfect classification but generated only the bearing template. Rank 32 retained conversational responses with a reported score of 86%.