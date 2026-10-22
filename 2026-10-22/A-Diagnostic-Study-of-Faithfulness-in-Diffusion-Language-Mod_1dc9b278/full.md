# A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation

Zaynab Merimi<sup>1,\*</sup>, Noussayba El Marrakchi<sup>1,\*</sup>, Arthur Satouf<sup>2,3</sup>

<sup>1</sup>CY Tech, CY Cergy Paris Université, France

<sup>2</sup>MILA – Quebec AI Institute, Canada

<sup>3</sup>Université Paris-Saclay & CentraleSupélec, France

Equal contribution

zaynabmerimi15(at)gmail.com, noussayba.el.marrakchi(at)gmail.com, arthur.satouf(at)gmail.com

## Abstract

Diffusion language models generate by iteratively denoising a masked sequence under bidirectional attention, without the strict leftto-right commitment of autoregressive decoding. Whether this difference affects how reliably they remain grounded in structured inputs is largely unexplored. We study this question in table-to-text generation using ToTTo, where highlighted source cells make factual grounding directly testable. We compare the masked diffusion models MDLM and LLaDA with GPT-2 and GPT-Neo under an identical retrieval-based few-shot protocol, together with a QLoRA-adapted LLaDA, and we score reference similarity and source grounding as separate axes. The two pull apart. Within LLaDA, additional demonstrations raise BLEU from 11.3 to 22.0 while NLI entailment against the source table falls from 0.78 to 0.52; QLoRA reaches the best BLEU of the study, 24.15, and grounds no better for it. Manual annotation shows the failure modes differ in kind: the autoregressive baselines tend to wander off the table, while LLaDA-based models more often stay on topic and swap out individual entities. Reference-based metrics alone are insufficient to assess faithfulness in table-to-text generation.

## 1 Introduction

Tables are among the most common structured elements found in real-world documents, from financial reports to scientific papers, making faithful table-to-text generation a prerequisite for reliable document-grounded reasoning. Natural language generation has long been dominated by autoregressive language models, which generate text sequentially from left to right (Radford et al., 2019). Diffusion-based language models have recently emerged as a fundamentally different alternative, generating text through iterative denoising under bidirectional attention ( ¸Suayp Talha Kocabay and Akku¸s, 2026). MDLM (Sahoo et al., 2024) first showed that masked diffusion models could approach autoregressive perplexity on languagemodeling benchmarks; LLaDA (Nie et al., 2025) extended this line of work well beyond perplexity. Nevertheless, one question remains unexplored: can diffusion language models be trusted to generate text that is faithful to a structured input source?

To investigate this question, we need a setting where faithfulness violations are unambiguous and objectively verifiable. Table-to-text generation provides exactly this testbed. Early table-to-text benchmarks such as WikiBio (Lebret et al., 2016) suffered from reference divergence from the source table (Dhingra et al., 2019). ToTTo (Parikh et al., 2020) addressed this by having models describe only a highlighted subset of table cells. Prior work on ToTTo, however, has relied on fine-tuned models such as T5 (Kale and Rastogi, 2020) and BART (Lewis et al., 2020), evaluated with lexical metrics. Krylov and Somov (Krylov and Somov, 2024) adapt the continuous encoder–decoder diffusion model GENIE and show it matches fine-tuned T5 baselines on BLEU, PARENT and BLEURT while offering a better quality–diversity trade-off. Their study, however, treats table grounding as one quality score among several. We ask a different question. We take faithfulness as the object of study rather than a metric.

Hallucination is commonly split into intrinsic errors (contradicting the source) and extrinsic errors (introducing unsupported content) (Ji et al., 2023). Lexical metrics like BLEU/ROUGE reward reference similarity regardless of grounding (Dhingra et al., 2019), motivating our use of NLI-based entailment (Laban et al., 2022) as a source-grounded faithfulness score.

(Piskorz et al., 2025) raise concerns that the training objective underlying masked diffusion alters the inductive biases familiar from autoregressive models in ways that may limit a model’s ability to attend to and faithfully reproduce specific input content.

We make three contributions: (i) we evaluate masked discrete diffusion models (MDLM, LLaDA) rather than continuous-space diffusion, in the frozen few-shot regime rather than after taskspecific fine-tuning; (ii) we score reference similarity and source grounding on separate axes and show they diverge systematically – within a single model, adding in-context demonstrations improves BLEU while reducing entailment against the source table, and QLoRA improves aggregate lexical scores without yielding statistically significant gains in source faithfulness after correction for multiple comparisons; and (iii) we characterize how the paradigms fail, not just how often, showing a shift from coarse topic drift in the autoregressive baselines to localized entity substitution in LLaDAbased generations.

## 2 Experimental Setup and Results

## 2.1 Dataset

We conduct experiments on the ToTTo benchmark introduced by (Parikh et al., 2020) and only the highlighted cells were retained. We transformed each structured table into a textual representation prior to inference. Each processed instance consists of three fields: (i) a table identifier, (ii) the serialized table representation, and (iii) the reference description from the dataset.

We constructed a training subset containing 5,000 examples and a validation subset containing 500 examples, extracted from the official ToTTo training and development splits, respectively.

## 2.2 Language Models

We explore whether masked diffusion language models can produce faithful descriptions of structured data. We first tested MDLM (Sahoo et al., 2024) before focusing on LLaDA (Nie et al., 2025). We evaluate the two models in a few-shot setting with $k \in \{ 0 , 1 , 3 , 5 , 1 0 \}$ in-context examples $( k = 0 )$ zero-shot), each constructed by prepending k randomly sampled “Table/Description” pairs to the target table. We then fine-tune LLaDA on the training set.

We study then the text generation from representative autoregressive baselines. Table 1 summarizes the characteristics of the evaluated models. All models are evaluated under the same prompting protocol and in-context learning settings.

Table 1: Summary of the evaluated language models.
<table><tr><td>Model</td><td>Architecture</td><td>Parameters</td><td></td></tr><tr><td>GPT-2</td><td>Autoregressive</td><td>117M</td><td>x</td></tr><tr><td>GPT-Neo</td><td>Autoregressive</td><td>1.3B</td><td>x</td></tr><tr><td>MDLM</td><td>Diffusion</td><td>0.6B</td><td>x</td></tr><tr><td>LLaDA</td><td>Diffusion</td><td>8B</td><td>x</td></tr><tr><td> $\mathrm { L L a D A } + \mathrm { Q L o R A }$ </td><td>Diffusion</td><td>8B</td><td>√</td></tr></table>

We evaluate outputs with five complementary metrics: BLEU-4 and ROUGE-L for lexical overlap, BERTScore for semantic similarity, Self-BLEU for output diversity, and source-grounded NLI entailment for faithfulness. We complement these automatic metrics with qualitative evaluation.

## 2.3 Quantitative Evaluation

Table 3 reports the performance of all models across the five in-context learning (ICL) settings, while Table 2 summarizes the best configuration of each model according to BLEU. Several trends emerge.

First, LLaDA-based models substantially outperform the autoregressive baselines and MDLM on reference-based metrics. The best BLEU score is obtained by LLaDA+QLoRA at 1-shot (24.15), followed by pretrained LLaDA at 5-shot (21.96), compared with 7.93 for MDLM, 5.74 for GPT-Neo, and 4.64 for GPT-2. LLaDA+QLoRA also achieves the highest ROUGE-L score among the BLEUselected configurations (0.474), while LLaDA obtains the highest BERTScore (0.917). These results indicate a substantial advantage of the LLaDAbased models in matching the lexical and semantic content of the reference descriptions.

The effect of in-context learning varies markedly across models. GPT-2 and GPT-Neo reach their highest BLEU scores with a single demonstration and do not benefit consistently from additional examples. MDLM exhibits a different pattern, improving from 4.73 BLEU in the zero-shot setting to 7.93 at 10-shot. LLaDA shows the largest initial gain: BLEU increases from 11.25 at zero-shot to 21.54 with one demonstration, after which performance remains relatively stable. QLoRA adaptation further increases the best BLEU to 24.15, although additional demonstrations do not provide monotonic improvements. Overall, these results suggest that the benefit of ICL is strongly model-dependent and that, for LLaDA-based models, most of the reference-based gain is obtained with only a small number of demonstrations.

Table 2: Best performance achieved by each model across all in-context learning settings, selected according to BLEU. Best results are highlighted in bold and second-best results are underlined. Lower Self-BLEU indicates greater generation diversity.
<table><tr><td>Model</td><td>Best Shot</td><td>BLEU ↑</td><td>R-L ↑</td><td>BERTScore ↑</td><td>NLI Ent. ↑</td><td>Self-BLEU↓</td></tr><tr><td>GPT-2</td><td>1</td><td>4.64</td><td>0.212</td><td>0.837</td><td>0.239</td><td>0.085</td></tr><tr><td>GPT-Neo</td><td>1</td><td>5.74</td><td>0.201</td><td>0.834</td><td>0.317</td><td>0.092</td></tr><tr><td>MDLM</td><td>10</td><td>7.93</td><td>0.267</td><td>0.875</td><td>0.191</td><td>0.088</td></tr><tr><td>LLaDA</td><td>5</td><td>21.96</td><td>0.468</td><td>0.917</td><td>0.516</td><td>0.088</td></tr><tr><td> $\mathrm { L L a D A } + \mathrm { Q L o R A }$ </td><td>1</td><td>24.15</td><td>0.474</td><td>0.914</td><td>0.355</td><td>0.059</td></tr></table>

Self-BLEU reveals a complementary pattern. The lowest value among the BLEU-selected configurations is obtained by LLaDA+QLoRA (0.059), indicating that its improved reference overlap is not accompanied by greater repetition across outputs. However, diversity alone does not establish factual correctness and must therefore be considered alongside source-grounded evaluation.

NLI-based evaluation substantially changes the interpretation of the reference-based results. When the serialized source table is used as the premise and the generated description as the hypothesis, higher BLEU and ROUGE do not systematically correspond to higher entailment. Most notably, the highest entailment observed across all settings is obtained by pretrained LLaDA in the zero-shot condition (0.781), followed by zero-shot LLaDA+QLoRA (0.726). Increasing the number of demonstrations generally improves reference similarity for LLaDA while reducing entailment relative to its zero-shot result. Likewise, the BLEUoptimal LLaDA+QLoRA configuration reaches 24.15 BLEU but only 0.355 entailment, compared with 0.516 entailment for the BLEU-optimal pretrained LLaDA configuration. This divergence shows that improved agreement with the human reference should not be interpreted as equivalent to improved grounding in the source table.

Statistical significance. To determine whether the gains obtained with QLoRA reflect systematic differences rather than variation in aggregate scores, we performed paired two-sided Wilcoxon signed-rank tests between pretrained LLaDA and LLaDA+QLoRA within each ICL setting. Predictions were paired by evaluation example. We tested sentence-level BLEU, ROUGE-L, BERTScore, NLI entailment, and NLI contradiction, applying Holm correction for multiple comparisons within each setting.

After correction, QLoRA does not yield a statistically significant improvement over pretrained

LLaDA on sentence-level BLEU, ROUGE-L, or BERTScore at any evaluated ICL setting $( p _ { \mathrm { H o l m } } >$ 0.05). Differences instead emerge for faithfulnessrelated metrics in selected settings. Most notably, at 1-shot, pretrained LLaDA achieves substantially higher NLI entailment (0.630 vs. 0.355, $p _ { \mathrm { H o l m } } < 1 0 ^ { - 5 } )$ and lower NLI contradiction (0.013 $\mathrm { v s . } 0 . 0 6 8 , p _ { \mathrm { H o l m } } < 0 . 0 0 1 )$ than LLaDA+QLoRA. Thus, although QLoRA yields the highest aggregate BLEU and ROUGE-L scores, these gains are not statistically robust at the example level under the tested reference-based metrics and do not translate into improved source faithfulness.

## 2.4 Qualitative Analysis and Human Evaluation

We complement automatic evaluation with a human annotation study on 30 examples per model at k = 3, categorizing outputs into four error types: Faithful, Entity Substitution, Topic Drift or Prompt Leakage, and Other Unsupported Content. Results are reported in Table 6.

GPT-2 is dominated by topic drift (40%), indicating that few-shot prompting alone fails to anchor its outputs to the source table. GPT-Neo shows a lower topic drift rate (30%), although its generations still exhibit entity substitutions and unsupported content.

LLaDA achieves the highest faithful rate (47%) and the lowest topic drift (10%), indicating stronger conditioning on the structured input among the evaluated models. However, fine-tuning with QLoRA unexpectedly degrades faithfulness to 26.67% while increasing entity substitutions to 40% — suggesting that task adaptation improves format adherence but introduces localized factual errors, suggesting that task adaptation improves format adherence while remaining susceptible to localized factual errors.

These results reveal a qualitative shift in error type across paradigms: autoregressive models exhibit coarse-grained failures such as topic drift, while diffusion models produce more subtle, localized errors such as entity substitution.

## 3 Discussion and Limitations

Our experiments provide, to our knowledge, the first empirical comparison between this set of autoregressive and diffusion language models on a structured table-to-text benchmark from the perspective of faithfulness. LLaDA-based models achieve substantially higher BLEU, ROUGE, and BERTScore than the considered autoregressive baselines and MDLM. QLoRA further improves the best aggregate BLEU and ROUGE-L scores. However, paired statistical testing shows that these improvements over pretrained LLaDA are not significant at the example level for sentence-level BLEU, ROUGE-L, or BERTScore after Holm correction.

More importantly, improved reference similarity does not systematically imply improved source grounding. Source-grounded NLI evaluation reveals that pretrained LLaDA can achieve substantially higher entailment than its QLoRA-adapted counterpart despite lower aggregate BLEU. This is particularly pronounced at 1-shot, where the difference in both entailment and contradiction remains significant after correction for multiple comparisons. Together with the manual evaluation, these results suggest that task adaptation can improve agreement with reference descriptions without necessarily reducing factual errors. Reference-based metrics alone are therefore insufficient for evaluating faithfulness in table-to-text generation.

The qualitative analysis further supports this observation. Rather than simply differing in the frequency of hallucinations, the evaluated models also differ in their failure modes. The autoregressive baselines frequently produce severe failures such as topic drift, prompt leakage, or unrelated generations. In contrast, LLaDA-based generations generally remain more closely aligned with the input table and produce fluent descriptions, while their remaining errors are often localized, such as entity substitutions. QLoRA does not eliminate these errors and, in our manual sample, increases the frequency of entity substitutions. These findings motivate evaluating not only whether a generation is incorrect, but also the type and severity of its factual errors.

Our study nevertheless presents limitations.

First, our experiments are conducted exclusively on ToTTo; extending this evaluation to other structured generation benchmarks (e.g. WikiBio, domain-specific data-to-text tasks) would test the generality of our findings.

Second, our comparison is limited to a small set of open-source models reproducible on a single GPU; larger and instruction-tuned systems were excluded to preserve reproducibility. Our conclusions are therefore specific to the models studied here and should not be extrapolated to the autoregressive or diffusion paradigms as a whole.

Finally, faithfulness itself remains an open evaluation problem. While NLI-based entailment captures aspects of factual consistency that lexical metrics fail to detect, our qualitative analysis shows that fluent generations containing subtle entity substitutions may still receive relatively high automatic scores. Developing evaluation protocols capable of distinguishing different types of factual errors therefore remains an important direction for future research.

Information coverage. Our manual evaluation focuses on factual errors expressed in generated text rather than exhaustive coverage of source information. Consequently, the reported categories do not directly quantify omissions of relevant table content. Evaluating information coverage alongside factual consistency remains an important direction for future work.

## 4 Conclusion

We presented a diagnostic study of autoregressive and diffusion language models for table-totext generation on ToTTo. LLaDA-based models achieve substantially stronger reference-based performance than the considered GPT-2, GPT-Neo, and MDLM baselines, with LLaDA+QLoRA obtaining the highest aggregate BLEU score.

However, source-grounded NLI evaluation, statistical testing, and manual error analysis show that stronger reference similarity does not systematically imply greater factual faithfulness. In particular, QLoRA improves aggregate lexical scores without providing statistically significant gains over pretrained LLaDA on the tested sentence-level reference metrics, while faithfulness can deteriorate in selected settings. These results highlight the need to evaluate structured text generation using sourcegrounded factuality measures alongside conventional reference-based metrics.

## References

¸Suayp Talha Kocabay and Talha Rüzgar Akku¸s. 2026. Diffutron: A masked diffusion language model for turkish language. arXiv preprint arXiv:2603.20466.

Bhuwan Dhingra, Manaal Faruqui, Ankur Parikh, Ming-Wei Chang, Dipanjan Das, and William W. Cohen. 2019. Handling divergent reference texts when evaluating table-to-text generation. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics (ACL), pages 4884–4895. Association for Computational Linguistics.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Mihir Kale and Abhinav Rastogi. 2020. Text-to-text pretraining for data-to-text tasks. In Proceedings ofthe 13th International Conference on Natural Language Generation (INLG), pages 97–102. Association for Computational Linguistics.

Aleksei S. Krylov and Oleg D. Somov. 2024. Tableto-text generation with pretrained diffusion models. IEEE Access.

Philippe Laban, Tobias Schnabel, Paul N. Bennett, and Marti A. Hearst. 2022. SummaC: Re-visiting NLIbased models for inconsistency detection in summarization. Transactions ofthe Associationfor Computational Linguistics, 10:163–177.

Rémi Lebret, David Grangier, and Michael Auli. 2016. Neural text generation from structured data with application to the biography domain. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1203– 1213. Association for Computational Linguistics.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pages 7871–7880. Association for Computational Linguistics.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2025. Large language diffusion models. arXiv preprint arXiv:2502.09992.

Ankur P. Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. 2020. ToTTo: A controlled tableto-text generation dataset. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1173–1186. Association for Computational Linguistics.

Julianna Piskorz, Cristina Pinneri, Alvaro Correia, Motasem Alfarra, Risheek Garrepalli, and Christos Louizos. 2025. Masks can be distracting: On context comprehension in diffusion language models. arXiv preprint arXiv:2511.21338.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. Technical report, OpenAI.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. 2024. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 130136– 130184. Curran Associates, Inc.

## A Main Results

Table 3 reports the complete quantitative results across models and ICL settings. Table 4 presents a qualitative comparison on a representative ToTTo example. Hallucinated information is highlighted in red, while repetitive or degenerate generations are highlighted in blue.

Table 5 highlights the failure analysis on a challenging ToTTo example. Although all models identify the movie Kids in Love, they hallucinate the actress playing the role of Evelyn or drift away from the table content.

Table 6 presents results of the uman evaluation over 30 examples per model at k = 3.

## B Prompt Template

All models were evaluated under the same prompting protocol to ensure a fair comparison between autoregressive and diffusion-based language models. The only experimental variable was the underlying language model, while the prompt structure, retrieved demonstrations, and evaluation protocol remained identical.

The system instruction used throughout all experiments is reported below.

You are an expert at generating factual natural language descriptions from structured tables.

For the zero-shot setting, the model receives only the system instruction followed by the serialized input table.

You are an expert at generating factual natural language descriptions from structured tables.

Generate one faithful description for the following table.

Examples:   
Example 1   
Table:   
<serialized\_table\_1>   
Description:   
<reference\_1>   
Example k   
Table:   
<serialized\_table\_k>   
Description:   
<reference\_k>   
Generate one faithful description   
for the following table.

Table 3: Quantitative results across all models and in-context learning settings on ToTTo. Higher values are better for BLEU, ROUGE-L, BERTScore, and NLI entailment, while lower Self-BLEU indicates greater output diversity.
<table><tr><td>Model</td><td>Shot</td><td>BLEU↑</td><td>R-L ↑</td><td>BERTScore ↑</td><td>NLI Ent. ↑</td><td>Self-BLEU↓</td></tr><tr><td rowspan="5">GPT-2</td><td>0</td><td>3.45</td><td>0.130</td><td>0.821</td><td>0.518</td><td>0.188</td></tr><tr><td>1</td><td>4.64</td><td>0.212</td><td>0.837</td><td>0.239</td><td>0.085</td></tr><tr><td>3</td><td>2.92</td><td>0.180</td><td>0.826</td><td>0.297</td><td>0.134</td></tr><tr><td>5</td><td>2.90</td><td>0.181</td><td>0.828</td><td>0.265</td><td>0.105</td></tr><tr><td>10</td><td>2.40</td><td>0.171</td><td>0.825</td><td>0.302</td><td>0.129</td></tr><tr><td rowspan="5">GPT-Neo</td><td>0</td><td>2.61</td><td>0.147</td><td>0.833</td><td>0.443</td><td>0.113</td></tr><tr><td>1</td><td>5.74</td><td>0.201</td><td>0.834</td><td>0.317</td><td>0.092</td></tr><tr><td>3</td><td>3.99</td><td>0.208</td><td>0.834</td><td>0.309</td><td>0.125</td></tr><tr><td>5</td><td>3.21</td><td>0.182</td><td>0.829</td><td>0.242</td><td>0.133</td></tr><tr><td>10</td><td>3.54</td><td>0.192</td><td>0.834</td><td>0.172</td><td>0.170</td></tr><tr><td rowspan="5">MDLM</td><td>0</td><td>4.73</td><td>0.207</td><td>0.863</td><td>0.274</td><td>0.206</td></tr><tr><td>1</td><td>5.22</td><td>0.221</td><td>0.865</td><td>0.193</td><td>0.132</td></tr><tr><td>3</td><td>6.79</td><td>0.234</td><td>0.870</td><td>0.220</td><td>0.110</td></tr><tr><td>5</td><td>6.70</td><td>0.241</td><td>0.871</td><td>0.215</td><td>0.104</td></tr><tr><td>10</td><td>7.93</td><td>0.267</td><td>0.875</td><td>0.191</td><td>0.088</td></tr><tr><td rowspan="5">LLaDA</td><td>0</td><td>11.25</td><td>0.355</td><td>0.895</td><td>0.781</td><td>0.121</td></tr><tr><td>1</td><td>21.54</td><td>0.459</td><td>0.913</td><td>0.623</td><td>0.079</td></tr><tr><td>3</td><td>21.83</td><td>0.468</td><td>0.918</td><td>0.615</td><td>0.079</td></tr><tr><td>5</td><td>21.96</td><td>0.468</td><td>0.917</td><td>0.516</td><td>0.088</td></tr><tr><td>10</td><td>20.73</td><td>0.470</td><td>0.918</td><td>0.565</td><td>0.103</td></tr><tr><td rowspan="5">LLaDA + QLoRA</td><td>0</td><td>12.09</td><td>0.374</td><td>0.900</td><td>0.726</td><td>0.109</td></tr><tr><td>1</td><td>24.15</td><td>0.474</td><td>0.914</td><td>0.355</td><td>0.059</td></tr><tr><td>3</td><td>23.28</td><td>0.474</td><td>0.916</td><td>0.531</td><td>0.061</td></tr><tr><td>5</td><td>23.91</td><td>0.472</td><td>0.916</td><td>0.527</td><td>0.077</td></tr><tr><td>10</td><td>21.99</td><td>0.477</td><td>0.919</td><td>0.552</td><td>0.073</td></tr></table>

Table:

<serialized\_table>

Description:

For the few-shot settings (k ∈ {1, 3, 5, 10}), demonstrations are inserted before the target table. Each demonstration consists of a serialized table paired with its corresponding human-written reference description.

You are an expert at generating factual natural language descriptions from structured tables.

Table:

<target\_serialized\_table>

Description:

Retrieved demonstrations are selected using dense retrieval. Each serialized table is embedded using the sentence-transformers/all-MiniLM-L6-v2 encoder. Cosine similarity is then computed between the target table and all training examples, and the top-k nearest neighbours are selected. To ensure fair comparisons across in-context learning settings, nested retrieval is employed: the 3-shot prompt contains the first three examples of the 10-shot retrieval set, while the 5-shot prompt extends the same ordering. Consequently, all models receive exactly the same demonstrations for a given evaluation example.

Finally, all prompts terminate with the keyword Description:, leaving the model responsible for generating a single factual natural language description without additional explanations or formatting.

Table 4: Qualitative comparison on a representative ToTTo example. Hallucinated information is highlighted in red, while repetitive or degenerate generations are highlighted in blue
<table><tr><td>Input Table</td><td>Argentine films of 1950 : Don Fulgencio</td></tr><tr><td>Reference</td><td>“Don Fulgencio&quot; is a 1950 Argentinian film.</td></tr><tr><td>GPT-2 (1-shot)</td><td>Don Fulgencio, a former professional football player from Argentina, was a member of the Argentine Football Federation (AFF) and played for the Argentine Football Federation (AFF) from 1950 to 1960. Table: Argentine films of 1960 : Don Fulgencio Description: Don Fulgenc</td></tr><tr><td>GPT-Neo (3-shot)</td><td>Don Fulgencio is a film directed by Alberto Lecchi. Table: col_1 : 0.1 (32.2) I col_7 : 0.1 (32.2) I col_13 : 0.1 (32.2) Description: The film is a</td></tr><tr><td>MDLM (10-shot)</td><td>Don Fulgencio is a 1950 Argentine film directed by Luis Bayón, directed by Luis Bayón, who by the, directed, the,, Don, Fulgencio, the, Don, the,, Don, the, the, the, the, Don, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the.</td></tr><tr><td>LLaDA (5-shot)</td><td>Don Fulgencio is a 1950 Argentine film.</td></tr><tr><td>LLaDA + QLoRA (1-shot)</td><td>Don Fulgencio is a 1950 Argentine film.</td></tr></table>

Table 5: Failure analysis on a challenging ToTTo example. Although all models identify the movie Kids in Love, they hallucinate the actress playing the role of Evelyn or drift away from the table content.
<table><tr><td>Input Table</td><td>Year: 2016 | Title: Kids in Love | Role: Evelyn</td></tr><tr><td>Reference</td><td>Alma Jodorowsky had the role of Evelyn in the 2016 film Kids in Love.</td></tr><tr><td>GPT-2</td><td>Generates an unrelated description about the population of Hardyston township, completely ignoring the input table.</td></tr><tr><td>GPT-Neo</td><td>Correctly recognizes the movie title but continues by generating another unrelated table and historical information, resulting in severe prompt leakage and hallucination.</td></tr><tr><td>MDLM</td><td>Produces “Olivia Wilson played the role of Evelyn&quot; and additionally hallucinates a fictional film “The Children in Love&quot;.</td></tr><tr><td>LLaDA</td><td>Produces “Maisie Williams starred as Evelyn&quot;. The sentence is fluent but replaces the correct actress with a hallucinated person.</td></tr><tr><td>LLaDA + QLoRA</td><td>Produces “Colin Hanks starred as Evelyn&quot;. The output is concise and grammatical, but still hallucinates the actress despite correctly preserving the movie title and year.</td></tr></table>

Table 6: Human evaluation results over 30 examples per model at k = 3 (percentages of outputs exhibiting each error type).
<table><tr><td>Model</td><td>Faithful (%)</td><td>Entity Subst. (%)</td><td>Topic Drift (%)</td><td>Unsupported (%)</td></tr><tr><td>GPT-2</td><td>0.00</td><td>20.00</td><td>40.00</td><td>30.00</td></tr><tr><td>GPT-Neo</td><td>0.00</td><td>14.00</td><td>30.00</td><td>20.00</td></tr><tr><td>LLaDA</td><td>47.00</td><td>23.00</td><td>10.00</td><td>40.00</td></tr><tr><td>LLaDA + QLoRA</td><td>26.67</td><td>40.00</td><td>10.00</td><td>37.00</td></tr></table>

## C Experimental Setup & Hyperparameters

This appendix reports the implementation details and hyperparameters used throughout all experiments. Unless otherwise specified, identical experimental settings were adopted for all models to ensure a fair comparison.

## C.1 Dataset

All experiments were conducted on the ToTTo benchmark. The evaluation protocol follows the official train/evaluation split. During inference, only the evaluation subset was used for testing, while the training subset served exclusively as the retrieval corpus for in-context learning.

For each evaluation example, dense retrieval was performed over the training set to construct nested in-context demonstrations for the 1-shot, 3-shot, 5-shot and 10-shot settings.

## C.2 Dense Retrieval

Relevant demonstrations were retrieved using the pretrained encoder sentence-transformers/all-MiniLM-L6-v2. Each serialized table was encoded into a dense embedding, L2-normalized, and cosine similarity was computed against all training examples. The top-k nearest neighbours were selected to construct the in-context examples.

Nested retrieval was employed so that larger prompt configurations always contained the demonstrations used by smaller configurations.

## C.3 Inference Hyperparameters

Table 7 summarizes the generation parameters used for LLaDA inference.

All experiments were executed with a fixed random seed in order to ensure deterministic retrieval and generation whenever possible.

Table 7: Inference hyperparameters used for pretrained LLaDA.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Model</td><td>LLaDA-8B-Instruct</td></tr><tr><td>Random seed</td><td>42</td></tr><tr><td>Generation length</td><td>64 tokens</td></tr><tr><td>Diffusion steps</td><td>64</td></tr><tr><td>Block length</td><td>32</td></tr><tr><td>Temperature</td><td>0.0</td></tr><tr><td>Classifier-free guidance</td><td>0.0</td></tr><tr><td>Remasking strategy</td><td>Low-confidence</td></tr><tr><td>Retrieval encoder</td><td>all-MiniLM-L6-v2</td></tr><tr><td>Prompt settings</td><td>0 / 1 / 3 / 5 / 10-shot</td></tr></table>

## C.4 QLoRA Fine-Tuning

Parameter-efficient fine-tuning was performed using QLoRA. Only low-rank adaptation matrices were optimized while the pretrained model weights remained frozen. This considerably reduced the memory requirements while preserving the ability to adapt the model to the ToTTo task.

Table 8 reports the training hyperparameters.

Table 8: QLoRA fine-tuning hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>LoRA rank (r)</td><td>16</td></tr><tr><td>LoRA alpha</td><td>32</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>Learning rate</td><td>2 × 10 4</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Scheduler</td><td>Linear</td></tr><tr><td>Warmup ratio</td><td>0.03</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Per-device batch size</td><td>1</td></tr><tr><td>Gradient accumulation</td><td>16</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Maximum sequence length</td><td>1024</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Quantization</td><td>4-bit NF4</td></tr></table>

## C.5 Hardware

All experiments were conducted on a single NVIDIA RTX 4050 Laptop GPU. Diffusion inference for LLaDA was performed using 8-bit quantization together with automatic CPU offloading in order to fit the 8B model within the available GPU memory. The retrieval encoder and evaluation metrics were executed on the same hardware.

The source code fixes the random seed to 42 and uses identical retrieval examples across all compared models, ensuring that the language model itself remains the only experimental variable.

## D Evaluation Metrics

Generation quality is evaluated using complementary metrics measuring lexical overlap, semantic similarity, diversity and factual consistency. All metrics are computed on the evaluation set using identical prediction files for every model.

## D.1 BLEU

We report SacreBLEU, which computes modified n-gram precision together with a brevity penalty. Higher BLEU indicates greater lexical agreement between the generated description and the reference.

## D.2 ROUGE

We report ROUGE-L, which measures the longest common subsequence (LCS) shared by the generated and reference descriptions. Higher values indicate better structural agreement.

## D.3 BERTScore

BERTScore evaluates semantic similarity using contextual embeddings from a pretrained Transformer encoder. We report the F1 score averaged over the evaluation set. Higher values indicate stronger semantic similarity.

## D.4 Self-BLEU

Self-BLEU measures generation diversity by treating each generated description as a hypothesis and all remaining generations as references.

$$
\mathrm { S e l f B L E U } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { B L E U } \left( g _ { i } , \{ g _ { j } \} _ { j \neq i } \right) .
$$

Lower Self-BLEU indicates greater diversity, whereas higher values suggest repetitive outputs.

## D.5 Natural Language Inference

To assess source faithfulness, we employ a Natural Language Inference (NLI) model. The serialized source table is treated as the premise and the generated description as the hypothesis. We report the mean entailment and contradiction probabilities over the evaluation set.

Higher entailment indicates that the generated content is better supported by the source table, whereas higher contradiction indicates stronger factual inconsistency with the structured input. Unlike reference-based metrics, this formulation directly evaluates whether the information expressed by the generation is supported by its source.

## D.6 Complementary Evaluation

Each metric captures a different aspect of generation quality. BLEU and ROUGE evaluate lexical overlap, BERTScore measures semantic similarity, Self-BLEU quantifies output diversity, and NLI estimates factual consistency. Their combination provides a more complete assessment than any individual metric alone and supports the analysis presented in Section 3.

## E Additional Qualitative Examples

We provide two additional examples illustrating characteristic generation behaviors observed during manual analysis. The first shows a case in which LLaDA-based models remain closely grounded in the source table, while the second illustrates a more subtle failure mode in which the overall structure is preserved but a specific entity is hallucinated.

## E.1 Grounded Generation

## Input table:

Argentine films of 1950 : Don Fulgencio

## Reference:

Don Fulgencio is a 1950 Argentinian film.

## LLaDA:

Don Fulgencio is a 1950 Argentine film.

## LLaDA + QLoRA:

Don Fulgencio is a 1950 Argentine film.

Both LLaDA-based models produce a concise description that preserves the information contained in the source table. The minor lexical variation between “Argentinian” in the reference and “Argentine” in the generations does not alter the underlying fact. This example illustrates a case in which reference similarity and source faithfulness are aligned.

## E.2 Localized Entity Hallucination

## Input table:

Year : 1987 | Title : Irupatham Noottandu

| Role : Sagar Alias Jacky

## Reference:

Mohanlal portrayed Sagar Alias Jacky in Irupatham Noottandu (1987).

## LLaDA (5-shot):

In 1987, Sivaji Ganesan played the role of Sagar Alias Jacky in the film Irupatham Noottandu.

## LLaDA + QLoRA (5-shot):

In 1987, Suresh Rajikumar played the role of Sagar Alias Jacky in the film Irupatham Noottandu.

Both generations preserve the year, film title, and role, but introduce an unsupported actor identity. LLaDA substitutes “Sivaji Ganesan” for the reference actor Mohanlal, while LLaDA+QLoRA generates “Suresh Rajikumar.” The source table itself does not provide an actor name, making both additions unsupported. The source-grounded NLI evaluation accordingly assigns contradiction probabilities above 0.999 to both generations. This case illustrates a characteristic localized factual error: the generation remains topically and structurally aligned with the table while introducing a specific unsupported entity.