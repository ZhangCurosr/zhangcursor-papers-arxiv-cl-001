# Benchmarking Arabic Dialects for Math Problems

Noor Abo Mokh,<sup>1</sup> Teresa Lynn,<sup>1</sup> Nizar Habash,<sup>1,2</sup> Ahmed Elshabrawy,<sup>1</sup>

Sara Alhajeri,<sup>1</sup> Alaa Elsetohy,<sup>1</sup> Preslav Nakov,<sup>1</sup> Alham Fikri Aji<sup>1</sup>

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence,

<sup>2</sup>New York University Abu Dhabi

{noor.abomokh,alham.fikri}@mbzuai.ac.ae

## Abstract

Mathematical reasoning remains underexplored for low-resource languages, including Dialectal Arabic. To bridge this gap, we introduce the DAR-MGSM (Dialectal Arabic Multilingual Grade School Math) Benchmark, the first manually curated benchmark for evaluating mathematical reasoning in Dialectal Arabic. DAR-MGSM consists of math word problems (MWPs) requiring multi-step reasoning and covers five major Arabic dialects alongside Modern Standard Arabic (MSA). The benchmark comprises two parallel versions: translated-only and translated & culturally localized. The latter version adapts entities and references to regional varieties. We evaluate eleven Arabic-centric and multilingual large language models on DAR-MGSM. The results show that the performance generally declines on the localized version, highlighting the challenges introduced by culturally grounded adaptations. However, the magnitude of this effect varies across models, with Arabic-centric models exhibiting different levels of sensitivity to localization. A mechanistic interpretability analysis shows that translated and localized variants of the same MWPs often induce distinct internal reasoning trajectories, even when aggregate accuracy remains similar. Overall, our results show that mathematical reasoning performance depends on both linguistic variety and cultural context.

## 1 Introduction

Large language models (LLMs) have demonstrated strong performance on complex tasks like mathematical reasoning in resource-rich languages such as English (Cobbe et al., 2021; Hendrycks et al., 2021a,b) and French (Hoang Ha, 2025). However, these capabilities do not consistently extend to lowresource languages (E Sobhani et al., 2026; Huang et al., 2025), nor to languages with numerous variants. Arabic is a prominent example: despite its rich dialectal diversity, most existing evaluation mathematical benchmarks focus on Modern Standard Arabic (MSA).

![](images/20becca83b07359d23c5b4ed499f4afe7bfab3461ba0c2f851bad549612d2d7d.jpg)  
Figure 1: An illustrative Math Word Problem adapted from MGSM-Rev2 in three versions: (i) Original English, (ii) Translated to MSA, and (iii) Localized to Egyptian Arabic dialect and entities.

The focus on MSA is understandable. Math textbooks and educational materials across the Arab world are written in MSA. Yet, teachers use Dialectal Arabic (DA) extensively in instruction. Amin and Badreddine (2020) document the extensive use of dialect in science classes, while MSA is used in textbooks. Other studies such as Amara (1996) mention extensive use of dialect in math classes to explain concepts.

This reflects Arabic’s diglossic nature (Ferguson, 1959), an critical challenge for LLM evaluation. Existing benchmarks focusing on MSA provide only a partial view of mathematical reasoning capabilities in Arabic. To bridge this gap, we aim to develop an LLM math word problem (MWP) benchmark that reflects classroom real-world conditions in the Arab world.

Leveraging existing benchmarks, we translate MWPs from English into Arabic, to further allow us to investigate LLM biases across MSA, and Arabic dialects. While translation captures linguistic differences, it often creates texts alien to Arab culture. American English MWPs, for instance, commonly use US-centric entities such as Western personal names and locations, or currencies, even when presented to Arabic-speaking users.<sup>1</sup> See Figure 1 for an illustrative example.

Such choices influence how naturally problems are interpreted and processed.

To investigate these challenges, we distinguish between two orthogonal benchmark dimensions: linguistic translation, which adapts the language of a problem, and cultural entity alignment or localization, which adapts culturally grounded entities for names, locations, currencies, and numeral conventions. This distinction enables a controlled study of whether culturally aligned benchmark variants elicit different model behavior than direct translations of the same mathematical problem.

Additionally, we benchmark localization with mechanistic interpretability, providing insight not only into whether localization affects performance, but also how it influences internal model reasoning processes.

Our contributions are as follows:

• We introduce DAR-MGSM (Dialectal Arabic Multilingual Grade School Math Benchmark), the first benchmark for evaluating grade-school mathematical reasoning in MSA and five major Arabic varieties: Egyptian (EGY), Emirati (UAE), Jordanian (JOR), Moroccan (MOR), and Saudi (KSA).

• We release two parallel benchmark variants: a manually translated dataset and a translated & culturally localized dataset.

• We conduct the first systematic evaluation of dialectal Arabic mathematical reasoning, benchmarking multiple Arabic-centric and multilingual LLMs.

• We use mechanistic interpretability to compare model behavior on translated and localized data.

Our results show that performance generally declines on localized data compared to direct translations, with varying effects across a variety of models and dialects. The interpretability analysis reveals that localization alters LLM reasoning, highlighting the need for culturally grounded evaluation in Arabic and other multilingual settings, where translation alone may mask differences in model capabilities.

## 2 Related Work

Mathematical reasoning benchmarks Evaluation of LLMs on mathematical reasoning has expanded substantially beyond English in recent years. One of the widely used benchmarks is GSM8K (Cobbe et al., 2021) containing approximately 8,500 grade-school MWPs. Building on GSM8K, Shi et al. (2023) introduced MGSM, extending to ten additional languages translated from English. MGSM-Rev2 (Cobbe et al., 2021; Shi et al., 2023; Peter et al., 2026) is a revised version of MGSM. MGSM8KInstruct (Chen et al., 2024), is a multilingual instruction-following dataset built on MGSM. Other reasoning benchmarks, such as GPQA (Rein et al., 2024), remain predominantly English-centric. In the above benchmarks, Arabic is typically represented only through MSA, but not dialects. We bridge this gap by evaluating on language varieties, providing parallel grade-school math benchmarks for MSA and five major Arabic dialects.

Arabic mathematical reasoning Within the Arabic NLP landscape, mathematical reasoning resources remain comparatively limited. Multilingual evaluation suites such as BenchMAX (Huang et al., 2025) and MathMist (E Sobhani et al., 2026) include MSA as one of several evaluation languages. More recently, Alenezi and Salman (2026) introduced large-scale Arabic mathematical reasoning datasets, including AGSM8K, Qudurat, and ArabicMWPs, where AGSM8K consists of GSM8K-style problems translated into MSA. ArMATH (Alghamdi et al., 2022) provides a collection of manually authored MWPs in MSA designed to support the development of automatic MWP solvers. Other dialectal benchmarks focus on related reasoning tasks, such as multiple-choice question answering (Altakrori et al., 2026), rather than open-ended mathematical problem solving. To the best of our knowledge, no benchmark currently evaluates grade-school mathematical reasoning across the Arabic dialects. Consequently, the effect of dialectal variation on LLM mathematical reasoning capabilities remains largely unexplored.

In contrast, we provide a benchmark specifically designed for evaluation covering multiple dialects.

Translation vs. localization A methodological limitation shared by many multilingual reasoning benchmarks is a reliance on direct translation from English. For example, MGSM was intentionally translated without localization to facilitate crosslingual comparison (Shi et al., 2023). Similarly, AGSM8K (Alenezi and Salman, 2026) relies on machine translation followed by manual review. Recent work has argued that translation alone is insufficient for culturally grounded evaluation. Studies in multilingual and culturally aware NLP demonstrate that direct translations can obscure important sociocultural differences and reduce linguistic naturalness, motivating the need for localization in benchmark design (Farsi et al., 2025; Zhou et al., 2025). This issue is particularly relevant for Arabic, where dialects are closely tied to specific regional and cultural contexts. Our work builds on these insights by explicitly separating translation and localization as two distinct dimensions of the benchmark. Unlike prior Arabic mathematical reasoning benchmarks, we provide both translated and culturally localized versions of the same MWPs. This design enables controlled evaluation of the impact of localization on mathematical reasoning performance while preserving the underlying problem structure.

Multilingual reasoning challenges Recent studies have shown that multilingual reasoning remains challenging for current LLMs. Models often achieve higher accuracy when reasoning in English than when reasoning directly in other languages, reflecting the predominance of English in training and optimization pipelines (Saji et al., 2026). Particularly, Saji et al. (2026) demonstrate that reasoning in English yields higher final-answer accuracy than reasoning in many other languages. These findings motivate our investigation of localized Arabic MWPs, where linguistic variation and cultural alignment can be studied independently. By comparing translated and localized variants of the same MWP, we isolate the effect of cultural grounding on mathematical reasoning performance and better understand how benchmark design influences model behavior.

Residual stream interpretability Mechanistic interpretability often treats the transformer residual stream as a shared computational workspace, where attention heads and MLPs read from and write to common subspaces (Elhage et al., 2021). Methods such as tuned lens probe intermediate residual representations, while causal tracing, activation patching, and path patching test whether specific hidden states or model components are necessary for a given behavior (Belrose et al., 2023; Meng et al., 2022; Olsson et al., 2022; Nanda et al., 2023). More recent work studies reasoning as trajectories through residual-stream geometry and decomposes representations using sparse autoencoders and superposition analyses (Elhage et al., 2022; Bricken et al., 2023; Templeton et al., 2024; Manson, 2025; Li et al., 2025). While prior work has primarily used these methods to study reasoning, factual recall, and circuit-level behavior, little attention has been paid to multilingual or culturally grounded evaluation settings. Here we bridge this gap.

## 3 Dataset Curation

We extended MGSM-Rev2 (Cobbe et al., 2021; Shi et al., 2023; Peter et al., 2026), to include MSA<sup>2</sup> and five Arabic Dialects: Egyptian, Emirati, Jordanian, Moroccan, Saudi. MGSM-Rev2 comprises 250 selected MWPs from the GSM8K English benchmark (Cobbe et al., 2021). The English versions were the source text for translation for all the multilingual variants in MGSM and MGSM-Rev2 and also represent our starting point. The MWP from MGSM-Rev2 in Figure 1 demonstrates the US-centric nature of the original benchmark data. We therefore undertake both a translation and a localization approach to curating the data. Below, we describe both versions: standalone and separately.

## 3.1 Manual Translation

We hired professional translators to translate the MGSM-Rev2 250 MWPs into MSA and the five Arabic dialects. Both a dedicated translator and a reviewer were assigned to each Arabic variety (see Appendix A for additional information on the annotators). As an additional quality check, native speakers conducted an in-house review of the professional translations. Correction rates ranged from 4.0% for Saudi Arabic to 49.6% for MSA, mostly due to misinterpretation of English expressions encoding mathematical relationships (See Appendix A).

<table><tr><td>ENG</td><td>&lt;name&gt; has an egg farm in &lt;location&gt;. His chickens produce &lt;num1&gt; eggs per day and he sells them for &lt;num2&gt; &lt;currency&gt; per dozen. How much does he make on eggs per week?</td></tr><tr><td>MSA</td><td>u1&gt;. location&gt;name&gt; r </td></tr><tr><td>EGY</td><td>. oatio e&gt;   </td></tr><tr><td>JOR</td><td>&gt; . ai    </td></tr><tr><td>KSA</td><td>i  &gt; </td></tr><tr><td>MOR</td><td>.oati a&gt; </td></tr><tr><td>UAE</td><td> &gt;  . t       </td></tr></table>

Table 1: Examples from DAR-MGSM localizations of MWPs for EGY, JOR, KSA, MOR, UAE, and MSA.
<table><tr><td></td><td>ENG</td><td>Translated</td><td>MSA</td><td>EGY</td><td>JOR</td><td>KSA</td><td>MOR</td><td>UAE</td></tr><tr><td>&lt;name&gt;</td><td>Lloyd</td><td>Lloyd</td><td>Musa</td><td>Metwally</td><td>Walid</td><td>Abdul Rahim</td><td>Ibrahim</td><td>Mumin</td></tr><tr><td>&lt;location&gt;</td><td>Toulouse</td><td>j Toulouse</td><td>j Toulouse</td><td>Cairo</td><td></td><td></td><td>Marrakech</td><td>Al-Ain</td></tr><tr><td>&lt;num1&gt;</td><td>252</td><td>252</td><td>YOY</td><td>YOY</td><td>Irbid YOY</td><td>Riyadh YOY</td><td>252</td><td>YOr</td></tr><tr><td>&lt;num2&gt;</td><td>5</td><td>5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5</td><td>0</td></tr><tr><td>&lt;currency&gt;</td><td>dollar</td><td>dollar</td><td></td><td></td><td></td><td>J</td><td></td><td></td></tr></table>

Table 2: Entity realizations across English, translated Arabic, MSA, EGY, JOR, KSA, MOR, and UAE.

The translation guidelines emphasized that entities such as places, names, and currencies should remain in the original form, i.e., should not be localized (see examples in Table 2). Additionally, we advised that dialectal translations should be natural with minimal overlap with MSA. For borrowed words, we asked annotators to adhere to linguistic forms used naturally in their dialect. For instance, the word for email should be ÉJÖ<sup>ß@</sup>. We had also allowed for spelling variation, where the meaning remained intact (see translation guidelines in Appendix B). The final translated dataset consists of 250 MWPs per dialect, i.e., a total of 1,500 MWPs covering six varieties.

## 3.2 Localization

Since direct translation would fail to incorporate cultural alignment with the dialects, we augmented the translated data with a semi-automated localization step, by replacing math question’s entities with region-specific equivalents. We identified four entity types to be localized: names, locations, currencies, and numerals. See Table 2 for examples of entity replacements.

The localizable entities were semi-automatically replaced throughout the MSA dataset with placeholders (e.g., [person\_m] for a human male name). Anchoring these placeholders to the translated text maintains structural and logical integrity, which is critical for such tasks. When multiple entities of the same category appeared in a single MWP, we assigned a unique index i to each entity variable to distinguish between different instances within the same category. As an example, if a MWP contained more than one male name, we assigned each name an index (e.g., [person\_m\_1], [person\_m\_2]). This indexing scheme preserves the distinctions between the entities while maintaining consistent placeholder types. We then projected the placeholders onto the dialectal translations using a structured heuristic. The final phase of localization involved an automatic mapping of the variable placeholders to region-specific replacements. We replaced the names and the locations with regionally relevant counter-parts. We also adjusted the financial units to reflect country-specific currencies. Additionally, we localized numerical values and percentage signs to the Arabic-Indic numeral system for all dialects except Moroccan which uses Arabic numeral system. This ensured that both the linguistic and the mathematical components of the MWP aligned culturally with the target regions (see examples in Tables 1 and 2). A total of 51 MWPs did not have entities requiring localization (Appendix C for more details). The dataset went through a final human validation step. The name category had the most issues with 47 wrong name replacement across all dialects (less than 1% for all replacements). For example, our automatic scripts assumed that a name would appear once in the math problem, which led to failed replacements in cases where a name was mentioned twice or more. We show the surface differences induced by each localization scheme in Table 8.

## 4 Experimental Setup

We experimented with several Arabic-centric and multilingual LLMs on their ability to answer MWPs from our dataset (for more details, see Appendix D). We evaluated the following multilingual models: Qwen-3.6, Gemma4 models, and GPT-5.4. We further evaluated the following Arabiccentric ones on dialectal models such as Atlas-Chat, NileChat, and Jais that provide support for different dialects. We used OpenRouter (OpenRouter, 2026) where applicable, otherwise relying on a local GPU for inference. The models were prompted as follows:

Solve the provided math problem. End with ‘The answer is: [number]’.

We then extracted the numerical values and computed accuracy against a reference answer. Results were normalized by removing thousand separators. We note that Moroccan uses periods as thousand separators while other dialects use commas. We experimented with individual replacements. We manually audited model outputs to assess whether answer-extraction failures contributed to the reported errors. Section 5.3 describes the audit, and Appendix H provides the category breakdown.

## 5 Results

## 5.1 Benchmark Results

The overarching takeaway is that cultural localization exposes model fragility that direct translation conceals.

Overall evaluation results on DAR-MGSM Table 3 shows that the accuracy of Arabic-centric and multilingual models degrades when localized entities are introduced. This degradation can vary by model size and dialect. Individual replacements show only slight changes compared to all replacements.

Degradation can vary depending on model size While most models perform worse on localized data, flagship models like GPT 5.4 and Gemma 4 only show slight degradation. We observe that performance improves as the model size increases.

Localizations can impact a model’s ability to correctly answer a MWP In Figure 2, we also observe that across models, the accuracy drops for all Arabic varieties. Performance appears to vary depending on the type of culturally-aware replacement (see Tables 10 and 11). For example, Mary vs Fatema, dirhams vs \$ or ’dollar’, and so on. This reveals that adding localizations can impact on the model’s ability to answer the MWP correctly.

Certain dialects are more vulnerable Results in Table 10 and Table 11 also reveal consistent model under-performance across both the translated and localized Moroccan MWPs in particular.

The largest drops in accuracy between translation-only and localization occur for MSA and Egyptian (-3.3 points each).

Statistical significance varies across different models Table 3 shows that, for flagship models that are large size, the results are not statistically significant.<sup>3</sup> This does not indicate invariance, as upon further investigation, the models do not incorrectly answer the same questions across the translated and the localized MWPs. In smaller models such as Nile-Chat (Shang et al., 2025a) and Atlas-Chat (Shang et al., 2025b), however, the differences between translated and localized are statistically significant. This can indicate that these Arabiccentric models are more sensitive to regional differences. One possible explanation is that localizing MWPs can introduce challenges for these smaller models, and may result in performance degradation. Thus, the accuracy results do not provide a unified result: while some models show little change when introducing localizations, the MWPs that the models could correctly answer can differ. We further investigated this behavior focusing on Gemma 4 models through interpretability study, to understand the models behavior under localization.

<table><tr><td></td><td colspan="2">MSA</td><td colspan="2">EGY</td><td colspan="2">JOR</td><td colspan="2">KSA</td><td colspan="2">MOR</td><td colspan="2">UAE</td><td colspan="2">Average</td></tr><tr><td></td><td>T</td><td>L</td><td>T</td><td>L</td><td>T</td><td>L</td><td>T</td><td>L</td><td></td><td>T L</td><td>T</td><td>L</td><td>T L</td></tr><tr><td>Atlas-Chat-2B</td><td>28.8</td><td>23.6</td><td>22.8</td><td>13.2</td><td>24.8</td><td>20.8</td><td>21.6</td><td>18.4</td><td>24.4</td><td>23.6</td><td>28.0</td><td>20.0</td><td>25.1 19.9</td></tr><tr><td>Nile-Chat-4B</td><td>38.0</td><td>28.4</td><td>35.2</td><td>32.4</td><td>36.8</td><td>31.2</td><td>35.2</td><td>24.4</td><td>15.2</td><td>14.8</td><td>836.4 36.0</td><td></td><td>32.8 27.9</td></tr><tr><td>Atlas-Chat-9B</td><td>66.0</td><td>61.6</td><td>55.2</td><td>52.0</td><td>61.6</td><td>58.4</td><td>56.4</td><td>55.2</td><td>58.4</td><td>60.0</td><td>62.461.2</td><td></td><td>60.058.1</td></tr><tr><td>c4ai-r7b</td><td>72.4</td><td>69.2</td><td>67.2</td><td>62.4</td><td>68.0</td><td>65.6</td><td>62.8</td><td>57.2</td><td>54.4</td><td>55.2</td><td>71.672.8</td><td></td><td>66.1 63.7</td></tr><tr><td>Jais2-8B-Chat</td><td>68.8</td><td>68.8</td><td>66.4</td><td>53.2</td><td>70.8</td><td>65.6</td><td>62.0</td><td>54.8</td><td>56.4</td><td></td><td>54.4 70.0</td><td>67.2</td><td>65.7 60.7</td></tr><tr><td>Qwen2.5-7B</td><td>75.6</td><td>69.6</td><td>69.6</td><td>63.6</td><td>72.0</td><td>72.0</td><td>62.0</td><td>65.2</td><td>38.8</td><td>38.4 76.8</td><td></td><td>74.0</td><td>65.8 63.8</td></tr><tr><td>Gemma4-2B</td><td>73.6</td><td>71.6</td><td>66.8</td><td>69.2</td><td>67.6</td><td>66.8</td><td>63.2</td><td>61.6</td><td>50.4</td><td>45.2</td><td>73.2 74.0</td><td></td><td>65.8 64.7</td></tr><tr><td>Gemma4-4B</td><td>84.4</td><td>81.2</td><td>80.0</td><td>79.6</td><td>82.8</td><td>84.0</td><td>75.2</td><td>73.2</td><td>66.4</td><td>66.8</td><td>85.683.2</td><td></td><td>79.1 78.0</td></tr><tr><td>Gemma4-26B</td><td>85.6</td><td>86.0</td><td>88.0</td><td>87.6</td><td>90.8</td><td>90.4</td><td>78.4</td><td>80.0</td><td>80.4</td><td>79.6</td><td>91.6</td><td>90.0</td><td>85.8 85.6</td></tr><tr><td>Qwen-3.6</td><td>86.4</td><td>84.8</td><td>87.2</td><td>87.6</td><td>91.2</td><td>90.4</td><td>78.0</td><td>76.8</td><td>79.2</td><td>80.0</td><td>91.2</td><td>90.0</td><td>85.5 84.9</td></tr><tr><td>GPT-5.4</td><td>90.0</td><td>89.2</td><td>89.2</td><td>90.0</td><td>93.2</td><td>94.8</td><td>85.6</td><td>86.4</td><td>84.4</td><td>84.8</td><td>92.0</td><td>93.2</td><td>89.1 89.7</td></tr><tr><td>Average</td><td></td><td>70.066.7</td><td></td><td>66.1 62.8</td><td></td><td>69.1 67.3</td><td></td><td></td><td></td><td>61.9 59.4 55.3 54.8 70.8 69.2</td><td></td><td></td><td>65.5 63.4</td></tr></table>

Table 3: Accuracy (%) of 11 models on translated (T) and localized (L) problems for MSA and five dialects. Bold: the higher of T and L. Underlined: T and L differ significantly (McNemar’s test, $p < 0 . 0 5 )$ .

![](images/e11739b81a3937046010ae4fb0b92d4a2bfe96ba6fb84ad94471e714b6a51c0b.jpg)  
Figure 2: Performance by region and localization.

## 5.2 Interpretability

We investigate the impact of localizations on a model’s ability to answer MWP correctly. Our hypothesis is that the model’s internal residual-stream trajectory changes, and so does the answer.

Sampling robustness of localization effects For each problem, we compare a baseline prompt (Translated) against localized variants (Localized) on the Gemma 4 models only for tractability. We divide examples into four groups: both correct, both wrong, baseline correct/localized wrong, and baseline wrong/localized correct. We refer to the latter two groups as discordant examples. We compute Root Mean Square (RMS) trajectory distances by model, dialect, localization scheme, layer, and correctness group to whether localized and non-localized versions diverge when correctness changes.

We use activation patching to test whether localization-relevant activations causally drive the observed discordant behavioral shifts. We patch entity-span representations in both directions and check if the model’s answer and residual trajectory shift toward the source condition. We compare source patching to controls: no patching, self-patching, shuffled non-localized sources, and patching an unrelated same-length span (See Appendix G and Figure 3). We then tested whether localization effects could be explained by ordinary stochastic decoding. We generated three samples for the translated condition and three for the localized condition and compared seed-to-seed residualstream RMS distances against matched translatedto-localized RMS distances.

![](images/abd2d552e055d8dd398819a059c125eb4e63d2495ecf4d48c8723f1324004623.jpg)

![](images/07732710eed7a84de2a0d1b0fedeb2ff0a1f2420aed8ca9839db369029eddb2e.jpg)  
Figure 3: Activation patching correctness by baseline. Source patching partially recovers source-condition behavior on discordant examples.

Localization changes which examples are solved Table 3 shows that aggregate accuracy is broadly stable across localization schemes for Gemma 4 models. However, Table 13 shows that paired outcomes are not stable. Under localization, models produce 316 discordant outcomes over 3,582 localizable pairs, corresponding to a discordance rate of 8.8%, while the mean accuracy change is -1.2. Thus, aggregate accuracy alone masks the effect: localization moves individual examples between the correct and the incorrect sets.

Smaller models are more sensitive, but larger models are not invariant The discordance effect decreases with model scale. Table 4 shows that, under full localization, Gemma4-2B has the highest discordance rate at 12.9%, while Gemma4-26B has the lowest at 5.6%. Larger models are therefore more stable under localization, but they exhibit paired examples where the baseline and localized prompts differ in correctness. This supports the idea that localizations can alter model predictions even for larger models.

Correctness flips coincide with large residualstream trajectory shifts We next ask whether paired behavioral changes correspond to changes in internal computation. Figure 5 and Table 12 summarize the main mechanistic result: examples that remain correct in both conditions have small residual-stream distances, while examples that flip correctness have much larger distances (Figure 4 support the same conclusion). Pairs correct in both baseline and localized forms have mean RMS around 0.29.

<table><tr><td>Model</td><td>N</td><td>Discord.</td><td>Discord. Rate</td><td>Mean Acc. ∆</td></tr><tr><td>Gemma4-2B</td><td>1,194</td><td>154</td><td>12.9%</td><td>-1.7 pp</td></tr><tr><td>Gemma4-4B</td><td>1,194</td><td>95</td><td>8.0%</td><td>-1.8 pp</td></tr><tr><td>Gemma4-26B</td><td>1,194</td><td>67</td><td>5.6%</td><td>-0.3 pp</td></tr></table>

Table 4: Model-level paired discordance under full localization. Smaller models are more sensitive, but localization effects remain present across scales. We exclude the 51 non-localizable MWPs that have differences in the numerals used and have slight discordance; however, we find it more interesting to study discordance among examples with bigger localization differences.

In contrast, pairs where only the baseline is correct have mean RMS around 1.50, and pairs where only the localized version is correct have mean RMS around 1.50. Pairs wrong in both conditions are intermediate, with mean RMS around 1.22. This supports the hypothesis that when localization changes the answer, it also moves the model onto a different residual-stream trajectory.

Table 14 shows that matched translated-tolocalized residual-stream distances exceed withintranslated seed-to-seed distances in 17 of 18 model– dialect cells (although Gemma4-2B MSA is very close to a tie at 1.004). The effect is most pronounced for Gemma4-26B, whose localizationto-sampling RMS ratios range from 1.07 to 1.45. Gemma4-2B shows smaller shifts, with ratios from 1.004 to 1.214, while Gemma4-4B is more variable and falls below 1 only for MSA (0.97). Localization correctness-flip rates exceed sampling flip rates in 13 of 18 cells, are equal in one, and are lower in four. Thus, localization generally changes residual trajectories and correctness more than seed variation alone, although the magnitude depends on the model and dialect. These results suggest that the localization effect is not explained solely by stochastic decoding, but they do not independently establish a causal mechanism.

Activation patching supports a causal role for entity-span representations As shown in Figure 3, RMS shifts show that answer flips coincide with trajectory changes, but do not establish causality. We therefore patch residual representations at the changed localization spans in both directions on discordant examples. Source patching outperforms the control-span baseline, reaching 54.7% best-layer correctness and a 0.31 best-layer sourcecloseness gain, compared with 35.7% and 0.22 for unrelated-span patching. This supports a causal role for representations in shifting the target toward both the source answer and source residual trajectory, although it does not isolate an exclusively example-specific semantic mechanism.

The effect is localized to entity spans, but not uniquely to matched entity identity The shuffled-source baseline is also strong, reaching 52.3% best-layer correctness and a 0.31 best-layer source- closeness gain, close to the matched sourcepatching results of 54.7% and 0.31. Because this control patches changed-span activations from mismatched examples, its performance suggests that recovery is not driven solely by transferring the exact matched name, location, currency, or numeral representation. We interpret the results as evidence that representations at localized spans are causally relevant, while the small gap between matched and shuffled sources leaves unresolved whether the operative signal encodes specific entity identity, a general source-condition representation, or a broader residual-state feature associated with those spans.

To summarize our results, these mechanistic shifts reveal that culturally adapting a MWP does not merely alter its surface vocabulary—it fundamentally changes the model’s reasoning pathway. This proves that benchmarks relying solely on translated text may mask vulnerabilities in a model’s real-world problem-solving capabilities.

## 5.3 Error Analysis

To further understand the results, we perform manual analysis of errors in UAE and JOR by categorizing the errors into five categories: (a) Reasoning/setup error, when the model is providing incorrect interpretation from the beginning (e.g., units); (b) Final step error, where reasoning is correct but the final step is wrong.; (c) Missing step, when a model fails to include a necessary step.; (d) localization error, when the model struggles with localized entities; and (e) translation error. Among the analyzed errors, reasoning/setup errors such as misinterpreting the percentage/proportional relationships is the most frequent category, or incorrect interpretation of comparative expressions such as ’double’. We observed arithmetic errors, such as providing the wrong sum. In Figure 6, the first example 175 + 140 + 280 is 595 not 695.

Some examples produced alternative but internally consistent interpretations of the problem statement. This could suggest that Some errors arise from ambiguity in the problem formulation rather than from incorrect reasoning. Interestingly, in the UAE dataset, we observe that incorrect MWP is because the reasoning language is Chinese (see examples in Appendix H from JOR and UAE).

## 6 Conclusion and Future Work

We introduced DAR-MGSM, the first manually curated benchmark for mathematical reasoning across five Arabic dialects and Modern Standard Arabic, with parallel translated-only and culturally localized versions. Our evaluation of eleven LLMs showed that localization generally reduces accuracy and can substantially change which problems models solve correctly, even when overall performance remains similar. Interpretability analysis linked these changes to shifts in internal representations. These findings highlight the importance of evaluating mathematical reasoning across both linguistic varieties and cultural contexts.

In future work, we plan to expand DAR-MGSM’s coverage and to investigate methods for improving reasoning consistency across equivalent problem formulations. We also aim to extend this evaluation approach to other multilingual settings, supporting the development of educational AI that work reliably across languages and local contexts.

## Limitations

DAR-MGSM covers five Arabic dialects and Modern Standard Arabic, with results scoped to these varieties and the evaluated models. Cultural localization focuses on names, locations, currencies, and numeral conventions rather than all aspects of cultural context. Localization applies only to problems containing relevant entities, so comparisons for individual localization categories use the corresponding subsets.

## Ethics

DAR-MGSM is derived from MGSM-Rev2, which is distributed under the CC BY-SA 4.0 license. We use and adapt the source data in accordance with its license terms and provide attribution to its creators. The benchmark is intended to support research on mathematical reasoning across Arabic varieties and cultural contexts.

## References

Reem Alenezi and Ayed Atallah Salman. 2026. On LLM prompting techniques for Arabic language arithmetic reasoning. In The 7th Workshop on Open-Source Arabic Corpora and Processing Tools (OS-ACT7) with 5 Shared Tasks, pages 106–114, Palma, Mallorca (Spain). Association for Computational Linguistics.

Reem Alghamdi, Zhenwen Liang, and Xiangliang Zhang. 2022. ArMATH: a Dataset for Solving Arabic Math Word Problems. In Proceedings of the Language Resources and Evaluation Conference, pages 351–362, Marseille, France. European Language Resources Association.

Malik H. Altakrori, Nizar Habash, Teresa Lynn, Younes Samih, Abed Alhakim Freihat, Kirill Chirkunov, Muhammed AbuOdeh, Radu Florian, Preslav Nakov, and Alham Fikri Aji. 2026. DialectalArabicMMLU: Benchmarking dialectal capabilities in Arabic and multilingual language models. In Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026), pages 3199–3219, Palma, Mallorca, Spain. European Language Resources Association (ELRA).

Muhammad Hasan Amara. 1996. Arabic Diglossia in the Classroom Assumptions and Reality. In Language and culture in the Near East, pages 131–142. Brill.

Tamer Amin and Diala Badreddine. 2020. Teaching science in Arabic: diglossia and discourse patterns in the elementary classroom. International Journal of Science Education, 42(14):2290–2330.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. 2023. Eliciting Latent Predictions from Transformers with the Tuned Lens. CoRR, abs/2303.08112.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, and 6 others. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Technical report, Anthropic.

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Dongmei Zhang, and Jia Li. 2024. Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 7001–7016, Miami, Florida, USA. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Tasnim Mohiuddin, Md Mofijul Islam, and Swakkhar Shatabda. 2026. MathMist: A parallel multilingual benchmark dataset for mathematical problem solving and reasoning. In Findings of the Association for Computational Linguistics: EACL 2026, pages 2524–2550, Rabat, Morocco. Association for Computational Linguistics.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. Technical report, Anthropic.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, and 6 others. 2021. A Mathematical Framework for Transformer Circuits. Technical report, Anthropic.

Farhan Farsi, Farnaz Aghababaloo, Shahriar Shariati Motlagh, Parsa Ghofrani, MohammadAli Sadraei-Javaheri, Shayan Bali, Amir Hossein Shabani, Farbod Bijary, Ghazal Zamaninejad, AmirMohammad Salehoof, and Saeedeh Momtazi. 2025. MELAC: Massive Evaluation of Large Language Models with Alignment of Culture in Persian Language. In Proceedings ofthe 14th International Joint Conference

on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 1933–1950, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Charles A Ferguson. 1959. Diglossia. Word, 15(2):325– 340.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring Mathematical Problem Solving with the MATH Dataset. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks.

Huy Hoang Ha. 2025. Pensez: Moins de données, meilleur raisonnement – repenser les LLM français. In Actes des 32ème Conférence sur le Traitement Automatique des Langues Naturelles (TALN), volume 1 : articles scientifiques originaux, pages 573–598, Marseille, France. ATALA & ARIA.

Xu Huang, Wenhao Zhu, Hanxu Hu, Conghui He, Lei Li, Shujian Huang, and Fei Yuan. 2025. BenchMAX: A Comprehensive Multilingual Evaluation Suite for Large Language Models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 16751–16774, Suzhou, China. Association for Computational Linguistics.

Bo Li, Guanzhi Deng, Ronghao Chen, Junrong Yue, Shuo Zhang, Qinghua Zhao, Linqi Song, and Lijie Wen. 2025. REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model. Preprint, arXiv:2509.22518.

Rob Manson. 2025. Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams. Preprint, arXiv:2507.21107.

Quinn McNemar. 1947. Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. Psychometrika, 12(2):153–157.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. 2023. Progress Measures for Grokking via Mechanistic Interpretability. In International Conference on Learning Representations.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, and 7 others. 2022. Incontext Learning and Induction Heads. Technical report, Anthropic.

OpenRouter. 2026. OpenRouter: Unified API for Large Language Models. https://openrouter.ai. Accessed: 2026-05-22.

Jan-Thorsten Peter, David Vilar, Tobias Domhan, Dan Malkin, and Markus Freitag. 2026. Mind the gap... or not? How translation errors and evaluation details skew multilingual results. In Proceedings ofthe Fifth Workshop on Generation, Evaluation and Metrics (GEM), pages 191–204, San Diego, California, USA. Association for Computational Linguistics.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In Proceedings of the First Conference on Language Modeling: COLM 2024, Philadelphia, PA, USA.

Alan Saji, Raj Dabre, Anoop Kunchukuttan, and Ratish Puduppully. 2026. The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 329–344, Rabat, Morocco. Association for Computational Linguistics.

Guokan Shang, Hadi Abdine, Ahmad Chamma, Amr Mohamed, Mohamed Anwar, Abdelaziz Bounhar, Omar El Herraoui, Preslav Nakov, Michalis Vazirgiannis, and Eric P. Xing. 2025a. Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts. In Proceedings of The Third Arabic Natural Language Processing Conference, pages 306– 322, Suzhou, China. Association for Computational Linguistics.

Guokan Shang, Hadi Abdine, Yousef Khoubrane, Amr Mohamed, Yassine Abbahaddou, Sofiane Ennadir, Imane Momayiz, Xuguang Ren, Eric Moulines, Preslav Nakov, Michalis Vazirgiannis, and Eric P. Xing. 2025b. Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect. In Proceedings of the First Workshop on Language Models for Low-Resource Languages, pages 9–30, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language Models Are Multilingual Chain-of-Thought Reasoners. In International Conference on Learning Representations.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, and 3 others. 2024. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Technical report, Anthropic.

Naitian Zhou, David Bamman, and Isaac L. Bleaman. 2025. Culture is Not Trivia: Sociocultural Theory for Cultural NLP. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25869– 25886, Vienna, Austria. Association for Computational Linguistics.

## A Annotator Demographics

<table><tr><td>ID</td><td>Dataset</td><td>Native</td><td>Residence</td><td>Age</td><td>Gender</td><td>Degree</td><td>Task</td><td>Background</td></tr><tr><td>A1</td><td>MSA</td><td>EGY</td><td>EGY</td><td>30s</td><td>F</td><td>PhD</td><td>T</td><td>L, LA</td></tr><tr><td>A2</td><td>MSA</td><td>UAE</td><td>UAE</td><td>40s</td><td>F</td><td>MA</td><td>R</td><td>L, CL</td></tr><tr><td>A3</td><td>MSA</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr><tr><td>A4</td><td>EGY</td><td>EGY</td><td>EGY</td><td>30s</td><td>F</td><td>PhD</td><td>T</td><td>L, LA</td></tr><tr><td>A5</td><td>EGY</td><td>EGY</td><td>UAE</td><td>40s</td><td>F</td><td>MA</td><td>R</td><td>L, CL</td></tr><tr><td>A6</td><td>EGY</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr><tr><td>A7</td><td>JOR</td><td>JOR</td><td>JOR</td><td>30s</td><td>M</td><td>BA</td><td>T</td><td>L, LA</td></tr><tr><td>A8</td><td>JOR</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>R</td><td>CL</td></tr><tr><td>A9</td><td>JOR</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr><tr><td>A10</td><td>KSA</td><td>KSA</td><td>KSA</td><td>20s</td><td>F</td><td>BA</td><td>T</td><td>CS</td></tr><tr><td>A11</td><td>KSA</td><td>KSA</td><td>KSA</td><td>30s</td><td>F</td><td>BA</td><td>R</td><td>LA</td></tr><tr><td>A12</td><td>KSA</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr><tr><td>A13</td><td>MOR</td><td>MOR</td><td>MOR</td><td>20s</td><td>M</td><td>MA</td><td>T</td><td>CS</td></tr><tr><td>A14</td><td>MOR</td><td>MOR</td><td>MOR</td><td>30s</td><td>M</td><td>BA</td><td>R</td><td>LA</td></tr><tr><td>A15</td><td>MOR</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr><tr><td>A16</td><td>UAE</td><td>UAE</td><td>US</td><td>40s</td><td>F</td><td>PhD</td><td>T</td><td>LA</td></tr><tr><td>A17</td><td>UAE</td><td>UAE</td><td>UAE</td><td>20s</td><td>M</td><td>MS</td><td>R</td><td>LA</td></tr><tr><td>A18</td><td>UAE</td><td>PAL</td><td>UAE</td><td>30s</td><td>F</td><td>PhD</td><td>V</td><td>CL</td></tr></table>

Table 5: Annotator demographics and roles for the translation task. There are three roles: Translate (T), Review (R), and Validate (V). The annotator background experience includes: Linguistics (L), Linguistic Annotator (LA), Computational Linguist (CL) and Computer Science (CS). All annotators are native speakers of Arabic, and they worked on their own native dialect or a dialect spoken in a country where they resided in. Annotators were paid \$15/hr.

<table><tr><td>Variety</td><td>Corrected / Total</td><td>Rate</td></tr><tr><td>MSA</td><td>124/250</td><td>49.6%</td></tr><tr><td>Egyptian</td><td>41/250</td><td>16.4%</td></tr><tr><td>Jordanian</td><td>77/250</td><td>30.8%</td></tr><tr><td>Moroccan</td><td>90/250</td><td>36.0%</td></tr><tr><td>Emirati</td><td>70/250</td><td>28.0%</td></tr><tr><td>Saudi</td><td>10/250</td><td>4.0%</td></tr></table>

Table 6: Number and percentage of problems requiring major edits or retranslation during internal quality assurance after handling the translations by our external vendor. Counts refer to problems, rather than individual edits, and describe corrections during dataset construction. These figures describe corrections during dataset curation, rather than residual error rates in the final benchmark. We conducted an additional internal quality assurance pipeline.

## B Translation Guidelines

These are the guidelines as they were presented to our annotators.

## Math Translation Guidelines

This task involves translating math problems in English into Arabic dialects and reviewing them for accuracy, fluency and naturalness. Each translation task is performed by two translators: one responsible for the translation, the other for reviewing it. See below for the translator and reviewer tasks for clear instructions (see example sheet in the appendix below).

You are provided with a translation request in an Excel sheet with four columns as detailed below:

<table><tr><td>Column</td><td>Content</td><td>Description</td></tr><tr><td>A</td><td>English source text</td><td>This is the original source text in English</td></tr><tr><td>B</td><td>Dialect Translation</td><td>This is to add the translation in your dialect</td></tr><tr><td>C</td><td>Reject/Accept (choose one from list)</td><td>Choose whether you accept or reject the translation from the provided list</td></tr><tr><td>D</td><td>Justification (If Column C is &quot;Reject&quot;)</td><td>If you rejected the translation, please provide an explanation of why it was rejected</td></tr></table>

## Responsibilities and task description (for more details check the general instructions):

## The translator’s task

1. Preparation

• Open the excel file provided to you and ensure it is titled in your dialect.

Ensure it has all four columns with the English source (Col A), Dialect Translation (Col B), Reject/Accept (Col C), and Justification (Col D).

2. Translation

• Read the full sentence to ensure you understand it before translating.

• Translate the math problem from English (Column A) to your native dialect (Column B).

• Use simple, concise, and direct language, maintaining the original meaning and clarity of the question.

• After finishing the translation, make sure to check for the following: Is the translation clear? Is the translation clear?

o Does it sound natural in my dialect?

## The reviewer’s task

1. Review the translation

• Review the translation in Column B.

• Decide whether to accept or reject the translation based whether it meets the following requirement:

o The translation should accurately reflect the meaning of the English text.

o The names of places/proper nouns/currency should be kept as is.

o The translation should sound natural in your dialect, with minimal overlap with MSA.

o Borrowed words should be used naturally. For instance, is the word ايميل used and not بريد الكتروني as applicable per dialect.

o The translation should follow the grammatical rules of your dialect.

## 2. Feedback

• If rejecting, provide a justification in Column D.

As a quality assurance step, we will review accepted translations randomly to check the quality of the review process. A satisfactory translation would be following the checklist and general instructions in the guidelines.

## General instructions:

<sup></sup> Confidence in translation: Only work on a sheet if you are confident in your translation capabilities for each specific question. Do not translate in cases where you doubt your understanding of the question. In such cases, simply move to the next line to avoid unnecessary effort. Please add a note of explanation in the comment column.

 Support tools: It is acceptable to use dictionaries or term bases to find the most suitable term for a word in a sentence. However, automatic translation is strictly forbidden as these translated texts will be part of a scientific study that cannot be influenced by translation technology.

<sup></sup> Make sure that the final translation reads like a question-answer conversation. For example: EN: Salem goes on a hiking trip for 3 days. The length of the trail is 24 kilometers. If he walks the same number of kilometers a day, how many kilometers will he walk every day?

$$
\begin{array} { r l } &  \mathrm { M S A : ~ | \hat { \omega } _ \varepsilon | ^ 2 - | \hat { \omega } _ \varepsilon | ^ 2 - | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | } \\ &  \mathrm  ~ | \hat { \omega } _ \varepsilon | ^ 2 - | \hat { \omega } _ \varepsilon | ^ 2 ~ | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat { \omega } _ \varepsilon | \leq | \hat  \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { P A : ~ c l ~ j = g \int _ { \mathcal { S } _ \epsilon } ^ \epsilon | \tilde { \lambda } | ^ \epsilon | \Delta \epsilon ~ _ { \mathcal { S } _ \epsilon } \mathrm { d } \tilde { \lambda } | ^ { 2 } ~ } \mathrm { ~ a s ~ } \mathrm { ~ c o s i t i a l ~ i \sum _ { \mathcal { S } _ \epsilon \in \mathcal { S } _ \epsilon } | \mathcal { S } _ \epsilon | \Delta \epsilon ~ _ { \mathcal { S } _ \epsilon } | \Delta \epsilon ~ _ { \mathcal { S } _ \epsilon } | \Delta \epsilon ~ _ { \mathcal { S } _ \epsilon } ~ } } \\ & { \mathrm { ~ c l ~ g \ ' _ { \mathcal { S } _ \epsilon } \mathrm { ~ a s ~ } \it { \mathcal { S } _ \epsilon } ^ { \lambda _ { \mathcal { S } _ \epsilon } } \mathrm { ~ c l \tilde { \lambda } _ { \mathcal { S } _ \epsilon } \mathrm { ~ b a s } ^ \lambda _ { \mathcal { S } _ \epsilon } \mathrm { ~ c l \tilde { \lambda } _ { \mathcal { S } _ \epsilon } \mathrm { ~ b a s } ^ \lambda _ { \mathcal { S } _ \epsilon } } } } } \\ & { \mathrm { ~ } } \end{array}
$$

<sup></sup> Do not try to invent terms to make your translation “appear dialectical”. This holds the same for terms that do not have an equivalent term in Arabic (e.g. house flipping), please provide a brief description.

<sup></sup> For borrowed words in your variety, use the borrowed word as is to mimic the real dialect use. For instance, if the word "email" is commonly used in its transliterated form in your dialect (not translated as $" , ( " , \dot { \circ } 5 , \dot { \circ } 5 ) $ بريد use the form " إيميل ".

<sup></sup> While spelling variation is expected, it is advised to maintain consistent spelling for recurring dialectal terms within one dialect, For example, in the Egyptian dialect, the future tense is usually marked with ه at the beginning of the verb, like هكلمك ") I will call you"). Some speakers, however, prefer to use ح instead of ,ه as in . حكلمك Whichever variant is used should be consistent for the dialect. See more details on acceptable variants in the orthographic guidelines part below.

Maintain consistency with the number format in the source dataset. Use digits when digits are provided, and words when numbers are written out (e.g., dozens, fractions, or dual forms), as this reflects the dataset’s structure. For instance, if you have the number one written out in digits as $^ { 6 } 1 \%$ then keep it as is (even if this does not follow the Arabic language rules).

<sup></sup> Any numeral system (Hindu-Arabic $( 1 . 4 . 5 . 4 . . )$ or Arabic $( 1 , 2 , 3 . . . ) )$ is acceptable in this dataset, use whatever is natural for your writing. This holds true for punctuation marks as well. See the table below for acceptable number/ punctuation marks variants.

<sup></sup> Entities such as proper nouns (e.g. names of places, people, organisations, and products) and units of measurement such as currencies, metres, miles, etc. should remain as is, please do not localize them for this task.

## Orthographic Guidelines

Since DA has no official orthography standards, Arabs writing in DA tend to use a set of variant spellings that either capture the utterance phonology primarily or are inspired by Modern Standard Arabic orthographic decisions (etymological or historical spelling) (Habash et al., 2018). These are considered dialectical variants, where they do not impact readability and comprehension, but a variant is used consistently in a dialect. In cases where a variation affects readability, comprehension, or changes the meaning, the variant should be considered a spelling error. For example, the word “ سال “ when written with a hamza ( سأل ( means asked, whereas when written without a hamza (,) سال it means spelled or leaked. Since the hamza alters the meaning, using the wrong form would be treated as a spelling error.

The following is a motivating example from Habash et al., (2018) showing 20+ spellings of the Egyptian Arabic word /mabi’ulha:š/ ‘he does not say it’ and their frequencies from a Google Search (September 29, 2017). Since all of these variants clearly indicate the intended meaning, they are all acceptable variants for the purpose of our task at hand. Note that the variations include different decisions on how to represent the glottal stop whose MSA cognate is $/ 9 / 9$ shortened long vowels, attaching or detaching the negation proclitic mA/m, etc.

<table><tr><td rowspan=1 colspan=1>Arabic Orthography</td><td rowspan=1 colspan=1>Arabic Transliteration</td><td rowspan=1 colspan=1>Frequency</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mbyqwlhAš</td><td rowspan=1 colspan=1>≈26,000</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mA byqwlhAš</td><td rowspan=1 colspan=1>≈13,000</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mAbqlhAš, mbqwlhAš,mbqlhAš, mA bqlhAš,mAbyqwlhAš</td><td rowspan=1 colspan=1>≤ 10,000</td></tr><tr><td rowspan=1 colspan=1>le loen</td><td rowspan=1 colspan=1>mAbqwlhAš, mA bqwlhAš,mbyqlhAš, mA byqlhAš</td><td rowspan=1 colspan=1>≤ 1,000</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mbýlhAš, mAbywlhAš,mA bywlhAš, mAbywlhAš</td><td rowspan=1 colspan=1>≤ 100</td></tr><tr><td rowspan=1 colspan=1> 0</td><td rowspan=1 colspan=1>mA bywlhAš, mAbýlhAš,mbyýwlhAš, mA byýlhAš,mAbýwlhAš, mA býlhAš,mA bwlhAš, mbýwlhAš,mbywlhAš, mAbwlhAš,mbwlhAš</td><td rowspan=1 colspan=1>≤ 10</td></tr></table>

We present below a list of common orthographic phenomena that we consider as dialectal variants. Some of these are the result of letter shape similarity and other phonological similarities. Please note, that not all “similar letters” are acceptable variants: if they are visually similar but phonologically not similar (don’t sound similar), they are most likely errors.

<table><tr><td colspan="1" rowspan="1">Variant Set Name</td><td colspan="1" rowspan="1">Variant Characters</td><td colspan="1" rowspan="1">Dialectal Variants Examples</td></tr><tr><td colspan="1" rowspan="1">Ta Marbuta /Ha</td><td colspan="1" rowspan="1">0/0</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Alif Maqsura/Ya</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">!</td></tr><tr><td colspan="1" rowspan="1">Alif Hamzas</td><td colspan="1" rowspan="1">wyi</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Hamza Forms</td><td colspan="1" rowspan="1">i/e/is/3</td><td colspan="1" rowspan="1">//</td></tr><tr><td colspan="1" rowspan="1">EtymologicalVariants</td><td colspan="1" rowspan="1">S/iS///</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">EtymologicalVariants</td><td colspan="1" rowspan="1">上/</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">EtymologicalVariants</td><td colspan="1" rowspan="1">已/</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">EtymologicalVariants</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">EtymologicalVariants</td><td colspan="1" rowspan="1">3/3/</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Splits</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Merges</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Shortening &amp;Lengthening</td><td colspan="1" rowspan="1"> $/ \leq 6 / 1 6 / s$ </td><td colspan="1" rowspan="1"></td></tr><tr><td></td><td>(long vowel spelling variant of short vowel spelling via elided diacritics)</td><td></td></tr></table>

It is not possible to create a full, exact list of acceptable variants for all possible contexts, as some contexts can lead the reader to a different unintended meaning. For example, while Ta Marbuta and Ta Mamduda may be interchangeable safely in contexts like , الحاره جمعيت / الحارة جمعية they may not be in limited contexts like حركة .الجدعان / حركت الجدعان

Acceptable number/punctuation marks variants

<table><tr><td>1,2,3,4...</td><td></td></tr><tr><td>%,</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>2</td><td></td></tr></table>

## Appendix A:

Screenshot of Excel sheet

<table><tr><td rowspan=1 colspan=1>English source text</td><td rowspan=1 colspan=1>Dialect translation</td><td rowspan=1 colspan=1>Accept/Reject</td><td rowspan=1 colspan=1>Justification (If Column C is &quot;Reject&quot;)</td></tr><tr><td rowspan=1 colspan=1>James decides to run 3 sprints 3 times a week. Heruns 60 meters each sprint. How many total metersdoes he run a week?</td><td rowspan=1 colspan=1>Accep</td><td rowspan=1 colspan=1>t</td><td rowspan=1 colspan=1>日</td></tr></table>

## C Localizations

<table><tr><td></td><td>EN</td><td>MSA</td><td>EGY</td><td>JOR</td><td>KSA</td><td>MOR</td><td>UAE</td></tr><tr><td>Avg. local./MWP</td><td>5.79</td><td>5.30</td><td>5.32</td><td>5.64</td><td>5.55</td><td>5.59</td><td>5.40</td></tr></table>

Table 7: Number of localizations made per line on EN data compared to the other dialects.

<table><tr><td>Scheme</td><td>N</td><td>Char diff</td><td>Word diff</td><td>Char edit</td><td>Word edit</td></tr><tr><td>All</td><td>1194</td><td>10.8%</td><td>18.0%</td><td>19.1</td><td>6.2</td></tr><tr><td>Names</td><td>1190</td><td>6.3%</td><td>7.2%</td><td>10.9</td><td>2.4</td></tr><tr><td>Currency</td><td>522</td><td>5.7%</td><td>11.4%</td><td>10.6</td><td>4.2</td></tr><tr><td>Location</td><td>173</td><td>2.2%</td><td>1.5%</td><td>4.4</td><td>0.5</td></tr><tr><td>Numerals</td><td>1329</td><td>3.4%</td><td>9.3%</td><td>5.9</td><td>3.1</td></tr></table>

Table 8: Average surface difference by localization scheme. Localization edits are small and targeted rather than broad rewrites.

## D Additional Details on the Experimental Setup

<table><tr><td>Model</td><td></td><td>Identifier</td><td>Ar</td><td>En Access</td><td></td><td>Licence</td></tr><tr><td>1</td><td>Atlas-Chat-2B</td><td>MBZUAI-Paris/Atlas-Chat-2B</td><td></td><td></td><td>Local</td><td>Gemma ToU</td></tr><tr><td>2</td><td>Nile-Chat-4B</td><td>MBZUAI-Paris/Nile-Chat-4B</td><td></td><td></td><td>Local</td><td>Gemma ToU</td></tr><tr><td>3</td><td>Atlas-Chat-9B</td><td>MBZUAI-Paris/Atlas-Chat-9B</td><td></td><td></td><td>Local</td><td>Gemma ToU</td></tr><tr><td></td><td>4 c4ai-r7b</td><td>CohereLabs/c4ai-command-r7b-arabic-02-2025</td><td></td><td></td><td>Local</td><td>CC BY-NC 4.0</td></tr><tr><td>5</td><td>Jais2-8B-Chat</td><td>inception42/Jais-2-8B-Chat</td><td></td><td></td><td>Local</td><td>Apache 2.0</td></tr><tr><td></td><td>6 Qwen2.5-7B</td><td>Qwen/Qwen2.5-7B-Instruct</td><td></td><td></td><td>Local</td><td>Apache 2.0</td></tr><tr><td>7</td><td>Gemma4-2B</td><td>google/gemma-4-E2B-it</td><td></td><td></td><td>Local</td><td>Apache 2.0</td></tr><tr><td></td><td>8 Gemma4-4B</td><td>google/gemma-4-E4B-it</td><td></td><td></td><td>Local</td><td>Apache 2.0</td></tr><tr><td>9</td><td>Gemma4-26B</td><td>google/gemma-4-26B-A4B-it</td><td></td><td></td><td>Local</td><td>Apache 2.0</td></tr><tr><td>10</td><td>Qwen-3.6</td><td>qwen/qwen3.6-plus</td><td></td><td></td><td>OpenRouter API terms</td><td></td></tr><tr><td></td><td>11 GPT-5.4</td><td>openai/gpt-5.4</td><td></td><td></td><td>OpenRouter API terms</td><td></td></tr></table>

Table 9: Evaluated models, in the order of Table 3. Identifiers refer to the Hugging Face Hub (local models) or OpenRouter (API models). Ar and En mark the languages each model supports (Arabic, English). Gemma ToU: Google’s Gemma Terms of Use, inherited from the Gemma base models of Atlas-Chat and Nile-Chat. Gemma4-2B and Gemma4-4B are the E2B and E4B variants (2B and 4B effective parameters); Gemma4-26B is a mixture-ofexperts model with 4B active parameters.

Decoding Reasoning (thinking) is disabled for GPT-5.4, Qwen-3.6 and the Gemma 4 models. All local models and Qwen-3.6 use greedy decoding (temperature 0). OpenRouter does not pass a temperature setting to GPT-5.4, so GPT-5.4 uses the provider’s default sampling. Each model is run once per variety and condition; no scores are averaged over repeated runs. Responses that reach the token limit are scored as generated.

Prompting Except for the Gemma 4 models, every model receives the instruction in Section 4 as the system message and the problem as the user message, formatted with the model’s own chat template; when a template has no system role, the instruction is prepended to the user message. These models generate at most 1,024 new tokens.

Gemma 4 The Gemma 4 models use the pipeline of the interpretability analysis (Section 5.2) with greedy decoding: at most 512 new tokens, the system prompt “You are a helpful assistant.”, and an instruction to end with “Final answer: [number]”.

Hardware The Arabic-centric models and Qwen2.5-7B run in Hugging Face Transformers in bf16 without quantization on a single NVIDIA A100 40GB GPU. The Gemma 4 models run on a machine with 4 Intel Xeon Gold 6150 CPU cores (2.70 GHz) and a single NVIDIA Tesla V100- SXM2 32GB GPU. GPT-5.4 and Qwen-3.6 are queried through OpenRouter, at a total cost of approximately \$100.

Run-to-run variation In MSA, the place-only and currency-only questions are identical to the translated ones, so any difference between these three conditions reflects run-to-run variation, not localization. Seven of the eleven models give identical answers in all three. For the other four, accuracy varies by at most 0.8 points (2.8 for Qwen-3.6), and the average over all models varies by only 0.2 points.

## E Interpretability

![](images/435ef3dd65087a4f9f0a7472f69ad9fb065e6b4b1700e60226cc71f62d08a814.jpg)  
Figure 4: Residual-stream path distance by correctness group; dashed line = unrelated-problem average.

## F Results

<table><tr><td></td><td>T</td><td>L</td><td>Name</td><td>Curr.</td><td>Loc.</td><td>Num.</td></tr><tr><td>MSA EGY JOR</td><td>70.0 66.1 69.1</td><td>66.7 62.8 67.3</td><td>67.5 66.9 68.7</td><td>69.9 63.6 68.5</td><td>70.1 66.0 69.1</td><td>69.2 65.5 68.1</td></tr></table>

Table 10: Model evaluation average results across MSA and the dialects: EGY, JOR, KSA, MOR and UAE. We report accuracy for the different localizations schemes over the 250 problems. T denotes Translated data; L denotes Translated and fully Localized data, Name denotes name replacements only, Curr. denotes currency replacements only, Loc. denotes location replacements only and Num. denotes numerical value replacements only.

<table><tr><td></td><td>Name</td><td>Curr.</td><td>Loc.</td><td>Num.</td></tr><tr><td>MSA</td><td>65.7</td><td>71.0</td><td>75.8</td><td>69.2</td></tr><tr><td>EGY</td><td>66.7</td><td>67.2</td><td>78.8</td><td>65.5</td></tr><tr><td>JOR</td><td>67.0</td><td>64.8</td><td>78.8</td><td>68.1</td></tr><tr><td>KSA</td><td>61.9</td><td>60.6</td><td>58.6</td><td>60.6</td></tr><tr><td>MOR</td><td>53.7</td><td>56.3</td><td>55.6</td><td>55.5</td></tr><tr><td>UAE</td><td>68.6</td><td>70.3</td><td>84.8</td><td>69.1</td></tr></table>

Table 11: Model evaluation average results across MSA and the dialects: EGY, JOR, KSA, MOR and UAE. We report accuracy for the different localizations schemes. Each scheme is evaluated only on the questions it applies to (localizable entities): Name, Curr., Loc. and Num..

<table><tr><td>Correctness group</td><td>N</td><td>Mean RMS</td></tr><tr><td>Baseline only correct</td><td>556</td><td>1.50</td></tr><tr><td>Localized only correct</td><td>470</td><td>1.50</td></tr><tr><td>Neither correct</td><td>2722</td><td>1.22</td></tr><tr><td>Both correct</td><td>9476</td><td>0.29</td></tr></table>

Table 12: Mean residual-stream RMS distance by paired correctness group, pooled across the three Gemma4 models, six dialects, and localization schemes. Discordant examples exhibit substantially larger trajectory shifts than examples that are correct in both conditions.

<table><tr><td>Scheme</td><td>N</td><td>Discordant</td><td>Discord. rate</td><td>mean acc. ∆</td><td>Localized only / Baseline only</td></tr><tr><td>All</td><td>3582</td><td>316</td><td>8.8%</td><td>-1.2 pp</td><td>136 / 180</td></tr><tr><td>Names</td><td>3570</td><td>311</td><td>8.7%</td><td>-0.2 pp</td><td>152 /159</td></tr><tr><td>Currency</td><td>1566</td><td>153</td><td>9.8%</td><td>-3.0 pp</td><td>53 / 100</td></tr><tr><td>Location</td><td>519</td><td>29</td><td>5.6%</td><td>+0.2 pp</td><td>15 / 14</td></tr><tr><td>Numerals</td><td>3987</td><td>217</td><td>5.4%</td><td>+0.3 pp</td><td>114 / 103</td></tr><tr><td>Total</td><td>13224</td><td>1026</td><td>7.8%</td><td>-0.6 pp</td><td>470 /556</td></tr></table>

Table 13: Paired discordance by localization scheme, pooled across the three Gemma4 models and six dialects. Only eligible, quality-filtered pairs are included for each scheme, so N varies by scheme. The final column reports cases where only the localized prompt is correct and where only the translated baseline is correct; these totals match the discordant-group sizes in Table 12. Accuracy ∆ is localized minus translated accuracy in percentage points.

<table><tr><td>Model</td><td>Dialect</td><td>Loc./S. RMS</td><td>Samp. flip</td><td>Loc. flip</td></tr><tr><td rowspan="6">Gemma4-2B</td><td>MSA</td><td>1.004</td><td>16.0%</td><td>12.0%</td></tr><tr><td>EGY</td><td>1.025</td><td>12.0%</td><td>13.3%</td></tr><tr><td>JOR</td><td>1.214</td><td>4.0%</td><td>14.7%</td></tr><tr><td>KSA</td><td>1.024</td><td>9.3%</td><td>11.3%</td></tr><tr><td>MOR</td><td>1.111</td><td>13.3%</td><td>18.7%</td></tr><tr><td>UAE</td><td>1.043</td><td>9.3%</td><td>12.0%</td></tr><tr><td rowspan="6">Gemma4-4B</td><td>MSA</td><td>0.965</td><td>4.0%</td><td>6.7%</td></tr><tr><td>EGY</td><td>1.140</td><td>4.0%</td><td>8.0%</td></tr><tr><td>JOR</td><td>1.091</td><td>4.0%</td><td>6.0%</td></tr><tr><td>KSA</td><td>1.083</td><td>16.0%</td><td>12.0%</td></tr><tr><td>MOR</td><td>1.158</td><td>8.0%</td><td>8.0%</td></tr><tr><td>UAE</td><td>1.153</td><td>5.3%</td><td>4.7%</td></tr><tr><td rowspan="6">Gemma4-26B</td><td>MSA</td><td>1.197</td><td>2.7%</td><td>3.3%</td></tr><tr><td>EGY</td><td>1.259</td><td>5.3%</td><td>6.0%</td></tr><tr><td>JOR</td><td>1.073</td><td>6.7%</td><td>4.0%</td></tr><tr><td>KSA</td><td>1.221</td><td>9.3%</td><td>11.3%</td></tr><tr><td>MOR</td><td>1.182</td><td>4.0%</td><td>4.7%</td></tr><tr><td>UAE</td><td>1.449</td><td>1.3%</td><td>5.3%</td></tr></table>

Table 14: Sampling robustness under full localization. Each model–dialect cell uses up to 50 matched problems and three generation seeds. Loc./S. RMS is the ratio of matched translated-to-localized RMS distance to within-translated seed-to-seed RMS distance, computed from unrounded mean values. Values above 1 indicate that localization changes residual trajectories more than sampling variation in the translated condition: this holds in 17 of 18 cells (smallest margin 1.004 for Gemma4-2B/MSA); the only exception is Gemma4-4B/MSA (0.965, italicized). Bold marks the largest ratio within each model. Flip rates report correctness changes across the corresponding paired comparisons; localization flip rates exceed sampling flip rates in 13 of 18 cells, equal them in 1, and fall below in 4.

![](images/051ac82b0f4044788ccb5216c0970e8c8d41b1cadde29cd2535c77d7728f486f.jpg)

![](images/cd6638dd670f83ed0143e64ce2826bd7ed7bd222243e626bb88083a3b6b52eaa.jpg)

![](images/372e1735e69e27818a667fe5551663ee4bdd461705f642fed99abed5fe3f3132.jpg)  
Figure 5: Residual-stream RMS distance by paired correctness group. Discordant examples show substantially larger trajectory separation than examples that remain correct in both conditions.

## G Activation Patching Procedure

We use activation patching to test whether correctness changes on discordant examples are causally linked to the localized entity tokens. A discordant example is one where the model answers correctly in one condition but incorrectly in the other, e.g., the translated non-localized prompt is correct while the localized prompt is wrong, or vice versa. For each such pair, we run patching in both directions: from the non- localized prompt into the localized prompt, and from the localized prompt into the non-localized prompt. If the localized entity representation is causally involved in the answer flip, then replacing the target prompt’s entity-span residual stream with the source prompt’s corresponding entity-span residual stream should move the patched generation toward the source condition, both in answer correctness and in residual-stream geometry.

Let $x ^ { s r c }$ denote the source prompt and $x ^ { t g t }$ denote the target prompt. The source is the condition whose internal representation we patch from, and the target is the condition whose forward pass is modified. For each transformer layer $\ell ,$ we run the target prompt while replacing selected residualstream vectors at selected token positions with the corresponding source residual-stream vectors:

$$
h _ { \ell , p _ { i } } ^ { t g t }  h _ { \ell , q _ { i } } ^ { s r c } ,
$$

where $p _ { i }$ is a target token position and $q _ { i }$ is the matched source token position. We then decode from the patched target run and evaluate whether the patched answer becomes correct, whether it matches the source answer, and whether the patched residual trajectory moves closer to the source trajectory.

Span identification. For each localized/nonlocalized prompt pair, we first identify the changed text spans between the two problem statements. Rather than assuming a fixed name list, we compare the two strings and extract the non-matching contiguous spans in order of appearance. These spans are then paired by order: the first changed span in the localized prompt is paired with the first changed span in the non-localized prompt, the second with the second, and so on. This handles examples with multiple changed names, locations, or currencies without needing to decide manually which entity should correspond to which one. For source patching, we select the relevant localized entity span and its aligned non- localized span.

Tokenization and multi-token spans. After identifying character-level spans, we map each span to token positions using the model tokenizer and the final chat-formatted prompt. A single entity may correspond to one token or multiple tokens. For example, a name may be split into several subword tokens, and a location may occupy multiple Arabic or Latin-script tokens. We therefore patch spans as token sequences rather than as single tokens. If the source span tokenizes to positions

$$
( q _ { 1 } , q _ { 2 } , \ldots , q _ { m } )
$$

and the target span tokenizes to positions

$$
( p _ { 1 } , p _ { 2 } , \ldots , p _ { n } ) ,
$$

we patch the aligned positions in order. When $m =$ $n ,$ this is a direct one-to-one token replacement. When the token lengths differ, we patch the overlapping prefix up to min $( m , n )$ tokens. Thus, the operation remains localized to the aligned entity span while avoiding invalid assumptions about tokenizer granularity. The recorded source\_token\_count, target\_token\_count, and patch\_token\_count allow us to audit these cases.

Layer sweep. We repeat the patch for each residual-stream layer from early to late layers. This gives a layer-wise causal profile: if a name or location representation matters only after it has been integrated into the model’s reasoning state, patching should have stronger effects at those layers. For each layer, we generate an answer from the patched target run and compute both behavioral and geometric metrics.

Behavioral metric. The primary behavioral metric is patched correctness:

$$
\mathrm { C o r r e c t } _ { \ell } ^ { p a t c h } = \nl { \nl } ^ { k } \{ \mathrm { a n s w e r } ( x _ { p a t c h e d , \ell } ^ { t g t } ) = y \} ,
$$

where $y$ is the gold answer. We also compare the patched answer to the clean source and clean target answers to see whether the patch moves the model toward the source condition.

Geometric metric. We also measure whether the patched residual trajectory moves closer to the source trajectory. Let $R ^ { s r c }$ be the residual-stream trajectory of the clean source run, $R ^ { t g t }$ the clean target trajectory, and $R _ { \ell } ^ { p a t c h }$ the patched target trajectory after patching at layer ℓ. We compute RMS distances between trajectories:

$$
d ( R ^ { a } , R ^ { b } ) = \sqrt { \frac { 1 } { | \mathcal { L } | } \sum _ { k \in \mathcal { L } } \left. R _ { k } ^ { a } - R _ { k } ^ { b } \right. _ { 2 } ^ { 2 } } .
$$

The clean target-to-source distance $d ( R ^ { t g t } , R ^ { s r c } )$ is the unpatched baseline. The patched-to-source distance $d ( R _ { \ell } ^ { p a t c h } , R ^ { s r c } )$ tells us whether the patch moved the target trajectory toward the source. We report a source-distance ratio:

$$
\mathrm { S o u r c e R a t i o } _ { \ell } = \frac { d ( R _ { \ell } ^ { p a t c h } , R ^ { s r c } ) } { d ( R ^ { t g t } , R ^ { s r c } ) } ,
$$

and a source-closeness gain:

$$
\mathrm { G a i n } _ { \ell } = 1 - \mathrm { S o u r c e R a t i o } _ { \ell } .
$$

A positive gain means the patched run is closer to the source trajectory than the clean target was; a gain near zero means the patch did not move the trajectory; a negative gain means the patch moved it farther away.

Patching methods and controls. We compare the entity-span source patch against several controls:

• source\_patch: patch the matched entity span from the source condition into the target condition. This is the main causal intervention. If entity localization drives the correctness flip, this patch should move the target answer and residual trajectory toward the source condition.

• no\_patch: run the target condition without modifying activations. This gives the unpatched target behavior and residual trajectory.

• self\_patch: patch the target span using activations from the same target condition. This controls for the mechanics of patching itself; because the patched vectors come from the same condition, it should not systematically change behavior.

• shuffled\_source: patch from a source example that is not the matched problem. This tests whether any source-condition activation is sufficient, or whether the effect depends on the correct problem-matched entity representation.

• control\_span: patch an unrelated samelength span rather than the localized entity span. This tests whether the effect is specific to the entity tokens rather than caused by overwriting arbitrary positions in the prompt.

Interpretation. The causal prediction is that source\_patch should outperform the controls on discordant examples: it should increase the probability of recovering the source-side correct answer and produce a positive source-closeness gain. Strong evidence requires that this effect is larger than self\_patch, control\_span, and shuffled\_source; otherwise the result could be explained by patching artifacts, arbitrary position overwrites, or generic source-condition activations rather than the localized entity representation itself.

## H Error Analysis Examples

<table><tr><td>Dialect</td><td>Subset</td><td>Question</td><td>Model Reasoning</td><td>Reference Generated Answer</td><td>Error Type Answer</td></tr><tr><td></td><td>Translated</td><td> $\sin \alpha \sin \alpha \cos \alpha \cos \alpha \cos \alpha \cos \alpha \cos \alpha$   $1 7 5 \approx 0 . 5 5 . 0 1 0 0 ( 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0$   $c = 3 5 \div 5 5 = 3 5 1 = 3 4 1 = 2 5 3 . 5 2 \cdots 1 1 5$   $v ^ { 2 } = v _ { 1 } ^ { 2 } t$   $9 0 0 0 0 0 0 0 0 0 0 0 0 0$ </td><td> $i 0 0 3 5 \div 5 9 1 = 3 3 1 1 2 0 5 ( 1 7 5 0 - 1 1 7 5 1 ) 2 0 0 \div 5 1 5 1$   $1 4 0 = 3 5 - 1 7 5 = 3 5 1 1 2 \sin 6 ( 2 1 2 ) 1 2 1 2$   $= 2 1 2 0 0 0 ( 0 . 5 1 1 ) 2 1 0 0 0 0 ( 5 1 ) 2 1 2 0 0 0 ( 5 1 ) 5 1 5$   ${ } ^ { 2 8 0 } = 1 4 0 \ast { } ^ { 2 }$   $= 2 8 0 + 1 4 0 + 1 7 5 = 5 3 . 0 5 2 4 . 0 1 2 + 1 5 5 . 0 5 5$ </td><td>595</td><td>695 Final Step</td></tr><tr><td></td><td></td><td>A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamonds, and twice the number of emeralds than the were there in the chest?</td><td>If the number of diamonds is 175, and the number of rubies is 35 less than the number of diamonds, then the number of rubie  $\mathrm { s } = 1 7 5 - 3 5$  = 140. If the number of emeralds is double the number rubies. How many of the gems of rubies, then the number of emeralds = 2 * 140 = 280.</td><td></td><td></td></tr><tr><td></td><td>Localized</td><td> $u \times 2 . 1 \approx 1 2 . 1$  lio 5  $\therefore \sin x = \sin x .$   $1 0 1 , 1 5 , 5 5 , 5 5 , 1 3 , 6 5 , 1 5 , 7 1$ </td><td> $( - 2 ) , 0 5 6 < 0 . 5 6 < 0 . 5 6 < 0 . 5 6 < 0 . 5 6 < 0 . 5 6 < 0 . 5 6 < 0 . 1 < 0 . 5 6 < 0 . 1 < 0 . 1 6 < 0 . 1 6 < 0 . 1 5 6 < 0 . 1 6$   $t = \sqrt { 2 } , \sqrt [ 4 ] { 0 } , \sqrt [ 1 5 ] { 1 } , \approx \sqrt [ 1 5 ] { 1 + \sqrt [ 1 5 ] { 1 + \sqrt [ 1 5 ] { 1 + \sqrt [ 1 5 ] { 1 + \sqrt [ 1 5 ] { 1 } } } } }$ </td><td>104</td><td>99 Final Step</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>66</td><td></td></tr><tr><td></td><td></td><td> $p _ { 1 } L _ { 1 } = 1 6 . 6 , 1 5 \leq 1 1 1 = 4 . 1 5 ( d m ^ { 3 } )$ </td><td> $= 3 3 ^ { \ast } 2 = 1 , \dot { 5 } 1 \ldots \sin 1 1 \ldots \cos 1 0 0 0 \ldots \sin 1 \ldots 5 .$   $1 1 ( 0 ) 1 ( 1 1 0 0 ) = 1 0 0 ( 1 2 0 0 ) = 1 1 ( 0 1 1 0 0 ) = 1 1 0 0 ( 1 1 0 0 )$ </td><td></td><td></td></tr><tr><td></td><td></td><td> $3 1 0 1 \ldots 5 1 5$  </td><td></td><td> $\tan ^ { \cdot } \tan ^ { \angle 1 } \alpha \cot \beta \cot$   $b _ { 4 } = 3 3 : 1 0 1 , 5 3 .$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $\therefore \Delta 1 ( - 1 ) = \Delta 2 ( - 1 ) \Delta 1 ( - 1 ) = \Delta 4$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $i \omega s 0 1 , \ldots = 5 - i , i \leq 1 1 , i \omega$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $6 6 + 3 3 = 5$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>=  $v L = 9 9 s = 6 0 \sqrt { 3 } v \times 1 1 \approx ( \dot { 0 } \dot { 3 } )$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>The answer is: 99</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>she comes across a pair of</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>boots that fit her shoe budget. shoes &quot;x&quot;</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>However, she has to choose</td><td>Shireen is shoe shopping when To solve this problem, we first need to know</td><td></td><td></td></tr><tr><td></td><td></td><td>between the boots and two</td><td>the price of the shoes. Let&#x27;s call the price of the</td><td></td><td></td></tr><tr><td></td><td></td><td>pairs of high heels that together cost five dollars less</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>We know that the price of the two heels is:</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>- One heel: 33 dinars</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>- The second heel: double the price of the first</td><td></td><td></td></tr><tr><td></td><td></td><td>than the boots. If one pair of</td><td>heel = 2 * 33 = 66 dinars</td><td></td><td></td></tr><tr><td></td><td></td><td>heels costs 33 dinars and the</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>We also know that the two heels are 5 dinars</td><td></td><td></td></tr><tr><td></td><td></td><td>other costs twice as much,</td><td>cheaper than the shoes, so we can write the</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>boots?</td><td>how many dinars are the</td><td>following equation:</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Price of the shoes - 5 = Price of the two heels</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>X = 33 + 66</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>X = 99</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Therefore, the price of the shoes is 99 dinars.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>The answer is: 99</td><td></td><td></td><td></td></tr><tr><td></td></table>

Figure 6: Examples of errors in Jordanian Arabic.

![](images/93bad173e68099c9a576de9604e3085417f4690a0a1d466633f6834664e7f036.jpg)  
Figure 7: Examples of errors in Emirati Arabic.