# Cross-Dataset Stability of Expert-Informed Skill Prompting and Fine-Tuning for Chinese Metaphor Identification

Yufeng Wu and Meichun Liu

City University of Hong Kong

Abstract—Metaphor-identification performance can change markedly across datasets that differ in text distribution and annotation policy. We examine whether a fixed expert-informed procedure produces a more even cross-dataset profile than taskspecific parameter adaptation. Four prespecified conditions are compared for Chinese sentence-level metaphor identification: BERT fine-tuning (BERT-FT), QLoRA-based large language model fine-tuning (LLM-FT), direct zero-shot LLM prompting (LLM-ZS), and zero-shot prompting with a frozen procedural Skill (Skill-ZS). The Skill operationalizes established criteria involving contextual meaning, basic meaning, contrast, and comparison. Evaluation covers CMRE Test and two external datasets, CCIME and CMC. Fine-tuned scores are means over three seeds, whereas each zero-shot score comes from one deterministic configuration. Fine-tuning remains strongest on the native test set: BERT-FT reaches 91.76 Macro-F1. LLM-FT has the highest external mean (83.52), while Skill-ZS is close at 82.92 and has both the highest external floor (82.64) and the smallest observed range across all three datasets (4.08 points). In the matched zero-shot comparison, adding the Skill reduces metaphorical predictions on every dataset. This sharply lowers false positives on CCIME but increases false negatives on CMRE Test and CMC. The results position expert-informed Skill prompting as a complementary route to more even observed cross-dataset performance, while fine-tuning retains its advantage in nativedata accuracy. To our knowledge, this is the first study to compare an expert-informed procedural Skill with task-specific fine-tuning in the same cross-dataset evaluation of Chinese sentence-level metaphor identification.

Index Terms—Chinese metaphor identification, cross-dataset evaluation, expert-informed Skill, fine-tuning, large language models

## I. INTRODUCTION

Metaphor-identification systems often change behavior when the evaluation dataset changes. Differences in genre, construction mix, class prevalence, and annotation policy can alter both overall performance and the balance between false positives and false negatives [1], [2], [3]. A high sourcealigned score therefore captures only one part of the evaluation problem; the external performance floor and the variation across datasets are also consequential.

The task itself depends on deciding whether an expression has a contextual meaning that contrasts with, yet can be understood in relation to, a more basic meaning. Manual procedures such as MIP and MIPVU organize this judgment into explicit steps [4], [5]. Computational systems have represented related knowledge through model architecture, contextual interaction, explicit basic-meaning modules, prompts, and rule scripts [6], [7], [8]. These alternatives place linguistic expertise in different parts of a system and may consequently produce different cross-dataset profiles.

Fine-tuning places task adaptation in parameters learned from labeled examples. An instruction Skill instead supplies an operational procedure at inference time while leaving the model parameters unchanged. We use Skill to denote a reusable instruction artifact that specifies a multistep inference procedure rather than an in-context demonstration set or learned adapter. Its criteria may transfer across datasets differently from regularities learned from one training distribution.

We evaluate these routes under a common Chinese sentencelevel formulation. BERT-FT and LLM-FT represent supervised parameter adaptation. Skill-ZS supplies the base LLM with a frozen expert-informed procedure derived from published annotation protocols and metaphor-identification research. Direct LLM-ZS uses the same base model and runtime without the Skill, providing a matched comparison of the procedure’s aggregate effect.

We ask three questions. RQ1 asks how the four conditions differ in observed cross-dataset stability, summarized by native Macro-F1, external mean, external floor, the signed native– external mean gap, and three-dataset range. RQ2 examines the dataset- and class-specific performance trade-offs behind these summaries using accuracy, metaphor-class precision, recall, F1, and Macro-F1. RQ3 asks how adding the Skill redistributes aggregate decisions relative to the matched LLM-ZS condition. To our knowledge, this is the first study to compare an expert-informed procedural Skill with task-specific fine-tuning in the same cross-dataset evaluation of Chinese sentence-level metaphor identification [9], [10], [11]. It further contributes a criterion-specific account of their complementary performance profiles and an aggregate analysis of the decision shift introduced by the Skill.

## II. RELATED WORK

## A. Expert Knowledge in Metaphor Identification

MIP and MIPVU make metaphor decisions more reproducible by requiring analysts to establish contextual meaning, identify a more basic meaning, and assess contrast and comparison [4], [5]. Computational models have incorporated these principles in different forms. Sequential metaphor identification uses theory-inspired contextual signals, MelBERT models contextualized interaction, and explicit basic-meaning modeling represents the comparison directly [6], [7], [8]. ContrastWSD further combines sense information with an MIP-based contrast test [12].

Generative models allow operational knowledge to be expressed in prompts rather than encoded in model architecture. Recent work evaluates zero-shot metaphor labels and rationales, MIPVU-inspired LLM procedures, and explicit rule scripts for Chinese metaphor identification [13], [9], [10]. These studies motivate an expert-informed Skill. The present Skill adapts selected operational principles to sentence-level binary classification rather than implementing the full lexicalunit MIPVU procedure.

## B. Fine-Tuning, Prompting, and Dataset Change

Metaphor-identification systems do not transfer uniformly across datasets. Cross-lingual research established that learned representations can support transfer across language-specific datasets [14]. Later studies directly evaluated cross-dataset and cross-language generalization, including under multiple outof-distribution settings, as well as cross-genre performance losses [1], [2], [3]. Broader NLP research likewise distinguishes source-aligned evaluation from performance under a changed target distribution [15].

LLM-based studies add a second comparison: parameter adaptation versus direct or procedure-guided prompting. Recent studies examine surface-sensitive LLM metaphor judgments and multilingual evaluation of metaphor detection [16], [17]. Fuoli et al. compare retrieval-augmented generation (RAG), prompt engineering, and fine-tuning for fulltext metaphor identification [11]. The present study focuses on Chinese sentence classification and asks whether explicit procedural guidance produces a more even observed crossdataset profile. It combines a four-regime comparison with a matched direct-versus-Skill zero-shot intervention, allowing performance level and cross-dataset variation to be considered together.

## III. METHODS

We compare four ways of supplying task knowledge while holding the sentence-level label space and evaluation rows constant. CMRE Dev was used for supervised model selection, and both zero-shot conditions were finalized before formal evaluation. Implementation-level configurations and reproducibility records are provided in the supplementary package.

## A. Task, Data, and Overlap Control

Each sentence receives either the metaphorical or nonmetaphorical label. CMRE supplies the training, development, and native test partitions [18]. We used its released sentence labels and retained simile relations labeled as positive. CCIME contributes the 1,200 publicly labeled Task 1 development rows [19], and CMC contributes its released held-out Chinese split [20]. Only their binary metaphor labels were used. Table I reports the retained data.

TABLE I  
DATASET COMPOSITION AFTER CMRE TRAIN/DEV OVERLAP REMOVAL.
<table><tr><td>Dataset</td><td>Total</td><td>Metaphorical</td><td>Non-metaphorical</td></tr><tr><td>CMRE Train</td><td>6,794</td><td>3,397</td><td>3,397</td></tr><tr><td>CMRE Dev</td><td>850</td><td>425</td><td>425</td></tr><tr><td>CMRE Test</td><td>850</td><td>425</td><td>425</td></tr><tr><td>CCIME</td><td>1,162</td><td>567</td><td>595</td></tr><tr><td>CMC</td><td>698</td><td>274</td><td>424</td></tr></table>

TABLE II  
KNOWLEDGE-SUPPLY REGIMES AND FORMAL RUN BASIS.
<table><tr><td>Condition</td><td>Labels</td><td>Updated</td><td>Skill</td><td>Runs</td></tr><tr><td>BERT-FT</td><td>CMRE</td><td>full model</td><td>no</td><td>3</td></tr><tr><td>LLM-FT</td><td>CMRE</td><td>QLoRA</td><td>no</td><td>3</td></tr><tr><td>LLM-ZS</td><td>none</td><td>none</td><td>no</td><td>1</td></tr><tr><td>Skill-ZS</td><td>none</td><td>none</td><td>yes</td><td>1</td></tr></table>

Before evaluation, exact and normalized-exact matches with CMRE Train or Dev were removed, excluding 38 CCIME rows and two CMC rows. Released duplicate rows within an evaluation view retained distinct identifiers so that the original row-level units were preserved. The resulting rows were shared by all four conditions. The annotation policies nonetheless differ: CMRE retains released-positive similes as metaphorical, whereas CCIME places similes and hyperbole in its non-metaphorical class. We therefore interpret CCIME as combining dataset shift with annotation-policy shift. Dataset versions, normalization rules, duplicate records, and overlap reports are documented in the supplement.

## B. Four Knowledge-Supply Regimes

Table II summarizes the four knowledge-supply regimes. BERT-FT and LLM-FT learn from CMRE labels; LLM-ZS and Skill-ZS receive no labeled examples. The four-way comparison describes practical systems with different architectures and adaptation costs. The zero-shot pair forms the controlled comparison: both use the same base model, runtime, prompt shell, output format, and evaluation rows, with the frozen Skill as the only condition-level addition.

## C. Fine-Tuned Conditions

BERT-FT used hfl/chinese-robertawwm-ext-large with a two-label sequence-classification head [21]. The full model was optimized with AdamW at a learning rate of $2 \times 1 0 ^ { - 5 }$ , effective batch size 16, and maximum sequence length 512 for up to five epochs. Checkpoint selection maximized CMRE Dev Macro-F1, with lower Dev loss and then earlier epoch as tie-breakers. A fixed 0.5 threshold produced the labels.

LLM-FT adapted the text-only Qwen/Qwen3.6-27B model [22] with 4-bit NF4 QLoRA [23], [24]. Rank 8 and α = 16 yielded 23.76M trainable parameters. Assistant-only targets contained one JSON label, with loss restricted to target tokens. AdamW used a learning rate of $1 0 ^ { - 4 }$ , effective batch size 16, and maximum sequence length 256 for up to three epochs. Checkpoints were selected by the same CMRE Dev criterion as BERT-FT.

Both fine-tuned conditions used seeds 42, 43, and 44 and selected one checkpoint independently per seed before test prediction. The supplement reports the complete optimization, target-module, model-revision, and runtime configurations.

## D. Zero-Shot Control and Expert-Informed Skill

LLM-ZS used the same Qwen base without parameter updates. Skill-ZS added the frozen Skill to this matched zeroshot configuration. We use zero-shot to mean that no labeled CMRE, CCIME, or CMC example was used for parameter updates or supplied as an in-context demonstration. The direct prompt was dataset-agnostic and example-free.

The Skill translates established identification principles into six operations: establish sentence context, identify a candidate mapping, compare contextual and basic meanings, test contrast and comparison, reject unsupported cases, and aggregate the analysis to one sentence label. Its wording was refined through seven iterations on 60 AI-assisted synthetic specification sentences, balanced between the two labels. These items were used to revise the instruction rather than to estimate performance. They were excluded from formal evaluation, and no formal evaluation sentence, score, or error informed Skill development.

Both zero-shot conditions used deterministic decoding and the same constrained JSON label format. All 2,710 outputs per condition satisfied the output schema directly. The exact prompts, Skill text, decoding configuration, parser specification, and output records are included in the supplement. Because the matched pair differs only by the Skill, their aggregate output difference isolates the condition-level change associated with adding the full procedure; instruction-level effects require separate ablations.

## E. Evaluation Measures

We report accuracy and metaphor-class precision, recall, and F1. Macro-F1, the unweighted mean of both class F1 scores, is the primary cross-dataset measure because class prevalence differs. The stability analysis considers four complementary summaries: the mean across CCIME and CMC, the lower of those two external scores, the signed difference between CMRE Test and the external mean, and the range across all three datasets. A small range is interpreted together with the performance floor, because uniformly low scores would also produce little variation. Fine-tuned values are means and sample SDs over three seeds; zero-shot values are deterministic point estimates.

## IV. RESULTS

The analysis begins with the cross-dataset profile that motivates the comparison, then examines the class-specific tradeoffs and matched decision shift behind it. Fig. 1 visualizes the

TABLE III  
OBSERVED CROSS-DATASET STABILITY CRITERIA IN MACRO-F1 POINTS, COMPUTED FROM UNROUNDED VALUES. NATIVE − EXT. MEAN IS SIGNED.
<table><tr><td>Method</td><td>Ext. mean Ext. floor</td><td></td><td>Native – ext. mean</td><td>Range</td></tr><tr><td>BERT-FT</td><td>80.69</td><td>76.70</td><td>11.07</td><td>15.06</td></tr><tr><td>LLM-FT</td><td>83.52</td><td>76.78</td><td>7.65</td><td>14.39</td></tr><tr><td>LLM-ZS</td><td>78.75</td><td>66.74</td><td>3.12</td><td>24.02</td></tr><tr><td>Skill-ZS</td><td>82.92</td><td>82.64</td><td>-3.80</td><td>4.08</td></tr></table>

Macro-F1 and prediction profiles. Dataset scores are rounded to two decimals in Table $\mathrm { I V } ;$ the stability criteria in Table III use the corresponding unrounded values.

## A. Observed Cross-Dataset Stability

The main pattern is a separation between peak performance and cross-dataset evenness. BERT-FT has the highest native Macro-F1, and LLM-FT has the highest external mean at 83.52. Skill-ZS is only 0.60 points lower on the external mean, while attaining the highest external floor (82.64) and the smallest observed three-dataset range (4.08). Its floor exceeds those of LLM-FT and BERT-FT by 5.86 and 5.94 points, respectively. RQ1 therefore identifies Skill-ZS as having the most even observed profile among the four conditions, rather than the highest score on every criterion.

The signed native–external gap further distinguishes the profiles. Both fine-tuned models score higher on CMRE Test than on their external average, while Skill-ZS scores 3.80 points higher on average externally than natively. This direction is descriptive rather than intrinsically desirable; together, the external floor and range show that the Skill’s observed stability is coupled with a lower native peak.

The different run designs limit the precision with which these point differences can be interpreted. BERT-FT and LLM-FT provide sample SDs across three seeds; the largest displayed dispersion is LLM-FT on CCIME at 2.77 points. The LLM-ZS and Skill-ZS results are deterministic point estimates under their frozen configurations and provide no comparable estimate of between-run dispersion. We therefore report rankings and differences descriptively, without significance claims for cross-regime contrasts. The figure reflects this distinction by drawing whiskers only for fine-tuned conditions.

## B. Dataset and Class-Specific Trade-offs

On CMRE Test, BERT-FT and LLM-FT have the two highest Macro-F1 values but different class profiles: BERT-FT has higher metaphor precision (92.70 versus 88.95), whereas LLM-FT has higher recall (94.04 versus 90.67). Skill-ZS raises precision relative to LLM-ZS (88.58 versus 83.62) while lowering recall from 79.29 to 67.53.

On CCIME, Skill-ZS raises precision relative to LLM-ZS from 61.80 to 77.56 while maintaining similar recall (90.83 versus 91.01). Accuracy, metaphor F1, and Macro-F1 increase by 14.54, 10.06, and 15.90 points. LLM-FT has the highest recall but lower precision than BERT-FT, leaving their Macro-F1 values nearly equal.

![](images/6e6f4a1df619d265abb1b55c6ad6d8d2898e7eea0b2802947708eae804ff453f.jpg)

B Native–external trade-off  
![](images/9dffde0081c0923929ab04560ad4a4cb870d527bb47a03aa540424d1e1461279.jpg)

![](images/406d30956c1392d5bc55fb014bd5823c9125d3c12b6aea9b68305931df2dca4e.jpg)  
Fig. 1. Complementary profiles. (A) Macro-F1 by dataset. (B) CMRE Test Macro-F1 versus the observed external floor; the diagonal marks equality. (C) Rates of metaphorical predictions for the matched zero-shot pair and gold prevalence; arrows run from LLM-ZS to Skill-ZS. Whiskers show sample SD only for the three-seed fine-tuned conditions.

TABLE IV  
CLASSIFICATION PERFORMANCE (%). P, R, AND F1 REFER TO THE METAPHORICAL CLASS. FINE-TUNED VALUES ARE MEANS ± SAMPLE SDS OVER THREE SEEDS; ZERO-SHOT VALUES ARE POINT ESTIMATES. BOLD MARKS THE HIGHEST OBSERVED VALUE IN EACH DATASET–METRIC COLUMN.
<table><tr><td>Dataset</td><td>Method</td><td>Accuracy</td><td>P</td><td>R</td><td>F1</td><td>Macro-F1</td></tr><tr><td rowspan="4">CMRE Test</td><td>BERT-FT</td><td>91.76±0.51</td><td>92.70±0.16</td><td> $9 0 . 6 7 { \pm } 1 . 3 0 \ $ </td><td>91.67±0.59</td><td>91.76±0.51</td></tr><tr><td>LLM-FT</td><td>91.18±0.51</td><td>88.95±0.62</td><td>94.04±0.59</td><td>91.42±0.49</td><td> $9 1 . 1 7 { \pm } 0 . 5 1 $ </td></tr><tr><td>LLM-ZS</td><td>81.88</td><td>83.62</td><td>79.29</td><td>81.40</td><td>81.87</td></tr><tr><td>Skill-ZS</td><td>79.41</td><td>88.58</td><td>67.53</td><td>76.64</td><td>79.12</td></tr><tr><td rowspan="4">CCIME</td><td>BERT-FT</td><td>76.97±0.28</td><td>70.85±0.67</td><td> $8 9 . 7 1 { \pm } 1 . 1 7 $ </td><td>79.17±0.12</td><td>76.70±0.34</td></tr><tr><td>LLM-FT</td><td>77.48±2.46</td><td>69.35±2.57</td><td>96.77±1.37</td><td>80.77±1.67</td><td>76.78±2.77</td></tr><tr><td>LLM-ZS</td><td>68.16</td><td>61.80</td><td>91.01</td><td>73.61</td><td>66.74</td></tr><tr><td>Skill-ZS</td><td>82.70</td><td>77.56</td><td>90.83</td><td>83.67</td><td>82.64</td></tr><tr><td rowspan="4">CMC</td><td>BERT-FT</td><td>85.53±1.74</td><td>83.44±3.06</td><td>78.83±1.32</td><td>81.06±2.10</td><td>84.68±1.79</td></tr><tr><td>LLM-FT</td><td>90.74±1.30</td><td>88.86±1.24</td><td>87.35±3.04</td><td>88.08±1.83</td><td>90.25±1.41</td></tr><tr><td>LLM-ZS</td><td>91.12</td><td>87.06</td><td>90.88</td><td>88.93</td><td>90.76</td></tr><tr><td>Skill-ZS</td><td>84.67</td><td>88.48</td><td>70.07</td><td>78.21</td><td>83.19</td></tr></table>

On CMC, LLM-ZS has the highest accuracy, recall, metaphor F1, and Macro-F1, while LLM-FT has the highest precision. Skill-ZS raises precision slightly relative to LLM-ZS (88.48 versus 87.06) but lowers recall from 90.88 to 70.07. RQ2 therefore shows that the Skill’s more even Macro-F1 profile emerges from dataset-specific precision–recall tradeoffs rather than uniform gains in every measure.

Both fine-tuned models have their lowest Macro-F1 on CCIME. Skill-ZS instead scores 3.52 points higher on CCIME and 4.08 points higher on CMC than on CMRE Test. CCIME’s treatment of similes and hyperbole as non-metaphorical offers one plausible explanation for this contrast, alongside other differences in dataset composition.

## C. Aggregate Decision Redistribution

Table V reports aggregate confusion counts and classprediction rates for the matched zero-shot conditions. Skill-ZS predicts the metaphorical class less often on every dataset. On CCIME, its rate is 57.14%, compared with 71.86% for LLM-ZS; false positives fall from 319 to 149, while false negatives change from 51 to 52. On CMRE Test and CMC, false positives also fall, but false negatives rise by 50 and 57, respectively. RQ3 therefore identifies a consistent reduction in metaphorical predictions with dataset-dependent effects on error types.

Gold prevalence clarifies the contrast. On CCIME, LLM-ZS predicts 23.06 percentage points more metaphorical cases than the gold distribution, whereas Skill-ZS reduces that gap to 8.35 points. On CMRE Test and CMC, direct LLM-ZS is closer to gold prevalence than Skill-ZS. The same change in prediction tendency therefore improves alignment with the gold class prevalence on one dataset and worsens it on two others.

## V. DISCUSSION

The results reveal a trade-off between source-aligned peak performance and evenness across the observed datasets. We first interpret the Skill’s cross-dataset profile, then relate that profile to its matched decision shift and to the choice between knowledge-supply regimes.

## A. A More Even Observed Cross-Dataset Profile

The central finding concerns the shape of performance across datasets. Fine-tuning provides the strongest sourcealigned results, and LLM-FT achieves the highest external average. Skill-ZS, by contrast, combines a competitive external mean with the highest observed external floor and the narrowest three-dataset range. Read together, the floor and range indicate that the Skill yields a more even performance level across the observed datasets, even though it does not maximize native performance.

TABLE V  
AGGREGATE CONFUSION COUNTS AND RATES OF METAPHORICAL PREDICTIONS FOR MATCHED ZERO-SHOT CONDITIONS.
<table><tr><td>Dataset</td><td>Method</td><td>TN</td><td>FP</td><td>FN</td><td>TP</td><td>Predicted metaphorical</td><td>Gold metaphorical</td></tr><tr><td rowspan="2">CMRE Test</td><td>LLM-ZS</td><td>359</td><td>66</td><td>88</td><td>337</td><td>47.41%</td><td>50.00%</td></tr><tr><td>Skill-ZS</td><td>388</td><td>37</td><td>138</td><td>287</td><td>38.12%</td><td>50.00%</td></tr><tr><td rowspan="2">CCIME</td><td>LLM-ZS</td><td>276</td><td>319</td><td>51</td><td>516</td><td>71.86%</td><td>48.80%</td></tr><tr><td>Skill-ZS</td><td>446</td><td>149</td><td>52</td><td>515</td><td>57.14%</td><td>48.80%</td></tr><tr><td rowspan="2">CMC</td><td>LLM-ZS</td><td>387</td><td>37</td><td>25</td><td>249</td><td>40.97%</td><td>39.26%</td></tr><tr><td>Skill-ZS</td><td>399</td><td>25</td><td>82</td><td>192</td><td>31.09%</td><td>39.26%</td></tr></table>

This pattern extends cross-dataset evaluation beyond a single source-to-target drop. Prior work shows that genre, annotation consistency, and evaluation setting can alter metaphoridentification performance [1], [2], [3]. The present comparison distinguishes at least three desirable properties: a high native peak, a high external average, and a high external floor with limited variation. The explicit contextual- and basicmeaning criteria in the Skill may be less tied to CMRE-specific label regularities than supervised parameter updates, offering a plausible account of its more even profile. The results therefore support evaluating procedural guidance by both its level and its consistency across datasets.

## B. Stability Through a Decision Trade-off

The matched zero-shot comparison shows how this profile is produced at the output level. Adding the Skill consistently lowers the rate of metaphorical predictions. On CCIME, where the direct model strongly overpredicts the positive class relative to gold prevalence, this shift removes many false positives with almost no recall loss. On CMRE Test and CMC, the same shift moves the predictions below the gold metaphor prevalence and produces substantially more false negatives. The Skill thus changes the operating tendency of the base model rather than delivering a uniform accuracy improvement.

Annotation policy provides a useful interpretation of this asymmetry. CCIME assigns similes and hyperbole to the nonmetaphorical class, whereas CMRE retains released-positive similes. A stricter requirement for supported contextual–basic contrast can therefore align well with CCIME’s negativeclass boundary while suppressing true positives under another policy. Stability here refers to aggregate performance rather than invariant errors: the balance between precision and recall still changes by dataset. Similar operational criteria have been implemented through model structure, explicit sense comparison, prompting, and rule scripts [6], [7], [12], [9], [10].

## C. Choosing a Knowledge-Supply Regime

The comparison suggests that the preferred regime depends on the evaluation objective. Fine-tuning is favored when maximizing performance on the labeled source distribution is central, and LLM-FT offers the strongest external average in this study. Skill-ZS becomes attractive when the observed external floor and variation across datasets receive greater weight. Because its procedural criteria remain explicit at inference time, the Skill also supports direct revision and inspection without another parameter-update cycle.

These routes are complementary rather than mutually exclusive. One promising direction is to combine supervised adaptation with procedural checks that constrain overly broad metaphor predictions. Such hybrids should be assessed with class-specific measures, prespecified decision costs, controlled rule ablations, and prompt-sensitivity tests so that gains in average performance can be separated from gains in crossdataset stability.

## VI. LIMITATIONS

The empirical scope contains four prespecified conditions, one base LLM, one expert-informed Skill, and three Chinese sentence-level datasets. CCIME and CMC provide two observed external settings; their external mean, external floor, and the three-dataset range do not establish behavior on unobserved genres, languages, annotation schemes, or metaphor constructions. Annotation policies are not fully aligned: CMRE retains source-positive similes as metaphorical, whereas CCIME assigns similes and hyperbole to its non-metaphorical class. Transfer to CCIME therefore partly measures adaptation to a changed labeling policy rather than domain change alone.

Fine-tuned conditions provide means and sample SDs across three seeds, whereas each zero-shot condition contributes one deterministic formal run. This asymmetry prevents comparable uncertainty estimates across all four conditions. The Skill was refined on 60 AI-assisted synthetic specification sentences, and the exact authoring-model identifier was not retained. The study did not independently reannotate those items or obtain external expert validation of every rule. Aggregate confusion counts identify the condition-level shift introduced by the complete Skill but cannot identify instruction-level causes; controlled ablation and prespecified instance-level analysis are needed for causal attribution.

The evaluation is sentence-level and binary. It neither identifies the exact metaphor-related word nor directly measures explanation quality, procedural compliance, calibration, or human interpretability. The two external datasets also differ in several properties, including resource construction and class distribution, so their performance differences cannot be attributed to a single domain-shift factor.

## VII. CONCLUSION

Across three Chinese metaphor datasets, the frozen expertinformed Skill produces the most even observed performance profile: it has the highest external floor and the smallest three-dataset Macro-F1 range, while its external mean remains within 0.60 points of LLM-FT. Fine-tuning retains the strongest native performance, and LLM-FT attains the highest external average. These outcomes distinguish crossdataset stability from peak accuracy and show why both should be reported. To our knowledge, this is the first study to compare an expert-informed procedural Skill with task-specific fine-tuning in the same cross-dataset evaluation of Chinese sentence-level metaphor identification.

The matched zero-shot comparison connects the Skill’s profile to a consistent reduction in metaphorical predictions. This shift substantially reduces false positives on CCIME while increasing false negatives on CMRE Test and CMC, demonstrating that a stable aggregate profile can contain different class-specific trade-offs. Expert-informed procedural prompting therefore provides a complementary route for supplying task knowledge: it yields a comparatively high performance floor across the observed datasets without updating model parameters. The profiles also motivate hybrid systems in which explicit operational criteria constrain training or posttraining decisions while fine-tuning preserves source-specific sensitivity.

## ETHICS AND DATA AVAILABILITY

The study uses existing research datasets and modelgenerated predictions. No new personal or human-subject data were collected. The 60 synthetic specification sentences were produced through an AI-assisted authoring process for instruction development and are not presented as human annotation. Dataset access and redistribution remain subject to original release terms. The supplementary package will include preprocessing records, prompts, the frozen Skill, runtime configurations, instance-level predictions, evaluation code, and the figure source where redistribution terms permit.

## REFERENCES

[1] E. Aghazadeh, M. Fayyaz, and Y. Yaghoobzadeh, “Metaphors in pretrained language models: Probing and generalization across datasets and languages,” in Proc. ACL, 2022, pp. 2037–2050.

[2] O. O. Uduehi and R. C. Bunescu, “An expectation-realization model for metaphor detection,” in Proc. FigLang, 2024, pp. 79–84.

[3] S. Reimann and T. Scheffler, “Metaphors in online religious communication: A detailed dataset and cross-genre metaphor detection,” in Proc. LREC-COLING, 2024, pp. 11 236–11 246.

[4] Pragglejaz Group, “MIP: A method for identifying metaphorically used words in discourse,” Metaphor and Symbol, vol. 22, no. 1, pp. 1–39, 2007.

[5] G. J. Steen, A. G. Dorst, J. B. Herrmann, A. A. Kaal, T. Krennmayr, and T. Pasma, A Method for Linguistic Metaphor Identification: From MIP to MIPVU. John Benjamins, 2010.

[6] R. Mao, C. Lin, and F. Guerin, “End-to-end sequential metaphor identification inspired by linguistic theories,” in Proc. ACL, 2019, pp. 3888–3898.

[7] M. Choi et al., “MelBERT: Metaphor detection via contextualized late interaction using metaphorical identification theories,” in Proc. NAACL-HLT, 2021, pp. 1763–1773.

[8] Y. Li, S. Wang, C. Lin, and F. Guerin, “Metaphor detection via explicit basic meanings modelling,” in Proc. ACL, 2023, pp. 91–100.

[9] S. Reimann and T. Scheffler, “Using large language models to perform MIPVU-inspired automatic metaphor detection,” in Proc. Analogy-Angle II, 2025, pp. 10–21.

[10] W. Huang and M. Liu, “Interpretable Chinese metaphor identification via LLM-assisted MIPVU rule script generation: A comparative protocol study,” arXiv preprint arXiv:2603.10784, 2026.

[11] M. Fuoli, W. Huang, J. Littlemore, S. Turner, and E. Wilding, “Metaphor identification using large language models: A comparison of RAG, prompt engineering, and fine-tuning,” Applied Corpus Linguistics, vol. 6, no. 2, 2026, Art. no. 100204, doi: 10.1016/j.acorp.2026.100204.

[12] M. Elzohbi and R. Zhao, “ContrastWSD: Enhancing metaphor detection with word sense disambiguation following the metaphor identification procedure,” in Proc. LREC-COLING, 2024, pp. 3907–3915.

[13] P. Chen, C. Yang, and Q. Huang, “Merely judging metaphor is not enough: Research on reasonable metaphor detection,” in Findings of EMNLP, 2024, pp. 5850–5860.

[14] Y. Tsvetkov, L. Boytsov, A. Gershman, E. Nyberg, and C. Dyer, “Metaphor detection with cross-lingual model transfer,” in Proc. ACL, 2014, pp. 248–258.

[15] L. Yang et al., “Out-of-distribution generalization in natural language processing: Past, present, and future,” in Proc. EMNLP, 2023, pp. 4533– 4559.

[16] E. Sanchez-Bayona and R. Agerri, “Metaphor and large language models: When surface features matter more than deep understanding,” in Findings of ACL, 2025, pp. 17 462–17 477.

[17] E. Sanchez-Bayona and R. Agerri, “Meta4XNLI: A cross-lingual parallel corpus for metaphor detection and interpretation,” Computational Linguistics, vol. 52, no. 1, pp. 191–235, 2026.

[18] G. Chen et al., “Chinese metaphorical relation extraction,” in Findings of EMNLP, 2023, pp. 9085–9095.

[19] BLCU ICALL, “CCL2026 Chinese conversational meaning and metaphor evaluation,” 2026, [Online]. Available: https://github.com/ blcuicall/CCIME2026. Accessed: Aug. 24, 2026.

[20] Y. Li, C. Lin, and F. Guerin, “CM-Gen: A neural framework for Chinese metaphor generation with explicit context modelling,” in Proc. COLING, 2022, pp. 6468–6479.

[21] Y. Cui, W. Che, T. Liu, B. Qin, S. Wang, and G. Hu, “Revisiting pretrained models for Chinese natural language processing,” in Findings of EMNLP, 2020, pp. 657–668.

[22] Qwen Team, “Qwen3.6-27B model card,” 2026, [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-27B. Accessed: Aug. 24, 2026.

[23] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022, [Online]. Available: https://openreview.net/forum? id=nZeVKeeFYf9. Accessed: Aug. 24, 2026.

[24] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 10 088–10 115.