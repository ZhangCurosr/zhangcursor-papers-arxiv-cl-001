# Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing

Yi Liu University of Science and Technology of China scnuliuyi@mail.ustc.edu.cn

## Abstract

Large language models (LLMs) are increasingly used to analyze and rewrite news, yet current framing studies mainly evaluate generation, detection, or whether rewritten text appears more neutral. They do not directly show whether a model can undo a known framing transformation while keeping the facts fixed. We introduce a controlled inversion test over three established textual realizations of framing: evaluative lexis, agency realization, and information salience. Across 60 news articles and three intervention strengths, this yields 540 paired variants with preserved atomic facts and recorded edits. Across Qwen, DeepSeek, and Kimi, factual preservation remains near 0.84, whereas intervention reversal is 0.044–0.068. Even when both framing type and direction are recognized correctly, pooled reversal reaches 0.071. These results reveal a clear separation between factual fidelity, framing recognition, and framing inversion: recognizing how an article is framed does not imply that the framing can be undone.

keywords: large language models, news framing, framing inversion, computational social science, robustness

## 1 Introduction

LLMs are becoming practical instruments for computational social science, supporting annotation, explanation, and large scale analysis of social text [1–3]. News is a consequential setting because framing can change how the same event is interpreted without requiring a change in its core facts. Communication theory characterizes framing through selection and salience, including the placement of information [4], while news discourse analysis locates framing cues across syntactic, thematic, and rhetorical organization [5]. Linguistic work supplies corresponding realization mechanisms: appraisal theory treats evaluative wording as a systematic resource for stance [6], and transitivity analyses of news show that grammatical choices about actors and goals shape representations of responsibility [7]. This motivates three realization levels: evaluative lexis, agency realization, and information salience, corresponding to lexical, clause, and discourse resources in news framing.

Recent LLM work measures biased news generation, framing detection, summary bias, and neutralization [8–12]. These tasks reveal whether framing is produced, recognized, or reduced, but not whether a model can reverse a known presentation change. We make the forward transformation observable. From a reference article x, we construct $y = T ( x ; o , s , d )$ while preserving an atomic factual inventory and recording the injected edits. The resulting pair turns framing into an inverse problem: can a model identify what changed in y, and can it undo those changes without altering the facts? Figure 1 summarizes this distinction; Figure 2 gives the full benchmark and task interface.

![](images/2feb56759f085f8016eac7965e0b2d0c978b9ab8e12cb598cfccbaf28b466c17.jpg)  
Figure 1: Controlled inversion asks whether a fact-preserving frame can be recognized and then reversed.

This formulation makes a distinction that aggregate bias scores are not designed to isolate: recognition and inversion are diferent operations. The contributions of this paper are summarized as follows:

• Controlled framing inversion benchmark. We introduce paired news variants in which fact-preserving framing interventions are explicitly constructed and the injected presentation changes are recorded.

• Multi-stage evaluation protocol. We separate framing recognition, factual preservation, and intervention reversal as independent capabilities, enabling direct measurement of whether recognized framing can be undone.

• Capability separation analysis. Experiments across three contemporary LLM families reveal that factual fidelity remains substantially easier than presentation-structure recovery, showing that recognizing a framing intervention does not imply successful inversion.

## 2 Related Work

News framing and computational media bias. Classic framing work links news interpretation to selection, salience, responsibility, and recurring issue frames [4,13,14]. Pan and Kosicki operationalize framing through syntactic, script, thematic, and rhetorical structures [5]; linguistic accounts complement this view with evaluative stance [6], actor representation through transitivity [7], and the organization of news stories as discourse [15]. Computational resources then make framing and bias measurable at scale, including the Media Frames Corpus, BASIL, MBIC, and broader surveys of media bias [16–19].

LLM measurement, framing, and rewriting. LLMs are now used as annotators and social measurement tools [1–3]. News-focused work covers generated bias, framing detection, summary fairness, and neutralization [8–10, 12, 20, 21]; related reframing work studies span-level bias rewriting and positive reframing [11, 22]. HELM and broader surveys motivate multi-dimensional evaluation [23, 24], while BBQ, CrowS-Pairs, and BOLD provide complementary bias tests [25–27]. Prompt formatting and example order afect model behavior [28, 29], and explicit reasoning can alter performance or faithfulness [30–34]. Controlled inversion adds a diferent target: because the source, factual inventory, and forward intervention are known, reconstruction can be scored against the presentation change that produced the input.

![](images/f3c0d0ff5676d0aded838e3801e380979b5d9174213450dacfe6881e54c318e1.jpg)  
Figure 2: Controlled benchmark, fixed D0 to R0 interface, and a real S011 case. The right panel uses three variants at medium strength from one source to visualize the headline separation: DeepSeek recognizes the favorable salience intervention, while reconstruction preserves the injected ordering (IRR = 0).

## 3 Controlled Framing Inversion

## 3.1 Problem and framing operators

Let x be a reference news article and F(x) its atomic factual inventory. We construct

$$
y = T ( x ; o , s , d ) , \qquad F ( y ) = F ( x ) ,\tag{1}
$$

where o denotes the framing operator, s ∈ {low, medium, high} its strength, and d ∈ {favorable, unfavorable} its direction toward a target actor. The generator also records the injected edit set $E = \{ e _ { 1 } , \ldots , e _ { k } \}$ An evaluated model first predicts the framing, $\hat { z } ~ = ~ f ( y )$ , and then reconstructs $\hat { x } ~ = ~ g ( y , \hat { z } )$ Reconstruction receives the model’s own detection output rather than the ground truth intervention.

The operators instantiate the three realization sites motivated above. Lexical framing manipulates evaluative wording, corresponding to linguistic resources for attitude and stance [6]; sentence and paragraph order remain fixed, and low, medium, and high strength use two, four, and six edits. Agency prominence manipulates grammatical subject, voice, and attribution prominence, drawing on the role of transitivity and actor assignment in news representation [7]; the three strengths use one, two, and three interventions without changing who performed each action. Salience framing manipulates placement and document prominence, directly reflecting framing through selection and salience [4, 5]; it ranges from a local reorder to reorganization of the lead and document order. No operator may add or delete source facts, alter entities or quantities, or introduce new causal claims.

## 3.2 Construction and evaluation tasks

The benchmark contains 60 English news sources, each canonicalized to 180–260 words and represented by 6–10 atomic facts. Three operators at three strengths yield nine framed variants per source, for 540 variants in total; direction is assigned once per source and operator pair with 90 favorable and 90 unfavorable assignments. A disjoint 10-source development set fixes prompt templates and intervention budgets before evaluation.

Controlled variants are produced with GLM-5.2 under a constrained generation and validation procedure. The generator receives the reference article, immutable facts, target actor, operator, direction, and strength, and returns the transformed article together with its edit map. Validation enforces factual preservation, absence of unsupported additions, the requested framing condition, professional plausibility, and operator isolation. All 540 constructed variants satisfy these checks, giving a benchmark of 540 validated framed articles plus 60 clean controls.

Each evaluated model then performs two tasks under a fixed interface. Detection (D0) receives one article, is explicitly told that no presentation problem may be present, considers lexical choice, agency prominence, and salience ordering, and returns a compact JSON record with issue type, direction, target actor, confidence, evidence, and explanation. Reconstruction (R0) receives the framed article y and that same model’s D0 record, and produces a factually faithful, minimally framed rewrite while preserving supported propositions, names, dates, quantities, quotations, attribution, actor roles, and causal relations. The source x, atomic facts $F ( x )$ , ground truth operator and direction, and edit set E are withheld; exact original wording is not required.

Prompt protocol. Given documented sensitivity to prompt formatting and order [28, 29], development fixes task wording, field order, and output schema before evaluation; D0 and R0 are then held fixed across model families. Figure 2 summarizes the benchmark, task interface, and a real example of recognition versus inversion.

## 3.3 Metrics

Detection is summarized by four-class macro-F1, with direction accuracy and the false positive rate on clean news as diagnostics. Reconstruction uses factual preservation and intervention reversal. We compute an atomic fact preservation score, FactF1, from deterministic lexical and numeric compatibility with the source facts. Let $E _ { \mathrm { r e v } } ( { \hat { x } } ) \subseteq E$ contain the injected edits reversed in reconstruction ˆx. We define

$$
\mathrm { I R R } ( \hat { x } ; E ) = \frac { | E _ { \mathrm { r e v } } ( \hat { x } ) | } { | E | } .\tag{2}
$$

For conditional analysis, $D _ { \mathrm { { c o r r e c t } } }$ requires the two variables that parameterize the intervention, operator and direction, to match ground truth; target actor text is auxiliary. We pool reversed and injected edits within $D _ { \mathrm { { c o r r e c t } } }$ and its complement. Uncertainty for reconstruction metrics is estimated with 5,000 bootstrap replicates clustered by source article.

## 4 Experiments and Results

## 4.1 Evaluation models

We evaluate DeepSeek-V4-Flash, Qwen-Plus, and Kimi-K2.6 under direct inference with tool and web access disabled; output coverage for each model is reported in Table 1. We also evaluate each provider’s native thinking configuration on the 180 variants at high strength with the task prompt unchanged, using it as a robustness check across inference configurations; Table 4 reports matched coverage.

Table 1: Core direct results. Panel (a) reports four-class macro-F1, direction accuracy, clean false positive rate, joint type and direction recognition, and coverage. Panel (b) reports reconstruction quality with 95% confidence intervals from bootstrap resampling clustered by source.  
(a) Detection and calibration
<table><tr><td>Model</td><td>F1</td><td>Dir.</td><td>FP↓</td><td>Exact</td><td>D0 n</td><td>Clean n</td></tr><tr><td>DeepSeek</td><td>.247</td><td>.217</td><td>.167</td><td>.128</td><td>600</td><td>60</td></tr><tr><td>Qwen</td><td>.236</td><td>.226</td><td>.250</td><td>.130</td><td>600</td><td>58</td></tr><tr><td>Kimi</td><td>.374</td><td>.533</td><td>.754</td><td>.312</td><td>571</td><td>57</td></tr></table>

(b) Reconstruction
<table><tr><td>Model</td><td>FactF1</td><td>F1 CI</td><td>IRR</td><td>IRR CI</td><td>R0 n</td></tr><tr><td>DeepSeek</td><td>.840</td><td>[.818,.862]</td><td>.044</td><td>[.032,.057]</td><td>540</td></tr><tr><td>Qwen</td><td>.834</td><td>[.815,.853]</td><td>.068</td><td>[.051,.085]</td><td>522</td></tr><tr><td>Kimi</td><td>.837</td><td>[.816,.859]</td><td>.054</td><td>[.041,.068]</td><td>513</td></tr></table>

Table 2: Reversal conditioned on recognition. “Exact” requires the correct framing type and direction; “Other” contains all remaining aligned reconstructions. Pooled counts are 75 of 1,052 reversed edits for Exact and 139 of 2,832 for Other.
<table><tr><td>Model</td><td colspan="4">Exact n Other n IRR |Exact IRR |Other Gain</td></tr><tr><td>DeepSeek</td><td>69 471</td><td>.066</td><td> $. 0 3 8 + . 0 2 7$ </td><td></td></tr><tr><td>Qwen</td><td>68 454</td><td>.092</td><td> $. 0 6 2 + . 0 3 0$ </td><td></td></tr><tr><td>Kimi</td><td>160 353</td><td>.065</td><td> $. 0 4 6 ~ { + . 0 1 8 }$ </td><td></td></tr><tr><td>Pooled</td><td>297 1278</td><td>.071</td><td> $. 0 4 9 + . 0 2 2$ </td><td></td></tr></table>

## 4.2 Facts survive while frames persist

Across all three model families, reconstruction preserves factual content far more reliably than it reverses the injected framing: FactF1 remains near 0.84, whereas IRR lies at 0.044–0.068 (Table 1). This is the paper’s first separation: factual fidelity and framing recovery measure diferent capabilities. The model profiles separate further across stages: Kimi leads Macro-F1 and exact recognition, Qwen attains the highest overall IRR, and DeepSeek the lowest clean false positive rate. Controlled inversion therefore exposes sensitivity, calibration, and recovery as distinct axes rather than a single aggregate rank.

## 4.3 Recognizing is not reversing

Exact recognition occurs in 18.9% of aligned reconstructions. Conditioning on exact recognition raises pooled IRR from 0.049 to 0.071 (Table 2), a 45% relative increase; gains across models range from 0.018 to 0.030. Yet 75 of 1,052 injected edits are reversed in this subset, so 92.9% of logged interventions remain despite correct type and direction. The S011 case in Figure 2(c) makes this separation concrete: DeepSeek identifies the salience mechanism and favorable direction, while reconstruction leaves the framed ordering unchanged, yielding IRR = 0. The conditional counts therefore isolate inversion as an additional operation rather than an automatic consequence of recognition.

Table 3: Results by mechanism. Detection reports class F1; reconstruction reports IRR for the three injected framing operators.
<table><tr><td>Class</td><td>DeepSeek Qwen Kimi Det. IRR Det. IRR Det. IRR</td></tr><tr><td>None</td><td>.212 .203 .164</td></tr><tr><td>Lexical</td><td>.486.047.393 .078 .604 .041</td></tr><tr><td></td><td>Agency .000 .061 .163 .078 .419 .091</td></tr><tr><td></td><td>Salience .290 .012 .187 .025.308 .038</td></tr></table>

Table 4: Native thinking on framing at high strength. Deltas are thinking minus direct scores on paired outputs; task prompts are unchanged.
<table><tr><td>Model Det. nR0 n ∆Dir. ∆F1∆FactF1∆IRR</td></tr><tr><td>DeepSeek 180 180-.033-.041 -.007 +.025</td></tr><tr><td>Qwen 179174-.060-.042 -.005-.021</td></tr><tr><td>Kimi 157156-.286-.186 -.007 +.000</td></tr></table>

## 4.4 Where framing is encoded matters

The mechanism pattern aligns with the linguistic locus of the intervention. Evaluative lexis is locally explicit and has the highest class F1 for every model. Salience is distributed over placement and document structure and has the lowest reversal rate in all three families. The ranking across stages is especially informative: lexical framing is the easiest mechanism to detect in all three models, yet agency has higher IRR for DeepSeek and Kimi and ties lexical for Qwen. Thus cue visibility does not determine edit recovery. FactF1 remains in the narrow 0.83–0.84 range across operators, concentrating the mechanism efect in framing recovery rather than factual retention.

Strength reveals a visibility gradient. From low to high strength, direction accuracy rises by .16, .13, and .09 for DeepSeek, Qwen, and Kimi, while IRR rises by .043, .047, and .038. FactF1 changes much less. The same ordering therefore appears across all three model families and both recognition and inversion: weaker framing preserves the factual substrate while making the presentation change less visible. Importantly, IRR at high strength remains below .09 for every model (.059, .085, and .069), so the inversion separation persists even at the most visible end of the controlled scale.

## 4.5 The separation persists under native thinking

Native thinking leaves the central separation intact. IRR shifts heterogeneously across models (+.025 for DeepSeek, -.021 for Qwen, and +.000 for Kimi), while FactF1 changes by less than .01 for every model. The separation between recognition and inversion therefore remains visible under a second inference configuration.

## 4.6 Regularities across models

Three regularities recur across all model families: exact recognition raises IRR, salience is the least reversible operator, and increasing intervention strength raises both direction accuracy and IRR while FactF1 remains comparatively stable. The mechanism margins are sizable: lexical detection exceeds the strongest alternative framing class by .185–.206 F1, while salience IRR is 59–80% below each model’s strongest operator. At the same time, the model leading each axis changes: Kimi on recognition, Qwen on reversal, and DeepSeek on clean calibration. The benchmark therefore reveals a reproducible structure across attitude, responsibility, and discourse prominence while keeping recognition, calibration, and recovery empirically distinct.

## 5 Discussion

Controlled inversion turns LLM news analysis into a measurement problem with multiple axes. Clean controls expose calibration, framed variants expose sensitivity, and the source article together with the logged edit map separates factual fidelity from actual correction. The resulting profiles show why corpus coding, editorial auditing, and data curation benefit from reporting these quantities separately rather than compressing them into one quality score [23, 24]. They also motivate model selection by stage because recognition, clean calibration, and inversion peak in diferent model families.

The ranking itself demonstrates why the stages should remain separate. Kimi’s exact recognition (.312) exceeds Qwen’s (.130), while Qwen has higher IRR (.068 versus .054). A single aggregate score would hide this reversal in model order. Controlled inversion instead makes the handof from detector to reconstructor measurable under one inverse objective with fixed facts, allowing an auditable pipeline to choose detection and recovery components for the capability each stage actually requires.

For editing with LLMs, the same decomposition also provides a verifiable workflow: detection identifies the presentation mechanism, reconstruction performs the edit, and IRR checks the result against the known intervention rather than against surface fluency alone. This makes controlled inversion useful not only as a benchmark, but also as a design pattern for auditing whether an automated editorial correction actually changed the intended representational choice while preserving content.

The linguistic grounding also yields a direct robustness target. Lexical evaluation is comparatively visible, whereas agency and especially salience require tracking who is foregrounded and where otherwise valid facts are placed. Because paired articles share the same factual inventory, supervision can target the logged presentation change itself: evaluative markers for lexical framing, actor and attribution realization for agency, and information order for salience. Controlled inversion thus turns a broad debiasing goal into verifiable, linguistically grounded corrections while keeping content fixed.

## 6 Conclusion

We introduced controlled inversion for fact-preserving news framing, grounding interventions in evaluative lexis, agency realization, and discourse salience. Across three LLM families, the benchmark separates factual fidelity, framing recognition, and framing inversion, making framing recovery directly measurable. Exact recognition consistently raises reversal but does not determine it; salience is the least reversible operator, and stronger interventions are easier to recover while factual fidelity remains stable. By scoring reconstruction against the known forward intervention, controlled inversion makes recovery the target by construction rather than neutrality by impression. It therefore provides an objective based on recorded edits for LLMs that preserve news facts while changing their presentation.

![](images/1c58d4c5a328e717b2ae240139821bde19d0b59c9543d5ec42332b08b0572fbc.jpg)  
Figure 3: Framing strength acts as a visibility axis: direction accuracy and IRR both increase from low to high strength for all three model families.

## References

[1] C. Ziems, W. Held, O. Shaikh, J. Chen, Z. Zhang, and D. Yang, “Can large language models transform computational social science?” Comput. Linguistics, vol. 50, no. 1, pp. 237–291, 2024.

[2] F. Gilardi, M. Alizadeh, and M. Kubli, “ChatGPT outperforms crowd workers for textannotation tasks,” Proc. Natl. Acad. Sci. U.S.A., vol. 120, no. 30, e2305016120, 2023.

[3] L. P. Argyle, E. C. Busby, N. Fulda, J. R. Gubler, C. Rytting, and D. Wingate, “Out of one, many: Using language models to simulate human samples,” Political Analysis, vol. 31, no. 3, pp. 337–351, 2023.

[4] R. M. Entman, “Framing: Toward clarification of a fractured paradigm,” J. Commun., vol. 43, no. 4, pp. 51–58, 1993.

[5] Z. Pan and G. M. Kosicki, “Framing analysis: An approach to news discourse,” Political Communication, vol. 10, no. 1, pp. 55–75, 1993.

[6] J. R. Martin and P. R. R. White, The Language of Evaluation: Appraisal in English. Basingstoke, U.K.: Palgrave Macmillan, 2005.

[7] J. Li, “Transitivity and lexical cohesion: Press representations of a political disaster and its actors,” J. Pragmatics, vol. 42, no. 12, pp. 3444–3458, 2010.

[8] J. Yoo and Y. Shin, “Fair or Framed? Political bias in news articles generated by LLMs,” in Proc. EMNLP, 2025, pp. 16904–16930.

[9] V. Pastorino and N. S. Moosavi, “Frame In, Frame Out: Measuring framing bias in LLMgenerated news summaries,” in Proc. \*SEM, 2026, pp. 378–384.

[10] V. Pastorino, J. A. Sivakumar, and N. S. Moosavi, “Decoding news narratives: A critical analysis of large language models in framing detection,” in Proc. PoliticalNLP @ LREC, 2026, pp. 17–28.

[11] A. Y. Radwan, A. ElKady, S. Chaduvula, M. Hafez, A. Krishnan, and S. Raza, “UnBias-Plus: Detect, explain, and rewrite bias,” arXiv:2606.23412, 2026.

[12] A. Mitra et al., “Breaking Bias: A context-aware multimodal framework for detecting and neutralizing ideological bias in news,” ACM Trans. Knowl. Discov. Data, vol. 20, no. 3, Art. no. 38, 2026.

[13] D. A. Scheufele, “Framing as a theory of media efects,” J. Commun., vol. 49, no. 1, pp. 103–122, 1999.

[14] H. A. Semetko and P. M. Valkenburg, “Framing European politics: A content analysis of press and television news,” J. Commun., vol. 50, no. 2, pp. 93–109, 2000.

[15] A. Bell, The Language of News Media. Oxford, U.K.: Blackwell, 1991.

[16] D. Card, A. E. Boydstun, J. H. Gross, P. Resnik, and N. A. Smith, “The Media Frames Corpus: Annotations of frames across issues,” in Proc. ACL-IJCNLP, 2015, pp. 438–444.

[17] L. Fan et al., “In plain sight: Media bias through the lens of factual reporting,” in Proc. EMNLP-IJCNLP, 2019, pp. 6343–6349.

[18] T. Spinde, L. Rudnitckaia, K. Sinha, F. Hamborg, B. Gipp, and K. Donnay, “MBIC–A media bias annotation dataset including annotator characteristics,” in Proc. iConference, 2021.

[19] F. Hamborg, K. Donnay, and B. Gipp, “Automated identification of media bias in news articles: An interdisciplinary literature review,” Int. J. Digit. Libr., vol. 20, pp. 391–415, 2019.

[20] A. Tohidi, S. Haider, and D. J. Watts, “Rethinking news framing with large language models,” Sci. Rep., vol. 15, Art. no. 45592, 2025.

[21] N. Huang, I. Maab, and J. Yamagishi, “When bigger isn’t better: A comprehensive fairness evaluation of political bias in multi-news summarisation,” in Proc. ACL, 2026, pp. 19532–19563.

[22] C. Ziems, M. Li, A. Zhang, and D. Yang, “Inducing positive perspectives with text reframing,” in Proc. ACL, 2022, pp. 3682–3700.

[23] P. Liang et al., “Holistic evaluation of language models,” Trans. Mach. Learn. Res., 2023.

[24] I. O. Gallegos et al., “Bias and fairness in large language models: A survey,” Comput. Linguistics, vol. 50, no. 3, pp. 1097–1179, 2024.

[25] A. Parrish et al., “BBQ: A hand-built bias benchmark for question answering,” in Findings ACL, 2022, pp. 2086–2105.

[26] N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman, “CrowS-Pairs: A challenge dataset for measuring social biases in masked language models,” in Proc. EMNLP, 2020, pp. 1953–1967.

[27] J. Dhamala et al., “BOLD: Dataset and metrics for measuring biases in open-ended language generation,” in Proc. ACM FAccT, 2021, pp. 862–872.

[28] M. Sclar, Y. Choi, Y. Tsvetkov, and A. Suhr, “Quantifying language models’ sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting,” in Proc. ICLR, 2024.

[29] Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp, “Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity,” in Proc. ACL, 2022, pp. 8086–8098.

[30] J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Adv. Neural Inf. Process. Syst., vol. 35, pp. 24824–24837, 2022.

[31] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, “Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023.

[32] T. Lanham et al., “Measuring faithfulness in chain-of-thought reasoning,” arXiv:2307.13702, 2023.

[33] G. Chochlakis, N. M. Pandiyan, K. Lerman, and S. Narayanan, “Larger language models don’t care how you think: Why chain-of-thought prompting fails in subjective tasks,” in Proc. ICASSP, 2025, pp. 1–5.

[34] H. Zhao, C. Zhao, Y. Li, Z. Zhang, and G. Liu, “Thinking in a Crowd: How auxiliary information shapes LLM reasoning,” in Proc. ICASSP, 2026.