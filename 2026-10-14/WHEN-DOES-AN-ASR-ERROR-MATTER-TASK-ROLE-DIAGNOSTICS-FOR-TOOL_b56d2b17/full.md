# WHEN DOES AN ASR ERROR MATTER? TASK-ROLE DIAGNOSTICS FOR TOOL-USING VOICE AGENTS

Xiaolong Wang<sup>∗</sup> Shijie Su Naqi Xiong

Leibniz University Hannover, Hannover, Germany

## ABSTRACT

In cascaded voice agents, ASR errors can have markedly different downstream consequences: a small transcription error may leave a tool call unchanged or alter its intent or arguments. Conventional word error rate (WER), however, does not capture these differences. We introduce SpeechEvalX, a paired diagnostic protocol that evaluates how task roles affected by ASR errors influence structured tool calls. Controlled edits are matched by source utterance, edit count, and operation, while task-role labels are compared with lexical severity, BERT cosine distance, Semantic-WER, and binary criticality. Across 594 source utterances, joint call accuracy falls from 82.3% with reference transcripts to 71.9–74.9% across three ASR front ends. Number and negation edits cause 21.9–25.7 percentage-point losses, versus 1.4–2.9 points for control edits. Detailed task-role labels improve source-grouped ROC-AUC from 0.73 to 0.78 over binary criticality, showing that downstream impact depends strongly on the task role affected.

Index Terms— ASR evaluation, voice agents, tool calling, error analysis, spoken language understanding

## 1. INTRODUCTION

A one-word recognition error may be harmless, or it may change a quantity, destination, or negation and thereby alter the requested action. For the cascade

$$
\mathrm { s p e e c h }  \mathrm { A S R }  \mathrm { L L M }  \mathrm { s t r u c t u r e d } \mathrm { a c t i o n } ,\tag{1}
$$

the relevant outcome is agreement between the predicted action and the original request; transcript fidelity alone does not determine it.

Task-aware ASR metrics and controlled-noise studies motivate evaluation beyond WER [1, 2, 3]. Paired text–audio tool-calling evaluation also separates decision, tool-selection, and argument errors [27]; neither clean-text controls nor this taxonomy is novel by itself.

SpeechEvalX asks whether task-role labels explain action failure beyond simpler descriptions at the same lexical edit severity. The primary comparison fixes the source utterance, edit count, edit operation, agent configuration, and reference action. A predictive analysis compares lexical features, BERT cosine distance, Semantic-WER, binary critical-span overlap, and the full role taxonomy, isolating the information added by role identity.

Our contributions are a paired diagnostic protocol and a direct test of task-role information beyond binary criticality. Matched edits quantify sensitivity by role; source-grouped failure prediction and second-model transfer test diagnostic value. Benchmark-audio errors separately describe practical mixtures. Code and recipes are available at github.com/invoker-qqwrv/SpeechEvalX.

![](images/111a3bf2fcbdb9bc2b1e5c66ec0d84a6c73c2a6f91be92ce1dba6ecad7ee9567.jpg)  
Fig. 1. Matched example: equal lexical severity, but only the number edit changes the structured action relative to the original reference.

## 2. RELATED WORK

Semantic Distance and Semantic-WER evaluate transcription quality with downstream usability in mind [1, 2]. Controlled transcriptionnoise analysis studies the effects of error type and severity on language-understanding tasks [3], while ASR-aware named-entity work shows that aggregate word errors can conceal failures on salient spans [4]. End-to-end and stack-propagation approaches connect speech to structured semantics [5, 6]. SLURP and SLUE provide task-oriented spoken-language resources [7, 8]; SUPERB and SUPERB-SG broaden capability evaluation [9, 10].

Whisper, Paraformer, and FunAudioLLM provide complementary ASR front ends [11, 12, 13]. VoiceBench and VoiceAgent-Bench evaluate broader voice-assistant and agent capabilities [17, 18]. ToolLLM and ToolACE study function selection and argument generation [19, 20], while AgentBench, BFCL, and the τ-bench family extend evaluation to agents and grounded outcomes [21, 22, 23, 24]. BFCL Audio, From Text to Voice, τ -Voice, and EVA-Bench address spoken tool use or end-to-end interaction [25, 27, 28, 26]. SpeechEvalX is deliberately narrower: it isolates the effect of transcript errors on structured actions through matched within-utterance edits and measures the information added by task-role labels.

## 3. PAIRED TASK-ROLE DIAGNOSTICS

## 3.1. Evaluation unit and reference action

For source i, let $r _ { i }$ be a verified reference transcript, $h _ { i }$ an observed ASR hypothesis on benchmark audio or a controlled corruption, and z<sub>i</sub> a reference action. An action is a single tool call $( f _ { i } , \theta _ { i } )$ or an explicit no-call decision. Multi-call cases require a separate adapter and are outside the primary analysis. For a fixed downstream configuration m, $\hat { z } _ { i m } ( x )$ is the prediction from input x, and

$$
y _ { i m } ( x ) = \mathbb { I } \{ \hat { z } _ { i m } ( x ) \equiv z _ { i } \} .\tag{2}
$$

Equivalence follows a frozen benchmark-specific scoring rule. It is not defined by agreement with the model output from the reference transcript, because that output can itself be wrong.

Table 1. Operational task-role taxonomy for span annotation.
<table><tr><td>Role</td><td>Inclusion rule</td></tr><tr><td>Number</td><td>Quantity, time, date, ordinal, or identifier aligned to an action value</td></tr><tr><td>Entity</td><td>Named person, place, organization, product, or account aligned to an action value</td></tr><tr><td>Negation</td><td>Overt cue whose scope changes whether or how the action is executed</td></tr><tr><td>Intent cue</td><td>Predicate or phrase that selects the call or no- call intent</td></tr><tr><td>Other slot</td><td>Schema-grounded argument value not covered above</td></tr><tr><td>Function word</td><td>Closed-class token outside every task-bearing span</td></tr><tr><td>Other non- critical</td><td>Remaining material outside every task-bearing span</td></tr></table>

Each pair links the same reference action to raw and canonical outputs, transcript alignment, span annotations, edit provenance, and configuration identifiers. Prompt, schema, context, decoding, parser, and scoring remain fixed; reference actions and labels are hidden from the agent. We audited 50 prespecified sources with three runs each for residual inference nondeterminism.

## 3.2. Span annotation and matched edits

The roles are number, entity, negation, intent cue, other slot value, function word, and other non-critical. They are assigned from the reference instruction and schema using rules fixed before scoring. Numeric and entity spans precede residual slot values, and negation precedes function-word membership. Remaining overlaps follow a fixed tie-break or are excluded; unresolved or implicit arguments are not forced into a span.

The first five roles are task-bearing and the last two are controls; “non-critical” means that no annotated schema role is present, not that an edit is guaranteed to be harmless. Two annotators independently labeled 120 sources containing 684 candidate edit spans. Exact span-boundary F1 was 0.93, task-role Cohen’s κ was 0.89, and eligibility agreement was 95.0%. Annotators were blinded to downstream model outputs. Disagreements were adjudicated against the frozen tool schema and annotation guideline before model scoring.

Define lexical severity with a fixed word tokenizer:

$$
S ( r _ { i } , h _ { i } ) = D _ { \mathrm { l e v } } ( r _ { i } , h _ { i } ) / | r _ { i } | .\tag{3}
$$

Primary interventions use the same source utterance, edit count, and edit operation. As Fig. 1 shows, replacing “the” with $\mathbf { \ddot { a } } ^ { , 5 }$ and $^ { \mathfrak { s } \mathfrak { s } } 2 2 ^ { \mathfrak { s } \mathfrak { s } }$ with “32” in a temperature request are both single substitutions. The original action remains the target because the altered transcript represents a recognition error, not a revised request. Matching controls lexical severity; it does not equalize semantic change or acoustic likelihood.

Role contrasts use the common eligible set $I _ { c d }$ with support reported; means from different sets are descriptive only. If a source has several variants in one role–operation stratum, an outcomeindependent frozen rule selects one, preventing prolific sources from dominating. Confusion-derived edits use development sources only. Observed multi-edit hypotheses retain multiple labels and are analyzed separately.

Table 2. Paired action outcomes. Both paths use the same original reference.
<table><tr><td>Reference</td><td>Corrupted</td><td>Interpretation</td></tr><tr><td>1</td><td>1</td><td>Preserved success</td></tr><tr><td>1</td><td>0</td><td>Harmed pair (H)</td></tr><tr><td>0</td><td>1</td><td>Rescued pair (R)</td></tr><tr><td>0</td><td>0</td><td>Persistent failure</td></tr></table>

## 3.3. Final-output error categories

Reference-correct/corrupted-incorrect pairs receive one category in this precedence order: malformed output; incorrect call/no-call decision; wrong tool; invalid or missing argument structure; and incorrect argument value. Precedence resolves overlaps but is not an execution timeline or claim about internal reasoning. Other pairs are reported separately; unresolved cases are unclassified.

## 3.4. Scoring and denominators

Joint call accuracy (JCA) is correct actions divided by all eligible examples, including no-call decisions. It measures structured-action accuracy, not external task completion. Malformed calls are failures; instrumentation failures and coverage are reported separately, and eligibility is outcome-independent.

Let $\dot { N } _ { C }$ count call-reference examples, $N _ { 0 }$ no-call references, C correct calls, and $C _ { 0 }$ correct no-call decisions. Then

$$
\mathrm { J C A } = ( C + C _ { 0 } ) / ( N _ { C } + N _ { 0 } ) .\tag{4}
$$

For transparent decomposition, let $T$ count predictions on callreference examples with a recoverable, correct tool name, including predictions with incorrect arguments. We report tool accuracy $T / N _ { C }$ and conditional argument exact match (Arg-EM), C/T. The latter requires a valid call and matching reference arguments under the frozen rule. Thus call-only JCA is $( T / N _ { C } ) ( C / T ) = C / N _ { C }$ when denominators are nonzero. Undefined ratios are reported as unavailable. No-call accuracy is $C _ { 0 } / N _ { 0 }$ .

For benchmark-audio hypotheses, corpus WER is total word edit distance divided by total reference words, not mean utterance WER. Text normalization and benchmark aliases are recorded.

## 3.5. Paired effects and uncertainty

For one variant per source and role c on eligible set $I _ { c } ,$ define the paired JCA loss

$$
L _ { c } = \frac { 1 } { | I _ { c } | } \sum _ { i \in I _ { c } } [ y _ { i m } ( r _ { i } ) - y _ { i m } ( h _ { i } ^ { ( c ) } ) ] = \frac { H _ { c } - R _ { c } } { | I _ { c } | } .\tag{5}
$$

A positive value denotes a net loss in action accuracy after the edit. We report N, H, and R with $L _ { c } .$ . Pairwise role contrasts restrict both conditions to $I _ { c d } ;$ bootstrap resampling operates on source groups. With one paired variant per source, McNemar’s test compares corrupted success outcomes for two roles, followed by Holm correction over prespecified contrasts. Multiple variants require clustered analysis rather than treatment as independent samples. Persistent failures are not counted as new harm caused by the altered transcript.

## 4. VALIDATION DESIGN AND REPRODUCIBILITY

## 4.1. Data, systems, and frozen rules

We sampled 600 candidates with seed 2027 from the English singletool subset of krutrim-ai-labs/VoiceAgentBench and the English single-turn subset of BFCL Audio [18, 25]. Six candidates failed prespecified eligibility checks and were excluded without replacement. The final cohort contains 594 source utterances, split evenly between benchmarks (297 each), with 540 call and 54 no-call references. Audio is decoded to 16-kHz mono float32 before inference.

Controlled-edit eligibility was determined before downstream inference. The 210-source intervention set is the intersection of sources that admit one unambiguous single-edit variant for every compared role under the frozen schema; the other 384 sources remain in the benchmark-audio ASR analysis. No source was selected using model outcomes. A fixed rule selects one variant per source and role, so every row of Table 4 uses the same sources.

The ASR front ends are Whisper large-v3, SenseVoice-Small, and Paraformer-large [11, 13, 12]. The primary and sensitivity agents are Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct [15, 16]. Exact repository identifiers are frozen in the released configuration. Inference uses bfloat16, temperature 0, top-p 1, at most 256 new tokens, and seed 2027. Prompt, schema, parser, canonicalizer, and scoring rules are fixed across paired conditions.

For WER, reference and hypothesis text undergo Unicode NFKC normalization, lowercasing, and punctuation deletion; numbers are not semantically rewritten. We report corpus WER, defined as total word edit distance divided by the total number of reference words.

## 4.2. Information added by task-role labels

For source utterances that yield a correct action from the reference transcript, the endpoint is action failure after corruption. We compare cumulative feature sets using the same regularized logisticregression family. Each edit variant is a prediction sample, and the source utterance is the grouping unit. B0 contains lexical severity, edit operation, length, and benchmark. B1 adds BERT embedding cosine distance, computed from bert-base-uncased by mean-pooling all token representations and taking cosine distance [14]. B2 adds Semantic-WER with its named-entity and sentiment weighting, a 0.6 word-similarity threshold, and the default insertion weight IW = 1 [2]. B3 adds a binary critical-span flag, and B4 adds the detailed task-role labels.

The primary contrast is B4 versus B3: both include the lexical and semantic controls, while B4 additionally identifies the role of the altered span. This contrast tests whether knowing the detailed task role adds information beyond knowing that a critical span was affected.

All variants of a source, as well as duplicate or paraphrase-linked templates when identifiable, remain in one split. Hyperparameters and preprocessing are fitted on training data only. We use the same five source-grouped folds for all feature sets and report out-of-fold ROC-AUC, average precision (AP), and paired source-bootstrap intervals for score differences. AUC is calculated only when an evaluation fold contains both outcome classes. Because reference transcripts and actions may be used to construct features, this is an offline diagnostic analysis.

## 4.3. Validation on a second model and ASR errors

The sensitivity agent processes the same reference inputs and frozen edit set, without selecting examples from the primary agent’s outcomes. We use the same source-grouped partitions and apply the diagnostic predictor trained on the primary model to the second model without refitting. ASR hypotheses from benchmark audio retain overlapping task-role labels and are analyzed separately from controlled single-edit interventions.

Table 3. ASR hypotheses on benchmark audio: 594 source utterances (%). Tool is accuracy on call-reference cases; Arg-EM is conditional on a recoverable correct tool.
<table><tr><td>Input</td><td>WER</td><td>Tool</td><td>Arg-EM</td><td>No-call</td><td>JCA</td></tr><tr><td>Reference</td><td>一</td><td>91.3</td><td>89.5</td><td>88.9</td><td>82.3</td></tr><tr><td>Whisper large-v3</td><td>6.3</td><td>86.9</td><td>85.1</td><td>85.2</td><td>74.9</td></tr><tr><td>SenseVoice-Small</td><td>7.2</td><td>86.1</td><td>83.9</td><td>83.3</td><td>73.2</td></tr><tr><td>Paraformer</td><td>8.1</td><td>85.2</td><td>83.3</td><td>81.5</td><td>71.9</td></tr></table>

Table 4. Matched interventions on the same 210 sources. $L _ { c }$ is the net JCA loss in percentage points; intervals are source-bootstrap 95% CIs.
<table><tr><td>Role</td><td>H</td><td>R</td><td> $L _ { c }$ </td><td>95% CI</td></tr><tr><td>Other non-critical</td><td>5</td><td>2</td><td>1.4</td><td>[-0.5, 3.8]</td></tr><tr><td>Function word</td><td>8</td><td>2</td><td>2.9</td><td>[0.5, 5.7]</td></tr><tr><td>Other slot value</td><td>24</td><td>4</td><td>9.5</td><td>[5.7, 13.8]</td></tr><tr><td>Intent cue</td><td>30</td><td>4</td><td>12.4</td><td>[8.1, 17.1]</td></tr><tr><td>Entity</td><td>40</td><td>4</td><td>17.1</td><td>[12.4, 22.4]</td></tr><tr><td>Number</td><td>51</td><td>5</td><td>21.9</td><td>[16.7, 27.6]</td></tr><tr><td>Negation</td><td>58</td><td>4</td><td>25.7</td><td>[20.0, 31.9]</td></tr></table>

For ASR hypotheses from benchmark audio, we report role prevalence and action failures within supported severity, benchmark, and role strata. Multi-edit hypotheses require multi-label features; their failures are not attributed entirely to each individual role. These errors are observational, so the associated failure rate is not an isolated effect of a single role. We therefore do not generalize controlled single-edit results to observed multi-edit errors without this separate validation. Multiple synthesized voices from one text remain in the same source group.

## 5. RESULTS

The 540 call and 54 no-call references yield 489, 445, 435, and 427 correct actions for reference transcripts and the three ASR conditions. Relative to reference transcripts, ASR hypotheses from benchmark audio reduce JCA by 7.4–10.4 percentage points. The error rate with reference transcripts is specific to the evaluated configuration and should not be interpreted as an irreducible floor. The repeated-inference audit comprised 50 source utterances and three runs per source; the prespecified first run provides all reported aggregate scores, while the repeated runs are retained to inspect nondeterminism.

Equal edit count does not imply equal risk of action failure. For the primary model, the mean paired JCA loss is 17.3 points across the five task-bearing roles and 2.1 points across the two controls; prespecified contrasts between task-bearing and control roles remain significant after Holm correction (p < 0.01).

Table 5 tests the central claim: detailed task-role labels add predictive information beyond lexical severity, semantic metrics, and binary criticality. B4 improves ROC-AUC over B3 from 0.73 to 0.78 (∆ = 0.05, 95% CI [0.02, 0.08]; paired source-bootstrap p = 0.003), and AP increases from 0.62 to 0.68. When the predictor is transferred to Llama-3.1-8B-Instruct without refitting, B4 reaches 0.74 versus 0.69 for B3. The direction is consistent across agents, supporting transfer of the diagnostic pattern. Without a corresponding uncertainty estimate, however, it does not establish a statistically significant transfer gain.

Table 5. Diagnostic value in source-grouped five-fold failure prediction. Feature sets are cumulative; the primary contrast is B4 versus B3.
<table><tr><td>Model</td><td>Added information</td><td>ROC-AUC</td><td>AP</td></tr><tr><td>B0</td><td>Lexical controls</td><td>0.62</td><td>0.48</td></tr><tr><td>B1</td><td>+ BERT embedding cosine</td><td>0.68</td><td>0.55</td></tr><tr><td>B2</td><td>+ Semantic-WER</td><td>0.70</td><td>0.58</td></tr><tr><td>B3</td><td>+ Critical-span flag</td><td>0.73</td><td>0.62</td></tr><tr><td>B4</td><td>+ Detailed task roles</td><td>0.78</td><td>0.68</td></tr><tr><td colspan="2">Primary contrast: B4–B3</td><td>+0.05</td><td>+0.06</td></tr></table>

ROC-AUC gain: 95% CI [0.02, 0.08], $p = 0 . 0 0 3$ (paired source bootstrap). AP gain is a point estimate. Endpoint: action failure after corruption among cases that are correct with the reference transcript.

Table 6. Paired JCA loss across downstream agents (percentage points).
<table><tr><td>Role</td><td>Primary</td><td>Sensitivity</td></tr><tr><td>Other non-critical</td><td>1.4</td><td>1.0</td></tr><tr><td>Function word</td><td>2.9</td><td>3.3</td></tr><tr><td>Other slot value</td><td>9.5</td><td>8.6</td></tr><tr><td>Intent cue</td><td>12.4</td><td>10.5</td></tr><tr><td>Entity</td><td>17.1</td><td>15.7</td></tr><tr><td>Number</td><td>21.9</td><td>19.0</td></tr><tr><td>Negation</td><td>25.7</td><td>23.3</td></tr></table>

## 5.1. Observed ASR errors and final-output failures

This analysis pools the three ASR conditions and allows overlapping role labels. Among hypotheses containing at least one error, the proportions followed by action failure are 35/104 for number, 37/128 for entity, 20/42 for negation, 23/87 for intent cue, 29/159 for other slot values, 19/201 for function words, and 17/236 for other noncritical edits. The numerators sum to 180 role-label occurrences, not necessarily 180 distinct failed hypotheses. These descriptive associations do not estimate the incremental B4–B3 benefit for observed ASR errors.

Argument-value mismatch is the most common final-output error, while 30.1% of harmed pairs change the call/no-call decision, tool choice, or output validity. Removing within-source severity matching raises the mean loss across task-bearing roles from 17.3 to 20.6 points, indicating that unmatched comparisons can exaggerate role effects. Restricting the analysis to deterministic substitutions gives 16.8 versus 2.0 points for task-bearing and control roles. Leave-domain-out B4 ROC-AUC is 0.72 when trained on VoiceAgentBench and tested on BFCL Audio, and 0.70 in the reverse direction.

## 6. DISCUSSION AND CONCLUSION

SpeechEvalX tests whether task-role annotations add diagnostic value under explicit controls for lexical severity. Paired action scoring separates failures caused by transcript corruption from failures already present with the reference transcript, while retaining cases in which corruption unexpectedly corrects an action. Final-output categories describe observed disagreements without claiming to locate a failure in time or reveal a hidden causal mechanism; the protocol is intentionally narrower than a general benchmark for voice agents.

Table 7. Final-output error categories among 216 harmed controlled pairs.
<table><tr><td>Category</td><td>Count</td><td>Share (%)</td></tr><tr><td>Call/no-call decision</td><td>17</td><td>7.9</td></tr><tr><td>Wrong tool</td><td>42</td><td>19.4</td></tr><tr><td>Argument structure</td><td>48</td><td>22.2</td></tr><tr><td>Argument value</td><td>103</td><td>47.7</td></tr><tr><td>Malformed output</td><td>6</td><td>2.8</td></tr></table>

The main evidence is the B4–B3 contrast, which isolates the information provided by role identity after accounting for lexical controls, semantic metrics, and binary criticality. The role-level ordering of paired JCA loss is complementary evidence. The proportions from observed ASR errors describe practical error mixtures; showing the same incremental value on those mixtures requires a separate predictive comparison.

During development, these diagnostics can support ASR and downstream-model selection, distinguish tool-selection failures from argument-construction failures, and prioritize regression tests for high-impact roles. Their practical value is to explain when transcript errors alter structured actions and to guide error analysis. Because the analysis uses reference transcripts and actions, it is intended for offline evaluation rather than deployment-time detection.

## 6.1. Audit trail and aggregation

Every aggregate result can be traced from source IDs to raw and canonical outputs, parser status, transcript alignments, task-role labels, edit provenance, and configuration identifiers. Analyses of benchmark-audio ASR, matched effects, source-grouped prediction, and model transfer use a common result registry. For observed ASR errors with overlapping labels, counts of role-label occurrences and counts of distinct harmed pairs are different aggregation units.

## 6.2. Threats to validity

Matched edits control lexical severity, but not semantic distance or acoustic likelihood. Observed ASR hypotheses can contain correlated edits. The 54 no-call cases, English-only data, current tool schemas, and synthetic speech limit external validity. Source support and class balance affect the precision of estimates from source-grouped prediction. Broader validation requires additional languages, APIs, agent families, and a separate analysis of the information added by task-role labels for observed ASR errors.

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This study used publicly available benchmark data and involved no new data collection or direct interaction with human participants. No additional ethical approval was required. The authors declare no conflicts of interest.

## 8. ACKNOWLEDGMENT

No external funding was received. OpenAI ChatGPT was used for language editing; all technical content and results were reviewed and verified by the authors.

## 9. REFERENCES

[1] S. Kim et al., “Semantic Distance: A new metric for ASR performance analysis towards spoken language understanding,” in Proc. Interspeech, 2021.

[2] S. Roy, “Semantic-WER: A unified metric for the evaluation of ASR transcript for end usability,” arXiv:2106.02016, 2021.

[3] O. Shapira, S. E. Chazan, and A. D. N. Cohen, “Measuring the effect of transcription noise on downstream language understanding tasks,” in Proc. ACL, 2025.

[4] P. Szymanski et al., “Why aren’t we NER yet? Artifacts of ASR errors in named entity recognition in spontaneous speech transcripts,” in Proc. ACL, 2023.

[5] L. Lugosch et al., “Speech model pre-training for end-to-end spoken language understanding,” in Proc. Interspeech, 2019.

[6] L. Qin et al., “A stack-propagation framework with token-level intent detection for spoken language understanding,” in Proc. EMNLP-IJCNLP, 2019.

[7] E. Bastianelli et al., “SLURP: A spoken language understanding resource package,” in Proc. EMNLP, 2020.

[8] S. Shon et al., “SLUE: New benchmark tasks for spoken language understanding evaluation on natural speech,” in Proc. ICASSP, 2022.

[9] S.-W. Yang et al., “SUPERB: Speech processing universal performance benchmark,” in Proc. Interspeech, 2021.

[10] H.-S. Tsai et al., “SUPERB-SG: Enhanced speech processing universal performance benchmark for semantic and generative capabilities,” in Proc. ACL, 2022.

[11] A. Radford et al., “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023.

[12] Z. Gao et al., “Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition,” in Proc. Interspeech, 2022.

[13] K. An et al., “FunAudioLLM: Voice understanding and generation foundation models for natural interaction between humans and LLMs,” arXiv:2407.04051, 2024.

[14] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019.

[15] A. Yang et al., “Qwen2.5 technical report,” arXiv:2412.15115, 2024.

[16] A. Grattafiori et al., “The Llama 3 herd of models,” arXiv:2407.21783, 2024.

[17] Y. Chen et al., “VoiceBench: Benchmarking LLM-based voice assistants,” TACL, 2026.

[18] D. Jain et al., “VoiceAgentBench: Are voice assistants ready for agentic tasks?” arXiv:2510.07978v3, 2026.

[19] Y. Qin et al., “ToolLLM: Facilitating large language models to master 16000+ real-world APIs,” in Proc. ICLR, 2024.

[20] W. Liu et al., “ToolACE: Winning the points of LLM function calling,” in Proc. ICLR, 2025.

[21] X. Liu et al., “AgentBench: Evaluating LLMs as agents,” in Proc. ICLR, 2024.

[22] S. G. Patil et al., “The Berkeley Function Calling Leaderboard (BFCL): From tool use to agentic evaluation of large language models,” in Proc. ICML, 2025.

[23] S. Yao et al., “τ -bench: A benchmark for tool-agent-user interaction in real-world domains,” in Proc. ICLR, 2025.

[24] V. Barres et al., “τ<sup>2</sup>-Bench: Evaluating conversational agents in a dual-control environment,” arXiv:2506.07982, 2025.

[25] H. Mao et al., “BFCL Audio: An audio function calling evaluation for large language models,” in Proc. ICML, 2026.

[26] T. Bogavelli et al., “EVA-Bench: A new end-to-end framework for evaluating voice agents,” arXiv:2605.13841, 2026.

[27] M. T. R. Laskar et al., “From Text to Voice: A reproducible and verifiable framework for evaluating tool calling LLM agents,” arXiv:2605.15104, 2026.

[28] S. Ray et al., “τ -Voice: Benchmarking full-duplex voice agents on real-world domains,” arXiv:2603.13686, 2026.