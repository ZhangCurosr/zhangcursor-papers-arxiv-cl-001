# Honeyed Words, Hidden Daggers: Benchmarking and Training LLMs to See Through Mock Politeness

Caixin Kang<sup>1,2,\*</sup>, Sitong Gong<sup>2,\*</sup>, Tianyu Yan<sup>2,\*</sup>, Kaipeng Zhang<sup>2</sup>, Yoichi Sato<sup>1</sup>, Yifei Huang<sup>1,2,†</sup>

<sup>1</sup>The University of Tokyo <sup>2</sup>Alaya Lab

<sup>\*</sup>Equal contribution <sup>†</sup>Corresponding author

The Chinese idiom 口蜜腹剑 (“honeyed words, hidden daggers”) names speech that is polite in form but hostile in function. We present ZhenhuanBench, a 1,592-item benchmark built from expert commentary on Empresses in the Palace, covering speaker intent, emotion–language mismatch, and per-listener uptake. Text-only evaluation covers 34 LLMs. GPT-6 Astra leads the leaderboard at 73.48% overall, while its listener-reaction score is 58.37%. The annotations also supply supervised traces and string-verifiable training rewards. With 338 gold traces, an 8B model outperforms teacher distillation on identical items. Subsequent RL saturates, and chance correction reduces the estimated recoverable performance gap by 7.5×. Our CPR-GRPO loop (Compress–Probe–Replenish) uses this estimate to weight gold supervision, reaching 67.0 on public multiple-choice tasks and 74.5 under the training-matched hidden protocol. Without target-task training, CPR outperforms Base, SFT, and GRPO across six Chinese and English pragmatic and sarcasm evaluations. On three external datasets, training with CPR yields higher mean Macro-F1 than SFT and GRPO.

Project page: https://ut-vision-social.github.io/ Email: cxkang@iis.u-tokyo.ac.jp, hyf015@gmail.com Date: October 12, 2026

![](images/79b2bc41d895eca55379615fc44b292035a9af124e7c98ed94834d4f22ee846e.jpg)

## 1 Introduction

In Empresses in the Palace (甄嬛传), a consort congratulates a newly pregnant rival with a single warm line, 「到底是富察姐姐有福气，三五日 间就有喜了。」 <sub>(“Sister</sub> <sub>Fucha</sub> <sub>is</sub> <sub>truly</sub> <sub>blessed,</sub> with child in just a handful of days.”). The words are impeccably kind, yet everyone in the hall knows the Emperor has scarcely visited her, so the congratulation quietly asks how the child could be his. The Empress catches the barb and cannot smile; a less astute consort takes it as sincere praise. This is mock politeness, a pragmatic register in which politeness is the weapon rather than the message (Culpeper, 1996; Haugh, 2015). Reading it requires recovering the speaker’s true intent, noticing that emotion and linguistic surface point in opposite directions, and tracking which listeners decode the attack and which take it at face value (Bell, 1984). Figure 1 shows how a single expert-anchored annotation captures all three layers, and how the same structure later becomes training signal.

We study whether LLMs understand mock politeness, whether this ability can be trained into a small model, and how the training signal contributes to its performance. Existing resources provide limited scale and annotation detail for these questions. The largest prior Chinese mockpoliteness evaluation contains 100 items with three-way labels and no theory-of-mind dimension (Zhang et al., 2026), and the classic Chinese implicature benchmark is nearly saturated by strong models (Yue et al., 2024).

Our answer to the first question is Zhenhuan-Bench, a 1,592-item benchmark whose gold labels derive from 47 hours of expert commentary aligned to the original screenplay. Each aligned utterance carries the speaker’s real intent, an emotion– language mismatch label, and per-listener uptake judgments, and from one annotation we derive four tasks probing intent, second-order theory of mind, multi-listener uptake, and Culpeper-style politeness classification. Text-only evaluation spans 34 LLMs. GPT-6 Astra leads the leaderboard with 73.48 overall and 58.37 on listener reactions; Gemini 3.7 Flash leads the MCQ tasks with 78.97 (§4). Providing expert commentary raises Gemma-4-31B’s hidden-set accuracy from 65.7 to 85.2.

Our training study asks how to use these expert interpretations to train mock-politeness understanding. Annotation-Grounded Verifiable Rewards (AGVR) turn the annotated relations into supervision for an 8B model. With 338 traces and LoRA, an 8B model rises from 36.8 to 55.7 on the leaderboard’s MCQ tasks under supervision plus one GRPO round, above GPT-5, and gold traces beat chain-of-thought distilled from Kimi-

![](images/e967c35b18fc57f8453be83585a0553546312b1f0db34e320b8ff96015f080d7.jpg)  
Figure 1: Overview of ZhenhuanBench and CPR-GRPO. (1) 47 hours of expert commentary become 1,506 structured annotations and four tasks, with 81.4 percent of candidate items rejected by quality gates. (2) A sample scene with multi-listener context translated into derived tasks T1–T4. (3) Training that reuses annotations as rewards and traces through Compress, Probe, and Replenish iterations (§5.4).

K2.6 on identical items. Diagnosing why further RL saturates, we find the gap between pass@k and greedy decoding shrinks 7.5× after correction for multiple-choice chance; the set of recoverable items decreases with each GRPO round. This diagnosis yields CPR-GRPO, a Compress–Probe– Replenish loop that directs further supervision from existing gold traces toward recoverable errors. The 8B model reaches 67.0 on the leaderboard's MCQ tasks, and 74.5 under the training-matched hidden protocol. The trained model also improves over Base on Chinese and English sarcasm tests and retains its aggregate accuracy under character renaming. In summary, our contributions are as follows:

• We construct ZhenhuanBench (§3), the largest theory-grounded Chinese mock-politeness benchmark, whose expert-anchored annotations each yield four tasks over one scene, carry graded multi-source provenance, and double as training signal.

• We develop CPR-GRPO (§5.1, §5.4) to train mock-politeness understanding from limited expert annotations, combining judge-free AGVR rewards with chance-corrected selection and weighting of existing gold traces. With 338 traces, the 8B model reaches 67.0 on the leaderboard’s MCQ tasks.

• We present extensive experiments and findings (§4, §5). Text-only evaluation covers 34 LLMs, including controlled reasoning comparisons. Training controls show that gold annotation outperforms teacher distillation and interface-only SFT enables subsequent GRPO gains; chance correction reduces the estimated recoverable gap by 7.5×.

## 2 Related Work

Chinese pragmatics benchmarks. Swordsman-Imp evaluates sitcom implicature and is nearly saturated (Yue et al., 2024), the Mask of Civility targets mock politeness directly but is small, coarsegrained, and unreleased (Zhang et al., 2026), and broader suites such as PUB lack the multi-listener dimension (Sravanthi et al., 2024). LLM pragmatic competence is uneven and tuning-sensitive (Hu et al., 2023; Ruis et al., 2023), and computational politeness predicts perceived politeness from surface features (Danescu-Niculescu-Mizil et al., 2013) rather than asking whether a polite surface hides hostile intent. ZhenhuanBench is an order of magnitude larger than the closest mock-politeness resource, adds per-listener uptake and mismatch structure, and is designed to be trainable (Table A.1, Appendix A).

Theory of mind and social intelligence. Machine ToM is usually probed with synthetic falsebelief vignettes (Le et al., 2019; Kim et al., 2023; Gandhi et al., 2023), on which strong models remain brittle (Sap et al., 2022), while broader benchmarks test social commonsense or interactive episodes (Sap et al., 2019; Zhou et al., 2024) and multimodal work reads deception, first impressions, and multi-speaker structure from what is visible and audible (Kang et al., 2025, 2026b; Ouyang et al., 2025). Related systems span interactive digital humans (Cai et al., 2025), social video generation (Ouyang et al., 2026), narrative agents (Huang et al., 2025), and proactive assistance with temporal memory (Gong et al., 2026a,b). ZhenhuanBench instead isolates the case where the signal is the language itself, with second-order questions from natural dialogue rather than constructed vignettes, several co-present listeners, and expert-documented rather than stipulated gold.

Distillation and RLVR. Rationale-based training uses human explanations (Camburu et al., 2018) and model-generated reasoning (Hsieh et al., 2023; Zelikman et al., 2022); our traces are instead programmatic rewrites of human-anchored annotation (§5.2). RLHF trains models from human preferences (Ouyang et al., 2022), while RLVR methods such as GRPO use verifiable outcomes, chiefly in mathematics and code (Shao et al., 2024; DeepSeek-AI et al., 2025), with recent work on social coordination (Kang et al., 2026a); AGVR uses exact matches against annotation fields to score pragmatic judgments. Yue et al. (2025) show that RLVR sharpens rather than expands ability, which we replicate and refine with chance correction. Our GRPO adopts DAPO’s zero-variance filtering and asymmetric clip (Yu et al., 2025). Where variance curricula reweight by raw variance (Jiang et al., 2025), off-policy guidance mixes gold into rollout groups (Yan et al., 2025), and self-distillation retrains on model outputs (Gulcehre et al., 2023), our loop assigns additional gold supervision to examples in a chance-corrected recoverable band. Chance correction is a classical psychometric tool (Cohen, 1960); to our knowledge, no prior work has used it to gate RL training.

## 3 ZhenhuanBench

## 3.1 Expert Commentary to Structured Gold

We construct gold labels by extracting and auditing interpretations from expert commentary, with corroborating evidence retrieved from an independent English recap podcast and the source novel. Our primary source is the first 110 parts (47 hours of audio) of an ongoing video commentary series on Empresses in the Palace by a dedicated analyst (Shulan Xiaojiejie, 2025), whose close readings of individual lines provide exactly the intent-level analysis the task needs (source selection and reception in Appendix A). We transcribe the commentary with Whisper (Radford et al., 2023), align commentary passages to the 18,297-utterance screenplay (1,214 scenes), and extract 1,506 alignments, each pairing an utterance with the expert’s reading.

Cross-source corroboration uses two additional textual sources. Chasing Dramas, an independent English recap podcast (Karen and Cathy, 2019), contributes 68 episode transcripts (about 249k words), and the source novel by Liu Lianzi (Liu Lianzi, 2007) contributes 1.26M characters including inner monologue. We index both sources with BGE-M3 (Chen et al., 2024) and retrieve passages for each alignment to compare the commentary’s interpretation with the podcast’s account and the novel’s narration.

An LLM extractor converts each alignment into four structured fields with verbatim quote anchoring, and a separate audit pass checks fidelity to the commentary (Appendix A). real\_intent states what the speaker is actually doing; emotion\_language\_mismatch labels affect against surface as aligned, partial\_mismatch, or inverted, operationalizing implicational mismatch (Haugh, 2015); third\_party\_reactions lists each listener present with an uptake judgment (yes, no, partial) and a one-sentence reason, operationalizing audience design (Bell, 1984); and rhetoric\_sub\_type supports Culpeper-style classification (Culpeper, 1996).

## 3.2 Tasks

From each alignment we derive up to four items. T1 asks the speaker’s true intent (4-way MCQ), the implicature-recovery problem of Grice (1975). T2 asks what a named listener privately concludes (4- way MCQ), a second-order theory-of-mind judgment (Premack and Woodruff, 1978; Perner and Wimmer, 1985). T3 asks for free-text descriptions of every listener’s reaction (scored by an LLM judge at evaluation only), operationalizing audience design (Bell, 1984). T4 asks for a threeway politeness classification, genuine politeness, mock politeness, or overt impoliteness, the facebased contrast of politeness theory (Brown and Levinson, 1987) operationalized through Culpeper (1996). Deriving multiple tasks from one annotation enables cross-task consistency and peralignment diagnostics unavailable to independentitem benchmarks (§4.4). Item construction enforces an anti-leakage checklist, with neutral phrasing, scene truncation before consequences unfold, length-balanced options, and three option permutations per MCQ (Appendix A).

![](images/5f07425344ecd8361c7883af6c981521a9d8b51aa360d0c8c5c0c631e14e6d97.jpg)  
Figure 2: ZhenhuanBench composition (1,592 items). Inner ring, the three splits (training and hidden episodedisjoint); outer ring, tasks T1–T4 shaded dark to light. Full counts in Appendix A.

## 3.3 Splits and Scale

ZhenhuanBench contains 1,592 items, split into 791 public, 319 hidden-test, and 482 training items with episode-level isolation between training and hidden episodes (Figure 2; per-task-by-split counts in Appendix A). Sampling ran in two symmetric passes over the full commentary to avoid coverage bias. T4 is the scarcest task (212 items) because it requires a confident Culpeper class from the commentary, while T3 contributes 429 free-text items that no MCQ-only protocol would cover. Training and hidden sets share no episode. All splits use the same drama and cast.

## 3.4 Quality

During construction, every MCQ passes an optionsonly guessing filter using Claude Haiku-4.5 or GPT-4o-mini: the probe attempts the item from the answer options alone, with the scene withheld, and any item it can solve without the scene is rewritten, removing answer-selection artifacts in the spirit of adversarial dataset filtering (Le Bras et al., 2020; Gururangan et al., 2018). Across two assembly passes the item writer proposed 8,544 candidates and the automated checks rejected 81.4 percent, leaving the 1,592 final items (full funnel in Table A.3).

Each item records its primary source, supporting quotations, and any cross-source corroboration. The gold label is anchored in the expert commentary for all 1,592 items, verbatim in 823 of them and as a supported paraphrase in the remaining 769, and 1,047 items (65.8 percent) additionally carry a quoted secondary passage that corroborates the same reading, 1,023 citing the English podcast and 28 the novel, with 4 items citing both. A gt\_strength field grades this support from 1 to 3 (79, 1,303, and 210 items), so the benchmark can be filtered by how independently each label is attested. On the public set, mean discrimination— the score difference between the upper and lower halves of a 24-model analysis subset ranked by overall score—is positive for all four tasks, from +0.16 to +0.24.

## 4 Evaluating Frontier Models

## 4.1 Protocol

Prompts provide the scene, cast, target utterance, and question (P\_context). MCQs use JSON answers, exact-match scoring, and majority voting over three option permutations. Qwen3.8-Max scores T3 reactions against the gold descriptions on a 0/1/2 scale, with temperature 0 and thinking disabled (Appendix D). Each T3 item averages its listener scores and divides by two; the overall score averages all 791 items. Controlled reasoning comparisons fix the model version and vary its reasoning setting (Appendix G.1).

## 4.2 Leaderboard

GPT-6 Astra leads Table 1 with 73.48 overall, followed by Gemini 3.7 Flash (72.28). Kimi K3 is the highest-scoring open-weight model at 66.76. DeepSeek V4 Pro scores highest on speaker intent (T1), while GPT-6 Astra leads on listener judgments (T2/T3). GPT-6 Astra and Gemini 3.7 Flash tie on politeness classification (T4), and Gemini also leads the MCQ tasks with 78.97. Predicting each listener’s reaction in free text remains difficult, with the best T3 score reaching only 58.37.

<table><tr><td>Model</td><td>All</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>Model</td><td>All</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td></tr><tr><td colspan="10">Open-weight models</td></tr><tr><td>Kimi K3</td><td>66.8</td><td>72.4</td><td>67.9</td><td>53.4</td><td>77.4</td><td>Qwen3-8B (base)</td><td>34.2</td><td>41.4</td><td>34.2</td><td>26.9</td><td>32.1</td></tr><tr><td>DeepSeek V4 Pro</td><td>62.5</td><td>76.2</td><td>58.8</td><td>49.4</td><td>66.0</td><td>GLM-4.6V-Flash</td><td>33.2</td><td>33.9</td><td>33.8</td><td>28.8</td><td>38.7</td></tr><tr><td>Kimi-K2.6</td><td>62.0</td><td>66.1</td><td>65.4</td><td>48.2</td><td>71.7</td><td>Qwen3.5-4B</td><td>32.9</td><td>38.5</td><td>35.8</td><td>15.8</td><td>47.2</td></tr><tr><td>Gemma-4-31B</td><td>59.6</td><td>64.4</td><td>62.9</td><td>42.0</td><td>75.5</td><td>InternVL3-8B</td><td>31.8</td><td>29.7</td><td>32.5</td><td>26.0</td><td>46.2</td></tr><tr><td>DeepSeek-V4-Flash</td><td>56.8</td><td>67.8</td><td>52.1</td><td>45.3</td><td>65.1</td><td>GLM-4.1V-9B</td><td>31.1</td><td>30.5</td><td>35.8</td><td>25.8</td><td>32.1</td></tr><tr><td>Qwen3-VL-235B</td><td>52.4</td><td>56.1</td><td>49.2</td><td>42.6</td><td>70.8</td><td>Qwen3-VL-4B</td><td>30.3</td><td>32.2</td><td>32.5</td><td>22.0</td><td>36.8</td></tr><tr><td>GLM-4.6V</td><td>47.4</td><td>53.1</td><td>46.2</td><td>37.4</td><td>56.6</td><td>GLM-4.5-Air</td><td>30.1</td><td>28.5</td><td>28.7</td><td>29.5</td><td>37.7</td></tr><tr><td>Qwen3-VL-30B</td><td>44.4</td><td>53.1</td><td>41.2</td><td>36.9</td><td>46.2</td><td>MiniCPM-o-2.6</td><td>29.4</td><td>28.0</td><td>31.2</td><td>21.7</td><td>43.4</td></tr><tr><td>Qwen3-32B</td><td>44.1</td><td>51.0</td><td>43.8</td><td>34.4</td><td>48.1</td><td>MiMo-VL-7B</td><td>28.3</td><td>25.9</td><td>29.2</td><td>28.7</td><td>31.1</td></tr><tr><td>Qwen3.5-9B</td><td>41.6</td><td>47.3</td><td>36.2</td><td>31.1</td><td>61.3</td><td>Qwen2.5-VL-7B</td><td>28.3</td><td>31.4</td><td>30.4</td><td>22.6</td><td>27.4</td></tr><tr><td>Qwen3-14B</td><td>39.1</td><td>46.9</td><td>34.2</td><td>30.9</td><td>49.1</td><td>Qwen3.5-2B</td><td>26.0</td><td>32.2</td><td>27.5</td><td>12.6</td><td>34.9</td></tr><tr><td>Qwen3-VL-8B</td><td>37.4</td><td>36.4</td><td>35.8</td><td>32.0</td><td>53.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Proprietary models</td></tr><tr><td>GPT-6 Astra</td><td>73.5</td><td>73.2</td><td>79.6</td><td>58.4</td><td>89.6</td><td>GPT-5</td><td>52.3</td><td>52.3 </td><td>51.7</td><td>49.5</td><td>59.4</td></tr><tr><td>Gemini 3.7 Flash</td><td>72.3</td><td>74.5</td><td>78.8</td><td>53.3</td><td>89.6</td><td>GPT-40</td><td>49.3</td><td>51.0</td><td>50.4</td><td>39.0</td><td>63.2</td></tr><tr><td>Qwen3.8-Max</td><td>62.5</td><td>62.8</td><td>64.2</td><td>49.8</td><td>83.0</td><td>Haiku-4.5</td><td>44.4</td><td>44.8</td><td>45.0</td><td>40.3</td><td>50.0</td></tr><tr><td>Sonnet-4.6</td><td>58.9</td><td>59.4</td><td>60.4</td><td>50.6</td><td>70.8</td><td>GPT-5-mini</td><td>38.1</td><td>35.1</td><td>37.5</td><td>35.7</td><td>50.9</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>58.6</td><td>66.1</td><td>57.1</td><td>45.8</td><td>69.8</td><td>GPT-4o-mini</td><td>30.8</td><td>26.8</td><td>29.6</td><td>28.3</td><td>47.2</td></tr><tr><td>CPR-GRPO (Qwen3-8B)</td><td>56.7</td><td>66.1</td><td>67.5</td><td>27.4</td><td>67.9</td><td>Trained reference; MCQ accuracy: 67.0</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Public-set leaderboard (percent): 33 models and a trained 8B reference. Qwen3.8-Max judges the models free-text T3 responses. All is the mean score over all 791 items. Within each group, models rank by All down the left block, then the right. Best scores across all models are bold.

Using Qwen3.8-Max and Claude Sonnet-4.6 as alternative T3 judges yields highly similar model rankings across 34 configurations, with Spearman correlations of 0.985 for T3 and 0.999 for overall scores. Appendix F.1 reports detailed cross-judge results.

Training gains. CPR raises Qwen3-8B’s MCQ accuracy from 36.75 to 67.01 and its overall score from 34.20 to 56.69. Its T3 score is close to Base (27.39 versus 26.94). The training analysis in §5 examines how annotation traces and RL contribute to the MCQ gains.

Reasoning budgets. Increasing reasoning effort does not consistently improve mock-politeness understanding. DeepSeek V4 Pro improves from 58.36 overall with thinking off to 68.44 at max effort. Qwen3.8-Max performs best with thinking off (63.43), while Qwen3.8-27B peaks at low effort (54.64). The benefits of additional reasoning are model-dependent rather than uniformly positive (Appendix G.1).

Character-name sensitivity. On 508 qualitychecked items, two consistent renamings change CPR’s mean accuracy from 63.39 to 64.07, versus 36.22 to 32.48 for Base and 67.91 to 58.17 for Qwen3.8-Max. The intervention changes names and aliases while preserving dialogue and role relations (Appendix G.4). CPR retains its accuracy under character renaming, whereas Base and Qwen3.8-Max are more sensitive to charactername cues.

## 4.3 Text Ablations and Oracle Commentary

On the hidden set we vary the prompt for three models (base 8B, our trained model, Gemma-4-31B). Knowledge-enhanced prompting, which injects the textbook definition of mock politeness and helped in prior work (Zhang et al., 2026), hurts all three models here, by −11.5, −15.8, and −4.6 points respectively (Table B.1), with verbatim prompts in Appendix B. Removing the cast block barely matters. Adding full scene context raises accuracy by 3.7 points for Gemma and 6.5 points for CPR, while lowering Base accuracy by 3.3 points. Appending a short excerpt of the expert’s commentary (an oracle hint) lifts Gemma-4-31B to 85.2, the base model to 60.6, and our model to 75.5. The commentary supplies the expert’s interpretation of the target utterance, including its implied intent and listeners’ understanding. Subtitle-cropped frames yield no significant gain over mismatched frames; audio helps MiniCPM-o-2.6 (+3.1) and leaves Gemini-2.5-Pro unchanged (Appendix E).

<table><tr><td>Model</td><td></td><td>Consist. Per-align. mean None right</td><td></td></tr><tr><td>Kimi-K2.6</td><td>42.9</td><td>65.5</td><td>13.5</td></tr><tr><td>Gemma-4-31B</td><td>39.7</td><td>63.7</td><td>14.1</td></tr><tr><td>Sonnet-4.6</td><td>36.5</td><td>60.4</td><td>16.7</td></tr><tr><td>GPT-5</td><td>27.6</td><td>53.2</td><td>21.2</td></tr><tr><td>Haiku-4.5</td><td>19.9</td><td>42.2</td><td>34.6</td></tr><tr><td>Qwen3-8B (base)</td><td>12.2</td><td>35.4</td><td>41.0</td></tr><tr><td>CPR-GRPO (ours)</td><td>37.2</td><td>64.1</td><td>10.9</td></tr></table>

Table 2: Schema-enabled diagnostics over 156 alignments with at least two MCQ items (T1/T2/T4; percent). “Consist.” counts alignments with all sibling MCQs correct, “None right” with all wrong. The shaded row is our trained 8B model (§5).

## 4.4 Schema-Enabled Diagnostics

Because multiple tasks share an alignment, we can measure whether a model understands a scene coherently. Cross-task consistency (all sibling MCQs correct) separates models far more sharply than accuracy (Table 2). For Kimi-K2.6 versus Base 8B, MCQ accuracy differs by 1.8× (66.8/36.8), compared with 3.5× for consistency (42.9/12.2); GPT-5 and Haiku-4.5 score 27.6 and 19.9. Coherent scene-level understanding is rarer than per-item accuracy suggests. The trained model raises consistency from 12.2 to 37.2 (above Sonnet-4.6) and reduces the none-right rate from 41.0 to 10.9, the lowest value in Table 2. Mismatch-confusion analysis appears in Appendix F.1.

## 5 Training Study

Understanding mock politeness requires relating polite wording to speaker intent and listener uptake in context. Our annotations record these relations as intent, emotion–language mismatch, and listener reactions, providing supervision for training. As RL gains saturate, CPR-GRPO uses a chance-corrected recovery estimate to select and weight existing gold traces for further supervision.

Unless noted otherwise, training uses Qwen3- 8B with LoRA (Hu et al., 2022); Appendix F.4 replicates the supervised stage, one GRPO round, and one full loop iteration on further backbones. Unless noted, we evaluate on the MCQ items of the held-out hidden test, which shares no episode with training, using paired bootstrap tests. The public-set results use the MCQ protocol of Table 1, reported as the micro average over its MCQ tasks (T1, T2, T4); the overall column of Table 1 additionally includes judge-scored T3.

![](images/b712af2878ee3cfbca7729eddfff2f3494b83675d792b1c4d6007203de3dab44.jpg)  
Figure 3: Leaderboard-protocol MCQ accuracy (public set). Training raises 8B accuracy from 36.8 to 67.0; the shaded band spans GPT-5 to Kimi-K2.6.

## 5.1 AGVR: Annotation-Grounded Verifiable Rewards

For SFT, AGVR converts each aligned annotation into a five-step trace: evidence, mismatch, per-listener uptake, intent, and a JSON answer. Each step is instantiated programmatically from gold fields. For RL, the same fields define stringverifiable rewards (answer, mismatch, and uptake matches plus a −0.5 format gate). Training rewards are computed directly from the gold annotation fields. Supervision lifts the 8B model from 36.8 to 52.3 on this metric and one GRPO round on the same annotations to 55.7, above GPT-5 (53.3); and on the 253 items the teacher Kimi-K2.6 could itself solve, gold traces beat its rejectionsampled chain-of-thought 55.6 to 49.5. Auxiliary reward heads reduce variability across seeds (Appendix F.2). SFT accuracy increases with goldtrace count (Figure F.1, Appendix F.2).

## 5.2 Effects of Output Format and Supervised Content

GRPO from the base model changes accuracy by −0.5 points (p=0.9). A two-by-two study (Table F.2, Appendix F.2) separates the interface (the five-step-plus-JSON output format that the reward function parses and scores) from the content (gold reasoning). Interface-only SFT replaces reasoning spans with placeholders and reaches 56.9/55.6; subsequent GRPO adds +3.3/+3.2. Gold reasoning supplied as unparseable prose scores 1.9/3.7, with 95 percent parse failures and 51 percent zerovariance rollout groups. Interface-only SFT thus enables GRPO gains in both tested seeds. A shuffledcontent control gains +2.8, compared with +25.5 for the full recipe. Answer-plus-intent supervision without the five-step structure scores 59.3 versus 62.0 (Table 3). These controls identify contributions from correct answer supervision and a parseable reward interface.

<table><tr><td>Cold-start variant</td><td>Score Paired</td><td> $\Delta$ </td></tr><tr><td>Gold scaffold (full recipe)</td><td>62.0</td><td>+25.5</td></tr><tr><td>Answer + intent, no structure</td><td>59.3</td><td> $+ 2 2 . 7$ </td></tr><tr><td>Gold, matched 253 items</td><td>55.6</td><td> $+ 1 8 . 1$ </td></tr><tr><td>Teacher CoT, same 253 items</td><td>49.5</td><td> $+ 1 0 . 6$ </td></tr><tr><td>Shuffled content</td><td>40.8</td><td> $+ 2 . 8$ </td></tr></table>

Table 3: SFT-stage attribution on the hidden set (percent; paired $\Delta$ against the base of the same evaluation session). Content drives the gain; gold beats teacher CoT on identical items.

![](images/4354a7160dab4e86b265e92d3b0d82ff9cc09d933c5b3904eda5ec9781415ce5.jpg)  
Figure 4: Pure GRPO plateaus near 70 on the hidden set (shaded band); adding Replenish (orange, two seeds) reaches 74.5 at round 4.

## 5.3 Chance-Corrected Analysis of GRPO

GRPO after SFT adds +4 to +5 points to greedy accuracy. pass@8 stays flat (93.1 to 93.5) while greedy and majority vote rise, replicating the sharpening view of RLVR (Yue et al., 2025). The pass@16-minus-greedy gap is 31.4 points; random guessing already gives pass@16 above 99 percent on 4-way items. Chance-corrected, the recoverable excess is only 4.2 points (RL) and 4.8 (SFT), a 7.5× inflation that declines monotonically down the pipeline (5.7, 4.8, 4.2) even as greedy accuracy climbs thirty points. The recoverable band decreases during training. Grouping greedywrong items by chance-corrected recovery shows that fewer items remain recoverable as GRPO gains diminish; the round-3 increment is +0.9.

## 5.4 CPR-GRPO: Compress, Probe, Replenish

CPR-GRPO uses the recoverable band to allocate further supervision from the expert-grounded gold traces through three steps (Algorithm 1 and Figure C.1, Appendix C). Compress runs one tokenlevel GRPO round (asymmetric clipping 0.2/0.28, zero-variance filtering, k3 KL to the round-start policy; any RLVR optimizer fits this slot). Probe re-samples the training set 16 times per item and buckets by chance-corrected recovery. Replenish oversamples gold traces of sub-modal and strongrecovery items (4× and 2×) for one epoch of hard cross-entropy.

![](images/d64c79c1d32b7b29ad23d51ff09e0bbac5af4293ac2118c62b8453a3eda5f14e.jpg)  
Figure 5: After Replenish, correct-answer rates are highest among recoverable errors (orange) and lower among low-recovery errors (gray); most previously correct answers are retained (blue).

Reward, objective, and probe. Formally, let $\mathcal { D } =$ $\{ x _ { i } \}$ be the training items, each with $k _ { i }$ options and gold fields $( a _ { i } ^ { * } , m _ { i } ^ { * } , u _ { i } ^ { * } )$ for answer, mismatch label, and per-listener uptake, and let $\mathcal { G } = \{ \tau _ { i } \}$ be the programmatic gold traces of §5.1. The verifiable reward of a rollout $y$ is

$$
\begin{array} { r } { r ( y \mid x _ { i } ) = 1 [ \operatorname { a n s } ( y ) = a _ { i } ^ { * } ] + \lambda 1 [ \operatorname* { m i s } ( y ) = m _ { i } ^ { * } ] } \\ { + \lambda \mathrm { F } _ { \mathrm { a u d } } ( y , u _ { i } ^ { * } ) - \frac { 1 } { 2 } \mathbb { 1 } [ \operatorname* { m a l f } ( y ) ] , } \end{array}\tag{1}
$$

with $\lambda { = } 0 . 3 , \mathrm { F _ { a u d } }$ the fraction of listeners whose uptake tag matches gold, and malf(·) flagging malformed outputs. Within a group of $G$ rollouts the advantage is $\hat { A } _ { g } = ( r _ { g } - \mu _ { i } ) / ( \sigma _ { i } + \epsilon )$ zero-variance groups are dropped, and the Compress step maximizes a token-level clipped objective with asymmetric clipping and a k3 KL anchor to the round-start policy (Eq. C.1, Appendix C). The Probe step estimates, for every item, the empirical correct rate $\hat { p } _ { i }$ over $G ^ { \prime }$ =16 fresh samples and the greedy correctness $g _ { i }$ , and converts them into a chance-corrected recovery excess and a replenish-

ment weight

$$
\begin{array} { r } { \begin{array} { r } { e _ { i } = \operatorname* { m a x } \bigl ( 0 , ~ \hat { p } _ { i } - \frac { 1 } { k _ { i } } \bigr ) , } \\ { w _ { i } = \left\{ \begin{array} { l l } { 4 } & { g _ { i } = 0 , ~ e _ { i } > \delta , ~ \hat { p } _ { i } \leq \frac { 1 } { 2 } ~ ( \mathrm { s u b - m o d a l } ) } \\ { 2 } & { g _ { i } = 0 , ~ \hat { p } _ { i } > \frac { 1 } { 2 } ~ ( \mathrm { s t r o n g ~ r e c o v e r y } ) } \\ { 1 } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array} } \end{array}\tag{2}
$$

with margin $\delta { = } 0 . 0 6$ . Items with $e _ { i } \approx 0$ get no extra budget, separating the probe from classical hard-example mining. Appendix F.3 reports sensitivity to the weights and margin. The Replenish step takes one epoch of weighted cross-entropy on gold traces, $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = - \sum _ { i } w _ { i } \log \pi _ { \boldsymbol { \theta } } ( \tau _ { i } \mid x _ { i } ) } \end{array}$ with no KL anchor to overwrite confidently wrong modes.

Gains from replenishment. Pure GRPO plateaus near 70 (rounds 2 to 5 give 66.7, 67.6, 70.4, 69.4), whereas Replenish reaches 68.5/68.5 at round 2 and 71.8/74.5 at round 4 (Figure 4). Repeated evaluations give 74.5–77.3 before and 75.5–75.9 after another Compress step (Appendix F.2). Figure 5 reports changes within the probe-defined groups. Replenish flips 61 and 80 percent of the sub-modal and strong-recovery items against 10 percent of the unrecoverable bucket, keeps 85 percent of previously correct ones, and re-probing identifies 81 recoverable items at round 4, compared with 146 at round 2. The same targeting reproduces on two further backbones (Appendix F.4). Further Compress gains have paired intervals spanning zero. Replenish lowers scores from 75.9 to 70.4 and 69.4 to 68.1, with 1 and 8 sub-modal items remaining (Appendix F.3). Under the leaderboard protocol the best checkpoint scores 67.0 on the MCQ tasks (Figure 3), above Sonnet-4.6 (61.9) and level with Kimi-K2.6 (66.8).

## 5.5 Training Controls

Table F.3 (Appendix F.3) reports the training controls. Anchoring the same gold traces inside RL groups as off-policy members, in the spirit of LUFFY (Yan et al., 2025), yields 65.3, compared with 66.7 for GRPO. Pure GRPO scores 69.4 at round 5. Rejection sampling adds +1.0, DPO at $\beta { = } 0 . 0 5$ reduces accuracy by 8.3 points, and five SFT epochs score 61.6 versus 62.0 after three epochs. CPR-GRPO achieves the highest accuracy among these training strategies.

## 5.6 Transfer to External Tasks

We test both the fixed ZhenhuanBench checkpoints and the training recipe on external data.

Fixed-checkpoint transfer. Without target-data training, CPR exceeds Base, SFT, and GRPO in all six evaluations in Table 4. CPR reaches 47.73 Macro-F1 on ToSarcasm and 86.00 accuracy on iSarcasmEval C, versus 42.79 and 82.00 for Base. Appendix G.2 details the datasets and evaluation protocols.

<table><tr><td>Dataset</td><td>Base</td><td>SFT</td><td>GRPO</td><td>CPR</td></tr><tr><td>MultiPragEval (Zh)</td><td>65.78</td><td>64.00</td><td>65.78</td><td>66.33</td></tr><tr><td>MultiPragEval (En)</td><td>65.56</td><td>65.89</td><td>65.78</td><td>67.11</td></tr><tr><td>ToSarcasm</td><td>42.79</td><td>46.68</td><td>45.93</td><td>47.73</td></tr><tr><td>without topic</td><td>45.90</td><td>51.34</td><td>50.84</td><td>51.89</td></tr><tr><td>iSarcasmEval A</td><td>28.07</td><td>28.96</td><td>30.88</td><td>31.40</td></tr><tr><td>iSarcasmEval C</td><td>82.00</td><td>83.50</td><td>85.50</td><td>86.00</td></tr></table>

Table 4: Transfer without target-data training. Scores are accuracy, except ToSarcasm (Macro-F1) and iSarcasmEval A (sarcastic-class F1). Details in Appendix G.2.

Retraining on external datasets. We independently train Qwen3-8B on FGVIrony, CPED, and SemEval-2018 Task 3B with three seeds. Table 5 compares Base, SFT, GRPO, CPR, and wrong-only replenishment. CPR exceeds GRPO by 2.33, 3.03, 7.50, and 4.10 Macro-F1 points across the four settings, and also exceeds SFT and wrong-only replenishment. Appendix G.3 gives the data splits and training settings.

<table><tr><td>Method</td><td>FGV</td><td>CPED-B</td><td>CPED-N</td><td>SemEval</td></tr><tr><td>Base</td><td>17.77</td><td>23.97</td><td>23.97</td><td>19.05</td></tr><tr><td>SFT</td><td>32.09</td><td>28.75</td><td>31.61</td><td>38.80</td></tr><tr><td>GRPO</td><td>38.19</td><td>29.07</td><td>32.48</td><td>43.16</td></tr><tr><td>Wrong-only</td><td>32.59</td><td>28.19</td><td>37.36</td><td>41.17</td></tr><tr><td>CPR</td><td>40.52</td><td>32.10</td><td>39.98</td><td>47.26</td></tr></table>

Table 5: External retraining: mean Macro-F1 over three seeds for trained methods; Base uses one fixed checkpoint. B/N: balanced/natural CPED sampling. Details in Appendix G.3.

## 6 Conclusion

ZhenhuanBench evaluates mock politeness through speaker intent, emotion–language mismatch, and listener uptake. GPT-6 Astra leads the leaderboard with 73.48 overall, while listener-reaction prediction remains difficult across models. Gold annotation traces outperform teacher reasoning on matched items, and interface-only supervision enables subsequent GRPO gains. Chance correction reduces the estimated recoverable performance gap by 7.5×; CPR-GRPO uses this estimate to weight gold supervision and raises an 8B model’s public MCQ accuracy from 36.8 to 67.0. Fixedcheckpoint evaluation shows gains on Chinese and English sarcasm tasks, and character renaming preserves CPR’s aggregate accuracy. Larger reasoning budgets improve DeepSeek’s performance; Qwen models peak at off or low effort. Together, these findings show how expert pragmatic interpretations can support both the evaluation and training of models to understand mock politeness.

## Limitations

The benchmark draws its scenes from a single Chinese drama. Training and hidden splits use different episodes but share characters and setting. Overlap with model pretraining data is unknown. T3 is scored by an LLM judge. Its absolute scores vary across judges.

## Ethics Statement

ZhenhuanBench draws on publicly accessible, copyrighted creative works: a television screenplay, fan commentary, a recap podcast, and a novel, cited in §3.1. The release contains structured annotations, short evidence quotations, and episode and scene references; it excludes the original video and audio. The commentary author’s account is cited, and the reception statistics in Appendix A come from public profile pages. The dialogue and characters are fictional. Items contain indirect verbal aggression characteristic of mock politeness, with no slurs or graphic material. Potential misuse includes assistance with manipulative writing. Our experiments train and evaluate comprehension.

## References

Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, and Walid Magdy. 2022. SemEval-2022 task 6: iSarcasmEval, intended sarcasm detection in English and Arabic. In Proceedings ofthe 16th International Workshop on Semantic Evaluation (SemEval-2022), pages 802–814.

Anthropic. 2025. Claude Haiku 4.5. https: //www.anthropic.com/news/claude-haiku-4-5. Claude Haiku 4.5; accessed 2026-05-04.

Anthropic. 2026. Claude Sonnet 4.6. https:// www.anthropic.com/news/claude-sonnet-4-6. Claude Sonnet 4.6; accessed 2026-05-04.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025b. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Allan Bell. 1984. Language style as audience design. Language in society, 13(2):145–204.

Penelope Brown and Stephen C Levinson. 1987. Politeness: Some universals in language usage, volume 4. Cambridge university press.

Yiyi Cai, Xuangeng Chu, Xiwei Gao, Sitong Gong, Yifei Huang, Caixin Kang, Kunhang Li, Haiyang Liu, Ruicong Liu, Yun Liu, Dianwen Ng, Zixiong Su, Erwin Wu, Yuhan Wu, Dingkun Yan, Tianyu Yan, Chang Zeng, Bo Zheng, and You Zhou. 2025. Towards interactive intelligence for digital humans. arXiv preprint arXiv:2512.13674.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems, 31.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3-embedding: Multilinguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Yirong Chen, Weiquan Fan, Xiaofen Xing, Jianxin Pang, Minlie Huang, Wenjing Han, Qianfeng Tie, and Xiangmin Xu. 2022. CPED: A large-scale Chinese personalized and emotional dialogue dataset for conversational AI. arXiv preprint arXiv:2205.14727.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1):37–46.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Jonathan Culpeper. 1996. Towards an anatomy of impoliteness. Journal of pragmatics, 25(3):349–367.

Cristian Danescu-Niculescu-Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, and Christopher Potts. 2013. A computational approach to politeness with application to social factors. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 250–259.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437.

Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah Goodman. 2023. Understanding social reasoning in language models with language models. Advances in Neural Information Processing Systems, 36:13518–13529.

GLM-V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. 2025. GLM-4.5V and GLM-4.1V-Thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006.

Sitong Gong, Caixin Kang, Tianyu Yan, Guo Chen, Bo Zheng, Kaipeng Zhang, Yunzhi Zhuge, Xiang Ruan, Huchuan Lu, and Yifei Huang. 2026a. GROVE: Growing and reasoning over temporally stratified memory from streaming video experience. arXiv preprint arXiv:2608.02392.

Sitong Gong, Tianyu Yan, Caixin Kang, Bo Zheng, Xiang Ruan, Huchuan Lu, Kaipeng Zhang, Yoichi Sato, and Yifei Huang. 2026b. Vinci2: Providing proactive assistance in continuous egocentric videos. arXiv preprint arXiv:2607.11523.

Google DeepMind. 2026a. Gemini 3.1 pro. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/ gemini-3-1-pro/. Gemini 3.1 family; accessed 2026-05-04.

Google DeepMind. 2026b. Gemma 4. https://blog.google/innovation-and-ai/ technology/developers-tools/gemma-4/. Gemma-4-31B-it; accessed 2026-05-04.

Herbert P Grice. 1975. Logic and conversation. In Speech acts, pages 41–58. Brill.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. 2023. Reinforced selftraining (rest) for language modeling. arXiv preprint arXiv:2308.08998.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R Bowman, and Noah A Smith. 2018. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 107–112.

Michael Haugh. 2015. Im/politeness implicatures, volume 11 of Mouton Series in Pragmatics. De Gruyter Mouton.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Jennifer Hu, Sammy Floyd, Olessia Jouravlev, Evelina Fedorenko, and Edward Gibson. 2023. A finegrained comparison of pragmatic language understanding in humans and language models. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 4194–4213.

Yifei Huang, Tianyu Yan, Sitong Gong, Xiwei Gao, Caixin Kang, Ruicong Liu, Huchuan Lu, and Bo Zheng. 2025. Living the novel: A system for generating self-training timeline-aware conversational agents from novels. arXiv preprint arXiv:2512.07474.

Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, and Hao Wang. 2025. Vcrl: Variance-based curriculum reinforcement learning for large language models. arXiv preprint arXiv:2509.19803.

Caixin Kang, Yifei Huang, Liangyang Ouyang, Mingfang Zhang, Ruicong Liu, and Yoichi Sato. 2025. Can MLLMs read the room? a multimodal benchmark for assessing deception in multi-party social interactions. arXiv preprint arXiv:2511.16221.

Caixin Kang, Ke Wang, Yuchuan Wu, Xiaoqiang Wang, Ling Li, Bingxue Qiu, Yongbin Li, and Yoichi Sato. 2026a. What makes a social agent’s interactions effective? learning from verifiable outcomes in paired worlds. arXiv preprint arXiv:2610.20195.

Caixin Kang, Tianyu Yan, Sitong Gong, Mingfang Zhang, Liangyang Ouyang, Ruicong Liu, Bo Zheng, Huchuan Lu, Kaipeng Zhang, Yoichi Sato, and Yifei Huang. 2026b. Perception or prejudice: Can MLLMs go beyond first impressions of personality? arXiv preprint arXiv:2605.22109.

Karen and Cathy. 2019. Chasing dramas. https: //www.chasingdramas.com. English-language podcast explaining cultural and historical references in Chinese historical dramas, 500+ episodes since August 2019; 68 episode transcripts for Empresses in the Palace used for corroboration. Accessed 2026-07-31.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Le Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14397–14413.

Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. 2025. Kimi K2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877.

Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish Sabharwal, and Yejin Choi. 2020. Adversarial filters of dataset biases. In International conference on machine learning, pages 1078–1088. Pmlr.

Bin Liang, Zijie Lin, Bing Qin, and Ruifeng Xu. 2022. Topic-oriented sarcasm detection: New task, new dataset and new method. In Proceedings ofthe 21st Chinese National Conference on Computational Linguistics, pages 557–568.

Liu Lianzi. 2007. Hou Gong: Zhen Huan Zhuan (Empresses in the Palace), seven volumes. Huashan Literature and Art Press. First serialized online; print publication from 2007 (vols. 1–3 Huashan Literature and Art Press, vols. 4–5 Guangxi Normal University Press, vols. 6–7 Chongqing Press). About 1.26M characters.

OpenAI. 2025. GPT-5. https://openai.com/index/ introducing-gpt-5/. Large language model.

OpenAI, Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. GPT-4o system card. arXiv preprint arXiv:2410.21276.

Liangyang Ouyang, Yifei Huang, Mingfang Zhang, Caixin Kang, Ryosuke Furuta, and Yoichi Sato. 2025. Multi-speaker attention alignment for multimodal social interaction. arXiv preprint arXiv:2511.17952.

Liangyang Ouyang, Ruicong Liu, Caixin Kang, Yifei Huang, and Yoichi Sato. 2026. SocialDirector: Training-free social interaction control for multi-person video generation. arXiv preprint arXiv:2605.10079.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.

Dojun Park, Jiwoo Lee, Hyeyun Jeong, Seohyun Park, Youngeun Koo, Soonha Hwang, Seonwoo Park, and Sungeun Lee. 2024. MultiPragEval: Multilingual pragmatic evaluation of large language models. In Proceedings of the 2nd GenBench Workshop on Generalisation (Benchmarking) in NLP, pages 96–119.

Josef Perner and Heinz Wimmer. 1985. “john thinks that mary thinks that. . . ” attribution of second-order beliefs by 5-to 10-year-old children. Journal of experimental child psychology, 39(3):437–471.

David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515–526.

Qwen Team. 2026. Qwen3.5. https://qwen.ai/ blog?id=qwen3.5. Large language model.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR.

Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. 2023. The goldilocks of pragmatic understanding: Finetuning strategy matters for implicature resolution by llms. Advances in Neural Information Processing Systems, 36:20827–20905.

Maarten Sap, Ronan Le Bras, Daniel Fried, and Yejin Choi. 2022. Neural theory-of-mind? on the limits of social intelligence in large lms. In Proceedings of the 2022 conference on empirical methods in natural language processing, pages 3762–3780.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social iqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4463–4473.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Shulan Xiaojiejie. 2025. Xi pin zhen huan zhuan (a close reading of empresses in the palace), video commentary series. https://space.bilibili.com/ 618442764. Bilibili video series, ongoing (part 136 as of July 2026); parts 1–110 (about 47 hours of audio) used in this work. Accessed 2026-07-31.

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. Pub: A pragmatics understanding benchmark for assessing llms’ pragmatics capabilities. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12075–12097.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793.

Cynthia Van Hee, Els Lefever, and Véronique Hoste. 2018. SemEval-2018 task 3: Irony detection in English tweets. In Proceedings ofthe 12th International Workshop on Semantic Evaluation, pages 39–50.

Zhiyuan Wen, Rui Wang, Qianlong Wang, Lin Gui, Yunfei Long, Shiwei Chen, Bin Liang, Min Yang, and Ruifeng Xu. 2025. FGVIrony: A Chinese dataset of fine-grained verbal irony. Information Processing & Management, 62(5):104169.

Xiaomi LLM-Core Team. 2025. MiMo-VL technical report. arXiv preprint arXiv:2506.03569.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. 2025. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. 2025. Learning to reason under off-policy guidance. In Advances in Neural Information Processing Systems, volume 38, pages 117157–117186.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, et al. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pages 113222–113244.

Shisen Yue, Siyuan Song, Xinyuan Cheng, and Hai Hu. 2024. Do large language models understand conversational implicature-a case study with a chinese sitcom. In Proceedings ofthe 23rd Chinese National Conference on Computational Linguistics (Volume 1: Main Conference), pages 1270–1285.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, volume 38.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488.

Yitong Zhang, Yuhan Xiang, and Mingxuan Liu. 2026. The mask of civility: Benchmarking chinese mock politeness comprehension in large language models. arXiv preprint arXiv:2602.03107.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. 2024. Sotopia: Interactive evaluation for social intelligence in language agents. In International Conference on Learning Representations.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

## Supplementary Material

Appendix Contents   
A Dataset Construction and Statistics 14   
B Evaluation Prompts 14   
C Training Details 15   
D T3 Judge 16   
E Multimodal Probes: an Audit Protocol 16   
F Additional Results 17   
F.1 Evaluation Robustness 17   
F.2 Training Analyses 17   
F.3 CPR-GRPO Loop Ablations 18   
F.4 Cross-Backbone Replication 19   
G Model Comparisons and Transfer 19   
G.1 Model Performance and Reason  
ing Settings 19   
G.2 External Evaluation of Fixed   
Checkpoints . 20   
G.3 Retraining on External Datasets 20   
G.4 Character Renaming 21

<table><tr><td>Benchmark</td><td>Size Tasks ToM Listener Train.</td><td></td><td></td><td></td></tr><tr><td>SwordsmanImp</td><td>200</td><td>1</td><td></td><td></td></tr><tr><td>Mask of Civility</td><td>100</td><td>1</td><td></td><td></td></tr><tr><td>PUB (En)</td><td>28k</td><td>14</td><td>partial</td><td></td></tr><tr><td>EmotionQueen (En)</td><td>10k</td><td>4</td><td></td><td></td></tr><tr><td>ZhenhuanBench</td><td>1,592</td><td>4</td><td></td><td></td></tr></table>

Table A.1: Pragmatics benchmarks. “Listener” marks per-listener uptake structure; “Train.” marks annotations designed to double as training signal.

## A Dataset Construction and Statistics

Table A.2 gives the per-task-by-split item counts and Table A.3 the construction funnel from raw alignments to final items. The paragraphs below document the annotation schema, the extraction and audit passes, and the assembly checklist behind those counts.

<table><tr><td>Split</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>Total</td></tr><tr><td>Public</td><td>239</td><td>240</td><td>206</td><td>106</td><td>791</td></tr><tr><td>Hidden test</td><td>92</td><td>82</td><td>103</td><td>42</td><td>319</td></tr><tr><td>Training</td><td>140</td><td>158</td><td>120</td><td>64</td><td>482</td></tr><tr><td>All</td><td>471</td><td>480</td><td>429</td><td>212</td><td>1,592</td></tr></table>

Table A.2: ZhenhuanBench items by task and split. Aligning 76 episodes (18,297 utterances, 1,214 scenes) with the 110-part expert commentary (47 hours) yields 1,506 alignments. The 68 podcast transcripts and 1.26M-character novel provide cross-source corroboration. Training and hidden episodes are disjoint.

Annotation schema. Each of the 1,506 alignments pairs one screenplay utterance with the expert’s reading of it and carries four structured fields. real\_intent is a one-sentence statement of what the speaker is doing (e.g., 借道贺暗指 皇上极少召幸, 公开质疑龙种真伪, “using a congratulation to hint that the Emperor rarely summons her, publicly casting doubt on whether the child is genuine,” for the running example of Figure 1). emotion\_language\_mismatch takes one of aligned, partial\_mismatch, or inverted, operationalizing the implicational mismatch account of mock politeness. third\_party\_reactions lists every listener present with an uptake tag in {yes, no, partial} and a one-sentence reason. rhetoric\_sub\_type records the rhetorical device and supports the Culpeper three-way classification.

Extraction and audit. Extraction runs in two passes. The first pass converts a commentary span into candidate fields with a verbatim quote anchor from the commentary, and a second audit pass verifies each field against its anchor and repairs or discards unsupported fields. On the full set the audit flagged 292 of the 1,506 alignments (19.4 percent), the leading error types being plot summary written as intent (136), character misattribution (62), hallucinated detail (61), and event misattribution (33), and repaired 269 of them against their anchors.

<table><tr><td>Stage</td><td>Count</td></tr><tr><td>Aligned annotations flagged by audit pass (repaired)</td><td>1,506 292 (269)</td></tr><tr><td>Candidate items (two assembly passes)</td><td>8,544</td></tr><tr><td>rejected: option/gold length balance</td><td>4,471</td></tr><tr><td>rejected: options-only probe†</td><td>1,816</td></tr><tr><td>rejected: anti-leakage and consistency</td><td>665</td></tr><tr><td>Final items (18.6% of candidates)</td><td>1,592</td></tr></table>

Table A.3: Construction funnel. <sup>†</sup>still solved from options alone after two rewrites. Anti-leakage rejections include circular evidence and future-tense narration.

Item assembly. Items are assembled under an anti-leakage checklist. Questions use neutral phrasing that does not presuppose hostility, scenes are truncated before the consequences of the utterance unfold, options are length-balanced within 15 percent, every MCQ is evaluated under three option permutations, and sibling items derived from one alignment never share option text.

Primary-source selection. We selected the commentary for its line-by-line analysis of dialogue and quotable evidence for the annotated interpretations. The account is certified by Bilibili as a wellknown film-and-television creator and has about 491,000 followers; the 47-video archive of parts 1–110 used in our corpus has 32.8 million plays (accessed 2026-07-31). Each extracted field carries a verbatim quote anchor and undergoes an audit against the commentary. The English recap podcast and source novel provide cross-source corroboration (§3.4).

## B Evaluation Prompts

The main condition P\_context presents, in order, the cast block (name, rank, and a one-line persona for every character present), the scene header, the full scene dialogue up to the truncation point, the target utterance, the question, the options, and a JSON answer contract. P\_lite drops the cast block and scene. The knowledge-enhanced condition

<table><tr><td>Prompt condition</td><td>Base 8B CPR-GRPO Gemma-31B</td><td></td></tr><tr><td>Dialogue only</td><td>44.0</td><td>52.8 62.0</td></tr><tr><td>+ Full scene (main)</td><td>40.7</td><td>59.3 65.7</td></tr><tr><td>- cast block</td><td>41.2</td><td>58.3 66.2</td></tr><tr><td>+ Knowledge (defn.)</td><td>29.2</td><td>43.5 61.1</td></tr><tr><td>+ Oracle commentary</td><td>60.6</td><td>75.5 85.2</td></tr></table>

Table B.1: Text ablations on the hidden set (percent). The injected definition lowers all three scores; expert commentary yields the highest accuracy for each model.

P\_definition prepends the following definition, given verbatim with an English gloss.

【Haugh 2014 mock politeness 定义(理论参考<sub>)</sub>】<sub>mock politeness</sub> 指外形上礼貌<sub>(</sub>敬语<sub>/</sub>谦辞/客套) 但内核为非礼貌(攻击/讽刺/施压) 的语言现象。形式与功能错位<sub>,</sub>用礼貌形式包装非礼貌意图。识别要点(1) 表面话语寄存器为礼貌<sub>(2)</sub> 真实意图为攻击或操控<sub>(3)</sub> 听者通常听得懂<sub>,</sub>但因规范无法当场揭破。

English gloss. [Haugh 2014 mock politeness definition, theoretical reference] Mock politeness is speech whose form is polite (honorifics, deference, courtesy formulas) but whose substance is impolite (attack, sarcasm, coercion). Form and function are misaligned, a polite wrapper around an impolite intent. Cues are (1) a polite surface register, (2) a real intent to attack or manipulate, (3) listeners who usually understand but cannot expose the attack on the spot.

The oracle condition appends a block labeled 【专家解说提示】 (“expert commentary hint”) containing the expert’s commentary excerpt for the aligned utterance. The scaffold prompt used for training and hidden-set evaluation appends a five-step output contract requiring <evidence>, <mismatch>, <listeners>, and <intent> tags before the JSON answer. These prompt comparisons disable thinking. Reasoning-setting experiments use the configurations in Appendix G.1. RL-trained checkpoints repeated prompt text when evaluated with a thinking template absent from training.

## C Training Details

All main runs use Qwen3-8B with LoRA rank 64, α=128, dropout 0.05, on all seven projection matrices (174.6M trainable parameters, 2.1 percent), on a single H200 GPU. SFT over 338 traces, one GRPO round, one probe, and one Replenish epoch each complete within hours; a full loop iteration fits within a single GPU-day. SFT uses learning rate 10<sup>−5</sup>, three epochs over 338 traces. Each GRPO round samples G=6 rollouts per item at temperature 0.8 from the round-start policy, drops zero-variance groups, and trains one epoch with

```latex
Algorithm 1 One CPR-GRPO iteration
Require: policy π<sub>θ</sub> after gold-trace SFT; items D with gold
$( a ^ { * } , m ^ { * } , u ^ { * } )$ ; gold traces G; group size G; probe size G<sup>′</sup>;
margin δ
1: {Compress (one GRPO round)}
2: π<sub>0</sub> ← π<sub>θ</sub>; for each x<sub>i</sub>: sample G rollouts from π<sub>0</sub>, score
with Eq. 1
3: drop zero-variance groups; $\hat { A } _ { g }  ( r _ { g } - \mu _ { i } ) / ( \sigma _ { i } + \epsilon )$
4: update θ by maximizing the clipped GRPO objective
(Eq. C.1)
5: {Probe (chance-corrected band diagnosis)}
6: for each x<sub>i</sub>: greedy correctness g<sub>i</sub>; pˆ<sub>i</sub> ← correct rate over
G<sup>′</sup> samples at T=0.8
7: compute excess e<sub>i</sub> and weights w<sub>i</sub> by Eq. 2
8: {Replenish (gold-targeted distillation)}
9: one epoch of weighted CE: $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = - \sum _ { } } \end{array}$ w<sub>i</sub> log π<sub>θ</sub>(τ<sub>i</sub> |
x<sub>i</sub>) (no KL anchor)
10: Return the updated policy.
```

## CPR-GRPO

![](images/2b3ad7afb3651090d6da89ca2514c32e07435b7aa9184cbffe281e1a81bc7a01.jpg)  
Figure C.1: The CPR-GRPO loop. Compress runs GRPO, Probe estimates chance-corrected recovery, and Replenish weights gold traces using those estimates.

token-level importance ratios, asymmetric clipping $( \varepsilon _ { l } , \varepsilon _ { h } ) = ( 0 . 2 , 0 . 2 8 )$ , and a k3 KL penalty $\scriptstyle ( \beta = 0 . 0 3 )$ to the round-start policy π<sub>0</sub>, which acts as a per-round reference reset. The full Compress objective is

$$
\begin{array} { r l } & { \mathcal { I } ( \theta ) = \mathbb { E } \Big [ \frac { 1 } { \vert y \vert } \sum _ { t } \operatorname* { m i n } \big ( \rho _ { t } \hat { A } , ~ \bar { \rho } _ { t } \hat { A } \big ) \Big ] } \\ & { \quad \quad - \beta \mathrm { D } _ { k 3 } \big ( \pi _ { \theta } \vert \vert \pi _ { 0 } \big ) , } \end{array}\tag{C.1}
$$

where $\rho _ { t } = \pi _ { \theta } ( y _ { t } \mid \cdot ) / \pi _ { 0 } ( y _ { t } \mid \cdot )$ is the token importance ratio and $\bar { \rho } _ { t } = \mathrm { c l i p } { \left( \rho _ { t } , 1 - \varepsilon _ { l } , 1 + \varepsilon _ { h } \right) }$ its clipped counterpart. Rewards follow Eq. 1 with $\lambda { = } 0 . 3$ . The probe uses $G ^ { \prime } { = } 1 6$ samples at temperature 0.8, margin $\delta { = } 0 . 0 6$ , and the Replenish step is one epoch of hard cross-entropy with oversampling weights 4 and 2 and no KL anchor. With sequencelevel importance ratios, the KL term diverged to above $\mathrm { 3 \times 1 0 ^ { 4 } }$ within one epoch, degrading the policy below its SFT starting point. Token-level ratios with the same data trained stably (final policy loss

$$
3 { \times } 1 0 ^ { - 3 } , \mathrm { K L } \approx 0 . 0 0 9 ) .
$$

## D T3 Judge

T3 responses are scored per listener using the following rubric. The leaderboard and reasoning comparisons use Qwen3.8-Max, temperature 0, thinking disabled, and a 256-token output cap. The judge receives the scene, speaker, target utterance, listener, reference reaction, and predicted reaction. Model identities and previous scores are omitted.

【任务】对比模型输出的<sub>listener</sub> 反应描述<sub>vs</sub>gold listener 反应描述, 评分0/1/2。【评分细则】0 分: 模型反应与gold 主旨明显不同/完全没答到点/方向相反。1 分: 部分正确, 抓到了<sub>gold</sub> 的某一面但缺关键细节或有小错。<sub>2</sub>分: 与gold 主旨一致, 关键判断(听懂/没懂/反击/顺从) 都对,用词可以不同。

English gloss. [Task] Compare the model’s description of each listener’s reaction against the gold description and score 0/1/2. [Rubric] 0, the response clearly departs from the gold reading, misses the point entirely, or points the opposite way. 1, partially correct, capturing one aspect of the gold but missing key details or making a minor error. 2, consistent with the gold reading, with the key judgments (understood or not, strikes back or complies) all correct; wording may differ.

Scores are averaged over listeners and normalized to [0, 1] per item. The judge appears only at evaluation time and never in training.

## E Multimodal Probes: an Audit Protocol

We test whether frames from the scene help models read mock politeness. Multimodal socialinteraction work reports that visual and auditory cues carry signal in multi-party scenes (Kang et al., 2025; Ouyang et al., 2025), motivating an evaluation of visual context for mock-politeness understanding. Each probe item pairs a benchmark MCQ with six frames spanning $[ t _ { 0 } - 2 \mathrm { s } , t _ { 0 } { + } 6 \mathrm { s } ]$ around the utterance moment, located by bigram containment against Whisper segments. Three conditions are compared under identical text: T (text only), TV (text plus the item’s own frames), and VC (text plus another item’s frames under a fixed derangement), so TV−VC measures the difference between matched and mismatched visual context. Two frame sources are used: commentary-video frames (70 items) and episode frames (56 items), the latter also in a subtitle-cropped variant since episode frames carry hardcoded subtitles. Table E.1 gives the full matrix.

Both matched and mismatched frames improve accuracy over text alone, with TV−T gains of +4 to +11 points. On episode frames, Sonnet’s TV−VC is −4.8 points (95% CI [−8.9, −0.6]); after subtitle cropping, it is 0.0. All subtitle-cropped conditions in Table E.1 have TV−VC confidence intervals that include zero.

<table><tr><td>Model</td><td>Frames</td><td>T</td><td>TV VC</td><td>TV-VC</td></tr><tr><td>Qwen3-VL-8B</td><td>comm.</td><td>46.7</td><td>51.0 51.0</td><td>+0.0</td></tr><tr><td>Qwen3-VL-8B</td><td>epis.</td><td>40.5</td><td>44.6 43.5</td><td>+1.2</td></tr><tr><td>Qwen3-VL-8B</td><td>crop</td><td>39.9</td><td>44.6 43.5</td><td>+1.1</td></tr><tr><td>Sonnet-4.6</td><td>epis.</td><td>58.3</td><td>60.7 65.5</td><td>-4.8*</td></tr><tr><td>Sonnet-4.6</td><td>crop</td><td>58.3</td><td>62.5 62.5</td><td>+0.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>epis.</td><td>67.9</td><td>70.8 74.4</td><td>-3.6</td></tr><tr><td>Gemini-2.5-Pro</td><td>crop</td><td>71.4</td><td>73.2 75.6</td><td>-2.4</td></tr></table>

Table E.1: Vision probe (percent; paired). <sup>∗</sup>95% CI [−8.9, −0.6]; all other TV−VC differences include zero.

Scaffolding and reaction-shot variants. We vary visual response scaffolding and frame timing to test how models use visible expressions and listener reactions. We run a two-by-two on Qwen3-VL-30B-A3B over a 138-item probe set (414 paired observations per cell, all subtitlecropped): frames at the utterance moment versus reaction shots at $[ t _ { 0 } + 2 \mathrm { s } , t _ { 0 } + 1 0 \mathrm { s } ]$ , each answered directly versus with a describe-then-answer visual chain of thought that first describes every visible character’s expression and body language. The mismatched-frame control is kept in every cell. TV−VC is +0.7 [−1.4, +3.1], +0.2 [−1.4, +1.9], +0.5 [−1.9, +2.9], and +1.0 [−1.2, +3.1] across the four cells, with all four confidence intervals including zero.

An audio probe. We test whether prosodic cues improve performance using the same protocol on the 138-item set. Each item pairs its MCQ with the 8-second original-audio clip of the utterance moment (16 kHz mono, cut by the same Whisper timestamps), and three conditions are compared, T (text only), TA (text plus the item’s own audio), and AC (text plus another item’s audio under the same fixed derangement).

We compare matched and mismatched audio against the text-only baseline to distinguish their effects. On both MiniCPM-o-2.6 and Gemini-2.5- Pro (Comanici et al., 2025), matched audio beats the mismatched control by an identical TA−AC of +3.6, significantly so in both cases, yet the decomposition against text-only is opposite. For MiniCPM, matched audio raises accuracy over text $\mathrm { ( T A { - } T = + 3 . 1 \ [ + 0 . 2 , + 6 . 3 ] ) }$ while a random clip is neutral $( \mathrm { A C - T = - 0 . 5 } )$ . For Gemini, matched audio leaves accuracy unchanged (TA− $- \mathrm { T } = + 0 . 0$ [−3.1, +3.1]) and mismatched audio lowers accuracy $( \mathrm { A C - T = - 3 . 6 \ [ - 6 . 8 , - 0 . 5 ] ) }$

Qwen3-Omni-30B-A3B (Xu et al., 2025), has negative point estimates for both audio comparisons (TA− $- \mathrm { T } = - 2 . 9 .$ , TA− $\mathbf { \partial } \cdot \mathbf { A C } = - 2 . 4 .$ , both intervals spanning zero), and its visual comparison gives $( \mathbf { T } \mathbf { V } - \mathbf { V } \mathbf { C } = - 0 . 2 )$ . Matched-minus-mismatched differences reflect changes in either condition; comparison with text alone distinguishes improvements from losses under mismatched inputs. The same confound explains the vision numbers, where both matched and mismatched frames raise Gemini’s accuracy over text $( \mathrm { T V - T } = + 4 . 3 , \mathrm { V C - T } = + 6 . 8 )$ so TV−VC is negative. Audio improves MiniCPM in this evaluation; corresponding gains are absent for Gemini and Qwen3-Omni.

## F Additional Results

## F.1 Evaluation Robustness

T3 judge robustness. We score the same T3 responses with Qwen3.8-Max and Sonnet-4.6 under the rubric in Appendix D. Across all 34 configurations, Spearman correlation is ρ = 0.985 for T3 and $\rho = 0 . 9 9 9$ for overall scores (Table F.1). MCQ scores are fixed.

Mismatch confusion. On the hidden set, both trained models over-predict inverted and struggle most on partial\_mismatch (gold distribution 92/71/53 for inverted, partial, aligned; majority baseline 42.6). The SFT model classifies mismatch at 50.0 percent and the RL model at 50.9, with 42 of 71 partial items misread as inverted by the SFT model, indicating frequent confusion between partial and inverted mismatch.

## F.2 Training Analyses

Performance across prompts and tasks. CPR scores 59.3 under the standard hidden prompt, 74.5 under the training-matched scaffold prompt, and 67.0 on public MCQs under the leaderboard protocol. The public-set score uses the leaderboard prompt in Table 1. CPR scores 27.39 on public T3, close to Base at 26.94. Training on gold why phrases monotonically hurts (34.3, 32.0, 28.5). The real\_intent field states the intended interpretation; why records the evidence supporting an uptake label. Training on the latter lowers reactionprediction scores in these experiments.

![](images/e705b2a5a870ecc55d788f3dc824f74fded944909dc2691cfc1e93e56d6aef31.jpg)  
Figure F.1: AGVR SFT data efficiency on the hidden set. Accuracy crosses the base band (mean and SD of 62 evaluations) between 96 and 168 traces and saturates near 62.5.

Auxiliary rewards. Across three independent runs, answer-only GRPO scores $6 5 . 1 \pm 2 . 8$ and the three-head mix scores $6 6 . 2 \pm 0 . 8 ( \mathrm { m e a n } \pm \mathrm { S D } )$ Auxiliary rewards reduce variability across runs, with a smaller change in mean accuracy.

Data efficiency. Figure F.1 varies the number of gold traces in AGVR SFT (nested subsets, three epochs, identical otherwise). The hidden-set score crosses the base between 96 and 168 traces and reaches 62.5 at the full 338, with clearly diminishing returns past 253 (a further 85 traces add 3.2 points). The supervised stage uses 338 humananchored traces and requires no model-generated rationales.

Chance-Corrected pass@k. Figure F.2 plots naive pass@k against its chance-corrected counterpart for the base, SFT, and RL checkpoints. At k=16, naive pass@k approaches 95 percent while chance-corrected coverage approaches zero as random-guess coverage increases. The chancecorrected value at k=1 rises with training, from 23.9 at the base to 33.0 after SFT and 38.2 after RL. This is the full-curve form of the 7.5× inflation reported in §5.3.

Item-level flow. Tracking each of the 216 hidden items across base, SFT, GRPO round 2, Replenish, and the loop peak shows the stages behave differently. SFT gains 67 items but overwrites 21 of the base model’s 88 (76 percent retention); GRPO retains 92 percent of previously correct answers; Replenish retains 84 percent and corrects the most errors. Twenty-two items (10 percent) are never solved at any stage, indicating errors that persist through all evaluated stages, and 54 (25 percent) are solved at every stage.

<table><tr><td>Model</td><td>Q-T3</td><td>S-T3</td><td>Q-All</td><td>S-All</td><td>Model</td><td>Q-T3</td><td>S-T3</td><td>Q-All</td><td>S-All</td></tr><tr><td>GPT-6 Astra</td><td>58.37</td><td>61.08</td><td>73.48</td><td>74.19</td><td>Qwen3-32B</td><td>34.43</td><td>36.65</td><td>44.11</td><td>44.69</td></tr><tr><td>Gemini 3.7 Flash</td><td>53.28</td><td>54.37</td><td>72.28</td><td>72.57</td><td>Qwen3.5-9B</td><td>31.07</td><td>31.03</td><td>41.59</td><td>41.58</td></tr><tr><td>Kimi K3</td><td>53.44</td><td>55.83</td><td>66.76</td><td>67.38</td><td>Qwen3-14B</td><td>30.91</td><td>32.40</td><td>39.15</td><td>39.54</td></tr><tr><td>DeepSeek V4 Pro</td><td>49.39</td><td>52.99</td><td>62.55</td><td>63.49</td><td>GPT-5-mini</td><td>35.68</td><td>35.88</td><td>38.12</td><td>38.17</td></tr><tr><td>Qwen3.8-Max</td><td>49.84</td><td>49.47</td><td>62.54</td><td>62.44</td><td>Qwen3-VL-8B</td><td>31.96</td><td>36.29</td><td>37.40</td><td>38.53</td></tr><tr><td>Kimi-K2.6</td><td>48.18</td><td>46.20</td><td>61.98</td><td>61.46</td><td>Qwen3-8B</td><td>26.94</td><td>30.99</td><td>34.20</td><td>35.25</td></tr><tr><td>Gemma-4-31B</td><td>42.03</td><td>44.50</td><td>59.62</td><td>60.26</td><td>GLM-4.6V-Flash</td><td>28.80</td><td>33.45</td><td>33.16</td><td>34.38</td></tr><tr><td>Sonnet-4.6</td><td>50.61</td><td>55.34</td><td>58.94</td><td>60.18</td><td>Qwen3.5-4B</td><td>15.82</td><td>16.91</td><td>32.94</td><td>33.23</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>45.75</td><td>47.82</td><td>58.57</td><td>59.10</td><td>InternVL3-8B</td><td>26.01</td><td>28.64</td><td>31.81</td><td>32.49</td></tr><tr><td>DeepSeek-V4-Flash</td><td>45.27</td><td>44.09</td><td>56.80</td><td>56.49</td><td>GLM-4.1V-9B</td><td>25.81</td><td>29.25</td><td>31.12</td><td>32.02</td></tr><tr><td>CPR-GRPO</td><td>27.39</td><td>30.99</td><td>56.69</td><td>57.63</td><td>GPT-4o-mini</td><td>28.28</td><td>29.98</td><td>30.75</td><td>31.19</td></tr><tr><td>Qwen3-VL-235B</td><td>42.56</td><td>44.62</td><td>52.42</td><td>52.96</td><td>Qwen3-VL-4B</td><td>22.01</td><td>25.69</td><td>30.26</td><td>31.22</td></tr><tr><td>GPT-5</td><td>49.47</td><td>51.09</td><td>52.33</td><td>52.75</td><td>GLM-4.5-Air</td><td>29.49</td><td>29.25</td><td>30.06</td><td>29.99</td></tr><tr><td>GPT-40</td><td>38.96</td><td>41.63</td><td>49.34</td><td>50.03</td><td>MiniCPM-o-2.6</td><td>21.72</td><td>25.16</td><td>29.42</td><td>30.32</td></tr><tr><td>GLM-4.6V</td><td>37.38</td><td>41.87</td><td>47.41</td><td>48.58</td><td>MiMo-VL-7B</td><td>28.72</td><td>29.69</td><td>28.34</td><td>28.59</td></tr><tr><td>Qwen3-VL-30B</td><td>36.89</td><td>41.14</td><td>44.37</td><td>45.48</td><td>Qwen2.5-VL-7B</td><td>22.57</td><td>22.90</td><td>28.26</td><td>28.34</td></tr><tr><td>Haiku-4.5</td><td>40.29</td><td>43.61</td><td>44.37</td><td>45.24</td><td>Qwen3.5-2B</td><td>12.62</td><td>12.78</td><td>26.04</td><td>26.09</td></tr></table>

Table F.1: Cross-judge scores on identical responses (percent). Q: Qwen3.8-Max; S: Sonnet-4.6. MCQ scores are fixed. Both judges use the same rubric with thinking disabled.

<table><tr><td>Cold start</td><td>SFT</td><td>+GRPO</td><td>Observation</td></tr><tr><td>Full recipe</td><td>62.0/61.1</td><td>66.7/66.7</td><td>baseline</td></tr><tr><td>Interface only</td><td>56.9/55.6</td><td>60.2/58.8</td><td>GRPO gain</td></tr><tr><td>Content, no iface</td><td>1.9/3.7</td><td>5.6/25.0</td><td>Parse failures</td></tr><tr><td>None (base)</td><td></td><td>40.3</td><td>No gain</td></tr></table>

Table F.2: Interface-alignment two-by-two on the hidden set, two seeds. Interface-only SFT enables GRPO gains in both seeds; unparseable gold prose yields low scores.

Evaluation reproducibility. Repeated greedy decoding of the same 216 hidden items produces checkpoint-dependent score variation. Base averages 38.6 with SD 1.7 over 62 evaluations (range 34.3–41.7); parse failures are at most 5.6 percent and are uncorrelated with score. SFT scores range from 58.8 to 62.0. Repeated scores are 66.2/66.2/66.7 for GRPO round 2, 75.5/75.9/75.9 for the additional Compress checkpoint, and 74.5/76.4/77.3 for the round-4 checkpoint. Paired bootstrap intervals compare predictions on the same items within an evaluation.

## F.3 CPR-GRPO Loop Ablations

Additional uses of recovery estimates. The band supports three uses, a hard filter (in-band groups at G=16 reach 68.1/71.3), a soft advantage weight (excess-weighted GRPO beats its uniform control on identical rollouts, unlike raw-variance weighting (Jiang et al., 2025)), and the Replenish gate.

![](images/16d145088bd8b0374b300d5175177d2ed7e45605f08fd88c4433e8e77a4096f4.jpg)  
Figure F.2: Naive pass@k (solid) vs. chance-corrected coverage (dashed). Chance-corrected coverage approaches zero at k=16.

Replenish constant sensitivity. Rebuilding the Replenish set from the same round-2 diagnosis under alternative constants and retraining from the same checkpoint gives 67.6 for flattened weights 2/2, 65.7 for aggressive weights 8/4, and 63.0 for a tightened margin δ=0.12, against 68.5 for the default 4/2/0.06. Equal weights remove the submodal emphasis, and the larger 8/4 weights lower accuracy. Tightening the margin reduces the selected sub-modal set from 29 items at δ=0.06 to 20.

Excess-weighting dose response. From the round-2 checkpoint, one sampling of 16 rollouts per item gives 68.5, 71.3, 73.1, 72.2, 69.9 for α = 0, 2, 4, 6, 8, an inverted-U peaking near α=4. A second independent sampling gives 70.4, 73.1,

<table><tr><td>Control</td><td>Score</td><td>Factor tested</td></tr><tr><td>GA-GRPO (gold in groups)</td><td>65.3</td><td>re-exposure</td></tr><tr><td>Round-5 pure iteration RFT (rejection sampling)</td><td>69.4 63.0</td><td>more compute self-distill</td></tr><tr><td>DPO (β=0.05)</td><td>53.7</td><td>preference opt.</td></tr><tr><td>More SFT (5 epochs)</td><td>61.6</td><td>more gradient</td></tr><tr><td>Ans-only vs. 3-head</td><td>65.1/66.2</td><td></td></tr><tr><td>Teacher CoT (253 items)</td><td>49.5/55.6</td><td>reward richness teacher swap</td></tr></table>

Table F.3: Controls on the hidden set. Vanilla GRPO from the same start scores 66.7; CPR-GRPO reaches 71.8/74.5 in two seeds.

69.9 for $\alpha = 0 , 2 ,$ , 4. $\mathbf { A } \mathbf { t } \ \alpha { = } 2$ , weighting improves over the corresponding uniform control by 2.8 and 2.7 points in the two samples. The two samples reach their highest scores at α=4 and $\alpha { = } 2 .$ , respectively. Halving the probe to $G ^ { \prime } { = } 8$ shifts the diagnosed band only slightly (25 and 30 items against 29 and 33) and reaches 66.2 against 68.5 for the $G ^ { \prime } { = } 1 6$ default, a 2.3-point decrease with half as many probe samples.

Additional iteration. From the 74.5 checkpoint, Compress reaches 75.9 (T1/T2/T4 of 75.0/76.8/76.2), the probe finds one sub-modal and 17 strong-recovery items with a residual excess of 10 item-equivalents on the training set, and an additional Replenish step yields 70.4. From the 68.5 checkpoint, Compress reaches 69.4, the probe finds 8 and 27 with 15 item-equivalents, and an additional Replenish step yields 68.1. Paired confidence intervals for both Compress increments include zero (see Evaluation reproducibility, Appendix F.2). Excess-weighted GRPO scores 69.0 and 75.5 for the additional Compress step, compared with 69.4 and 75.9 for unweighted GRPO. Earlier-round weighting results appear in Excessweighting dose response above.

## F.4 Cross-Backbone Replication

We apply the same recipe to GLM-4-9B (Team GLM et al., 2024) and Qwen3-14B. On GLM-4- 9B, AGVR SFT lifts the hidden-set score from 42.1 to 54.2 (paired +12.0, 95% CI [+4.2, +19.9], $p = 0 . 0 0 4 )$ . In the paired GRPO comparison, SFT scores 55.6 and one Compress round raises accuracy to 60.2 (+4.6). On Qwen3-14B the identical pipeline goes 37.0 to 70.4 after SFT (paired +33.3) and 71.3 after one GRPO round, so the 14B SFT endpoint already exceeds the 8B model’s GRPO endpoint. GRPO adds +4.6, +4.7, and +0.9 points on GLM-4-9B, Qwen3-8B, and Qwen3-14B, respectively; the smallest gain occurs at the highest

SFT endpoint. SFT and GRPO improve scores on all three tested backbones.

Full-loop transfer. The complete Compress– Probe–Replenish iteration also transfers to both backbones. On GLM-4-9B, from the Compress checkpoint (58.3 in the paired evaluation), the probe diagnoses 37 sub-modal and 34 strongrecovery items $( \Sigma e \ : = \ : 2 1 . 4 )$ , and one Replenish epoch lifts the same 216 items to 60.6 (paired +2.3, 95% CI $[ - 5 . 1 , + 9 . 7 ] ;$ , reducing the sub-modal set on re-probing (11 items, $\Sigma e = 1 3 . 6 )$ . Flips concentrate where diagnosed, with 78 and 88 percent of sub-modal and strong-recovery training items flipping correct against 32 percent of the untargeted greedy-wrong bucket, at 91 percent retention. A second Compress round changes accuracy by +4.2 points (95% CI [−2.3, +10.6]).

On Qwen3-14B the probe finds a smaller band (17 sub-modal and 19 strong-recovery items, $\Sigma e =$ 12.0), yet one Replenish epoch raises hidden-set accuracy from 69.9 to 75.5 (paired +5.6, 95% CI $[ - 0 . 5 , + 1 1 . 6 ] )$ , reducing the sub-modal set from 17 to 6 items while re-probing identifies further strong-recovery items. The targeting contrast is clear for strong-recovery items (84 percent flip correct) but weak for the already-small sub-modal bucket. A second Compress round changes the hidden score by −2.3 points (95% CI [−8.3, +3.7]) and reduces the two recovery groups to 8 items each $( \Sigma e = 5 . 5 )$

Recovery estimates across backbones. Singleiteration Replenish gains are +1.8, +2.3, and +5.6 points on the three backbones. Both GLM-4-9B and Qwen3-14B gain at Σe ≈ 12, while the additional Compress step changes their scores by +4.2 and −2.3 points, respectively. These comparisons relate recovery estimates to the observed gains from further training.

## G Model Comparisons and Transfer

## G.1 Model Performance and Reasoning Settings

Scoring. The leaderboard and reasoning comparisons use the 791 public items: 585 MCQs (T1/T2/T4) and 206 T3 items. MCQs use three option permutations and majority voting with the fixed tie rule. Qwen3.8-Max applies the rubric in Appendix D. Overall score is (585 MCQ + 206 T3)/791. Invalid final answers receive zero.

<table><tr><td>Model</td><td>Off</td><td>Low</td><td>High/Med. Max/Xhigh</td><td></td></tr><tr><td>DeepSeek V4 Pro</td><td>58.36</td><td>61.13</td><td>65.53</td><td>68.44</td></tr><tr><td>Qwen3.8-Max</td><td>63.43</td><td>63.42</td><td>61.60</td><td>62.31</td></tr><tr><td>Qwen3.8-27B</td><td>51.57</td><td>54.64</td><td>54.09</td><td>52.13</td></tr></table>

Table G.1: Overall score across reasoning settings, with Qwen3.8-Max judging T3. DeepSeek uses off/low/high-/max; Qwen uses off/low/medium/xhigh, with a 16K thinking budget for Qwen-Max medium.

Models and reasoning settings. Table 1 spans 11 model families (Yang et al., 2025; Qwen Team, 2026; Bai et al., 2025a,b; Kimi Team et al., 2025; DeepSeek-AI et al., 2024; Google DeepMind, 2026b; Team GLM et al., 2024; GLM-V Team et al., 2025; Zhu et al., 2025; Yao et al., 2024; Xiaomi LLM-Core Team, 2025; OpenAI, 2025; OpenAI et al., 2024; Anthropic, 2026, 2025; Google DeepMind, 2026a). The reasoning comparison fixes each model version and the public items, varies the reasoning setting, and allows up to 32,768 output tokens. Temperature is zero where supported. Qwen3.8-27B runs locally in BF16. Each setting contributes one generation pass.

DeepSeek gains 10.07 points from off to max (scene-bootstrap 95% CI [7.41, 12.80]). Qwen-Max’s off and low settings score almost identically; all its enabled-versus-off confidence intervals include zero. Qwen-27B improves by 3.07 points from off to low ([0.23, 6.02]), with smaller gains at higher budgets. Across the 12 configurations, overall-score rankings under Qwen and GLM-5.3 judging are identical (Spearman ρ = 1.00).

## G.2 External Evaluation of Fixed Checkpoints

We evaluate the Qwen3-8B Base, SFT, GRPO, and CPR checkpoints without training on target-dataset labels. SFT is trained on 338 gold traces.

MultiPragEval (Park et al., 2024) contributes 300 Chinese and 300 English pragmatic-inference questions; accuracy averages three option permutations. ToSarcasm (Liang et al., 2022) uses its 973-item Chinese test set, evaluated both with the topic and after removing it, and reports binary Macro-F1. English iSarcasmEval (Abu Farha et al., 2022) uses 1,400 Task-A items (sarcastic-class F1) and 200 Task-C pairs (accuracy under the original order). Each pair contrasts a sarcastic text with its nonsarcastic rephrase.

Table 4 in §5.6 reports the results. CPR improves over Base on all six evaluations, including +4.94

<table><tr><td>Dataset</td><td>Warm</td><td>Pool</td><td>Dev</td><td>Test</td></tr><tr><td>FGVIrony</td><td>1,024</td><td>2,048</td><td>958</td><td>1,973</td></tr><tr><td>CPED (each regime)</td><td>456</td><td>912</td><td>1,687</td><td>27,438</td></tr><tr><td>SemEval Task 3B</td><td>192</td><td>384</td><td>767</td><td>784</td></tr></table>

Table G.2: External training and evaluation sizes. Warm and Pool are disjoint supervised-start and subsequent training sets.

Macro-F1 on ToSarcasm and +3.33 sarcastic-class F1 on iSarcasmEval A. Removing the topic raises ToSarcasm scores for every checkpoint.

## G.3 Retraining on External Datasets

Tasks and splits. We independently train Qwen3- 8B on seven-class Chinese FGVIrony (Wen et al., 2025), 19 dialogue acts from Chinese CPED (Chen et al., 2022), and four-class English irony from SemEval-2018 Task 3B (Van Hee et al., 2018). FGVIrony training examples sharing a news component with the test set are removed; development data are grouped by news component. CPED uses dialogue-disjoint warm-start and subsequent training pools, in balanced and natural-frequency regimes. Balanced sampling takes 24 warm-start and 48 pool examples per class; natural sampling keeps the same total sizes. Both regimes use the same full official test set. SemEval uses the official cleaned test inputs joined to gold labels by identifier.

Training. All runs use LoRA rank 64 and scaling 128, two warm-start SFT epochs, and learning rates 10<sup>−5</sup> for SFT and $5 \times 1 0 ^ { - 6 }$ for GRPO. Compress uses six sampled responses per item; Probe uses 16, with chance level 1/K, excess threshold 0.06, and replenishment weights 4/2/1. Rewards check the gold class; malformed answers receive −0.5. FGVIrony runs three cycles and selects checkpoints on development Macro-F1. CPED and SemEval use one fixed cycle. Checkpoint selection is completed before test evaluation. Wrong-only replenishment samples greedy errors after Compress and matches CPR’s supervised output-token budget. Evaluation uses greedy decoding with thinking disabled.

Table 5 in §5.6 reports the three-seed means. CPR exceeds GRPO by 2.33, 3.03, 7.50, and 4.10 points across the four settings, and also exceeds wrong-only replenishment. Prioritizing recoverable errors is more effective here than sampling all greedy errors indiscriminately.

<table><tr><td>Model</td><td>Orig.</td><td>A</td><td>B Mean</td><td>∆</td></tr><tr><td>Base</td><td>36.22</td><td>32.09</td><td>32.87 32.48</td><td>-3.74</td></tr><tr><td>CPR</td><td>63.39</td><td>64.17</td><td>63.98 64.07</td><td>+0.69</td></tr><tr><td>DeepSeek V4 Pro</td><td>61.81</td><td>56.69</td><td>59.84 58.27</td><td>-3.54</td></tr><tr><td>Qwen3.8-Max</td><td>67.91</td><td>58.07</td><td>58.27 58.17</td><td>-9.74</td></tr><tr><td>Qwen3.8-27B</td><td>56.50</td><td>52.36</td><td>52.95 52.66</td><td>-3.84</td></tr></table>

Table G.3: Accuracy under two character-renaming maps on the same 508 items. Changes are percentage points.

## G.4 Character Renaming

Construction and checks. We exclude 20 pilot scenes from the 585 public MCQs, leaving 525 candidates. Two consistent pseudonym maps replace names, aliases, and name-bearing titles throughout the inputs while retaining rank, relationships, dialogue, and option meanings. Programmatic checks verify reversible replacement. Kimi K3 and GLM-5.3 review semantic consistency without access to gold answers. Flagged replacements are corrected and rechecked; a coding assistant checks remaining flags against the source text. Seventeen unresolved or name-dependent items are excluded, yielding 508 items from 170 scenes. The final set is frozen before evaluation.

Evaluation. We evaluate the original and both renamed versions with thinking disabled, temperature zero, a 1,024-token output cap, and three option permutations. Accuracy uses the same strict JSON parser and majority aggregation for all versions. Table G.3 reports Base, CPR, DeepSeek V4 Pro, Qwen3.8-Max, and Qwen3.8-27B; ∆ is the mean of A and B minus the original score. Dialogue and plot information remain available, so the intervention measures sensitivity to character names.

CPR retains its aggregate accuracy under both mappings, whereas Base and the three larger comparison models lose 3.54–9.74 points on average. Qwen3.8-Max shows the largest name sensitivity in this comparison.