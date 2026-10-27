# RubriX: A Fast, Fine-Tunable Non-Autoregressive Rubric Judge

Balamurugan Paramasivam CobuildX balamurugan.p@cobuildx.ai

Sudalai Rajkumar CobuildX srk@cobuildx.ai

## Abstract

Rubric-based evaluation asks which level of a rubric a response meets. It is usually done by prompting a large language model (LLM) as a judge, which needs a generative call per judgment and reads every level description, so its cost grows with the rubric. We present RubriX, a rubric judge whose architecture is inspired by contrastive language models (CLM): a LoRA-adapted encoder embeds the input and each level description, and a softmax over scaled cosine similarities gives a probability for every level without text generation. Trained on public rubric data and proprietary synthetic data, RubriX reaches 75.4% exact accuracy on held-out rubrics, against 57.1% for TypeSafe’s Jev and 49.8% for a Qwen3-8B judge. On eight human-labelled benchmark sets it is within 6 percentage points of Jev’s accuracy on most of them, while Jev keeps a larger lead on two and ranks responses best on average; against the results reported in the G-Eval paper, RubriX outperforms G-Eval with GPT-3.5 on every SummEval dimension and both QAGS subsets. With level embeddings cached, RubriX’s latency stays at about 28 ms per judgment as level descriptions grow from 10 to 3,000 tokens, while Jev’s (approximate network time subtracted) rises from 44 ms to 211 ms and the Qwen3-8B judge’s from 32 ms to 516 ms, making RubriX 7.7× faster than Jev and 18.8× faster than the Qwen3-8B judge at 3,000 tokens per level.

## 1 Introduction

Evaluating AI systems increasingly comes down to one repeated decision: given a response and a rubric, which level of the rubric does the response meet? This is usually done by prompting a large language model as a judge [29], as in the rubric-based metrics of Ragas [10, 23] and the G-Eval metric [18] implemented in DeepEval [5]. These approaches are flexible, since any criterion can be written as text, but every judgment needs a full generative LLM call over the whole rubric, which makes them slow and costly when applied to every response and to long rubrics. Their scores also depend on prompt wording, decoding settings and output parsing.

RubriX takes a diferent approach, with an architecture inspired by contrastive language models (CLM) [2]. The response and each rubric level are embedded with a language-model encoder, and small trained heads score how well the response matches each level. A rubric is scored without text generation, returning a probability for every level, and the level embeddings can be computed once per rubric and reused for every response. Our contributions are:

• A non-autoregressive rubric judge whose cost does not grow with the rubric. RubriX scores a response against a rubric’s own level descriptions with a bi-encoder: one LoRA-adapted encoder embeds the input and each level separately, so a rubric of any size is scored without generation, and its level embeddings are computed once and reused for every response it judges.

• Ordinal training for a CLM-inspired architecture. The original CLM treats options as unordered. We train with cross-entropy over each rubric’s own levels plus a ranked probability score whose weight adapts to the running magnitudes of the two losses, so the ordinal term keeps about an equal share of the objective throughout training without a hand-tuned weight. Weighting losses adaptively from their running statistics is established in multi-task learning [4, 12]; we apply a simple form of it to the ordinal term.

• Training data. RubriX is trained on a criterion-capped mix of public rubric datasets and a proprietary dataset.

• Evaluation and latency. We compare RubriX with Jev [25], a Qwen3-8B judge [21] and the published G-Eval results [18] on held-out rubrics and eight human-labelled benchmark sets, by accuracy and by ranking correlation, and show that with cached level embeddings its latency stays flat as rubrics gain levels and longer level descriptions, while judges that read the levels slow down.

The RubriX-8B model is open source and available on Hugging Face at https://huggingface.   
co/CobuildX-AI/RubriX-8B, together with code to load it and score rubrics.

## 2 Related Work

LLM-as-a-judge. Strong LLMs agree well with human preferences when prompted to evaluate responses [29], which has made them the default evaluator for open-ended generation. They are, however, sensitive to prompt format and position, and every judgment is a generative call. A rubric judge of this kind also places the level descriptions inside its prompt, so every judgment reprocesses the whole rubric: its cost grows with the number of levels and with the length of each description, and the rubric must fit in the model’s context window together with the input. In our measurements (Section 5.4), a Qwen3-8B judge with a 32k context holds at most 20 levels of 1,500 tokens each.

Rubric-based evaluation in Ragas. Ragas [10] provides a rubric-based scoring metric for LLM outputs [23]. The user defines a rubric as a description for each score level, typically 1 to 5. An LLM judge reads the input, the response and optionally a reference, then returns the score whose description fits best, along with written feedback. Ragas also supports instance-specific rubrics, where each example carries its own level descriptions.

G-Eval. G-Eval [18] uses an LLM with chain-of-thought to evaluate generated text against user-defined criteria. The model first expands the criteria into evaluation steps, then scores the output in a form-filling step. Instead of taking the single generated score, G-Eval weights each possible score by the probability the model assigns to it, which gives finer-grained scores with better alignment to human judgments. DeepEval [5] implements this approach as its GEval metric for custom criteria. We compare with G-Eval’s published results on SummEval and QAGS.

Jev. Jev from TypeSafe [25] is a hosted model built for decisions rather than generation. Given a context and a set of options, it returns a probability for each option in a single pass, with no text to generate or parse. Rubric scoring maps onto it directly: the criterion becomes the question and the level descriptions become ordered options. Jev is available as a hosted API; its weights are not released, and a score question accepts at most 10 options.

Contrastive scoring. Contrastive objectives such as InfoNCE [20] and CLIP [22] learn a shared embedding space in which matching pairs score highly, using a temperature-scaled cosine similarity. CLM [2] applies this idea to choice tasks with a frozen LLM encoder and trained projection heads. RubriX’s architecture is inspired by CLM; it adapts the encoder and trains with an ordinal objective for rubric scoring.

Summary. Ragas and G-Eval treat evaluation as a generation task, with the cost and sensitivity that implies. Jev shows that scoring the rubric’s options directly is practical; its weights are not released. This leaves room for a rubric scorer that scores options directly and that can also be trained and adapted openly.

## 3 Method

![](images/d08e09befb7f658dddc55fbb3aa128bba1ac40054fd2521f49921cdc93aa30a8.jpg)  
Figure 1: The RubriX scorer. The input and each level description are encoded separately by the same encoder, projected by two heads, and compared by temperature-scaled cosine similarity. A softmax over the rubric’s own K levels gives a probability for each level. Level embeddings do not depend on the input, so they can be computed once per rubric and cached.

## 3.1 Scoring model

RubriX scores a response against a rubric with a contrastive approach (Figure 1). The input x (the conversation, the response to evaluate and the criterion) and each rubric level description $\ell _ { k }$ are embedded separately by a language-model encoder, Qwen3-8B [21], with no text generation. Each text’s embedding is its final hidden state, the last layer’s hidden state at the last token, L2- normalised. The same encoder, with one shared set of adapters (Section 3.3), embeds both. Two projection heads, f for inputs and g for levels, each a three-layer MLP (4096 →1536→1536→512 with LayerNorm and GELU), map the embeddings into a shared space. For a rubric with K levels, the score of level k is

$$
s _ { k } = \tau \cdot \cos { \left( f ( x ) , g ( \ell _ { k } ) \right) } ,\tag{1}
$$

and the predicted probability of level k is

$$
p _ { k } = \frac { \exp ( s _ { k } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } ) } .\tag{2}
$$

The temperature $\tau = 1 0 0$ is inherited from CLM, where it is a learned parameter capped at 100. Each example competes only against the levels of its own rubric, never against other rubrics’ levels.

## 3.2 Training objective

For a gold level $y ,$ the primary objective is cross-entropy over the rubric’s levels:

$$
\mathcal { L } _ { \mathrm { C E } } = - \log p _ { y } .\tag{3}
$$

Cross-entropy treats the levels as unordered: predicting a level two steps away costs the same as predicting a neighbouring one. We add the ranked probability score (RPS) [9], which compares the predicted cumulative distribution with the gold one:

$$
\mathcal { L } _ { \mathrm { R P S } } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K - 1 } \Big ( P _ { k } - \mathbf { 1 } [ y \leq k ] \Big ) ^ { 2 } , \qquad P _ { k } = \sum _ { j = 1 } ^ { k } p _ { j } .\tag{4}
$$

Here $P _ { k }$ is the predicted probability that the level is at most k. A prediction one level from the gold level costs less than one several levels away. RPS is much smaller than cross-entropy in magnitude, so a fixed weight lets it contribute little. We therefore weight it adaptively, so that on average it contributes as much as cross-entropy:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { t } \mathcal { L } _ { \mathrm { R P S } } , \qquad \lambda _ { t } = \alpha \frac { \overline { { \mathcal { L } } } _ { \mathrm { C E } } } { \overline { { \mathcal { L } } } _ { \mathrm { R P S } } } ,\tag{5}
$$

where the bars are exponential moving averages (decay 0.99) of the two losses over micro-batches, computed from detached values so that no gradient flows through $\lambda _ { t } ,$ seeded with the first micro-batch and capped at 1000; we use $\alpha = 1$ . This is a simple instance of adaptive loss balancing [4, 12]: instead of a fixed weight, each term is scaled by its own running magnitude.

## 3.3 Encoder adaptation

Low-rank adapters [13] are added to all attention and MLP projections of the encoder and trained jointly with the heads, so that the encoder itself can learn features that separate adjacent levels. This trains 43.6M adapter parameters (about 0.5% of the encoder) and 18.9M head parameters. The heads start from the released CLM-8B heads [2]. For serving, the adapters are merged into the encoder weights, so inference has the cost of the base encoder.

## 4 Experimental Setup

## 4.1 Training data

The training set (Table 1) has 34,261 examples covering 2,284 distinct criteria. It combines public rubric-labelled data with a proprietary dataset. The proprietary dataset contains 8,039 rubric-labelled examples over 324 criteria, covering long-form answers to expert questions, multi-turn conversations, and other open-ended writing tasks. It was generated with a large language model and filtered by blind re-grading: an example was kept only when an independent grader, without seeing the intended level, assigned exactly that level. The public part is drawn from seven sources and capped at 25 examples per criterion, balanced across levels within each criterion, so that sources with a handful of fixed criteria (HelpSteer’s attributes, for example) contribute few examples and the many custom rubrics of the Feedback Collection [15] and the principles of HelpSteer3 [19] contribute most. The test set is split from training by criterion: no criterion or level description of the held-out test set appears in training, and some criteria (HelpSteer’s complexity and HelpSteer2’s verbosity, for example) are held out entirely. The validation data, used only to select the checkpoint, partly shares criteria with training.

Table 1: Training data. Public sources are capped at 25 examples per criterion, level-balanced.
<table><tr><td>Source</td><td>Criteria</td><td>Levels</td><td>Examples</td></tr><tr><td>Feedback Collection [15]</td><td>898</td><td>5</td><td>22,450</td></tr><tr><td>HelpSteer3, principles [19]</td><td>1,046</td><td>2</td><td>3,372</td></tr><tr><td>HelpSteer, HelpSteer2 [27, 28]</td><td>8</td><td>5</td><td>200</td></tr><tr><td>HelpSteer3, feedback [19]</td><td></td><td>3 5</td><td>75</td></tr><tr><td>UltraFeedback [8]</td><td></td><td>4 5</td><td>100</td></tr><tr><td>ASAP 2.0 [7]</td><td>1</td><td>6</td><td>25</td></tr><tr><td>Proprietary dataset</td><td>324</td><td>2-5</td><td>8,039</td></tr><tr><td>Total</td><td>2,284</td><td></td><td>34,261</td></tr></table>

## 4.2 Benchmarks

We evaluate on six human-labelled benchmarks (Table 2). JudgmentBench scores each answer with pass/fail checklist items and with penalty items that count errors; we report these separately as JudgmentBench, binary and JudgmentBench, multi-level, with the counts grouped into four levels (none, one, a few, many). QAGS is reported per subset. Rubric text is kept verbatim where the benchmark provides it. Where a benchmark has no level descriptions (HealthBench and the JudgmentBench checklist items are met/not-met, JudgmentBench’s penalty items are counts, SummEval has 1–5 ratings and QAGS a yes/no question), short generic level descriptions were added, the same for every judge, since there is no other way to pose those items as a choice between levels. For QAGS, the criterion is the question its workers answered, asked of the whole summary: “Is the summary factually supported by the article?”.

Table 2: Evaluation benchmarks.
<table><tr><td>Benchmark</td><td>What is judged</td><td>Labels from</td><td>Judgments</td></tr><tr><td>HealthBench [1]</td><td>health conversations, consensus criteria (met / not met)</td><td>2–5 physicians</td><td>29,511</td></tr><tr><td>BiGGen [16]</td><td>responses with instance-specific rubrics, nine capabilities (5 levels)</td><td>human raters</td><td>3,194</td></tr><tr><td>ELLIPSE [6]</td><td>learner essays, overall proficiency and six two trained raters traits (5 levels)</td><td></td><td>45,374</td></tr><tr><td>SummEval [11]</td><td>news summaries, four dimensions (5 levels) three experts</td><td></td><td>6,400</td></tr><tr><td>JudgmentBench, binary [14]</td><td>legal answers, checklist items (met / not met)</td><td>practising lawyers</td><td>17,365</td></tr><tr><td></td><td>JudgmentBench, multi-level [14] legal answers, counted errors (4 levels)</td><td>practising lawyers</td><td>2,782</td></tr><tr><td>QAGS-CNN [26]</td><td>multi-sentence model summaries of CNN/DailyMail articles (supported / not)</td><td>three crowd workers</td><td>235</td></tr><tr><td>QAGS-XSUM [26]</td><td>one-sentence model summaries of BBC articles (supported / not)</td><td>three crowd workers</td><td>239</td></tr></table>

## 4.3 Baselines

Jev [25] receives one score question per judgment: the rubric’s criterion is the question’s instructions, and its level descriptions, lowest first, are the question’s criteria. Jev returns a probability for each criterion, which we use as its distribution over levels. Qwen3-8B [21], the same base model as RubriX’s encoder, is used as a zero-shot judge with thinking disabled: the levels are listed lowest first and lettered A, B, C, . . . , and the level distribution is the softmax of the first generated token’s log-probabilities restricted to those letters. G-Eval [18] results are copied from its paper for the two benchmarks it shares with us, SummEval (its Table 1) and QAGS (its Table 3), with GPT-4 (G-Eval-4) and GPT-3.5 (G-Eval-3.5).

## 4.4 Metrics

Exact accuracy is the share of judgments whose most likely level equals the gold level, defined as the level nearest to the raters’ mean score, with exact halves assigned to the lower level (on the held-out set, the dataset’s own label). For QAGS, whose human score is the share of a summary’s sentences that the workers’ majority judged supported, a summary is labelled supported only if every one of its sentences is, and unsupported if even one is not. Within-one accuracy counts a prediction at most one level away as correct; it is reported for multi-level benchmarks only, since on two levels it is always 100%. Every judge is scored on the same judgments.

Ranking correlation: we report summary-level Spearman ρ and Kendall τ [3]. Judgments are grouped by input and criterion, for example one SummEval article’s 16 summaries for one dimension; within each group the judge’s scores are correlated with the human scores (Kendall τ as τ-b, which handles ties), so the question is whether a judge orders the responses to the same input the way people do. Group correlations are averaged within each criterion, then within each category, and the reported AVG is the mean over a benchmark’s categories. A judge’s score is its probability-weighted level, $\sum _ { k } p _ { k } k .$ , the same score G-Eval uses (Jev’s probabilities, returned to two decimals, are renormalised to sum to one). Groups with fewer than three responses, or where the human score or the judge’s score never changes, are skipped. QAGS has one summary per article, so, as in G-Eval, each subset is one correlation over all its summaries.

## 4.5 Implementation details

The encoder is Qwen3-8B in bf16 on one H100 80GB GPU, with gradient checkpointing. LoRA uses rank 16, $\alpha = 3 2$ and dropout 0.05 on all seven attention and MLP projections. We train with AdamW (no weight decay; learning rates $1 0 ^ { - 4 }$ for the adapters and $5 \times 1 0 ^ { - 4 }$ for the heads), micro-batches of 8 with 4 accumulation steps (an efective batch of 32), 5% linear warm-up then linear decay, and gradient clipping at 1.0, for one epoch of 1,071 steps. The checkpoint with the lowest log-loss on a fixed sample of the validation set was kept (step 1,068). Training took 10,212 seconds (2.8 hours) over 32.4M tokens. For evaluation, the adapters are merged into the encoder, which is served in bf16 with vLLM [17].

## 5 Results

## 5.1 Held-out rubrics

The test set held out from the training mix contains rubrics whose criteria and level descriptions never appear in training: 6,749 judgments, from the Feedback Collection (4,999), the proprietary dataset (1,172), HelpSteer3 principles (378), HelpSteer (100) and HelpSteer2 (100). RubriX is far ahead (Figure 2): 75.4% exact accuracy against 57.1% for Jev and 49.8% for the Qwen3-8B judge, and 97.6% within one level against 89.6% and 88.4%. These rubrics come from the same sources as the training data, so this result measures generalisation to new criteria of a familiar kind.

![](images/aea776dc6790078daa969b2e1662515b3aa4883fbd6e3f868d7d847ae430ed55.jpg)  
Figure 2: Held-out test set: 6,749 judgments on rubrics whose criteria and level descriptions never appear in training.

## 5.2 Human-labelled benchmarks: accuracy

Figures 3 and 4 report accuracy on the benchmarks. Jev has the highest exact accuracy on six of the eight sets (HealthBench, BiGGen, SummEval, JudgmentBench binary and both QAGS subsets), but RubriX is close behind on most of them: by 1.7 points on QAGS-CNN, 3.0 on QAGS-XSUM, 4.8 on HealthBench, 5.2 on BiGGen and 1.2 on ELLIPSE, where the Qwen3-8B judge leads. The larger gaps are SummEval (11.6 points) and JudgmentBench binary (14.1 points). The Qwen3-8B judge is the most accurate on ELLIPSE and JudgmentBench multi-level, where RubriX is ahead of Jev. RubriX is close to Jev on QAGS (80.4% against 82.1% on CNN, 75.7% against 78.7% on XSUM) and has the best within-one accuracy on JudgmentBench multi-level (96.4%), where Jev reaches 42.4%.

![](images/5f8692926330692ecea4e1d95d55a8c502f53e1bd1d0ebff019dfdadff6a4dfa.jpg)  
Figure 3: Exact accuracy on the human-labelled benchmarks.

![](images/eb0b59aac4ed619181f8087f9766ac434bf7527794ed8112c263429dd69f70e9.jpg)  
Figure 4: Within-one accuracy on the multi-level benchmarks.

## 5.3 Human-labelled benchmarks: ranking correlation

These correlations answer a diferent question from accuracy: given several responses to the same input, does the judge put them in the same order as people do, even when it does not land on the exact level? Table 3 reports Spearman $\rho$ and Kendall τ per category. Jev has the highest average correlation on five of the six benchmarks; on BiGGen the Qwen3-8B judge is level with it (0.527 against 0.526 in $\rho )$

Against G-Eval, on the two benchmarks it reports:

• SummEval: RubriX averages $\rho = 0 . 4 8 3$ , above G-Eval-3.5 (0.401) and below G-Eval-4 (0.514); on coherence it matches G-Eval-4 (0.581 against 0.582). Jev averages 0.541, above G-Eval-4, and is best on three of the four dimensions; G-Eval-4 is best on relevance.

• QAGS: RubriX has the highest Spearman correlation on QAGS-CNN (0.719, against 0.715 for Jev and 0.685 for G-Eval-4), but is lower on QAGS-XSUM (0.521, against 0.537 for G-Eval-4 and 0.653 for Jev), whose one-sentence summaries are more abstractive. Averaged over both subsets, RubriX (0.620) is level with G-Eval-4 (0.611) and the Qwen3-8B judge (0.616), and Jev (0.684) is ahead.

RubriX is above G-Eval-3.5 in every SummEval dimension and both QAGS subsets, in both $\rho$ and τ .

On ELLIPSE and HealthBench, RubriX ranks second, between Jev and the Qwen3-8B judge. On BiGGen, where each group holds only about four responses to one instance, it is third. On JudgmentBench, every judge is close to zero $( \rho \le 0 . 0 6 8$ on average): ordering the answers to the same legal task by their checklist and error items is not something any of these judges does reliably.

## 5.4 Latency and rubric size

A judge that reads the rubric inside its input pays for every level on every judgment. RubriX embeds each level description once per rubric, so a judgment on a rubric it has seen needs only the input embedded. We measured latency one request at a time, on dummy text so that token counts are exact: an 800-token response, a 30-token criterion, and level texts of a fixed length, with 110 judgments per setting (30 for the cache miss at 50 and 100 levels, whose rubrics hold 75,000 and 150,000 tokens) and a diferent response for every judgment. Two settings were varied: the number of levels, from 2 to 100 levels of 1,500 tokens each, and the length of each level, from 10 to 3,000 tokens with 5 levels. RubriX (the evaluated model’s weights, adapters merged) and the Qwen3-8B judge ran on one H100 80GB with vLLM, with vLLM’s prefix cache of for both; Jev was called through its API.

Figures 5 and 6 show the result. With its level embeddings cached, RubriX takes 27.5–28.3 ms per judgment in every setting, up to a 100-level rubric of 150,000 tokens. The Qwen3-8B judge grows from 108.5 ms at 2 levels to 1,241.3 ms at 20 levels of 1,500 tokens, the most its 32k context can hold, and from 31.7 ms to 516.2 ms as levels grow from 10 to 3,000 tokens. Jev grows from 73.1 ms to 199.7 ms between 2 and 10 levels, the maximum its score questions accept, and from 44.0 ms to 210.9 ms with level length. A cache miss, when RubriX meets a new rubric and embeds its levels in the same request, costs about as much as the Qwen3-8B judge: 108.2 ms at 2 levels, 4,090.2 ms for a 100-level, 150,000-token rubric, and 422.6 ms at 5 levels of 3,000 tokens. It happens once per rubric.

Training cost. RubriX trained 62.5M parameters (43.6M adapter and 18.9M head parameters) for one epoch in 2.8 hours on one H100. At inference, each judgment needs one encoder forward pass for its input (the conversation, the response and the criterion). Each level description is embedded in its own pass only once per rubric, and the vectors are reused for every response judged on that rubric; no text is generated.

Table 3: Ranking correlation with human scores: summary-level Spearman ρ and Kendall τ within groups of responses to the same input, averaged per category; AVG is the mean over a benchmark’s categories. G-Eval values are copied from its paper (Table 1 for SummEval, Table 3 for QAGS); it does not report the other benchmarks. Bold marks the best judge in each row and underline the second best, separately for ρ and τ. A group is one article’s summaries for one dimension (SummEval), the essays written to one prompt for one trait (ELLIPSE), one conversation’s replies for one criterion (HealthBench), one instance’s responses (BiGGen) or the answers to one task for one rubric item (JudgmentBench); for QAGS each subset is one group.
<table><tr><td></td><td></td><td colspan="2">RubriX</td><td colspan="2">Jev</td><td colspan="2">Qwen3-8B judge</td><td colspan="2">G-Eval-4</td><td colspan="2">G-Eval-3.5</td></tr><tr><td>Benchmark</td><td>Category</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td></tr><tr><td>SummEval</td><td>Coherence</td><td>0.581</td><td>0.450</td><td>0.649</td><td>0.510</td><td>0.432</td><td>0.330</td><td>0.582</td><td>0.457</td><td>0.440</td><td>0.335</td></tr><tr><td></td><td>Consistency</td><td>0.496</td><td>0.413</td><td>0.559</td><td>0.473</td><td>0.458</td><td>0.377</td><td>0.507</td><td>0.425</td><td>0.386</td><td>0.318</td></tr><tr><td></td><td>Fluency</td><td>0.435</td><td>0.360</td><td>0.484</td><td>0.402</td><td>0.461</td><td>0.381</td><td>0.455</td><td>0.378</td><td>0.424</td><td>0.347</td></tr><tr><td></td><td>Relevance</td><td>0.422</td><td>0.328</td><td>0.474</td><td>0.370</td><td>0.360</td><td>0.280</td><td>0.547</td><td>0.433</td><td>0.385</td><td>0.293</td></tr><tr><td></td><td>AVG</td><td>0.483</td><td>0.388</td><td>0.541</td><td>0.439</td><td>0.428</td><td>0.342</td><td>0.514</td><td>0.418</td><td>0.401</td><td>0.320</td></tr><tr><td>QAGS</td><td>QAGS-CNN</td><td>0.719</td><td>0.583</td><td>0.715</td><td>0.579</td><td>0.656</td><td>0.518</td><td>0.685</td><td>0.591</td><td>0.516</td><td>0.410</td></tr><tr><td></td><td>QAGS-XSUM</td><td>0.521</td><td>0.427</td><td>0.653</td><td>0.547</td><td>0.576</td><td>0.472</td><td>0.537</td><td>0.472</td><td>0.406</td><td>0.343</td></tr><tr><td></td><td>ÁVG</td><td>0.620</td><td>0.505</td><td>0.684</td><td>0.563</td><td>0.616</td><td>0.495</td><td>0.611</td><td>0.525</td><td>0.461</td><td>0.377</td></tr><tr><td>ELLIPSE</td><td>Overall</td><td>0.589</td><td>0.463</td><td>0.651</td><td>0.520</td><td>0.520</td><td>0.403</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Cohesion</td><td>0.510</td><td>0.395</td><td>0.579</td><td>0.453</td><td>0.475</td><td>0.364</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Syntax</td><td>0.516</td><td>0.399</td><td>0.590</td><td>0.462</td><td>0.494</td><td>0.379</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Vocabulary</td><td>0.516</td><td>0.402</td><td>0.569</td><td>0.447</td><td>0.520</td><td>0.406</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Phraseology</td><td>0.472</td><td>0.362</td><td>0.566</td><td>0.441</td><td>0.467</td><td>0.358</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Grammar</td><td>0.532</td><td>0.410</td><td>0.615</td><td>0.482</td><td>0.424</td><td>0.324</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Conventions</td><td>0.470</td><td>0.360</td><td>0.636</td><td>0.500</td><td>0.497</td><td>0.380</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AVG</td><td>0.515</td><td>0.399</td><td>0.601</td><td>0.472</td><td>0.485</td><td>0.373</td><td></td><td></td><td></td><td></td></tr><tr><td>HealthBench</td><td>Communication</td><td>0.286</td><td>0.259</td><td>0.361</td><td>0.332</td><td>0.226</td><td>0.204</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Complex responses</td><td>0.131</td><td>0.120</td><td>0.162</td><td>0.149</td><td>0.155</td><td>0.139</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Context seeking</td><td>0.484</td><td>0.437</td><td>0.565</td><td>0.518</td><td>0.317</td><td>0.285</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Emergency referrals</td><td>0.429</td><td>0.388</td><td>0.493</td><td>0.453</td><td>0.357</td><td>0.322</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Global health</td><td>0.438</td><td>0.398</td><td>0.436</td><td>0.398</td><td>0.340</td><td>0.306</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Health data tasks</td><td>0.314</td><td>0.286</td><td>0.584</td><td>0.535</td><td>0.355</td><td>0.323</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Hedging</td><td>0.154</td><td>0.140</td><td>0.483</td><td>0.447</td><td>0.160</td><td>0.147</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AVG</td><td>0.320</td><td>0.290</td><td>0.441</td><td>0.405</td><td>0.273</td><td>0.246</td><td></td><td></td><td></td><td></td></tr><tr><td>BiGGen</td><td>Grounding</td><td>0.549</td><td>0.513</td><td>0.643</td><td>0.598</td><td>0.632</td><td>0.577</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Instruction following</td><td>0.405</td><td>0.359</td><td>0.560</td><td>0.507</td><td>0.550</td><td>0.492</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Multilingual</td><td>0.418</td><td>0.342</td><td>0.427</td><td>0.365</td><td>0.474</td><td>0.408</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Planning</td><td>0.507</td><td>0.444</td><td>0.539</td><td>0.489</td><td>0.580</td><td>0.512</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Reasoning</td><td>0.453</td><td>0.408</td><td>0.533</td><td>0.496</td><td>0.542</td><td>0.498</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Refinement</td><td>0.363</td><td>0.320</td><td>0.577</td><td>0.513</td><td>0.514</td><td>0.468</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Safety</td><td>0.362</td><td>0.325</td><td>0.527</td><td>0.476</td><td>0.545</td><td>0.478</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Theory of mind</td><td>0.404</td><td>0.362</td><td>0.458</td><td>0.415</td><td>0.438</td><td>0.390</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Tool usage AVG</td><td>0.442 0.434</td><td>0.394 0.385</td><td>0.465 0.526</td><td>0.414 0.475</td><td>0.466 0.527</td><td>0.424 0.472</td><td></td><td></td><td></td></table>

## 6 Limitations

• Separate encoding. The input and the level descriptions are encoded separately and meet only at the final similarity, so the model never reads a response alongside a level description. This limits verdicts that depend on checking a specific fact against the rubric.

• Model capacity. RubriX uses an 8B encoder with small projection heads (18.9M parameters), which likely limits how far it generalises beyond the rubric styles it was trained on; we have not trained larger encoders to test this. Continuing training for a second epoch, from the final model on a partly refreshed data mix, raised validation log-loss at every check, so training stopped early and the first-epoch model was kept: with this encoder and head size, more training on this data does not help.

![](images/187e60ea7160d344c0c6135211ad4720086b4d9b666c4dbe1c58dc2ff2c92e86.jpg)  
Figure 5: Median latency per judgment, one request at a time, against the number of levels (1,500 tokens each); labels give the median in ms. Latency axis on a log scale. Jev’s line has an approximate network time subtracted: 232.6 ms, the median time, over 50 fresh connections from our client, from the end of the connection handshake to the first byte of the reply to an empty request to its API. RubriX and the Qwen3-8B judge were timed on the H100 itself and show processing only. Jev accepts at most 10 levels and the Qwen3-8B judge’s 32k context holds at most 20 levels of this length.

![](images/3bcdf7069c5376fa8db177c85fee40e20af9cd3f1c835d715ef8caefef52101b.jpg)  
Figure 6: Median latency per judgment against the length of each level description (5 levels), as in Figure 5.

• Dependence on fine-tuning data. RubriX responds strongly to fine-tuning: on held-out rubrics from its training sources it leads Jev by 18.3 percentage points in exact accuracy, but on human-labelled benchmarks of other formats it trails Jev on most sets. It is therefore most useful when it can be fine-tuned on rubrics like the ones it will be applied to; on unfamiliar rubric styles, without such data, a general-purpose judge such as Jev is stronger.

## 7 Future Work

• Larger encoders. Training RubriX on bigger base models, to measure how much of the gap to Jev on unfamiliar rubric styles is a matter of model size.

• More training data. Scaling the training data, in size and in the variety of criteria and formats, and analysing how accuracy and ranking on held-out rubrics and human-labelled benchmarks change with it.

• Domain-specific fine-tuning. Fine-tuning on data from a particular use case and comparing how the model then performs on that domain’s rubrics against general rubrics.

• Rubrics with hundreds of levels. Fine-tuning for scoring tasks that need very many levels. RubriX’s cost does not grow with the number of levels, while an LLM judge must fit every level description in its context, which at this scale needs long-context models, for example with extended rotary position embeddings (RoPE) [24]; we will compare the two in accuracy and eficiency.

## 8 Conclusion

RubriX, whose architecture is inspired by contrastive language models, scores rubrics with a LoRA-adapted encoder, returning a probability for every level without generation. On held-out rubrics it is well ahead of Jev and a Qwen3-8B judge. On human-labelled benchmarks it is competitive: it is within 6 percentage points of Jev’s accuracy on most sets, Jev keeps a larger lead on SummEval and JudgmentBench binary and ranks best on average, while RubriX leads on QAGS-CNN and matches G-Eval with GPT-4 on SummEval coherence. Its clearest advantage is cost: with cached level embeddings its latency does not depend on the size of the rubric, so long, many-level rubrics cost no more to apply than short ones. RubriX is therefore best suited to settings where a few labelled examples from the target domain are available to fine-tune on: trained with at most 25 examples per criterion, it beats Jev and the Qwen3-8B judge by a wide margin on held-out rubrics from the same sources (75.4% exact accuracy against 57.1% and 49.8%), its fine-tuning took 2.8 hours on a single H100, and each judgment then costs one encoder pass.

## References

[1] R. K. Arora et al. HealthBench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

[2] Contrastive-LM. CLM: Contrastive language models, and the CLM-v0.1-8B checkpoint. https: //github.com/Contrastive-LM/CLM.

[3] M. Bhandari, P. N. Gour, A. Ashfaq, P. Liu, and G. Neubig. Re-evaluating evaluation in text summarization. In Proceedings of EMNLP, pages 9347–9359, 2020.

[4] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings of ICML, 2018.

[5] Confident AI. DeepEval: The LLM evaluation framework. https://github.com/confident-ai/ deepeval.

[6] S. A. Crossley et al. The English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) corpus. International Journal of Learner Corpus Research, 9(2):248–269, 2023.

[7] ASAP 2.0. Automated Student Assessment Prize 2.0 essay corpus. https://github.com/ scrosseye/ASAP\_2.0.

[8] G. Cui et al. UltraFeedback: Boosting language models with scaled AI feedback. In Proceedings of ICML, 2024.

[9] E. S. Epstein. A scoring system for probability forecasts of ranked categories. Journal of Applied Meteorology, 8(6):985–987, 1969.

[10] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert. RAGAs: Automated evaluation of retrieval augmented generation. In Proceedings of EACL: System Demonstrations, 2024.

[11] A. R. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher, and D. Radev. SummEval: Reevaluating summarization evaluation. Transactions of the Association for Computational Linguistics, 9:391–409, 2021.

[12] R. Groenendijk, S. Karaoglu, T. Gevers, and T. Mensink. Multi-loss weighting with coeficient of variations. In Proceedings of WACV, pages 1469–1478, 2021.

[13] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In Proceedings of ICLR, 2022.

[14] JudgmentBench. JudgmentBench: legal tasks with rubric scores from practising lawyers. https: //huggingface.co/datasets/judgmentbench/JudgmentBench, 2026.

[15] S. Kim et al. Prometheus: Inducing fine-grained evaluation capability in language models. In Proceedings of ICLR, 2024.

[16] S. Kim et al. The BiGGen Bench: A principled benchmark for fine-grained evaluation of language models with language models. In Proceedings of NAACL, 2025. arXiv:2406.05761.

[17] W. Kwon et al. Eficient memory management for large language model serving with PagedAttention. In Proceedings of SOSP, 2023.

[18] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu. G-Eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of EMNLP, 2023.

[19] NVIDIA. HelpSteer3 dataset. https://huggingface.co/datasets/nvidia/HelpSteer3, 2025.

[20] A. van den Oord, Y. Li, and O. Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[21] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[22] A. Radford et al. Learning transferable visual models from natural language supervision. In Proceedings of ICML, 2021.

[23] Ragas. Rubrics-based evaluation metrics, Ragas documentation. https://docs.ragas.io.

[24] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. arXiv:2104.09864.

[25] TypeSafe AI. Jev model documentation. https://docs.typesafe.ai/models.

[26] A. Wang, K. Cho, and M. Lewis. Asking and answering questions to evaluate the factual consistency of summaries. In Proceedings of ACL, 2020.

[27] Z. Wang et al. HelpSteer: Multi-attribute helpfulness dataset for SteerLM. In Proceedings of NAACL, 2024.

[28] Z. Wang et al. HelpSteer2: Open-source dataset for training top-performing reward models. In Advances in NeurIPS, Datasets and Benchmarks Track, 2024.

[29] L. Zheng et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in NeurIPS, Datasets and Benchmarks Track, 2023.