# Decide, Don’t Generate: Competitive Dimensional ABSA with Jev’s Typed Decisions

Yiqun Zhang Peidong Wang Zihan Wang Shi Feng<sup>\*</sup> Northeastern University, China

## Abstract

Aspect-based sentiment analysis (ABSA) has largely turned to text generation. We show that competitive dimensional ABSA does not need it. Using Jev, a frozen model that answers typed questions with rubric scores, label probabilities, and yes/no judgments, we decompose all three tasks of SemEval-2026 Task III Track A into such decisions and align them with the annotation scheme through 488 coefficients fitted on CPU, with no text generation and no backbone tuning. On valence–arousal regression over ten corpora in six languages, the system reaches 1.0645 RMSE, the lowest aggregate error of any participating system. On triplet and quadruplet extraction, it reaches 52.09 and 44.06 continuous F1, above fine-tuned Llama-3.3-70B and GPT-OSS-120B baselines. Analyses and ablations show where the accuracy comes from: supervised calibration roughly halves the raw regression error, exact valence–arousal would add only 4.5 F1 to extraction, and the learned combination of span-boundary evidence, not any single signal, carries the extraction systems.<sup>1</sup>

## 1 Introduction

Generating sentiment structures has become a prominent approach to aspect-based sentiment analysis (ABSA). Representative milestones include unified BART-based sequence prediction (Yan et al., 2021), T5-based paraphrase generation for aspect sentiment quadruples (Zhang et al., 2021), and multi-view prompting over output orders (Gou et al., 2023). In a title-filtered audit of the literature (Figure 1), the share of methods that use a text-generative model rises from 38% in 2021–2023 to about 65% in 2024–2025. This count includes auxiliary uses such as augmentation and scoring, not only generation of the final tuples. The trend raises a question: does competitive sentiment analysis require generating text?

![](images/c8bb3339ecaa09c18cebc85a186a42d5b783d9741c375b1a69a1b848a8e67fe0.jpg)  
Figure 1: Share of ABSA methods using a textgenerative model at any stage, by publication period (protocol in Appendix D).

Before this shift, most ABSA systems were discriminative: an encoder supplied representations for classification or structured selection, as in BERT (Devlin et al., 2019) and its sentence-pair formulation for ABSA (Sun et al., 2019). When the answer is a label, a span, or a number, direct prediction is a natural fit. Generation offers one output space for all of them, at the price of backbone adaptation, repeated decoding, and output validation.

TypeSafe’s System One framing gives a reason to revisit direct prediction. It presents Jev as a model for fast, structured decisions: given a state and a typed question, it returns an answer with probabilities (TypeSafe, 2026). This resembles the role of BERT-style classifiers, with a question interface in place of task-specific output heads. We call the interface discriminative without assuming anything about Jev’s architecture or pretraining objective.

Our testbed is dimensional ABSA. It extends feature-level opinion mining (Hu and Liu, 2004) and polarity-based ABSA (Pontiki et al., 2014, 2016) with continuous valence and arousal: how positive an evaluation is and how activated the expressed feeling is (Lee et al., 2026). SemEval-2026 Task III (DimABSA, officially numbered Task 3) defines three increasingly structured tasks in its Track A (Yu et al., 2026): given-aspect regression (DimASR), aspect–opinion triplet extraction (DimASTE), and categorized quadruplet prediction (DimASQP), which we call Tasks 1–3 (T1–T3 in tables). The latter two credit a tuple only when its spans match exactly, so they test structural prediction as well as numerical estimation.

![](images/05f6317c776af94e678dd9078d95b7d8265708e7fffc7eb522b910e33b80ec07.jpg)  
Figure 2: Pipeline for the three tasks. A frozen model (snowflake) answers every question with rubric scores, label probabilities, or yes/no judgments; “Fit” marks learned postprocessors, fitted on training data except the Task 2 reranker (development data). Task 1 calibrates the scores of a given aspect; Task 2 proposes, checks, and selects aspect–opinion pairs and scores their VA; Task 3 adds a category to each pair.

Figure 2 shows our system, which decides rather than generates. Training-corpus statistics provide lexical candidates, boundary conventions, demonstrations, and category priors. Jev provides rubric scores, token and category probabilities, and pair judgments. Ridge regression, logistic reranking, and a small fusion model align these decisions with the annotation scheme. No component generates text, as an intermediate or a final answer, and no backbone weights are updated; the system does use labeled data, through 488 coefficients fitted on CPU.

## Our contributions are:

• A decision-based pipeline for all three tasks that generates no text and tunes no backbone; its task adaptation is 488 coefficients fitted on CPU (Section 3).

• The lowest ten-corpus T1 aggregate of any participating system (1.0645 RMSE), and T2/T3 scores (52.09/44.06 cF1) above fine-tuned Llama-3.3-70B and GPT-OSS-120B baselines, over six languages and four domains (Section 4.2).

• Ablations and error analysis of where the accuracy comes from: calibration roughly halves the raw regression error, the learned combination of boundary evidence carries extraction (Section 4.3), and the remaining extraction error is structural: gold pairs are lost in proposing and in selecting spans, while adding categories costs us less than any leading system (Section 4.4).

## 2 Related Work

## 2.1 ABSA formulations

SemEval established aspect-level polarity prediction (Pontiki et al., 2014, 2016); DimABSA extends it to continuous valence–arousal ratings (Russell, 1980; Lee et al., 2026), evaluated jointly with exact tuple structure through continuous F1 (Yu et al., 2026). Discriminative approaches use BERT (Devlin et al., 2019), including sentencepair classification (Sun et al., 2019). Generative approaches serialize sentiment structures through unified BART prediction (Yan et al., 2021), T5 paraphrases (Zhang et al., 2021), or multi-view output-order prompting (Gou et al., 2023). We revisit direct prediction for all three dimensional tasks, constructing tuples through candidate selection rather than text decoding.

## 2.2 DimABSA systems and calibration

Published systems combine backbone adaptation (PAI, TeleAI, and PALI; Ruan et al., 2026; Zhou et al., 2026; Chen, 2026), retrieved demonstrations and ensembling (Takoyaki; Yamada et al., 2026), or repeated structured generation (nchellwig; Hellwig et al., 2026). TeamLasse separates generative extraction from encoder-based VA regression (Strothe, 2026); Table 1 summarizes each system’s adaptation. Our pipeline combines BM25-retrieved annotation examples (Robertson and Zaragoza, 2009) with Jev’s fixed SCORE, CHOICE, and NOUL interfaces (TypeSafe, 2026). Unlike contextual calibration, which estimates answer bias from contentfree inputs (Zhao et al., 2021), our VA calibration fits benchmark labels: its gains rely on supervision as well as prompting.

## 3 Method

## 3.1 Tasks and decision interface

Let x be a review, a an aspect, o an opinion, and c a category. Sentiment is a vector $y = ( v , r ) \in$ $[ 1 , 9 ] ^ { 2 }$ , where r denotes arousal to distinguish it from aspect a. Task 1 predicts y given $( x , a )$ . Task 2 predicts a set of $( a , o , y )$ tuples given x. Task 3 predicts $( a , c , o , y )$ tuples. Explicit terms are substrings of the input; an implicit aspect uses the benchmark sentinel NULL where permitted.

All three tasks use the same model through three operations. CHOICE returns a distribution over a supplied finite label inventory. NOUL returns a scalar judgment for a binary proposition. SCORE returns a distribution over ordered rubric levels and its expected zero-based index. For dimension d with nine levels, we map that score to the benchmark scale as

$$
z _ { d } = 1 + \sum _ { k = 0 } ^ { 8 } k p _ { d } ( k \mid x , a , o ) ,\tag{1}
$$

omitting o for Task 1. Valence levels run from negative to positive evaluation and arousal levels from calm to activated feeling. Each question asks about the target aspect rather than the tone of the whole review. Figure 2 shows how the operations compose; Appendix E provides the prompt templates and complete scoring criteria.

## 3.2 Task 1: given-aspect regression

Fixed demonstrations. Each corpus uses nine fixed training examples: the earliest eligible record in each of nine equal-width valence bands, with empty bands filled in file order. The state contains the review and these labeled examples; two SCORE questions per aspect give raw valence and arousal. A BM25-retrieved alternative was tried on development data and not adopted (Section 4.3).

Joint calibration. A corpus-specific regression maps the raw outputs $\boldsymbol { z } = \left( z _ { v } , z _ { r } \right)$ to benchmark labels. Define

$$
{ \phi } ( z ) = [ { z } _ { v } , { z } _ { r } , | { z } _ { v } - 5 | , { z } _ { v } { z } _ { r } ] ^ { \top } .\tag{2}
$$

After standardizing these features with training means $\mu$ and standard deviations $s ,$ the prediction is

$$
\hat { y } = \mathrm { c l i p } _ { [ 1 , 9 ] } \bigg ( W ^ { \top } \frac { \phi ( z ) - \mu } { s } + b \bigg ) .\tag{3}
$$

We fit $W \in \mathbb { R } ^ { 4 \times 2 }$ and $b \in \mathbb { R } ^ { 2 }$ by ridge regression with an unpenalized intercept. The extremity and interaction terms let predicted arousal depend on how extreme the valence is. Five-fold grouped cross-validation on a training sample selects the ridge penalty, and development data select the calibration family. The shrinkage variant in Table 3 fits each dimension independently as $\bar { y } _ { d } + \alpha _ { d } ( z _ { d } - \bar { z } _ { d } )$

## 3.3 Task 2: dimensional triplet extraction

Token decisions and candidate lattice. Deterministic tokenization separates Chinese and Japanese characters, other words, and punctuation. For each role (aspect or opinion), CHOICE gives B/I/O probabilities per token. Candidates include the argmax BIO spans and alternative spans supported by the token marginals. For tokens i through j, the lattice score is

$$
\begin{array} { l } { { \displaystyle { \cal L } ( i , j ) = \left[ p _ { i } ( B ) + p _ { i } ( I ) p _ { i - 1 } ( O ) \right] } } \\ { { \displaystyle ~ \cdot \prod _ { k = i + 1 } ^ { j } p _ { k } ( I ) \left[ 1 - p _ { j + 1 } ( I ) \right] , } } \end{array}\tag{4}
$$

with boundary values $p _ { 0 } ( O ) = 1$ and $p _ { n + 1 } ( I ) = 0$ This is a heuristic score, not a normalized span distribution. We keep spans of at most 12 tokens with $L ( i , j ) \ge 0 . 2$ , add variants suggested by training boundary statistics, and include literal matches from the training lexicon. Each non-overlapping aspect–opinion combination receives a NOUL pair judgment.

Boundary checks and extensions. A pair judgment says whether a relation is plausible, but the metric also requires the dataset’s exact boundaries.

Further NOUL questions therefore ask whether a candidate is exactly one annotated phrase and whether a pair follows the dataset’s relation convention. Their context includes four same-corpus training reviews retrieved by BM25. Each check is repeated with character-bigram, character-trigram, and word retrieval, using Jieba for Chinese words. Opinion candidates are also extended by up to three tokens to the left and two to the right, with no internal punctuation and at most 12 tokens; an extension enters the pair pool when its span check is at least 0.5.

Learned pair selection. For each candidate whose lattice-stage pair judgment is at least 0.3, a feature vector $f ( x , a , o )$ combines model judgments, BIO support, boundary checks, training counts, edge statistics, length, distance, competing variants, extension membership, the mean and minimum check logits across retrieval views, and each check relative to the strongest overlapping rival. A logistic reranker predicts

$$
q ( a , o \mid x ) = \sigma \big ( w _ { g } ^ { \top } f ( x , a , o ) + b _ { g } \big ) ,\tag{5}
$$

where $g$ is a language group: English, Chinese, Japanese, or a shared Russian/Tatar/Ukrainian group. Each group model is fitted on development labels with an L2 penalty of 5, and features also encode corpus identity. Greedy decoding takes candidates in descending score, stops below 0.25, and drops a pair only when both its aspect and its opinion overlap an already selected pair, so one aspect can pair with several opinions and vice versa.

Pair-conditioned VA. For each selected pair, two SCORE questions estimate VA conditioned on both a and o. Instead of Task 1’s ridge model, an affine map per corpus m and dimension, fitted on about 1,000 training gold pairs, gives

$$
\begin{array} { r } { \hat { y } _ { d } = \exp _ { [ 1 , 9 ] } ( \beta _ { m , d } z _ { d } + \gamma _ { m , d } ) . } \end{array}\tag{6}
$$

## 3.4 Task 3: category enrichment

Task 3 takes the predicted Task 2 pairs and their VA values. For each pair, CHOICE scores the corpus’s category inventory. Each option names an entity and describes its attribute; the state includes four retrieved annotated reviews and a glossary of the three most frequent training aspects of each category.

We also estimate category priors and lexical distributions from training counts. With add-one prior

$\pi _ { c } ,$ the aspect lookup is

$$
P ( c \mid a ) = \frac { n ( a , c ) + \pi _ { c } } { n ( a ) + 1 } ,\tag{7}
$$

and the opinion lookup is analogous. Surface forms are lowercased, and NULL has no lexical counts. A six-dimensional vector $h _ { c }$ contains the log model probability, the log aspect lookup and its seen-intraining interaction, the log prior, and the log opinion lookup and its seen interaction. Model probabilities are floored at $1 0 ^ { - 3 }$ before taking logs. A conditional-logit model gives

$$
P ( c \mid x , a , o ) = \frac { \exp ( u _ { g } ^ { \top } h _ { c } ) } { \sum _ { c ^ { \prime } } \exp ( u _ { g } ^ { \top } h _ { c ^ { \prime } } ) } .\tag{8}
$$

The six weights $u _ { g }$ are fitted on about 1,000 training gold pairs per corpus, with L2 penalty 1 and no intercept. During fitting, counts exclude all annotations sharing the sampled review’s normalized text, and retrieval excludes that text. The highestscoring category is attached to the pair without changing its spans or VA.

## 4 Experiments

## 4.1 Setup

Data. We use the public Track A splits of SemEval-2026 Task III (Yu et al., 2026). Task 1 covers ten corpora in English, Chinese, Japanese, Russian, Tatar, and Ukrainian across restaurants, laptops, hotels, and finance; Tasks 2 and 3 cover the eight non-finance corpora. The Task 1 test set has 9,658 reviews and 16,186 aspect annotations; the Task 2 and Task 3 test sets share 6,690 reviews with 14,262 triplets and 14,263 quadruplets (Appendix A).

Metrics. We use the unchanged official scorer with Task 1 normalization disabled. Its joint error is

$$
\mathrm { R M S E } _ { \mathrm { V A } } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \hat { y } _ { i } - y _ { i } \| _ { 2 } ^ { 2 } } ,\tag{9}
$$

pooled over all annotations of the ten corpora (micro). For Tasks 2 and 3, each exact structural match contributes

$$
t _ { i } = \operatorname* { m a x } \left( 0 , 1 - \frac { \| \hat { y } _ { i } - y _ { i } \| _ { 2 } } { \sqrt { 1 2 8 } } \right) ,\tag{10}
$$

where a match requires aspect and opinion, plus category for Task 3. Continuous precision and recall divide the summed contributions by the predicted and gold tuple counts; continuous F1 (cF1) is their harmonic mean, reported ×100 and averaged over the eight corpora.

<table><tr><td>System</td><td>Task adaptation</td><td>Tunes backbone</td><td>Generates text</td><td>T1 RMSE↓</td><td>T2 cF1 ↑</td><td>T3 cF1 ↑</td></tr><tr><td>SemEval-2026 participants</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PAI (Ruan et al., 2026)</td><td>LoRA + VA alignment</td><td></td><td></td><td>1.0663</td><td>57.73</td><td></td></tr><tr><td>TeleAI (Zhou et al., 2026)</td><td>LoRA + regression head</td><td></td><td> T2/3</td><td>1.0737</td><td>55.66</td><td>31.26</td></tr><tr><td>PALI (Chen, 2026)</td><td>LoRA adapters</td><td></td><td></td><td>1.1340</td><td>57.50</td><td>49.20</td></tr><tr><td>Takoyaki (Yamada et al., 2026)</td><td>Retrieval + rules</td><td></td><td></td><td></td><td>56.20</td><td>48.03</td></tr><tr><td>nchellwig (Hellwig et al., 2026)</td><td>LoRA</td><td></td><td></td><td></td><td>56.55</td><td>47.19</td></tr><tr><td>TeamLasse (Strothe, 2026)</td><td>LoRA + encoder regressor</td><td></td><td>• T2/3</td><td></td><td>53.43</td><td>44.33</td></tr><tr><td>Model baselines of Lee et al. (2026)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama-3.3-70B (Meta, 2024)</td><td>4-bit QLoRA</td><td></td><td></td><td>2.5683</td><td>46.40</td><td>38.62</td></tr><tr><td>GPT-OSS-120B (OpenAI, 2025)</td><td>4-bit QLoRA</td><td></td><td></td><td>1.2362</td><td>45.71</td><td>37.27</td></tr><tr><td>Kimi K2 Thinking (Moonshot AI, 2025)</td><td>One-shot prompting</td><td>O</td><td></td><td>1.8873</td><td>38.59</td><td>26.95</td></tr><tr><td>Ours</td><td>488 coefficients on CPU</td><td>O</td><td>O</td><td>1.0645</td><td>52.09</td><td>44.06</td></tr></table>

Table 1: Test results and task adaptation. T1: micro RMSE over ten corpora; T2/T3: macro cF1 over eight corpora. • yes, ◦ no. Participant aggregates are computed from the per-corpus scores of Yu et al. (2026, Tables 6–8); –: not every corpus reported. Best score per column in bold.
<table><tr><td></td><td colspan="2">Task 1: RMSE↓</td><td colspan="3">Task 2: cF1 ↑</td><td colspan="3">Task 3: cF1 ↑</td></tr><tr><td>Corpus</td><td>Ours</td><td>Best</td><td>Ours</td><td>Best</td><td>Exact VA</td><td>Ours</td><td>Best</td><td>Cat. acc.</td></tr><tr><td>English restaurant</td><td>1.2163</td><td>1.1035ª</td><td>68.21</td><td>70.21f</td><td>+5.31</td><td>63.48</td><td>65.14f</td><td>93.0</td></tr><tr><td>English laptop</td><td>1.2086</td><td>1.2408ªa</td><td>62.25</td><td>63.66f</td><td>+5.54</td><td>37.38</td><td>42.27f</td><td>60.2</td></tr><tr><td>Japanese hotel</td><td>0.6454</td><td>0.5561b</td><td>50.03</td><td>58.37b</td><td>+2.60</td><td>37.59</td><td>42.52g</td><td>75.3</td></tr><tr><td>Japanese finance</td><td>0.7296</td><td>0.6581b</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Russian restaurant</td><td>1.3290</td><td>1.2190c</td><td>51.26</td><td>57.93c</td><td>+5.64</td><td>46.80</td><td>55.99c</td><td>91.3</td></tr><tr><td>Tatar restaurant</td><td>1.4604</td><td>1.5294º</td><td>45.09</td><td>51.19h</td><td>+5.61</td><td>42.03</td><td>47.36f</td><td>93.1</td></tr><tr><td>Ukrainian restaurant</td><td>1.3464</td><td>1.1888c</td><td>50.24</td><td>57.87</td><td>+5.65</td><td>46.84</td><td>54.37c</td><td>93.2</td></tr><tr><td>Chinese restaurant</td><td>0.9591</td><td>0.9256d</td><td>50.21</td><td>56.38c</td><td>+3.29</td><td>46.51</td><td>55.21i</td><td>92.6</td></tr><tr><td>Chinese laptop</td><td>0.7611</td><td> $0 . 6 1 0 3 ^ { \mathrm { b } }$ </td><td>39.43</td><td>53.08g</td><td>+2.07</td><td>31.88</td><td>48.24i</td><td>80.8</td></tr><tr><td>Chinese finance</td><td>0.5823</td><td>0.4841e</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggregate</td><td>1.0645</td><td>1.0663c</td><td>52.09</td><td>57.73c</td><td>+4.46</td><td>44.06</td><td>49.20g</td><td>85.0</td></tr></table>

Table 2: Per-corpus test results. Best: best published score for the corpus, from Yu et al. (2026); for the aggregate, the best complete-coverage aggregate of Table 1. Superscripts: (a) LogSigma (Hikal et al., 2026), (b) TeleAI (Zhou et al., 2026), (c) PAI (Ruan et al., 2026), (d) ICT-NLP (Huang et al., 2026), (e) HUS@NLP-VNU (Cao et al., 2026), (f) Takoyaki (Yamada et al., 2026), (g) PALI (Chen, 2026), (h) nchellwig (Hellwig et al., 2026), (i) NYCU Speech Lab. Bold: better than Best. Exact VA: gain if our extracted pairs had gold VA. Cat. acc.: category accuracy (%) on predicted pairs that match a gold pair, highlighted below 80. The finance corpora have Task 1 data only.

Protocol. We use jev-1.13.0 with the official data and scorer (Appendix B). Every fitted component uses training labels except the Task 2 reranker, which is trained on development labels and scored out of fold. Development data also select all design choices and thresholds, and test labels are used only for evaluation; Appendix B gives sampling and selection details. Retrieved examples exclude training texts that occur in development or test.

## 4.2 Competitive without generation or tuning

Table 1 compares our system with six participant systems and three model baselines evaluated by Lee et al. (2026). It is the only system that neither tunes a backbone nor generates text in any task: its whole task adaptation is 488 coefficients fitted on CPU (100 in the Task 1 ridge models, 332 in the Task 2 rerankers, 32 in the pair-level VA maps, and 24 in category fusion). Because the competition ranks each corpus separately, we reconstruct participant aggregates from the published per-corpus scores. The systems also differ in backbone and supervision, so the comparison places our results in context rather than isolating the effect of generation.

<table><tr><td>Variant (test)</td><td>Score</td><td>∆</td></tr><tr><td colspan="3">Task 1, micro RMSE ↓</td></tr><tr><td>Full system</td><td>1.0645</td><td></td></tr><tr><td>— joint terms (shrinkage)</td><td>1.1203</td><td>+0.0558</td></tr><tr><td>— demonstrations (zero-shot)</td><td>1.1015</td><td>+0.0370</td></tr><tr><td>— calibration (raw scores)</td><td>2.0731</td><td>+1.0086</td></tr><tr><td colspan="3">Task 2, exact-match pair F1 ↑</td></tr><tr><td>Full system — opinion extensions</td><td>56.55</td><td></td></tr><tr><td>— extra retrieval views</td><td>56.47 56.03</td><td>-0.08 -0.52</td></tr><tr><td>— lexicon pair answers</td><td>56.22</td><td>-0.33</td></tr><tr><td>— example-conditioned checks</td><td>54.14</td><td>-2.41</td></tr><tr><td>— lattice (argmax spans only)</td><td>50.46</td><td>-6.09</td></tr><tr><td>– reranker (pair judgment only)</td><td>41.24</td><td>-15.31</td></tr><tr><td colspan="3">Task 3, cF1 ↑</td></tr><tr><td>Full system</td><td>44.06</td><td></td></tr><tr><td>training lookups</td><td>44.08</td><td>+0.02</td></tr><tr><td>— model category decision</td><td>38.03</td><td>-6.03</td></tr></table>

Table 3: Component ablations on test (T1 micro over ten corpora, T2/T3 macro over eight). Each variant removes one component and refits the learned postprocessor on the data the final system uses. Removing the T2 checks also removes the extensions they admit. Losses of at least 0.01 RMSE or one F1 point are highlighted.

Regression. Our system obtains 1.0645 micro RMSE, the lowest aggregate among the 14 teams that report all ten corpora and 0.0018 below PAI’s 1.0663 (Ruan et al., 2026). Without participant predictions this margin cannot be tested for significance. Per corpus (Table 2), our system beats the best published score on English laptop and Tatar restaurant and trails it on the other eight.

Extraction. Task 2 reaches 52.09 cF1, between the sixth and seventh of the 12 complete-coverage participants, and Task 3 reaches 44.06, between the fifth and sixth of 9. Both exceed the fine-tuned Llama-3.3-70B and GPT-OSS-120B baselines. The two other systems without backbone tuning both generate text: our system trails Takoyaki’s retrieval-and-rules pipeline by about 4 points on each task and exceeds one-shot Kimi K2 Thinking by 13.5 and 17.1 points.

## 4.3 What each component contributes

Table 3 removes one component at a time from each final system. Each variant refits its learned postprocessor on the same data as the final system (training data for Tasks 1 and 3, development data for the Task 2 reranker); the ablations are post hoc and informed no design choice. For Task 2 we report exact-match pair F1, i.e., cF1 with gold VA, which isolates the structural decisions being

<table><tr><td>Corpus</td><td>Not proposed</td><td>Not selected</td><td>Found</td><td>Near-miss FP</td></tr><tr><td>Eng. rest.</td><td>15.5</td><td>14.4</td><td>70.0</td><td>46.6</td></tr><tr><td>Eng. laptop</td><td>17.1</td><td>20.7</td><td>62.2</td><td>46.7</td></tr><tr><td>Jpn. hotel</td><td>17.2</td><td>34.5</td><td>48.3</td><td>36.3</td></tr><tr><td>Rus. rest.</td><td>26.0</td><td>21.3</td><td>52.7</td><td>32.5</td></tr><tr><td>Tat. rest.</td><td>27.2</td><td>28.5</td><td>44.4</td><td>30.4</td></tr><tr><td>Ukr. rest.</td><td>27.9</td><td>20.3</td><td>51.8</td><td>34.0</td></tr><tr><td>Zho. rest.</td><td>14.0</td><td>37.2</td><td>48.8</td><td>49.0</td></tr><tr><td>Zho. laptop</td><td>28.6</td><td>37.3</td><td>34.1</td><td>66.3</td></tr><tr><td>Macro</td><td>21.7</td><td>26.8</td><td>51.5</td><td>42.7</td></tr></table>

Table 4: Where Task 2 loses gold pairs on test (%): never proposed as a candidate, proposed but not selected, or found; the larger loss per corpus is in bold. Near-miss FP: share of wrongly selected pairs whose aspect and opinion both overlap one gold pair.

ablated.

Task 1: calibration matters most. Raw scores reach only 2.0731 RMSE even with nine demonstrations: nine rubric levels do not by themselves put the model’s scores on the gold scale, but a few coefficients per corpus do. Calibration removes about half of the error. Its joint terms, which let predicted arousal depend on how extreme the valence is, are worth 0.0558 over independent shrinkage, and the demonstrations add 0.0370 once scores are calibrated. On development data the joint terms improve RMSE by 0.0506, with a paired bootstrap 95% interval of [−0.0626, −0.0395], while retrieving examples with BM25 instead of fixing them does not help (0.8613 against 0.8572).

Task 2: boundary decisions matter most. Replacing the reranker, and all the evidence it combines, by the lattice pair judgment alone (with a threshold chosen on development data) loses 15.31 points: no single signal decides boundaries well; their learned combination does. Restricting candidates to the argmax BIO spans loses 6.09, the value of the lattice, and removing the exampleconditioned checks loses 2.41. The extra retrieval views, the lexicon pair answers, and the opinion extensions each add less than a point.

Task 3: the model’s category decision carries the signal. Without the model’s category probabilities, training lookups and the prior reach only 38.03 cF1 (−6.03). Without the lookups, the model decision and prior alone match the full fusion (44.08 against 44.06).

<table><tr><td>System</td><td>T2</td><td>T3</td><td>Loss</td></tr><tr><td>PALI (Chen, 2026)</td><td>57.50</td><td>49.20</td><td>8.30</td></tr><tr><td>Takoyaki (Yamada et al., 2026)</td><td>56.20</td><td>48.03</td><td>8.17</td></tr><tr><td>nchellwig (Hellwig et al., 2026)</td><td>56.55</td><td>47.19</td><td>9.36</td></tr><tr><td>TeamLasse (Strothe, 2026)</td><td>53.43</td><td>44.33</td><td>9.10</td></tr><tr><td>Ours</td><td>52.09</td><td>44.06</td><td>8.03</td></tr></table>

Table 5: Macro cF1 on Tasks 2 and 3, and the loss when a category is added to the extracted pairs, for the systems that report every corpus of both tasks and score at least as high as ours on both. Smallest loss in bold.

## 4.4 Remaining extraction error is structural

Spans, not sentiment values. Gold VA on our extracted pairs would add only 4.46 cF1 to Task 2 (Table 2), so most of the error lies in which spans are extracted. Table 4 locates it. Averaged over corpora, 21.7% of the gold pairs are never proposed as candidates and 26.8% are proposed but not selected, and 42.7% of the wrongly selected pairs are boundary near-misses that overlap a gold pair on both roles. The two losses split by language: the Russian, Tatar, and Ukrainian corpora lose more than a quarter of their gold pairs before selection, whereas the Chinese and Japanese corpora lose over a third among proposed candidates. Chinese laptop, our weakest corpus (39.43 against PALI’s 53.08; Chen, 2026), suffers from both, and two thirds of its false positives are near-misses. The remaining gap lies in candidate coverage and boundary selection.

Categories cost no more than for the leading systems. Adding a category lowers our macro cF1 from 52.09 to 44.06. This loss of 8.03 points is the smallest among the systems that match or exceed us on both tasks (Table 5), so our 5.14-point gap to PALI on Task 3 is inherited from the Task 2 pairs. Category accuracy on matched pairs exceeds 91% in every restaurant corpus but is 60.2% in English laptop and 75.3% in Japanese hotel, whose inventories have 113–121 and 44 labels.

## 5 Conclusion

Deciding instead of generating is enough for competitive dimensional ABSA. With Jev’s typed decisions, corpus statistics, and 488 coefficients fitted on CPU, our system obtains the lowest ten-corpus Task 1 aggregate of any participating system and outperforms fine-tuned Llama-3.3-70B and GPT-OSS-120B baselines on extraction, without text generation or backbone tuning. What makes it work is alignment with the annotation scheme: a few coefficients per corpus put the model’s scores on the gold scale, the remaining extraction error lies in proposing and selecting span boundaries rather than in sentiment values or categories, and ablations show that the learned combination of boundary evidence carries extraction. Closing that gap, and comparing the latency and cost of decision composition with generative systems under matched conditions, are the natural next steps.

## References

An Cao, Lam Hoang, Le Ngoc Toan, and Ha Linh. 2026. HUS@NLP-VNU at SemEval-2026 task 3: Dualstream syntax-aware modeling and direct preference optimization for dimensional ABSA. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 2200–2208, San Diego, California, USA. Association for Computational Linguistics.

Cheng Chen. 2026. PALI at SemEval-2026 task 3: LoRA fine-tuning with validation for DimABSA. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 641–649, San Diego, California, USA. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Zhibin Gou, Qingyan Guo, and Yujiu Yang. 2023. MvP: Multi-view prompting improves aspect sentiment tuple prediction. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4380–4397, Toronto, Canada. Association for Computational Linguistics.

Nils Constantin Hellwig, Jakob Fehle, Udo Kruschwitz, and Christian Wolff. 2026. nchellwig at SemEval-2026 task 3: Self-consistent structured generation (SCSG) for dimensional aspect-based sentiment analysis using large language models. In Proceedings of the 20th International Workshop on Semantic Evaluation (2026), pages 37–47, San Diego, California, USA. Association for Computational Linguistics.

Baraa Hikal, Jonas Becker, and Bela Gipp. 2026. LogSigma at SemEval-2026 task 3: Uncertaintyweighted multitask learning for dimensional aspectbased sentiment analysis. In Proceedings of the

20th International Workshop on Semantic Evaluation (2026), pages 1237–1257, San Diego, California, USA. Association for Computational Linguistics.

Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings ofthe Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 168–177. Association for Computing Machinery.

Liyuan Huang, Jiawei He, Wutao Shen, Lin Li, and Jin Zhang. 2026. ICT-NLP at SemEval-2026 task 3: Less is more — multilingual encoder with joint training and adaptive ensemble for dimensional aspect sentiment regression. In Proceedings of the 20th International Workshop on Semantic Evaluation (2026), pages 950–957, San Diego, California, USA. Association for Computational Linguistics.

Lung-Hao Lee, Liang-Chih Yu, Natalia Loukachevitch, Ilseyar Alimova, Alexander Panchenko, Tzu-Mi Lin, Zhe-Yu Xu, Jian-Yu Zhou, Guangmin Zheng, Jin Wang, Sharanya Awasthi, Jonas Becker, Jan Philip Wahle, Terry Ruas, Shamsuddeen Hassan Muhammad, and Saif M. Mohammad. 2026. DimABSA: Building multilingual and multidomain datasets for dimensional aspect-based sentiment analysis. arXiv preprint arXiv:2601.23022. Version 3.

Meta. 2024. Llama 3.3 model card. 70B Instruct release, December 6, 2024. Accessed September 28, 2026.

Moonshot AI. 2025. Kimi K2 Thinking model card. Accessed September 28, 2026.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad AL-Smadi, Mahmoud Al-Ayyoub, Yanyan Zhao, Bing Qin, Orphée De Clercq, Véronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia Loukachevitch, Evgeniy Kotelnikov, Nuria Bel, Salud María Jiménez-Zafra, and Gül¸sen Eryigit.˘ 2016. SemEval-2016 task 5: Aspect based sentiment analysis. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 19–30, San Diego, California. Association for Computational Linguistics.

Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect based sentiment analysis. In Proceedings ofthe 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35, Dublin, Ireland. Association for Computational Linguistics.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 4(1–2):1–174.

Zhihao Ruan, Kaifeng Yang, Cheng Chen, Wenwen Dai, and Wenjia Mao. 2026. PAI at SemEval-2026 task 3: An LLM and data redistribution adaptationbased predictive strategy for valence-arousal scores. In Proceedings of the 20th International Workshop on Semantic Evaluation (2026), pages 1489–1494, San Diego, California, USA. Association for Computational Linguistics.

James A. Russell. 1980. A circumplex model of affect. Journal ofPersonality and Social Psychology, 39(6):1161–1178.

Lasse Strothe. 2026. TeamLasse at SemEval-2026 task 3: A hybrid generative-discriminative framework for dimensional aspect-based sentiment analysis. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 2155–2162, San Diego, California, USA. Association for Computational Linguistics.

Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 380–385, Minneapolis, Minnesota. Association for Computational Linguistics.

TypeSafe. 2026. Jev: System one and typed decision primitives. Product documentation. Accessed September 28, 2026.

Kosuke Yamada, Sho Takase, and Ryosuke Kohita. 2026. Takoyaki at SemEval-2026 task 3: Ensembling LLM predictions using demonstration retrieval for dimensional aspect-based sentiment analysis. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 1707–1723, San Diego, California, USA. Association for Computational Linguistics.

Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng Zhang. 2021. A unified generative framework for aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2416–2429, Online. Association for Computational Linguistics.

Liang-Chih Yu, Jonas Becker, Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Lung-Hao Lee, Ying-Lung Lin, Jin Wang, Jan Philip Wahle, Terry Ruas, Natalia Loukachevitch, Alexander Panchenko, Ilseyar Alimova, Lilian Diana Awuor Wanzare, Nelson Odhiambo, Bela Gipp, Kai-Wei Chang, and Saif Mohammad. 2026. SemEval-2026 task 3: Dimensional aspect-based sentiment analysis (DimABSA). In Proceedings of the 20th International Workshop on Semantic Evaluation (2026), pages 3753–3778, San Diego, California, USA. Association for Computational Linguistics.

Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan, Lidong Bing, and Wai Lam. 2021. Aspect sentiment quad prediction as paraphrase generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9209– 9219, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12697–12706. PMLR.

Yan Zhou, Wangshicheng Wang, Shiquan Wang, Mengjiao Bao, Ruiyu Fang, Shuangyong Song, Yongxiang Li, and Xuelong Li. 2026. TeleAI at SemEval-2026 task 3: Large language models for dimensional aspect-based sentiment analysis. In Proceedings ofthe 20th International Workshop on Semantic Evaluation (2026), pages 1846–1852, San Diego, California, USA. Association for Computational Linguistics.

## A Dataset Details

Table 6 reports counts of the official data before task-specific training filters. Training files for the non-finance corpora contain quadruplet annotations reused across tasks. Their counts therefore differ from the number of distinct Task 1 aspect targets. In English laptop test, Task 3 has 1,975 quadruplets and Task 2 has 1,974 triplets, so each task is evaluated against its own gold file. Task 3 reuses predicted Task 2 pairs, which does not require the gold inventories to agree.

## B Implementation and Reproducibility

We use jev-1.13.0 and the official DimABSA data and scorer at commit

bdc93be1224106ae7d3eb9   
5739c02a76ed4ae8a1

of the task repository. External comparisons use the published, rounded scores.

Supervision and selection. Task 1 demonstrations, the Task 1 ridge models, the Task 2 lexicon, boundary statistics and affine VA maps, and the Task 3 lookups and fusion weights are fitted on training labels. The Task 2 logistic reranker is trained on development labels; five-fold out-of-fold predictions by record give its development scores. Because features and thresholds were also chosen on development data, these scores are not nested estimates of the full selection procedure, and the folds do not group parallel Russian/Tatar/Ukrainian translations. The final Task 2 revision (opinion extensions, three retrieval views, and rival features) gained 1.50 cF1 out of fold but 0.41 on test; it changed several components at once.

Task 1 calibration uses 256 training text groups per corpus, 2,563 records and 4,664 VA annotations in total. Repeated texts stay in one group, and parallel translations share groups and folds. The sample excludes the fixed demonstrations and any text whose ID or normalized form occurs in development or test. The seed is 20260923, and ridge penalties are selected from {0.1, 1, 3, 10, 30, 100, 300}. A new calibration replaces the previous one only if it improves development RMSE by at least 0.02 and a paired cluster bootstrap (2,000 draws) puts the 95% interval of the change below zero.

Overlap handling. The Task 1 audit finds 23 train–test text overlaps in Japanese hotel; the Task 2 subset has two normalized-text overlaps with train. Retrieved examples exclude training texts that occur in development or test, and Task 3 applies the same exclusion to its statistics and glossary. Task 2’s lexicon and boundary counts use the full training split, so overlap removal is incomplete for that task.

Task 2 candidate details. Token BIO questions are batched at 48 questions per request, lattice pair checks at 32 pairs, and pair-conditioned VA at 16 pairs. The BIO state contains two synthetic examples. Training edge-affix variants use an inclusion/exclusion proportion of at least 0.9 with support of at least 20 occurrences; the maximum affix length is four tokens for Chinese/Japanese and two otherwise. Implicit aspects are disabled for English following the benchmark documentation and elsewhere when the training implicit-aspect rate is below 5%; only Japanese hotel meets the retained policy. Retrieved examples come from the same corpus.

Task 3 fitting details. The implemented inventory contains 14 English restaurant categories, 12 categories in the other restaurant corpora, 44 Japanese hotel categories, and 113–121 laptop categories. These counts come from the eligible training pools, not from a canonical label scheme. Fitting samples accumulate complete training reviews until at least 1,000 annotated pairs are covered per corpus. Count features leave out all annotations sharing the sampled review’s normalized text. Retrieved demonstrations also omit that text, but the category glossary is built from the whole eligible training pool, so the sampled text is not removed from every glossary entry. Each of the four category variants refits its own fusion weights; “model + prior” is therefore not the raw model argmax.

<table><tr><td>Corpus</td><td>Train rev.</td><td>T1 dev rev.</td><td>T1 test rev.</td><td>T1 test ann.</td><td>T2 dev rev.</td><td>T2 test rev.</td><td>T2 test ann.</td></tr><tr><td>English restaurant</td><td>2284</td><td>200</td><td>1000</td><td>1504</td><td>200</td><td>1000</td><td>2129</td></tr><tr><td>English laptop</td><td>4076</td><td>200</td><td>1000</td><td>1421</td><td>200</td><td>1000</td><td>1974</td></tr><tr><td>Japanese hotel</td><td>1600</td><td>200</td><td>800</td><td>1092</td><td>200</td><td>800</td><td>1443</td></tr><tr><td>Japanese finance</td><td>1024</td><td>200</td><td>800</td><td>1302</td><td></td><td></td><td></td></tr><tr><td>Russian restaurant</td><td>1240</td><td>56</td><td>1072</td><td>1637</td><td>48</td><td>630</td><td>1310</td></tr><tr><td>Tatar restaurant</td><td>1240</td><td>56</td><td>1072</td><td>1637</td><td>48</td><td>630</td><td>1310</td></tr><tr><td>Ukrainian restaurant</td><td>1240</td><td>56</td><td>1072</td><td>1637</td><td>48</td><td>630</td><td>1310</td></tr><tr><td>Chinese restaurant</td><td>6050</td><td>300</td><td>1000</td><td>1929</td><td>300</td><td>1000</td><td>2861</td></tr><tr><td>Chinese laptop</td><td>3490</td><td>300</td><td>1000</td><td>1673</td><td>300</td><td>1000</td><td>1925</td></tr><tr><td>Chinese finance</td><td>1000</td><td>200</td><td>842</td><td>2354</td><td>一</td><td>一</td><td>一</td></tr></table>

Table 6: Dataset statistics (rev.: reviews; ann.: annotations). Tasks 2 and 3 share reviews; Task 3 has one more English laptop test annotation.

## C Interpreting the Diagnostics

The exact-VA diagnostic in Table 2 scores our extracted pairs with gold VA. It removes numerical error on structurally matched predictions and keeps the extracted pairs unchanged, so it bounds what improving VA alone can gain for that pair set; it says nothing about candidate coverage, recall, or category selection. Category accuracy likewise conditions on matched pairs and ignores missing or spurious pairs. In our results, the ratio of Task 3 to Task 2 cF1 is close to this conditional accuracy, but not identical to it, because VA weights and gold tuple counts also enter.

## D Literature-Audit Protocol

Figure 1 describes a bounded corpus, not all of ABSA. We searched ACL Anthology metadata for ACL, EMNLP, NAACL, EACL, and COLING main proceedings and associated Findings published in 2004–2025, including LREC-COLING 2024. The case-insensitive title rule requires both “aspect” and “sentiment”, or the standalone abbreviation “ABSA”. Workshops and demonstrations are outside the scope. The start year follows early feature-level opinion mining (Hu and Liu, 2004); matching papers begin in 2008. The Anthology snapshot is commit 51279f83, retrieved September 28, 2026.

We additionally searched official AAAI, NeurIPS, and ICML proceedings for 2004–2025 and ICLR conference programs for 2013–2025 with the same rule. AAAI’s older directories were retrieved through a public reader proxy; AAAI was not held in 2009. ICLR, NeurIPS, and ICML produced no matching titles, which does not mean they publish no ABSA research. Source URLs, retrieval hashes, track filters, and zero-hit records are preserved in the audit directory.

<table><tr><td>Year</td><td>D</td><td>G</td><td>Other</td><td>n</td></tr><tr><td>2008</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>2010</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>2013</td><td>0</td><td>0</td><td>2</td><td>2</td></tr><tr><td>2014</td><td>1</td><td>0</td><td>3</td><td>4</td></tr><tr><td>2015</td><td>2</td><td>1</td><td>1</td><td>4</td></tr><tr><td>2016</td><td>6</td><td>1</td><td>0</td><td>7</td></tr><tr><td>2017</td><td>2</td><td>0</td><td>1</td><td>3</td></tr><tr><td>2018</td><td>17</td><td>1</td><td>2</td><td>20</td></tr><tr><td>2019</td><td>22</td><td>0</td><td>1</td><td>23</td></tr><tr><td>2020</td><td>29</td><td>1</td><td>0</td><td>30</td></tr><tr><td>2021</td><td>28</td><td>9</td><td>4</td><td>41</td></tr><tr><td>2022</td><td>21</td><td>10</td><td>0</td><td>31</td></tr><tr><td>2023</td><td>9</td><td>19</td><td>0</td><td>28</td></tr><tr><td>2024</td><td>23</td><td>30</td><td>1</td><td>54</td></tr><tr><td>2025</td><td>2</td><td>23</td><td>3</td><td>28</td></tr><tr><td>Total</td><td>162</td><td>95</td><td>20</td><td>277</td></tr></table>

Table 7: Annual paper counts behind Figure 1. G: the proposed method uses a text-generative model; D: it does not; Other: latent methods or unresolved model use. Years without included papers are omitted.

The combined search returned 293 venueeligible candidates: 261 from the Anthology and 32 from AAAI. We excluded 16 dataset-only, diagnostic, or non-ABSA-prediction papers, leaving 277 papers. Task or dataset papers remain eligible when they introduce or adapt an actual predictor or training intervention. Each included paper contributes once, regardless of the number of proposed variants or evaluated tasks. Models used only as comparison baselines or mentioned in related work do not affect its category.

Model-use codebook. Generative means that a proposed method or its tested variant uses a textgenerative model at any stage: resource construction, training augmentation, representation extraction, candidate scoring, preprocessing, or task inference. Mixed pipelines count as Generative even when their final task head is discriminative. This includes GPT-, T5-, and BART-family models, translation systems, and autoregressive language-model features such as ELMo and XLNet. Using only the encoder of a text-generative pretrained model also qualifies. The category thus measures model use, not whether a system generates text at inference.

Discriminative covers direct label, rating, tag, span, table, or action decisions without identified text-generative model use. BERT/RoBERTa masked-language-model encoders do not qualify as text-generative models under this codebook; neither does masked-token substitution alone. A taskspecific pointer or transition decoder without a text-generative language model is not automatically Generative. Early statistical topic models, VAEs, and RBMs also do not qualify merely because they have a probabilistic generative formulation. Latent/discovery approaches are retained as Other; a discriminative predictor using a latent auxiliary objective remains Discriminative unless a text-generative model is also used.

We also retain unresolved model use as Other rather than assuming that an undisclosed component is non-generative. For example, UGTS names AMRLib and GraphMerge names the Berkeley parser without specifying a checkpoint. Conversely, APARN names SPRING, whose documented BART backbone establishes Generative preprocessing. Other contains 11 latent/discovery papers and nine papers with unresolved model use.

Publisher full texts were temporarily unavailable for part of the AAAI expansion. Of its 30 included papers, 12 were checked against full texts, ten against author or associated implementations and dependency documentation, and one topic model against its abstract. Seven remain unresolved; their abstracts establish task eligibility but cannot establish absence of text-generative dependencies. These papers stay in the denominator. Individual records distinguish evidence types and link the inspected sources.

Aggregation and limitations. The figure pools papers within five publication periods. Each percentage divides the category count by all included papers in that period, retaining Other in the denominator. Table 7 gives annual counts; no observation is imputed for years without matching papers. In 2024–2025, 53 of 82 papers have identified textgenerative use and four remain unresolved. Assigning all four to Generative would raise that share from 64.6% to 69.5%.

The early 2004–2013 period contains only four papers and cannot establish the field’s original method distribution. Unequal period lengths, evolving venue coverage, title vocabulary, and changing task composition further limit interpretation. Coding was model-assisted with targeted source checks, without independent double annotation, so no interannotator agreement is available. The accompanying analysis/absa-trend/ directory provides titles, links, evidence, exclusions, and the counting protocol. The figure builder computes percentages directly from those records.

## E Prompt Templates

We document the prompt templates used by the final three-task pipeline. Each request consists of a shared state and a dictionary of typed questions; each question specifies its type, instructions, and, for SCORE or CHOICE, criteria. The text below preserves the implemented wording, with line wrapping for presentation. Braced names such as {aspect} are substitution slots, not literal input. Corpus-specific reviews, demonstrations, category inventories, and glossaries are filled at runtime.

## E.1 Shared valence–arousal rubrics

Both Task 1 and the pair-scoring stage of Task 2 use type: score, with the ordered criteria in Table 8. These nine level descriptions are our rubric; the 1–9 scale and the short dimension definitions below follow the benchmark (Lee et al., 2026). The returned expected index is zero-based and is shifted by one before calibration (Equation 1).

Dimension definitions. The corresponding sentence is appended to the question:

Valence: 1 = most negative, 9 = most   
positive.

Arousal: 1 = calm/low intensity, 9 =   
excited/high intensity.

Dimension-specific focus. The instructions.focus field ends with the following text for valence and arousal, respectively:

<table><tr><td>Level</td><td>Valence criterion</td><td>Arousal criterion</td></tr><tr><td>1</td><td>Strongly negative: a severe fault, harsh or contemptuous complaint</td><td>Very calm, low energy: the aspect arouses no feeling at all; the writer is indifferent</td></tr><tr><td>2</td><td>Clearly negative: the aspect is described as bad or</td><td>Calm: the aspect is regarded without emotional charge</td></tr><tr><td>3</td><td>disappointing Moderately negative: real criticism, but not emphatic</td><td>Somewhat calm: only the faintest feeling about the aspect</td></tr><tr><td>4</td><td>Mildly negative: a small complaint or a slight reservation</td><td>Mildly calm: a low-energy, subdued feeling</td></tr><tr><td>5</td><td>Neutral or mixed: no clear polarity, or praise and criticism cancel out</td><td>Moderate: an ordinary, middle-of-the-road level of feeling</td></tr><tr><td>6</td><td>Mildly positive: a small or lukewarm compliment</td><td>Moderately activated: the feeling runs a little above ordinary</td></tr><tr><td>7</td><td>Moderately positive: the aspect is described as good</td><td>Activated, excited: a clearly energised feeling about the</td></tr><tr><td>8</td><td>Clearly positive: strong approval, the aspect is praised</td><td>aspect Strongly activated: high energy, intensely felt</td></tr><tr><td>9</td><td>Strongly positive: enthusiastic praise, superlatives, delight</td><td>Extremely activated, high energy: furious or thrilled; the strongest feeling</td></tr></table>

Table 8: Verbatim ordered criteria for the shared SCORE questions. Level numbers show the benchmark scale; the API criterion indices are 0–8.

Judge only the sentiment directed at   
this aspect; ignore sentiment toward   
any other aspect in the text.

Judge only the feeling directed at this   
aspect; ignore feeling toward any other   
aspect in the text. Arousal is how   
activated that feeling is – how calm or   
how excited – not how positive or   
negative it is.

Implicit aspects. For VA questions, the aspect "{aspect}" becomes the following phrase when the aspect is NULL:

the aspect that is left implicit and   
never named in the text

## E.2 Task 1: given-aspect regression

The state has two fields: review\_to\_score contains the input review, and labelled\_examples contains nine fixed training demonstrations. Each demonstration has review, aspect, valence, and arousal fields; the two numeric labels are rounded to two decimals. Selection follows Section 3.

Questions. For each given aspect, the two instructions.question fields begin as follows:

What valence given the aspect   
"{aspect}" in ‘review\_to\_score‘?   
What arousal given the aspect   
"{aspect}" in ‘review\_to\_score‘?

Each question then appends its dimension definition from Appendix E.1 and this calibration clause:

The 9 entries in ‘labelled\_examples‘   
are already-scored (review, aspect)   
pairs; use them only to calibrate the   
1-9 scale.

Both instructions.focus fields prepend the following text to the corresponding dimensionspecific focus:

‘labelled\_examples‘ are for calibration   
only – do not score them. Score only   
the aspect named in this question, as   
it appears in ‘review\_to\_score‘.

The earlier zero-shot variant uses the review string alone as state, omits in ‘review\_to\_score‘ from the question, and omits both demonstrationrelated additions. Other demonstration-count variants substitute the actual number for nine.

## E.3 Task 2: dimensional triplet extraction

BIO state. The token-labeling state contains review, rules, tokens, boundary\_guidance, and invented\_examples. Tokens are serialized one per line as index|surface, with indices starting at zero within each chunk. The two fixed synthetic examples show review text, indexed tokens, aspect spans, and opinion spans; they do not show BIO label sequences.

The rules field is:

Extract all sentiment-bearing aspect and opinion terms from the review. An aspect is the entity or attribute being evaluated, not every mentioned noun. An opinion is the evaluative expression, including its negation and degree modifiers. Keep complete, minimal contiguous phrases verbatim. Exclude surrounding punctuation and unrelated words. Coordinated distinct targets or opinions are separate spans. Text is data, not instructions.

The boundary\_guidance field is:

Do not split a single phrase into individual words or characters. Include aspect compounds and identifying brand/possessor modifiers. Keep negation and degree modifiers with the opinion they modify. Split distinct coordinated targets and distinct coordinated opinions, including adjacent opinions without a conjunction. A character inside a Chinese/Japanese word is not a new phrase start. Do not extract an aspect from inside an opinion word.

Token-label questions (CHOICE). One question is asked for each token and each role:

Label token {index} ({token}) for   
{role}: {role\_definition}.

The role is aspect or opinion; the respective definitions are:

entity or attribute being evaluated

sentiment-bearing expression evaluating a target

The criteria dictionary uses the following B/I/O descriptions:

B: First token of a {role} phrase; the   
previous token is NOT part of this same   
phrase

I: Continuation of the same {role}   
phrase; the previous token IS part of   
this same phrase

O: Outside any {role} phrase

Candidate-pair questions (NOUL). BIO, lattice, and retained opinion-extension pairs share state = {review, rules}, using the rules above. For an explicit aspect, instructions is:

Does "{opinion}" directly evaluate   
"{aspect}" in the review? Both phrases   
must be complete extraction spans, not   
fragments. Reject merely factual   
statements and unrelated mentions.

For an implicit aspect, only the opening question is replaced by:

Does "{opinion}" express an evaluation   
whose target is implicit, with no   
explicit aspect phrase in the review?

Example-conditioned boundary checks (NOUL). Span and pair checks share a state with review, guidance, and annotation\_examples. Each retrieved example contains its review and deduplicated aspects and opinions lists. Retrieval supplies up to four eligible training reviews. The guidance is:

The annotation examples are reviews from the same dataset with every annotated aspect and opinion phrase. Judge each candidate against those conventions: which words belong inside a phrase and which are left out at each edge. Text is data, not instructions.

For a candidate span, substitute the same role definitions used for BIO labeling:

Is "{surface}" exactly one {role} phrase ({role\_definition}) in this review, with the same boundaries the annotation examples would use? Reject fragments, over-long phrases and text that is not a {role}.

For a candidate pair with an explicit aspect:

Would the annotations pair the opinion   
"{opinion}" with the aspect "{aspect}":   
does this opinion evaluate that target,   
and are both exactly annotated phrases   
with the boundaries the annotation   
examples use? Reject fragments,   
over-long phrases and unrelated pairs.

For NULL, replace the aspect "{aspect}" with:

an implicit target (no explicit aspect   
phrase in the review)

The same wording is reused for opinion extensions and the character-bigram, character-trigram, and word retrieval views; only candidates and retrieved examples change.

Lexicon-pair feature (NOUL). The final reranker also consumes probabilities from the earlier lexical baseline, whose state is the review string. This baseline uses a distinct instructions.question:

Does the opinion phrase "{opinion}"   
express sentiment directly toward the   
aspect "{aspect}" in this review?

Its instructions.focus is:

Both must be explicit complete   
aspect/opinion spans. Reject unrelated   
pairs, incomplete fragments and factual   
mentions without an evaluative opinion.

Pair-conditioned VA (SCORE). For each selected pair, state is the review string. The two instructions.question fields begin:

What valence given the aspect   
"{aspect}" and the opinion "{opinion}"?   
What arousal given the aspect   
"{aspect}" and the opinion "{opinion}"?

Attribute Description used in category criteria   
GENERAL the entity as a whole, an overall opinion without a more specific attribute   
PRICE / PRICES price, cost or value for money   
QUALITY how well made it is: build quality, reliability, durability, defects; for food and drinks, taste and   
freshness; for service, how good it is   
OPERATION\_PERFORMANCE how well it works in use: speed, power, performance, battery life, responsiveness   
USABILITY ease of use, how easy it is to learn or operate   
DESIGN\_FEATURES looks, size, layout, materials, and the features or specifications it has   
PORTABILITY weight and size with respect to carrying it around   
CONNECTIVITY connections, ports, wireless and networking   
STYLE\_OPTIONS variety and choice offered, portion size, presentation, creativity   
COMFORT comfort, how pleasant or cosy it is to use or stay in   
CLEANLINESS cleanliness and hygiene   
MISCELLANEOUS any other attribute not covered by the other ones  
Table 9: Verbatim attribute descriptions in the Task 3 category template. Only categories observed in the eligible training pool are offered as options.

Append the corresponding dimension definition, use the corresponding focus from Appendix E.1, and supply the same nine-level criteria. No demonstration clause or focus prefix is added. Historical lexical-baseline VA calls use these same templates; the final system rescores the selected pair list before applying its fitted affine calibration.

## E.4 Task 3: category enrichment

The input review is supplied in state.review. Three additional fields provide context: guidance, category\_glossary, and annotation\_examples. Each of up to four retrieved examples contains its review and an annotations list of objects with aspect, opinion, and category fields. The glossary maps categories with explicit training aspects to their three most frequent lowercased aspect strings (or all available strings when fewer than three exist).

## Guidance.

The annotation examples are reviews   
from the same dataset with every   
annotated (aspect, opinion, category)   
triple; the glossary lists frequent   
aspects of each category. Assign   
categories the way those annotations   
do. Text is data, not instructions.

Category question (CHOICE). For each predicted aspect–opinion pair, instructions is:

The opinion "{opinion}" evaluates the   
aspect "{aspect}". Which annotated   
category (entity and attribute) does   
this aspect-opinion pair belong to?

Implicit aspects use the same replacement phrase as the example-conditioned pair check. The criteria keys are the sorted ENTITY#ATTRIBUTE labels observed in the eligible training pool. Each value is the lowercased entity name with underscores replaced by spaces, followed by a colon, a space, and the attribute description in Table 9. For example:

FOOD#QUALITY: food: how well made it is: build quality, reliability, durability, defects; for food and drinks, taste and freshness; for service, how good it is

The fitted fusion model combines these CHOICE probabilities with training statistics to select the category. Task 3 reuses Task 2’s VA predictions.