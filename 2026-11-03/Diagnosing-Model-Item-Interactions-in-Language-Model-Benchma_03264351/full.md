# Diagnosing Model–Item Interactions in Language Model Benchmarks with LSIRM

Jinwen Luo<sup>1</sup>, Han Jiang<sup>2</sup>, Ziang Xiao<sup>2</sup>, Minjeong Jeon<sup>1</sup>, and Susu Zhang<sup>3</sup>

<sup>1</sup>University of California, Los Angeles <sup>2</sup>Johns Hopkins University <sup>3</sup>University of Illinois Urbana-Champaign

## Abstract

Language models with similar benchmark accuracy can succeed on diferent items. Overall scores do not show where their performance difers or which item properties might explain those diferences. We demonstrate how the latent space item response model (LSIRM) can help evaluators investigate these patterns and their consequences for model comparison. We analyze 1,348 language models on 5,761 BBH items and compare ways of presenting and scoring answers across BBH, GPQA, MMLU-Pro, and MuSR. We examine LSIRM maps, which summarize diferences in performance across models and items, alongside item properties and records of correct and incorrect responses. In BBH, nearby items on these maps often share the same correct answer, such as “Yes” or “No.” Across the four benchmarks, this pattern is stronger when answers are scored using option labels, such as A, B, C, and D, than when using the text of the options. We then estimate each model’s tendency to perform better or worse on items with particular correct answers and quantify how this tendency contributes to its score. This separates an estimated contribution associated with correct answers from the remaining diferences in item performance. Two models’ total numbers of correct answers difered by only 14 out of 5,761 items. Excluding the estimated answer-category contribution from their scores moved their ranks hundreds of places in opposite directions, while keeping all other estimates unchanged. The approach helps evaluators identify diferences hidden by overall accuracy and understand how the choice of what to include in a score afects model rankings.

## 1 Introduction

In the Hugging Face Open LLM Leaderboard response data analyzed here (Fourrier et al., 2024), OpenBuddy Llama-3-70B and Smaug Llama-3-70B achieve accuracies of 64.97% and 64.85% on BIG-Bench Hard (BBH) (Suzgun et al., 2023). Their totals difer by seven correct answers out of 5,761 items. Yet OpenBuddy performs better on questions about adjective order, whereas Smaug performs better on logical deduction involving three objects (Figure 1A–B). Their similar overall accuracy therefore conceals diferences in where they succeed and fail. These diferences may matter when choosing a model for particular tasks.

Diferences also appear within the same task. Two Mistral 7B variants, Base-SFT-CPO and Base-SFT-RDPO, difer by 17 correct answers across BBH and by four on the 250 Ruin Names items, which ask models to identify humorous edits to names. CPO performs better on items whose correct answer is A, while RDPO performs better when C is correct, with gaps of approximately

![](images/c8aa3da6834859635e32ed414eb46d81fc00746143fd374c23e3859237f61d59.jpg)  
D Ruin Names by correct option

![](images/c619dcfa244b50d0b27061f5d7ff3033eb4e09a3e7757fc3e99a88c3c49dc99b.jpg)

![](images/7f693a277721f7e3e2af7dcc16433de40dea529b91fa51ed84c1f37dace26037.jpg)  
Figure 1: Similar overall accuracy conceals diferences across tasks and correct-answer groups. (A–B) OpenBuddy and Smaug, two 70B models, difer by seven correct responses across 5,761 BBH items, yet their ordering reverses between adjective order (Hyperbaton) and logical deduction with three objects (250 items per task). (C–D) The Mistral 7B Base-SFT-CPO and Base-SFT-RDPO models difer by 17 correct responses overall and four on Ruin Names (109 versus 113 out of 250), yet their ordering reverses between items whose correct answers are A and C. Bars show observed accuracy; n denotes the number of items. In D, A–D group items by the correct option in the answer key. The panels present selected descriptive examples from the same BBH response matrix; full model identifiers appear in Table S3.

30 and 35 percentage points, respectively (Figure 1C–D). Even a task-level average can conceal diferent patterns of success across its items.

We use the latent space item response model (LSIRM) (Jeon et al., 2021) as a diagnostic tool to examine these model–item diferences across a BBH cohort of 1,348 language models. LSIRM represents models and items in a shared space while accounting for overall model performance and item dificulty. A shorter distance between a model and an item corresponds to a higher predicted probability of a correct response, with the other estimates held fixed. We examine the resulting interaction map alongside item properties and records of correct and incorrect responses to investigate what distinguishes models with similar overall accuracy.

In BBH, nearby items on the map often share the same correct answer, such as “Yes” or “No,”

or the same correct option label, such as A or B. Interpreting this grouping requires considering how answers are presented and scored. An evaluator may score the likelihood assigned to an option label, score the full option text, or extract an answer from a generated response. These procedures can produce diferent correctness patterns from the same model.

Previous research documents preferences for particular option labels and sensitivity to option order (Zheng et al., 2024), diferences between generated-answer and first-token evaluation (Wang et al., 2024a), and dependence on label presentation and prediction mode (Nowak et al., 2026). The efects of adjustments for response preferences also vary across evaluation conditions (Bhatt and Ivanova, 2026). These findings provide reasons to investigate answer representation as an explanation for the grouping observed in LSIRM maps. In a separate evaluation across BBH, GPQA, MMLU-Pro, and MuSR, we vary answer presentation and scoring to examine how these choices afect that grouping.

To quantify the diferences associated with correct-answer categories, we add terms that estimate each model’s tendency to perform better or worse on items depending on which answer is correct. We estimate these terms jointly with overall model performance, item dificulty, and the remaining model–item interactions. This gives an explicit estimate of the answer-associated contribution to each model’s fitted score and allows us to inspect the performance diferences that remain.

In the BBH analysis, adding these terms reduces grouping by correct-answer category in the refitted map, while distinct model profiles remain. Calculating scores with and without the estimated answer contribution, while holding all fitted estimates fixed, changes model ordering. The decomposition and score comparisons let evaluators report where models difer, how those diferences relate to correct-answer categories, and how their rankings depend on the contributions included in a score.

## 2 A diagnostic use of LSIRM

## 2.1 From IRT to model–item interaction

Item response theory (IRT) models benchmark performance through the responses of individual models to individual items. Language models are treated as respondents, benchmark questions as items, and correctness as the observed response (Lalor et al., 2024; Vania et al., 2021). Let $Y _ { m i } \in \{ 0 , 1 \}$ indicate whether model m answers item i correctly, and let $p _ { m i }$ denote its probability of a correct response.

The Rasch, or one-parameter logistic (1PL), model separates an overall model efect from item easiness (Rasch, 1960):

$$
\mathrm { l o g i t } ( p _ { m i } ) = \theta _ { m } + \beta _ { i } .\tag{1}
$$

Here, $\theta _ { m }$ represents overall model performance and $\beta _ { i }$ represents item easiness; equivalently, $- \beta _ { i }$ is item dificulty. A model with a larger $\theta _ { m }$ has a higher predicted probability of success, and an item with a larger $\beta _ { i }$ is easier for the evaluated models.

The two-parameter logistic (2PL) model adds item discrimination (Lord, 1980):

$$
\mathrm { l o g i t } ( p _ { m i } ) = a _ { i } ( \theta _ { m } + \beta _ { i } ) , \qquad a _ { i } > 0 .\tag{2}
$$

Discrimination allows some items to distinguish models with higher and lower overall performance more sharply than others. The term $a _ { i } \theta _ { m }$ allows the efect of overall model performance to vary across items, while retaining a common ordering: a model with a higher $\theta _ { m }$ has a higher predicted probability of success on every item.

Models can also difer systematically in which items favor them. One model may perform relatively well on one group of questions and relatively poorly on another, even when its overall performance resembles that of a second model. LSIRM represents these diferences by extending the Rasch model with a distance-based model–item interaction (Jeon et al., 2021):

$$
\mathrm { l o g i t } ( p _ { m i } ) = \theta _ { m } + \beta _ { i } - \| z _ { m } - w _ { i } \| .\tag{3}
$$

The coordinates $z _ { m }$ and $w _ { i }$ place model m and item i in a shared space. Their Euclidean distance contributes to the response logit alongside the overall model and item efects. Holding those efects fixed, a shorter distance implies a higher predicted probability of a correct response. We use a two-dimensional latent space and fix the distance multiplier to one $( \gamma = 1 )$ . The model and item coordinates are estimated jointly with the other parameters and are not standardized to unit variance. Thus, fixing the multiplier still allows the estimated distances to vary in magnitude.

## 2.2 Interpreting the interaction map

The fitted coordinates form an interaction map. LSIRM directly models distances between models and items. The shared geometry also gives meaning to neighborhoods: nearby items have similar distance profiles to the model cohort, and nearby models have similar distance profiles to the item set. These relationships help identify groups of items that distinguish models with similar overall performance.

The map describes the interaction contribution. Predicted correctness also depends on the overall model and item efects in Equation 3. We therefore examine distances alongside observed accuracy and item properties. Rotating or reflecting the map preserves its distances and fitted probabilities, so we omit coordinate axes and interpret relative positions.

Figure 2 illustrates this representation for 1,348 language models and 5,761 BBH items. Panel A shows the full map. Panel B highlights Navigation, Object Counting, and Temporal Sequences, retaining all 250 items from each task. These annotations connect regions of the map to the questions being evaluated and reveal variation among items within a task.

Panel C colors models by observed BBH accuracy. Panel D compares the distances of two similarly accurate models, OpenBuddy and Smaug, to each item. An item closer to one model has a more favorable distance contribution for that model. The comparison directs attention to items for which the two models’ fitted interaction contributions difer.

## 2.3 From a map pattern to a diagnostic question

Navigation provides an example of how to investigate a pattern in the map. Its items occupy two separated concentrations. Inspection of the item records shows that one contains the 145 items whose correct answer is “No,” whereas the other contains the 105 items whose correct answer is “Yes.” Correct-answer identity therefore distinguishes the two groups within this task.

Interpreting this separation requires examining both the items and the evaluation procedure. Items with diferent correct answers may difer in their content or demands. Models may also perform diferently depending on how answers are presented and scored. The task-by-task analysis in Section 4 examines whether the association extends across BBH (Figure 3), and the protocol comparisons assess its dependence on evaluation procedures.

We then examine the consequences for model comparison. By estimating each model’s tendency to perform better or worse on items with particular correct answers, we can report the associated

D Relative distance to two models  
A BBH interaction map  
![](images/809d48e43e1694c393cc4cea0d26b023be0b8ba284559f50504da9caa85d6bc8.jpg)

B Selected BBH tasks  
![](images/70d7afb5f5d4ed96574bdefe8ca79c1b380a7870d48d2c46f57adcceb3723419.jpg)

Object counting (250)

Temporal sequences (250)

C Observed BBH accuracy  
![](images/6f25f2439ddc804230fc0ddd5b0c9ac55adf00ac59ffb1de75fd9bc91433d593.jpg)

![](images/171ae0afb44fde114b6669bae00aa14a473a40ffa5ff18b8a15d3d4653fe5b22.jpg)  
Figure 2: Interpreting a BBH interaction map through item and model annotations. All panels retain identical fitted positions, orientation, and scale. (A) The full map contains 1,348 models and 5,761 items. (B) Navigation, Object Counting, and Temporal Sequences are highlighted by task, retaining all 250 items per task. (C) Model points are colored by observed accuracy across the 5,761 items. (D) Open circles identify OpenBuddy and Smaug. Item colors represent distance to Smaug minus distance to OpenBuddy: red indicates closer proximity to OpenBuddy, blue to Smaug, and white equal distance. Color intensity reflects the size of the diference. Fitted correctness probabilities combine model–item distances with the model and item efects.

contribution separately and inspect the interaction that remains. Scores calculated with and without this contribution show how the choice of included components afects model rankings.

## 3 Data and methods

We investigate the association between BBH item neighborhoods and correct answers, then estimate how that association contributes to model profiles and rankings. A separate four-benchmark evaluation examines whether changing answer presentation or scoring changes the interaction pattern. Held-out prediction assesses whether interaction improves response prediction after task and correct-answer diferences are represented.

## 3.1 BBH data and item neighborhoods

The main analysis uses item-level scored responses from the Hugging Face Open LLM Leaderboard (Fourrier et al., 2024). The retained BBH matrix (Suzgun et al., 2023) contains 1,348 language models and 5,761 items, coded 1 for correct and 0 for incorrect. These models form the union of 50 overlapping stratified draws of 100 models from a leaderboard-matched pool of 3,732 models. The retained data contain 24 task labels, including count-specific task variants.

All BBH decomposition and ranking comparisons use the same correctness matrix, model cohort, item order, and equal item weights. We annotate the initial LSIRM map with task labels and correct answers after fitting, then inspect the corresponding correctness records to interpret the neighborhoods. The correct-answer annotation identifies what an item requires; the response record indicates whether a model received credit.

We assess answer association within tasks because tasks difer in both content and available answers. For each item, we identify its ten nearest neighbors from the same task, excluding the item itself. In Navigation, for example, this asks whether an item whose correct answer is “Yes” has other “Yes” items among its nearest Navigation neighbors.

Let O denote the average proportion of neighbors sharing the item’s correct answer, and let N denote the mean agreement under 200 permutations of correct-answer annotations within tasks. We calculate

$$
A = { \frac { O - N } { 1 - N } } .\tag{4}
$$

A value of zero corresponds to the permutation expectation; a value of one indicates that every selected neighbor shares the item’s correct answer. For BBH summaries, we pool observed and permuted agreement with equal item weights before normalizing. The same permutation bank is used across fitted specifications.

## 3.2 Answer presentation and scoring

An association with correct answers may depend on how the evaluation presents and scores those answers. We examine this possibility in a separate evaluation of 18 open-weight models on 2,399 items from BBH, GPQA, MMLU-Pro, and MuSR (Suzgun et al., 2023; Rein et al., 2023; Wang et al., 2024b; Sprague et al., 2024). The items form 30 blocks defined by domain and number of answer options. One GPQA item was excluded across all models and conditions; its identity and recorded exclusion reason appear in Appendix A.4.

We compare three protocols: text scoring without labels, text scoring with labels, and label scoring. Table 1 illustrates their diferences with a constructed question, “Which animal is a mammal?” The options and their order remain the same across conditions.

The two text-scoring protocols compare the likelihoods of the same full option strings. Their contrast measures the efect of displaying labels while keeping text scoring fixed. Text scoring with labels and label scoring use identical prompts but compare diferent continuations. In the example, the correct candidate is “bottlenose dolphin” under text scoring and “(B)” under label scoring.

Table 1: Constructed example of the three evaluation protocols. The question is “Which animal is a mammal?” The final column lists the candidate continuations whose likelihoods are compared. Parenthetical codes identify the corresponding conditions in the analysis records.
<table><tr><td>Protocol</td><td>Options displayed</td><td>Candidates scored</td></tr><tr><td rowspan="5">Text scoring without labels (U0)</td><td>great white shark</td><td>great white shark</td></tr><tr><td>bottlenose dolphin</td><td>bottlenose dolphin</td></tr><tr><td>rainbow trout</td><td>rainbow trout common octopus</td></tr><tr><td>common octopus</td><td></td></tr><tr><td>(A) great white shark</td><td>great white shark</td></tr><tr><td rowspan="5">(LT0)</td><td>(B) bottlenose dolphin</td><td>bottlenose dolphin</td></tr><tr><td>(C) rainbow trout</td><td>rainbow trout</td></tr><tr><td>Text scoring with labels (D) common octopus</td><td>common octopus</td></tr><tr><td></td><td></td></tr><tr><td>(A) great white shark (B) bottlenose dolphin</td><td>(A) (B)</td></tr><tr><td rowspan="2">Label scoring (L0)</td><td>(C) rainbow trout</td><td>(C)</td></tr><tr><td>(D) common octopus</td><td>(D)</td></tr></table>

For each protocol, the evaluator selects the candidate with the largest summed conditional log likelihood, without length normalization. The comparison between text and label scoring therefore includes diferences in continuation length and tokenization.

We fit LSIRM separately to each protocol and assess alignment within each domain-by-optioncount block. For the text-scoring protocols, categories identify the correct option’s position; for label scoring, they identify its label. These categories correspond under the original identifier assignment. Equation 4 is applied within each block, and the normalized values are averaged equally across blocks.

All 2,399 items enter estimation. Primary alignment contrasts summarize the 2,159 items in 27 blocks with at least three options; the display contrast across all 30 blocks is reported separately. Conditional 95% intervals use 10,000 paired block-bootstrap draws with fitted maps held fixed.

Identifier reassignment provides a direct check of which choices models retain. We compare label scoring under the original identifiers with a condition that changes identifiers while preserving option wording and position. Across 43,182 paired choices, we record whether each model retains the same option or the same identifier. Retaining an option also retains its position in this design.

To assess cohort sensitivity, we repeat the fits with and without SmolLM2-360M-Instruct on identical items. We also fit the label-scoring data with model-specific efects for ten identifier groups to examine changes in the map and rankings after representing that association explicitly.

## 3.3 Representing the answer-associated contribution

The Navigation example suggests that correct-answer identity can distinguish items within a task. We quantify this association by allowing each model to have a diferent efect for each correct-answer category. For example, a model can have a positive efect for items whose correct answer is “Yes” and a negative efect for those whose correct answer is “No.”

We compare three LSIRM specifications. Let $c _ { i }$ denote item i’s correct-answer category and $t _ { i }$ its task. For specification k, write $\eta _ { m i } ^ { ( k ) } = \log \mathrm { i t } ( p _ { m i } ^ { ( k ) } )$ . The baseline LSIRM, M0, estimates overall model efects, item easiness, and model–item distances. M1 adds an answer efect shared across tasks. M2 allows that efect to difer by task:

$$
\mathrm { M 0 } ; \quad \eta _ { m i } ^ { ( 0 ) } = \theta _ { m } ^ { ( 0 ) } + \beta _ { i } ^ { ( 0 ) } - \| z _ { m } ^ { ( 0 ) } - w _ { i } ^ { ( 0 ) } \| ,\tag{5a}
$$

$$
\mathrm { M 1 } \colon \ \eta _ { m i } ^ { ( 1 ) } = \theta _ { m } ^ { ( 1 ) } + \beta _ { i } ^ { ( 1 ) } - \| z _ { m } ^ { ( 1 ) } - w _ { i } ^ { ( 1 ) } \| + \underbrace { \delta _ { m , c _ { i } } ^ { ( 1 ) } } _ { \mathrm { V o r } } \quad ,\tag{5b}
$$

$$
\mathrm { M 2 } \colon \eta _ { m i } ^ { ( 2 ) } = \theta _ { m } ^ { ( 2 ) } + \beta _ { i } ^ { ( 2 ) } - \| z _ { m } ^ { ( 2 ) } - w _ { i } ^ { ( 2 ) } \| + \underbrace { \delta _ { m , ( t _ { i } , c _ { i } ) } ^ { ( 2 ) } } _ { \mathrm { t a s k - a n s w e r ~ e f f e c t } } .\tag{5c}
$$

In M1, each model has one efect for each correct-answer category. Its efect for “Yes,” for example, is shared across tasks with that category. M2 estimates a separate efect for each task–answer combination, allowing the association with ${ } ^ { 6 6 } \mathrm { Y e s } ^ { 9 9 }$ to vary across tasks.

M1 contains 41 answer groups and provides the primary decomposition. M2 contains 123 task–answer groups and examines whether allowing answer efects to vary by task changes the findings. Numeric Object Counting answers remain distinct in both specifications. The answer-family summaries group these 17 numeric categories under “Other.”

For each model, the efects are centered equally across all indexed groups:

$$
\sum _ { c \in \mathcal { C } } \delta _ { m , c } ^ { ( 1 ) } = 0 , \qquad \sum _ { ( t , c ) \in \mathcal { G } } \delta _ { m , ( t , c ) } ^ { ( 2 ) } = 0 ,\tag{6}
$$

where C contains the 41 answer categories and $\mathcal { G }$ the 123 observed task–answer combinations.

The superscripts distinguish separately estimated fits. Each specification re-estimates model efects, item efects, and coordinates jointly with its answer terms. For the summaries below, we suppress the superscripts when referring to a single fit and write $I _ { m i } = - \| z _ { m } - w _ { i } \|$ for its interaction contribution.

We examine the answer efects alongside the interaction that remains after joint estimation. For an item subset T, such as the 250 Navigation items, the relative interaction profile is

$$
R _ { m T } = \frac { 1 } { | T | } \sum _ { i \in T } I _ { m i } - \frac { 1 } { J } \sum _ { i = 1 } ^ { J } I _ { m i } ,\tag{7}
$$

where $J = 5 , 7 6 1$ . A positive value indicates a more favorable interaction contribution for that subset than for the model’s average item. Observed accuracy and the estimated answer contribution accompany these profiles.

## 3.4 Scores and ranking comparisons

The decomposition allows us to compare scores that include or exclude the estimated answer contribution. Let $g ( i ) = c _ { i }$ in M1 and $\boldsymbol { g } ( i ) = ( t _ { i } , c _ { i } )$ in M2. Define the item-weighted means

$$
\bar { I } _ { m } = \frac { 1 } { J } \sum _ { i = 1 } ^ { J } I _ { m i } , \qquad \bar { \delta } _ { m } = \frac { 1 } { J } \sum _ { i = 1 } ^ { J } \delta _ { m , g ( i ) } .
$$

We calculate

$$
s _ { m } = \theta _ { m } + \bar { I } _ { m } , \qquad q _ { m } = s _ { m } + \bar { \delta } _ { m } .\tag{8}
$$

The score $s _ { m }$ combines the overall model efect and average interaction; $q _ { m }$ also includes the mean answer contribution. In M0, we set $\bar { \delta } _ { m } = 0$ , so the scores coincide. Although answer efects are centered equally across groups, their item-weighted mean can difer from zero because groups contain diferent numbers of items.

Both scores are on the logit scale. The mean fitted logit is $q _ { m } + \bar { \beta } .$ where $\bar { \beta }$ is the mean item easiness efect. Within a fit, every model is evaluated on the same items, so this common item term does not afect the ranking. Mean predicted accuracy would instead average the fitted probabilities across items.

Within-fit comparisons hold every estimate fixed and rank models by $q _ { m }$ and $s _ { m }$ . Excluding the answer contribution changes each score by exactly − $\cdot \bar { \delta } _ { m } \colon$ a positive contribution lowers the score when removed, whereas a negative contribution raises it. Comparing the resulting rankings shows how ordering depends on including that contribution.

Between-fit comparisons rank models by $s _ { m }$ from M0, M1, and M2. Here, adding answer terms leads to re-estimation of model efects, item efects, and coordinates. Score diferences therefore include changes in the overall model efects and average interactions. The additive decomposition applies within each fitted specification.

We rank unrounded scores in descending order and assign average ranks to exact ties. Every BBH comparison retains the same 1,348 models and 5,761 equally weighted items. Full-cohort summaries describe the distribution of rank changes, while selected examples show contrasting outcomes.

For the protocol evaluation, we calculate $s _ { m }$ over all 2,399 items. Rankings from label scoring, with and without identifier efects, are compared with text scoring without labels as an alternative protocol reference. We also re-rank the shared 17 models within the 18-model fits, separating changes among those models from insertion of the additional model. The intended use of an evaluation determines which protocol and score components should define its ranking.

## 3.5 Prediction and numerical stability

Held-out prediction assesses whether interaction captures response variation after diferences across tasks and correct answers have been represented. We use one fixed 100-model BBH draw, selected among the 50 draws for performance and metadata representativeness. A model-by-task–answer specification is compared with its two-dimensional interaction extension using paired held-out log loss and Brier score.

Each of ten overlapping masks withholds 57,400 responses and retains 518,700 for estimation. The masking pool contains 5,740 items with at least two observed successes and two failures; the other 21 items remain in training. All 5,761 items enter each fit, and held-out responses enter neither estimation nor initialization. Both specifications use the historical 107-group coding, which combines numeric object-counting answers into one task-specific group. The comparison evaluates response prediction within this fixed model cohort and item pool.

We fit all specifications by minimizing the summed Bernoulli negative log likelihood with quadratic penalties using L-BFGS-B. The spread of the estimated coordinates is determined by the response likelihood and the fixed quadratic penalties specified in the supplementary methods. No separate coordinate-variance parameter is estimated. Numerical acceptance requires first-order stationarity and agreement between independently computed objectives and gradients. Additiona random starts and local perturbations assess the sensitivity of the BBH decomposition, alignment, and rankings. Where multiple accepted solutions are available, we retain the one with the lowest original penalized objective. Penalty coeficients, initialization procedures, numerical tolerances, and starting-value comparisons appear in Appendix A.1.

## 4 Results

## 4.1 Correct-answer categories distinguish item neighborhoods within tasks

Items from the same BBH task can occupy diferent regions of the interaction map depending on their correct answers (Figure 3). This separation is especially clear in Navigation and Web of Lies, where items with “Yes” as the correct answer form a diferent neighborhood from those with “No” as the correct answer. Other tasks, including Date Understanding and Logical Deduction with seven objects, show more overlap between answer categories. The task-by-task view therefore reveals both the answer-associated pattern and diferences in its strength across tasks.

The alignment measure in Equation 4 summarizes how strongly correct-answer categories distinguish item positions within tasks. In the baseline fit, alignment is highest for Yes/No items (0.9629), followed by True/False (0.8935), Valid/Invalid (0.7009), and letter answers (0.3650). These diferences motivate a closer examination of the response patterns behind the map.

Web of Lies provides a concrete example. Its two item neighborhoods correspond to sharply diferent performance profiles among individual models (Figure 4). BgGPT answers all 122 items whose correct answer is “Yes” correctly and none of the 128 whose correct answer is “No.” Hebrew-Gemma shows the opposite pattern, answering none of the “Yes” items and all 128 “No” items correctly. The calme model answers most items correctly in both groups: 111 of 122 “Yes” items and 118 of 128 “No” items. These correctness profiles connect the item neighborhoods to diferences among models; full model identifiers appear in Table S3.

## 4.2 Scoring labels increases answer alignment at a fixed prompt

The four-benchmark evaluation shows that the scored answer representation afects alignment even when the prompt is unchanged (Figure 5). Across the 27 matched blocks containing 2,159 items, label scoring produces greater alignment than text scoring with labels. The increase is 0.0630 in the 17-model cohort, with a conditional 95% interval of [0.0365, 0.0945]. Including the eighteenth model gives an increase of 0.0908 [0.0534, 0.1338] (Table S6). These intervals describe variation across blocks with the fitted maps held fixed.

Displaying labels while continuing to score option text has a smaller, negative estimated contrast: −0.0192 [−0.0419, 0.0011] for 17 models and −0.0227 [−0.0446, −0.0036] for 18 models. The increase in alignment thus occurs when the evaluator switches from scoring option text to scoring labels under the same labelled prompt. Under the raw likelihood rule used here, that switch also changes the length and tokenization of the scored continuation.

The choice records distinguish this map pattern from persistence in selecting a particular label. After labels are reassigned, models retain the same option in 57.27% of the 43,182 paired choices and the same label in 18.86%. Retaining the option is more common for 15 of the 18 models. Option wording and position remain fixed in this comparison, so retaining the option also retains its position. Greater alignment under label scoring therefore coexists with more frequent preservation of the selected option than of its label.

## BBH interaction map by task and correct answer

1,348 models; 5,761 items. Gray crosses: models. Gray dots: other items.

![](images/531ca2cf8db61a40dbbf76a152c4061cbd845a8c233109fe0eac55a2368ed0ce.jpg)  
Figure 3: BBH interaction map annotated by task and correct answer. The map was estimated from the correctness records of 1,348 models on 5,761 items. Each panel highlights the items from one task, with colors and marker shapes indicating their correct answers. Gray crosses represent models; gray dots represent items from other tasks. Panel titles report task item counts. All panels use the same fitted positions, orientation, and scale. Correct-answer categories were added as annotations after fitting. Coordinate axes are omitted because their orientation has no substantive interpretation.

a Answer-associated structure in the item map M0: 1,348 models; all 5,761 items  
![](images/7ca83f099d0007913f476df6f0eb4dd2da72059d0f0204d2e22d74d465e56560.jpg)

![](images/8f6ae8191dc98b2e8d22759adbfde852475e79e474c5a283bccdf939637995f7.jpg)  
c Observed correctness by gold key The same 250 web-of-lies items

![](images/cc7b0dd016eba76e524f87071fa9e61ef4327082a1a0e9a14e0aea6594f7960c.jpg)  
Figure 4: An LSIRM map connects model–item interactions to observed correctness patterns. a, The selected M0 fit uses 1,348 models and 5,761 BBH items. All item positions are retained in gray; color highlights the 250 Web of Lies items by their correct answer (gold key) (122 Yes, 128 No). Crosses locate three previously selected illustrative models. The map retains the saved coordinate scale and orientation; model–item distance contributes negatively to the fitted log odds of a correct response. b, Normalized answer alignment in each of the five answer families, using ten within-task item neighbors and 200 task-preserving answer-label permutations. Alignment is $A = ( O - N ) / ( 1 - N )$ , where O is observed agreement and N is mean agreement under permutation; values near one indicate greater local agreement than the permutation reference. Points describe the selected fit, without sampling intervals. c, Observed correctness for the same three models and 250 highlighted items. Cells report percentages and exact correct/total counts; shading encodes correctness. BgGPT and Hebrew-Gemma have opposite correctness patterns across correct-answer categories, whereas calme answers most items correctly under both keys. These records retain correctness and correct answers, not the models’ chosen answer strings. Full model identifiers are listed in Table S3.

This dependence on answer representation is consistent with previous findings on option-label preferences and evaluation procedures (Zheng et al., 2024; Wang et al., 2024a; Nowak et al., 2026). The present comparison shows how that dependence appears in an LSIRM map under the specified likelihood scoring rules. Together with the BBH response profiles, it motivates estimating the contribution associated with correct-answer categories explicitly.

## 4.3 The decomposition distinguishes models with similar accuracy

Adding answer efects reduces the association between correct answers and the remaining item geometry (Figure 6). On the same BBH response matrix, Yes/No alignment falls from 0.9629 in M0 to 0.3677 with pooled answer efects in M1 and 0.1077 with task–answer efects in M2. True/False alignment falls from 0.8935 to 0.0496 and 0.0352, respectively. All five answer families have lower alignment under both specifications than under M0 (Table S2). These reductions occur after joint re-estimation of the answer terms, model and item efects, and interaction coordinates.

Phi and DeepSeek illustrate how this decomposition distinguishes models with similar overall accuracy. They answer 3,253 and 3,239 items correctly, respectively, a diference of 14 out of 5,761 items. Yet on 2,158 items, exactly one of the two models is correct. Their similar totals therefore combine many opposing successes and failures.

In M1, the estimated mean answer contribution is +0.838 logits for Phi and −1.302 for DeepSeek. The contribution raises Phi’s answer-included score q relative to its answer-excluded score s and lowers DeepSeek’s. Figure 6a and Table S4 show how the overall model efect, mean interaction, and mean answer efect combine for each model. All components are calculated over the same 5,761 items and add on the fitted logit scale.

The remaining interaction profiles locate diferences within each model’s performance pattern. Among Yes/No items, Phi has a relative interaction of +0.571 and a mean answer efect of −0.053; DeepSeek’s corresponding values are +0.055 and −2.237. Relative interaction compares an item group with that model’s own all-item mean. It therefore describes where the distance contribution is more or less favorable within the model’s profile.

Object Counting shows why this relative profile should be read alongside observed accuracy. Phi has a larger relative interaction than DeepSeek, 0.787 versus 0.151, while its observed accuracy is lower, 39.2% versus 61.2%. The mean answer contributions are positive for both models, at 0.741 and 1.389. Fitted probabilities combine these terms with the overall model and item efects. The complete task profiles also retain the contrasts in adjective order, or hyperbaton (92.0% versus 56.0% accuracy), and geometric shapes (21.6% versus 55.6%; Figure S1).

Interaction also contributes to held-out prediction after task–answer efects are included. In the separate 100-model cohort with 107 task–answer groups, adding two-dimensional interaction lowers mean held-out log loss from 0.3538 to 0.3320, with improvement in every one of the ten overlapping masks. Mean Brier score falls from 0.1118 to 0.1047. Mask-level log losses appear in Table S5. The gain concerns responses held out from the same model cohort and item pool.

## 4.4 Including the answer contribution changes model rankings

The opposite answer contributions for Phi and DeepSeek produce opposite ranking movements within M1 (Figure 7). With all estimates held fixed, excluding the answer contribution lowers Phi’s score by 0.838 logits and raises DeepSeek’s by 1.302 logits. Re-ranking the same 1,348 models moves Phi from 396 on q to 814 on s, and DeepSeek from 344 to 97. Their answer-excluded ranks are 717

b Paired alignment contrasts 27 matched blocks; conditional 95% intervals a Constructing the M1 scores s = θ + mean I; q = s + mean δ

a Separate displayed identifiers from the scored answer
<table><tr><td>Arm</td><td>Option display</td><td>Scored continuation</td><td>Scoring: L0 – LT0</td></tr><tr><td>uo</td><td>No identifiers</td><td>Option text</td><td>Same labelled prompt</td></tr><tr><td>LTO</td><td>Identifiers printed</td><td>Option text</td><td>Printing: LT0 – U0</td></tr><tr><td>LO</td><td>Identifiers printed</td><td>Identifier</td><td>Same text-scoring rule</td></tr></table>

![](images/51454bb7324835b93f1ee318928dd1beb6823033e27f43e71fd2145354a42b12.jpg)  
c Choices after identifier reassignment 18 models × 2,399 items; fixed option order

![](images/1d3b3a99ceb3fd193edd40d9db9c1dd166ffbcccb33bc11e95b75973527fae2b.jpg)

![](images/bc63edec9e554e8a098acb46b655203b6b002c15efad2d63cc6ffde54ea81d74.jpg)  
Figure 5: The scored answer representation changes the diagnosed structure. a, U0 scores option text without printed identifiers; LT0 prints identifiers and retains text scoring; L0 uses the same labelled display and scores the identifier. Selections maximize raw summed continuation log likelihoods without length normalization. b, Mean paired diferences in normalized alignment across 27 matched domain-by-option-count blocks (2,159 items, at least three options). The 17-model cohort excludes SmolLM2-360M-Instruct; the 18-model cohort includes it. All fits use the same 2,399 items in 30 blocks. Identifier-minus-text contrasts are 0.0630 and 0.0908. Printing identifiers has smaller negative contrasts, −0.0192 and −0.0227. Bars are conditional 95% percentile intervals from 10,000 paired block-bootstrap draws, holding the selected fitted maps fixed; they exclude refitting and model-population uncertainty. c, Preservation of the selected content or identifier after the first identifier reassignment, with option wording and order fixed. All 18 models and all 2,399 items contribute (43,182 paired choices). Content is preserved in 57.27% of choices and the identifier in 18.86%; 15 of 18 models preserve content more often. Content and position coincide under fixed ordering. Points are observed proportions, with no sampling intervals. Model labels omit the instruction-tuned sufix; the base Qwen model is marked explicitly.

![](images/465b742dc12d6410ae76744aed6d3f42efa6c18770d23626b02f6a4a926629fe.jpg)  
b The remaining profile after conditioning

![](images/b3467afbe66a578f3a7afda16718253065b7e5183191b0b3ab1bca8b7b56b7ca.jpg)  
c Alignment across specifications

![](images/6a87bed14ee1c444691f9e97a9ff876c4dbd74e04e9b75a59cb38c8d0cf063ba.jpg)

d Interaction predicts held-out responses Separate fixed cohort: 100 models; 10 masks  
![](images/ca4fec7f2cfba2100f4365964d43f55bde2c5fdbc7b1b80a439752cff05dbc5f.jpg)  
Figure 6: Conditioning gives explicit score components and a remaining performance profile. a, Signed M1 components for Phi-3-mini-4K-instruct-cpo-simpo (Phi), DeepSeek-R1-Distill-Llama-70B (DeepSeek), and QwQ-LCoT1-Merged (QwQ). Dark bars mark $s _ { m } = \theta _ { m } + I _ { m }$ and $q _ { m } = s _ { m } + \bar { \delta } _ { m } ;$ lighter bars mark their constituent terms. Means weight all 5,761 items equally. Both scores omit the common mean item efect. The components add on the logit scale within the selected fit. Scores are calculated before rounding. b, Mean answer ofsets and relative interactions across all five answer families. Relative interaction subtracts each model’s all-item mean interaction; positive values indicate a less negative interaction in that family. Family sizes are 4,074 (Letter), 937 (Yes/No), and 250 each (True/False, Valid/Invalid, Other). Other contains the numeric Object Counting answers. These fitted profiles complement observed accuracy; all 24 task profiles appear in Supplementary Figure S1. c, Answer-key alignment in M0, M1 and M2 for the same 1,348-model matrix, normalized against 200 task-preserving key permutations. d, Separate fixed cohort of 100 models: ten overlapping paired masks, each with 57,400 held-out responses. Gray lines join task–answer-only fits and their interaction extensions, using the historical 107 task–answer groups; dark diamonds mark means. Mean log loss falls from 0.3538 to 0.3320, improving in every mask. Panels a–c show selected numerically accepted solutions without sampling intervals; saved-start sensitivity is reported separately.

places apart.

Each score change follows directly from the model’s mean answer contribution. Rank movement also depends on how the other models’ scores change. QwQ provides a low-movement example, selected by the rule in Appendix A.3: excluding its answer contribution moves its rank from 438 to 441. Across the full cohort, the median absolute within-fit movement is 61 places for M1 and 22 for M2 (Figure 7b).

Refitting the model produces a separate ranking comparison. Between M0 and M1, Phi moves from 368 to 814 on s, DeepSeek from 539 to 97, and QwQ from 436 to 441. These changes include re-estimation of the overall model efects and interaction after answer terms are added.

Across all models, the median absolute movement from M0.s is 70 places for M1.s and 31 for M2.s. The corresponding rank correlations are 0.9190 and 0.9842. Of the models in M0’s top 50, 28 remain in M1’s top 50 and 40 in M2’s (Table S1). Both the fitted specification and the decision to include its answer contribution therefore afect model ordering.

The main ranking contrasts persist across the saved local solutions. Among the preceding best M1 fit and four reoptimized perturbations, Phi’s s rank ranges from 814 to 816 and DeepSeek’s remains at 97. All five answer-family alignment summaries remain below M0 across the saved candidates for both M1 and M2. Detailed profiles show more variation; Tables S9 and S10 report the component and ranking ranges.

The protocol evaluation shows a further consequence of changing the fitted specification. Adding label efects and comparing the resulting s rankings reduces agreement with text scoring without labels. In the 17-model cohort, Spearman correlation with that reference falls from 0.9804 to 0.8824, while mean absolute rank distance rises from 0.588 to 1.765 places. The corresponding 18-model values are 0.9835 to 0.9009 and 0.556 to 1.667 (Figure 7c; Table S7).

Including the eighteenth model leaves the ordering of the shared 17 models unchanged in the selected fits; SmolLM2-360M-Instruct ranks eighteenth under each compared specification. In this evaluation, fitting label efects and ranking models by the resulting answer-excluded scores does not recover the ordering obtained by scoring option text. The ranking consequence of representing an answer efect is therefore distinct from the consequence of changing the scoring protocol.

## 5 Discussion

LSIRM helps evaluators examine diferences in benchmark performance that overall accuracy conceals. By accounting for overall model performance and item dificulty, it identifies patterns in which models succeed on which items. This study demonstrates how to interpret those patterns using item properties and response records, investigate the role of evaluation procedures, and quantify the contribution associated with a particular item property. The BBH example connects this diagnostic process to concrete outcomes: model profiles that distinguish similarly accurate models and score comparisons that show how including or excluding an estimated contribution changes their rankings.

The task-by-task maps reveal that correct-answer categories distinguish item neighborhoods within several BBH tasks. Previous research provides reasons to investigate this pattern. Studies have documented preferences for option identifiers, diferences between generated-answer and firsttoken evaluation, and sensitivity to label presentation and prediction mode (Zheng et al., 2024; Wang et al., 2024a; Nowak et al., 2026). Our comparison across BBH, GPQA, MMLU-Pro, and MuSR examines how these evaluation choices afect the interaction maps. With the labelled prompt held fixed, scoring option identifiers produces stronger alignment with correct-answer categories than

a Similar accuracy, different ranking consequences

Open circle: starting rank • Filled circle: ending rank; accuracy in parentheses

![](images/8538f0066529e6d8f20ce5d5400637281dbd0cb0c7a03d13217dacdbaafed6ff.jpg)  
b Rank movement over the complete cohort

![](images/57251c5216e5d1038498335459eed9df841496c50deb98bc90fc3465175fbe04.jpg)  
c Conditioning reduces agreement with the U0 reference  
Open circle: L0 • Filled circle: L0 conditioned

![](images/dcc11f47b3dfb4f4a4bb1b5136a09f47cd664c0313d4b6370912148f33997c8b.jpg)

Figure 7: Accounting for answer-associated components changes model comparison. a, Selected examples distinguish re-estimation across fits (M0.s to M1.s) from removing the explicit answer contribution within M1 (q to s). Open circles show starting ranks and filled circles ending ranks; labels give exact ranks and observed accuracy. Phi and DeepSeek difer by 14 correct answers out of 5,761 items but disagree on 2,158 of 5,761 correctness records. Their M1.s ranks are 814 and 97, a 717-place separation. QwQ is a low-movement comparator selected by the documented rule. b, Empirical cumulative distributions retain all 1,348 models for each comparison. Blue solid curves use M1 and purple dashed curves M2. Median absolute movements are 70 and 31 places across fits, and 61 and 22 places within fits. Ranks are computed from unrounded scores, with average ranks for exact ties; lower ranks indicate larger scores. c, Between-fit agreement of unconditioned and conditioned L0.s ranks with U0.s, using all 2,399 rank-evaluation items. The 17- and 18-model rows each use their complete cohort. Spearman correlation decreases from 0.980 to 0.882 for 17 models and from 0.983 to 0.901 for 18 models; mean absolute rank distance increases from 0.59 to 1.76 and from 0.56 to 1.67, respectively. The shared 17 models have identical selected ranks in both cohort analyses. The display describes ranking consequences under specified fits and protocols; U0 is an alternative evaluation reference rather than an external criterion of correct ability ranking.

scoring option text. Agreement between the map, previous findings, and this controlled comparison supports interpreting part of the observed structure in relation to answer representation and scoring.

The decomposition makes the association with correct answers explicit for individual models. Phi and DeepSeek difer by only 14 correct responses out of 5,761 items, yet their correctness records difer on more than a third of those items. Their estimated mean answer contributions have opposite signs, and their remaining interaction profiles difer across tasks and answer families. These outputs help explain how models with similar totals can have diferent patterns of performance. Reading the fitted contributions alongside observed task accuracy shows where an answer-associated pattern accompanies a performance diference and where further diferences remain. Because the model efects, item efects, answer efects, and interactions are estimated jointly, the decomposition depends on the chosen specification. An answer efect can combine associations with answer content, option labels, and option positions.

The ranking comparisons show the consequences of deciding which fitted contributions enter a score. Within one fit, comparing the score that includes the answer contribution (q) with the score that excludes it (s) holds every parameter estimate fixed. The resulting rank changes isolate the efect of that scoring decision. Comparing scores across separately fitted specifications also reflects changes in the estimated model efects and interactions. In the crossed evaluation, this between-fit adjustment reduced agreement with the ranking obtained by scoring option text without labels in both the 17- and 18-model cohorts. Fitting answer efects and ranking models by the resulting answer-excluded scores did not recover that reference ordering. A preferred ranking requires a stated evaluation target, such as performance under the protocol intended for deployment. The dependence of response-bias calibration on evaluation conditions provides a further reason to assess any adjustment against that target (Bhatt and Ivanova, 2026).

The remaining interaction also contains information useful for prediction. In the separate 100- model analysis, including interaction improved prediction for held-out responses after task–answer efects were included. This result supports retaining interaction when modeling responses within that cohort; interpreting particular neighborhoods requires evidence about their items and response patterns. The numerical checks address a diferent question. The selected fits met the stationarity criteria, and alignment and rankings remained similar across the examined local solutions, although some detailed components varied. Broad comparisons were more stable than every detail of the fitted profiles.

The analyses concern the models, items, and protocols represented in the data. The main BBH cohort combines overlapping draws, and the prediction analysis reuses one selected cohort under overlapping response masks. These designs support comparisons within the analyzed cohorts rather than population-wide estimates for language models. Incomplete version and collection metadata also limit reconstruction of the original benchmark evaluations.

For practical evaluation, an LSIRM report can present observed task accuracy, the estimated answer contribution, and the remaining interaction profile together. Evaluators can use the task profiles to examine performance relevant to an intended application and trace unusual patterns back to individual response records. Scores calculated with and without the answer contribution should use the same models, items, and item weights, so their comparison has a clear interpretation. Reporting both rankings makes the scoring decision visible and allows evaluators to judge which contributions belong in a comparison for their intended use.

Data and code availability. The supplementary material provides model identifiers, component and ranking summaries, and numerical checks. Analysis code, processed response matrices, and machine-readable source tables prepared for this study are not currently publicly available.

## References

Om Bhatt and Anna A. Ivanova. 2026. RBCorr: Response Bias Correction in Language Models. arXiv preprint arXiv:2602.12445. https://arxiv.org/abs/2602.12445.

Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. 2024. Open LLM Leaderboard v2. Hugging Face. https://huggingface.co/spaces/open-llm-leaderboard/open\_llm\_leaderboard.

Minjeong Jeon, Ick Hoon Jin, Michael Schweinberger, and Samuel Baugh. 2021. Mapping Unobserved Item–Respondent Interactions: A Latent Space Item Response Model with Interaction Map. Psychometrika, 86(2):378–403. https://doi.org/10.1007/s11336-021-09762-5.

John P. Lalor, Pedro Rodriguez, João Sedoc, and Jose Hernandez-Orallo. 2024. Item Response Theory for Natural Language Processing. Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Tutorial Abstracts, pages 9–13. https://doi.org/10.18653/v1/2024.eacl-tutorials.2.

Frederic M. Lord. 1980. Applications of Item Response Theory to Practical Testing Problems. Lawrence Erlbaum Associates, Hillsdale, NJ. https://doi.org/10.4324/9780203056615.

Mateusz Nowak, Xavier Cadet, and Peter Chin. 2026. ABCD: All Biases Come Disguised. arXiv preprint arXiv:2602.17445. https://arxiv.org/abs/2602.17445.

Georg Rasch. 1960. Probabilistic Models for Some Intelligence and Attainment Tests. Danish Institute for Educational Research, Copenhagen.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv preprint arXiv:2311.12022. https://arxiv.org/abs/2311.12022.

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. 2024. MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning. International Conference on Learning Representations. https://arxiv.org/abs/2310.16049.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051. https://doi.org/10.18653/v1/2023.findings-acl.824.

Clara Vania, Phu Mon Htut, William Huang, Dhara Mungra, Richard Yuanzhe Pang, Jason Phang, Haokun Liu, Kyunghyun Cho, and Samuel R. Bowman. 2021. Comparing Test Sets with Item Response Theory. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1141–1158. https://doi.org/10.18653/v1/2021.acl-long.92.

Xinpeng Wang, Chengzhi Hu, Bolei Ma, Paul Röttger, and Barbara Plank. 2024. Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think. Conference on Language Modeling. https://arxiv.org/abs/2404.08382v2.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. Advances in Neural Information Processing Systems, 37. https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. Large Language Models Are Not Robust Multiple Choice Selectors. International Conference on Learning Representations. https://arxiv.org/abs/2309.03882.

## A Supplementary methods and provenance

## A.1 Estimation and numerical checks

All retained models use the summed Bernoulli negative log likelihood with penalty

$$
0 . 0 5 \| \theta \| ^ { 2 } + 0 . 0 0 5 \| \beta \| ^ { 2 } + 0 . 2 2 2 ( \| Z \| _ { F } ^ { 2 } + \| W \| _ { F } ^ { 2 } ) + 0 . 0 2 \| \delta \| _ { F } ^ { 2 } ,\tag{9}
$$

omitting absent terms. Here, Z and W collect the model and item coordinates, respectively, and $\| \cdot \| _ { F }$ denotes the Frobenius norm. The implemented interaction is $- \sqrt { \operatorname* { m a x } ( d ^ { 2 } , 0 ) + 1 0 ^ { - 8 } }$ , where $d ^ { 2 }$ is the expanded squared Euclidean distance. Geometry has two dimensions and its multiplier is fixed to one. For each model, ofsets are centered equally over all indexed groups: 41 in BBH M1, 123 in BBH M2, 107 in the prediction comparison, and ten in crossed L0 conditioning. Task–answer coding does not imply separate centering within each task.

Initial SVD fits use marginal proportions clipped to [0.01, 0.99], additive logits, and residual SVD coordinates with square-root singular-value scaling. Held-out cells enter neither marginal calculations nor initialization. The BBH numerical sequence includes the initial multistart fits, additional random starts, and perturbations of the preceding best M1 and M2 solutions. Four local perturbations per conditioned specification were accepted. Larger perturbations found lower objectives, and the selected display solutions are those lower accepted endpoints. They were not themselves subjected to a further perturbation cycle. The local sensitivity summaries compare each preceding anchor with its four perturbation endpoints, giving five candidates per specification.

Acceptance requires original and independent maximum absolute gradients at most $1 0 ^ { - 4 }$ objective agreement within $1 0 ^ { - 6 }$ , gradient agreement within $1 0 ^ { - 7 }$ , centered-ofset row sums within 10<sup>−10</sup>, finite values, and a nonincreasing objective within $1 0 ^ { - 6 }$ . The selected M0, M1, and M2 gradients are all below 10<sup>−5</sup> (Table S8). All eight local perturbation endpoints meet the criteria, although recovery of the preceding objective difers by perturbation size. These checks establish stationarity of the retained endpoints under the stated objective. They do not establish a global optimum or a unique decomposition.

Across the five local candidates, the maximum diference in a family-alignment summary is 0.009063 for M1 and 0.000998 for M2. Phi’s M1.s rank ranges from 814 to 816, DeepSeek’s stays at 97, and QwQ’s ranges from 441 to 444. Their M2.s ranks remain 615, 229, and 434. Finer profiles can be less stable. For the 25 object-counting items with correct answer 8, the M1 relative-interaction range is 0.151 logits for Phi and 0.557 for DeepSeek. These are descriptive starting-solution ranges, not confidence intervals.

The prediction analysis retains 20 accepted fits, two specifications for each of ten masks. Two endpoints required tighter continuations to resolve independent numerical-agreement failures; the original record had accepted 18 of 20. The crossed check contains 16 accepted fits: four specifications in two cohorts, each using an SVD start and a start transferred from the other cohort. Reciprocal starts are related candidates, not independent random replicates. Selection always uses the original penalized objective, never alignment, rank movement, or held-out performance. The recorded fitting environment is Python 3.12.9, NumPy 2.5.4, SciPy 1.18.1, and JAX 0.12.0.1.

## A.2 Item weighting, prediction masks, and crossed sensitivity

BBH alignment uses ten nearest item neighbors within each task, excluding the focal item. Canonical item order resolves distance ties. We use a shared bank of 200 within-task permutations (seed 20261025), pool observed and null agreement with equal item weights within a reported family, and then apply Equation 4. The “Other” family contains Object Counting items and retains their 17 distinct numeric answer keys. For crossed maps, the same calculation is performed within each domain-by-option-count block using 200 shared within-block permutations (seed 20261026). Normalized block values are then averaged equally.

The crossed conditional intervals use 10,000 paired block-bootstrap draws. The same sampled blocks enter both arms of each contrast; fitted coordinates remain fixed. Seeds 20261027 and 20261028 generate the matched-27-block and full-30-block resampling banks, respectively; each bank is shared across arms and cohorts. The primary matched set has 27 blocks and 2,159 items. The full printing contrast includes 30 blocks and 2,399 items (Table S6). Scoring contrasts remain positive across the retained starting solutions. The selected increase in the scoring contrast after adding the eighteenth model is 0.0278, with conditional interval [0.0052, 0.0546]. Across all four starting-solution pairings the increase ranges from 0.0168 to 0.0295, and one interval includes zero. Full-30-block alignment is more start-sensitive than the primary matched-block summaries. Crossed rank comparisons vary by at most one place across retained starts; within-specification rank correlations are at least 0.99755 and top-five membership is unchanged.

Each prediction mask withholds 57,400 cells from the fixed 100 × 5, 761 matrix. Eligibility requires at least two observed successes and two failures per item, yielding 5,740 eligible items; the other 21 stay wholly in training. Mask 4 covers 5,739 distinct held-out items. Across ten masks there are 574,000 held-out occurrences but 373,751 distinct cells. Training-only estimation leaves 518,700 cells per mask. The historical 107-group coding collapses the 17 numeric object-counting keys into one task-specific group. Mask-level losses are reported in Table S5 without treating masks as independent populations.

## A.3 Example selection and response verification

Five previously named models were retained for verification: BgGPT, Hebrew-Gemma, calme, Phi, and DeepSeek. A sixth, QwQ, was selected to supply a concrete low-movement comparison. Eligible models had M0.s ranks within the central 80% of the cohort and observed accuracy between 0.2 and 0.7, excluding the five named models. Among the 1,076 eligible models, selection minimized the largest absolute movement across M0.s–M1.s, M0.s–M2.s, M1.q–M1.s, and M2.q–M2.s, breaking ties by total movement and then model identifier. This is a documented illustrative selection, not a population-sampling rule.

All 34,566 correctness cells for the six models were matched to their source item/model/score records, with no selected duplicate records. Component summaries were independently recomputed from the saved coordinates and parameters. For Phi and DeepSeek, 2,167 items are correct for both, 1,436 incorrect for both, 1,086 correct only for Phi, and 1,072 correct only for DeepSeek. Full task profiles and full identifiers accompany the manuscript. No chosen-answer string is inferred from a correctness-only record.

Four additional models provide the introductory comparisons (Figure 1). OpenBuddy and Smaug illustrate a reversal across tasks among two 70B models with similar BBH accuracy. The Mistral 7B Base-SFT-CPO and Base-SFT-RDPO pair illustrates a reversal across correct-answer categories within Ruin Names. These pairs were selected from the same response matrix to illustrate contrasting observed profiles. Each pair’s 11,522 correctness entries was separately matched to the source records, with no duplicate retained records. Table S3 lists all ten example identities. Navigation and Adjective order are the display names for the archived tasks navigate and hyperbaton; these aliases retain the same item sets.

## A.4 Collection records and analysis scope

The complete crossed table contains 499,482 observations across 16 arms. The main set consists of 2,399 items in 30 blocks; a separate binary-answer context set has 480 items in six blocks. The common excluded item is gpqa:gpqa\_extended\_356. Collection code attributes this exclusion to question-stem truncation; the underlying diagnostic log was not recovered. The historical SmolLM2-360M-Instruct exclusion appears in the fitting code and archived screening table, but the screen-generating code and threshold were not recovered. The current analysis directly checks inclusion rather than assigning a behavioral interpretation to its near-chance score.

The retained direct-choice comparison uses the first label reassignment, in which option wording and order are fixed and identifiers change by item-specific nonidentity cyclic ofsets. Content preservation and identifier preservation are disjoint in this comparison; the remaining 23.87% preserves neither. The second reassignment excludes each item’s first ofset and is not an independent draw. Reduced-option arms jointly change distractors and correct-option placement. These ancillary arms are not used to establish the present decomposition or ranking results.

The observational files identify models, items, and correctness, with exact hashes and ordered manifests. All 1,348 identifiers join to the saved leaderboard; 1,345 have recorded commit hashes. These hashes are leaderboard metadata, not verified bindings to the response runs. The original retrieval date, response-run model revisions, and complete observational harness metadata were not recovered. Crossed collection scripts request Python 3.11, lm-eval 0.4.13, and transformers 5.17.0, with bfloat16 for Gemma models and float16 otherwise. Complete executed version records and resolved model revisions are unavailable, so requested settings are distinguished from verified runtime metadata.

The available crossed raw archive contains 479,411 retained records, all matching the canonical merged table. Nine Gemma-27B arm files, totaling 20,071 retained observations, were not recovered locally; their records are present in the complete merged table used for fitting. No crossed per-option likelihood archive or completed length-normalized outcome analysis was recovered. The implemented response reducer uses the argmax of raw summed continuation likelihoods. These records support the reported correctness analyses while leaving parts of the original collection provenance incomplete.

The archived project account identifies choice preservation as its primary crossed comparison and the scoring contrast as secondary; it states that the plan was not independently registered. An independently dated original plan was not recovered. The present fit refinements and 17-versus-18- model check are subsequent analyses of existing responses. The local source ledger preserves their lineage and replaces earlier numerical summaries with the verified endpoints used here.

## B Supporting results

Table S1: Rank comparisons for all 1,348 BBH models on the same 5,761 items. Lower ranks indicate larger scores. Between-fit comparisons re-estimate all components; within-fit comparisons remove only the mean answer ofset. The last column counts models shared by the two top-50 sets.
<table><tr><td>Comparison</td><td>Scores</td><td> $\rho$ </td><td>Median move</td><td>Max. move Top-50 overlap</td></tr><tr><td>between fit</td><td> $\mathrm { M 0 . s \mathrm { \to M 1 . s } }$ </td><td>0.9190</td><td>70</td><td>687 28</td></tr><tr><td>between fit</td><td> $\mathrm { M 0 . s \mathrm { \to M 2 . s } }$ </td><td>0.9842</td><td>31</td><td>454 40</td></tr><tr><td>within fit</td><td> $\mathrm { M 1 . q {  } M 1 . s }$ </td><td>0.9297</td><td>61</td><td>755 30</td></tr><tr><td>within fit</td><td> $\mathrm { M 2 . q {  } M 2 . s }$ </td><td>0.9920</td><td>22</td><td>239 43</td></tr></table>

Table S2: Normalized answer-family alignment on the common BBH item set. Neighborhoods contain ten items within the same task; the null uses 200 within-task permutations. Other contains all 250 Object Counting items, with their 17 numeric answer keys retained separately. Values are selected-fit summaries.
<table><tr><td>Correct-answer family</td><td>Items</td><td>M0</td><td>M1</td><td>M2</td></tr><tr><td>Letter</td><td>4074</td><td>0.3650</td><td>0.2745</td><td>0.1006</td></tr><tr><td>TRUE/FALSE</td><td>250</td><td>0.8935</td><td>0.0496</td><td>0.0352</td></tr><tr><td> $\mathrm { Y e s } / \mathrm { N o }$ </td><td>937</td><td>0.9629</td><td>0.3677</td><td>0.1077</td></tr><tr><td>Valid/Invalid</td><td>250</td><td>0.7009</td><td>0.1300</td><td>0.0980</td></tr><tr><td>Other</td><td>250</td><td>0.1373</td><td>0.0830</td><td>0.0984</td></tr></table>

Table S3: Exact identities of the ten BBH examples used in the text and figures. Correctness is calculated over the same 5,761 items per model. Accuracy is a proportion. The introductory pairs and the six diagnostic/ranking examples have separate source-record checks described in Appendix A.3.
<table><tr><td>Short name</td><td>Model identifier</td><td>Correct</td><td>Accuracy</td></tr><tr><td>OpenBuddy</td><td> $\mathrm { 0 p e n B u d d y / o p e n b u d d y - 1 1 a m a 3 - 7 0 b - v 2 1 . 2 - 3 2 k }$ </td><td>3743</td><td>0.6497</td></tr><tr><td>Smaug</td><td>abacusai/Smaug-Llama-3-70B-Instruct-32K</td><td>3736</td><td>0.6485</td></tr><tr><td>CPO</td><td>princeton-nlp/Mistral-7B-Base-SFT-CPO</td><td>2514</td><td>0.4364</td></tr><tr><td>RDPO</td><td>princeton-nlp/Mistral-7B-Base-SFT-RDP0</td><td>2531</td><td>0.4393</td></tr><tr><td>BgGPT</td><td>INSAIT-Institute/BgGPT-Gemma-2-27B-IT-v1.0</td><td>1657</td><td>0.2876</td></tr><tr><td>Hebrew-Gemma</td><td>yam-peleg/Hebrew-Gemma-11B-Instruct</td><td>2314</td><td>0.4017</td></tr><tr><td>calme</td><td>MaziyarPanahi/calme-2.4-rys-78b</td><td>4182</td><td>0.7259</td></tr><tr><td>Phi</td><td> $\mathrm { S y e d \mathrm { - } H a s a n \mathrm { - } 8 5 0 3 / P h i \mathrm { - } 3 \mathrm { - } m i n i \mathrm { - } 4 K \mathrm { - } i n s t r u c t \mathrm { - } c p o \mathrm { - } s i m p o \left( i \mathrm { - } \mathrm { ~ s i n p o ~ a ~ t ~ } \right) }$ </td><td>3253</td><td>0.5647</td></tr><tr><td>DeepSeek</td><td> $\mathtt { d e e p s e e k - a i / D e e p S e e k - R 1 - D i s t i 1 1 - L 1 a m a - 7 0 B }$ </td><td>3239</td><td>0.5622</td></tr><tr><td>QwQ</td><td> $\mathrm { p r i t h i v M L m o d s } / \mathrm { Q w Q - L C o T 1 - M e r g e d }$ </td><td>3148</td><td>0.5464</td></tr></table>

Table S4: M1 score construction for the ranking examples. All components are on the logit scale and use equal weights over 5,761 items: $s = \theta + \bar { I }$ and $\begin{array} { r } { q = s + \bar { \delta } } \end{array}$ . The model-independent mean item efect is omitted from both scores. Components and scores are rounded after calculation. Values are conditional on the selected numerically accepted fit.
<table><tr><td>Model</td><td>θ</td><td>ī</td><td>δ</td><td>S</td><td>q</td><td>Rank s</td><td>Rank q</td></tr><tr><td>Phi</td><td>-0.616</td><td>-2.708</td><td>0.838</td><td>-3.323</td><td>-2.486</td><td>814</td><td>396</td></tr><tr><td>DeepSeek</td><td>3.886</td><td>-4.806</td><td>-1.302</td><td>-0.920</td><td>-2.222</td><td>97</td><td>344</td></tr><tr><td>QwQ</td><td>0.262</td><td>-2.748</td><td>-0.076</td><td>-2.486</td><td>-2.561</td><td>441</td><td>438</td></tr></table>

Table S5: Paired held-out log loss for one fixed 100-model BBH cohort and ten overlapping response masks. Each mask withholds 57,400 responses and retains 518,700 for estimation. Both specifications use the historical 107 task–answer groups. Gain is the task–answer-only loss minus the loss after adding interaction; it is not model-population uncertainty.
<table><tr><td>Mask</td><td>Task-answer only</td><td>With interaction</td><td>Gain</td></tr><tr><td>1</td><td>0.354357</td><td>0.330746</td><td>0.023611</td></tr><tr><td>2</td><td>0.350713</td><td>0.329354</td><td>0.021359</td></tr><tr><td>3</td><td>0.352481</td><td>0.332135</td><td>0.020346</td></tr><tr><td>4</td><td>0.352671</td><td>0.330756</td><td>0.021915</td></tr><tr><td>5</td><td>0.354487</td><td>0.331522</td><td>0.022965</td></tr><tr><td>6</td><td>0.348527</td><td>0.326901</td><td>0.021626</td></tr><tr><td>7</td><td>0.355191</td><td>0.333507</td><td>0.021684</td></tr><tr><td>8</td><td>0.358360</td><td>0.335874</td><td>0.022486</td></tr><tr><td>9</td><td>0.354385</td><td>0.334492</td><td>0.019893</td></tr><tr><td>10</td><td>0.356642</td><td>0.334561</td><td>0.022081</td></tr><tr><td>Mean</td><td>0.353781</td><td>0.331985</td><td>0.021797</td></tr></table>

Table S6: Selected crossed-evaluation alignment contrasts for 17 and 18 models. Primary matched contrasts use 27 blocks and 2,159 items with at least three options; the supplementary full-set contrast uses 30 blocks and 2,399 items. Identifier minus text denotes label scoring minus text scoring with labels (L0–LT0); printed minus unprinted denotes text scoring with labels minus text scoring without labels (LT0–U0). Intervals are conditional 95% paired block-bootstrap intervals from 10,000 draws, with fitted maps held fixed.
<table><tr><td>Models</td><td>Blocks</td><td>Contrast</td><td>Estimate</td><td>95% interval</td></tr><tr><td>17</td><td>27</td><td>Identifier minus text</td><td>0.0630</td><td>[0.0365, 0.0945]</td></tr><tr><td>17</td><td>27</td><td>Printed minus unprinted</td><td>-0.0192</td><td>[-0.0419, 0.0011]</td></tr><tr><td>17</td><td>30</td><td>Printed minus unprinted</td><td>-0.0288</td><td>[-0.0521, -0.0073]</td></tr><tr><td>18</td><td>27</td><td>Identifier minus text</td><td>0.0908</td><td>[0.0534, 0.1338]</td></tr><tr><td>18</td><td>27</td><td>Printed minus unprinted</td><td>-0.0227</td><td>[-0.0446, -0.0036]</td></tr><tr><td>18</td><td>30</td><td>Printed minus unprinted</td><td>-0.0273</td><td>[-0.0472, -0.0087]</td></tr></table>

Table S7: Between-fit agreement of label-scoring ranks before and after fitting identifier efects (L0.s) with text-scoring ranks without labels (U0.s). Ranks are computed over 2,399 items. Shared-17 ranks from the 18-model fits are recomputed among the common models. Mean distance is the mean absolute diference in rank; U0 is an alternative evaluation reference.
<table><tr><td>Fit / ranked set</td><td>before  $\rho$ </td><td>after  $\rho$ </td><td>Distance before</td><td>Distance after</td><td>Closer</td><td>Farther</td></tr><tr><td>17 / 17</td><td></td><td>0.9804 0.8824</td><td>0.588</td><td>1.765</td><td>1</td><td>9</td></tr><tr><td>18 / 17</td><td>0.9804</td><td>0.8824</td><td>0.588</td><td>1.765</td><td>1</td><td>9</td></tr><tr><td>18 / 18</td><td>0.9835</td><td>0.9009</td><td>0.556</td><td>1.667</td><td>1</td><td>9</td></tr></table>

Table S8: Numerical acceptance of retained fits. The acceptance threshold is a maximum absolute gradient of $1 0 ^ { - 4 }$ , alongside finite parameters, independently recomputed objectives and gradients, and input identity checks. The BBH rows are the three selected display fits; crossed counts include both SVD and reciprocal starts. Acceptance establishes first-order stationarity under these checks.
<table><tr><td>Analysis</td><td>Fits</td><td>Accepted</td><td>Largest absolute gradient</td></tr><tr><td>BBH M0</td><td>1</td><td>1</td><td>6.83e-06</td></tr><tr><td>BBH M1</td><td>1</td><td>1</td><td>9.83e-06</td></tr><tr><td>BBH M2</td><td>1</td><td>1</td><td>9.19e-06</td></tr><tr><td>Held-out BBH</td><td>20</td><td>20</td><td>9.88e-06</td></tr><tr><td>Crossed evaluation</td><td>16</td><td>16</td><td>9.93e-06</td></tr></table>

Table S9: Starting-solution sensitivity within each anchor-plus-four-perturbation set. Each specification has five numerically accepted candidates. Minimum rank correlation and maximum movement cover both s and q across all pairs; alignment covers all five answer families. These descriptive ranges are not confidence intervals. Component values can vary even when ranks and alignment remain close.
<table><tr><td></td><td></td><td></td><td>Specification Candidates Minimum rank ρ Maximum move</td><td>Maximum family change</td></tr><tr><td>M1</td><td>5</td><td>0.9999755</td><td>13</td><td>0.009063</td></tr><tr><td>M2</td><td>5</td><td>0.9999997</td><td>4</td><td>0.000998</td></tr></table>

Table S10: Component and rank ranges for the three ranking examples across each specification’s five saved local candidates. Mean interaction, mean ofset and s use logit units. Ranges include the preceding anchor and four perturbations; they describe starting-solution sensitivity, not sampling uncertainty.
<table><tr><td>Specification</td><td>Model</td><td colspan="2">ī</td><td colspan="2"></td><td>Rank s</td></tr><tr><td>M1</td><td>Phi</td><td>[-2.708, -2.693]</td><td>[0.836, 0.838]</td><td>[-3.323, -3.312]</td><td></td><td>[814, 816]</td></tr><tr><td>M1</td><td>DeepSeek</td><td>[-4.806, -4.795]</td><td>[-1.302, -1.286]</td><td>[-0.926, -0.920]</td><td></td><td>[97, 97]</td></tr><tr><td>M1</td><td>QwQ</td><td>[-2.748, -2.735]</td><td>[-0.076, -0.074]</td><td>[-2.486, -2.478]</td><td></td><td>[441, 444]</td></tr><tr><td>M2</td><td>Phi</td><td>[-2.474, -2.473]</td><td>[0.176, 0.176]</td><td>[-2.219, -2.218]</td><td></td><td>[615, 615]</td></tr><tr><td>M2</td><td>DeepSeek</td><td>[-2.682, -2.682]</td><td>[-0.710, -0.710]</td><td>[-1.213, -1.213]</td><td></td><td>[229, 229]</td></tr><tr><td>M2</td><td>QwQ</td><td>[-2.548, -2.547]</td><td>[-0.178, -0.177]</td><td>[-1.870, -1.869]</td><td></td><td>[434, 434]</td></tr></table>

Table S11: All 24 archived BBH task profiles for Phi (P) and DeepSeek (D) in the selected M1 fit. Accuracy is observed correctness. Ofsets (δ) and relative interactions (R) are task means in logit units; R subtracts the model-specific mean interaction over all items. Task-relative interaction therefore describes the remaining fitted profile after the explicit answer component is represented. Display names Navigation and Adjective order correspond to the archived identifiers navigate and hyperbaton.
<table><tr><td>Task</td><td>Acc. P</td><td>Acc. D</td><td> $\delta _ { P }$ </td><td> $\delta _ { D }$ </td><td> $R _ { P }$ </td><td> $R _ { D }$ </td></tr><tr><td>boolean expressions</td><td>0.804</td><td>0.868</td><td>-0.019</td><td>-1.519</td><td>0.208</td><td>0.294</td></tr><tr><td>causal judgement</td><td>0.615</td><td>0.519</td><td>-0.115</td><td>-1.933</td><td>0.484</td><td>0.094</td></tr><tr><td>date understanding</td><td>0.560</td><td>0.656</td><td>1.219</td><td>-1.341</td><td>0.174</td><td>0.238</td></tr><tr><td>disambiguation qa</td><td>0.716</td><td>0.396</td><td>1.186</td><td>-1.429</td><td>0.207</td><td>-0.177</td></tr><tr><td>formal fallacies</td><td>0.624</td><td>0.652</td><td>0.468</td><td>-1.360</td><td>0.197</td><td>-0.121</td></tr><tr><td>geometric shapes</td><td>0.216</td><td>0.556</td><td>0.240</td><td>-0.803</td><td>0.542</td><td>0.306</td></tr><tr><td>Adjective order (Hyperbaton)</td><td>0.920</td><td>0.560</td><td>1.537</td><td>-2.127</td><td>-1.931</td><td>-2.288</td></tr><tr><td>Logical deduction (5)</td><td>0.552</td><td>0.504</td><td>1.254</td><td>-1.202</td><td>0.116</td><td>0.056</td></tr><tr><td>Logical deduction (7)</td><td>0.512</td><td>0.496</td><td>1.174</td><td>-0.951</td><td>0.323</td><td>0.318</td></tr><tr><td>Logical deduction (3)</td><td>0.828</td><td>0.548</td><td>1.139</td><td>-1.306</td><td>-0.279</td><td>0.486</td></tr><tr><td>movie recommendation</td><td>0.624</td><td>0.728</td><td>1.168</td><td>-0.947</td><td>0.617</td><td>0.483</td></tr><tr><td>Navigation</td><td>0.636</td><td>0.420</td><td>0.007</td><td>-2.531</td><td>0.785</td><td>0.219</td></tr><tr><td>object counting</td><td>0.392</td><td>0.612</td><td>0.741</td><td>1.389</td><td>0.787</td><td>0.151</td></tr><tr><td>penguins in a table</td><td>0.582</td><td>0.596</td><td>1.238</td><td>-1.116</td><td>-0.093</td><td>0.211</td></tr><tr><td>reasoning about colored objects</td><td>0.488</td><td>0.700</td><td>0.383</td><td>-1.138</td><td>0.087</td><td>1.169</td></tr><tr><td>ruin names</td><td>0.652</td><td>0.864</td><td>1.246</td><td>-1.039</td><td>-0.082</td><td>0.936</td></tr><tr><td>salient translation error detection</td><td>0.568</td><td>0.616</td><td>1.214</td><td>-1.325</td><td>-0.029</td><td>0.507</td></tr><tr><td>snarks</td><td>0.787</td><td>0.657</td><td>1.541</td><td>-2.159</td><td>-0.850</td><td>-0.798</td></tr><tr><td>sports understanding</td><td>0.592</td><td>0.484</td><td>-0.042</td><td>-2.289</td><td>0.921</td><td>0.294</td></tr><tr><td>temporal sequences</td><td>0.680</td><td>0.888</td><td>1.190</td><td>-0.940</td><td>-1.851</td><td>0.144</td></tr><tr><td>Tracking objects (5)</td><td>0.252</td><td>0.172</td><td>1.234</td><td>-1.196</td><td>-0.181</td><td>-0.754</td></tr><tr><td>Tracking objects (7)</td><td>0.228</td><td>0.148</td><td>1.166</td><td>-0.879</td><td>-0.043</td><td>-0.392</td></tr><tr><td>Tracking objects (3)</td><td>0.320</td><td>0.332</td><td>1.145</td><td>-1.324</td><td>-0.340</td><td>-1.118</td></tr><tr><td>web of lies</td><td>0.488</td><td>0.552</td><td>-0.077</td><td>-2.119</td><td>0.070</td><td>-0.378</td></tr></table>

All 24 task profiles in the selected M1 fit Items per task in parentheses; all 5,761 items retained  
![](images/adad1058f9ffde2369afe6c15916ff7d52bf9d081d64b2a3a7cc45806e1fee0f.jpg)  
Figure S1: Complete task profiles for the verified Phi/DeepSeek pair. Observed correctness, mean explicit answer ofset and relative interaction are shown for all 24 tasks in the selected M1 fit. The row labels give item counts; together the rows retain all 5,761 benchmark items. Phi denotes Phi-3-mini-4K-instruct cpo-simpo and DeepSeek denotes DeepSeek-R1-Distill-Llama-70B. Each fitted component is averaged with equal weights over the task’s items. Relative interaction subtracts each model’s mean interaction over the full item set, so it describes a task-specific departure from that model’s overall interaction level. Components use logits; observed accuracy is a percentage. The row labels Navigate and Hyperbaton correspond to Navigation and Adjective order in the main text. These selected-fit profiles carry no sampling intervals. Full model identifiers and task-level summaries are reported in Tables S3 and S11.