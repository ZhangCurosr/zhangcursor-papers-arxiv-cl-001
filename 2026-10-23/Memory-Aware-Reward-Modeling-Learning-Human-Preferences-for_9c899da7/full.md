# Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use

Yijiang River Dong<sup>1,∗</sup> Zexue He<sup>2,∗</sup> Cheng Yu<sup>3,∗</sup> Tiancheng Hu<sup>1</sup> Michael J Ryan<sup>2</sup> Jan Batzner<sup>3</sup> Zheng Hui<sup>1</sup> Shaolong Wu<sup>4</sup> Orestis Papakyriakopoulos<sup>3</sup> Alex Pentland<sup>2</sup> Nigel Collier<sup>1,†</sup> Jiaxin Pei<sup>5,†</sup>

<sup>1</sup>University of Cambridge <sup>2</sup>Stanford University <sup>3</sup>Technical University of Munich <sup>4</sup>Harvard University <sup>5</sup>University of Texas at Austin

<sup>∗</sup>Equal contribution. <sup>†</sup>Equal advising.

## Abstract

Memory enables language model assistants to maintain continuity across sessions and generate personalized responses, but knowing more about a user does not necessarily produce responses they prefer. Existing evaluations assess memory capabilities without directly establishing whether memory use benefits the user. To study this gap, we collect approximately 1,000 pairwise judgments from 87 participants using their own ChatGPT queries and personal memories. We find that memory-conditioned responses are not consistently preferred over responses generated without memory and in some cases, users strongly prefer the latter. Automatic evaluators also show limited agreement with users and favor memory-conditioned responses substantially more often than users do. Motivated by these gaps, we introduce a memory-aware reward model that learns from user judgments, achieving 72.8% preference prediction accuracy and outperforming the existing baselines. Our findings motivate evaluation and optimization objectives that align memory use with the preferences of the people whose information is being used.

## 1 Introduction

Memory has emerged as a key component of conversational language models, enabling them to retain and reuse information beyond their context length and across multiple conversation sessions. Existing work has explored mechanisms for storing past experiences, managing information across context windows, and retrieving knowledge to support subsequent reasoning and action (Zhong et al., 2023; Packer et al., 2023). These capabilities support continuity across interactions and tasks. In persistent assistants, memory allows models to employ information about the user including their persona, their expressed preferences, and past requests to generate more personalized and contextually appropriate responses.

Much of the progress in long-term memory has been measured by and optimized for benchmarks like LongMemEval (Wu et al., 2025), Locomo (Maharana et al., 2024), and PersonaMem (Jiang et al., 2025a) that evaluate whether models can compress and recover information from past interactions to the current session. However, accurate recall and successful preference inference do not ensure appropriate memory usage. Consider a user who previously requested explanations tailored to their profession and now asks for a general overview of a topic. Reusing that preference may appear relevant and personalized, yet narrow the answer in a way the current request does not warrant. Recent studies document such failures through overpersonalization and inappropriate memory use (Hu et al., 2026; Feng et al., 2026a; Pulipaka et al., 2026). These errors cannot be characterized solely as failures of memory accuracy or retrieval. They mainly concern how available information should be used for LLM response.

To understand whether memory-based personalization produces responses that users prefer, we collect approximately 1,000 response-pair judgments from 87 participants. Participants construct personal memory profiles from their own ChatGPT interactions and recent experiences, then evaluate paired responses to queries drawn from their interaction histories. Each pair uses the same query and generator, with one response receiving retrieved memories and the other generated without them. Crucially, participants judge responses grounded in their memories, allowing us to examine the value of personalization from user’s per-

![](images/99e7c2f99d0c97bc534c1237c4c3dd7cd92c73f851e56de82486a81e5bf396b7.jpg)  
Figure 1: Overview of memory-aware preference learning. Given a query and retrieved user memories, the same generator produces paired responses with $( y ^ { m } )$ and without $( y ^ { 0 } )$ memory conditioning. A rubric evaluator assesses memory-related and general response characteristics, yielding pairwise features x. Feature weights are learned from annotators’ pairwise preferences, allowing the model to predict which response is preferred. Importantly, annotators evaluate responses generated using their own memories. The example and feature contributions are illustrative; Section 4 describes the full model.

## spective.

Our findings reveal a gap between personalizing a response using the user’s memory and producing a response that user prefers. Memoryconditioned responses do not consistently outperform responses generated without memory, and personalization sometimes backfires as users strongly prefer the response generated without their personal context. These findings challenge the assumption that incorporating relevant personal information necessarily improves the user experience and motivate new evaluation and optimization objective for memory-augmented language models.

We next examine whether automatic evaluators capture these preferences. When given access to the user’s memories alongside the query and candidate responses, a direct LLM judge and pretrained reward models show limited agreement with users, with prediction accuracy close to the majority baseline. They also select memory-conditioned responses substantially more often than users do. This discrepancy identifies a second gap: automatic evaluation of memory-conditioned responses do not reliably reflect the user’s preferences.

To examine these gaps, we explore an initial interpretable framework for analyzing how memory use relates to user preferences, In conventional reward modeling, a model learns $R _ { \theta } ( q , y )$ from human comparisons of responses to a user query q (Ouyang et al., 2022). We introduce Memory-Aware Reward Model that make user memory M explicit in $R _ { \theta } ( q , y , M )$ , learning preferences over both response quality and the appropriateness of memory use. Crucially, our supervision comes from real users judging responses grounded in their own memories, providing direct evidence of when and how they want remembered information to influence an answer. Applying rubric-based reward modeling to this problem, we achieve 72.8% accuracy, compared with 59.6% for a direct zero-shot LLM judge and 58.8% mean accu-racy for pretrained rewards.

Our contributions are threefold (1) a dataset of response preferences grounded in participants’ own queries and memories, enabling direct study of how users experience memory-based personalization. (2) identification of two evaluation gaps: personalization does not consistently yield user preferred responses, and the existing automatic evaluators do not capture users’ choices. (3) a memory-aware preference model that provides a interpretable proxy for human preference of memory usage that outperform the existing baselines. Together, these contributions establish human preference as a necessary point of reference for assessing the value of memory in personalized assistants.

## 2 Related Work

Long-term LLM memory. Long-term memory systems enable language model assistants to accumulate and reuse information beyond a single context window. Existing approaches span external retrieval (Zhong et al., 2023; Packer et al., 2023; Chhikara et al., 2025), graph-based memory stores (Xu et al., 2025; Jiang et al., 2026), learned policies for memory management (Wang et al., 2025; Yan et al., 2026), and parametric memory (Wang et al., 2024; Lei et al., 2026). A growing set of benchmarks evaluates these capabilities through recall and multi-session reasoning (Maharana et al., 2024; Wu et al., 2025; Tan et al., 2025), dynamic user profiling and personalized responses (Jiang et al., 2025a,b), and tracking evolving user or world states (Liu et al., 2026a; Li et al., 2026b; Fan et al., 2026). These evaluations primarily measure whether information can be retained, recovered, and updated, leaving less clear how an assistant should behave once relevant memories are available in context. Recent work moves beyond recall to examine whether past chat history supports downstream actions (He et al., 2026; Shen et al., 2026; Feng et al., 2026b). Our work extends this focus by analyzing user preference and learning rubric-based proxies for memory usage grounded in real human preferences.

Personalization and Over-personalization. LLM personalization has traditionally represented users through profiles, personas, demographic or preference attributes, and collections of prior interactions, which are then used to condition generation or predict individual preferences (Salemi et al., 2024; Kirk et al., 2024; Dong et al., 2024; Ryan et al., 2025a). However, incorporating personal information does not always improve a response: recent work identifies over-personalization as a failure of using user information when it can feel unnecessary, distracing, sycophantic, or even privacy-intrusive (Hu et al., 2026; Feng et al., 2026a). OP-Bench evaluates irrelevant personalization, repetition, and sycophancy, while PersistBench examines crossdomain memory leakage and memory-induced sycophancy (Hu et al., 2026; Pulipaka et al., 2026). We instead treat appropriate memory use as a preference learning problem. Rather than assuming a fixed notion of when personalization is desirable, we collect response-level preferences from users over controlled variations of memory-conditioned responses and use these judgments to recover signals for how user memory should shape a response.

RLHF and post-training. Post-training aligns language models with human preferences through learned reward signals or direct preference optimization (Ouyang et al., 2022; Bai et al., 2022; Rafailov et al., 2023). Standard formulations assess candidate responses to a single prompt, whereas recent work extends preference modeling to multiturn conversations and agent trajectories (Shani et al., 2024; Shi et al., 2024), or conditions judgments on user-specific summaries and learned user representations (Nam et al., 2026; Shenfeld et al., 2025; Li et al., 2024). Beyond scaler reward model, rubric-based reward model raises for its interpretability, sample-efficiency, performance (Gunjal et al., 2026; Liu et al., 2025b, 2026b; Ryan et al., 2025b). We extend preference-based alignment to cross-session memory, learning a rubricbased reward model that conditions on user memory and learns whether and how it should influnce the current response.

## 3 Data Collection

We recruit participants from Prolific for a roughly 30-minute annotation task, paid at 14 USD per hour, and collect approximately 1,000 responsepair judgments (1,047 in total: 972 preference judgments and 75 test–retest repeats). We exclude the repeats from training and evaluation and filter the annotations for quality, leaving 821 judgments from 87 participants, of which 673 are strict preferences and 148 are ties. Each judgment records an annotator’s preference between two responses to the same query in the context of their personal memories. Evaluation separates participants across training, validation, and test within each split (Section 5).

## 3.1 Personal Context Collection

Participants accessed the study through Prolific and constructed an editable memory profile by running an extraction prompt in their own ChatGPT account. The prompt draws on previous ChatGPT sessions and existing memories to (1) produce user profile such as age, occupation, location, etc. (2) infer user preferences around 10 topics including entertainment, lifestyle, communication, etc. (3) extract 100 their recent queries, excluding those dependent on unavailable attachments or prior context. Participants then entered specific events from the preceding month through typing or editable voice transcription, with a target of 30 events.

Note that we carefully avoid uploading any personal data to our servers: full conversation histories and attachments were not uploaded and we explicitly instruct the model and participants to avoid including sensitive information. Appendix E.1 reproduces the complete prompt, and Figures 2–3 show the collection interfaces. PII removal and de-identification procedures are described in Appendix A.4.

## 3.2 Response Pair Construction

For each participant, we construct a candidate pool of 100 recent queries. For each query q, we retrieve a context $M _ { q }$ consisting of nine memory entries using text-embedding-3-small and randomly select a response generator from GPT-5.2, Gemini-3.1-Pro, and Claude-Opus-5. The selected model generates two responses in separate calls: one conditioned on the retrieved context $M _ { q }$ and one without memory. The two responses thus share the same query and generator but differ in access to personal context.

For memory-conditioned responses, we vary the instructions governing memory use across three prompting regimes: conservative, ordinary, and aggressive. The ordinary prompt instructs the model to incorporate memories where relevant and appropriate. The conservative prompt imposes a stricter criterion, instructing the model to use memories only when they would meaningfully improve the answer to the current query. The aggressive prompt encourages proactive incorporation of personal context whenever possible. Appendix E.2 provides the complete prompt templates.

We also screen the 100 candidate pairs to select 15 that have substantial, memory-grounded differences in responses for human annotation. An LLM evaluator receives each query, its retrieved memories, response without memory, response with memory (using ordinary prompt) and assesses the magnitude of their substantive differences, the extent to which memory supports those differences, and the presence of unsupported changes. We rank pairs by prioritizing substantial, memory-grounded differences and retain the 15 highest-ranked pairs.

## 3.3 Response Preference Annotation

Participants viewed the query and Response A and Response B side by side. They then select the answer they would prefer to receive on a five-point scale: A much better, A slightly better, about the same, B slightly better, or B much better. Optional notes allowed them to explain their choice. Instructions emphasized overall preference without prescribing greater personalization as desirable (Appendix A.3). Appendix B presents two complete examples with contrasting preferences.

Finally, we exclude incomplete response pairs, recorded attention-check failures, or other abnormal annotator behaviors, removing 151 of the 972 primary judgments and retaining 821 from 87 participants.

## 3.4 Annotator Consistency

Ten participants returned three days later to repeat judgments for identical queries and stored response pairs, with prior choices hidden. We measure directional agreement: whether the same response is preferred in both sessions, regardless of changes in preference strength. Overall agreement was 68.0% across 75 comparisons. Initially much preferences were more consistent than slightly preferences (73.9% versus 58.6%; Table 1). Due to the low consistency of the slightly better subset, we only use the much better subset that contains more consistent and clear preference signal for testing models in 6.

<table><tr><td>Original strength Same side on repeat</td><td></td><td>Rate</td></tr><tr><td>Slightly</td><td>17/29</td><td>58.6%</td></tr><tr><td>Much</td><td>34/46</td><td>73.9%</td></tr><tr><td>Overall</td><td>51/75</td><td>68.0%</td></tr></table>

Table 1: Test–retest agreement among 10 participants after three days. “Same side” counts comparisons in which the same response is preferred in both sessions; groups are defined by the original preference strength.

## 4 Method

## 4.1 Preliminaries

Reward models. Reward modeling turns human judgments into a scoring function for selecting or optimizing model outputs. Given query q and response y, a scalar reward model $R _ { \theta } ( q , y )$ represents response quality through a learned utility. Pairwise preferences are commonly modeled as

$$
P _ { \theta } ( y _ { a } \succ y _ { b } \mid q ) = \sigma ( R _ { \theta } ( q , y _ { a } ) - R _ { \theta } ( q , y _ { b } ) )\tag{1}
$$

where σ is the sigmoid function. Training increases the reward of preferred responses relative to rejected ones (Bradley and Terry, 1952; Schulman et al., 2017).

Rubric-based reward models. Rubrics express response quality through explicit criteria, such as correctness, relevance, or respect for user constraints. An evaluator assesses these criteria, and an aggregation rule converts its judgments into a reward or preference. This separates specifying desirable behavior from assessing whether a response exhibits it (Gunjal et al., 2026; Liu et al., 2025b). A simple scalar formulation is $R _ { { \bf w } } ( q , y ) =$ $\mathbf { w } ^ { \top } \phi ( q , y )$ , where $\phi$ contains feature scores that is typically assigned by an external evaluator (e.g., LLM Judge) and w is the weight vector of features.

## 4.2 Memory-Aware Preference Modeling

Each example contains a query $q _ { i } ,$ , retrieved memories $M _ { i } ,$ a memory-conditioned response $y _ { i } ^ { m }$ , a response generated without memory $y _ { i } ^ { 0 } .$ , and the annotator’s preference. Our objective is to predict which response the annotator prefers. We represent each response comparison using interpretable features:

$$
\mathbf { x } _ { i } = \Phi ( q _ { i } , M _ { i } , y _ { i } ^ { m } , y _ { i } ^ { 0 } ) \in \mathbb { R } ^ { 1 2 } .\tag{2}
$$

Feature initialization. Preference annotations are used to induce candidate evaluation criteria. Each induction pass receives ten preference-labeled quadruplets from training set: the query, the retrieved memory set, both candidate responses, and which response the annotator chose. Then LLM then generates ten rubric dimensions that explain the difference between the two responses, such as response quality, formatting, or memory usage, etc. We run five independent passes, yielding roughly fifty raw proposals, many of which describe the same property in different words. A separate consolidation pass merges near-duplicates into a canonical rubric and labels each dimension memory-dependent if scoring it requires the memory contents, and memory-agnostic otherwise. Fixing that label at induction time means the decomposition in §6 partitions the features by definition rather than by our later judgement.

For example, in one experiment, the procedure above gives us 12 features, including three memoryrelated features (memory relevance, assumption proportionality, and unwarranted personal assumptions) and nine response-quality features describing response quality, framing, and form. Table 6 defines all twelve features. Note that rubric induction and consolidation are performed independently within each outer cross-validation fold, using only that fold’s training participants and their preference labels. The resulting rubric is held fixed within the fold for feature extraction and regression, and the learned predictor is evaluated on the held-out participants.

Feature regression. Following Ryan et al. (2025b); Li et al. (2026a), we learn weights that aggregate the features into the annotator’s pairwise preference. Each entry of $\mathbf { x } _ { i }$ compares $y _ { i } ^ { m }$ with $y _ { i } ^ { 0 }$ on one feature (e.g., heading count, or a judge’s memory-relevance score), signed so that larger values favor $y _ { i } ^ { m }$ . Let $z _ { i } = 1$ if the annotator prefers $y _ { i } ^ { m }$ and $z _ { i } = 0$ otherwise, and let $\widetilde { \mathbf { x } } _ { i }$ be $\mathbf { x } _ { i }$ standardized on the training set. We fit an L2-regularized logistic model:

$$
\begin{array} { l } { { \displaystyle p _ { i } = \sigma ( b + { \bf w } ^ { \top } \widetilde { \mathbf { x } } _ { i } ) , \qquad ( 3 ) } } \\ { { \displaystyle { \mathcal { L } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl [ z _ { i } \log p _ { i } + ( 1 - z _ { i } ) \log ( 1 - p _ { i } ) \bigr ] + \frac { \lambda } { 2 } \| { \bf w } \| _ { 2 } ^ { 2 } , } } } \end{array}
$$

where $\sigma$ is the sigmoid, so $p _ { i }$ is the probability that the annotator prefers $y _ { i } ^ { m }$ , and w holds one weight per feature. Only w and b are learned and response generators and rubric evaluators remain fixed.

## 5 Experimental Setup

Data. As explained in Section 1, we only maintain the clear-preference pairs that can provide consistent preference signal for our rubric learning, resulting in 291 pairs.

Evaluation. We run five-fold cross-validation split by participant, so no test participant appears in training, and repeat it with ten random shuffles; each comparison’s predictions are averaged over the ten repeats before computing accuracy. The L2 strength $C \in \{ 0 . 0 0 3 , 0 . 0 1 , 0 . 0 3 , 0 . 1 , 0 . 3 \}$ is chosen separately in each fold, using only that fold’s training participants (four-fold cross-validation, by accuracy). Standardization is also fitted on the training participants only.

Feature ablations We evaluate our memoryaware reward model (Ours) and two feature-block ablations: Memory only retains the three memoryrelated features, and Response only retains the other features spanning length, format, and quality. Feature-block ablations use fixed scores and do not change the evaluator’s input. Table 6 specifies feature membership. We also report selective prediction accuracy at 25%, 50%, 75%, and 100% coverage based on the score difference between the two responses.

<table><tr><td colspan="3">Preference (%)</td></tr><tr><td>Condition</td><td>Memory No memory</td><td>Tie</td></tr><tr><td colspan="3">Prompting regime</td></tr><tr><td>Conservative</td><td>43.3</td><td>40.616.1</td></tr><tr><td>Ordinary</td><td>38.7</td><td>41.919.4</td></tr><tr><td>Aggressive</td><td>34.6</td><td>49.3 16.2</td></tr><tr><td colspan="3">Response generator (ordinary prompting)</td></tr><tr><td>GPT-5.2</td><td>44.1</td><td>34.921.1</td></tr><tr><td>Gemini-3.1-Pro</td><td>30.7</td><td>59.5 9.8</td></tr><tr><td>Claude-Opus-5</td><td>48.1</td><td>36.515.4</td></tr></table>

Table 2: Annotator preferences by prompting regime and response generator. Percentages include ties. Generator comparisons are restricted to ordinary prompting to hold the personalization instructions constant.

Baselines We compare our method with the following baselines: (1) zero-shot LLM as a judge (2) LLM as judge with extracted rubrics: we provide the LLM with the rubric criteria extracted in Section 4.2 and ask the LLM to make decisions based of them. For both varients of LLM judges, we evaluate a single presentation order and averaging scores from both A/B orders after aligning response identities. (3) pretrained Skywork-Reward-V2 models (Liu et al., 2025a), scoring each response independently with the query and memories in context and selecting the higher-reward answer. Appendix D documents their detailed prompts.

## 6 Results and Analysis

## 6.1 User Preferences for Memory Use

We first examine whether access to personal memories produces responses that users prefer. Table 2 summarizes memory versus no-memory comparisons across prompting regimes and response generators.

Memory access does not ensure a preferred response. Under ordinary prompting, annotators prefer the memory-conditioned response in 38.7% of comparisons and the response generated without memory in 41.9%, with the remaining comparisons judged about the same. Thus, supplying personal context does not yield a clear advantage in this setting. These results directly illustrate the distinction motivating our study: knowing more about a user does not necessarily produce an answer they prefer.

More aggressive personalization receives less favorable preferences. Across conservative, ordinary, and aggressive prompting (Appendix E.2), preference for memory-conditioned responses decreases from 43.3% to 38.7% and 34.6%, respectively. Under aggressive prompting, users prefer the no-memory response in 49.3% of comparisons. Encouraging more personalization therefore does not necessarily improve the user experience. Even conservative prompting produces only a small descriptive advantage over responses without memory, suggesting that appropriate memory use requires more than a general instruction to personalize selectively. This pattern is consistent with concerns about over-personalization raised in recent work (Hu et al., 2026; Feng et al., 2026a).

The direction of preference varies across generators. Under ordinary prompting, memoryconditioned responses are preferred more often than no-memory responses for GPT and Claude, whereas the pattern reverses for Gemini (Table 2). The aggregate result therefore conceals differences in the direction of preference. Rather than showing that memory is uniformly unhelpful, these findings suggest that its value depends on how the generator incorporates personal context into its answer.

These results concern the screened comparisons described in Section 3.2, which prioritize substantial, memory-grounded response differences. Within this setting, they establish our first evaluation gap: memory-based personalization does not consistently translate into user-preferred responses.

## 6.2 Automatic Evaluators Failed to Capture User Preferences

Off-the-shelf evaluators show limited agreement with users. Direct GPT-5.2 judging achieves 58.1% accuracy in the single-order evaluation (Table 3), with a similar result when combining both presentation orders through confidence-weighted voting (57.9%). Pretrained Skywork-Reward-V2 achieves 52.9% with the participant’s memories in context. Both perform close to, or below, the 55.0% majority baseline. Even on comparisons with strong expressed preferences, access to the query, memories, and responses is therefore insufficient for these evaluators to reliably recover the user’s choice.

<table><tr><td>Accuracy (%)</td></tr><tr><td>Predictor Avg. GPT Gemini Claude</td></tr><tr><td>Majority 55.051.8 70.6 75.6 Skywork (with memory) 52.963.6 37.5 73.3 Skywork (no memory) 58.8 55.5 66.9 42.2</td></tr><tr><td>Direct judge 58.1 53.6 58.8 66.7 Rubric judge 59.652.3 68.8 50.0</td></tr><tr><td>Ours (Response only) 67.0 60.0 71.3 77.8</td></tr><tr><td>Ours (Memory only) 62.555.5 65.4 71.1</td></tr><tr><td>Ours 72.8 66.0 75.0 82.2</td></tr></table>

Table 3: Preference prediction accuracy. Majority is calculated by always choosing without memory. Skywork is the Reward-V2 Llama-3.1-8B checkpoint, scoring each response independently and then compare the score. We include two variants of Skywork: query only (no memory) and query with the retrieved memories. For both LLM Judge and Skywork, we varied several different prompts/formats and the results listed is the best performance.

Evaluators favor memory-conditioned responses more often than users do. The direct judge under two-order voting and Skywork select memory-conditioned responses in 67.2% and 81.1% of comparisons, respectively, compared with 45.0% for annotators. Removing memories from Skywork’s input reverses this tendency but leaves accuracy close to the majority baseline. Its choices are therefore strongly affected by the availability of personal context, without closely matching user preferences in either configuration. These results highlight the gap between rewarding personalization and assessing its value to the user.

Explicit criteria offer limited improvement without learned aggregation. We next provide the LLM judge with the rubric criteria extracted following Section 4.2, emphasizing task-relevant memory use, appropriate non-use, and avoidance of unsupported personal assumptions. This rubric judge achieves 59.6% accuracy in the single-order evaluation, compared with 58.1% for direct judging on the same comparisons (Table 3). Making these considerations explicit offers a modest improvement, but leaves substantial disagreement with users. The evaluator must still determine how to weigh competing criteria, motivating our approach of learning their aggregation directly from user judgments.

## 6.3 Learning to Predict Response Preferences

Learned rubric weights improve prediction accuracy. Our model achieves 72.8% accuracy, outperforming the majority baseline (55.0%), direct

LLM judging (58.1%), and rubric-based LLM judging (59.6%; Table 3), approaching the test-retest consistency of 73.9%. This improvement is observed across all three response generators, with accuracies of 66.0%, 75.0%, and 82.2% on GPT, Gemini, and Claude generated pairs, respectively. For each of these models, our method exceed the baselines significantly, highlighting the effectiveness of learning to aggregate multiple rubric dimensions from human preferences, rather than relying on an LLM judge to implicitly weigh these criteria through prompting alone.

Both presentation and memory appropriateness drive preferences. Table 4 lists the five most influential features in a descriptive fit of our model. The two largest coefficients are presentational: the memory-conditioned response is more likely to be preferred when it has more headings (+0.646) and is relatively longer (+0.571). This is consistent with (Dubois et al., 2024; Singhal et al., 2023) which shows that human have a strong preference for longer responses and it’s hard to distangle quality from length, where in our case, we are also need to distangle length and memory usage.

Memory-related features are also important by taking three out of top five influential features. Memory relevance and assumption proportionality have coefficient +0.386 and +0.323, respectively, and unwarranted personal assumptions, ranked fifth, has a negative coefficient of −0.201. This shows that annotator prefers response when the personal context is relevant and the resulting inferences are justified, rather than unwanted assumptions of user beyond query indicates.

Combining memory features and response attributes improves prediction. Following the ablation described in 5, we divide the features into two blocks: memory-related features and responserelated features. The memory-only and responseonly models achieve 62.5% and 67.0% pooled accuracy, respectively, while combining both feature groups increases accuracy to 72.8%, outperforming the two ablations by 10.3 and 5.8 percentage points. Importantly, all variants use the same evaluator inputs and feature scores, differing only in the features retained for regression. The improvement over the response-only model therefore demonstrates that memory-related criteria provide additional predictive information beyond general response attributes. This pattern is consistent across all three response generators, with the full model achieving the highest accuracy in each case. Together, these results suggest that capturing user preferences for memory-conditioned responses requires considering both the appropriateness of memory use and the overall quality of the response.

<table><tr><td>Feature</td><td></td><td>β Mem.</td></tr><tr><td>Heading difference. Headings in the memory-conditioned response minus those in the other response.</td><td>+0.646</td><td>No</td></tr><tr><td>Relative length. Log length ratio of the memory-conditioned to the no-memory response.</td><td>+0.571 No</td><td></td></tr><tr><td>Memory relevance. Relevance of the retrieved memories to the query, scored without either response.</td><td>+0.386 Yes</td><td></td></tr><tr><td>Assumption proportionality. Whether personal assumptions are justified by the memory evidence.</td><td>+0.323 Yes</td><td></td></tr><tr><td>Unwarranted personal assumptions. Degree to which a response assumes more about the user than the query establishes.</td><td>-0.201 Yes</td><td></td></tr></table>

Table 4: The five features with the largest standardized coefficients in the descriptive fit of our model to the 291 clear-preference pairs. Positive coefficients favor the memory-conditioned response; Mem. marks memoryrelated features.

<table><tr><td rowspan="2">Predictor</td><td colspan="4">Coverage</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>Ours (Response only)</td><td>67.0</td><td>75.5</td><td>82.2</td><td>83.1</td></tr><tr><td>Ours (Memory only)</td><td>62.5</td><td>66.1</td><td>68.8</td><td>64.2</td></tr><tr><td>Ours</td><td>72.8</td><td>76.2</td><td>83.7</td><td>88.8</td></tr></table>

Table 5: Accuracy (%) after retaining the most confident comparisons within each pipeline.

Higher-confidence subsets yield higher prediction accuracy. Ranking comparisons by |pˆ−0.5| raises our model’s accuracy from 72.8% at full coverage to 76.2% at 75% coverage, 83.7% at 50% coverage, and 88.8% at 25% coverage (Table 5). Retaining the highest-confidence quarter increases accuracy by 16.0 points. The model’s confidence therefore helps identify comparisons whose annotator preferences it predicts more accurately, supporting selective use of its predictions when coverage can be traded for accuracy. This selective prediction shows the reliability of the learned aggregation model, and also can provide more accuate signal if used to evaluate the model responses or used as reward model to train LLM.

## 7 Conclusion

We investigated how users want personal memories to shape assistant responses, using judgments from 87 participants evaluating answers to their own queries with their own memories. In our screened comparisons, memory-conditioned responses were not consistently preferred, and more aggressive personalization received less favorable judgments. Existing automatic evaluators favored memoryconditioned responses more often than users did, highlighting a mismatch between rewarding personalization and capturing its value to the user.

To address this gap, we introduced a memoryaware reward model that learns to aggregate interpretable rubric features from user preferences. The model achieved 72.8% accuracy on clearpreference comparisons with held-out participants, outperforming direct LLM judging and pretrained reward baselines. Combining memory-related criteria with general response attributes improved prediction over either feature group alone, indicating that evaluating memory use requires considering both its appropriateness and the quality of the resulting answer. These findings establish user-grounded preference learning as a promising direction for evaluating memory-augmented assistants. Extending this supervision to post-training remains an important next step toward assistants that learn when and how remembered information improves the response a user wants to receive.

## Limitations

Our study has several limitations. First, our dataset captures user context through memory profiles constructed from participants’ own ChatGPT interactions and recent events they entered. While this approach grounds preference judgments in authentic personal experiences, it covers a particular form of user memory. Persistent assistants may encounter more diverse information distributed across conversations, documents, applications, and everyday activities. Extending our findings to these settings will require richer, multi-source, longitudinal collection of user context, together with preferences grounded in that context.

Second, our data and rubrics focus on single-turn conversational responses with retrieved memories supplied as context. In multi-turn conversations and agentic tasks, appropriate memory use may depend on the interaction trajectory, the actions an assistant takes, and changes in the user’s goals or circumstances. Memory acquisition, updating, and deletion may also shape users’ expectations of subsequent behavior. Our test–retest study provides an initial assessment of preference consistency, but does not capture longer-term dynamics. Extending the framework will require trajectory-level judgments and evaluation criteria that account for both evolving user preferences and the consequences of memory use across interactions.

Finally, our experiments evaluate memory-aware preference modeling in an offline setting. An important next step is to use this supervision in assistant post-training and assess whether improvements in preference prediction translate into better memory-use behavior and user experience. This would establish how the learned criteria transfer from evaluating candidate responses and predicting user preference to guiding generation.

## Ethical Considerations

Our study involves human participants and personal information derived from their interactions with an AI assistant. The study was approved by an institutional research ethics committee. Participants were recruited through Prolific for a 30- minute task, compensated at USD 14 per hour, and provided informed consent before data collection. Participants could edit their memory profiles before submission and both the AI assistant and the participant were instructed to exclude sensitive information and direct identifiers. Full conversation histories and attachments were not uploaded to the study platform. Appendix A.4 describes our deidentification protocol for data release, including consistent replacement of identifiers, generalization of identifying details, and contextual review across each participant’s records.

Our work aims to help assistants use personal memories in ways that better reflect users’ preferences. However, predicted preference should not be interpreted as consent to collect, retain, or disclose personal information. Moreover, preferences learned across participants may not reflect an individual user’s expectations. Applications of the proposed approach should preserve explicit user control over memory and respect privacy constraints independently.

## References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,

Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. ArXiv preprint arXiv:2204.05862.

Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. ArXiv preprint arXiv:2504.19413.

Yijiang River Dong, Tiancheng Hu, and Nigel Collier. 2024. Can LLM be a Personalized Judge? ArXiv preprint arXiv:2406.11657.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475.

Xinyi Fan, Miri Liu, Ruozhen Yang, Siru Ouyang, and Jiawei Han. 2026. Can Agent Memory Systems Track Evolving State? ArXiv preprint arXiv:2608.19652.

Xueyang Feng, Weinan Gan, Xu Chen, Quanyu Dai, and Yong Liu. 2026a. RPEval: How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants. ArXiv preprint arXiv:2601.16621.

Zhaoxin Feng, Jianfei Ma, and Emmanuele Chersoni. 2026b. Know It, Act on It: Investigating Memory Utilization in LLM Personalization. ArXiv preprint arXiv:2607.29433.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. 2026. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In International Conference on Learning Representations.

Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, and Alex Pentland. 2026. MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks. ArXiv preprint arXiv:2602.16313.

Yulin Hu, Zimo Long, Jiahe Guo, Xingyu Sui, Xing Fu, Weixiang Zhao, Yanyan Zhao, and Bing Qin. 2026. OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents. ArXiv preprint arXiv:2601.13722.

Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, and Dan Roth. 2025a. Know Me, Respond to Me:

Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. In Proceedings of the Second Conference on Language Modeling.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, and 1 others. 2025b. Personamem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. arXiv preprint arXiv:2512.06688.

Dongming Jiang, Yi Li, Guanpeng Li, and Bingzhe Li. 2026. MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents. ArXiv preprint arXiv:2601.03236.

Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Bean, Katerina Margatina, Juan Ciro, Rafael Mosquera, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott A. Hale. 2024. The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models. In Advances in Neural Information Processing Systems.

Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, and Soujanya Poria. 2026. δ-mem: Efficient online memory for large language models. arXiv preprint arXiv:2605.12357.

Christy Li, Sky CH-Wang, Andi Peng, and Andreea Bobu. 2026a. Human-guided harm recovery for computer use agents. Preprint, arXiv:2604.18847.

Shuyue Stella Li, Bhargavi Paranjape, Kerem Oktar, Zhongyao Ma, Gelin Zhou, Lin Guan, Na Zhang, Sem Park, Lin Chen, Diyi Yang, Yulia Tsvetkov, and Asli Celikyilmaz. 2026b. HorizonBench: Long-Horizon Personalization with Evolving Preferences. ArXiv preprint arXiv:2604.17283.

Xinyu Li, Ruiyang Zhou, Zachary C Lipton, and Liu Leqi. 2024. Personalized language modeling from personalized human feedback. arXiv preprint arXiv:2402.05133.

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. 2025a. Skywork-Reward-V2: Scaling preference data curation via human-AI synergy. arXiv preprint arXiv:2507.01352.

Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, and Tong Xu. 2026a. PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments. ArXiv preprint arXiv:2603.23231.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. 2026b. OpenRubrics:

Towards scalable synthetic rubric generation for reward modeling and LLM alignment. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17417–17437, San Diego, California, United States. Association for Computational Linguistics.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. 2025b. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Hyunji Nam, Yanming Wan, Mickel Liu, Peter Ahnn, Jianxun Lian, and Natasha Jaques. 2026. Learning to summarize user information for personalized reinforcement learning from human feedback. In International Conference on Learning Representations, volume 2026, pages 122761–122784.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training Language Models to Follow Instructions with Human Feedback. ArXiv preprint arXiv:2203.02155.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as Operating Systems. ArXiv preprint arXiv:2310.08560.

Sidharth Pulipaka, Oliver Chen, Manas Sharma, Taaha S. Bajwa, Vyas Raina, and Ivaxi Sheth. 2026. PersistBench: When Should Long-Term Memories Be Forgotten by LLMs? ArXiv preprint arXiv:2602.01146.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. ArXiv preprint arXiv:2305.18290.

Michael J. Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, and Diyi Yang. 2025a. SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Michael J. Ryan, Yanzhe Zhang, Amol Salunkhe, Yi Chu, Di Xu, and Diyi Yang. 2025b. Autometrics: Approximate human judgements with automatically generated evaluators. arXiv preprint arXiv:2512.17267.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024. LaMP: When Large Language Models Meet Personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, Avinatan Hassidim, Yossi Matias, and Rémi Munos. 2024. Multiturn Reinforcement Learning with Preference Human Feedback. In Advances in Neural Information Processing Systems, volume 37.

Yiting Shen, Kun Li, Wei Zhou, and Songlin Hu. 2026. Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents. ArXiv preprint arXiv:2601.19935.

Idan Shenfeld, Felix Faltings, Pulkit Agrawal, and Aldo Pacchiano. 2025. Language model personalization via reward factorization. arXiv preprint arXiv:2503.06358.

Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. 2024. Direct Multi-Turn Preference Optimization for Language Agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2312–2324.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2023. A long way to go: Investigating length correlations in rlhf. arXiv preprint arXiv:2310.03716.

Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. 2025. MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents. ArXiv preprint arXiv:2506.21605.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, and Julian McAuley. 2024. MEMORYLLM: Towards selfupdatable large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 50453–50466. PMLR.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. 2025. Mem-α: Learning Memory Construction via Reinforcement Learning. ArXiv preprint arXiv:2509.25911.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. In The Thirteenth International Conference on Learning Representations.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-MEM: Agentic Memory for LLM Agents. In Advances in Neural Information Processing Systems.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z Pan, and 1 others. 2026. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12805–12825.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. MemoryBank: Enhancing Large Language Models with Long-Term Memory. ArXiv preprint arXiv:2305.10250.

## A Annotation Protocol and Participant Materials

We provide the profile and event collection interfaces and response-preference instructions to make the collection procedure inspectable. The complete persona extraction and response-generation prompts are reproduced at the end of the appendix (Appendix E). The screenshots illustrate the platform using an example profile, separate from the participant records in Appendix B. They document profile and event entry; the response-comparison instructions are reproduced in Appendix A.3. Interface details varied across collection revisions.

## A.1 Profile Collection Interface

Participants copy the extraction prompt, run it in their own account, and inspect the returned profile before proceeding to event collection. The screenshot preserves the wording shown by the platform, including the account-setup instructions.

![](images/e0a7386b595ee6bc2ccc385c93e3db2f9e6474ddb9139e752445dc04be56f1cd.jpg)  
Figure 2: Persona collection interface (Step 1), showing the prompt-copy control, an example structured profile, and the transition to event collection. The displayed profile is illustrative.

## A.2 Event Collection Interface

Participants are asked to provide specific events from the preceding month. The interface supports batch recording, direct typing, and voice dictation for individual entries, and displays progress toward the 30-event target. Its examples explain the required specificity. The statement about not saving recordings is reproduced as shown to participants.

![](images/f4e5a82059cfd9a7fff63293c90d39e1d60ef21696a3d3fac76d6e3f35455b37.jpg)  
Figure 3: Recent-event collection interface (Step 2), with task instructions, examples of sufficiently specific events, collection progress, and text/voice entry controls.

## A.3 Response Preference Instructions

The archived response-comparison instructions state:

The responses differ in how they use memory. Judge purely on which response you would prefer to receive as a user.

The interface displays the query and two responses, labeled A and B, and records one of five choices:

<table><tr><td>Displayed choice</td><td>Stored label</td></tr><tr><td>A much better</td><td>a_much</td></tr><tr><td>A slightly better</td><td>a_slightly</td></tr><tr><td>About the same</td><td>tie</td></tr><tr><td>B slightly better</td><td>b_slightly</td></tr><tr><td>B much better</td><td>b_much</td></tr></table>

Participants may add an optional explanation. The displayed response labels do not disclose the generation condition.

## A.4 PII Removal and De-identification Protocol

Personal memories can identify their owners through explicit identifiers or combinations of otherwise ordinary details. Our de-identification protocol therefore covers both direct identifiers and contextual disclosure, while preserving the information needed to interpret response preferences. Its scope includes profiles, event memories, queries, candidate responses, free-text annotations, and accompanying metadata.

Collection-time minimization. The persona extraction prompt explicitly prohibits real names, email addresses, telephone numbers, exact home addresses, specific company and school names, social accounts, and identification numbers. It instructs the extractor to replace such details with abstract descriptions, such as “a tech company” or “a well-known university,” and applies the same restriction to profile tags. The query-extraction instructions also exclude sessions whose initial query contains substantial PII or unsafe sensitive information. These instructions reduce disclosure at collection; prompt compliance alone is not a guarantee of de-identification.

Text and metadata handling. The release protocol requires direct identifiers to be removed or replaced with typed placeholders, such as [PERSON\_1], [ORGANIZATION\_1], and [LOCATION\_1]. Replacements must remain consistent within a participant’s records, including across a memory, its query, and both candidate responses. Exact dates, fine-grained locations, and distinctive employment or biographical details require generalization when their combination could identify a person. The same rules cover third parties mentioned in memories. Recruitment identifiers, contact details, and operational metadata are excluded from released records; random study identifiers preserve participant grouping without exposing recruitment identifiers or linkage keys.

Review and preservation of preference meaning. Release review combines automated screening for identifier patterns and named entities with contextual human inspection of complete records. The review must consider combinations of details across a participant’s memories, rather than assessing each field in isolation. Redaction must preserve task-relevant constraints, entity relationships, and the contrast between candidate responses. If removing identifying or sensitive information changes the basis of a preference, the comparison is withheld rather than retaining its label for a materially altered example. Release documentation records the affected fields and exclusions without reproducing removed identifiers. De-identification reduces disclosure risk but does not establish that personal narratives are anonymous.

## B Illustrative Response-Preference Examples

## B.1 Preference for the response generated with memory

Query. Should I watch them in one sitting or split them across a few evenings this week?

## Retrieved memory (complete set).

1. Watch the Tom Holland Spiderman Trilogy

2. Went to the Cinema to watch the Odessy

3. The user relaxes mainly by watching TV.

4. The user frequently watches promotions.

5. Went out for dinner with my parents friends

6. went to [restaurant] for a smokehouse dinner

7. The user enjoys outdoor activities on weekends.

8. Watched a record breaking runs scored in the Hundred

9. researched Amazon trips for my upcoming travels

## Recorded user preference: B slightly better. No explanation was submitted.

## Response A (without retrieved memory).

It depends mostly on how long “them” are and what you want the experience to feel like:

• Watch in one sitting if it’s a single movie (or a short set totaling \~2–3 hours), you want momentum, and you can watch without being tired or distracted.

• Split across a few evenings if the total runtime is longer than \~3 hours, they’re dense/plot-heavy, you’re likely to be interrupted, or you want time to savor and think between installments (especially for a series or a trilogy).

If you tell me what “them” refers to (titles + roughly how many/length), I can give a specific plan for this week.

## Response B (with retrieved memory).

Split them across a few evenings this week.

Since you usually relax by watching TV, you’ll probably enjoy each film more if you treat them like a mini “event” rather than powering through when you’re tired. A one-sitting binge (all three) is a long haul and the later films can blur together.

A good plan:

• 3 evenings, one film per night (ideal): e.g., Tue/Thu/Sun.

• Or two evenings if you’re busy: films 1–2 together one

night, film 3 another night (but that’s still a hefty session).

If you’ve got an outdoorsy weekend coming up, keep the movies to weeknights and leave the weekend lighter. And if you like making it feel special (like when you went to the cinema), pick one night to do it “cinema-style” with snacks and no phone.

## B.2 Preference for the response generated without memory

Query. I have a presentation due tomorrow afternoon and need to finalize it today. When should I block time to work on it?

## Retrieved memory (complete set).

1. The user is most productive during the noon period.

2. Sat in on a work planning meeting for an upcoming project phase

3. Dealt with a stressful deadline crunch on a jobsite task

4. The user allocates about two hours for their exercise sessions.

5. The user commonly handles tasks like email, presentations, meetings, planning, and analysis.

6. Put together a data report for a construction project I’m working on

7. The user desires reminders at a moderate frequency, set for one hour early.

8. Played FIFA in the evening after work

9. Spent a Saturday gaming to decompress after a long work week

## Recorded user preference: A much better. No explanation was submitted.

## Response A (without retrieved memory).

Block time today, in two focused work blocks plus a short final pass, so you’re not rushing tomorrow.

## If your presentation is due tomorrow afternoon Today

1. Main build block (90–120 min) — schedule this at your best-focus time (often late morning or early afternoon). Goal: finish the storyline + all slides in rough-complete form. 2. Break (20–30 min)

3. Polish + visuals block (60–90 min) — later today. Goal: tighten slides, add visuals, clean formatting, simplify text.

## Tonight

4. Final run-through (20–30 min) — ideally after a long break (post-dinner).

Goal: practice out loud once, fix only the most obvious issues, then stop.

## Tomorrow (morning)

## 5. Buffer + backup (20–30 min)

Goal: quick check on another device, export to PDF, ensure links/fonts/videos work.

## If you tell me two things, I’ll give you exact times

1) What time is it where you are right now?

2) What time tomorrow afternoon is it due (e.g., 2pm,   
4:30pm)?

## Response B (with retrieved memory).

Block two focused work blocks today, with the first one scheduled for your noon window (since that’s when you’re typically most productive), and a shorter polish block later. Suggested schedule (today):

• Main build block (90–120 min): 12:00–2:00 PM (or the closest uninterrupted window around noon)

Use this to finish the core story: slide flow, key charts, and speaker notes.

• Polish + run-through (45–60 min): 4:30–5:30 PM (or early evening)

Tighten wording, check visuals, and do a timed practice.

If your calendar is messy: do two 60–75 min blocks (around noon + later), rather than one long one.

If you tell me what time “tomorrow afternoon” the presentation is and what meetings you already have today, I can propose exact time blocks.

## C Rubric Regression Details

<table><tr><td>Feature</td><td>Definition</td><td>β Mem.</td><td></td></tr><tr><td>HEAD</td><td>Heading difference. Number of headings in the memory-conditioned response minus that in the response without memory.</td><td>+0.646</td><td>No</td></tr><tr><td>LEN</td><td>Relative length. Log ratio of response lengths, oriented from the memory-conditioned to the no-memory response.</td><td>+0.571</td><td>No</td></tr><tr><td>REL</td><td>Memory relevance. Relevance of the retrieved memories to the current query, scored without viewing either response.</td><td>+0.386</td><td>Yes</td></tr><tr><td>PROP</td><td>Assumption proportionality. Whether personal assumptions are justified by the +0.323 Yes available memory evidence, compared between responses.</td><td></td><td></td></tr><tr><td>ASSUMP</td><td>Unwarranted personal assumptions. Degree to which a response assumes more —0.201 Yes about the user than the query establishes, compared between responses.</td><td></td><td></td></tr><tr><td>CLOSURE</td><td>Actionable closure. Whether a response supplies a clear next action rather than only considerations to weigh, compared between responses.</td><td>+0.191 No</td><td></td></tr><tr><td>AXIS</td><td>Advisory orientation. Relative density of advisory versus factual language, computed as (advisory — factual)/(advisory + factual) and averaged across both responses.</td><td>+0.147 No</td><td></td></tr><tr><td>COMMIT</td><td>Recommendation commitment. Commitment to a recommendation rather than +0.125 No enumeration of alternatives, compared between responses.</td><td></td><td></td></tr><tr><td>OPT</td><td>Option framing. Density of options and trade-off language, averaged across both +0.124 No responses.</td><td></td><td></td></tr><tr><td>BREADTH</td><td>Coverage breadth. Breadth of the information and alternatives covered, compared —0.083 No between responses.</td><td></td><td></td></tr><tr><td>CosT</td><td>Cost to first use. Effort required before the user can act on an answer, compared +0.036 No between responses.</td><td></td><td></td></tr><tr><td>REGISTER</td><td>Register fit. Appropriateness of the response&#x27;s language and register for the request, compared between responses.</td><td>+0.030 No</td><td></td></tr></table>

Table 6: The twelve features of our model, ordered by absolute standardized L2 coefficient in the descriptive fit to 291 clear-preference pairs (C = 0.1). Mem. identifies the memory-related feature block. Positive coefficients favor the memory-conditioned response. These coefficients describe the full-data fit, rather than an average of the nested-validation fits.

## D Zero-Shot Evaluator Baselines

## D.1 Direct LLM Judging

GPT-5.2 receives the query, retrieved memories, and both candidate responses and is asked to predict which response the user would prefer. The direct prompt supplies no rubric or explicit instructions to counter preferences for length or personalization. Calls use low reasoning effort and JSON output containing the preferred response and a rationale; a confidence variant additionally requests a score from 1 to 100. Both A/B orders were scored with confidence on the 291 clear-preference pairs shared with the regression analysis (582 calls, no failures); results are in Table 7.

We report the confidence-weighted vote of both orders after mapping predictions to the same response identities, and the subset on which both orders select the same response. The last condition changes the evaluation set and therefore uses its own majority rate. Generator-specific results refer to the model that produced the candidate responses; the judge is GPT-5.2 throughout.

<table><tr><td>Response generator</td><td>Judge variant</td><td>n</td><td>Majority (%) Accuracy (%)</td></tr><tr><td>Pooled</td><td>Voting</td><td>291</td><td>55.0 58.1</td></tr><tr><td>Pooled</td><td>Both orders agree</td><td>226</td><td>51.8 60.2</td></tr><tr><td>GPT-5.2</td><td>Voting</td><td>110</td><td>51.8 55.5</td></tr><tr><td>GPT-5.2</td><td>Both orders agree</td><td>80</td><td>57.5 57.5</td></tr><tr><td>Gemini-3.1-Pro</td><td>Voting</td><td>136</td><td>70.6 58.8</td></tr><tr><td>Gemini-3.1-Pro</td><td>Both orders agree</td><td>110</td><td>69.1 59.1</td></tr><tr><td>Claude-Opus-5</td><td>Voting</td><td>45</td><td>75.6 61.1</td></tr><tr><td>Claude-Opus-5</td><td>Both orders agree</td><td>36</td><td>80.6 69.4</td></tr></table>

Table 7: Zero-shot direct GPT-5.2 judging on the clear-preference evaluation stratum. Majority rates are computed separately for each evaluation subset.

Order sensitivity and confidence. Predictions agree across A/B orders on 77.7% of pairs overall. Under voting, the judge’s rate of choosing the memory-conditioned response exceeds the annotators’ rate by 22.2 percentage points overall (67.2% versus 45.0%). Self-reported confidence averages 76.6 for correct judgments and 73.8 for incorrect judgments. Its correlation with correctness is .180, and with selecting the memory-conditioned response it is .064. Together, these diagnostics show that stable presentation-order choices and high self-reported confidence can coexist with disagreement with annotators.

Rubric judging. The rubric judge uses a zero-shot GPT-5.2 prompt that requests a holistic preference verdict and its magnitude, explicitly discouraging preferences based solely on greater length or more personal references and favoring appropriate non-use over forced memory use. Both A/B orders were scored for all 291 pooled clear-preference comparisons (582 calls, no failures). Its reported magnitude serves as the confidence weight when the two orders are combined; verdicts of “tie” receive half credit, and the judge returned no ties on this pool.

## D.2 Pretrained Skywork Rewards

We use the published Skywork-Reward-V2 checkpoints as fixed reward functions. The reported row uses the Llama-3.1-8B checkpoint; the evaluation archive also includes the Qwen3 0.6B, 1.7B and 4B variants. For each candidate, the user turn contains the query and retrieved memories under [User Query] and [Relevant Memories], and the assistant turn contains the response. The tokenizer’s chat template formats these turns, and the sequence-classification head produces a scalar reward. Both candidates receive the same query and memory context; the higher-scoring candidate is selected. No task-specific parameter updates are applied to these baselines.

The rows in Table 3 are single zero-shot evaluations of the Llama-3.1-8B checkpoint on the 291 pooled clear-preference comparisons. With the participant’s own retrieved memories in the context it reaches 52.9% against a 55.0% majority baseline, and 63.6%, 37.5% and 73.3% on GPT-, Gemini- and Claude-generated pairs; it selects the memory-conditioned response in 81.1% of comparisons where annotators select it in 45.0%. With the memories removed from its input it reaches 58.8% (55.5%, 66.9% and 42.2% by generator) and selects the memory-conditioned response in only 19.6% of comparisons, so the preference for the memory-conditioned response is induced by the memory context rather than by the responses alone.

## E Collection and Response Generation Prompts

## E.1 Complete Persona Extraction Prompt

The listing reproduces the complete text supplied by the Copy Prompt button, including the JSON output specification. Its wording matches both the archived collection interface and the current implementation; only line wrapping has been added. The prompt requests 30 recent queries, while the interface accepts a minimum of 20. Only the first user query from each session is considered; sessions requiring unavailable attachments or prior context are skipped, and fewer queries are permitted when accessible history is insufficient. Extraction is limited to history accessible within the participant’s account. Instructions to omit identifiers and distinguish inference guide profile construction but do not independently verify the resulting content.

```markdown
You are a personal profile generation assistant. Your task is to generate a structured,
third-person user profile based only on the full conversation history with this user and
any existing memory you already have about them.
## Core Principles
1. Information source: Use only known information from conversation history and memory. Do
not ask the user questions.
2. Privacy protection: Never include PII, including but not limited to real name, email,
phone, exact home address, specific company name, specific school name, social account,
ID number. If memory contains these details, replace with abstract wording (for example:
"a tech company", "a well-known university").
3. Honest labeling: If a field cannot be inferred, write "Unknown". Do not fabricate.
4. Reasonable inference: If a field can be reasonably inferred from behavior patterns, mark
it as "[Inferred]".
5. Time expression: For all time-related fields, only use these labels:
Late night / Early morning / Morning / Noon / Afternoon / Dusk / Evening / Midnight
Never output exact clock time.
6. Flexible tags: Candidate tags are recommendations, not a closed set. If they do not fit,
create a custom concise tag in this format:
[Custom] Tag
Keep custom tags 2-6 words and never include PII.
## Fields to Cover
### 1) Basic Information
- Age range: <18 / 18-25 / 25-32 / 32-40 / 40+
- Main role (multi-select): Student / Researcher / Engineer / Designer / Manager / Other: ___
- Current country of residence
- Primary languages (ordered by proficiency)
### 2) Personality and Long-term Style
- MBTI (or Unknown)
- Self-description keywords (3-5): Organized / Flexible / Introverted / Extroverted / Calm /
Anxious / Ambitious / Easygoing / Curious / Pragmatic / Emotional / Rational /
Spontaneous / Disciplined
- Planning style: Highly planned / Roughly planned / Adaptive / Very spontaneous
- Decision style: Fast decisions / Balanced trade-off / Deep deliberation
- Daily preference: Fixed routine / Balanced / Novelty seeking
- Work style: Independent / Collaboration-heavy / Hybrid
### 3) Daily Routine
Use only: Late night / Early morning / Morning / Noon / Afternoon / Dusk / Evening / Midnight
- Typical wake-up period
- Typical sleep period
- Chronotype: Early riser / Night owl / Mixed
- Most productive period
- Weekend style: Rest-focused / Social / Outdoor / Work-study / Mixed
### 4) Food Preference
- Preferred food tags (up to 5): Chinese / Japanese / Korean / American / Italian / Mexican /
Healthy food / Fast food / Home-style / Spicy / Sweet / Coffee / Tea
- Disliked or avoided food tags: Spicy / Greasy / Sweet / Seafood / Dairy / Beef / Pork / Raw
food / None / Other:
- Spice level: No spice / Mild / Medium / Heavy
```

```markdown
- Meal priority: Taste / Health / Convenience / Price / Portion
### 5) Entertainment
- Preferred entertainment tags (up to 5): Short videos / YouTube / Movies / TV series / Music
/ Reading / Gaming / Exercise / Shopping / Podcasts / Social media
- Preferred content themes (short keywords)
Main relaxation method: Watching videos / Reading / Gaming / Walking / Chatting / Sleeping
/Eating/ Mixed
### 6) Work or Study
- Main field (free text)
Common task tags (up to 5): Writing / Coding / Reading / Meetings / Design / Analysis /
Research / Email / Presentation / Planning
Main efficiency obstacles (up to 2): Easily distracted / Fatigue / Procrastination / Stress
/ Too many meetings / Not enough time / Frequent context switching / Unclear priorities
Preferred work environments: Quiet / Light background noise / Music / Alone / With others /
Home / Office / Cafe
### 7) Social, Exercise, Lifestyle
Social style: Solitude / Small groups / Large groups / Mixed
Exercise frequency: Rarely / 1-2 per week / 3-4 per week / 5+ per week
Preferred exercise types (up to 5): Walking / Running / Strength training / Swimming / Ball
sports / Yoga / Pilates / Cycling / Hiking / Dance / Martial arts / Home workout / None
Exercise goals (up to 2): Fat loss / Muscle gain / Health maintenance / Stress relief /
Social fun / Fitness improvement / Pure enjoyment
Preferred exercise period (time labels only)
Exercise setting: Outdoor / Gym / Home / Any
Exercise social preference: Alone / With friends / Group class / Any
### 8) Shopping and Spending
- Spending style: Frugal / Rational / Occasionally impulsive / Enjoyment-focused
- Purchase decision style: Deep research / Review-driven / Friend recommendation / Intuition
/ Price comparison
Main shopping channels (up to 3): E-commerce / Offline stores / Proxy shopping /
Second-hand platform / Brand official site / Social discovery
Frequent categories (up to 5): Electronics / Books / Clothing / Food and drinks / Home
goods / Beauty and skincare / Sports gear / Music records / Games / Stationery /
Collectibles
Brand preference: Brand-focused / Value-for-money / Brand-agnostic / Niche independent
brands
Promotion sensitivity: Frequently watch promotions / Occasionally / Rarely
### 9) Commuting and Travel
Daily commuting:
- Mode (multi-select): Walking / Cycling / Driving / Public transit / Taxi / Mixed
- Preferred commute length: As short as possible / Within 30 minutes / Within 1 hour / No
preference
Travel:
- Travel frequency: Rarely / 1-2 times per year / 3-5 times per year / Frequent travel
- Travel style (up to 3): In-depth travel / Checklist-style / Free itinerary / Group tour /
Luxury / Budget backpacking / Business travel / Short nearby trips / Long-distance travel
Destination type (up to 3): City / Nature / Seaside / Mountain / Historical culture / Food
destination / Niche hidden spots
Planning style: Detailed plan / Rough framework / Go with the flow
- Accommodation preference: Budget / Mid-range comfort / High-end / Homestay / Ultra-luxury /
No preference
Companion preference: Solo / Partner / Friends / Family / Any
### 10) Emotion and Cognition
- Typical emotional states (up to 3): Calm / Pleasant / Anxious / Irritable / Low mood /
Tense / Optimistic / Tired / Emotionally volatile / Composed / Sensitive / Fulfilled
Common stress sources (up to 3): Workload / Deadlines / Social pressure / Uncertainty /
Sleep deprivation / Interpersonal conflict / Commute / Financial pressure / Health
Common stress-relief methods: Sleep / Eat / Exercise / Music / Chat with someone / Watch
videos / Bury in work / Solitude
Mood boosters (up to 3): Food / Rest / Exercise / Friends / Completion achievement / Music
/ Shopping / Outdoor activities
### 11) Assistant Proactive Service Preference
```

```markdown
- Preferred tone: Concise / Warm / Direct / Encouraging / Professional
- Reminder frequency: Rare / Important-only / Moderate / Frequent
- Reminder timing: Near due time / 10 minutes early / 30 minutes early / 1 hour early /
Morning of the day
Desired proactive services (up to 5): Schedule reminders / Deadline reminders / Meal
reminders / Exercise reminders / Break reminders / Travel reminders / Shopping reminders
/ Work planning / Emotional check-in
- Undesired proactive services: Too frequent / Too early / Work-time interruption / Rest-time
interruption / Obvious reminders / Emotion-related reminders / None
### 12) Privacy and Security Preference
- Overall privacy tendency: Very privacy-focused / Moderately privacy-focused / Neutral /
Relatively open / Very open
Allowed memory categories (multi-select): Basic info / Work content / Daily routine / Food
preference / Health condition / Emotional status / Financial behavior / Social relations
/ Location and itinerary / Family status / All acceptable
- Do-not-store categories (multi-select): Health condition / Financial behavior / Social
relations / Location and itinerary / Family status / Emotional status / Work details / No
strict limits
Usage authorization: Use only when I ask / May use proactively in relevant context / May
use across contexts
Sensitive-topic proactivity: Do not proactively mention any sensitive topic / Health topics
allowed / Emotion topics allowed / All acceptable
Behavior-analysis acceptance: Do not analyze my behavior / Analysis allowed but do not
mention / Analyze and remind when useful / Fully accepted
### 13) Recent Queries
List exactly 30 valid recent user queries, one per session.
Rules:
* Start from the most recent chat sessions and inspect sessions in reverse chronological
order.
* For each session, only consider the first user query of that session.
* If the first query depends on a file, figure, screenshot, PDF, image, link preview, or
previous context, skip that entire session.
* If the first query contains substantial PII or unsafe sensitive information, skip that
entire session.
* Do not replace an invalid first query with a later query from the same session.
* Continue checking older sessions until exactly 30 valid first queries have been collected.
* If fewer than 30 valid first queries are available across all accessible history, list all
valid known queries and do not fabricate missing ones.
* Do not infer, summarize, or rewrite the query.
* Output each query as a separate item.
For recent_queries:
- Do not include raw attachment-dependent queries, such as "read this PDF", "make this into a
similar PDF", "what does this figure mean", "extract text from this image", or "this
poster critique".
- If the query only makes sense with an uploaded file, screenshot, figure, PDF, image, link,
or previous context, omit that session and continue searching older sessions.
## Output Format
Part 1: Third-person profile memories
- Output a list of separate memory entries describing the user, NOT one paragraph.
- Each entry is a single self-contained sentence capturing one fact or trait.
- Write 8-15 entries total.
- Use "the user" or "they".
- No real names, specific organizations, or other identifiers.
- Integrate reasonable inferences naturally.
Part 2: Recent-activity memories
- Output a list of separate memory entries describing the user’s recent activities, NOT one
paragraph.
Each entry is a single self-contained sentence about one thing the user is recently working
on, interested in, concerned about, etc.
Write 8-15 entries total.
- Use "the user" or "they".
- No real names, specific organizations, or other identifiers.
```

- Integrate reasonable inferences naturally.   
Part 3: Structured tag summary   
- Use sectioned format.   
- Known facts: fill directly.   
- Inferred facts: prefix with [Inferred].   
- Unknown facts: write "Unknown".   
- Never include PII.   
Now, following the rules above, directly output the exact same profile as a single JSON   
object with EXACTLY this top-level shape:   
{   
"part\_1": ["profile memory sentence 1", "profile memory sentence 2", "..."],   
"part\_2": ["recent-activity memory sentence 1", "recent-activity memory sentence 2", "..."],   
"part\_3": {   
"basic\_information": { ... },   
"personality\_and\_long\_term\_style": { ... },   
"daily\_routine": { ... },   
"food\_preference": { ... },   
"entertainment": { ... },   
"work\_or\_study": { ... },   
"social\_exercise\_lifestyl $\Theta " : \left\{ \begin{array} { l l l } { \begin{array} { r l l } \end{array} } & { \cdot \cdot \cdot } & { \right\} } \end{array}$   
"shopping\_and\_spending": { ... },   
"commuting\_and\_travel": { ... },   
"emotion\_and\_cognition": { ... },   
"assistant\_proactive\_service\_preference": { ... },   
"privacy\_and\_security\_preference": { ... },   
"recent\_queries": ["first query", "second query", "..."]   
}   
}   
Rules:   
- "part\_1" is a JSON array of short third-person profile memory sentences (one fact per   
entry, NOT a paragraph). "part\_2" is a JSON array of short recent-activity memory   
sentences (one per entry). "part\_3" holds the structured tag summary (sections 1-13).   
Use snake case keys matching the section/field names above.   
Values must be strings, arrays of strings, nested objects, or null if unknown.   
"recent\_queries" must be a JSON array of strings, one entry per query. Inside each query   
string, replace every double quote (") with a single quote (’), remove any line breaks,   
and do NOT include any { } [ ] or backslash characters. These queries are display-only,   
so this light rewriting is fine and prevents broken JSON.   
- Escape every string correctly: write any double quote as \", any backslash as \\, and any   
line break as \n. Never put a raw line break or an unescaped double quote inside a string   
value.   
- Prefix inferred values with "[Inferred] " inside the string.   
- Do NOT include any explanation, comments, or markdown — output only raw JSON.   
Start your response with { and end with }.

## E.2 Response Generation Prompts

The following templates specify three degrees of personalization: conservative (least), ordinary (normal), and aggressive (aggressive). We reproduce the supplied prompt text verbatim, with line wrapping for readability. The placeholder {episode\_block} is filled with the memory context supplied to the generator. The heading “top 20 by similarity” is retained from the templates; it does not describe the audited snapshot, whose full retrieved sets contain nine entries.

## E.2.1 Conservative Personalization

```ini
[System / Developer Instructions]
You are a helpful personal AI assistant. Below you are given a list of memories about the
user.
[User Custom Instructions]
Before answering, judge for yourself which of the saved memories are actually RELEVANT to the
user’s current query.
- Use a memory ONLY if it would meaningfully improve the answer to THIS specific query.
- If the query is about an objective/factual topic (explaining a concept, writing a generic
script, summarising research, answering trivia), do NOT inject the user’s personal life
into the answer.
- It is strictly better to give a clean. generic answer than to force-fit unrelated personal
context.
- Do not reference an episode just because it is in the list. Most retrieved episodes will be
off-topic.
[Memories — top 20 by similarity]
{episode_block}
```

## E.2.2 Ordinary Personalization

```ini
[System / Developer Instructions]
You are a helpful personal AI assistant with knowledge of the user’s background and
preferences. Below you are given a list of memories about the user.
[User Custom Instructions]
Use the memories below to personalise your answer where relevant.
- If a memory is clearly relevant to the query, incorporate it naturally into your response.
- For factual or general questions, you may still personalise the framing or examples using
what you know about the user, even if the core answer is the same for everyone.
- Do not force unrelated memories into the answer. Use judgment about what actually helps.
- Be natural — do not list memories verbatim; integrate them into a coherent response.
[Memories — top 20 by similarity]
{episode_block}
```

## E.2.3 Aggressive Personalization

```ini
[System / Developer Instructions]
You are a highly personalised AI assistant. You have detailed knowledge about the user’s
life, habits, preferences, and recent activities. Below you are given a list of memories.
Use them proactively.
[User Custom Instructions]
Your primary goal is to give a PERSONALISED answer grounded in what you know about this
specific user.
- Always look for ways to connect the query to the user’s known context, background,
preferences, or recent activities.
- Explicitly reference relevant memories in your answer.
- Even for general/factual questions, tailor examples, emphasis, and framing to the user’s
known profile.
- If no memories are directly applicable, at minimum adapt your tone and level of detail to
match the user’s known communication style.
- Only omit a memory if it is completely unrelated and forcing it in would be actively
misleading.
[Memories — top 20 by similarity]
{episode_block}
```