# RCCM: Diagnosing Gains and Harms in Reasoning-Prompt Transfer

Yuqi Zhang Di Zhang

## Abstract

A reasoning prompt can improve one deployment and degrade another. RCCM (Reasoning-Control Capability Mismatch) examines these reversals by pairing answers to the same questions with records of how generation ends. Across five open-weight deployments and nine strategies, aggregate accuracy hides substantial question-level gains and harms; a commonbudget MATH500 evaluation examines transfer on a separate sample. On 60 additional questions, concise reasoning improves Phi’s automatic answer matching at a 2,048-token cap across three collections, but sharply reduces Qwen’s. Author-reviewed final-answer commitments give a smaller, statistically unresolved Phi gain. Reserving an answer stage also improves matching, with additional input and call costs. These results show why a prompt benefit must be interpreted alongside its execution and evaluation: matching a reference answer need not mean delivering it, and an apparent compression benefit may admit another explanation. RCCM records this evidence for reproducible transfer comparisons.

## 1 Introduction

Transferring a successful reasoning prompt to another deployment does not necessarily preserve its benefit. An advantage can reverse across models (Wang et al., 2025), and prompt choice and formatting affect performance even within a model (Mizrahi et al., 2024; Sclar et al., 2024; Zhuo et al., 2024). With reasoning prompts, changing the text also changes the requested procedure: shorten a derivation, organize it into stages, or check an earlier answer. To understand the resulting accuracy change, we need to know which answers improve and which become wrong.

Consider Chain of Draft (CoD), which asks for terse intermediate steps (Xu et al., 2025). On the same 120 development questions, our Phi deployment answers 83 correctly with Chain of Thought (CoT) and 98 with CoD. Qwen moves in the opposite direction, from 88 to 56. The paired answers reveal how those totals arise: CoD rescues 20 CoT errors and harms five correct answers on Phi, but rescues five and harms 37 on Qwen (Figure 1). The same strategy therefore plays opposite roles. Its net effect depends on the gains and harms it produces under each execution configuration.

These strategies impose different demands on generation. CoT requests intermediate steps (Wei et al., 2022); CoD requests brief drafts (Xu et al., 2025); checking and repair request inspection and revision (Madaan et al., 2023; Huang et al., 2024). A requested operation, however, is not its observed execution, and execution is not task success. A shorter response may still be wrong; a correctly extracted number may still be an intermediate candidate. Understanding transfer requires following the requested operation through the response to the answer being evaluated.

RCCM studies reasoning-control capability mismatch through the requests a strategy makes and the outcomes a deployment produces. A deployment comprises the checkpoint, chat wrapper, decoding policy, output budget, and answer parser. For each fixed strategy comparison, we pair responses to common questions and relate answer changes to execution events. The resulting record shows which questions change correctness, how generation ends, and what the evaluator accepts as an answer.

The Phi observations motivate two tests of the apparent compression benefit: a concise request on additional questions and an explicit answer stage under the same generated-token cap. We also review whether the original responses commit to a correct final answer. These comparisons distinguish automatic matching from answer delivery and test explanations beyond brevity.

We make three contributions. First, a paired execution model decomposes a prompt’s net benefit into execution frequencies and the gains and harms within each category. Second, experiments across deployments expose strategy reversals and test a compression benefit using repeated collections, reviewed answer commitments, and answer-delivery controls. Third, a reproducible transfer record specifies the strategy, execution configuration, evaluator, costs, and validation scope, so the comparison can be checked and reused.

## 2 Related Work

Designing and adapting reasoning. CoT, selfconsistency, and self-refinement alter reasoning or answer aggregation (Wei et al., 2022; Wang et al., 2023; Madaan et al., 2023). CoD and Sketch-of-Thought use compact representations (Xu et al., 2025; Aytes et al., 2025); decoding can also expose reasoning paths without explicit CoT instructions (Wang and Zhou, 2024). Format-Adapter generates and selects reasoning formats under a test-time budget (Wang et al., 2026). Here, the procedure is fixed: we study how its answers and execution change when it is used on another deployment.

Compression and budget control. Concise CoT reports model-dependent accuracy costs (Renze and Guven, 2024), while reasoning-step experiments vary demonstration granularity (Jin et al., 2024). Lee et al. (2025) examine 31 compression prompts, question-level length thresholds, and accuracy–length tradeoffs. TALE estimates perquestion token budgets (Han et al., 2025); Mid-Think combines token-level cues to induce intermediate reasoning budgets (Yang et al., 2026). Strictcap evaluation compares truncation with an earlystop answer phase (Sun et al., 2025), and budgetaware strategy evaluation accounts for inference costs (Wang et al., 2024). Our comparison follows a fixed prompt edit on common questions, linking each rescue or harm to paired stopping states. An additional question sample and two hard caps test whether the development pattern persists.

CLARO combines attribute-guided prompting with reinforcement learning (Schlesinger et al., 2026); Budget Guidance trains a remaining-length predictor to steer generation (Li et al., 2026). Zhou et al. (2026) analyze positive and negative answer flips as reasoning budgets increase. REFRAIN adjusts stopping thresholds online using reflectiveredundancy signals, without fine-tuning the reasoning model (Sun et al., 2026). RCCM relates answer flips to execution changes under a fixed prompt edit. This joint analysis, followed by evaluator and delivery controls, distinguishes the study from budget-controller design.

Sensitivity and transfer. Multi-prompt and sensitivity studies document dependence on prompt choice (Mizrahi et al., 2024; Sclar et al., 2024; Lu et al., 2024; Zhuo et al., 2024); matching rules also affect measured sensitivity (Hua et al., 2025). PromptBridge learns a mapping from calibrated source–target prompt pairs (Wang et al., 2025), while Beyond the Answer compares scientific prompting across models (Pandey et al., 2026). Instance-adaptive CoT addresses instancelevel harm (Yuan et al., 2024). Our comparison keeps the prompt edit fixed and tests how its benefit changes with execution conditions, without learning a cross-model mapping.

Control requirements and task outcomes. The Format Tax separates prompt-level format costs from decoder constraints, analyzes answer flips and length, and tests decoupled reasoning and formatting (Lee et al., 2026). We share its distinction between an execution requirement and answer correctness. RCCM measures the joint stop-state and answer-transition distribution for a fixed edit and hard cap, then checks whether answer matching, post-thought scope, and reviewed answer commitment support the same interpretation. We distinguish reference-assisted repair from intrinsic feedback (Huang et al., 2024) and observable responses from the computation that produces them (Turpin et al., 2023).

Procedural-execution tests assess adherence to prescribed arithmetic steps (Panda et al., 2026). Our responses are freely generated, so a stop state cannot establish that the requested intermediate steps were followed. Human-aligned multiplechoice evaluation likewise distinguishes automatic matches from answer judgments (Molfese et al., 2025). For open answers, we keep matching, structural scope, and commitment review as separate endpoints.

## 3 From Control Requests to Answer Changes

A paired model of execution. Fix baseline and target strategies $m _ { 0 } , m _ { 1 }$ , including their templates and any multi-call procedure. Draw a question X from a declared distribution Q, which may be uniform over a fixed benchmark. Under deployment e, let $( O _ { 0 } , O _ { 1 } )$ be the paired response records, including stop reasons and interface failures. Their conditional joint law $K _ { e } ( \cdot \mid X )$ describes the collection protocol, with deterministic generation as a special case. A fixed evaluator $s ( X , O _ { j } ) \in \{ 0 , 1 \}$ assigns the answer-match label $Y _ { j }$ . An interface failure yields no successful match; the latent answer’s mathematical correctness remains unknown. RCCM models these observable records rather than an unmeasured internal capacity.

Gains, harms, and execution. Write $D = Y _ { 1 }$ $Y _ { 0 }$ . Its expectation $\delta _ { e }$ is the target’s advantage, in probability units. The events $D = 1$ and $D = - 1$ are rescues and harms. On n common questions their counts give

$$
{ \widehat { \delta } } _ { e } = ( R - H ) / n , \qquad \Delta \mathrm { A c c u r a c y } = 1 0 0 { \widehat { \delta } } _ { e } ( \mathrm { p p } ) .\tag{1}
$$

They compare responses to the same question, not necessarily a revision of the same response. Let $Z$ be an exhaustive execution category shared across deployments; our analysis uses the pair of stop states (length, normal, or interface failure). Define $w _ { e } ( z ) = \operatorname* { P r } _ { e } ( Z = z )$ and $\mu _ { e } ( z ) = \mathbb { E } _ { e } [ D \mid Z = z ]$ Then

$$
\delta _ { e } = \sum _ { z } w _ { e } ( z ) \mu _ { e } ( z ) .\tag{2}
$$

The contribution of an execution category depends on how often it occurs and whether its answer changes are beneficial. Table 4 gives the empirical decomposition.

What must persist for a benefit to transfer? Consider another deployment $e ^ { \prime }$ with the same strategies, question distribution, evaluator, and category definitions. Put $v _ { z } = \mathrm { m i n } \{ w _ { e } ( z ) , w _ { e ^ { \prime } } ( z ) \}$ and $\begin{array} { r } { \mathrm { T V } ( w _ { e } , w _ { e ^ { \prime } } ) = \frac { 1 } { 2 } \sum _ { z } | w _ { e } ( z ) - w _ { e ^ { \prime } } ( z ) | } \end{array}$ . An elementary bound is

$$
\begin{array} { r l } { \displaystyle | \delta _ { e ^ { \prime } } - \delta _ { e } | } & { \le \mathrm { 2 T V } \big ( w _ { e } , w _ { e ^ { \prime } } \big ) } \\ { \displaystyle } & { \phantom { \le } + \sum _ { z } v _ { z } | \mu _ { e ^ { \prime } } ( z ) - \mu _ { e } ( z ) | . } \end{array}\tag{3}
$$

Terms with $v _ { z } = 0$ contribute zero. The bound separates shifts in execution frequency from shifts in conditional task benefit. Identical stopping distributions alone cannot preserve a benefit: every pair could stop normally while changing from all rescues to all harms. Both deployments’ joint outcomes are needed to evaluate the bound; its purpose is to account for an observed transfer difference, not to predict one from source records alone.

Appendix G proves the bound and covers absent categories, evaluator changes, and sampling assumptions. We estimate execution frequencies and conditional gains and harms, then examine evaluator sensitivity. Stopping is observed after generation; the bound does not identify its causal effect. The additional-question study tests the fixed edits motivated by these measurements.

## 4 Experimental Setup

Deployments and data. The development panel runs Qwen 2.5 7B Instruct, Gemma 3 4B, Llama

3.2 3B Instruct, Mistral 7B, and Phi-4 Mini Reasoning 3.8B through Ollama. Its 120 questions comprise 40 GSM8K problems (Cobbe et al., 2021) and two 40-question MATH strata chosen using pilot difficulty (Hendrycks et al., 2021). We set temperature to zero and supply no explicit system message; deployment defaults may still apply. Output caps are 2,048 tokens for Qwen/Gemma, 1,024 for Llama/Mistral, and 4,096 for Phi. Appendix A separates retained historical identities from newer runtime snapshots and identifies missing serving metadata.

MATH500 evaluates six candidates on Qwen, Gemma, and Llama at a common 2,048-token cap. None of its 500 question texts exactly matches the development panel; sources and indices follow MATH/PRM800K (Hendrycks et al., 2021; Lightman et al., 2024). Diagnostic probes use 60 selected development questions, 20 per stratum. A subsequent study uses a separate fixed 60-question sample after excluding exact matches to recorded local inputs. Table 1 distinguishes these question sets from repeated generations and configurations.

Strategies. The nine development strategies are CoT, Concise CoT, CoD, Key-Step Draft (KSD), Full-Retry, KSPAC-Repair, Contrastive Key-Step Prompting (CKP), Meta-Cognitive Router (MCR-Route), and Progressive Key-Step Reasoning (PKR). CoT and CoD instantiate the published ideas under our settings. The other names identify local templates or workflows. MCR-Route is a development-calibrated routing reference. MATH500 uses CoT/Concise/CoD/KSD and two single-generation variants, KSPAC-Check and MCR-Instruction, rather than the development repair and routing pipelines. Appendix A gives the exact bindings.

Scoring and inference. The primary score is answer matching after the archived extraction procedure. We retain raw text, extraction provenance, and stop reasons. Exact paired McNemar tests (McNemar, 1947; Fagerland et al., 2013) receive Holm correction (Holm, 1979) over all 36 pairs per nine-strategy deployment and all 15 pairs per MATH500 deployment. A statistical top band contains candidates not significantly separated from the point winner in that family, without asserting equivalence. Pointwise 95% paired bootstrap intervals (Efron, 1979) use 20,000 resamples of common questions. Numerical near-best is a separate criterion: within max(2, ⌈0.03n⌉) matches of the observed best. Original collections contain one generation per request; later repeated collections retain the same questions and do not enlarge the independent sample.

<table><tr><td>Block</td><td>Interface</td><td>Items</td><td>Models</td><td>Outcomes</td></tr><tr><td>Development</td><td>Historical</td><td>120</td><td>5</td><td>5,400</td></tr><tr><td>MATH500</td><td>Historical</td><td>500</td><td>3</td><td>9,000</td></tr><tr><td>Additional study</td><td>Chat</td><td>60</td><td>2</td><td>960 assigned</td></tr><tr><td>Repeated collections</td><td>Chat</td><td>Same 60</td><td>2</td><td>960 served</td></tr><tr><td>Answer-phase controls</td><td>Native raw</td><td>Same 60</td><td>2</td><td>960 logical / 1,049 calls</td></tr></table>

Table 1: Main evidence blocks. The last three share one 60-question set, not 180 independent questions. The original additional study has 957 served outputs and three terminal interface errors. Raw controls are analyzed separately from chat collections. Historical configurations are not replaced by current model identities.

<table><tr><td>Strategy</td><td>Qwen</td><td>Gemma</td><td>Llama</td><td>Mistral</td><td>Phi</td></tr><tr><td>CoT</td><td>73.3</td><td>70.0</td><td>57.5</td><td>10.0</td><td>69.2</td></tr><tr><td>Concise</td><td>72.5</td><td>65.8</td><td>57.5</td><td>5.0</td><td>77.5</td></tr><tr><td>CoD</td><td>46.7</td><td>50.8</td><td>50.0</td><td>9.2</td><td>81.7</td></tr><tr><td>KSD</td><td>73.3</td><td>72.5</td><td>70.0</td><td>5.8</td><td>68.3</td></tr><tr><td>Full-Retry</td><td>60.0</td><td>66.7</td><td>55.0</td><td>19.2</td><td>60.0</td></tr><tr><td>KSPAC-Repair</td><td>60.0</td><td>66.7</td><td>50.8</td><td>16.7</td><td>64.2</td></tr><tr><td>CKP</td><td>72.5</td><td>67.5</td><td>60.8</td><td>15.8</td><td>77.5</td></tr><tr><td>MCR-Route</td><td>69.2</td><td>75.0</td><td>68.3</td><td>13.3</td><td>72.5</td></tr><tr><td>PKR</td><td>67.5</td><td>65.8</td><td>37.5</td><td>6.7</td><td>63.3</td></tr></table>

Table 2: Development answer accuracy (%), n = 120 per cell. Bold marks a point maximum. MCR-Route uses development information; its score is not unseendata routing performance.

## 5 How Fixed Strategies Change Roles

Opposite effects under the same request. Table 2 reports all nine strategies. CoD improves Phi by 15 correct answers and reduces Qwen by 32. Llama benefits from KSD, moving from 69 to 84 correct answers, while Phi moves from 83 to 82. These paired effects within each deployment cannot be summarized by the models’ overall accuracy ordering.

The rankings in Table 2 also differ. Mean pairwise Spearman correlation among Qwen, Gemma, Llama, and Phi is 0.394, ranging from −0.139 for Gemma–Phi to 0.831 for Gemma–Llama. Mistral’s scores are 5.0–19.2%, providing a low-performing reference; those scores alone do not diagnose an interface fault.

Equal accuracy hides different answers. On Qwen, CoT and KSD both score 88/120, but KSD rescues ten errors and harms ten correct answers. Twenty questions change correctness without changing accuracy. On Llama, CoD rescues 15 CoT errors but introduces 24 harms, giving a net loss. Figure 1 shows these cancellations. Such complementary answers offer an opportunity for selection, provided a selector can recognize them without access to the reference.

<table><tr><td>Strategy</td><td>Qwen</td><td>Gemma</td><td>Llama</td></tr><tr><td>CoT</td><td>72.6</td><td>70.8</td><td>42.0</td></tr><tr><td>Concise CoT</td><td>69.8</td><td>68.6</td><td>34.6</td></tr><tr><td>CoD</td><td>39.4</td><td>50.6</td><td>24.6</td></tr><tr><td>KSD</td><td>67.8</td><td>68.0</td><td>37.8</td></tr><tr><td>KSPAC-Check</td><td>71.4</td><td>74.8</td><td>41.4</td></tr><tr><td>MCR-Instruction</td><td>69.4</td><td>69.4</td><td>40.2</td></tr></table>

Table 3: MATH500 accuracy (%), n = 500 per cell, common 2,048-token cap. Check and Instruction are single-generation variants, distinct from the development pipelines.

A common budget on MATH500. At the common 2,048-token cap (Table 3), CoD trails CoT on all three deployments. Qwen and Llama score highest with CoT, Gemma with KSPAC-Check. Gemma’s 4.0-point Check advantage over CoT is not separated after Holm correction (p = 0.1059); both remain in its statistical top band. Qwen’s band excludes only CoD; Llama’s excludes CoD and Concise CoT.

The six-candidate oracle scores 86.0%, 84.8%, and 68.2% on Qwen, Gemma, and Llama, exceeding the best fixed scores by 13.4, 10.0, and 26.2 points. It measures complementarity: selection cannot use the reference answer.

Matching and sampling checks. Replaying extraction and matching reproduces the archived correctness labels for all 5,400 development and 9,000 MATH500 responses. Two extracted strings differ, but their correctness labels do not. An additional normalization check converts seven development errors to matches; the point winners are unchanged, apart from a new Qwen tie. Appendix E reports a separate five-sample plurality analysis with its ties and generation settings. These checks examine how extraction, normalization, and decoding choices affect the comparison.

Which prompts rescue more answers than they damage?  
![](images/ece02bcce13c954efc2e73d3d6e63eba75880b3240da5887838a6e55577f51fa.jpg)  
CC = Concise CoT; connecting segment joins two statistics of one paired comparison.  
Figure 1: Paired gains and harms relative to CoT on 120 development questions per deployment. CC denotes Concise CoT. Blue circles show baseline errors rescued by the target; orange squares show correct baseline answers harmed. Their directed difference is the net accuracy change, not a confidence interval. Deployment-specific caps are given in Section 4.

## 6 Execution, Evaluation, and Answer Delivery

## 6.1 What changes in the diagnostic probes?

On Phi’s 60 selected development probes at a 2,048- token cap, the concise request scores 40/60 versus 28/60 for full CoT. It rescues 14 errors and harms two correct answers: +20.0 points, 95% interval [8.3, 31.7], Holm p = 0.0293 across the seven reported probe comparisons. Median emitted tokens fall from 2,048 to 1,106; actual length stops fall from 34 to 21.

Table 4 shows where this gain occurs. Thirteen questions move from a length stop to a normal stop, and 12 also move from wrong to correct. The 21 length-to-length pairs contain two rescues and two harms, canceling in the net score. Equation 2 recovers the overall gain as (13/60)(12/13) = 0.20. Its concentration in length-to-normal pairs motivates testing a shorter request on additional questions. Answer-only prompting gives 33/60 with a median 1,445.5 tokens, further separating the requested brevity from the observed response length.

Structure and checking probes show numerical gains from multi-stage requests on Qwen and selfcheck on Gemma, but Gemma declines under repair. Appendix B lists all seven comparisons; only Phi full-to-concise passes their common correction.

<table><tr><td>Full → concise</td><td>n</td><td>R</td><td>H</td><td>Both</td><td>Neither</td></tr><tr><td>Length → normal</td><td>13</td><td>12</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Length → length</td><td>21</td><td>2</td><td>2</td><td>1</td><td>16</td></tr><tr><td>Normal → normal</td><td>26</td><td>0</td><td>0</td><td>25</td><td>1</td></tr><tr><td>Normal → length</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Total</td><td>60</td><td>14</td><td>2</td><td>26</td><td>18</td></tr></table>

Table 4: Phi development probes: stop-state and answer transitions. Both/Neither refer to correctness in both/neither condition. Row categories are exhaustive; stopping and correctness are different measurements.

The nine-condition profiles are marginal diagnostics, not isolated causal effects.

## 6.2 Additional questions and two budgets

We next fixed 60 additional questions, two deployments (Qwen and Phi), four prompts, and two budgets, yielding 960 assignments. The sample comprises 20 GSM8K, 20 MATH levels 1–3, and 20 MATH levels 4–5, selected by seeded withinstratum shuffling and recorded exact-text exclusions (Appendix I). Local registration specifies Phi full-to-concise at 2,048 tokens as the primary comparison, with all eight pair/deployment/budget contrasts in one Holm family. Exact requests, input hashes, model digests, and runtime settings were recorded before inference.

The study produced 957 responses and three terminal interface failures. A documented collection amendment preserved existing responses and completed missing assignments without changing the templates, questions, or scorer. We report answer matches per assigned question, list interface failures separately, and provide missing-answer bounds in the artifact. All three failures occur in Phi’s main CoD arm, outside the primary pair. The main and probe pairs retain different historical newline encodings; we therefore analyze them as specified prompt pairs, not a two-factor design.

Matching across collections. Phi improves from 37 to 45 matches at 2,048 tokens (+13.3 pp; Table 9). Seven rescues coincide with length-tonormal transitions; the eighth reverses that stop direction. At 4,096 tokens, the first collection has three rescues and three harms. Its descriptive budget contrast is −13.3 pp, interval [−23.3, −3.3], outside the corrected primary tests. Two further same-setting full/concise collections add 960 responses on these questions. Phi’s 2,048-token gains are +18.3/ + 23.3 pp (new eight-way Holm p = .0382/.00049); at 4,096 they are +11.7/+8.3 pp and do not pass correction. Qwen loses matches in every collection (Appendix J). The larger cap thus changes the observed effect without consistently eliminating it.

Requested compression and observed execution. The concise prompt asks for at most three short reasoning lines. Across the two budgets, Qwen emits a median 4–4.5 tokens, compared with about 936–941 for Phi. Qwen meets a whole-response three-line count on 57/60 questions at each budget; Phi, whose returned content includes marked thought text, meets it on none. The same instruction produces markedly different responses. Phi’s improved matching is therefore not evidence of literal compliance with the three-line request.

Three endpoints on the same answers. Table 5 applies three criteria to the same responses. The frozen scorer can match a number inside marked thought text. Structural scope considers unmarked output or the suffix of one closed thought section; missing or ambiguous scope counts as no observable match. For answer commitment, one author verified AI-assisted annotations with method names hidden. The resulting gain is smaller and statistically unresolved. All binary completions of the four unknown labels give [+3.3, +10.0] pp; this is a label-uncertainty range, not a confidence interval. The artifact supplies the rubric and blinded inputs. This review does not systematically assess the derivations (Appendix K).

Repeated-measure uncertainty. Across the three chat collections, Phi’s mean matching change is +18.3 pp at 2,048 and +6.7 at 4,096; Qwen’s is $- 5 0 . 6 / - 5 6 . 7 \mathrm { p p }$ . Table 6 keeps each question’s repeated observations together. It separates questioncluster uncertainty from a descriptive sensitivity that also resamples the three collections. These retrospective summaries exclude the raw answerphase block and leave the registered tests and independent question count unchanged (Appendix L).

## 6.3 Compression versus answer delivery

To test whether the benefit is specific to a shorter request, we compare it with reserving room for an answer. Following Sun et al. (2025), two answerphase controls reserve 128 tokens within the same generated-token cap, with or without a warning. A length stop triggers continuation; the controller can close an unmatched thought marker and supply an answer slot. All four strategies run through the same native raw interface on the original 60 questions. Generated-token caps are matched, but total computation is not.

At 2,048 tokens, both controls improve Phi’s matching over full after correction across all 24 comparisons (Table 7). Their differences from concise do not pass correction. Neither full-to-control contrast passes at 4,096, and neither control shows a corrected gain over full on Qwen. The three automatic scoring endpoints agree on this block; these new outputs have not received the separate commitment review (Appendix M).

Each Phi/2,048 control rescues eleven questions on a two-call target path and two on a one-call path. A has one two-call harm; W has none. These descriptive partitions do not isolate continuation: the arms also change first-stage allowance, injected text and possibly the warning. All paths and budgets are reported in Appendix L. Historical fitting and prompt revisions remain auxiliary evidence (Appendix H.1).

## 7 An RCCM Reporting Record

An RCCM record documents the comparison behind a transfer claim: the strategy, deployment, evaluator, costs, and validation scope, together with the observed execution and answer changes. It gives readers the evidence for a recommendation without prescribing how candidates must be selected.

<table><tr><td>Endpoint</td><td>Full</td><td>Concise</td><td>R/H</td><td>Change (pp)</td><td>Inference</td></tr><tr><td>Frozen matching</td><td>37</td><td>45</td><td>8/0</td><td>+13.3</td><td>8-way Holm  $p = . 0 3 1 2 5$ </td></tr><tr><td>Structural answer scope</td><td>32</td><td>44</td><td>12/0</td><td>+20.0</td><td>Sensitivity Holm  $p = . 0 0 2 9 3$ </td></tr><tr><td>Confirmed equivalent commitment</td><td>44</td><td>48</td><td>6/2</td><td>+6.7</td><td> $\mathrm { E x p l o r a t o r y } p = . 2 8 9 1$ </td></tr></table>

Table 5: Three endpoints on the original Phi/2,048 full-to-concise pair, $n = 6 0$ . Scope uses thought boundaries; commitment uses author-verified AI-assisted labels, with four uncertainties retained. Each endpoint has its own inference; none replaces the registered test.
<table><tr><td>Deployment / cap</td><td>Run-average (pp)</td><td>Question-cluster interval</td><td>Question + collection interval</td></tr><tr><td>Phi / 2048</td><td>+18.3</td><td>[11.1, 26.1]</td><td>[8.9,28.9]</td></tr><tr><td>Phi / 4096</td><td>+6.7</td><td>[0.0, 13.3]</td><td>[-2.2, 16.7]</td></tr><tr><td>Qwen / 2048</td><td>-50.6</td><td> $[ - 6 3 . { \dot { 3 } } , - 3 7 . 8 ]$ </td><td> $[ - 6 3 . 3 , - 3 7 . 2 ]$ </td></tr><tr><td>Qwen / 4096</td><td>-56.7</td><td> $[ - 6 8 . 3 , - 4 4 . 4 ]$ </td><td> $[ - 6 8 . 9 , - 4 3 . 9 ]$ </td></tr></table>

Table 6: Retrospective summaries of three chat collections on the same 60 questions. Pointwise 95% stratified bootstrap intervals keep paired observations together. The first conditions on the observed collections; the second also resamples their indices, a sensitivity based on only three collections. Raw controls are excluded; registered tests remain unchanged.

Propose an edit from the joint evidence. Phi’s development gains concentrate in length-to-normal transitions (Table 4), suggesting a concise request under a tighter cap. The edit, comparator, questions, budgets, and evaluator are then fixed. The probe motivates the test; it does not settle the recommendation.

Test the interpretation before adoption. Phi’s matching gains recur under the tested settings, but commitment review gives a smaller estimate, and an answer stage also improves matching. Recording these comparisons with their gains, harms, uncertainty, and costs lets a reader distinguish evidence that an edit helps from evidence for why it helps.

A worked record. In a fixed Qwen split, Concise CoT scores 27/30 on the probe and CoT 24/30. On the remaining 470 questions, CoT scores 339 and Concise 322. The 17-answer gap exceeds the declared 15-answer near-best tolerance. The record retains both the local win and the different validation outcome, rather than treating the probe winner as already transferable.

The historical RCCM-Guard candidate audit illustrates why these tasks should remain separate (Appendix D). Matched-size Direct has a higher near-best hit rate, while Guard more often retains the highest point estimate. These results describe a trade-off, not established selection or validationcost superiority. The reporting record can accompany either rule.

## 8 Conclusion

RCCM examines prompt transfer through paired answer changes and execution records. The same strategy can rescue some answers and damage others, even when aggregate accuracy is unchanged. Phi’s concise request illustrates the further distinction between a matching gain and its explanation: repeated collections, commitment review, and an answer-stage control each change how the benefit should be interpreted. Reporting these results together makes clear which procedure was tested and what evidence supports its reuse.

<table><tr><td>Arm</td><td>Matches</td><td>R/H vs F</td><td>Holm p</td><td>Calls</td><td>Output tok.</td><td>Input tok.</td></tr><tr><td>F: full</td><td>38</td><td>一</td><td>一</td><td>60</td><td>88991</td><td>6393</td></tr><tr><td>C: concise</td><td>41</td><td>6/3</td><td>1</td><td>60</td><td>70200</td><td>6513</td></tr><tr><td>A: full then answer</td><td>50</td><td>13/1</td><td>0.031128</td><td>86</td><td>88588</td><td>59556</td></tr><tr><td>W: warned then answer</td><td>51</td><td>13/0</td><td>0.0043945</td><td>89</td><td>86728</td><td>67369</td></tr></table>

Table 7: Phi/2,048 answer-phase comparison, $n = 6 0$ per arm. The native raw interface and generated-token cap are shared, not total compute. A/W reserve 128 tokens and continue only after a length stop. R/H and Holm p compare each arm with F in the full 24-test family. Commitment review does not cover these outputs.

## Limitations

The study covers English mathematics and the specified open-weight deployments and templates. Repeated collections and raw controls reuse the same 60 questions. Commitment annotation covers the original Phi pair through author-verified single review, not the later outputs or all rationales. Retained historical identities and current runtime snapshots describe their respective collections. The findings apply to the reported evaluators and execution configurations.

## Ethical Considerations

We use public mathematical benchmarks and model-generated answers, without recruiting external participants or collecting personal information. Sources and licenses are retained in the artifact. AI tools assisted research, coding, drafting, translation, and initial answer annotation; one author reports verifying every annotation individually. This is not an independent double review or rationale certification. The authors remain responsible for the scientific claims, references, and submission. The reporting workflow does not certify high-stakes use.

## References

Simon A. Aytes, Jinheon Baek, and Sung Ju Hwang. 2025. Sketch-of-thought: Efficient LLM reasoning with adaptive cognitive-inspired sketching. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24296– 24320, Suzhou, China. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

B. Efron. 1979. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7.

Morten W Fagerland, Stian Lydersen, and Petter Laake. 2013. The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional. BMC Medical Research Methodology, 13:91.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2025. Tokenbudget-aware LLM reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24842–24855, Vienna, Austria. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

S. Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6:65–70.

Andong Hua, Kenan Tang, Chenhe Gu, Jindong Gu, Eric Wong, and Yao Qin. 2025. Flaw or artifact? rethinking prompt sensitivity in evaluating LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 19889–19899, Suzhou, China. Association for Computational Linguistics.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations, volume 2024, pages 32808–32824.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024. The impact of reasoning step length on large language models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 1830–1842, Bangkok, Thailand. Association for Computational Linguistics.

Ayeong Lee, Ethan Che, and Tianyi Peng. 2025. How well do LLMs compress their own chain-ofthought? a token complexity approach. Preprint, arXiv:2503.01141.

Ivan Yee Lee, Loris D’Antoni, and Taylor Berg-Kirkpatrick. 2026. The format tax. Preprint, arXiv:2604.03616.

Junyan Li, Wenshuo Zhao, Yang Zhang, and Chuang Gan. 2026. Steering LLM thinking with budget guidance. In Findings of the Association for Computational Linguistics: ACL 2026, pages 37437–37451, San Diego, California, United States. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let's verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Sheng Lu, Hendrik Schuff, and Iryna Gurevych. 2024. How are prompts different in terms of sensitivity? In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5833–5856, Mexico City, Mexico. Association for Computational Linguistics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594. Curran Associates, Inc.

Quinn McNemar. 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12:153–157.

Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. 2024. State of what art? a call for multi-prompt LLM evaluation. Transactions of the Association for Computational Linguistics, 12:933–949.

Francesco Maria Molfese, Luca Moroni, Luca Gioffré, Alessandro Scirè, Simone Conia, and Roberto Navigli. 2025. Right answer, wrong score: Uncovering the inconsistencies of LLM evaluation in multiplechoice question answering. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 18477–18494, Vienna, Austria. Association for Computational Linguistics.

Sailesh Panda, Pritam Kadasi, Abhishek Upperwal, and Mayank Singh. 2026. When LLMs stop following steps: A diagnostic study of arithmetic procedural execution in language models. Preprint, arXiv:2605.00817.

Rohan Pandey, Eric Ye, and Michael Li. 2026. Beyond the answer: Decoding the behavior of llms as scientific reasoners. Preprint, arXiv:2603.28038.

Matthew Renze and Erhan Guven. 2024. The benefits of a concise chain of thought on problem-solving in large language models. In 2nd International Conference on Foundation and Large Language Models (FLLM), pages 476–483.

Oded Schlesinger, Young Kyung Kim, J. Matias Di Martino, and Guillermo Sapiro. 2026. CLARO: Controlled attribute-driven reasoning optimization for efficient chain-of-thought. In Findings of the Association for Computational Linguistics: ACL 2026, pages 26779–26799, San Diego, California, United States. Association for Computational Linguistics.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models'sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In International Conference on Learning Representations, volume 2024, pages 25055–25083.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. 2026. Stop when enough: Adaptive early-stopping for chain-of-thought reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27250–27268, San Diego, California, United States. Association for Computational Linguistics.

Yi Sun, Han Wang, Jiaqiang Li, Jiacheng Liu, Xiangyu Li, Hao Wen, Yizhen Yuan, Huiwen Zheng, Yan Liang, Yuanchun Li, and Yunxin Liu. 2025. An empirical study of LLM reasoning ability under strict output length constraint. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 7652–7671, Suzhou, China. Association for Computational Linguistics.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don't always say what they think: Unfaithful explanations in chain-ofthought prompting. In Advances in Neural Information Processing Systems, volume 36, pages 74952– 74965. Curran Associates, Inc.

Dingzirui Wang, Xuanliang Zhang, Rongyu Cao, Longxu Dou, Xianzhen Luo, Yingwei MA, Qingfu Zhu, Binhua Li, Fei Huang, and Yongbin Li. 2026. Format-adapter: Improving reasoning capability of LLMs by adapting suitable format. In Findings of the Association for Computational Linguistics: ACL 2026, pages 22408–22427, San Diego, California, United States. Association for Computational Linguistics.

Junlin Wang, Siddhartha Jain, Dejiao Zhang, Baishakhi Ray, Varun Kumar, and Ben Athiwaratkun. 2024. Reasoning in token economies: Budget-aware evaluation of LLM reasoning strategies. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19916–19939, Miami, Florida, USA. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations.

Xuezhi Wang and Denny Zhou. 2024. Chain-of-thought reasoning without prompting. In Advances in Neural Information Processing Systems, volume 37, pages 66383–66409. Curran Associates, Inc.

Yaxuan Wang, Quan Liu, Zhenting Wang, Zichao Li, Wei Wei, Yang Liu, and Yujia Bao. 2025. Promptbridge: Cross-model prompt transfer for large language models. Preprint, arXiv:2512.01420.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. 2025. Chain of draft: Thinking faster by writing less. Preprint, arXiv:2502.18600.

Van Yang, Shouren Wang, Debargha Ganguly, Xinpeng Li, Chaoda Song, Vikash Singh, Vipin Chaudhary, and Xiaotian Han. 2026. Mid-think: Training-free intermediate-budget reasoning via token-level triggers. In Findings of the Association for Computational Linguistics: ACL 2026, pages 6024–6038, San Diego, California, United States. Association for Computational Linguistics.

Xiaosong Yuan, Chen Shen, Shaotian Yan, Xiaofeng Zhang, Liang Xie, Wenxiao Wang, Renchu Guan, Ying Wang, and Jieping Ye. 2024. Instance-adaptive zero-shot chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 37, pages 125469–125486. Curran Associates, Inc.

Shu Zhou, Rui Ling, Junan Chen, Xin Wang, Tao Fan, and Hao Wang. 2026. When more thinking hurts: Overthinking in LLM test-time compute scaling. In Findings of the Association for Computational Linguistics: ACL 2026, pages 23967–23977, San Diego, California, United States. Association for Computational Linguistics.

Jingming Zhuo, Songyang Zhang, Xinyu Fang, Haodong Duan, Dahua Lin, and Kai Chen. 2024. ProSA: Assessing and understanding the prompt sensitivity of LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1950–1976, Miami, Florida, USA. Association for Computational Linguistics.

## A Data Blocks, Templates, and Deployment Records

This preprint includes a size-limited reproducibility subset in the arXiv ancillary directory: replay code, historical summary tables, and complete confirmation, repeatability, and answer-phase ledgers. The accompanying ARXIV\_SUBSET\_README.md lists included files and runnable checks. Long historical response and prompt ledgers remain in the separate full artifact; they are not included in this arXiv submission. Original records and frozen hashes in the supplied subset are unchanged.

The development tags are qwen2.5:7b, gemma3:4b, llama3.2:3b, mistral:7b, and phi4-mini-reasoning:3.8b. Historical records do not fully retain binary digests, quantization, and serving versions. The additional study captures the current model digests and server configuration before inference. These identify that study, not the earlier generations.

The model creators’ cards are linked in the accompanying asset-source index. The additional study records Ollama 0.32.6 and Q4\_K\_M quantization for both Qwen and Phi, with the exact model digests and native templates in confirmation/RUNTIME\_MANIFEST\_ANON.json. The GSM8K and MATH source releases carry MIT licenses; their notices are retained in the artifact. Model weights are not redistributed. The distributed responses and replay code are intended for reproducing the reported comparisons under the respective source terms.

The artifact provides data/templates.json, response-level template bindings, and original request text. CKP uses compact/verbose contrastive examples; MCR-Route combines classification with a development-calibrated routing table; PKR asks for uncertain steps to be identified and expanded. Main-panel KSPAC-Repair includes its multi-stage repair procedure. In MATH500, the archived keys kspac\_repair and mcr instead denote single-generation checking and planning requests, reported as KSPAC-Check and MCR-Instruction. Internal primary, repair, and fallback calls are retained where recorded.

Additional-study prompts. The main CoT instruction is “Solve the problem. Think step by step, then provide the final answer.” Main CoD asks for a minimal draft with at most five words per reasoning step and no full sentences unless necessary. Both use the historical literal backslash–n encoding around the problem and final-answer request. The full probe asks for “full step-by-step reasoning before giving the final answer.” The concise probe asks for “at most 3 short reasoning lines” and to “Avoid all extra explanation.” These probes use actual newlines. Each requests Final answer: <answer>. Exact bytes are supplied in confirmation/templates\_frozen.json and the

<table><tr><td>Block</td><td>Design</td><td>Role</td></tr><tr><td>Development</td><td>120 questions, five deployments, nine strategies</td><td>Paired gains/harms and rankings Fixed-candidate evaluation on another set</td></tr><tr><td>MATH500</td><td>500 questions, three deployments, six candidates; 2,048-token cap</td><td></td></tr><tr><td>Nine-condition profile Targeted probes</td><td>60 questions, four deployments, nine conditions 60 selected development questions per tested de-</td><td>Marginal behavior under requests Joint stop/answer transitions</td></tr><tr><td></td><td>ployment</td><td></td></tr><tr><td>Additional study</td><td>60 questions, two deployments, four prompts, two budgets</td><td>Fixed registered contrasts; 960 assigned outcomes</td></tr><tr><td>Retrospective fitting</td><td>7,140 records; 80/20 response-level split</td><td>Association with observed features</td></tr><tr><td>Other edits Sampling</td><td>Qwen 15 pairs; Falcon 120 per strategy 30 questions, four deployments, three strategies,</td><td>Supplied-rationale and prompt revisions Plurality and stop-event checks</td></tr><tr><td></td><td>five draws</td><td></td></tr><tr><td>Candidate audit</td><td>100 stratified 30/470 splits per MATH500 deploy- ment</td><td>Resampling sensitivity on cached re- sponses</td></tr></table>

Table 8: Distinct units and purposes. Repeated strategies and resampling do not add independent questions or deployments.

frozen request ledger.

The additional study uses temperature zero, seed 20261009, context length 8,192, and output caps 2,048/4,096. Registration is a local pre-run timestamp and hash record, not an external preregistration. Exact-text exclusion against recorded local inputs does not certify absence of semantic overlap or pretraining contamination. Collection amendments concern restart/status handling and terminal interface failures; they preserve original inputs and scoring. Three terminal errors occur in main CoD on Phi. At 2,048 tokens, its mathematical effect is bounded by [10.0, 11.7] points under the two possible scores for the missing answer; at 4,096 tokens, the bound is [8.3, 11.7]. Successful-match effects use the observed endpoint and retain all 60 assignments.

## A.1 Complete Registered Contrasts

Table 9 reports all eight registered comparisons on the additional question set, retaining the frozen endpoint and common correction family.

## B Operation Profiles and All Development Probes

Reference-assisted repair records. Archived repair probes provide the reference final answer to all three requests: error judgment, localization, and re-answering. The latter two also state that the input solution is wrong. They therefore evaluate responses with external answer information. The counts are retained below with their actual scoring definitions. The judgment task has only inputs labeled wrong, so always answering NO passes its rule; localization uses a text heuristic, not verified first-error annotations. These scores do not estimate independent detection or localization accuracy.

## C Uniform Answer-Scope Sensitivity

The frozen answer-matching endpoint uses the complete returned content. To check its dependence on marked thought text, we apply one retrospective rule to all 960 assignments. With no thought markers, use the whole output; with exactly one correctly ordered closed pair, use only the suffix after </think>; with an unclosed or ambiguous marked section, record no observable answer in the specified scope. Interface errors likewise provide no observed match. Apply the archived extractor and matcher to the available scope. The resulting endpoint is an observed outside-thought match, not mathematical correctness of an absent answer or a semantic commitment judgment. It remains separate from the registered analysis.

For Phi full/concise at 2,048 tokens, 24/16 served outputs have an unclosed or ambiguous marked scope; at 4,096 tokens, the counts are 14/13. Qwen has no unresolved marked scope. The primary frozen-score rescue involving a numeric fallback in unfinished thought text is retained in the original endpoint and excluded by this uniform sensitivity rule. Baseline-only matches are treated by exactly the same rule. Neither an extractor replay nor this structural check constitutes a human semantic review.

## D Historical Guard Rules and Candidate Audit

Let $a _ { m }$ be probe matches, $a _ { \mathrm { m a x } }$ the best probe count, and $p _ { m } , e _ { m } , h _ { m }$ the fractions of extraction failures, empty responses, and actual length stops. Let $u _ { m }$ be their union. A candidate is unsupported on the probe when $p _ { m } > 0 . 3 0 , e _ { m } > 0 . 0 5$ , or $h _ { m } > 0 . 5 0$ . Otherwise its supported condition is as follows; remaining cases are risky.

<table><tr><td>Deployment</td><td>Cap / pair</td><td>Base</td><td>Target</td><td>R/H</td><td>∆(pp)</td><td>Holm p</td><td>95% paired interval</td></tr><tr><td>Qwen</td><td>2048 / CoT-CoD</td><td>48</td><td>33</td><td>3/18</td><td>-25.0</td><td>.00894</td><td>[-38.3, -11.7]</td></tr><tr><td>Qwen</td><td>2048 / full–concise</td><td>42</td><td>12</td><td>1/31</td><td>-50.0</td><td> $1 . 0 8 \times 1 0 ^ { - 7 }$ </td><td>[−63.3, -36.7]</td></tr><tr><td>Qwen</td><td>4096 / CoT-CoD</td><td>46</td><td>31</td><td>5/20</td><td>-25.0</td><td>.02039</td><td>[−40.0, −10.0]</td></tr><tr><td>Qwen</td><td>4096 / full–concise</td><td>46</td><td>10</td><td>0/36</td><td>-60.0</td><td> $2 . 3 3 \times 1 0 ^ { - 1 0 }$ </td><td>[−71.7, −48.3]</td></tr><tr><td>Phi</td><td>2048 / CoT-CoD</td><td>37</td><td>43</td><td>8/2</td><td>+10.0</td><td>.32813</td><td>[0.0, 20.0]</td></tr><tr><td>Phi</td><td>2048 / full–concise</td><td>37</td><td>45</td><td>8/0</td><td>+13.3</td><td>.03125</td><td>[5.0, 21.7]</td></tr><tr><td>Phi</td><td>4096 / CoT-CoD</td><td>45</td><td>50</td><td>7/2</td><td>+8.3</td><td>.35938</td><td>[−1.7, 18.3]</td></tr><tr><td>Phi</td><td>4096 / full–concise</td><td>46</td><td>46</td><td>3/3</td><td>0.0</td><td>1.00000</td><td>[-8.3,8.3]</td></tr></table>

Table 9: All eight registered contrasts on 60 additional questions per cell. Counts use the frozen answer-matching endpoint; intervals are pointwise. Phi CoD has one terminal interface failure at 2,048 tokens and two at 4,096; these yield no successful match, with unknown mathematical correctness. Full/concise are probe templates, not the main CoT/CoD templates.

<table><tr><td>Request</td><td>Level</td><td>Q</td><td>G</td><td>L</td><td>P</td></tr><tr><td>Length</td><td>Answer-only</td><td>47.8</td><td>31.1</td><td>36.1</td><td>55.0</td></tr><tr><td rowspan="4">Structure</td><td>Concise</td><td>54.4</td><td>55.0</td><td>43.3</td><td>56.7</td></tr><tr><td>Full</td><td>58.3</td><td>51.1</td><td>43.9</td><td>46.1</td></tr><tr><td>Free</td><td>39.4</td><td>39.4</td><td>31.1</td><td>53.3</td></tr><tr><td>Simple</td><td>56.7</td><td>53.9</td><td>41.1</td><td>52.2</td></tr><tr><td rowspan="4">Checking</td><td>Multi-stage</td><td>64.4</td><td>43.9</td><td>51.1</td><td>52.2</td></tr><tr><td>None</td><td>48.9</td><td>40.0</td><td>44.4</td><td>55.0</td></tr><tr><td>Self-check</td><td>55.0</td><td>60.6</td><td>43.3</td><td>53.3</td></tr><tr><td>Repair</td><td>56.7</td><td>36.7</td><td>35.6</td><td>49.4</td></tr></table>

Table 10: Marginal accuracy (%) in the nine-condition design. Q/G/L/P denote Qwen/Gemma/Llama/Phi. Each level pools three conditions on the same 60 questions: 180 responses, not 180 independent questions. These are conditional summaries, not isolated factorial effects.

The archived term promotable denotes a completed audit disposition, not a pre-validation authorization:

$$
g _ { P } ( m ^ { * } ) = \mathrm { S } , \quad m ^ { * } \in \mathrm { N B } ( V ) , \quad m ^ { * } \notin \mathrm { F } ( V ) .\tag{4}
$$

Here NB is the numerical near-best set and F the declared validation reliability set. The other archived dispositions are local-only and unsupported. Their names and thresholds are retained for reproducing the earlier candidate audit; the main-text reporting record does not require this selector.

Candidates are ordered by match count, event union rate, mean emitted tokens, then the fixed order CoT, Concise CoT, CoD, KSD, KSPAC-Check, MCR-Instruction. Guard retains supported candidates, falling back to the first candidate if none is supported without changing its label. Validation near-best uses a max(2, ⌈0.03n⌉) match-count tolerance. The validation reliability set flags an event union rate above 5%, an accuracy gap of at least 15 points from the best candidate, or accuracy at most 20%. These are declared reporting criteria, not statistical risk guarantees.

Each MATH500 replay samples ten probe questions per difficulty stratum, with 30 probe and 470 validation questions. There are 100 repetitions per deployment, seed 20260726. A hit means at least one retained candidate belongs to the target set, including tied maxima; it is not recall of the entire set. Aggregate values first average within deployment and then across the three deployments.

Retention and evaluation budget. The 100 stratified 30/470 splits per deployment overlap and measure sensitivity to sampled probe questions. Highest-point hit rates are 54.7% for Direct top-2 and 72.3% for Guard; matching Direct’s candidate count to Guard gives 86.0% near-best and 66.3% highest-point hits. The table reports a hit in the target set, rather than recall of every member. On a new deployment, evaluating the full six-candidate reference pool costs $6 \times 3 0 = 1 8 0$ probe evaluations and $6 \times 4 7 0 = 2 , 8 2 0$ validation evaluations per split, plus any operation probes. Multi-stage strategies incur their internal calls as well. Cached replay is not new inference, and retaining fewer candidates does not avoid the cost of evaluating the reference pool.

The archived probe top-band comparator tests the point winner against its five alternatives; it retains that selection-dependent implementation as a candidate-retention reference. Main MATH500 inference instead uses all 15 pairs. Multiplying all interface thresholds by 0.75 or 1.25 changes none of 1,800 probe labels in this panel. Changing the compression tolerance from one answer to zero/two changes 103/105 labels and mean size to 1.83/2.53. Near-best tolerances of 2%/4% give Guard hit rates 80.7%/83.3%, versus 81.0% at 3%. These retrospective checks characterize the rules without selecting new thresholds from their outcomes.

<table><tr><td>Deployment</td><td>Baseline → target</td><td>Correct counts</td><td>∆(pp)</td><td>95% interval</td><td>Holm p</td></tr><tr><td>Qwen</td><td>Free → multi-stage</td><td>37 → 42</td><td>+8.3</td><td>[−1.7, 18.3]</td><td>.6797</td></tr><tr><td>Qwen</td><td>Simple → multi-stage</td><td>36 → 42</td><td>+10.0</td><td>[−1.7, 21.7]</td><td>.5923</td></tr><tr><td>Gemma</td><td>No-check → self-check</td><td>34 → 38</td><td>+6.7</td><td>[-8.3, 21.7]</td><td>.6797</td></tr><tr><td>Gemma</td><td>Self-check → repair</td><td>38 → 31</td><td>-11.7</td><td>[−23.3, 0.0]</td><td>.5923</td></tr><tr><td>Phi</td><td>Full → concise</td><td>28 → 40</td><td>+20.0</td><td>[8.3, 31.7]</td><td>.0293</td></tr><tr><td>Phi</td><td>Full → answer-only</td><td>28 → 33</td><td>+8.3</td><td>[−1.7, 20.0]</td><td>.6797</td></tr><tr><td>Phi</td><td>Answer-only → concise</td><td>33 → 40</td><td>+11.7</td><td>[0.0, 23.3]</td><td>.5537</td></tr></table>

Table 11: All seven targeted development comparisons, n = 60 paired questions each. Pointwise intervals and one common Holm family are retained. Qwen/Gemma caps are 1,024 tokens, Phi’s is 2,048. Edits can change wording and organization together.

<table><tr><td>Archived pass rule</td><td>Qwen</td><td>Gemma</td><td>Llama</td><td>Phi</td></tr><tr><td>Output NO</td><td>28/32</td><td>33/36</td><td>50/50</td><td>1/37</td></tr><tr><td>Localization text</td><td>32/32</td><td>36/36</td><td>50/50</td><td>32/37</td></tr><tr><td>Re-answer match</td><td>4/32</td><td>2/36</td><td>10/50</td><td>4/37</td></tr></table>

Table 12: Reference-assisted repair probes. Phi receives truncated visible traces; other inputs are complete. Localization requires at least 20 characters and specified error/step keywords. Requests are evaluated independently.

## E Sampling and Reproduction

The sampling block draws five answers per strategy for CoT, CoD, and KSD on 30 stratified questions in four deployments, using temperature 0.7, top-p 0.95, and seeds 101–105. It chooses the most frequent extracted answer string; ties prefer nonempty strings and then lexical order. Mathematical equivalence is not used to merge strings before voting. Of 360 voting units, 166 lack a three-of-five majority and 125 have a tied maximum. We therefore describe this as plurality aggregation.

The largest plurality accuracies are Qwen KSD 83.3%, Gemma KSD 80.0%, Llama CoT 73.3%, and Phi CoT 6.7%. Across 1,800 generations, there are 45 actual length stops, including 30 on Phi. The settings and ties accompany these results; they do not imply that sampling is generally ineffective.

The evidence package contains response and template indexes, scoring outputs, all comparison families, transitions, threshold replays, and figure inputs. Original extraction and matching code is preserved. The 14,400-response replay has zero correctness-label differences. Additional repeatability records and single-author AI-assisted annotations have separate indexes (Appendices J and K); they neither replace original scores nor certify all rationales. Replaying extraction, matching references, and judging answer commitments remain distinct checks.

## F Paired Effects Under the Primary Scoring

The following two tables give the complete CoTrelative comparisons used to interpret answer transitions in the two main data blocks.

## G Paired Execution Model: Assumptions and Derivations

Objects and assumptions. The input X includes the question and the evaluator’s reference information, which is not thereby supplied to the generator. Records contain text, execution events, and a terminal error symbol when no response is served. Strategies include their invocation and fallback rules. The response marginals of K<sub>e</sub>(· | X) must match the declared collection procedures. Independence between the two responses is unnecessary. With stochastic generation, rescue and harm probabilities depend on the pairing protocol, although their difference depends only on marginal accuracies.

The evaluator is a fixed, measurable binary function of the input and response. We treat answer matching as the operational endpoint; it is not a definition of the semantic validity of a rationale. A terminal interface failure receives zero for observed successful matching. Its unknown mathematical answer remains missing. Execution categories form a finite, exhaustive partition with the same meaning in each compared configuration. Our stop-pair partition has nine possible categories, including failure categories; unused categories have zero mass. A missing stop field would require its own category rather than silent exclusion.

<table><tr><td>Deployment</td><td>Cap / pair</td><td>Base</td><td>Target</td><td>R/H</td><td>∆(pp)</td><td>Holm p</td></tr><tr><td>Qwen</td><td>2048 / CoT-CoD</td><td>48</td><td>33</td><td>3/18</td><td>-25.0</td><td>.00745</td></tr><tr><td>Qwen</td><td>2048 / full–concise</td><td>42</td><td>12</td><td>1/31</td><td>-50.0</td><td> $1 . 0 8 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Qwen</td><td>4096 / CoT-CoD</td><td>46</td><td>31</td><td>5/20</td><td>-25.0</td><td>.01631</td></tr><tr><td>Qwen</td><td>4096 / full–concise</td><td>46</td><td>10</td><td>0/36</td><td>-60.0</td><td> $2 . 3 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Phi</td><td>2048 / CoT-CoD</td><td>33</td><td>41</td><td>12/4</td><td>+13.3</td><td>.23044</td></tr><tr><td>Phi</td><td>2048 / full–concise</td><td>32</td><td>44</td><td>12/0</td><td>+20.0</td><td>.00293</td></tr><tr><td>Phi</td><td>4096 / CoT-CoD</td><td>42</td><td>47</td><td>10/5</td><td>+8.3</td><td>.60352</td></tr><tr><td>Phi</td><td>4096 / full–concise</td><td>45</td><td>45</td><td>3/3</td><td>0.0</td><td>1.00000</td></tr></table>

Table 13: Retrospective scope sensitivity, all $n = 6 0$ assignments per cell. Holm is recalculated across these eigh sensitivity tests and does not replace Table 9. Missing scope is no observed delivered match.
<table><tr><td>Candidate</td><td>Supported condition</td></tr><tr><td>CoT</td><td> $u _ { \mathrm { C o T } } < 0 .$  40 and the best of the three compression candidates does not exceed CoT by at</td></tr><tr><td>Concise/CoD/KSD</td><td>least 10 points  $a _ { m } \geq a _ { \operatorname* { m a x } . } - 1$ </td></tr><tr><td>KSPAC-Check</td><td> $a _ { m } \equiv \operatorname* { m a x } ( a _ { \mathrm { C o T } } , a _ { \mathrm { K S D } } ) + 1$ </td></tr><tr><td>MCR-Instruction</td><td>The preceding count condition and  $u _ { m } - u _ { \mathrm { C o T } } \leq 0 . 1 0$ </td></tr></table>

Table 14: Development-derived probe rules, replayed with integer counts and denominators. Their provenance does not establish pre-registration before all development observations.

<table><tr><td>Rule</td><td>Size</td><td>Near-best</td><td>Highest point</td></tr><tr><td>Direct top-1</td><td>1.00</td><td>67.0</td><td>34.7</td></tr><tr><td>Direct top-2</td><td>2.00</td><td>86.0</td><td>54.7</td></tr><tr><td>Direct top-3</td><td>3.00</td><td>96.0</td><td>76.0</td></tr><tr><td>Probe top-band</td><td>5.55</td><td>100.0</td><td>100.0</td></tr><tr><td>Guard</td><td>2.18</td><td>81.0</td><td>72.3</td></tr><tr><td>Direct, matched size</td><td>2.18</td><td>86.0</td><td>66.3</td></tr></table>

Table 15: Candidate size and hit rates (%) across cached splits. Direct top-2 plus the same hard interface filter is unchanged at 86.0/54.7.

For the deployment comparison, strategies, Q, and the evaluator are held fixed. Changing the budget is part of changing the deployment. If the task distribution also changes, the algebra still holds for the resulting joint laws, but the difference then includes task shift and cannot be attributed to deployment alone. A scorer change is handled separately below. No structural equation for a latent “reasoning capacity” is assumed.

Paired decomposition. For binary labels, $D =$ $Y _ { 1 } - Y _ { 0 }$ equals one on a rescue, minus one on a harm, and zero otherwise. Hence

$$
\delta _ { e } = \operatorname* { P r } _ { e } ( Y _ { 0 } = 0 , Y _ { 1 } = 1 ) - \operatorname* { P r } _ { e } ( Y _ { 0 } = 1 , Y _ { 1 } = 0 ) .\tag{5}
$$

Partitioning these probabilities over Z proves Equation 2. On a finite ledger, let $n _ { z } , R _ { z } , H _ { ; }$ <sub>z</sub> be category

counts. Its exact empirical version is

$$
\widehat { \delta } _ { e } = \sum _ { z : n _ { z } > 0 } \frac { n _ { z } } { n } \frac { R _ { z } - H _ { z } } { n _ { z } } .\tag{6}
$$

Equal accuracy therefore constrains only the difference of rescues and harms. It does not require identical correct-answer sets. The Qwen CoT–KSD comparison in Figure 1 realizes this distinction with ten transitions in each direction.

Proof of the transfer bound. Abbreviate $w _ { e } ( z ) , w _ { e ^ { \prime } } ( z ) , \mu _ { e } ( z ) , \mu _ { e ^ { \prime } } ( z )$ by $w _ { z } , w _ { z } ^ { \prime } , \mu _ { z } , \mu _ { z } ^ { \prime }$ and put $v _ { z } = \operatorname* { m i n } ( w _ { z } , w _ { z } ^ { \prime } )$ . A zero-mass conditional mean may take any value in [−1, 1], since its coefficient is zero. Separate common and excess mass:

$$
\begin{array} { l } { { \displaystyle \delta _ { e ^ { \prime } } - \delta _ { e } = \sum _ { z } v _ { z } ( \mu _ { z } ^ { \prime } - \mu _ { z } ) } } \\ { { \displaystyle \qquad + \sum _ { z } ( w _ { z } ^ { \prime } - v _ { z } ) \mu _ { z } ^ { \prime } } } \\ { { \displaystyle \qquad - \sum _ { z } ( w _ { z } - v _ { z } ) \mu _ { z } . } } \end{array}\tag{7}
$$

Because $| \mu _ { z } | , | \mu _ { z } ^ { \prime } | ~ \leq ~ 1$ and both weight vectors sum to one, each excess-mass sum has total weight $\begin{array} { r } { 1 - \sum _ { z } v _ { z } = \mathrm { T V } ( w , w ^ { \prime } ) } \end{array}$ . Taking absolute values proves Equation 3. Its right-hand side B is at most two: $| \mu _ { z } ^ { \prime } \mathrm { ~ - ~ } \mu _ { z } | \ \le \ 2$ gives $\begin{array} { r } { B \leq 2 \mathrm { T V } ( w , w ^ { \prime } ) + 2 \sum _ { z } v _ { z } = 2 } \end{array}$ . The bound can nevertheless be loose for a particular comparison. If $\delta _ { e } > B ,$ , then $\delta _ { e ^ { \prime } } \geq \delta _ { e } - B > 0 .$ . This is a sufficient condition involving both deployments’ joint distributions, not a prediction rule using the source deployment alone.

<table><tr><td>Deployment</td><td>Target</td><td>n</td><td>Base</td><td>Target</td><td>R/H</td><td>∆(pp)</td><td>95% interval</td></tr><tr><td>Gemma</td><td>CoD</td><td>120</td><td>84</td><td>61</td><td>4/27</td><td>-19.2</td><td>[−27.5, -10.8]</td></tr><tr><td>Gemma</td><td>KSD</td><td>120</td><td>84</td><td>87</td><td>11/8</td><td>+2.5</td><td>[−4.2, 10.0]</td></tr><tr><td>Gemma</td><td>Concise</td><td>120</td><td>84</td><td>79</td><td>7/12</td><td>-4.2</td><td>[−11.7, 2.5]</td></tr><tr><td>Llama</td><td>CoD</td><td>120</td><td>69</td><td>60</td><td>15/24</td><td>-7.5</td><td>[−17.5, 2.5]</td></tr><tr><td>Llama</td><td>KSD</td><td>120</td><td>69</td><td>84</td><td>25/10</td><td>+12.5</td><td>[3.3, 21.7]</td></tr><tr><td>Llama</td><td>Concise</td><td>120</td><td>69</td><td>69</td><td>16/16</td><td>+0.0</td><td>[−9.2, 9.2]</td></tr><tr><td>Mistral</td><td>CoD</td><td>120</td><td>12</td><td>11</td><td>617</td><td>-0.8</td><td>[−6.7, 5.0]</td></tr><tr><td>Mistral</td><td>KSD</td><td>120</td><td>12</td><td>7</td><td>4/9</td><td>-4.2</td><td>[−10.0,1.7]</td></tr><tr><td>Mistral</td><td>Concise</td><td>120</td><td>12</td><td>6</td><td>4/10</td><td>-5.0</td><td>[−10.8, 0.8]</td></tr><tr><td>Phi</td><td>CoD</td><td>120</td><td>83</td><td>98</td><td>20/5</td><td>+12.5</td><td>[5.0, 20.8]</td></tr><tr><td>Phi</td><td>KSD</td><td>120</td><td>83</td><td>82</td><td>5/6</td><td>-0.8</td><td>[−5.8, 5.0]</td></tr><tr><td>Phi</td><td>Concise</td><td>120</td><td>83</td><td>93</td><td>16/6</td><td>+8.3</td><td>[0.8, 15.8]</td></tr><tr><td>Qwen</td><td>CoD</td><td>120</td><td>88</td><td>56</td><td>5/37</td><td>-26.7</td><td>[-35.8, -17.5]</td></tr><tr><td>Qwen</td><td>KSD</td><td>120</td><td>88</td><td>88</td><td>10/10</td><td>+0.0</td><td>[−7.5, 7.5]</td></tr><tr><td>Qwen</td><td>Concise</td><td>120</td><td>88</td><td>87</td><td>13/14</td><td>-0.8</td><td>[-9.2, 7.5]</td></tr></table>

Table 16: Development paired effects relative to CoT. Intervals are pointwise; full comparison families determine statistical top bands.
<table><tr><td>Deployment</td><td>Target</td><td>n</td><td>Base</td><td>Target</td><td> $R / H$ </td><td> $\Delta \left( \mathsf { p p } \right)$ </td><td>95% interval</td></tr><tr><td>Gemma</td><td>CoD</td><td>500</td><td>354</td><td>253</td><td>20/121</td><td>-20.2</td><td>[−24.6, -16.0]</td></tr><tr><td>Gemma</td><td>KSD</td><td>500</td><td>354</td><td>340</td><td>27/41</td><td>-2.8</td><td>[−6.0, 0.4]</td></tr><tr><td>Gemma</td><td>Concise</td><td>500</td><td>354</td><td>343</td><td>30/41</td><td>-2.2</td><td>[−5.6, 1.0]</td></tr><tr><td>Gemma</td><td>Check</td><td>500</td><td>354</td><td>374</td><td>41/21</td><td>+4.0</td><td>[1.0, 7.0]</td></tr><tr><td>Gemma</td><td>Instruction</td><td>500</td><td>354</td><td>347</td><td>33/40</td><td>-1.4</td><td>[−4.6, 2.0]</td></tr><tr><td>Llama</td><td>CoD</td><td>500</td><td>210</td><td>123</td><td>33/120</td><td>-17.4</td><td>[−22.0, −12.8]</td></tr><tr><td>Llama</td><td>KSD</td><td>500</td><td>210</td><td>189</td><td>48/69</td><td>-4.2</td><td>[-8.4,0.0]</td></tr><tr><td>Llama</td><td>Concise</td><td>500</td><td>210</td><td>173</td><td>52/89</td><td>-7.4</td><td>[−12.0, −2.8]</td></tr><tr><td>Llama</td><td>Check</td><td>500</td><td>210</td><td>207</td><td>65/68</td><td>-0.6</td><td>-5.2,4.0]</td></tr><tr><td>Llama</td><td>Instruction</td><td>500</td><td>210</td><td>201</td><td>54/63</td><td>-1.8</td><td>-6.2,2.4]</td></tr><tr><td>Qwen</td><td>CoD</td><td>500</td><td>363</td><td>197</td><td>15/181</td><td>-33.2</td><td>-38.0, -28.6]</td></tr><tr><td>Qwen</td><td>KSD</td><td>500</td><td>363</td><td>339</td><td>33/57</td><td>-4.8</td><td>-8.4, −1.0]</td></tr><tr><td>Qwen</td><td>Concise</td><td>500</td><td>363</td><td>349</td><td>36/50</td><td>-2.8</td><td>[−6.4,0.8]</td></tr><tr><td>Qwen</td><td>Check</td><td>500</td><td>363</td><td>357</td><td>34/40</td><td>-1.2</td><td>[−4.6, 2.2]</td></tr><tr><td>Qwen</td><td>Instruction</td><td>500</td><td>363</td><td>347</td><td>22/38</td><td>-3.2</td><td>[−6.2, −0.2]</td></tr></table>

Table 17: MATH500 paired effects relative to CoT. Intervals are pointwise; full comparison families determine statistical top bands.

Why stopping alone is insufficient. Consider one fixed question and deterministic response records whose stop category is normal-to-normal. In deployment e, the baseline answer is wrong and the target correct, giving $w _ { e } ( z ) ~ = ~ 1$ and $\mu _ { e } ( z ) = 1$ . In $e ^ { \prime } { \mathrm { . } }$ , let the baseline be correct and the target wrong, with the same stop category. Then $w _ { e ^ { \prime } } = w _ { e }$ but $\mu _ { e ^ { \prime } } ( z ) = - 1$ , and the effect changes from +1 to −1. This construction satisfies the model with a common evaluator and input distribution. Thus even the full stop-pair distribution cannot identify the direction of task benefit. Conditioning on a post-generation event also does not amount to intervening on that event.

Scorer sensitivity. Hold the response law fixed and apply two deterministic evaluators, s and se.

Let $\epsilon _ { j } = \operatorname* { P r } ( s ( X , O _ { j } ) \neq \tilde { s } ( X , O _ { j } ) )$ . Subtracting their paired effects and using the triangle inequality gives

$$
| \widetilde { \delta } _ { e } - \delta _ { e } | \leq \epsilon _ { 0 } + \epsilon _ { 1 } .\tag{8}
$$

Indeed, the absolute change in each binary score is exactly its disagreement indicator. This bound requires checking both arms. For Phi full–concise at 2,048 tokens, the uniform scope check changes five baseline matches and one target match to nonmatches, without adding matches. Thus $\epsilon _ { 0 } = 5 / 6 0$ $\epsilon _ { 1 } = 1 / 6 0$ , and the observed effect change, 4/60, is at most 6/60. Appendix C applies that check to every registered condition; the frozen primary result is not replaced.

Empirical and population readings. All decompositions apply exactly to the observed paired ledger without sampling assumptions. Reading its estimates as expectations for future items additionally requires a relevant sampling design and stability of the declared response procedures. Repeated runs on one item or overlapping resampling splits are not independent new items. The paired bootstrap preserves each observed response pair (Efron, 1979); its intervals quantify resampling variability and do not cover deployment shift. For independent, identically distributed response pairs, the exact conditional McNemar test conditions on the number of discordant pairs and uses a binomial probability of one half under marginal equality (McNemar, 1947; Fagerland et al., 2013). The fixed question strata and development selection therefore accompany the interpretation of those tests. Holm adjustment applies to the stated comparison family (Holm, 1979), not to an undisclosed search over candidate prompts.

The supplementary checks compare the derivations with finite distributions, boundary cases, and the reported empirical decomposition.

## H Additional Retrospective Controls

These analyses reuse the archived responses. They are additional retrospective controls, not new model generations or registered primary tests.

Information-matched fitting. All rows use the original 5,712/1,428 record split (seed 42), balanced logistic regression, and identical targets. Length is the log of recorded output tokens. Control features are the four archived compression, structure, verification, and repair descriptors. The split can share questions and deployments; AUC differences have not been significance-tested.

<table><tr><td>Features</td><td>Correctness</td><td>Interface event</td></tr><tr><td>Length</td><td>0.7029</td><td>0.8336</td></tr><tr><td>Deploy. + length</td><td>0.7940</td><td>0.9131</td></tr><tr><td>Method + deploy. + len.</td><td>0.8173</td><td>0.9433</td></tr><tr><td>Deploy. + control</td><td>0.7040</td><td>0.8920</td></tr><tr><td>Enriched</td><td>0.8021</td><td>0.9432</td></tr></table>

Table 18: Added same-split AUC controls. Enriched is the original descriptor set in Figure 2. Output length is observed after generation, not a pre-inference measurement.

## H.1 Historical Fitting and Prompt Edits

![](images/b75731181dcd06c8a2a364b72c533787524ab48c6d79c46584fd3b8358e86d1a.jpg)  
Figure 2: Retrospective fit and concrete prompt changes. (a) Method, Deploy., and Both use identifiers; Enriched combines deployment identity, control descriptions, and observed output length, followed by interactions. Targets are answer correctness and interface events. This is a record-level split, not unseen-deployment prediction. (b) Qwen compares supplied rationale lengths (15 pairs); Falcon uses separate CoD/KSD prompt revisions (120 pairs each); Phi uses the 60-question development full-to-concise contrast. Effects retain their own experimental settings and denominators.

![](images/4d6f741cf4e37c0f226da83b0c8c0d504b710be428d3707737d388728dfaec86.jpg)  
Tasks differ; effects are not pooled.

Figure 2 relates two further analyses of execution. On 7,140 responses split 80/20 by record, enriched features achieve AUC 0.802 for correctness and 0.943 for interface events. An added same-split control using method, deployment, and observed output length reaches 0.817 and 0.943, respectively (Appendix H). Control descriptions thus do not improve these point estimates over the simpler descriptor baseline. Questions and deployments can occur in both partitions; these are retrospective fits, not unseen-deployment predictions. The interfaceevent target is distinct from Guard’s validation reliability criterion.

The paired changes in the second panel show why the actual edit matters. Increasing supplied rationale length on Qwen changes 7/15 to 15/15 matches. Separate Falcon revisions change CoD from 3/120 to 32/120 and KSD from 64/120 to 35/120. They are method-specific prompt edits, not one common backend intervention. Phi’s point is the development contrast in Table 4.

Execution and answer scope. Table 19 retains the same full/concise responses as Table 9. The scope endpoint applies one structural rule to both arms. It does not certify a final semantic commitment.

At 4,096 tokens on Phi, three rescues in lengthto-normal pairs are offset by three harms in pairs ending at the length cap. On Qwen, all pairs stop normally despite the large loss in matches. Completion and correctness therefore remain distinct measurements.

Stratified sensitivity. A separate retrospective bootstrap resamples 20 questions within each of the three strata, using NumPy default\_rng(20261011) and 20,000 draws. For Phi at 2,048 tokens, its pointwise interval is [5.0, 23.3] pp, with stratum rescue/harm counts 2/0, 3/0, and 3/0. This sensitivity does not replace the registered pooled interval [5.0, 21.7] pp or its Holm family. It likewise does not estimate repeated-generation variation.

## I Additional-Sample Selection Replay

The source snapshot contains GSM8K test and seven MATH test subjects. Keeping MATH rows with a final boxed reference gives 6,319 candidates. A Python random.Random(20261009) generator shuffles each stratum in the order GSM8K, MATH levels 1–3, MATH levels 4–5. NFKC-normalized, case-folded alphanumeric question text identifies duplicates. The draw skips duplicates and recorded archived-input exclusions, takes the first 20 eligible items in each stratum, and shuffles the combined 60 using the same generator.

<table><tr><td>Deployment / cap</td><td>Request</td><td>Matches</td><td>Scope matches</td><td>Median tokens</td><td>Length stops</td><td>Unresolved scope</td></tr><tr><td>Qwen / 2048</td><td>full</td><td>42</td><td>42</td><td>426.5</td><td>0</td><td>0</td></tr><tr><td>Qwen / 2048</td><td>concise</td><td>12</td><td>12</td><td>4.5</td><td>0</td><td>0</td></tr><tr><td>Qwen / 4096</td><td>full</td><td>46</td><td>46</td><td>422.0</td><td>0</td><td>0</td></tr><tr><td>Qwen / 4096</td><td>concise</td><td>10</td><td>10</td><td>4.0</td><td>0</td><td>0</td></tr><tr><td>Phi / 2048</td><td>full</td><td>37</td><td>32</td><td>1864.0</td><td>26</td><td>24</td></tr><tr><td>Phi / 2048</td><td>concise</td><td>45</td><td>44</td><td>936.0</td><td>16</td><td>16</td></tr><tr><td>Phi / 4096</td><td>full</td><td>46</td><td>45</td><td>1858.0</td><td>15</td><td>14</td></tr><tr><td>Phi / 4096</td><td>concise</td><td>46</td><td>45</td><td>940.5</td><td>12</td><td>13</td></tr></table>

Table 19: Observed execution on the 60 additional questions in each cell. All 480 full/concise responses were served; the three interface failures occur in a different arm.
<table><tr><td>Deployment / cap</td><td>Stop pair</td><td>n</td><td>Rescue</td><td>Harm</td><td>Both</td><td>Neither</td></tr><tr><td>Qwen / 2048</td><td>normal→normal</td><td>60</td><td>1</td><td>31</td><td>11</td><td>17</td></tr><tr><td>Qwen / 4096</td><td>normal→normal</td><td>60</td><td>0</td><td>36</td><td>10</td><td>14</td></tr><tr><td>Phi / 2048</td><td>length→length</td><td>15</td><td>0</td><td>0</td><td>0</td><td>15</td></tr><tr><td>Phi / 2048</td><td>length→normal</td><td>11</td><td>7</td><td>0</td><td>4</td><td>0</td></tr><tr><td>Phi / 2048</td><td>normal→length</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Phi / 2048</td><td>normal→normal</td><td>33</td><td>0</td><td>0</td><td>33</td><td>0</td></tr><tr><td>Phi / 4096</td><td>length→length</td><td>10</td><td>0</td><td>1</td><td>0</td><td>9</td></tr><tr><td>Phi / 4096</td><td>length→normal</td><td>5</td><td>3</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Phi / 4096</td><td>normal→length</td><td>2</td><td>0</td><td>2</td><td>0</td><td>0</td></tr><tr><td>Phi / 4096</td><td>normal→normal</td><td>43</td><td>0</td><td>0</td><td>42</td><td>1</td></tr></table>

Table 20: Complete observed stop-pair categories for full-to-concise. Missing categories have zero observations. Row cells partition each 60-question comparison.

The encountered exclusion IDs are math\_test\_number\_theory\_0361 and math\_test\_precalculus\_0442. The artifact supplies the eight source files with dataset revisions and SHA256 hashes, the selected ledger, and scripts/replay\_selection.py. The replay checks all selected fields and their order conditional on these exclusions. It does not retrospectively authenticate all historical archive scans or the pre-run version of the selector source. No generated-answer score enters the draw.

The study estimates a balanced three-stratum comparison, rather than a prevalence-weighted score for the original benchmark mixture. The fixed primary analysis remains intact; the additional within-stratum sensitivity is reported separately.

## J Repeated Collection Under Fixed Settings

We collected the full/concise pair twice more on the same 60 questions at both caps and deployments: 960 new responses in 16 cells. Seed 20261009, temperature zero, the native templates, context length 8,192, model digests, and Ollama version 0.32.6 were retained. Each collection is reported separately; repeated responses are not additional inde-

pendent questions.

The eight new full-to-concise tests form a separate Holm family. Table 21 reports every contrast. The original four-prompt study and its eight-way correction are unchanged.

Phi’s gain repeats at 2,048 tokens and passes correction in both collections. At 4,096, the estimates are positive but do not pass correction. Together with the original zero net difference, these observations show variation at the larger cap rather than a consistent disappearance of the gain. Qwen loses matches in every collection.

Relative to the original collection, 882 of the 960 new outputs have the same correctness label, 916 have the same stop state, and 230 have exactly the same text. Generation therefore varies despite fixed sampling settings, although these records do not identify its source. The artifact verifies frozen requests, input and runtime-source hashes, and extracted-answer matches while preserving the original records.

## K Author-Verified Answer-Commitment Annotation

An additional retrospective review covered all 120 responses in the original Phi full/concise comparison at 2,048 tokens. A single author verified AIassisted initial annotations. Response order was randomized and method names, model tags, and automatic scores were hidden in the review kit. The author states that the mapping was not inspected before annotation; response style can nevertheless reveal a condition. This is an author-verified single review, not an independent double review.

<table><tr><td>Deployment</td><td>Cap</td><td>Collection</td><td>Base</td><td>Target</td><td>Rescue/Harm</td><td>Change (pp)</td><td>Holm p</td></tr><tr><td>Phi</td><td>2048</td><td>1</td><td>35</td><td>46</td><td>14/3</td><td>+18.3</td><td>0.0381775</td></tr><tr><td>Phi</td><td>2048</td><td>2</td><td>34</td><td>48</td><td>14/0</td><td>+23.3</td><td>0.000488281</td></tr><tr><td>Phi</td><td>4096</td><td>1</td><td>41</td><td>48</td><td>10/3</td><td>+11.7</td><td>0.18457</td></tr><tr><td>Phi</td><td>4096</td><td>2</td><td>46</td><td>51</td><td>6/1</td><td>+8.3</td><td>0.18457</td></tr><tr><td>Qwen</td><td>2048</td><td>1</td><td>42</td><td>12</td><td>2/32</td><td>-50.0</td><td> $3 . 4 6 9 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Qwen</td><td>2048</td><td>2</td><td>43</td><td>12</td><td>1/32</td><td>-51.7</td><td> $4 . 7 5 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Qwen</td><td>4096</td><td>1</td><td>44</td><td>10</td><td>0/34</td><td>-56.7</td><td> $9 . 3 1 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Qwen</td><td>4096</td><td>2</td><td>42</td><td>10</td><td>1/33</td><td>-53.3</td><td> $2 . 8 5 2 \times 1 0 ^ { - 8 }$ </td></tr></table>

Table 21: All eight new repeated-collection contrasts; each uses the same 60 paired questions and original answermatching endpoint. No new independent sample is claimed.
<table><tr><td>Deploy. / cap</td><td>Run</td><td>Label eq.</td><td>Stop eq.</td><td>Text eq.</td><td>Output tok.</td><td>Input tok.</td><td>Time (s)</td></tr><tr><td>Phi / 2048</td><td>1</td><td>103/120</td><td>107/120</td><td>0/120</td><td>163027</td><td>12906</td><td>2671.0</td></tr><tr><td>Phi / 2048</td><td>2</td><td>110/120</td><td>112/120</td><td>0/120</td><td>157502</td><td>12906</td><td>2556.5</td></tr><tr><td>Phi / 4096</td><td>1</td><td>109/120</td><td>110/120</td><td>0/120</td><td>236087</td><td>12906</td><td>4396.3</td></tr><tr><td>Phi / 4096</td><td>2</td><td>105/120</td><td>107/120</td><td>0/120</td><td>221535</td><td>12906</td><td>3589.0</td></tr><tr><td>Qwen / 2048</td><td>1</td><td>114/120</td><td>120/120</td><td>55/120</td><td>36508</td><td>13000</td><td>816.7</td></tr><tr><td>Qwen / 2048</td><td>2</td><td>115/120</td><td>120/120</td><td>60/120</td><td>36371</td><td>13000</td><td>795.3</td></tr><tr><td>Qwen / 4096</td><td>1</td><td>114/120</td><td>120/120</td><td>58/120</td><td>34998</td><td>13000</td><td>766.6</td></tr><tr><td>Qwen / 4096</td><td>2</td><td>112/120</td><td>120/120</td><td>57/120</td><td>36882</td><td>13000</td><td>805.3</td></tr></table>

Table 22: Agreement with the original collection and observed cost, pooling the 60 full and 60 concise responses only for descriptive agreement counts. Input-token counts need not equal uncached computation.

The rubric distinguishes final-answer commitment, reference equivalence, rationale validity, and uncertainty. An unclosed thought marker alone does not settle whether a response commits to an answer. Likewise, absence of a final answer is not a judgment of mathematical error. The artifact preserves every label and original note; the English manuscript reports their aggregate results.

The exact original English rubric, Chinese operational instructions, blinded response view and blank forms are preserved in answer\_annotations/original\_review\_kit.

Its then-pending wording is historical; the received author-verified labels are retained separately. The rubric asks for a readable, unambiguous commitment rather than an intermediate candidate, and does not decide commitment from a thought marker alone. Boundary illustrations include B002, accepted despite an unclosed marker after repeated conclusions; B076, where the author inferred a count from an exhaustive list; and B060/B113, unresolved because later candidates conflict. These are descriptions of the received judgments, not new rules retrospectively assigned to the original review. No labels were recoded in this revision.

<table><tr><td>Request</td><td>Equiv.</td><td>Not equiv.</td><td>No final</td><td>Uncertain</td></tr><tr><td>Full</td><td>44</td><td>1</td><td>13</td><td>2</td></tr><tr><td>Concise</td><td>48</td><td>0</td><td>10</td><td>2</td></tr></table>

Table 23: All 60 responses per request under the additional commitment rubric. Uncertain outcomes remain unresolved rather than being labeled mathematical errors.

Confirmed equivalent commitments give 44 versus 48 answers, with six rescues, two harms, and a +6.7 pp net gain. The exploratory paired test gives $p = 0 . 2 8 9 1$ . Across all 16 binary completions of the four unresolved labels, the gain ranges from +3.3 to +10.0 pp, with no nominal p below 0.05. This range describes the unresolved labels, not sampling uncertainty or annotation error.

There are 11 resolved disagreements with frozen answer matching and four unresolved cases. Among rationale labels, 118 are not checked and two are marked invalid despite equivalent final answers. Neither final-answer equivalence nor the original score replay validates the underlying derivation. The annotation is separate from all frozen test families and does not adjudicate the new repeated outputs.

<table><tr><td>Cell</td><td>Arm Calls</td><td></td><td>n</td><td>R H</td><td>B</td><td>N</td></tr><tr><td>Phi / 2048</td><td>A</td><td></td><td>1 34</td><td>2</td><td>0 32</td><td>0</td></tr><tr><td>Phi / 2048</td><td>A</td><td>2</td><td>26</td><td>11</td><td>1 5</td><td>9</td></tr><tr><td>Phi / 2048</td><td>W</td><td></td><td>1 31</td><td>2</td><td>0 29</td><td>0</td></tr><tr><td>Phi / 2048</td><td>W</td><td>2</td><td>29</td><td>11</td><td>0 9</td><td>9</td></tr><tr><td>Phi / 4096</td><td>A</td><td></td><td>1 46</td><td>0</td><td>0 45</td><td>1</td></tr><tr><td>Phi / 4096</td><td>A</td><td></td><td>2 14</td><td>9</td><td>0 1</td><td>4</td></tr><tr><td>Phi / 4096</td><td>W</td><td></td><td>1 42</td><td>0</td><td>0 41</td><td>1</td></tr><tr><td>Phi / 4096</td><td>W</td><td></td><td>2 18</td><td>8</td><td>0 5</td><td>5</td></tr><tr><td>Qwen / 2048</td><td>A</td><td>1</td><td>60</td><td>1</td><td>2 41 16</td><td></td></tr><tr><td>Qwen / 2048</td><td>A</td><td>2</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>Qwen / 2048</td><td>W</td><td></td><td>1 59</td><td>7</td><td>2 40 10</td><td></td></tr><tr><td>Qwen / 2048</td><td>W</td><td>2</td><td>1</td><td>0</td><td>1 0</td><td>0</td></tr><tr><td>Qwen / 4096</td><td>A</td><td></td><td>1 60</td><td>0</td><td>2 42</td><td>16</td></tr><tr><td>Qwen / 4096</td><td>A</td><td>2</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>Qwen / 4096</td><td>W</td><td></td><td>1 59</td><td>7</td><td>4 39</td><td>9</td></tr><tr><td>Qwen / 4096</td><td>W</td><td>2</td><td>1</td><td>0</td><td>1 0</td><td>0</td></tr></table>

Table 24: Full-to-A/W outcomes by realized target call path. R/H: rescue/harm; B/N: both/neither correct. Each two-row partition sums to its 60 pairs and registered R/H. Triggering is an observed event, not an assigned causal treatment.

## L Repeated-Collection and Path Summaries

We summarize the completed records retrospectively. The dated specification and code are archived separately from the original registered analyses. These summaries add no generations or primary tests.

For each model and budget, form a 60 × 3 matrix of concise-minus-full labels from the three chat collections. Average within question and then across the balanced strata. Question-cluster resampling retains all three collection observations. The joint sensitivity additionally resamples three collection indices globally within a draw, shared across the two budgets. Both use 20,000 draws and numpy default\_rng(20261012). This preserves within-question and cross-budget pairing; three observed collections do not identify the distribution of arbitrary future serving changes.

Table 6 reports both pointwise summaries. The paired 2,048-minus-4,096 effect contrasts are:

Phi: +11.7 pp, question-cluster interval [5.0, 18.9], joint interval [1.7, 21.1].

Qwen: +6.1 pp, question-cluster interval [1.1, 11.7], joint interval [0.0, 13.9].

Both contrasts are descriptive and add no primary significance decisions.

For Phi/2,048, both A and W rescue eleven questions on two-call target paths and two on one-call paths. A has one two-call harm; W has none. The comparisons also involve regenerated first-stage responses and the warning/budget procedure, so they do not isolate the effects of continuation, thought closure, or budget reservation.

## M Equal-Generated-Cap Answer-Phase Comparison

We compare full CoT, the concise request, and two answer-phase controls on the original 60 questions in each deployment/cap. This block reuses the questions rather than adding a held-out sample. The controls adapt the early-answer approach of Sun et al. (2025) as existing-method baselines for the compression interpretation.

All four arms use native prefixes through the same raw generation interface, with temperature zero, seed 20261009, context 8,192, and the recorded model digests and Ollama 0.32.6. The two controls reserve 128 tokens from cap B, generate at most B −128 initially, and continue the same assistant prefix only on a length stop. A normal stop uses the initial response. The warned control appends: If [ANSWER NOW] appears, stop the intermediate work and immediately give your best final answer.

The controller closes an unmatched thought marker, when present, then supplies [ANSWER NOW] and the fixed Final answer: slot. These are input injections, not generated answer text. The primary score uses this slot plus the generated continuation, or the original response when no continuation occurs. Prespecified sensitivities score the generated suffix alone and the whole returned trace. All 960 matching labels agree across the three endpoints; their three separately corrected 24-test families therefore have identical numerical results. This does not certify semantic answer commitment or intermediate reasoning.

At 2,048 tokens, Phi moves from 38 full matches to 50/51 with A/W, with 13 rescues and 1/0 harms. Both improvements pass the registered correction. Concise gives 41 matches; its contrasts with A/W have Holm p = .2920/.0952, so neither establishes superiority over concise. The full-to-A/W contrasts do not pass correction at 4,096. Qwen rarely requires continuation, and its controls show no significant difference from full after correction; this leaves equivalence unresolved.

<table><tr><td>Deployment / cap</td><td>Edit</td><td>Base</td><td>Target</td><td>R/H</td><td>Change (pp)</td><td>95% interval</td><td>Holm p</td></tr><tr><td>Phi / 2048</td><td>F → C</td><td>38</td><td>41</td><td>6/3</td><td>+5.0</td><td>[−5.0, 15.0]</td><td>1</td></tr><tr><td>Phi / 2048</td><td>F → A</td><td>38</td><td>50</td><td>13/1</td><td>+20.0</td><td>[10.0, 31.7]</td><td>0.031128</td></tr><tr><td>Phi / 2048</td><td>F → W</td><td>38</td><td>51</td><td>13/0</td><td>+21.7</td><td>[11.7, 31.7]</td><td>0.0043945</td></tr><tr><td>Phi / 2048</td><td> $\mathbf { C } \to \mathbf { A }$ </td><td>41</td><td>50</td><td>11/2</td><td>+15.0</td><td>[5.0, 25.0]</td><td>0.29199</td></tr><tr><td>Phi / 2048</td><td> $\mathrm { C } \to \mathrm { W }$ </td><td>41</td><td>51</td><td>11/1</td><td>+16.7</td><td>[6.7, 26.7]</td><td>0.095215</td></tr><tr><td>Phi / 2048</td><td> $\mathbf A \to \mathbf W$ </td><td>50</td><td>51</td><td>3/2</td><td>+1.7</td><td>[−5.0, 8.3]</td><td>1</td></tr><tr><td>Phi / 4096</td><td>F → C</td><td>46</td><td>50</td><td>4/0</td><td>+6.7</td><td>[1.7, 13.3]</td><td>1</td></tr><tr><td>Phi / 4096</td><td>F → A</td><td>46</td><td>55</td><td>9/0</td><td>+15.0</td><td>[6.7, 23.3]</td><td>0.0625</td></tr><tr><td>Phi / 4096</td><td>F → W</td><td>46</td><td>54</td><td>8/0</td><td>+13.3</td><td>[6.7, 21.7]</td><td>0.10938</td></tr><tr><td>Phi / 4096</td><td> $\mathbf { C } \to \mathbf { A }$ </td><td>50</td><td>55</td><td>5/0</td><td>+8.3</td><td>[1.7, 15.0]</td><td>0.75</td></tr><tr><td>Phi / 4096</td><td> $\mathrm { C } \to \mathrm { W }$ </td><td>50</td><td>54</td><td>6/2</td><td>+6.7</td><td>[−1.7, 15.0]</td><td>1</td></tr><tr><td>Phi / 4096</td><td>A → W</td><td>55</td><td>54</td><td>1/2</td><td>-1.7</td><td>[−6.7, 3.3]</td><td>1</td></tr><tr><td>Qwen / 2048</td><td> $\mathrm { F } \to \mathrm { C }$ </td><td>43</td><td>12</td><td>2/33</td><td>-51.7</td><td> $[ - 6 5 . 0 , - 3 6 . 7 ]$ </td><td> $7 . 7 1 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $\mathrm { Q w e n } / 2 0 4 8$ </td><td> $\mathrm { F } \to \mathbf { A }$ </td><td>43</td><td>42</td><td>1/2</td><td>-1.7</td><td>[−6.7, 3.3]</td><td>1</td></tr><tr><td> $\mathrm { Q w e n } / 2 0 4 8$ </td><td> $\mathrm { F } \to \mathbb { W }$ </td><td>43</td><td>47</td><td>7/3</td><td>+6.7</td><td> $[ - 3 . 3 , 1 6 . 7 ]$ </td><td>1</td></tr><tr><td> $\mathrm { Q w e n } / 2 0 4 8$ </td><td>C → A</td><td>12</td><td>42</td><td>32/2</td><td>+50.0</td><td>[35.0, 63.3]</td><td> $1 . 3 8 8 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $\mathrm { Q w e n } / 2 0 4 8$ </td><td>C → W</td><td>12</td><td>47</td><td>36/1</td><td>+58.3</td><td>[45.0, 70.0]</td><td> $1 . 3 2 7 \times 1 0 ^ { - 8 }$ </td></tr><tr><td> $\mathrm { Q w e n } / 2 0 4 8$ </td><td>A → W</td><td>42</td><td>47</td><td>9/4</td><td>+8.3</td><td>[−3.3, 20.0]</td><td>1</td></tr><tr><td>Qwen / 4096</td><td> $\mathrm { F } \to \mathrm { C }$ </td><td>44</td><td>13</td><td>1/32</td><td>-51.7</td><td>[−65.0, -38.3]</td><td> $1 . 7 4 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Qwen / 4096</td><td> $\mathrm { F } \to \mathbf { A }$ </td><td>44</td><td>42</td><td>0/2</td><td>-3.3</td><td>[−8.3, 0.0]</td><td>1</td></tr><tr><td>Qwen / 4096</td><td>F → W</td><td>44</td><td>46</td><td>7/5</td><td>+3.3</td><td>[−8.3, 15.0]</td><td>1</td></tr><tr><td>Qwen / 4096</td><td> $\mathbf { C } \to \mathbf { A }$ </td><td>13</td><td>42</td><td>31/2</td><td>+48.3</td><td>[33.3,61.7]</td><td> $2 . 4 8 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Qwen / 4096</td><td> $\mathrm { C } \to \mathrm { W }$ </td><td>13</td><td>46</td><td>34/1</td><td>+55.0</td><td>[41.7,66.7]</td><td> $4 . 8 2 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Qwen / 4096</td><td> $\mathbf A \to \mathbf W$ </td><td>42</td><td>46</td><td>9/5</td><td>+6.7</td><td>[−5.0, 18.3]</td><td>1</td></tr></table>

Table 25: All 24 registered primary comparisons on 60 paired questions each. F: full; C: concise; A: full then answer; W: warned then answer. One Holm family spans all cells and pairs. Intervals are pointwise stratified bootstrap intervals, not simultaneous intervals. Suffix-only and whole-trace sensitivity families give identical matching results; all 72 comparisons are supplied in the artifact.
<table><tr><td>Deployment / cap</td><td>Arm</td><td>Calls</td><td>Two-phase</td><td>Output tok.</td><td>Input tok.</td><td>Time (s)</td><td>Median tok.</td></tr><tr><td>Phi / 2048</td><td>F</td><td>60</td><td>0</td><td>88991</td><td>6393</td><td>1908.4</td><td>1498.0</td></tr><tr><td>Phi / 2048</td><td>C</td><td>60</td><td>0</td><td>70200</td><td>6513</td><td>1487.6</td><td>967.0</td></tr><tr><td>Phi / 2048</td><td>A</td><td>86</td><td>26</td><td>88588</td><td>59556</td><td>1910.8</td><td>1575.5</td></tr><tr><td>Phi / 2048</td><td>W</td><td>89</td><td>29</td><td>86728</td><td>67369</td><td>1876.2</td><td>1783.5</td></tr><tr><td>Phi / 4096</td><td>F</td><td>60</td><td>0</td><td>127810</td><td>6393</td><td>2757.9</td><td>1549.5</td></tr><tr><td>Phi / 4096</td><td>C</td><td>60</td><td>0</td><td>101592</td><td>6513</td><td>2199.5</td><td>1073.5</td></tr><tr><td>Phi / 4096</td><td>A</td><td>74</td><td>14</td><td>127568</td><td>63739</td><td>2759.8</td><td>1645.0</td></tr><tr><td>Phi / 4096</td><td>W</td><td>78</td><td>18</td><td>133237</td><td>81463</td><td>2892.5</td><td>1892.0</td></tr><tr><td>Qwen / 2048</td><td>F</td><td>60</td><td>0</td><td>35175</td><td>6440</td><td>1023.3</td><td>405.0</td></tr><tr><td>Qwen / 2048</td><td>C</td><td>60</td><td>0</td><td>892</td><td>6560</td><td>39.7</td><td>4.0</td></tr><tr><td>Qwen / 2048</td><td>A</td><td>60</td><td>0</td><td>36170</td><td>6440</td><td>1061.0</td><td>411.5</td></tr><tr><td>Qwen / 2048</td><td>W</td><td>61</td><td>1</td><td>30941</td><td>9720</td><td>905.6</td><td>355.5</td></tr><tr><td>Qwen / 4096</td><td>F</td><td>60</td><td>0</td><td>36373</td><td>6440</td><td>1059.3</td><td>414.5</td></tr><tr><td>Qwen / 4096</td><td>C</td><td>60</td><td>0</td><td>1013</td><td>6560</td><td>42.9</td><td>4.0</td></tr><tr><td>Qwen / 4096</td><td>A</td><td>60</td><td>0</td><td>36197</td><td>6440</td><td>1052.7</td><td>405.0</td></tr><tr><td>Qwen / 4096</td><td>W</td><td>61</td><td>1</td><td>34560</td><td>11767</td><td>1003.2</td><td>347.0</td></tr></table>

Table 26: Complete answer-phase cost ledger, 60 items per row. Each item remains within its generated-token cap across both stages. Injected prefixes consume input, not output, and extra calls/prefill mean this is not an equal-total-compute comparison. Input counts do not estimate uncached FLOPs.

The run completed 960 logical outcomes through 1,049 calls and 1,036,035 generated tokens. Frozen requests, runtime-source and scorer hashes, stage payloads, all three endpoints, exact tests, Holm correction, intervals and costs were checked. The complete artifact retains all attempts, calls and outcomes. These generations have not received the separate human review of the original 120 responses.