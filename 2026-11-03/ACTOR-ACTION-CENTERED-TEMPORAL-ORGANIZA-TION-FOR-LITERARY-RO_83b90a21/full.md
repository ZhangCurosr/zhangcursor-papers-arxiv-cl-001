# ACTOR: ACTION-CENTERED TEMPORAL ORGANIZA-TION FOR LITERARY ROLE-PLAYING AGENTS

Yuxiang Gao<sup>1</sup> Tianjie Ju<sup>2∗</sup> Zheng Wu<sup>3</sup> Zongru Wu<sup>3,2</sup> Hao Fei<sup>4</sup>

<sup>1</sup>Harbin Institute of Technology

<sup>2</sup>National University of Singapore

<sup>3</sup>Shanghai Jiao Tong University

<sup>4</sup>University of Oxford

## ABSTRACT

Large language agents for literary role-playing are typically modeled around dialogue generation, leaving characters’ non-dialogue behaviors and their dynamic evolution over story time underrepresented. In this paper, we propose ACTOR (Action-Centered Temporal Organization for Literary Role-Playing Agents), an automated framework for constructing such behavior trajectories from raw novel texts. ACTOR extracts dialogue and persona-relevant non-dialogue behaviors, represents each behavior using a Thought–Action–Observation structure, and organizes behavior trajectories according to the internal temporal order of the story. We build a behavior dataset from 143 diverse literary works. For each source behavior, we generate counterfactual behaviors that are contextually plausible yet inconsistent with the character’s persona. We find that existing models struggle to reliably infer character-consistent actions from historical behaviors and evolving character states. Using these behavior pairs, we train models with Regularized Preference Optimization (RPO), which augments DPO with a negative log-likelihood loss on the source behavior to jointly optimize behavior preference and generation. We further design a four-way next-behavior selection task to evaluate the model’s ability to faithfully mimic character actions. Across 17,560 evaluation instances, the model trained on ACTOR’s preference pairs achieves substantially improved accuracy. This result demonstrates that structured, temporally organized supervision enables models to leverage character profiles and prior behaviors to infer subsequent actions, greatly advancing literary role-playing beyond mere dialogue imitation. Code and data are available at https://github.com/SilentStar233/ACTOR.

## 1 INTRODUCTION

As research on role-playing language agents advances, researchers have constructed richer role representations and high-quality training datasets (Wang et al., 2026). Early studies draw on diverse textual sources such as scripts, biographies, and novels to build faithful role representations. These sources support retrieval, fine-tuning, and generation constraints, yielding text-grounded character profiles and literary worlds (Shao et al., 2023; Ran et al., 2025). Recent works further dynamically update character profiles and memories to simulate evolving character states across different story stages, effectively improving the faithfulness of role-playing (Xiao et al., 2026; Peng et al., 2026b).

However, fine-grained timestamp-aware behavior modeling cannot be adequately supported by these representations. Some prior works treat actions as embedded components of dialogue messages and thus mainly serve dialogue generation (Wang et al., 2025). Other existing lines of research predict characters’ next actions based on plot summaries or pre-constructed coarse-grained action sequences (Xu et al., 2025; Song et al., 2026). Consequently, literary-oriented RPLA still lacks character-centric atomic-level behavior trajectories. Meanwhile, existing methods generally organize character experiences following the narrative order of source texts, with little explicit distinction between the actual chronological order of events and their disclosure order in original literary works.

![](images/57de1fbd880f3b0e8a55133503346c6ef275bea49652f995f4fbd05ff03068c8.jpg)  
Figure 1: Overview of our behavior-modeling evaluation. Given character-related information, the model needs to select the most plausible next action from four options.

To address the above practical challenges, we propose a pipeline framework that automatically constructs character-behavior trajectories from full-length novels. Behaviors extracted by this framework include both verbal dialogues and non-dialogue actions, and the trajectories are organized strictly according to the actual chronological order of unfolding events, providing a unified data foundation for investigating character behaviors and their gradual evolution throughout the story progression. During dataset construction, we concurrently generate contextually plausible yet personamismatched counterfactual actions for each ground-truth behavior to support subsequent model training and evaluation. Finally, we build our curated dataset extracted from 143 literary works.

For model training, we select 115 works from the curated corpus and use the first 80% of each character’s behavior trajectory for training. As each source behavior is paired with a contextually plausible but persona-inconsistent counterfactual behavior, we use Regularized Preference Optimization (RPO), a joint preference-and-generation objective combining Direct Preference Optimization (DPO) with a negative log-likelihood loss on the source behavior. We treat the source behavior as chosen and one counterfactual behavior as rejected. These signals train the model to infer and generate actions consistent with the character’s current state and behavioral prior history in context.

To evaluate the model’s behavior-modeling capacity, we take the subsequent 20% of behaviors from each character trajectory within the aforementioned 115 works as temporal evaluation data. The remaining 28 unseen works excluded from training are entirely reserved for cross-book evaluation. For each evaluation instance, given the character profile, historical memory and current scene, the model is required to select the most plausible next action taken by the character among one ground-truth behavior and three distinct counterfactual behaviors. The former setup examines whether the model can continue observed character experiences, whereas the latter tests the model’s ability to predict behaviors based on structured character information without access to the source literary works.

The main contributions of this paper are summarized as follows:

• We propose ACTOR, an automated framework that extracts character-centered atomic behaviors from raw literary texts and organizes behavior trajectories according to their actual occurrence time in the story. ACTOR addresses the lack of character behavior data in existing resources.

• We formulate literary role-playing as an action-centered temporal modeling task: at a specified point in the story, the model predicts a character’s next behavior based on the character’s state and previous behavior history. This task extends literary role-playing beyond dialogue response generation to the unified modeling of dialogue and non-dialogue behaviors.

• We construct a next-behavior selection benchmark to objectively evaluate a model’s ability to imitate behavior based on character information, and use the behavior data generated by ACTOR to train models. Experiments on this benchmark show that training with these data improves a model’s ability to imitate the behavior of a specified character.

## 2 RELATED WORK

## 2.1 LITERARY CHARACTER ROLE-PLAYING GROUNDED IN TEXTUAL SOURCES

Early role-playing systems primarily ground language models in character descriptions or historical dialogue (Chen et al., 2024; Chen et al., 2025). Li et al. (2023) retrieve character-specific dialogue from scripts to better support response generation. Shao et al. (2023) reconstruct character experiences from biographical descriptions and train a dedicated agent for each character. Wang et al. (2024) extend role-conditioned instruction tuning to a broader range of diverse roles. Lu et al. (2024) generate and refine role-playing supervision through self-alignment.

Later studies construct richer character information from literary sources (Park et al., 2025; Kang et al., 2025). Wang et al. (2025) extract novel-sourced authentic conversations alongside scene context and character experiences, augmenting dialogue turns with associated actions and thoughts. Cheng et al. (2026) leverage fine-grained psychological attributes to organize character memory for improved long-term consistency. Du et al. (2026) reverse-synthesize character reasoning from literary dialogues and apply reinforcement learning to enhance character motivation modeling. Tang et al. (2026) structure book-derived memory from the character’s egocentric perspective to prevent god-view information leakage. These works enrich role-playing information, yet they are optimized for dialogue response generation rather than sequential behavior prediction.

## 2.2 TEMPORAL CHARACTER MODELING VIA ACTION TRAJECTORIES

Another line of research examines characters at specific points in a narrative (Ahn et al., 2024; Xu et al., 2025). Ahn et al. (2024) require models to respect chapter-specific knowledge boundaries and avoid referring to future events that remain unrevealed within the current narrative segment. Xu et al. (2025) reconstruct key decisions from a character’s prior experiences. Yuan et al. (2024) summarize character development from complete fictional works and evaluate whether the resulting profiles support character understanding. These studies show that character behavior depends not only on general personality, but also on experiences accumulated before the current story point.

Recent studies model character evolution based on plot progression and personal experiences (Yang et al., 2025b; Peng & Shang, 2025). Xiao et al. (2026) build temporally-ordered causal event graphs over novels for point-specific character profiles, while Zong et al. (2026) co-update character states and fictional worlds from dialogue and narrative text. Song et al. (2026) assess agent behavior across psychological phases of character development. Peng et al. (2026b) learn context-dependent behavioral rules from Fandom-sourced character-labeled action sequences and scene–action pairs, and Peng et al. (2026a) encode storylines as continuous actions with task-relevant history for next-action prediction. However, neither line of work extracts complete character action trajectories from novels nor reconstructs the genuine chronology of character experiences.

## 3 INTRODUCTION OF ACTOR

## 3.1 DATASET CONSTRUCTION

T-A-O behavior extraction We define character behavior as contextual utterances and distinctive non-dialogue actions in literary scenarios. Deliberate silence and avoidance are treated as valid behaviors, whereas routine actions that reflect no unique personal traits or active choices are filtered out. Each behavior follows a Thought–Action–Observation (T-A-O) structure: Thought describes the internal mental state and decision rationale before an action, Action corresponds to the verbal utterance or physical conduct, and Observation captures subsequent interpersonal reactions and environmental changes perceptible to the character. To efficiently manage long-text contexts, the extraction pipeline adopts a hierarchical granularity fallback, processing full plots first, followed by individual scenes and character-level contexts as a last resort.

![](images/3a00c760247f6ccddc0e7a577d29494eed923addaf2e24dcc9697f7954c19b41.jpg)  
Figure 2: ACTOR dataset construction pipeline. Plot-conditioned context is transformed into Thought–Action–Observation (T-A-O) units, followed by key-behavior validation, temporal and entity alignment, and dynamic profile updates for sustained changes. Each behavior is paired with three situation-plausible but target-character-inconsistent alternatives, which are refined to match the real action’s form and texture while leaving the real action unchanged.

Counterfactual alternatives For each behavior, the system generates alternative actions that are plausible in the current situation but inconsistent with the target character. So selecting the correct behavior requires genuine understanding of the character. These alternatives are generated together with the extracted behaviors and are refined after constructing character trajectories and profiles.

Key-behavior validation As shown in the validation module of Figure 2, we clearly distinguish ordinary and key behaviors. A key behavior must satisfy two core criteria: the character has reasonable alternative options at the moment, and the selected action alters persistent states or future situational possibilities. To reduce annotation noise from single-pass generation, the system first produces candidate key behaviors, identifies scenes in subsequent plots that reflect long-term behavioral impacts, and verifies the candidates through an independent evidence-checking procedure. Only evidence-supported candidates are retained as valid key behaviors.

Temporal alignment After behavior verification, we re-annotate the precise occurrence time of each scene using the original plot text. The textual reading order is used as the default timeline, while ordering anomalies potentially caused by flashbacks or reversed narration are detected and converted into preliminary adjustment proposals. Each proposal is independently checked against the source text, and only scene reorderings supported by sufficient textual evidence are applied. Character aliases and scattered mentions are then normalized to construct complete character-level behavioral trajectories in the corrected in-story chronological order.

Character data The resulting behavior trajectory of a character c is represented as:

$$
\begin{array} { r } { \mathcal { T } _ { c } = \left( b _ { 1 } , \dots , b _ { N _ { c } } \right) , \qquad b _ { i } = \left( s _ { i } , h _ { i } , a _ { i } , o _ { i } , r _ { i } , t _ { i } , \kappa _ { i } \right) , \quad t _ { 1 } \le \cdots \le t _ { N _ { c } } , } \end{array}\tag{1}
$$

where $s _ { i }$ denotes the scene context, $\left( { { h _ { i } } , { a _ { i } } , { o _ { i } } } \right)$ is the Thought–Action–Observation representation, $r _ { i }$ is the textual position, $t _ { i }$ is the reconstructed story-time position, and $\kappa _ { i } ~ \in ~ \{ 0 , 1 \}$ } indicates whether the behavior is a valid key behavior. The trajectory is ordered by story time, while the textual position is retained for source tracing.

Based on each trajectory, initial character profiles are derived from descriptions before the character’s first appearance. Each valid key behavior triggers a profile-update check over all experiences accumulated after the latest profile version. A new profile version is saved only when these experiences lead to sustained character changes.

Alternative refinement Finally, the initially generated counterfactuals are refined to match the length and structural form of real behaviors while remaining situationally reasonable and inconsistent with the target character. We then anonymize all datasets. Book titles and character names are withheld from the model during both training and evaluation to prevent leakage from the base model’s prior knowledge and improve generalization in character-action modeling.

## 3.2 ACTION PREFERENCE TRAINING

Based on the character behavioral trajectories, we formulate character action prediction as a conditional preference learning task. The input context x consists of four types of inputs before each target action: a dynamic profile of the current character state, long-term memory of validated key behaviors, short-term memory of recent scene interactions, and situational context containing background, temporal cues, and participant information. All inputs are strictly sampled from pre-action story content, ensuring the model only uses information accessible to the character at that moment.

In terms of training paradigm, our dataset is naturally suited to preference optimization. For the same prediction context, we possess both the character’s genuine behavior $\mathbf { y } ^ { + }$ extracted from the source text and counterfactual behaviors with matched length and format but inconsistent character profiles, which form an explicit relative preference relation: the genuine behavior should be preferred over actions the character would not take. Among the three counterfactual behaviors, the one closest in length to the genuine behavior is selected as $\mathbf { y } ^ { - }$ to minimize confounding effects from output-length differences. The source behavior and selected counterfactual are stored as chosen and rejected, respectively, defining the preference relation $\mathbf { y } ^ { + } \succ \mathbf { y } ^ { - }$

Direct Preference Optimization (DPO) learns this relation by optimizing the relative likelihood of the two behaviors against a fixed reference policy (Rafailov et al., 2023):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ^ { + } , \boldsymbol { y } ^ { - } ) \sim \mathcal { D } } \left[ \log \sigma \left( \beta \left( \log \frac { \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { + } \mid \boldsymbol { x } ) } { \pi _ { \mathrm { r e f } } ( \boldsymbol { y } ^ { + } \mid \boldsymbol { x } ) } - \log \frac { \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { - } \mid \boldsymbol { x } ) } { \pi _ { \mathrm { r e f } } ( \boldsymbol { y } ^ { - } \mid \boldsymbol { x } ) } \right) \right) \right] , } \end{array}\tag{2}
$$

where $\pi _ { \theta }$ is the policy being optimized, $\pi _ { \mathrm { r e f } }$ is the fixed reference policy, and $\beta$ controls the scaling of the preference objective.

However, this relative objective does not explicitly require the likelihood of the source behavior itself to increase. Consequently, it may only widen the preference margin during action selection, without increasing the probability of directly generating utterances or actions consistent with the target character. Following Liu et al. (2024), we therefore adopt RPO, which augments the DPO objective with a token-level negative log-likelihood loss on the source behavior:

$$
\mathcal { L } _ { \mathrm { N L L } } ( \theta ) = - \mathbb { E } _ { ( x , y ^ { + } ) \sim \mathcal { D } } \left[ \frac { 1 } { | y ^ { + } | } \sum _ { t = 1 } ^ { | y ^ { + } | } \log \pi _ { \theta } \left( y _ { t } ^ { + } \mid x , y _ { < t } ^ { + } \right) \right] .\tag{3}
$$

The final training objective is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R P O } } ( \theta ) = \mathcal { L } _ { \mathrm { D P O } } ( \theta ) + \alpha \mathcal { L } _ { \mathrm { N L L } } ( \theta ) , } \end{array}\tag{4}
$$

where $\alpha$ controls the contribution of direct supervision on the source behavior. We set $\beta = 0 . 1$ for all models and use $\alpha = 1 . 0$ for the Qwen-2.5 models and $\alpha = 0 . 2 5$ for Llama-3.1-8B. The resulting objective trains the model both to distinguish character-consistent behaviors from counterfactuals and to increase the likelihood of the source behavior itself.

## 4 EVALUATION

## 4.1 EVALUATION SETUP

Evaluated Models We evaluate six pretrained models as baselines: Qwen-2.5-Instruct 7B and 14B (Yang et al., 2025c), Llama-3.1-Instruct 8B (Grattafiori et al., 2024), Qwen-3-32B (Yang et al., 2025a), CoSER-8B (Wang et al., 2025), and HER-32B model (Du et al., 2026). These baselines cover general-purpose instruction-tuned large language models and existing specialized role-playing models for literary character simulation. We further evaluate three ACTOR models trained with RPO: ACTOR-Qwen-2.5-7B, ACTOR-Qwen-2.5-14B, and ACTOR-Llama-3.1-8B.

Table 1: Main results on the complete 17,560-instance evaluation set under the anonymous setting. All values are selection accuracy (%). Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td>Model</td><td>Overall</td><td>Temporal</td><td>Cross-work</td><td>Key</td><td>Non-key</td></tr><tr><td>Qwen-2.5-Instruct 7B</td><td>71.34</td><td>71.29</td><td>71.43</td><td>71.10</td><td>71.55</td></tr><tr><td>Qwen-2.5-Instruct 14B</td><td>76.96</td><td>77.04</td><td>76.79</td><td>76.85</td><td>77.05</td></tr><tr><td>Llama-3.1-Instruct 8B</td><td>69.15</td><td>68.45</td><td>70.59</td><td>69.97</td><td>68.43</td></tr><tr><td>Qwen-3-32B</td><td>78.30</td><td>77.97</td><td>78.96</td><td>79.14</td><td>77.56</td></tr><tr><td>CoSER-8B</td><td>57.41</td><td>56.89</td><td>58.46</td><td>60.56</td><td>54.63</td></tr><tr><td>HER-32B</td><td>65.65</td><td>65.02</td><td>66.93</td><td>67.24</td><td>64.25</td></tr><tr><td>ACTOR-Qwen-2.5-7B</td><td>80.20</td><td>80.23</td><td>80.14</td><td>80.21</td><td>80.19</td></tr><tr><td>ACTOR-Qwen-2.5-14B</td><td>86.00</td><td>86.16</td><td>85.68</td><td>86.40</td><td>85.65</td></tr><tr><td>ACTOR-Llama-3.1-8B</td><td>78.04</td><td>78.24</td><td>77.63</td><td>79.37</td><td>76.87</td></tr></table>

Evaluation Data The evaluation data are directly extracted from character behavior trajectories constructed from 143 literary works. For 115 works that contribute to training, the first 80% of each character trajectory is used to construct the training data, while the remaining 20% is reserved for temporal evaluation. The other 28 works are excluded from training and are used entirely for cross-work evaluation. The final evaluation set contains 17,560 test instances, including 11,770 temporal-evaluation instances and 5,790 cross-work instances.

Task and Metrics We formulate evaluation as a four-way next-behavior selection task. Given the character profile, behavioral memories, and current scene before the target behavior, the model must select the candidate behavior most likely to be performed by the target character. Each instance contains one source-text behavior and three counterfactual candidates. Selection accuracy is the primary metric. We additionally report accuracy on temporal and cross-work subsets, as well as on key and non-key behavior subsets, to examine model performance across different settings.

Open-ended Generation Evaluation Beyond next-behavior selection, we adopt the CoSER roleplaying evaluation framework (Wang et al., 2025) to examine open-ended character interaction. CoSER evaluates models through multi-turn scenarios and combines environment simulation, nextstate prediction, and language-model-based judging. It provides scores along four core dimensions: plot consistency, human-likeness, character fidelity, and plot quality. We use this evaluation to test whether ACTOR-derived training transfers from selecting character-consistent behaviors to generating coherent actions in dynamic interactive story settings.

## 4.2 MAIN RESULTS

Table 1 reports selection accuracy on the evaluation set under the anonymous setting.

All three ACTOR models outperform their corresponding backbones, with overall gains ranging from 8.86 to 9.04 percentage points. ACTOR-Qwen-2.5-14B achieves the highest overall accuracy of 86.00%, while ACTOR-Qwen-2.5-7B reaches 80.20% and surpasses the larger Qwen3-32B baseline. These results clearly show that ACTOR improves character-consistent behavior selection across diverse model families and sizes. Since all models are evaluated under the same strictly anonymous setting, the gains cannot be attributed to access to book titles or original character names. The comparison with Qwen3-32B further suggests that model scale alone does not substitute for targeted behavioral supervision tailored for literary character modeling.

The cross-work subset contains literary works excluded from training, making it a test of whether the learned behavior patterns transfer beyond familiar stories. ACTOR retains substantial gains on these unseen works: all three models improve over their backbones by 7.04–8.89 percentage points. The two Qwen-based models show similar gains on the temporal and cross-work subsets, whereas the Llama-based model has a smaller cross-work gain. These improvements are unlikely to arise solely from continuing trajectories seen during training. Instead, the learned behavioral supervision generalizes effectively to unseen characters and narrative scenarios.

Table 2: Representative cases from the evaluation results. Only decision-relevant evidence is shown.
<table><tr><td>Case</td><td>Relevant pre-action evidence</td><td>Qwen-2.5-7B- Instruct</td><td>ACTOR-Qwen-2.5- 7B / Source behavior</td></tr><tr><td>State trans- formation</td><td>Formerly cold and dismissive of poverty, the character later resolves to change and repeat- edly behaves with kindness and generosity. He only ap- pears severe when the clerk ar- rives late.</td><td>Punishes the clerk and refuses further support for his family.</td><td>Raises the clerk&#x27;s salary, supports his family, and asks him to add coal to the fire.</td></tr><tr><td>Dialogue continuity</td><td>The profile lists only the char- acter&#x27;s occupation. In the con- versation, she alternates be- tween teasing disclosures and assurances about another per- son&#x27;s visits.</td><td>Promises to disclose the information:“I&#x27;ll tell.&quot;</td><td>Reassures the inter- locutor that she will not disclose it: “Cer- tainly not.&quot;</td></tr><tr><td>Action adaptation</td><td>After declaring his willingness to kill an opponent and draw- ing his sword, the character is challenged by another charac- ter who intervenes.</td><td>Abandons the attack and shields the origi- nal target.</td><td>Redirects the attack toward the interven- ing character.</td></tr></table>

The improvements extend to both key and non-key behaviors. For example, ACTOR-Qwen-2.5- 7B gains 9.11 percentage points on key behaviors and 8.64 points on non-key behaviors; the other two ACTOR models show similarly balanced gains. ACTOR therefore improves the selection of ordinary persona-relevant behaviors as well as salient plot events. This matters for role-playing, where character consistency must also be maintained in less consequential, everyday interactions.

CoSER-8B and HER-32B obtain lower scores on this selection benchmark despite being developed for role-playing. Their results suggest that training for open-ended role-playing does not necessarily yield strong fine-grained action selection. The difference may reflect the objectives of these models and should not be taken as evidence of weaker role-playing ability overall. We therefore examine open-ended interaction separately using the CoSER framework.

## 4.3 QUALITATIVE CASE STUDY

To complement the quantitative results, we present three illustrative instances in which ACTOR-Qwen-2.5-7B selects the behavior consistent with the source text, whereas its backbone selects a counterfactual behavior. As shown in Table 2, these cases require different forms of behavioral reasoning: tracking a character’s long-term personality transformation, maintaining local dialogue coherence, and adapting an ongoing action to changes in the current scene. For readability, the contexts shown in the main text are manually summarized; more detailed versions of the three instances are provided in Tables 10–12 in Appendix C.

The first case concerns a character whose behavior has changed over time. The backbone selects a punitive action, which is superficially plausible given the character’s earlier reputation for coldness. However, the behavioral history shows he has recognized the consequences of his former conduct and repeatedly acted with kindness and generosity. His severe manner at the beginning of the current interaction is therefore performative rather than evidence of a return to his former state. ACTOR instead selects the source-consistent action of raising the clerk’s salary and supporting his family. Thus, ACTOR can prioritize a temporally updated character state over an earlier salient trait.

In the second case, the character profile provides little useful personality information beyond the character’s occupation. The decision must therefore be inferred primarily from the ongoing conversation, which alternates between direct answers, teasing disclosures, and assurances. The backbone predicts an eager promise to disclose the information, whereas ACTOR selects the brief assurance appearing in the source text. This example suggests that ACTOR can use the sequential organization of recent behaviors when profile cues are insufficient to determine the next action.

Table 3: Open-ended generation results under the CoSER evaluation framework.
<table><tr><td>Model</td><td>Plot Consistency</td><td>Human-likeness</td><td>Character Fidelity</td><td>Plot Quality</td><td>Average</td></tr><tr><td>CoSER-8B</td><td>16.07</td><td>6.96</td><td>30.83</td><td>25.31</td><td>19.79</td></tr><tr><td>Qwen-2.5-Instruct 7B</td><td>11.70</td><td>1.17</td><td>21.87</td><td>6.97</td><td>10.43</td></tr><tr><td>ACTOR-Qwen-2.5-7B</td><td>22.57</td><td>3.88</td><td>41.12</td><td>20.40</td><td>21.99</td></tr><tr><td>Qwen-2.5-Instruct 14B ACTOR-Qwen-2.5-14B</td><td>25.30</td><td>3.90</td><td>42.95</td><td>26.60</td><td>24.69</td></tr><tr><td></td><td>25.30</td><td>15.00</td><td>38.50</td><td>34.75</td><td>28.39</td></tr><tr><td>Llama-3.1-Instruct 8B</td><td>6.35</td><td>2.80</td><td>16.40</td><td>3.85</td><td>7.35</td></tr><tr><td>ACTOR-Llama-3.1-8B</td><td>16.75</td><td>7.65</td><td>21.10</td><td>23.80</td><td>17.33</td></tr></table>

The third case highlights ACTOR’s ability to maintain action continuity after a change in the scene. The character has already drawn his sword against an opponent when another character intervenes. ACTOR redirects the attack toward the intervening character, preserving the character’s existing intent while responding to the new situation. In contrast, the backbone incorrectly protects the original target, which conflicts with both the preceding attack and the character’s stated objective. The contrast indicates that effective prediction requires preserving the character’s initial intent while updating the immediate action in response to the intervention.

## 4.4 OPEN-ENDED GENERATION RESULTS

To verify whether ACTOR can generalize beyond candidate behavior selection, we test the models on open-ended multi-turn character interaction using the CoSER evaluation framework. GPT-5.6- Luna serves as both helper and judge model across all experiments. Since our experimental setup differs from the original CoSER work, we compare models under a unified evaluation protocol instead of directly matching our absolute scores against those reported in the original paper.

CoSER is constructed from multi-character dialogues extracted from authentic literary works, where each message integrates speech, physical actions, and inner thoughts. It formulates characterspecific supervised training by taking scene context and other characters’ utterances as input to predict the target character’s responses. Additionally, it optimizes environment modeling and nextspeaker prediction to facilitate multi-agent simulation. Such data structures and training objectives fundamentally differ from ours, yet our models still achieve performance gains.

As shown in Table 3, all three ACTOR models achieve higher overall scores than their corresponding backbone models, with particularly clear improvements for ACTOR-Qwen-2.5-7B and ACTOR-Llama-3.1-8B. The improvements are not uniform across individual dimensions because ACTOR is not trained on CoSER-specific interaction procedures or output formats. Nevertheless, the consis tent improvement in overall performance suggests that, even without explicit training on CoSER data, preference learning over temporally structured character trajectories can enable models to exhibit features associated with character dialogue style in open-ended interaction. These results also demonstrate ACTOR’s broader application potential. Details are shown in appendix D.

## 4.5 ABLATION STUDY

Figure 3 reports the context ablation under the anonymous setting. We keep the Current Scene unchanged and separately remove the Character Profile, Recent Memory, or Long-term Memory from the input context. We find that removing either the Character Profile or Recent Memory consistently reduces the performance of all three ACTOR models. This indicates stable character traits and recent behavioral context provide complementary signals for global personality reasoning and local behavioral adaptation. In contrast, removing Long-term Memory changes the accuracy of the two Qwen models by no more than 0.5 percentage points and slightly improves the Llama model. This difference may reflect the partial overlap between Long-term Memory and the Character Profile, making the marginal contribution of long-term information dependent on the backbone.

![](images/c016dad65c6cd1d05f52dd88464607157f4672dab5463c8dcf59709fd1849986.jpg)  
ACTOR-Qwen-7B

![](images/867c4b322a3c1cc2a6e2b45199d2013421746ac59f34a5fd835c6c15093fefbb.jpg)  
ACTOR-Qwen-14B

![](images/4ed164a8032147dc97893d119862ddafee7b354925b7329db6a1ed0b41723895.jpg)  
ACTOR-Llama-8B

Figure 3: Accuracy changes after removing different context components. Each panel corresponds to one ACTOR model, and the vertical reference line denotes the full-context performance. Positive values indicate performance drops after removing the component.  
![](images/1c3ce7ce9e0a3dc11e2f8867f0ff372c26d958068b798b0468759012e6598123.jpg)  
Figure 4: Selection accuracy under the anonymous and identity-provided settings.

Figure 4 compares model performance under anonymous and identity-provided settings. Supplying book titles and character names boosts accuracy for all models, as explicit identities help retrieve prior knowledge of familiar characters. ACTOR’s gains (0.51–1.82 pp) are smaller than those of baselines. ACTOR’s advantage therefore does not primarily depend on explicit character identities. Its strong anonymous-setting performance shows the model leverages provided character information and scene context, rather than memorized knowledge of known books or characters. This benefit is valuable for role-playing original or unseen characters.

## 5 CONCLUSION

We presented ACTOR, an automated framework for extracting and chronologically organizing dialogue and persona-relevant non-dialogue behaviors from literary texts. By pairing source behaviors with counterfactual alternatives, ACTOR formulates literary role-playing as an action-centered temporal prediction task and trains models with an RPO objective that combines preference learning with direct supervision on source behaviors. Experiments show that ACTOR consistently improves next-behavior prediction across different settings, with gains on unseen works indicating transfer beyond memorizing training trajectories. Ablation and qualitative analyses further highlight the importance of updated character profiles and recent behavioral memory. Supplementary evaluation with the CoSER evaluation framework suggests that these benefits extend to open-ended character interaction, although the framework may inherit noise from automatic data construction and currently relies on separate selection and generation protocols. Future work will improve construction reliability, broaden language and genre coverage, and develop unified evaluations.

## AI USE STATEMENT

We used generative AI tools to assist with extracting and cleaning character behavior data from lit erary texts, including behavior structuring, candidate key-behavior identification, temporal information organization, and counterfactual behavior generation. Generative AI was also used to assist with drafting and polishing the manuscript and with generating and revising figures. All AI-generated data were subject to programmatic validation, source-based evidence verification, and manual inspection. The authors also reviewed all AI-assisted text and figures. Generative AI was not used to replace the authors in formulating the core research questions, drawing theoretical conclusions, or interpreting experimental results. The authors take full responsibility for the final content, research claims, and all artifacts produced with the assistance of generative AI.

## ETHICS STATEMENT

This work involves no human participants, private personal data, or direct interaction with real individuals, so institutional review board approval is unnecessary. We construct character behavior trajectories from literary texts. To comply with copyright and licensing constraints, our released resource contains only derived anonymized records. Full original novels are not redistributed.

Automatically generated behaviors and counterfactuals may carry biases inherited from source literature. Biases can also be introduced by language models. Such biases may produce stereotyped portrayals. These portrayals touch on gender, culture, and social roles. For this reason, the resulting models cannot serve as reliable depictions of real people. They must not be deployed for high-stakes decision-making. We mitigate annotation errors through source-based validation alongside structural auditing. Manual inspection further supports this process. We take responsibility for ethical usage and thorough documentation of all released materials.

## REPRODUCIBILITY STATEMENT

The ACTOR data-construction pipeline is described in Section 3.1. The RPO training objective and its model-specific settings are provided in Section 3.2. The data splits, evaluated models, task formulation, and open-ended evaluation protocol are specified in Section 4.1. The appendix further provides dataset statistics and structural audits (Appendix A), the complete dataset schema (Appendix B) and the prompts used (Appendix E). Code and data are available at https://github.com/SilentStar233/ACTOR.

## REFERENCES

Jaewoo Ahn, Taehyun Lee, Junyoung Lim, Jin-Hwa Kim, Sangdoo Yun, Hwaran Lee, and Gunhee Kim. TimeChara: Evaluating point-in-time character hallucination of role-playing large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 3291–3325, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.197. URL https://aclanthology.org/2024.findings-acl.197/.

Jiangjie Chen, Xintao Wang, Rui Xu, Siyu Yuan, Yikai Zhang, Wei Shi, Jian Xie, Shuang Li, Ruihan Yang, Tinghui Zhu, Aili Chen, Nianqi Li, Lida Chen, Caiyu Hu, Siye Wu, Scott Ren, Ziquan Fu, and Yanghua Xiao. From persona to personalization: A survey on role-playing language agents. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https: //openreview.net/forum?id=xrO70E8UIZ. Survey Certification.

Nuo Chen, Yan Wang, Yang Deng, and Jia Li. The oscars of AI theater: A survey on role-playing with language models, 2025. URL https://arxiv.org/abs/2407.11484.

Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, and Yuan Zhang. PsyMem: Fine-grained psychological alignment and explicit memory control for advanced roleplaying LLMs. Transactions ofthe Associationfor Computational Linguistics, 14:510–529, 2026. doi: 10.1162/tacl.a.645. URL https://aclanthology.org/2026.tacl-1.24/.

Chengyu Du, Xintao Wang, Aili Chen, Weiyuan Li, Rui Xu, Junteng Liu, Zishan Huang, Rong Tian, Zijun Sun, Yuhao Li, Liheng Feng, Deming Ding, Pengyu Zhao, and Yanghua Xiao. HER: Human-like reasoning and reinforcement learning for LLM role-playing. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 25725–25762, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395- 1. doi: 10.18653/v1/2026.findings-acl.1283. URL https://aclanthology.org/2026. findings-acl.1283/.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Sua Kang, Chaelim Lee, Subin Jung, and Minsu Lee. Fic2bot: A scalable framework for personadriven chatbot generation from fiction. Electronics, 14(19), 2025. ISSN 2079-9292. doi: 10. 3390/electronics14193859. URL https://www.mdpi.com/2079-9292/14/19/3859.

Cheng Li, Ziang Leng, Chenxi Yan, Junyi Shen, Hao Wang, Weishi MI, Yaying Fei, Xiaoyang Feng, Song Yan, HaoSheng Wang, Linkang Zhan, Yaokai Jia, Pingyu Wu, and Haozhen Sun. Chatharuhi: Reviving anime character in reality via large language model, 2023. URL https: //arxiv.org/abs/2308.09597.

Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in RLHF: Your SFT loss is implicitly an adversarial regularizer. Advances in Neural Information Processing Systems, 37:138663–138697, 2024.

Keming Lu, Bowen Yu, Chang Zhou, and Jingren Zhou. Large language models are superpositions of all characters: Attaining arbitrary role-play via self-alignment. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7828–7840, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.423. URL https://aclanthology.org/2024.acl-long.423/.

Jeiyoon Park, Chanjun Park, and Heuiseok Lim. CharacterGPT: A persona reconstruction framework for role-playing agents. In Weizhu Chen, Yi Yang, Mohammad Kachuee, and Xue-Yong Fu (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track), pp. 287–303, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-194-0. doi: 10.18653/v1/2025.naacl-industry.24. URL https://aclanthology.org/2025.naacl-industry.24/.

Letian Peng and Jingbo Shang. Codifying character logic in role-playing. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=RoVS9vmpd2.

Letian Peng, Ziche Liu, Yiming Huang, Longfei Yun, Kun Zhou, Yupeng Hou, and Jingbo Shang. Bookmarks: Efficient active storyline memory for role-playing, 2026a. URL https://arxiv. org/abs/2605.14169.

Letian Peng, Kun Zhou, Longfei Yun, Yupeng Hou, and Jingbo Shang. Deriving character logic from storyline as codified decision trees. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 12480–12509, San Diego, California, United States, July 2026b. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/ 2026.acl-long.568. URL https://aclanthology.org/2026.acl-long.568/.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Yiting Ran, Xintao Wang, Tian Qiu, Jiaqing Liang, Yanghua Xiao, and Deqing Yang. BOOK-WORLD: From novels to interactive agent societies for story creation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd An nual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15898–15912, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.773. URL https://aclanthology. org/2025.acl-long.773/.

Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu. Character-LLM: A trainable agent for roleplaying. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13153–13187, Singapore, December 2023. Association for Computational Linguistics. URL https://aclanthology. org/2023.emnlp-main.814/.

Woojung Song, Nalim Kim, Sangjun Song, Chaewon Heo, Jongwon Lim, and Yohan Jo. Arcane: Do role-playing language agents stay in character at the right time?, 2026. URL https:// arxiv.org/abs/2606.05553.

Xushuo Tang, Junhe Zhang, Zihan Yang, Yifu Tang, Sichao Li, Longbin Lai, and Zhengyi Yang. Staying in character: Perspective-bounded memory for book-based role-playing agents, 2026. URL https://arxiv.org/abs/2606.25632.

Noah Wang, Z.Y. Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang, Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao Huang, Jie Fu, and Junran Peng. RoleLLM: Benchmarking, eliciting, and enhancing roleplaying abilities of large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 14743–14777, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.findings-acl.878. URL https://aclanthology.org/2024.findings-acl. 878/.

Xintao Wang, Heng Wang, Yifei Zhang, Xinfeng Yuan, Rui Xu, Jen tse Huang, Siyu Yuan, Haoran Guo, Jiangjie Chen, Shuchang Zhou, Wei Wang, and Yanghua Xiao. CoSER: Coordinating LLM-based persona simulation of established roles. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=BOrR7YqKUt.

Ye Wang, Jiaxing Chen, and Hongjiang Xiao. Role-playing agents driven by large language models: Current status, challenges, and future trends, 2026. URL https://arxiv.org/abs/2601. 10122.

Zhihao Xiao, Mengting Li, Xintao Wang, Linfeng Li, Limin Shui, Mengqi Ji, and Borui Cai. Dream: LLM-based Dynamic Role-playing via Event-Aware Memory Graph. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’26, pp. 5686–5697, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400722592. doi: 10.1145/3770855.3818027. URL https://doi.org/10.1145/ 3770855.3818027.

Rui Xu, Xintao Wang, Jiangjie Chen, Siyu Yuan, Xinfeng Yuan, Jiaqing Liang, Zulong Chen, Xiaoqingdong, and Yanghua Xiao. Character is destiny: Can persona-assigned language models make personal choices? In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 15038–15059, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.813. URL https: //aclanthology.org/2025.findings-emnlp.813/.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025a.

Table 4: Summary of the dataset.
<table><tr><td>Split</td><td>Works</td><td>Records</td><td>Characters</td><td>Scenes</td><td>Key</td><td>Non-key</td></tr><tr><td>Training</td><td>115</td><td>270,856</td><td>3,939</td><td>30,818</td><td>26,374</td><td>244,482</td></tr><tr><td>Evaluation</td><td>143</td><td>17,560</td><td>2,437</td><td>11,201</td><td>8,221</td><td>9,339</td></tr><tr><td>Temporal / cross-work</td><td>115 /28</td><td>17,560</td><td></td><td></td><td></td><td></td></tr></table>

Bohao Yang, Hainiu Xu, Jinhua Du, Ze Li, Yulan He, and Chenghua Lin. Evolvtrip: Enhancing literary character understanding with temporal theory-of-mind graphs, 2025b. URL https: //arxiv.org/abs/2506.13641.

Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025c. URL https://arxiv.org/abs/2412.15115.

Xinfeng Yuan, Siyu Yuan, Yuhan Cui, Tianhe Lin, Xintao Wang, Rui Xu, Jiangjie Chen, and Deqing Yang. Evaluating character understanding of large language models via character profiling from fictional works. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8015–8036, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.456. URL https://aclanthology.org/2024. emnlp-main.456/.

Qing Zong, Yue Guo, Mengxin Yang, Yiwen Guo, and Yangqiu Song. Evolvingworld: An openschema framework for co-evolving role-play agents and world model in interactive literary world, 2026. URL https://arxiv.org/abs/2607.17250.

## A DATASET STATISTICS

The final anonymized corpus contains behavior records from 143 non-empty literary works. The training partition contains 270,856 source preference records from 115 works. The evaluation partition contains 17,560 four-way next-behavior questions from 143 works: 11,770 temporal-evaluation instances from the 115 training works and 5,790 cross-work instances from 28 additional works. The evaluation set contains 8,221 key-behavior instances and 9,339 non-key-behavior instances. Table 4 summarizes the corpus composition.

The context supplied to each instance is compact but structured. In the evaluation set, the median character trajectory contains 6 recorded behaviors, the median profile contains 120 whitespacedelimited words, and the median recent and long-term memories contain 8 and 7 entries, respectively. Training records have longer trajectories because they include the complete training prefix; their median trajectory length is 15 behaviors. Profile freshness is reported per record and indicates whether the profile was immediately updated, moderately distant, or stale relative to the target behavior. Trajectory-length distributions are shown in Figures 5 and 6.

We also audit the structural validity of the released records. The rebuilding validation retained 288,416 records and excluded 279 records before release. All 17,560 evaluation instances contain exactly four options and a valid answer, while all 270,856 training records contain non-empty chosen and rejected behaviors. No duplicate record identifiers or malformed placeholder strings were detected. A small number of participant-description keys are composite or relationship-style expressions rather than single “Character n” placeholders; they are reported separately for manual review and are not counted as known-alias or malformed-placeholder failures. The complete audit is given in Table 5.

Finally, source chosen and rejected behaviors are closely matched in surface length: their medians are both 13 whitespace-delimited words, with means of 21.69 and 21.30 words, respectively. The corresponding ranges and percentiles are reported in Table 6; the distributions are visualized in Figures 7 and 8.

Table 5: Evaluation-question and preference-pair audit.
<table><tr><td>Audit</td><td>Training</td><td>Evaluation</td><td>Result</td></tr><tr><td>Records with required answer fields</td><td>270,856</td><td>17,560</td><td>valid</td></tr><tr><td>Invalid preference/options records</td><td>0</td><td>0</td><td>none</td></tr><tr><td>Duplicate record IDs</td><td>0</td><td>0</td><td>none</td></tr><tr><td>Malformed placeholders</td><td>0</td><td>0</td><td>none</td></tr><tr><td>Non-placeholder participant-key occurrences</td><td>764</td><td>65</td><td>manual review</td></tr><tr><td>Four-option questions</td><td></td><td>17,560</td><td>all evaluation records</td></tr></table>

Table 6: Length statistics for source preference pairs, measured in whitespace-delimited words.
<table><tr><td>Response</td><td>Count</td><td>Mean</td><td>Median</td><td>P10</td><td>P90</td><td>Min-Max</td></tr><tr><td>Chosen</td><td>270,856</td><td>21.6877</td><td>13</td><td>6</td><td>42</td><td>1-3635</td></tr><tr><td>Rejected</td><td>270,856</td><td>21.3041</td><td>13</td><td>5</td><td>41</td><td>2-3297</td></tr></table>

![](images/a44c60629561bd5abf5a4440d5d854766ff6f0228b5a72aec4b12ca995026d10.jpg)  
Figure 5: Evaluation trajectory-length distribution.

## B DATASET SCHEMA

Each ACTOR instance is centered on a target character and a behavior occurring at a particular point in the story. All instances contain the character profile, behavioral memories, and current-scene context available before the target behavior. Training instances additionally contain a preferred source behavior and a rejected counterfactual, whereas evaluation instances contain four candidate behaviors and the corresponding answer label. All textual fields are anonymized before being provided to the model.

Table 9 presents an actual anonymized evaluation instance from the released dataset. The textual fields and candidate options are preserved as stored in the dataset; only internal bookkeeping fields are omitted from the table.

For training instances, the same contextual fields are paired with chosen and rejected behaviors instead of four candidate options. This shared schema enables the extracted trajectories to support both preference-based training and objective next-behavior evaluation.

![](images/2e6b00f1a67c3bf6c9a5f8a1432bf7f4faa3154534ad16850cf61412f845205e.jpg)

Figure 6: Training trajectory-length distribution.  
![](images/45e5764b614e4d6c1db633fe9ab6a81b6021b3437b26c0b8b771f91882cdaece.jpg)

Figure 7: Length distribution of chosen behaviors.  
![](images/64e3e9a7629570106a841c1805d3e8c55b2418acbb92f4dd8d1e583a323d7622.jpg)  
Figure 8: Length distribution of rejected behaviors.

Table 7: Context fields shared by ACTOR training and evaluation instances.
<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>character</td><td>Anonymized la- bel</td><td>Anonymized name of the target character shown to the model.</td></tr><tr><td>profile</td><td>Structured ob- ject</td><td>The latest character-profile version available before the target behavior. It summarizes persistent traits, relationships, and established states.</td></tr><tr><td>long_term_memory</td><td>List of T-A-O records</td><td>Previously verified key behaviors that provide long- range behavioral history.</td></tr><tr><td>recent_memory</td><td>List of T-A-O records</td><td>Behaviors from the most recent preceding scenes, representing the local behavioral context.</td></tr><tr><td>scene</td><td>Structured ob- ject</td><td>The current-scene description, temporal informa- tion, present participants, and anonymized partic- ipant descriptions. It excludes the target behavior and all subsequent events.</td></tr><tr><td>is_critical</td><td>Boolean</td><td>Indicates whether the target behavior is a verified key behavior.</td></tr><tr><td>action_uid, scene_uid</td><td>Identifiers</td><td>Internal identifiers for the target behavior and its containing scene. They are used for traceability and are not required as model inputs.</td></tr></table>

Table 8: Split-specific fields in ACTOR training and evaluation instances.
<table><tr><td>Field</td><td>Split</td><td>Description</td></tr><tr><td>chosen</td><td>Training</td><td>The behavior performed by the target character in the source text. It is treated as the preferred response during RPO training.</td></tr><tr><td>rejected</td><td>Training</td><td>A contextually plausible but character-inconsistent counterfactual behavior paired with the source be- havior.</td></tr><tr><td>options</td><td>Evaluation</td><td>Four candidate next behaviors, consisting of one source behavior and three counterfactual behaviors.</td></tr><tr><td>correct_answer</td><td>Evaluation</td><td>The identifier of the candidate corresponding to the source behavior.</td></tr><tr><td>character_action_order Training /Evalu-</td><td>ation</td><td>The position of the target behavior in the character- level chronological trajectory.</td></tr><tr><td>split</td><td>Training</td><td>Partition label indicating that the instance belongs to the training prefix of a character trajectory.</td></tr><tr><td>sample_id, question_id</td><td>1 Training / Evalu- ation</td><td>Unique identifiers used to trace each instance during construction, validation, and evaluation.</td></tr></table>

## C DETAILED CASE-STUDY INSTANCES

Tables 10–12 provide additional details for the three cases discussed in Section 4.3. To maintain readability, we retain the evidence most relevant to each behavioral decision and use ellipses to indicate omitted profile, memory, or scene content.

Table 9: An actual anonymized next-behavior selection instance from the ACTOR evaluation set.
<table><tr><td>Component</td><td>Dataset content</td></tr><tr><td>Target character</td><td>Character 1</td></tr><tr><td>Character profile</td><td>Character 1 is a young law student and the heir presumptive to Char- acter 2, positioned by expectation as the future baronet. Beyond this social and familial standing, no stable temperament, values, interac- tion style, personal goals, or relationship orientation are evidenced before his first recorded action.</td></tr><tr><td>Recent memory</td><td>Thought: I do not feel eager to be drawn into the family&#x27;s notice. Action: Receives Character 2 overtures without warmth. Observation: Character 2 persists, making allowance for what he treats as youthful modesty.</td></tr><tr><td>Long-term memory</td><td>Thought: I do not feel eager to be drawn into the family&#x27;s notice. Action: Receives Character 2 overtures without warmth. Observation: Character 2 persists, making allowance for what he treats as youthful modesty.</td></tr><tr><td>Current scene</td><td>Time: in one of their spring excursions to London, when Character 3 was in her first bloom. Present: Character 3, Character 2, Character 1. The scene has just begun. Character 3 is the eldest daughter of Char- acter 2 and Character 4; Character 2 is a baronet and the father of Character 3, Character 5, and Character 6; Character 1 is the heir pre- sumptive to Character 2 and a young law student expected to become</td></tr><tr><td>Candidate A</td><td>the future baronet. Behaves agreeably during the introduction.</td></tr><tr><td>Candidate B</td><td>Regards Character 3 with open disdain throughout.</td></tr><tr><td>Candidate C</td><td>Defers slavishly to Character 2 guidance.</td></tr><tr><td>Candidate D</td><td>Presses Character 3 hand with ardent presumption.</td></tr><tr><td>Correct answer</td><td>A</td></tr></table>

## D COSER INTERACTION EXAMPLES

Tables 13 and 14 present two qualitative comparisons from the CoSER evaluation. Each model is evaluated in a separate simulation of the same initial scene. Context and character descriptions are summarized from the evaluation inputs. We select excerpts from the first eight character messages in each simulation, omitting next-speaker predictions and environment messages. Turn numbers preserve their original order; gaps indicate omitted turns, and ellipses mark omissions within a turn. Speaker attributions follow the recorded role fields, including inconsistencies; names are shortened in the first table and redundant name prefixes are omitted. Scores are scene-level averages of the four CoSER dimensions, computed from the full simulations rather than these excerpts. These selected cases illustrate interaction patterns and limitations.

In Table 13, the Qwen-2.5-Instruct 7B backbone initially attends to the dome model, but Tommaso’s next turn addresses himself by name. The interaction then settles into declarations of shared purpose, and both characters repeat almost the same handshake description instead of discussing the design. CoSER-8B avoids that repetition and raises a question about the dome’s shape, although the exchange soon turns to general reassurance. ACTOR-Qwen-2.5-7B sustains the subject more clearly: Tommaso asks about a flying buttress, Michelangelo responds in terms of support and visibility, and later turns discuss what the dome symbolizes. The useful contrast is whether the conversation develops beyond the initial scene cue. The backbone recognizes that cue but does not maintain a distinct or progressing exchange; ACTOR does so in these selected turns. This reading concerns conversational progression, not the architectural or historical correctness of the design claims.

Table 10: Detailed instance for Case 1, which requires tracking a long-term character transformation. Ellipses indicate omitted context.
<table><tr><td>Field</td><td>Content</td></tr><tr><td>Profile</td><td>Character 1 was formerly cold, solitary, suspicious, anti-Christmas, and dis- missive of poverty. After witnessing the consequences of his former life, he becomes grateful, humble, playful, openly generous, and eager to repair his past hardness. ...</td></tr><tr><td>Long-term memory</td><td>Thought: I now see that this neglected and plundered dead man&#x27;s case may be my own if my present life continues in this direction. Action: He recognizes that his own life is tending toward the same unloved death. Observation: .. . Thought: I must bind myself to a changed life and preserve the lessons I have learned. Action: He vows to honor Christmas throughout the year and to live by the lessons of the past, present, and future. Observation: . ..</td></tr><tr><td>Recent memory</td><td>He goes to church, walks through the streets with pleasure, treats children kindly, and questions beggars rather than ignoring them. He then approaches his nephew&#x27;s house, overcomes his anxiety, and joins the family gathering. . . .</td></tr><tr><td>Current scene</td><td>The character waits for his clerk, who arrives eighteen and a half minutes late. He initially challenges the clerk in a severe tone and pretends that he will no longer tolerate such conduct. His severity is described as feigned and suddenly playful, although the clerk remains alarmed.</td></tr><tr><td>Candidates</td><td>A: Punishes the clerk for arriving late and refuses to provide additional warmth or assistance. B: Offers only conditional assistance after scrutinizing the clerk&#x27;s family and conduct. C: Wishes the clerk a merry Christmas, raises his salary, promises to assist his family, and asks him to add coal to the fire.</td></tr><tr><td>Predictions</td><td>D: Offers help primarily as a public demonstration of his own generosity. Source behavior: C; Qwen-2.5-Instruct 7B: A; ACTOR-Qwen-2.5-7B: C.</td></tr></table>

Table 11: Detailed instance for Case 2, which requires maintaining local dialogue continuity. Ellipses indicate omitted context.
<table><tr><td>Field</td><td>Content</td></tr><tr><td>Profile</td><td>Character 1 is identified only as a chambermaid responsible for room care. No personal traits, social relationships, motives, history, or attitudes are available in the profile.</td></tr><tr><td>Long-term memory</td><td>The chambermaid has previously revealed that Character 3 visited nearly ev- ery day and that the visits were made to Character 4. She further reveals that Character 4 and Character 3 sometimes went out together. .. .</td></tr><tr><td>Recent memory</td><td>When asked about Character 4, the chambermaid answers directly that Char- acter 4 went out only a few minutes earlier. Character 2 finds the news strange and hurriedly searches through his belongings.</td></tr><tr><td>Current scene</td><td>During an extended conversation, the chambermaid alternates between direct answers, teasing remarks, and evasive responses. She discusses Character 3&#x27;s visits, tells Character 2 that the visits occurred nearly every day, and reveals that Character 3 and Character 4 sometimes went out together. Character 2 becomes visibly disturbed and questions her about the details. .. .</td></tr><tr><td>Candidates</td><td>A: promises: &quot;Certainly not.&quot; B: greedily hints: &quot;Pay me.&quot; C: haughtily says: &quot;Never, sir.&quot; D: eagerly says: &quot;I&#x27;ll tell.&quot;</td></tr><tr><td>Predictions</td><td>Source behavior: A; Qwen-2.5-Instruct 7B: D; ACTOR-Qwen-2.5-7B: A.</td></tr></table>

Table 14 shows a different failure in the Qwen-2.5-Instruct 7B backbone. It introduces Ellen’s concern and the tray of food, but the turn labelled Heathcliff adopts the caregiver’s perspective and then repeats Ellen’s appeal to eat almost verbatim. Later turns dwell on the wind and the room rather than developing a response to his refusal. CoSER-8B maintains a reciprocal dialogue, yet shifts from the supplied conflict toward a reassuring exchange. ACTOR-Qwen-2.5-7B keeps the offered food, refusal to eat, and withdrawal in view, but turns 3 and 5 are labelled Ellen Dean while voicing Heathcliff’s perspective. Stronger engagement with the scene therefore does not resolve speaker attribution. Although ACTOR has the highest average score here, its character-fidelity score is lower than the backbone’s (42.50 versus 62.50); the scene average should not be interpreted as improvement on every dimension.

Table 12: Detailed instance for Case 3, which requires adapting an ongoing action to a change in the current scene. Ellipses indicate omitted context.
<table><tr><td>Field</td><td>Content</td></tr><tr><td>Profile</td><td>Character 1 is a steward and trusted envoy whose loyalty lies primarily with his mistress and her political faction. Although normally self-protective under direct threat, he is willing to use violence against Character 5 to demonstrate his factional loyalty. ...</td></tr><tr><td>Long-term memory</td><td>Thought: If I meet Character 5, killing him will prove my allegiance to the party I serve. Action: He declares that, if he encounters Character 5, he will show which party he follows. Observation: . . Thought: Here is the proclaimed traitor I hoped to meet; killing him can raise my fortunes. Action: He draws his sword and threatens to kill Character 5. Observation: Character 10 intervenes between them.</td></tr><tr><td>Recent memory</td><td>The character repeatedly follows his mistress&#x27;s instructions, refuses to disclose her confidential letter, and prioritizes her orders despite warnings that the jour- ney is dangerous. .. .</td></tr><tr><td>Current scene</td><td>Character 1 prepares to attack Character 5. Character 10 steps between them and refuses to move. Character 1 threatens Character 10 and orders him to stand aside, but Character 10 remains defiant and prepares to resist the attack.</td></tr><tr><td>Candidates</td><td>A: Kneels before Character 10. B: Casts away his sword. C: Fights Character 10.</td></tr><tr><td>Predictions</td><td>D: Shields Character 5, whom he had intended to kill. Source behavior: C; Qwen-2.5-Instruct 7B: D; ACTOR-Qwen-2.5-7B: C.</td></tr></table>

Together, the examples suggest that ACTOR’s behavioral supervision can support more sustained exchanges and closer engagement with a scene’s conflict in open-ended interaction. They also expose a remaining failure mode: maintaining the correct speaker and perspective across turns. The examples complement the aggregate CoSER results without establishing how often either pattern occurs.

## E PROMPT

To facilitate reproducibility, this section lists the prompts used throughout the ACTOR model. Tables 15–17 present the prompts adopted in the action trajectory extraction stage. Tables 18–19 are for dynamic character-profile construction and counterfactual rewriting. Tables 20–21 provide the shared context and output prompts used for RPO training and four-way next-behavior evaluation.

## Table 13: Conversational progression in The Agony and the Ecstasy (scene 0006). The scene context is summarized and model outputs are excerpted from the recorded simulations.

Scene. Michelangelo and Tommaso examine a wooden model of St. Peter’s dome. After years of design work, Michelangelo is conscious of his declining health and wants his vision to survive him.

Characters and motivations. Michelangelo is independent and devoted to his craft; he wants Tommaso to understand the design. Tommaso admires his work, worries about his health, and wants to support him.

## Qwen-2.5-Instruct 7B

Scene score: 26.75

1. Michelangelo: . . . I stand before the model of the dome, my hands tracing the lines with a steady hand. . . .

2. Tommaso: . . . (Tommaso, you’re my inspiration. Let’s do this together.)

3. Michelangelo: . . . Together, we shall see this through. (We shake hands, a silent pact between us.)

4. Tommaso: (Our hands meet, a bond of friendship and shared purpose.) . . .

5. Michelangelo: (Our hands meet, a bond of friendship and shared purpose.) (I return his smile, a flicker of hope lighting my eyes.) . . .

6. Tommaso: (Our hands meet, a bond of friendship and shared purpose.) (I return his smile, a flicker of hope lighting my eyes.) . . .

## CoSER-8B

Scene score: 19.50

3. Michelangelo: . . . What do you think of making it more concave?

4. Tommaso: . . . What inspired this change?

5. Michelangelo: . . . I just feel that it needs more... movement. But I fear I may be losing my perspective.

6. Tommaso: . . . Your vision is unparalleled. Trust in your instincts. . . .

8. Tommaso: . . . Your work will outlive us all. How can I assist you in bringing this vision to life?

## ACTOR-Qwen-2.5-7B

Scene score: 46.00

1. Michelangelo: . . . ”Tommaso, I have a new idea for the dome,” . . .

2. Tommaso: . . . ”But why not use a flying buttress?” . .

3. Michelangelo: . . . ”It would support the weight, but it would also be visible. I want the dome to be a perfect sphere, hidden from view.”

5. Michelangelo: . . . ”It’s not just a building, Tommaso, it’s a symbol.”

6. Tommaso: . . . ”A symbol of what, Michelangelo?”

7. Michelangelo: . . . ”Of God’s creation, of man’s ingenuity, of our faith and our dreams.”

Table 14: Scene alignment and speaker attribution in Wuthering Heights (scene 0189). The scene context is summarized and model outputs are excerpted from the recorded simulations.

Scene. During Heathcliff’s final days, Ellen Dean visits his chamber. He has withdrawn from company and refuses food, while she worries about his deteriorating condition.

Characters and motivations. Ellen Dean is practical and compassionate, with a strong sense of duty toward those in her care. Heathcliff is preoccupied with rejoining Catherine and resists interruption. Her concern conflicts with his withdrawal.

## Qwen-2.5-Instruct 7B

Scene score: 23.75

1. Ellen Dean: . . . (I place the tray on the table, my hand trembling slightly as I watch Heathcliff’s shallow breathing.)

2. Heathcliff: [Heathcliff’s fate is sealed, yet I must fulfill my duty.] (I approach the bed, my steps hesitant, and gently place the tray beside him, hoping he might find some solace in the food.)

3. Ellen Dean: . . . ”Eat, Mr. Heathcliff. It’s the least I can do.”

4. Heathcliff: . . . ”Eat, Mr. Heathcliff. It’s the least I can do.”

7. Heathcliff: . . . The wind howls mournfully, as if echoing his fate. . . .

## CoSER-8B

Scene score: 26.50

2. Heathcliff: . . . I am... better. Thank you.

3. Ellen Dean: . . . I’ve brought you some food, sir. Shall I open the windows?

6. Heathcliff: . . . Do you... do you miss me, Nelly?

7. Ellen Dean: . . . I... I do, Mr. Heathcliff. You’ve been kind to me in the past.

8. Heathcliff: . . . Thank you, Nelly. That’s all I need to know.

## ACTOR-Qwen-2.5-7B

Scene score: 47.50

1. Ellen Dean: . . . Heathcliff, you’ve not eaten. I brought you some broth.

2. Heathcliff: [Heathcliff’s eyes flicker with a mix of anger and despair as he looks at the food.]

3. Ellen Dean: (Heathcliff looks up, his eyes hollow and distant) Nelly, I’ve no appetite. (His voice is a hoarse whisper) I’ve no need for food.

5. Ellen Dean: . . . I’ve seen enough of this world. (His voice is a mere shadow of its former self) I’ve tasted enough of its bitterness.

8. Ellen Dean: . . . Heathcliff, you’re not yourself. (She tries to keep her voice steady) You’ve been ill for days.

Table 15: Prompts for plot and character-context curation.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Plot Segmentation</td><td>(intro) Partition the supplied book chunk into consecutive, non- overlapping plot segments for downstream character-action extraction. (requirements) Return the exact first and last sentences of every segment. Preserve source-text order and continue truncated segments from the pre- vious chunk. Maintain the canonical character registry and report newly observed aliases. Do not create overlapping segments or include a broad parent plot together with a contained sub-plot. Return strict JSON only.</td></tr><tr><td>Overlap Repair</td><td>(intro) Repartition a source-text region that has been assigned to overlap- ping plot segments. (requirements) Cover the supplied region exactly once, without gaps or duplicated passages. Segments must be mutually exclusive and remain in source order. Start a new segment only when there is a meaningful change in the immediate goal, setting, interaction, or outcome. Return a parsimonious set of segments with exact first-sentence anchors.</td></tr><tr><td>Character Enrich- ment</td><td>(intro) Generate compact character details for each supplied character in a finalized plot. (requirements) Use only the plot summary and the supplied plot text. For every character, generate a short description of identity, social position, or relevant relationships, together with a concise account of the charac- ter&#x27;s actions, decisions, intentions, or experiences in this plot. Do not use outside knowledge or information from later plots. Preserve the supplied canonical names and return strict JSON.</td></tr><tr><td>Alias Grouping</td><td>(intro) Group character names that refer to the same person and select a canonical name. (requirements) Merge formal names, honorifics, nicknames, and diminu- tives only when the identity is supported. Do not merge family members merely because they share a surname or title. If a name is genuinely am- biguous, place it in the unresolved list rather than guessing. Every input name must appear exactly once in either a group or the unresolved list.</td></tr></table>

Table 16: Prompts for scene and T-A-O behavior extraction.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Scene Splitting</td><td>(intro) Split a finalized plot into consecutive scenes representing continu- ous interactions in one setting. (requirements) Scenes must jointly cover the entire plot text without gaps or overlaps. Start a new scene when the setting changes or the interaction shifts to a different focus. For each scene, provide a description, canoni- cal participants, relationship context, and the exact first sentence used to locate the scene in the source text.</td></tr><tr><td>T-A-O Extraction</td><td>(intro) Extract character behavior trajectories as Thought-Action- Observation loops from the supplied plot text. (requirements) An action is a decision-bearing behavior for which the character has plausible alternatives. Include consequential decisions, de- cisional speech such as refusal, persuasion, promise, deception, or threat, and deliberate silence or non-intervention. Exclude passive information exchange, routine movements, decorative gestures, and actions with no meaningful character choice. Each step must contain exactly one atomic decision. Thought is written from the character&#x27;s first-person perspective and must be grounded in the source text. Action describes the performed behavior or preserves a direct quotation when applicable. Observation records only reactions or envi- ronmental changes perceptible to the character; omniscient narration and later events are prohibited. For every behavior, generate exactly three situationally plausible but character-inconsistent alternatives. Mark a behavior as critical only when the character has a reasonable alternative and the selected behav- ior changes a persistent state or a future situational possibility. Record the scene&#x27;s actual time and supporting evidence, rather than dates mentioned only in plans, memories, or documents. Return strict JSON only.</td></tr></table>

Table 17: Prompts for criticality verification and timeline reconstruction.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Criticality Loca- tor</td><td>(intro) Locate later plots and scenes that may contain evidence of the long- term consequences of a candidate critical behavior. (requirements) Use only the supplied event ledger and later scenes. Do not select the candidate&#x27;s own scene, an earlier scene, or a scene that merely repeats its wording or emotional tone. Return at most three supporting scene identifiers and a concise explanation of the possible downstream consequence. Return an empty list when no later evidence can be identi- fied.</td></tr><tr><td>Criticality Verifier</td><td>(intro) Independently verify whether a candidate behavior should retain its critical-behavior label. (requirements) Keep a candidate only when the supplied later evidence contains a concrete event, relationship change, decision, or consequence that could plausibly differ under another reasonable choice. Every retained case must cite at least one supporting scene and action identifier. Vague plot importance, current-scene mood, or repeated wording does not con-</td></tr><tr><td>Scene-Time Rela- beling</td><td>stitute evidence. Return keep or downgrade in strict JSON. (intro) Re-annotate the time at which the actions and dialogue in each scene actually occur. (requirements) Use only explicit or directly grounded evidence from the original plot text. A future appointment, a remembered event, or a date on a document is not automatically the time of the current scene. If the scene time is not established, return null for both the time label and its evidence.</td></tr><tr><td>Timeline Anomaly Detection</td><td>(intro) Detect confident cases in which the story-time order of scenes dif- fers from their textual presentation order. (requirements) Treat textual order as the default timeline. Report only scenes or contiguous scene blocks whose primary actions and dialogue clearly occur earlier in story time. Do not move a scene merely because it contains a recollection, discussion of a past event, or a future plan. Return proposed moves and uncertain cases separately.</td></tr><tr><td>Timeline Review</td><td>(intro) Independently review each proposed timeline adjustment against the original plot text. (requirements) Check both the moved block and its insertion neighbor- hood. Return accept only when the boundaries and insertion point are supported, reject when the textual order should be preserved or ev- idence is insufficient, and modify when a different move is required. Provide source-grounded reasoning for every decision.</td></tr><tr><td>Entity Resolution</td><td>(intro) Resolve source-level character names into persistent character en- tities. (requirements) Group names only when the supplied time-ordered appear- ances establish that they refer to the same person. Do not merge people based only on shared surnames, titles, gender, or proximity. Every source name must appear exactly once in an entity group or in the unresolved list. Prefer conservative unresolved assignments to unsupported merges.</td></tr></table>

Table 18: Prompts for dynamic character-profile construction.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Profile Initializa- tion</td><td>(intro) Construct the initial dynamic profile of a character immediately before the character&#x27;s first recorded behavior. (requirements) Use only the pre-action character description. Do not use the first action, later events, future relationships, or outside knowledge of the novel. Summarize the character&#x27;s supported identity, social position, temperament, interaction style, current relationships, and relevant lived experience in a compact profile. Return an evidence assessment together with the profile.</td></tr><tr><td>Profile Update</td><td>(intro) Update a character profile after a valid critical behavior and the accumulated actions since the previous profile version. (requirements) Set changed to true only when the action window causes a durable change in identity, social position, stable attitudes, important relationships, goals, self-understanding, or relevant lived experience. Re- peated or temporary behavior does not require a new version. When changed=false, return a null profile. When changed=true, rewrite the complete current profile rather than producing a chronological action log. Do not use future events or counterfactual alternatives.</td></tr></table>

Table 19: Prompts for counterfactual rewriting and record-level anonymization.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Counterfactual Rewriting</td><td>(intro) Rewrite the initially generated counterfactual behaviors for a source behavior. (requirements) The three alternatives must remain plausible responses to the same scene while violating the target character&#x27;s personality, values, goals, or behavioral style. Match the real behavior in length, structural form, speech format, register, and degree of formality. Preserve a quota- tion for dialogue behaviors and avoid quotations for non-dialogue behav- iors. The alternatives must not be simple paraphrases and should differ</td></tr><tr><td>Record Anonymiza- tion</td><td>strings in strict JSON. (intro) Replace person names in a dataset record with consistent anony- mous character identifiers. (requirements) Identify every person-name form appearing in the record, including full names, nicknames, abbreviations, possessives, and title— name combinations. Map forms referring to the same person to the same Character N, ordered by first appearance. Preserve place names, ani- mal names, relational terms, structural identifiers, and labels required for</td></tr></table>

Table 20: Shared character-context prompt used for RPO training.
<table><tr><td>Prompt</td><td>Template and requirements</td><td></td></tr><tr><td rowspan="3">Character text</td><td rowspan="3">Con-</td><td>(intro) You are role-playing as the target character. Decide what the char- acter will do next based on the supplied character profile, memories, and current scene. (requirements) Treat the provided Character Profile, Long-term Memory,</td></tr><tr><td>Recent Memory, and Current Scene as the complete and authoritative his- tory for the task. Do not rely on prior knowledge of the original novel, reconstruct omitted events, or infer future developments. The profile de- scribes the current character state; long-term memory contains validated key behaviors; recent memory provides local behavioral context; and the current scene contains the situation immediately before the target behav-</td></tr><tr><td>ior. During training, the source behavior is provided as chosen and one coun- terfactual behavior is provided as rejected. The model is trained to prefer the source behavior while also increasing its conditional likelihood under the auxiliary generation loss.</td></tr></table>

Table 21: Prompt used for four-way next-behavior selection and context ablations.
<table><tr><td>Prompt</td><td>Template and requirements</td></tr><tr><td>Next-Behavior Se- lection</td><td>(intro) Choose the action that the target character would most likely take next. (requirements) Use only the supplied Character Profile, Long-term Mem- ory, Recent Memory, and Current Scene. Do not use prior knowledge of the original novel or any event that occurs after the target behavior. Four</td></tr><tr><td>Context Ablations</td><td>provide an explanation or any additional text. (intro) Evaluate the effect of individual character-information sources on next-behavior selection. (requirements) Use the same selection prompt while selectively removing the book identity, character profile, long-term memory, recent memory, or both profile and memory. All other scene and option information re- mains unchanged. The model must follow the same single-letter output constraint in every condition.</td></tr></table>