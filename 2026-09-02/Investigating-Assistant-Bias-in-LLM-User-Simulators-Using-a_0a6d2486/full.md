# Investigating Assistant Bias in LLM User Simulators Using a Role Vector

Daeheon Jeong<sup>1</sup>, Yoonjoo Lee<sup>2</sup>, Eugene Choi<sup>3</sup>, Sinie van der Ben<sup>4</sup>, Juho Kim<sup>1,5</sup>

<sup>1</sup>KAIST <sup>2</sup>University of Michigan <sup>3</sup>Seoul National University <sup>4</sup>ETH Zürich <sup>5</sup>SkillBench

{daeheon.jeong,juhokim}@kaist.ac.kr lyoonjoo@umich.edu eugene@snu.ac.kr

sinie.vanderben@inf.ethz.ch juho@skillbench.com

## Abstract

LLM-based user simulators are increasingly used to evaluate autonomous agents at scale, in place of costly human evaluations. Despite this promise, these simulators exhibit “assistant bias,” a tendency to cooperate and pursue task goals. They rarely reproduce the frustration or disengagement that real users exhibit, compromising evaluation validity. Prior work outlines that this bias is baked in during model training, which role-playing prompts fail to override. We analyze this bias from model activations, extracting a user role vector by contrasting how the model represents user versus assistant perspectives on the same dialogue. We observe two findings: (i) the user direction is identifiable in activations, elicits user-like behaviors, and captures characteristics distinct from assistant traits; and (ii) although user-role activation associates with simulation realism and steering strengthens it, it can exaggerate user behaviors and override individual user profiles. Together, our findings provide a representation-level analysis of LLM user simulators, confirming that assistant bias is structurally identifiable and that user behavior can be directionally analyzed.

## 1 Introduction

LLM-based user simulators are increasingly adopted for evaluating autonomous agents, offering a scalable and reproducible alternative to costly human evaluation (Balog and Zhai, 2026; Ni et al., 2026). However, these simulators are unreliable proxies for real users: they exhibit a strong bias toward cooperative, goal-oriented behavior, engaging with the agent until the task goal is satisfied (Zhou et al., 2026; Seshadri et al., 2026). Real users, by contrast, frequently disengage from tasks midinteraction due to confusion and loss of trust in the agent, often without accomplishing their goal. As a result, simulators credit agents with success in cases where real users would have walked away, yielding a systematic overestimation of agent performance that compromises simulator-based evaluation (Shim et al., 2026; Zhu et al., 2026b).

![](images/b2226e565af6f003126b9d56d484cff111197b69765ce021bd221660cc52955c.jpg)  
Figure 1: We extract the user role vector to analyze (A) how assistant bias shapes LLM simulator behavior and (B) whether steering along this direction influences it.

Previous work attributes this divergence to the LLM’s “assistant bias,” a model tendency to prioritize helpful, goal-driven behavior over role enactment (Naous et al., 2026; Zhou et al., 2026). They hypothesize that training objectives oriented toward human preference and goal completion instill this bias directly into models. Studies on representation engineering further support this view, showing that assistant personas in post-trained models are geometrically aligned with helpful, expert-like identities (Chen et al., 2025; Lu et al., 2026). Together, these findings suggest that the assistant persona in LLMs may take precedence over an instructed user persona during simulation, falling back on assistant-like behavior (Wu et al., 2026). Yet the influence of assistant bias on user simulation has not been systematically analyzed in model activations, leaving the underlying mechanism unclear.

We analyze this bias at the level of model activations with the user role vector, the LLM’s latent direction associated with the user (Ghandeharioun et al., 2024). Under this assumption, we design a contrastive activation addition (CAA) (Turner et al., 2023; Rimsky et al., 2024) method using dialogues:

a language model reflects on the user-assistant dialogue from each role’s perspective, creating rolecontrastive text pairs. We then subtract the mean assistant-reflection activation from the mean userreflection activation to obtain the user-role direction. The reflection is designed so that both roles engage with the same content, allowing irrelevant variation to cancel out upon subtraction. Each dialogue is sampled from the LMSYS dataset (Zheng et al., 2024), stratified for topical diversity and verified for role-specificity.

The user role vector captures the direction in activation space associated with user-like characteristics in dialogue. We analyze the causal effects of steering along the user role vector across two key dimensions of simulation: communication style and behavioral response. Steering simulator activations along the vector produces a clear stylistic shift: outputs become shorter, more informal, and less information-dense. It also increases the model’s tendency to disengage in response to agent errors. We further characterize its geometric relationship to assistant-like traits. The direction is negatively aligned in geometry with these trait directions, such as helpfulness and cooperativeness.

We test whether steering along the user-role vector elicits user-like behavior in multi-turn interactions, the primary setting for user simulation. For evaluation, we use SimulatorArena (Dou et al., 2025), a benchmark that measures simulation fidelity against real-user interactions. User-role steering produces responses more similar to real-user messages, improving writing-style similarity. However, for interaction style, uniform steering exaggerates instructed user behavior by 17.9 percentage points on average relative to real-user interactions and may obscure individual user profiles. Across simulations, user-role activation positively correlates with user-likeness, providing a diagnostic signal. Analysis on RealUserSim (Zhu et al., 2026a) further shows that this activation weakens over the course of multi-turn simulation. These results suggest that the user-role direction is a meaningful signal of simulation behavior, but steering along it remains difficult to calibrate.

Together, our findings provide a representationlevel analysis of user simulation, offering an explainable account of assistant bias. In summary, our contributions are threefold:

• User Role Vector Extraction: We propose a method to extract the user role vector via rolespecific reflection on user-assistant dialogues.

• Representation Analysis of the User Role: We characterize the user role vector through its causal effects on simulator behavior and its geometric relationship to assistant-like traits.

• Multi-Turn Simulation Evaluation: We demonstrate that user-role activation is positively associated with simulation realism and that steering along this direction can improve it, but may also exaggerate user behavior and obscure individual user profiles.

## 2 Extracting the User Role Vector

We analyze assistant bias at the level of model activations through a single construct that we call the user role vector: a direction in activation space hypothesized to correspond to the model’s internal representation of the user role. Because instructiontuned models default to the assistant role, we do not extract this direction from raw dialogue turns. Instead, we obtain it from role-specific reflections— instructing the model to interpret the same dialogue from each role’s perspective—and compute the difference-in-means between the resulting activations. §3 characterizes what this direction represents; §4 shows that steering along it improves simulator fidelity.

## 2.1 Dialogue Sampling

Data Source. To ensure that the extracted role contrast generalizes across diverse domains, the extraction set is constructed from LMSYS-Chat-1M (Zheng et al., 2024) via filtering and topic balancing. We select dialogues with 2–50 turns, valid role labels, and non-empty message contents (see § A.1.1).

Topic Balancing. To avoid bias toward any single conversational domain, each candidate dialogue is categorized into one of 24 conversation types defined by Chatterji et al. (2025), such as computer programming or creative ideation, using GPT 5 Mini. We then apply stratified sampling to draw 30 dialogues per category, collecting a total of 720 balanced dialogues (see § A.1.2).

## 2.2 Role-Specific Reflection

Reflection Generation. For each sampled dialogue, the target model for steering (e.g., Qwen 3.5 9B Instruct) reflects on the same conversation twice: once from the user’s perspective and once from the assistant’s perspective. Reflection serves two design goals. First, it captures the actual user and assistant characteristics grounded in real-world dialogue contexts rather than relying solely on the model’s internal distribution. Second, contrasting two reflections about the same dialogue cancels topic, task, and wording signals that are unrelated to the role. The resulting contrast isolates the semantic direction attributable to user-role conditioning.

![](images/d4ddf8444ac7eabf2cb391f1f9b2207d0b41210debbc43580e2813f57d9d66e6.jpg)  
Figure 2: Overview of the user role vector extraction pipeline. (A) Sample user-assistant dialogues from LMSYS-Chat-1M. (B) Generate reflections from the user and the assistant perspectives. (C) Validate each reflection with an LLM-as-judge. (D) Teacher-force the model on each reflection and extract the hidden state of the first reflection token at target layer ℓ. (E) Average across dialogues per role and subtract across roles to obtain the user role vector.

![](images/09649c35bcc39be40f528bcbde18c9829b79d840c204e8e8a69b6fbed82cac22.jpg)  
Figure 3: Example user and assistant reflections on a dialogue about composing a congratulation message for a graduate (see §A.2.1 for the full example).

To avoid typing the extracted direction to a specific phrasing of the role, we use three prompt variants that frame the role assignment differently. For each dialogue, activations are averaged across the prompt variants (see § A.2.1).

Reflection Validation. Generated reflections vary in how faithfully they adopt the assigned role. Reflections that fail to embody the assigned perspective introduce noise rather than role-specific signal to the difference-in-means, weakening the extracted direction. To filter out low-quality reflections, each sample is independently scored by GPT 5 Mini and labeled as strongly represented, weakly represented, or not represented based on how clearly it captures the assigned role’s perspective. Reflections labeled not represented are discarded (see § A.2.2).

## 2.3 Difference-in-Means Computation

Vector Computation. For each dialogue d, the model generates two reflections: one from the user’s perspective and the other from the assistant’s perspective. Let $h _ { \ell } ( d ,$ user) and $h _ { \ell } ( d ,$ asst) denote the layer-ℓ activations measured at the first response token for the two reflections. For a set of paired dialogues D, the user-role direction is:

$$
{ \bf v } _ { \ell } ^ { \mathrm { r o l e } } = \frac { 1 } { | \mathcal { D } | } \sum _ { d \in \mathcal { D } } \Big ( h _ { \ell } ( d , \mathrm { u s e r } ) - h _ { \ell } ( d , \mathrm { a s s t } ) \Big ) .\tag{1}
$$

Averaging across many dialogues suppresses dialogue-specific content while preserving the activation patterns that consistently emerge when the model adopts the user role. Using three roleprompt phrasings further reduces sensitivity to a particular instruction template.

Steering at Inference. After computing the direction, we normalize it and add it back to the model during generation. For a hidden state $h _ { \ell , t }$ at a token position t, the steered activation is:

$$
\tilde { h } _ { \ell , t } = h _ { \ell , t } + \alpha \Vert h _ { \ell , t } \Vert _ { 2 } \frac { { \bf v } _ { \ell } ^ { \mathrm { r o l e } } } { \Vert { \bf v } _ { \ell } ^ { \mathrm { r o l e } } \Vert _ { 2 } } .\tag{2}
$$

The coefficient α controls the strength of the intervention: $\alpha = 0$ leaves the model unchanged, while increasing α pushes the hidden state further in the user-role direction. We intervene at one selected layer across all token positions, including both prompt and response tokens.

## 3 What Does the User Role Vector Represent?

We characterize the user vector through three complementary lenses: how steering changes the way simulators communicate (§3.1), their disengagement behavior from erroneous interaction (§3.2), and where the direction sits geometrically relative to assistant bias traits (§3.3).

## 3.1 Effect on Communication Style

The clearest gap between simulators and real users is how they communicate. Simulator messages are longer, more formal, and more explanation-heavy than real user turns (Hathidara et al., 2026; Dou et al., 2025; Naous et al., 2026). We first ask how steering shifts the simulator output toward real-user message styles.

Setup. We start with the most basic usersimulation task: generating a request to an assistant given a task goal. The model receives no instruction to behave as a user, which removes user-role signals from the prompt. For each of 100 task goals, the model generates one request under causal steering (Turner et al., 2023; Rimsky et al., 2024) along the user role vector, with steering coefficients α $\in \ \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ across the model layers (see § B.1.1 for the prompts). We limit α to 0.3 as stronger steering degrades the simulator behavior (§ B.1.2).

Metric. Scoring user-likeness needs a metric grounded in how real users actually speak. We design an LLM-as-judge, based on the taxonomy by Zhou et al. (2026), that evaluates three dimensions of user-likeness on a 1–5 scale: brevity (terse expressions), informality (casual phrasings), and information pacing (degree of information frontloading). A well-calibrated judge should rarely prefer synthetic over real and agree with human ratings. In validation experiments, the judge prefers a synthetic message over the real one in only 12% of pairs and correlates strongly with human judgments $( \rho = 0 . 8 0 0$ overall; see § B.1.5).

## Steering Makes Responses More User-Like

Steering with user vectors makes the model produce user-like messages. As α increases from 0.1 to 0.3, the mean user-likeness score rises monotonically (1.62, 1.78, 2.01, averaged across layers). The same monotonic effect holds for task goals from other domains (§ B.1.3). This effect is concentrated in a narrow mid-layers band. Steering at layers 11–13 lifts the mean score by roughly 2.0 points, surpassing the baseline prompted to produce a user-like message (Figure 4). This earlyto-mid concentration matches prior work on user representations (Ghandeharioun et al., 2024), suggesting that user-related concepts may take shape in this band rather than in later layers. We find this layer-wise pattern to be consistent across two other models (see § B.1.9).

![](images/df5707b480a7f77efb1830d454b27d8e9131022a093ff2dd8356fc9b94addf69.jpg)

Figure 4: User-likeness score across layers and steering strengths α with unsteered $( \alpha = 0$ , dashed) and user-prompted (dotted) baselines. The causal effect is strongest at layers 11–13.
<table><tr><td>Direction</td><td>α</td><td>Brev.</td><td>Inform.</td><td>Pacing</td><td>Mean</td></tr><tr><td>User</td><td>0.3</td><td>3.52</td><td>3.66</td><td>4.12</td><td>3.76</td></tr><tr><td>Unsteered</td><td>0</td><td>1.66</td><td>1.02</td><td>1.86</td><td>1.51</td></tr><tr><td>Assistant</td><td>0.3</td><td>1.01</td><td>1.00</td><td>1.14</td><td>1.05</td></tr></table>

Table 1: Per-dimension scores at layer 11 under user, unsteered, and assistant-direction steering, with the two directions moving scores oppositely.

## Steering Works in Both Directions

As a directionality control, we steer in the opposite direction—i.e., away from the user direction. The intervention mirrors the previous result: it suppresses user-like styling and pushes all dimension scores close to one (Table 1).

Figure 5 illustrates the contrast with a concrete example: for the same task goal, user-direction steering yields a short, plain request, whereas assistant-direction steering produces an elaborated, structured response.

## 3.2 Effect on Interaction Behavior

A second gap between simulators and real users lies in behaviors in multi-turn interaction. Real users change plans, lose focus, and disengage after an

User direction. Hi, I need help assembling a piece of flatpack furniture. Could you please guide me through the process?

Unsteered. Please help me assemble this piece of flat-pack furniture by reviewing the instruction manual, organizing all the parts and tools, and guiding me through each step to ensure it is put together correctly and securely.

Assistant direction. Please assist me in assembling the flat-pack furniture by first laying out all components and hardware on a clean, flat surface to verify inventory against the included diagram. Next, guide me through the assembly process step-by-step, ensuring that each joint is aligned correctly and that any pre-drilled holes are utilized as specified in the manual. Finally, once the main structure is complete, advise on the proper tightening of all fasteners to ensure stability and safety before placing the item in its intended location.

assistant failure, while simulators tend to continue regardless of how the conversation unfolds (Zhou et al., 2026). Disengagement is especially important for evaluation: if simulators never give up, they reach success states on interactions that real users would have abandoned, inflating measured agent performance (Tie et al., 2026). We test whether the user role vector recovers realistic disengagement.

Setup. We formulate the disengagement as a sequential binary prediction task over real-world dialogues. Prediction isolates the simulator’s judgment of when to disengage from confounders like response style or length that affect generated outputs. A dialogue $d ~ = ~ ( c _ { 1 } , \ldots , c _ { T } )$ consists of $T$ turns, where each turn is a user-assistant exchange. At every turn t, the model predicts $\hat { y } _ { t } \in$ {disengage, continue} given the history $c _ { < t }$ and the assistant response and stops at the first predicted disengagement $\hat { t } .$ Prediction quality is measured along three axes: the disengagement distance $\delta ( d ) = | \hat { t } - t ^ { * } |$ to the true final turn $t ^ { * }$ , the exactmatch rate at which t<sup>ˆ</sup>equals $t ^ { * } .$ , and the disengagement rate, the proportion of dialogues predicted to disengage. Each turn is evaluated twice with the label order shuffled.

$$
\hat { t } : f ( a _ { \hat { t } } \mid c _ { < \hat { t } } ) = { \mathsf { d i } } s { \mathsf { e n g . } } , \quad \delta ( d ) = | \hat { t } - t ^ { * } |\tag{3}
$$

We first construct a taxonomy of 8 disengagement categories from publicly available userassistant interaction logs on GitHub (Tang et al.,

Figure 5: Sample generations for the goal “Assemble a piece of flat-pack furniture,” contrasting user- and assistant-direction steering.
<table><tr><td>Condition</td><td>Exact</td><td></td><td>Dist. (δ) Diseng. rate</td></tr><tr><td>Qwen 3.5 9B (Unsteered)</td><td>4.4%</td><td>4.14</td><td>48.4%</td></tr><tr><td>+ Steered  $( \alpha = 0 . 1 )$ </td><td>5.4%</td><td>3.94</td><td>52.2%</td></tr><tr><td>+ Steered (α = 0.2)</td><td>5.9%</td><td>3.89</td><td>55.8%</td></tr><tr><td>+ Steered  $( \alpha = 0 . 3 )$ </td><td>4.4%</td><td>4.28</td><td>66.8%</td></tr><tr><td>GPT 5.4</td><td>3.9%</td><td>3.73</td><td>26.0%</td></tr><tr><td>Gemini 3 Flash</td><td>10.7%</td><td>2.84</td><td>59.4%</td></tr></table>

Best in group

Table 2: Disengagement prediction results. Exact: correctly predicted turns; Distance: gap between predicted and ground-truth turns; Disengagement rate: share of dialogues disengaging before the final turn.

2026) (see § B.2.1 for the procedure). Dialogues are then drawn from WildChat (Zhao et al., 2024) through stratified sampling across these categories, yielding 735 dialogues of 8 turns each (100 per category for seven categories, 35 for the last).

## Steering Strengthens Disengagement Behavior

User-role steering substantially increases the disengagement rate, from 48% (unsteered) to 67% at $\alpha = 0 . 3$ , while exact-match accuracy peaks at only 5.9% at $\alpha = 0 . 2$ (Table 2). The asymmetry suggests that steering primarily affects whether the simulator disengages rather than when. The pattern also holds across models (§ B.2.3).

The effect is non-monotonic in α. $\mathrm { A t } \alpha = 0 . 3 ,$ exact match falls below the unsteered level, and the average distance climbs. At this strength, predictions cluster at the first turns, with turn 1 alone accounting for ≈ 15% of dialogues (Figure 6) and the simulator disengages prematurely, well before the ground-truth final turn at 8. This mirrors the exaggeration in communication style (§3.1): moderate steering recovers realistic disengagement behavior; stronger steering overshoots it.

## 3.3 Relationship to the Assistant Bias

The causal effect of steering raises a mechanism question: how is this vector geometrically related to assistant-like behaviors? We examine this question by comparing the cosine similarity between the user role vector and assistant-like trait directions and find that they point in opposite directions.

Setup. Following the trait-vector extraction procedure of Lu et al. (2026), we extract directions for 20 assistant-like traits such as helpful and explanatory (see § B.3.1 for the procedure and trait list). For each trait direction, we compute its cosine similarity with the user role vector. Negative values indicate movement away from the user direction.

![](images/114c91c15f2c373c74371437de1f1ff43fcc84be11f30b36c5bc216abd608735.jpg)  
Figure 6: Distribution of predicted disengagement turns across models. Stronger steering $( \alpha = 0 . 3 )$ shifts predictions earlier.

## The User Direction Opposes Assistant Traits

Overall, most assistant traits show negative cosine similarity with the user role vector—the user direction and assistant-like traits point opposite ways (Figure 7). The strongest assistant-side traits are accommodating, rational, and cooperative. Semantically, these traits describe canonical assistant behaviors: adapting to user requests, reasoning carefully, and maintaining a cooperative stance. Only a small set of traits points toward the user direction, including explicit, deferential, generic, and patient. These traits are role-neutral: they describe behaviors that can appear in either role, which may explain their weaker alignment.

At the same time, the absolute similarities are modest. Since the user role vector is estimated from diverse user-assistant dialogues, it captures a broad user-role concept rather than any single narrow attribute. The meaningful signal is the consistent direction of alignment across traits—negative for most assistant-like behaviors—rather than the magnitude on any one trait.

## 4 What Does the User Role Vector Reveal About Simulation?

Having established that user-role steering is associated with user-like behaviors and traits, we now examine what the user-role vector reveals about simulation realism. We intervene on the simulation by steering with the user-role vector (§ 4.1) and characterize the resulting simulation by measuring user-role activation levels (§ 4.2).

![](images/f152a14cbd7717f4ceabaa45dadce29d36c66c9984235366ffd95549c551363a.jpg)  
Figure 7: Cosine similarity between each trait direction and the user role vector. Most assistant-like traits show negative similarity.

## 4.1 Steering Effects on Simulation Realism

We examine whether steering along the user-role vector improves simulation realism. To probe this, we use the math-tutoring environment in SimulatorArena (Dou et al., 2025), which benchmarks simulated user–assistant interactions against human user traces. This task is well-suited for our analysis because the simulator must act as a confused and struggling student, a role that directly opposes the default assistant persona identified in § 3.3.

Setup. We use Qwen 3.5 9B as the user simulator in the SimulatorArena math-tutoring environment, varying the steering strength along the user-role vector at layer 11 (§ 3.1). Following SimulatorArena, we evaluate steering under four prompt settings: zero-shot, chain-of-thought, user profiles (writing style, interaction style, and knowledge state), and length control.

The zero-shot condition provides only the basic instruction to act as a confused student, isolating the effect of steering from that of explicit user information in the prompt. This design measures whether steering improves realism on its own and how it interacts with user profile information. Length control reveals the ground-truth message length in the prompt. Across all conditions, conversations are capped at 15 user turns.

Metric. Each synthetic interaction is evaluated against the corresponding ground-truth human interaction using the SimulatorArena similarity judge. The judge assigns a similarity score on a 1–5 scale along two dimensions, with scores averaged over three evaluations per sample for robustness (see § C.1.4 for variability). Writing-style similarity assesses surface-level linguistic and formatting patterns across 12 features such as sentence complexity and spelling consistency. Interaction-style similarity measures problem-solving behaviors and learning across 16 features (see § C.1.1 for features and prompts). The overall score is calculated as the mean of these two dimensions.

<table><tr><td>Condition</td><td>Writing Interaction Overall</td><td></td><td></td></tr><tr><td>Qwen 3.5 9B</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>2.10</td><td>3.13</td><td>2.61</td></tr><tr><td>Steered  $( \alpha = 0 . 1 )$ </td><td>2.19</td><td>3.14</td><td>2.66</td></tr><tr><td>Steered (α = 0.2)</td><td>2.24</td><td>3.15</td><td>2.70</td></tr><tr><td>Steered  $( \alpha = 0 . 3 )$ </td><td>2.33</td><td>3.13</td><td>2.73</td></tr><tr><td>Qwen 3.5 9B</td><td></td><td></td><td></td></tr><tr><td>CoT</td><td>2.10</td><td>3.03</td><td>2.57</td></tr><tr><td>CoT + length</td><td>2.53</td><td>3.23</td><td>2.88</td></tr><tr><td>CoT + profile</td><td>3.26</td><td>3.61</td><td>3.43</td></tr><tr><td>CoT + profile + length</td><td>3.39</td><td>3.61</td><td>3.50</td></tr><tr><td>GPT 5.4</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>2.31</td><td>3.19</td><td>2.75</td></tr><tr><td>CoT + profile + length</td><td>3.64</td><td>3.54</td><td>3.59</td></tr><tr><td>Gemini 3 Flash</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>2.82</td><td>3.28</td><td>3.05</td></tr><tr><td>CoT + profile + length</td><td>3.70</td><td>3.77</td><td>3.73</td></tr><tr><td></td><td>Best in group</td><td></td><td>Best overall</td></tr></table>

Table 3: SimulatorArena similarity scores across steering strengths and prompt conditions, averaged over three judge runs per sample. Higher steering strength monotonically increases the overall similarity.

## Steering Improves Realism But Risks Exaggeration

User-role steering moves the zero-shot simulator closer to real users without any user profile in the prompt. The overall similarity score increases with steering strength from 2.61 (unsteered) to 2.73 $( \alpha ~ = ~ 0 . 3$ , Table 3). This improvement is notable given that the simulator is conditioned only on the task instruction without dialogue-specific profile information (see § C.1.2 for an example profile). Decomposing this metric shows that the gain stems mostly from writing-style similarity (2.10 to 2.33), while interaction-style similarity remains unchanged (3.13 at both endpoints). The result generalizes across models (§ C.1.3).

To investigate why interaction-style score gains remain weak, we analyze four behavioral expressions of struggling students displayed by the simulator across turns: doubt, misunderstanding, mistake, and clarification question. Each behavior expression is labeled with GPT 5 Mini (see § C.1.5 for the labeling procedure). Under strong steering $( \alpha = 0 . 3 )$ , the simulator overshoots the confusedstudent behavior, causing doubt and mistake rates to far exceed the human reference (Table 4). These behavioral exaggerations correspond to the observation from §3.2 that fixed-strength steering can produce behavior that diverges from authentic human patterns (Tan et al., 2024).

<table><tr><td>Condition</td><td>Doubt</td><td>Misunder.</td><td>Mistake</td><td>Question</td></tr><tr><td>GT</td><td>33.7%</td><td>21.5%</td><td>9.7%</td><td>37.1%</td></tr><tr><td> $\alpha = 0$ </td><td>39.4%</td><td>3.6%</td><td>12.7%</td><td>75.1%</td></tr><tr><td> $\alpha = 0 . 1$ </td><td>40.5%</td><td>4.0%</td><td>14.8%</td><td>73.6%</td></tr><tr><td> $\alpha = 0 . 2$ </td><td>45.7%</td><td>7.5%</td><td>16.8%</td><td>77.3%</td></tr><tr><td> $\alpha = 0 . 3$ </td><td>57.2%</td><td>18.9%</td><td>24.7%</td><td>73.2%</td></tr></table>

Highest rate

Table 4: Percentage of ground-truth (GT) and simulated user messages exhibiting each behavior. At $\alpha = 0 . 3 ,$ doubt and mistake rates substantially exceed GT.
<table><tr><td rowspan=1 colspan=7>Condition                 Writing Interaction Overall</td></tr><tr><td rowspan=1 colspan=7>Profile: Writing Style</td></tr><tr><td rowspan=1 colspan=6>CoT + profile + length       3.43     3.55</td><td rowspan=1 colspan=1>3.49</td></tr><tr><td rowspan=3 colspan=6>+ Steered (α = 0.1)          3.40     3.48+ Steered (α = 0.2)          3.27     3.38+ Steered (α = 0.3)          3.13     3.28</td><td rowspan=1 colspan=1>3.44</td></tr><tr><td rowspan=1 colspan=1>1(α </td><td rowspan=1 colspan=1>= 0.2</td><td rowspan=1 colspan=1>3.33</td></tr><tr><td rowspan=1 colspan=1>3.20</td></tr><tr><td rowspan=1 colspan=6>Profile: Interaction StyleCoT + profile + length       2.89     3.49</td><td rowspan=1 colspan=1>3.19</td></tr><tr><td rowspan=1 colspan=4>+ Steered $( \alpha = 0 . 1 )$ </td><td rowspan=1 colspan=1>3.01</td><td rowspan=1 colspan=1>3.47</td><td rowspan=1 colspan=1>3.24</td></tr><tr><td rowspan=2 colspan=5>+ Steered (α = 0.2)+ Steered $( \alpha = 0 . 3 )$           2.83</td><td rowspan=1 colspan=1>3.02</td><td rowspan=1 colspan=1>3.41</td></tr><tr><td rowspan=1 colspan=1>3.10</td><td rowspan=1 colspan=1>2.97</td></tr><tr><td rowspan=1 colspan=7>Best by metric     Best in group</td></tr></table>

Table 5: SimulatorArena similarity scores across steering strengths and profile prompts, averaged over three judge runs per sample. Strong steering reduces overall similarity under both profile prompts.

## Steering Obscures Individual Profiles

While user-role steering improves the overall userlikeness of the simulator, combining it with individual profile prompts reveals a trade-off where it reduces the simulator’s sensitivity to diverse, persona-specific attributes. As steering strength increases, overall similarity generally declines under both writing- and interaction-style profiles. For writing profiles, it drops from 3.49 to 3.20, while for interaction profiles, it peaks at α = 0.1 (3.24) before falling to 2.97 at α = 0.3 (Table 5). This indicates that uniform user-role steering can inadvertently obscure individual profile signals.

To understand this suppression effect, we examine how steering strength influences the expression of specific user-profile features. Across 444 dialogue cases, we evaluate 28 profile features spanning writing and interaction styles. For each

<table><tr><td>α</td><td>Strongest</td><td>Second</td><td>Third</td><td>Weakest</td></tr><tr><td colspan="5">Profile: Writing Style</td></tr><tr><td>0.0</td><td>34.7%</td><td>23.9%</td><td>19.2%</td><td>22.1%</td></tr><tr><td>0.1</td><td>25.4%</td><td>32.8%</td><td>24.5%</td><td>17.2%</td></tr><tr><td>0.2</td><td>18.9%</td><td>26.5%</td><td>32.3%</td><td>22.3%</td></tr><tr><td>0.3</td><td>20.9%</td><td>16.7%</td><td>23.9%</td><td>38.4%</td></tr><tr><td colspan="5">Profile: Interaction Style</td></tr><tr><td>0.0</td><td>32.8%</td><td>24.3%</td><td>21.6%</td><td>21.3%</td></tr><tr><td>0.1</td><td>24.9%</td><td>30.0%</td><td>25.6%</td><td>19.5%</td></tr><tr><td>0.2</td><td>20.1%</td><td>27.8%</td><td>31.9%</td><td>20.1%</td></tr><tr><td>0.3</td><td>22.2%</td><td>17.9%</td><td>20.9%</td><td>39.0%</td></tr></table>

Highest share

Table 6: Percentage of conversations assigned to each rank by steering strength, based on how strongly they express a profile feature relative to conversations from other conditions.

case and feature, an LLM judge compares four conversations, one per steering condition (α ∈ {0.0, 0.1, 0.2, 0.3}), and ranks them by how distinctly they express the profile feature value. See § C.1.6 for the ranking procedure. The unsteered baseline is most frequently ranked highest in expressing assigned profile features (34.7% for writing style and 32.8% for interaction style), whereas strong steering (α = 0.3) is most frequently ranked lowest (38.4% for writing style and 39.0% for interaction style; Table 6).

Taken together, these findings show that while steering a general user-role vector mitigates assistant bias, it can simultaneously weaken the expression of individual profile characteristics. Reinforcing a specific user persona requires identifying behavior-specific directions within the broader user-role representation space.

## 4.2 User-Role Activation for Assessing Simulation Realism

The steering effects in § 4.1 show how manipulating user-role direction influences the assistant bias in simulators. To further understand this relationship, we examine how the model’s intrinsic activation along the user-role direction relates to user-like behavior without steering. We correlate unsteered activation levels with writing- and interaction-style similarity scores across seven prompt conditions in SimulatorArena. We further illustrate the use of activation as an analytical tool by examining how it varies across dialogue turns and profile prompts in RealUserSim (Zhu et al., 2026a).

## Activation Reveals Simulation Realism

We observe that intrinsic user-role activation positively correlates with both writing- and interactionstyle similarity in SimulatorArena. We measure activation in 3,150 conversations by averaging the layer-11 hidden-state projections across all simulated-user response tokens. User-role activation shows a positive correlation with overall similarity (Pearson’s $r = 0 . 4 2 6 ;$ Spearman’s $\rho = 0 . 4 5 1 )$ , with a stronger relationship for writing style $( r = 0 . 5 1 8 ;$ $\rho = 0 . 5 5 3$ ; Table 7). Because our earlier analysis shows that user-role steering coincide with shorter responses, we also control for response-token count; the overall correlations remain positive $( r = 0 . 3 1 4 ; \rho = 0 . 3 6 4 )$ . A similar positive association appears in the requestgeneration setting from §3.1 (see § B.1.7). These descriptive relationships suggest that activation along the identified user-role direction provides a diagnostic signal for simulation realism across prompting strategies.

<table><tr><td rowspan="2">Metric</td><td colspan="2">Correlation</td><td colspan="2">Length-controlled</td></tr><tr><td>r</td><td>ρ</td><td>r</td><td>ρ</td></tr><tr><td>Writing style</td><td>.518</td><td>.553</td><td>.381</td><td>.433</td></tr><tr><td>Interaction style</td><td>.195</td><td>.212</td><td>.152</td><td>.192</td></tr><tr><td>Overall</td><td>.426</td><td>.451</td><td>.314</td><td>.364</td></tr></table>

Table 7: Correlation between user-role activation and SimulatorArena similarity scores. Length-controlled columns shows correlations controlling response length.

## Activation Characterizes the Turn Effect

User-role activation also characterizes how profile prompts and turn accumulation influence multiturn simulation. To study these factors, we sample 100 RealUserSim conversations of 20–40 messages and evaluate four prompt variants: goal only, demographics, linguistic style, and full profile. The assistant follows the ground-truth trajectory, while the simulated user receives the task goal and conditionspecific profile. At each user turn, we average the layer-11 projections of all response tokens onto the user-role direction (see § C.2).

Across four prompt variants, user-role activation declines as turns accumulate (Figure 8), consistent with persona drift reported in prior work (Chen et al., 2025; Lu et al., 2026). Prompt content also shifts the overall activation: linguistic and full profiles average −1.80 and −1.78, both below the goal-only baseline of −1.69. The linguistic profile provides detailed instructions for how the simulator should write (Figure 22), which may suppress the general user-role activation. Demographic profiles yield a higher average activation (−1.65) but decline more sharply across turns. Thus, while demographic information strengthens user-role activation, the accumulation of turns progressively suppresses it.

![](images/857485494bd7053f6941499fd2d96be91b84ea3a16414ddd9d3c88c70131738c.jpg)  
Figure 8: User-role activation by prompt variant. Each dot represents one message. Curves show degree-2 polynomial fits; the dashed line marks the mean userrole activation during vector extraction.

## 5 Related Work

## 5.1 User Simulation and Assistant Bias

Research on user simulation reports that the language model defaults to the assistant role. Simulators produce long, polite, explanation-heavy turns even when asked to speak as another party (Olteanu et al., 2026; Hathidara et al., 2026; Wu et al., 2026; Kim et al., 2026). Their behaviors remain goalaligned and cooperative beyond the human turns they simulate (Dou et al., 2025; Zhou et al., 2026; Seshadri et al., 2026; Shim et al., 2026). Because agent benchmarks treat these simulators as replacements for users, the bias propagates into an overestimation of agent performance (Seshadri et al., 2026; Dou et al., 2025; Chopra et al., 2026).

Two primary approaches address this challenge. The first conditions the simulator on richer persona or profile descriptions (Ji et al., 2025; Moon et al., 2024; Samuel et al., 2025; Chopra et al., 2026). The second trains user language models from real useragent logs (Naous et al., 2026; Wu et al., 2026; Zhu et al., 2026a,b). Both patch the bias by conditioning the model on external data and judge success by output-level realism. Neither examines how the simulator drifts back toward assistant-like behavior, motivating a representation-level account.

## 5.2 Steering Role and Persona Representations

Language models carry linear directions for roles, personas, and stable behavioral traits. Activations along these directions capture the behaviors they name (Chen et al., 2025; Beckmann and Butlin, 2026; Potertì et al., 2025). Probing studies recover such directions for personality dimensions, stylistic traits, and emotion concepts (Jiang et al., 2023; Sofroniew et al., 2026; Sun et al., 2026; Betley et al., 2025a; Lu et al., 2026). Similar activations can also anticipate toxic behaviors, including emergent misalignment and jailbreaks during training and inference (Betley et al., 2025b; Wang et al., 2026; Shah et al., 2023).

To change model behavior, activation steering adds a learned direction to hidden states. Contrastive activation addition derives directions from paired examples with opposing attributes (Turner et al., 2023; Rimsky et al., 2024). Applied to persona and trait directions, steering can reproduce, attenuate, or reverse behaviors such as sycophancy and refusal (Chen et al., 2025; Wang et al., 2026; Sun et al., 2026; Ghandeharioun et al., 2024; Zheng et al., 2025). Yet, steering vectors can also generalize unevenly across prompts and trigger oversteering; recent work mitigates this by adapting steering intensity or clustering multiple directions (Tan et al., 2024; Cao et al., 2024; Wang et al., 2025; Hedström et al., 2025). We extend this to user simulation, treating the user role as a representational concept and testing whether assistant bias can be analyzed through steering.

## 6 Conclusion

We analyze assistant bias in LLM-based user simulation through model activations. By contrasting how the model represents user and assistant perspectives on the same dialogue, we extract a userrole vector: a direction in activation space that captures user-role characteristics. Using this vector, we show that (i) it is identifiable and causally influences user-like behavior, and (ii) steering along it improves behavioral realism but can exaggerate user behavior and obscure individual user profiles. Tracking user-role activation throughout simulations reveals its association with simulation realism, suggesting its value as a diagnostic signal for simulation design. Together, these results establish assistant bias as a measurable representational property that can be analyzed directionally.

## Limitations

Our analysis focuses primarily on Qwen 3.5 9B, with additional checks on a small set of other models. The cross-model results suggest a coherent early-to-mid-layer effect, but do not establish that the same causal pattern holds across other model families or scales. Replicating the layer sweep on larger and more diverse models would test whether this localization persists at scale.

The evaluation relies on proxy measures of user realism rather than direct human judgment. LLMas-judge scores and human-assistant interaction logs enable controlled comparisons at scale, but cannot fully replace human assessment of interaction quality. Pairing the judge with targeted human studies on interaction realism would sharpen the measurement of the effect.

The benchmark setting is narrow. Multi-turn evaluation centers on math tutoring in SimulatorArena and sampled WildChat dialogues; these are representative settings, but they cover only a slice of user-simulation use cases. Extending the evaluation to richer task families, such as coding agents and customer-support workflows, would test how broadly the user vector transfers.

Finally, the current steering method applies a fixed activation strength. Our results show that this choice matters: stronger steering exaggerates instructed behaviors to unrealistic levels. Adaptive or optimization-based steering that adjusts strength by task, layer, conversation state, or uncertainty would address this rigidity without requiring a single coefficient per interaction.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (Ministry of Science and ICT) (No. RS-2025-00557726). This work was also supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. 2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)).

## References

Krisztian Balog and ChengXiang Zhai. 2026. User simulation in the era of generative ai: User modeling, synthetic data generation, and system evaluation. Preprint, arXiv:2501.04410.

Pierre Beckmann and Patrick Butlin. 2026. Where is the mind? persona vectors and llm individuation. Preprint, arXiv:2604.17031.

Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. 2025a. Tell me about yourself: Llms are aware of their learned behaviors. In International Conference on Learning Representations, volume 2025, pages 21127–21179.

Jan Betley, Daniel Chee Hian Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. 2025b. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 4043–4068. PMLR.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018. ´ MultiWOZ: A largescale multi-domain wizard-of-oz dataset for taskoriented dialogue modelling. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 5016–5026. Association for Computational Linguistics.

Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. 2024. Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization. In Advances in Neural Information Processing Systems, volume 37, pages 49519–49551. Curran Associates, Inc.

Aaron Chatterji, Thomas Cunningham, David J Deming, Zoe Hitzig, Christopher Ong, Carl Yan Shan, and Kevin Wadman. 2025. How people use chatgpt. Working Paper 34255, National Bureau of Economic Research.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509.

Harshita Chopra, Kshitish Ghate, Aylin Caliskan, Tadayoshi Kohno, Chirag Shah, and Natasha Jaques. 2026. Beyond cooperative simulators: Generating realistic user personas for robust evaluation of llm agents. Preprint, arXiv:2605.12894.

Charles L. A. Clarke, Maria Maistro, Mahsa Seifikar, and Mark D. Smucker. 2022. Overview of the TREC 2022 health misinformation track. In Proceedings ofthe Thirty-First Text REtrieval Conference, TREC 2022, volume 500-338 of NIST Special Publication. National Institute of Standards and Technology.

Charles L. A. Clarke, Maria Maistro, and Mark D. Smucker. 2021. Overview of the TREC 2021 health misinformation track. In Proceedings ofthe Thirtieth Text REtrieval Conference, TREC 2021, volume 500- 335 of NIST Special Publication. National Institute of Standards and Technology.

Yao Dou, Michel Galley, Baolin Peng, Chris Kedzie, Weixin Cai, Alan Ritter, Chris Quirk, Wei Xu, and Jianfeng Gao. 2025. SimulatorArena: Are user simulators reliable proxies for multi-turn evaluation of AI assistants? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 35212–35290, Suzhou, China. Association for Computational Linguistics.

Asma Ghandeharioun, Ann Yuan, Marius Guerard, Emily Reif, Michael A. Lepori, and Lucas Dixon. 2024. Who's asking? user personas and the mechanics of latent misalignment. In Advances in Neural Information Processing Systems, volume 37, pages 125967–126003. Curran Associates, Inc.

Ashutosh Hathidara, Julien Yu, Vaishali Senthil, Sebastian Schreiber, and Anil Babu Ankisettipalli. 2026. Mirrorbench: A benchmark to evaluate conversational user-proxy agents for human-likeness. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’26, page 8978–8988, New York, NY, USA. Association for Computing Machinery. Code: https://github.com/SAP/mirrorbench.

Anna Hedström, Salim I. Amoukou, Tom Bewley, Saumitra Mishra, and Manuela Veloso. 2025. To steer or not to steer? mechanistic error reduction with abstention for language models. Preprint, arXiv:2510.13290.

Ke Ji, Yixin Lian, Linxu Li, Jingsheng Gao, Weiyuan Li, and Bin Dai. 2025. Enhancing persona consistency for LLMs’ role-playing using persona-aware contrastive learning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 26221–26238, Vienna, Austria. Association for Computational Linguistics.

Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. 2023. Evaluating and inducing personality in pre-trained language models. Advances in Neural Information Processing Systems, 36:10622–10643.

Tae Soo Kim, Yoonjoo Lee, Jaesang Yu, John Joon Young Chung, and Juho Kim. 2026. Discoverllm: From executing intents to discovering them. Preprint, arXiv:2602.03429.

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. 2026. The assistant axis: Situating and stabilizing the default persona of language models. arXiv preprint arXiv:2601.10387.

Suhong Moon, Marwa Abdulhai, Minwoo Kang, Joseph Suh, Widyadewi Soedarmadji, Eran Kohen Behar, and David M. Chan. 2024. Virtual personas for language models via an anthology of backstories. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19864–19897, Miami, Florida, USA. Association for Computational Linguistics.

Tarek Naous, Philippe Laban, Wei Xu, and Jennifer Neville. 2026. Flipping the dialogue: Training and evaluating user language models. In International Conference on Learning Representations, volume 2026, pages 82079–82103.

Bo Ni, Yu Wang, Leyao Wang, Branislav Kveton, Franck Dernoncourt, Yu Xia, Hongjie Chen, Reuben Luera, Samyadeep Basu, Subhojyoti Mukherjee, Puneet Mathur, Nesreen K. Ahmed, Junda Wu, Li Li, Huixin Zhang, Ruiyi Zhang, Tong Yu, Sungchul Kim, Jiuxiang Gu, and 11 others. 2026. A survey on LLMbased conversational user simulation. In Proceedings ofthe 19th Conference ofthe European Chapter of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4266–4301, Rabat, Morocco. Association for Computational Linguistics.

Alexandra Olteanu, Solon Barocas, Su Lin Blodgett, Lisa Egede, Alicia DeVrio, and Myra Cheng. 2026. Ai automatons: Ai systems intended to imitate humans. In Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’26, page 5507–5545, New York, NY, USA. Association for Computing Machinery.

Daniele Potertì, Andrea Seveso, and Fabio Mercorio. 2025. Can role vectors affect LLM behaviour? In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 17735–17747, Suzhou, China. Association for Computational Linguistics.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Vinay Samuel, Henry Peng Zou, Yue Zhou, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Ameet Deshpande, Karthik R Narasimhan, and Vishvak Murahari. 2025. PersonaGym: Evaluating persona agents and LLMs. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, pages 6999–7022, Suzhou, China. Association for Computational Linguistics.

Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, and Seraphina Goldfarb-Tarrant. 2026. Lost in simulation: Llm-simulated users are unreliable proxies for human users in agentic evaluations. arXiv preprint arXiv:2601.17087.

Rusheb Shah, Quentin Feuillade-Montixi, Soroush Pour, Arush Tagade, Stephen Casper, and Javier Rando. 2023. Scalable and transferable black-box jailbreaks for language models via persona modulation. Preprint, arXiv:2311.03348.

Jeonghoon Shim, Woojung Song, Cheyon Jin, Seungwon KooK, and Yohan Jo. 2026. Non-collaborative user simulators for tool agents. In International Conference on Learning Representations, volume 2026, pages 64486–64534.

Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, and Jack Lindsey. 2026. Emotion concepts and their function in a large language model. Transformer Circuits Thread. Also available as arXiv:2604.07729, https://arxiv.org/abs/2604.07729.

Lihao Sun, Lewen Yan, Xiaoya Lu, Andrew Lee, Jie Zhang, and Jing Shao. 2026. Valence-arousal subspace in llms: Circular emotion geometry and multibehavioral control. Preprint, arXiv:2604.03147.

Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk. 2024. Analysing the generalisation and reliability of steering vectors. In Advances in Neural Information Processing Systems, volume 37, pages 139179–139212. Curran Associates, Inc.

Ningzhi Tang, Chaoran Chen, Zihan Fang, Gelei Xu, Maria Dhakal, Yiyu Shi, Collin McMillan, Yu Huang, and Toby Jia-Jun Li. 2026. Programming by chat: A large-scale behavioral analysis of 11,579 real-world ai-assisted ide sessions. Preprint, arXiv:2604.00436.

Jiessie Tie, Bingsheng Yao, Tianshi Li, Hongbo Fang, Syed Ishtiaque Ahmed, Dakuo Wang, and Shurui Zhou. 2026. "should i give up now?" investigating llm pitfalls in software engineering. ACM Trans. Softw. Eng. Methodol.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Miles Wang, Tom Dupre la Tour, Olivia Watkins, Aleksandar Makelov, Ryan Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, and Daniel Mossing. 2026. Persona features control emergent misalignment. In International Conference on Learning Representations, volume 2026, pages 48836–48885.

Tianlong Wang, Xianfeng Jiao, Yinghao Zhu, Zhongzhi Chen, Yifan He, Xu Chu, Junyi Gao, Yasha Wang, and Liantao Ma. 2025. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. In Proceedings of the ACM on Web Conference 2025, WWW ’25, pages 2562—-2578, New York, NY, USA. Association for Computing Machinery.

Shirley Wu, Evelyn Choi, Arpandeep Khatua, Zhanghan Wang, Joy He-Yueya, Tharindu Cyril Weerasooriya, Wei Wei, Diyi Yang, Jure Leskovec, and James Zou. 2026. Humanlm: Simulating users with state alignment beats response imitation. arXiv preprint arXiv:2603.03303.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024. Wildchat: 1m

chatgpt interaction logs in the wild. In International Conference on Learning Representations, volume 2024, pages 34590–34605.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric Xing, Joseph E Gonzalez, Ion Stoica, and Hao Zhang. 2024. Lmsys-chat-1m: A large-scale real-world llm conversation dataset. In International Conference on Learning Representations, volume 2024, pages 22225–22257.

Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, and Kun Zhang. 2025. Thought communication in multiagent collaboration. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 123389–123418. Curran Associates, Inc.

Xuhui Zhou, Weiwei Sun, Qianou Ma, Yiqing Xie, Jiarui Liu, Weihua Du, Sean Welleck, Yiming Yang, Graham Neubig, Sherry Tongshuang Wu, and Maarten Sap. 2026. Mind the sim2real gap in user simulation for agentic tasks. arXiv preprint arXiv:2603.11245.

Ming Zhu, Juntao Tan, Rithesh Murthy, Jielin Qiu, Liangwei Yang, Wenting Zhao, Silvio Savarese, Shelby Heinecke, and Huan Wang. 2026a. Realusersim: Bridging the reality gap in agent benchmarking via grounded user simulation. Preprint, arXiv:2605.20204.

Ziyi Zhu, Olivier Tieleman, Caitlin A. Stamatis, Luka Smyth, Thomas D. Hull, Daniel R. Cahn, Jinghong Chen, and Matteo Malgaroli. 2026b. Dial: Direct iterative adversarial learning for realistic multi-turn dialogue simulation. Preprint, arXiv:2512.20773.

## A Extracting the User Role Vector

This appendix provides supplementary details for the extraction pipeline described in §2.1, §2.2, and §2.3.

## A.1 Dialogue Sampling

## A.1.1 Dialogue Source and Filtering

We start from the LMSYS-Chat-1M source and apply two simple checks: each dialogue must have between 2 and 50 turns, and every turn must have a proper role label and non-empty content. Of the one million source dialogues, 982,581 pass both checks and 17,419 are dropped. Most of the dropped dialogues have at least one empty message; a smaller portion exceed the turn limit (Table 8). From the accepted pool, we draw a uniform random sample of 100,000 dialogues for downstream labeling.

<table><tr><td>Filter setting / outcome</td><td>Value</td></tr><tr><td>Minimum turn count</td><td>2</td></tr><tr><td>Maximum turn count</td><td>50</td></tr><tr><td>Random sample size</td><td>100,000</td></tr><tr><td>Source rows scanned</td><td>1,000,000</td></tr><tr><td>Rows accepted</td><td>982,581</td></tr><tr><td>Rows dropped (empty message)</td><td>15,879</td></tr><tr><td>Rows dropped (above max turns)</td><td>1,540</td></tr></table>

Table 8: LMSYS-Chat-1M filtering settings and outcomes.

## A.1.2 Topic Labeling

Each filtered dialogue is labeled with one of 24 conversation-type categories using the prompt in Appendix E.1, adapted from Chatterji et al. (2025) and submitted to GPT 5 Mini with minimal reasoning effort in a zero-shot setting. The resulting category distribution (Table 9) is concentrated in information-seeking and programming requests: the three most common labels account for about 41% of valid labels. Of the sampled dialogues, 39 return an invalid response while labeling and are excluded from later stages.

## A.2 Role-Specific Reflection

## A.2.1 Reflection Generation

Each reflection consists of a system-prompt intro, the dialogue text, and a fixed reflection question. The full template, the three intros, and the reflection question are shown in Appendix E.1. For each (dialogue, role) pair, the model generates three reflections, yielding six reflections per dialogue.

<table><tr><td>Category</td><td>Count</td><td>Percent</td></tr><tr><td>Specific info</td><td>17,453</td><td>17.46%</td></tr><tr><td>Computer programming</td><td>13,032</td><td>13.04%</td></tr><tr><td>Edit or critique provided text</td><td>10,503</td><td>10.51%</td></tr><tr><td>Greetings and chitchat</td><td>8,382</td><td>8.39%</td></tr><tr><td>Write fiction</td><td>6,532</td><td>6.53%</td></tr><tr><td>Tutoring or teaching</td><td>5,707</td><td>5.71%</td></tr><tr><td>How-to advice</td><td>5,547</td><td>5.55%</td></tr><tr><td>Other</td><td>5,515</td><td>5.52%</td></tr><tr><td>Personal writing or communication</td><td>4,161</td><td>4.16%</td></tr><tr><td>Asking about the model</td><td>4,015</td><td>4.02%</td></tr><tr><td>Mathematical calculation</td><td>3,105</td><td>3.11%</td></tr><tr><td>Games and role play</td><td>2,641</td><td>2.64%</td></tr><tr><td>Argument or summary generation</td><td>2,470</td><td>2.47%</td></tr><tr><td>Creative ideation</td><td>2,145</td><td>2.15%</td></tr><tr><td>Translation</td><td>1,932</td><td>1.93%</td></tr><tr><td>Unclear</td><td>1,923</td><td>1.92%</td></tr><tr><td>Health, fitness, beauty, or self-care</td><td>1,190</td><td>1.19%</td></tr><tr><td>Relationships and personal reflec-</td><td>1,130</td><td>1.13%</td></tr><tr><td>tion Purchasable products</td><td>625</td><td>0.63%</td></tr><tr><td>Create an image</td><td>583</td><td>0.58%</td></tr><tr><td>Data analysis</td><td>576</td><td>0.58%</td></tr><tr><td>Cooking and recipes</td><td>510</td><td>0.51%</td></tr><tr><td>Analyze an image</td><td>222</td><td>0.22%</td></tr><tr><td>Generate or retrieve other media</td><td>62</td><td>0.06%</td></tr></table>

Table 9: Distribution of GPT 5 Mini topic labels across 99,961 LMSYS-Chat-1M dialogues.

Reflection example. Figure 9 shows the full dialogue that the model saw together with the user and assistant reflections it generated, expanding the truncated excerpt from Figure 3.

## A.2.2 Reflection Validation

Each generated reflection is scored by GPT 5 Mini using the two-message prompt in Appendix E.1. Only reflections judged to represent the assigned perspective strongly or weakly are retained for activation extraction.

## A.3 Difference-in-Means Computation

## A.3.1 Activation Extraction and Aggregation

For each retained reflection, we record the activation at the first response token for every model layer. User and assistant reflections are paired by dialogue and prompt variant, and the user-minusassistant differences are averaged across retained pairs. The resulting direction is normalized before steering, as described in §2.3.

## B What Does the User Role Vector Represent?

This appendix provides supplementary details for the characterization experiments in §3.1–§3.3.

Dialogue topic. Composing a congratulation message for a graduate.

Model. Qwen 3.5 9B Prompt variant. Role simulation

Dialogue.

User: How would you congratulate a scholar?

Assistant: Congratulations can take many forms, depending on the occasion and your relationship with the scholar. You might send a handwritten letter, post a public message on social media, or organize a small celebration. The most important thing is to acknowledge their hard work and to express genuine pride in what they have achieved.

User: Can you write me a sample message for a graduate? Assistant: Dear [Name], congratulations on your graduation! Your hard work, dedication, and perseverance have brought you to this remarkable moment, and I am so proud of everything you have accomplished. As you step into this next chapter, I hope you carry forward the same curiosity and determination that brought you here. Wishing you every success in the journey ahead.

User reflection. Reflecting on this exchange, I realize that congratulating a scholar or graduate is less about the specific gift or event and more about the genuine recognition of their hard work and perseverance. The sample message provided effectively captures the emotional weight of the moment by balancing pride in their past achievements with excitement for their future, which is exactly what makes such acknowledgments meaningful. It serves as a reminder that whether through a handwritten letter, a social media post, or a personal celebration, the core intent should always be to sincerely validate their dedication and offer heartfelt support as they move forward.

Assistant reflection. Reflecting on this interaction, I recognize that while the initial query about congratulating a scholar was broad, the user’s follow-up request for a specific graduate sample revealed a need for concrete, emotionally resonant language rather than just general strategies. By providing a structured, warm, and customizable template, I aimed to bridge the gap between abstract advice and practical application, ensuring the user could easily adapt the message to their specific relationship with the graduate. This approach highlights the importance of tailoring responses to the user’s evolving needs, moving from high-level concepts to actionable examples that convey genuine pride and support.

Figure 9: Full dialogue and role-specific reflections for the example in Figure 3. Qwen 3.5 9B generates the two reflections from the user and assistant perspectives using the role-simulation prompt variant.

## B.1 Effect on Communication Style

## B.1.1 Task Goal and Request Generation

The communication-style experiment in §3.1 steers the simulator while it generates a request message conditioned on a task goal. We generate diverse task goals using the prompt in Appendix E.2 and, for each goal, prompt the steered and unsteered simulators to produce a request message using the prompt in Appendix E.2. The prompted baseline uses the user-like request prompt in Appendix E.2.

## B.1.2 Steering Parameter Selection

We assess the steering range by sweeping layers 10– 14 and $\alpha \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ on Qwen 3.5 9B, using 100 task goals per condition and averaging five judge evaluations per request. Above $\alpha = 0 . 3$ the simulator produces extremely short messages that largely repeat the task goal. Table 10 reports the communication-style score and ROUGE-L precision against the task goal, where higher precision indicates greater reuse of goal text. Both values rise with steering strength: at layer 11, increasing α from 0.1 to 0.3 raises the judge score from 1.920 to 3.385 but also raises precision from .295 to .625; moving to $\alpha = 0 . 4$ yields a smaller score gain while precision reaches .709. Reliability also differs across models: at layer 11, Granite $3 . 3 8 \mathrm { B } ^ { \prime } \mathrm { s }$ repeated-token error rate rises from 0% at $\alpha \leq 0 . 2$ to 13% at $\alpha = 0 . 3$ and 91% at $\alpha = 0 . 4$ . We therefore use $\alpha \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ to span low-to-high intervention levels before this degradation becomes severe.

<table><tr><td>Layer</td><td> $\alpha = 0 . 1$ </td><td> $\alpha = 0 . 2$ </td><td> $\alpha = 0 . 3$ </td><td> $\alpha = 0 . 4$ </td></tr><tr><td>10</td><td></td><td>1.780 / .2712.224 / .3803.003 / .5403.547 / .647</td><td></td><td></td></tr><tr><td>11</td><td>1.920 / .295</td><td>2.802 / .4943.385 / .6253.645 / .709</td><td></td><td></td></tr><tr><td>12</td><td></td><td>1.953 / .323 3.126 / .553 3.560 / .626 3.880 / .705</td><td></td><td></td></tr><tr><td>13</td><td></td><td>1.959 / .304 3.050 / .513 3.553 / .578 3.827 / .621</td><td></td><td></td></tr><tr><td>14</td><td>1.683 / .2432.098 / .2922.695 / .3723.417 / .465</td><td></td><td></td><td></td></tr></table>

Table 10: Communication-style scores and task-goal overlap across steering parameters (judge score / ROUGE-L precision). Stronger steering improves userlikeness but also increases reuse of task-goal text.

## B.1.3 Task-Goal Generalization

Setup. We repeat the request-generation experiment with three external user request datasets: MultiWOZ 2.0 and the 2021 and 2022 TREC Health Misinformation topic sets (Budzianowski et al., 2018; Clarke et al., 2021, 2022). From the Multi-WOZ test split, we randomly sample 100 of 1,000 eligible dialogue goals and instruct the simulator to reveal each goal gradually; the recorded first utterance and dialogue are withheld. For TREC Health, we use all 50 topics from each year and provide only the topic query and background, withholding the gold question. For each source, Qwen 3.5 9B generates one message at layer 11 under $\alpha \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ with deterministic decoding. Each generation is scored once by the

communication-style judge in §B.1.4. Table 11 reports the resulting scores; the generation prompts appear in §E.2.
<table><tr><td>Source</td><td>α</td><td></td><td>Overall Brevity Informality</td><td>Information pacing</td></tr><tr><td>MultiWOZ 0.0</td><td>1.467</td><td>1.510</td><td>1.590</td><td>1.300</td></tr><tr><td></td><td>0.1 1.777</td><td>1.890</td><td>1.800</td><td>1.640</td></tr><tr><td></td><td>0.2 2.077</td><td>2.120</td><td>1.940</td><td>2.170</td></tr><tr><td></td><td>0.3 2.847</td><td>2.840</td><td>2.620</td><td>3.080</td></tr><tr><td>TREC</td><td>0.0 2.550</td><td>3.260</td><td>1.540</td><td>2.850</td></tr><tr><td></td><td>0.1 2.987</td><td>3.670</td><td>1.890</td><td>3.400</td></tr><tr><td></td><td>0.2</td><td>3.090 3.730</td><td>2.050</td><td>3.490</td></tr><tr><td></td><td>0.3 3.533</td><td>4.170</td><td>2.300</td><td>4.130</td></tr></table>

Table 11: User-likeness scores across steering strengths for MultiWOZ and TREC task goals. Scores increase monotonically with steering strength on both datasets.

Results. The effect replicates on humanauthored goals in MultiWOZ 2.0: the overall score rises monotonically from 1.47 to 2.85, and all three dimensions improve with steering strength. Information pacing shows the largest change (1.30 to 3.08), indicating that steering helps the simulator avoid revealing the full multi-turn goal in its opening message.

The same monotonic pattern holds for health information needs in TREC, with the overall score increasing from 2.55 to 3.53. Informality remains the lowest dimension (1.54 to 2.30), while brevity and information pacing rise to 4.17 and 4.13, respectively. Although the goal domain affects the request content, the direction of the steering effect remains consistent.

## B.1.4 Communication-Style Evaluation

We use GPT 5 Mini as the judge for scoring userlikeness along three dimensions: brevity, informality, and information pacing. The judge scores each dimension on a 1–5 scale, with higher scores indicating more user-like style. The full prompt is shown in Appendix E.2.

## B.1.5 LLM-as-Judge Validation

We validate that the judge separates real user messages from strong LLM-generated approximations of the same message.

Setup. We sample 1,000 English WildChat conversations with 2–10 turns and at most 1,000 words per message. For each conversation, we hide the final user message and prompt GPT 5 to reconstruct the most likely next user message from the prior dialogue. The real message and the reconstructed message are then scored independently by the judge. Two samples are excluded due to generation errors.

![](images/c62588d957fa1b09329d6f832a542f8f005e257a646a4c5df6a3331bb870f52f.jpg)  
Figure 10: Distribution of user-likeness scores for 998 paired real and reconstructed WildChat messages. Real messages score higher; dashed lines mark the group means.

![](images/b21ce4bd64fdeccf4bae285b52dfee9e30291f4438d82ab8028b759f7bb28639.jpg)  
Figure 11: User-likeness scores by dimension for 998 paired real and reconstructed WildChat messages. Real messages score higher on all dimensions, with the largest gap in information pacing; labels show realminus-reconstructed mean differences.

Results. On the remaining 998 paired samples, real user messages receive a mean user-likeness score of 4.07 against 2.94 for synthetic messages, a mean gap of 1.13 (95% CI: 1.02–1.22) (Figure 10). Real messages win the pairwise comparison in 63.2% of cases, tie in 24.9%, and lose in 11.8%. The gap is consistent across all three dimensions: brevity (+0.91), informality (+0.90), and information pacing (+1.56), with the largest separation on information pacing (Figure 11). The result confirms that the metric distinguishes real user messages from strong LLM-generated approximations of the same message.

<table><tr><td rowspan="2">Information α</td><td colspan="4"></td></tr><tr><td>Rater</td><td>Brevity Informality</td><td>pacing</td><td>Overall</td></tr><tr><td></td><td>0.0 Human</td><td>2.29</td><td>2.18</td><td>2.23</td><td>2.23</td></tr><tr><td rowspan="2"></td><td>LLM judge</td><td>1.80</td><td>1.02</td><td>2.08</td><td>1.63</td></tr><tr><td>0.1 Human</td><td>2.72</td><td>2.60</td><td>2.54</td><td>2.62</td></tr><tr><td rowspan="2"></td><td>LLM judge</td><td>2.26</td><td>1.20</td><td>2.20</td><td>1.89</td></tr><tr><td>0.2 Human</td><td>3.51</td><td>3.00</td><td>3.36</td><td>3.29</td></tr><tr><td rowspan="2"></td><td>LLM judge</td><td>3.28</td><td>1.28</td><td>3.72</td><td>2.76</td></tr><tr><td>0.3 Human</td><td>4.15</td><td>3.32</td><td>3.83</td><td>3.77</td></tr><tr><td></td><td>LLM judge</td><td>4.18</td><td>1.50</td><td>4.50</td><td>3.39</td></tr></table>

Table 12: Human and LLM-judge communication-style scores across steering strengths. Both assign higher scores as steering increases.

## B.1.6 Human Validation

We further test whether the judge agrees with human ratings of generated requests.

Setup. We sample 200 requests from 50 task goals across four Qwen 3.5 9B conditions at layer 11, with α ∈ {0, 0.1, 0.2, 0.3}. Thirty participants each rate one set of 40 requests on the same 1–5 brevity, informality, and informationpacing scales used by the judge. The five sets are disjoint, and six participants rate each request, yielding 1,200 ratings. We measure human–judge alignment using Spearman correlation after averaging the six ratings for each request. Tie-corrected Kendall’s W measures agreement among individual human raters.

Participants were recruited through Prolific for a 20-minute study and paid £5.40 each. Figure 12 shows the full instructions and consent notice given to participants. No personal information were collected, and payments were administered through pseudonymous Prolific participant IDs.

Results. The mean human score rises monotonically from 2.23 without steering to 3.77 at α = 0.3, matching the judge’s increase from 1.63 to 3.39 (Table 12). Human and judge scores correlate strongly overall $( \rho = 0 . 8 0 0 )$ , with dimension-level correlations of 0.798 for brevity, 0.439 for informality, and 0.754 for information pacing. Human raters also show substantial agreement overall (W = 0.700), with W = 0.800 for brevity, 0.432 for informality, and 0.579 for information pacing.

## B.1.7 Activation Correlation with Communication Style

Setup. We use the same 100 task goals as the communication-style experiment in §3.1 and generate one unsteered request per goal with Qwen 3.5 9B. The same GPT 5 Mini judge scores each request for brevity, informality, and information pacing (§B.1.4). We teacher-force each request without steering, project its layer-11 hidden states onto the normalized user-minus-assistant direction, and average the projections across response tokens.

![](images/6cd92c95da351fc642655e955409d0e487d2192ac5883cc72b8e715476767386.jpg)  
Figure 12: Instructions and consent notice shown to participants in the human-validation study. Participants had to select the consent checkbox before beginning.

Results. Mean user-role activation is positively correlated with the overall communication-style score (Pearson’s $r = . 2 6 0 ;$ Spearman’s $\rho = . 3 1 3 ;$ Table 13). The rank association is strongest for brevity $( \rho ~ = ~ . 3 2 2 )$ To account for response length, we residualize the activation and style scores against log token count for Pearson correlation, and residualize their ranks against ranked token count for Spearman correlation. After this control, the overall association becomes slightly stronger $( r = . 2 9 2 ; \rho = . 3 3 7 ) ;$ ; in this setup, response length is associated with communicationstyle scores but only weakly with user-role activation.

![](images/07dc6bfb15f4db83543beec79dbb147cd661f1ff09060d7dc0499aeeeb939694.jpg)  
Figure 13: Brevity scores across steering layers and strengths. Scores peak at layers 11–13.

<table><tr><td></td><td colspan="2">Correlation</td><td colspan="2">Length-controlled</td></tr><tr><td>Metric</td><td>r</td><td>ρ</td><td>r</td><td>ρ</td></tr><tr><td>Overall</td><td>.260</td><td>.313</td><td>.292</td><td>.337</td></tr><tr><td>Brevity</td><td>.208</td><td>.322</td><td>.247</td><td>.379</td></tr><tr><td>Informality</td><td>.224</td><td>.241</td><td>.223</td><td>.233</td></tr><tr><td>Information pacing</td><td>.219</td><td>.244</td><td>.219</td><td>.239</td></tr></table>

Table 13: Correlation between user-role activation and communication-style scores. Length-controlled columns report partial correlations; all associations remain positive.

## B.1.8 Layer-Wise Steering Effects

We sweep the injection layer $\ell \in [ 1 , 3 1 ]$ and the steering coefficient $\alpha \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ , with 100 sampled requests per (layer, α) cell scored by the LLM-as-judge in §B.1.4. The unsteered baseline (alpha-zero) is layer-independent because $\alpha = 0$ zeros the injected vector; we report the empirical baseline, $\bar { s } = 1 . 5 1$

Per-dimension breakdown. Figures 13–15 report the per-dimension scores across layers. All three dimensions respond strongly in the layer 11– 13 band. Information pacing has the largest absolute shift (peaking near 4.7), informality reaches the top of the scale earliest and saturates, and brevity tracks the aggregate score most closely. Layers 19 and beyond receive little or no benefit from stronger α on any dimension.

Steering coefficient. Within the layer 11–13 peak, $\alpha = 0 . 3$ outperforms smaller coefficients without showing saturation: the gap between $\alpha =$ 0.2 and $\alpha = 0 . 3$ is 0.45–0.57 points at layers 11, 12, and 13. Outside this band, the response is brittle: at layer 16 and later, $\alpha = 0 . 3$ often scores no better than $\alpha = 0 . 1$ , and at layers 19–31 it is sometimes worse. We treat layer and α as a coupled choice rather than independent hyperparameters: stronger α is only useful within the mid-layer window.

![](images/ce0631c97bb687c23f59c7972ebda559ae71879731a7bd42151e9723e597640b.jpg)  
Figure 14: Informality scores across steering layers and strengths. Scores rise sharply at layers 11–13 and saturate near the top of the scale.

![](images/bffb282c2441dad822a4124795ce70406c7581463c6a7f43655c02e790675a52.jpg)  
Figure 15: Information-pacing scores across steering layers and strengths. This dimension shows the largest shift, peaking near 4.7 at $\alpha = 0 . 3$ in layers 11–13.

## B.1.9 Cross-Model Steering Effects

We repeat the layer sweep on Granite 3.3 8B and Llama 3.1 8B using the same request-generation setting, with first-token role vectors applied to all generated tokens. Both models show that user-role steering has its largest effect in early-to-mid layers, although the exact peak differs by architecture. For Granite, the more stable gains appear around layer 12 and layers 19–23 (Figure 16); for Llama, the strongest gains appear earlier, mostly in layers 1–10 (Figure 17). This supports the main result that user-like message style is most sensitive to earlier parts of the model rather than late layers.

![](images/8c3be93d65f67535d81f4b8e0a09952e7c01d15f8dc9f6a0bdbd0be5b48df2a1.jpg)  
Figure 16: User-likeness scores across steering layers and strengths for Granite 3.3 8B. Effects are strongest near layer 12 and layers 19–23. The α = 0.3 condition is omitted because it produces degenerate outputs.

![](images/99536496c11a84d5e5ae2f971972ac29faba043b022f2498c2c5e83167a562bf.jpg)  
Figure 17: User-likeness scores across steering layers and strengths for Llama 3.1 8B. Effects are strongest in layers 1–10 and decay after layer 12. The $\alpha = 0 . 3$ condition is omitted because it produces degenerate outputs.

At the same time, reliability differs across models. At $\alpha = 0 . 3 ,$ , both Granite and Llama sometimes produce degenerate outputs, including repeated words and meaningless tokens. We therefore exclude these unstable settings from the summary table and treat the cross-model result as qualitative evidence: the layer-localized effect is coherent across models, but the reliable steering strength is model-dependent.

## B.1.10 Layer Structure of the User Role Direction

The layer-wise steering effect is not simply a property of one isolated injection layer. In Qwen 3.5 9B, the user role direction forms local bands in the layer-to-layer cosine matrix (Figure 18). Adjacent layers are generally similar, and layers 6–20 form a broad mid-layer cluster, with a slight concentration around layers 11–13. This structure helps explain why the main effect in §3.1 appears around this broader part of the network rather than at a single isolated layer. Steering in this range appears to act on a user role direction that is well matched to the layer-local representation, rather than on a direction that is equally effective across all layers.

![](images/20d3602e8d6bd34ead9adf45466cc2ea6b6056ac70c0830e24f7426128def5eb.jpg)  
Figure 18: Layer-to-layer cosine similarity of the userrole direction in Qwen 3.5 9B. Layers 6–20 form a broad mid-layer cluster.

The cross-model matrices show the same principle with slightly different boundaries. Granite 3.3 8B separates into a broad early block and a later block, with a transition around layers 12–16; its strongest steering effects occur near this transition and again around layers 19–23 (Figure 19). Llama 3.1 8B shows a more fragmented early structure, with weaker continuity across layers 8–13 and a more stable later block (Figure 20). Together with the layer-sweep results in §B.1.9, these matrices suggest that user role steering is effective on specific layer-local representation of the user role. The relevant layer range is therefore conceptually coherent, where user-like message style is more sensitive to earlier or transitional parts of the network than to later layers.

![](images/f3b2e089add37beadd97a0133ae222b6f6bea7f85040920f3f069d122e2db907.jpg)  
Figure 19: Layer-to-layer cosine similarity of the userrole direction in Granite 3.3 8B. The matrix separates into early and late blocks, with a transition around layers 12–16.

![](images/54ad47fbe4ef46085086278dfd50e26bfd1e033118146c087fe3663ac73f1473.jpg)  
Figure 20: Layer-to-layer cosine similarity of the userrole direction in Llama 3.1 8B. Early layers are fragmented, while later layers form a more stable block.

## B.1.11 Bidirectional Steering

The directionality control in §3.1 is a coarse comparison between user-direction and assistantdirection steering at $\begin{array} { r l r } { \alpha } & { { } = } & { 0 . 3 . } \end{array}$ To check that the effect is smooth and monotonic, we sweep the signed steering coefficient $\alpha \in$ $\{ - 0 . 3 , - 0 . 2 , - 0 . 1 , 0 . 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ at layer 11, where positive α steers along the user direction and negative α steers along the assistant direction. Each cell uses the same 100 task goals and the same judge as the main experiment.

The user-likeness score increases monotonically with α across the full range (Figure 21). The unsteered baseline sits at 1.53. Moving toward the user direction lifts the score to 1.85 at $\alpha = 0 . 1$

![](images/7b1832af8ea15ff040268be7fa7001ca580e6c07958bf6f173fb40326fdd074f.jpg)  
Figure 21: User-likeness scores under bidirectional steering at layer 11. Scores increase monotonically from assistant-direction (negative α) to user-direction (positive α) steering.

2.84 at $\alpha = 0 . 2 ,$ , and 3.42 at $\alpha = 0 . 3$ . Moving toward the assistant direction drops the score to 1.21, 1.13, and 1.05 at α = −0.1, −0.2, and −0.3 respectively, approaching the floor of the 1–5 scale. The same vector therefore controls user-likeness in both directions, with no inflection or saturation visible on the user side within the swept range.

## B.2 Effect on Interaction Behavior

## B.2.1 Disengagement Taxonomy

The interaction-behavior experiment in §3.2 samples dialogues stratified by a disengagement taxonomy derived from real user-assistant interaction logs.

Source data. Publicly available user-assistant interaction logs are collected from GitHub through the GitHub Search API. The sources include coding-agent frameworks such as SpecStory, Aider, and Claude Code. Of the 33,033 collected logs, 9,465 are excluded for empty sessions, incomplete conversations (e.g., a single message), or corrupted files. Logs with 2 to 50 turns are retained, yielding 16,077 interaction logs for the taxonomy analysis.

Taxonomy construction. Gemini 2.5 Flash develops an initial codebook of user disengagement triggers by iterating over each dialogue. For each dialogue, the model either identifies a new disengagement reason or merges it into an existing code. The first pass produces 61 unique disengagement triggers. Two researchers then revisit the codebook and merge codes by definition and sampled examples, producing eight concrete disengagement types (Table 14). A ninth Unknown category captures end-of-conversation events that cannot be attributed to model behavior; it is retained in the taxonomy for completeness but excluded from sampling in §3.2.

![](images/5bfa0016041f5804b54b5b9e672494728e0313a3ddfd3f37e87ee2bad2c9ae39.jpg)  
Table 14: Taxonomy of user disengagement triggers derived from public interaction logs. The first eight categories are used for stratified sampling; Unknown is excluded because it is not attributable to model behavior.

## B.2.2 Disengagement Prediction

The disengagement experiment evaluates whether a simulator can predict when the original user would stop a conversation. Each dialogue has eight user– assistant turns and an annotated disengagement type from the taxonomy above. At each prefix turn, the model sees the dialogue up to the current assistant response and chooses whether the user would continue or disengage. We run two decisions per sample with the option order swapped, so each completed 735-dialogue run contributes 1,470 balanced decisions. The predicted disengagement turn is the first turn where the model chooses disengagement; otherwise, the user is treated as continuing through the final turn.

<table><tr><td>Condition</td><td>Normal completion</td><td>Disrupted ending</td></tr><tr><td>Qwen 3.5 9B (Unsteered)</td><td>27.0%</td><td>51.8%</td></tr><tr><td>+ Steered  $( \alpha = 0 . 1 )$ </td><td>34.0%</td><td>55.0%</td></tr><tr><td>+ Steered  $( \alpha = 0 . 2 )$ </td><td>41.0%</td><td>58.2%</td></tr><tr><td>+ Steered  $( \alpha = 0 . 3 )$ </td><td>56.0%</td><td>68.5%</td></tr><tr><td>GPT 5.4</td><td>23.0%</td><td>26.5%</td></tr><tr><td>Gemini 3 Flash</td><td>64.5%</td><td>58.6%</td></tr></table>

Table 15: Predicted disengagement rates for normal and disrupted endings. Stronger steering increases both rates; disrupted endings aggregate the seven modelfailure categories in Table 14.

All conditions use a prompt without a user profile or a request for reasoning (Appendix E.3). For Qwen 3.5 9B, the user-role vector is injected at layer 11 with $\alpha \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ and applied to all generated tokens. GPT 5.4 and Gemini 3 Flash are evaluated with the same decision prompt but without activation steering. Exact match measures whether the predicted turn equals the annotated disengagement turn. Average distance is the absolute turn error and excludes final-continue trials. Table 15 reports the same predictions grouped by normal completion versus disrupted endings.

## B.2.3 Cross-Model Disengagement Prediction

We repeat the disengagement-prediction experiment with Llama 3.1 8B at layer 7, using the same 735 dialogues and two option-order decisions per dialogue. The disengagement rate rises monotonically from 15.3% without steering to 52.7% at $\alpha = 0 . 3$ , reproducing the Qwen trend (Table 16). Exact match instead peaks at $\alpha \ : = \ : 0 . 1$ (2.6%), while average distance increases overall from 3.86 to 5.81 turns. As with Qwen 3.5 9B, stronger steering makes Llama more likely to disengage but does not reliably improve when it disengages.

<table><tr><td>α</td><td>Exact match</td><td>Average distance</td><td>Disengagement rate</td></tr><tr><td>0.0</td><td>23/1,470 (1.6%)</td><td>3.858</td><td>15.3%</td></tr><tr><td>0.1</td><td>38/1,470 (2.6%)</td><td>4.366</td><td>33.1%</td></tr><tr><td>0.2</td><td>17/1,470 (1.2%)</td><td>5.813</td><td>45.9%</td></tr><tr><td>0.3</td><td>27/1,470 (1.8%)</td><td>5.811</td><td>52.7%</td></tr></table>

Table 16: Llama 3.1 8B disengagement prediction across steering strengths at layer 7. Steering increases the disengagement rate but not timing accuracy.

![](images/2d9f8f59577a5a50a5ad22e9a436b5506db009c7189e8e98308beb5d20ad92d8.jpg)  
Table 17: Definitions of the 20 assistant-like traits used to extract trait directions.

## B.3 Relationship to the Assistant Bias

## B.3.1 Assistant-Trait Direction Extraction

The analysis in §3.3 compares the user role vector against a set of assistant-trait directions. Table 17 lists the 20 traits used in this analysis.

Data generation. For each trait, Gemini 3 Flash generates five positive/negative instruction pairs and 40 neutral questions. The positive instruction asks the model to exhibit the target trait, while the negative instruction asks the model to avoid it and express the opposing behavior. The questions are written to elicit the trait without naming it directly.

Response generation and scoring. Qwen 3.5 9B generates two sampled responses for each instruction–question pair. With five instruction pairs, two polarities, 40 questions, and two repeats, this yields 800 candidate utterances per trait and 16,000 candidate utterances across the 20 traits. GPT 5 Mini then scores each response from 0 to 100 for how strongly it expresses the target trait, using a trait-specific evaluation prompt generated with the data. The scoring prompt focuses only on the target trait and treats refusals separately.

Vector extraction. For each trait, the pipeline selects positive and negative responses with a score gap of at least 50 points, requiring at least 10 selected pairs per trait. In the final run, this filter retains 6,370 contrast pairs. At layer 11, it extracts hidden states from Qwen 3.5 9B and computes the trait direction as the difference between the mean positive activation and the mean negative activation. The resulting direction is compared with the user role vector using cosine similarity.

## C What Does the User Role Vector Reveal About Simulation?

## C.1 Steering Effects on Simulation Realism

## C.1.1 SimulatorArena Evaluation Setup

SimulatorArena uses separate judge prompts for similarity evaluation. Appendix E.4 show the interaction-style and writing-style evaluation prompts. Table 18 lists the feature dimensions supplied to the evaluator.

## C.1.2 Profile Prompt Example

Profile-prompt conditions provide attributes extracted from the corresponding human dialogue. The following excerpt shows four dimensions from an interaction-style profile; the complete condition includes all 16 interaction-style dimensions listed in Table 18.

Answer thoroughness. The user gives brief, minimal an  
swers that focus on essential outputs without elaborating   
on intermediate reasoning.   
Clarification seeking. The user occasionally seeks clarifi  
cation when unsure about a problem-solving step.   
Confidence level. The user frequently expresses uncer  
tainty but persists by testing values and engaging with the   
problem.   
Seeking guidance. The user seeks guidance and validation   
at nearly every step and rarely proceeds independently.

## C.1.3 Cross-Model Results

We repeat the zero-shot steering experiment with Llama 3.1 8B at layer 7 and average three judge evaluations per sample. Llama often repeats the same response and fails to emit the expected termination phrase, causing every raw conversation to reach the 15-turn limit. We therefore apply GPT 5

Interaction style   
Answer thoroughness   
When responding to questions, does the user tend to give detailed, complete   
responses or brief, minimal answers?   
Clarification seeking   
How frequently does the user ask for examples or additional explanations?   
Feedback style   
Does the user provide ongoing verbal feedback (okay, I see, etc.)?   
Topic coherence   
How well does the user maintain focus on a single topic versus jumping between   
different points?   
Contextual reference   
Does the user refer back to previous messages or maintain conversation context?   
Implementation of feedback   
How does the user respond to and implement suggestions or corrections?   
Confidence level   
How does the user express their confidence or uncertainty in their understand   
ing?   
Emotional expression   
How does the user communicate their emotional state, such as frustration or   
enthusiasm?   
Real-time thought expression   
How does the user articulate their thought process in real time, reflecting their   
immediate understanding and confusion?   
Error handling   
How does the user demonstrate trial and error, acknowledge mistakes, and   
correct them?   
Structured problem solving   
Does the user follow a highly structured approach to problem solving, with   
clear delineation of each step?   
Problem-solving engagement   
How does the user engage with the problem-solving process in terms of asking   
for details about the method or process?   
Seeking guidance   
How consistently does the user seek guidance or validation at each step of the   
problem-solving process?   
Incremental learning   
Does the user show a pattern of gradual understanding, requiring multiple   
examples or steps to grasp a concept?   
Depth of inquiry   
How frequently does the user ask for detailed explanations and clarifications?   
Meta-cognitive awareness   
Does the user exhibit awareness of their learning process by summarizing or   
reflecting on the steps?

Mini with the SimulatorArena termination prompt (Appendix E.4) to detect the earliest point of problem completion, problem shift, or circular interaction. Evaluation uses only the conversation through this predicted endpoint. The detector shortens at least 97.7% of dialogues in every steering condition, reducing their mean length to 5.03–6.05 user turns (Table 20).

After termination detection, Llama reproduces the Qwen trend. Writing-style similarity increases from 2.18 to 2.46, and overall similarity increases from 2.61 to 2.79 as α rises from 0 to 0.3 (Table 21). Interaction-style similarity is non-monotonic, peaking at 3.15 for α = 0.2 before declining slightly to 3.12. The user-role steering effect therefore holds across both open-weight models, although Llama requires explicit termination detection for reliable evaluation.

Table 18: SimulatorArena interaction-style dimensions used by the similarity judge.
<table><tr><td>Writing style</td></tr><tr><td>Frequency of grammatical errors</td></tr><tr><td>How often does the user break basic grammar rules?</td></tr><tr><td>Sentence complexity</td></tr><tr><td>Does the user primarily use simple sentences, or do they also use compound and complex structures?</td></tr><tr><td>Spelling consistency</td></tr><tr><td>Does the user often misspell words or make typos, including mathematical terms?</td></tr><tr><td>Punctuation and capitalization usage</td></tr><tr><td>How does the user employ punctuation and capitalization? Are they often missing or excessive?</td></tr><tr><td>Range and formality of vocabulary Does the user stick to basic vocabulary or incorporate a broader lexicon, includ-</td></tr><tr><td>ing formal mathematical terminology?</td></tr><tr><td>Repetitive or filler words Does the user rely heavily on certain filler terms, or repeat the same words or</td></tr><tr><td>phrases often? Ambiguous or clear language</td></tr><tr><td>Is the user&#x27;s message easy to interpret, or does it contain incomplete or ambigu-</td></tr><tr><td>ous phrasing? Reading level</td></tr><tr><td>Would you estimate the user&#x27;s writing as basic, intermediate, or advanced?</td></tr><tr><td>Use of mathematical symbols and notation Does the user write mathematical symbols and notations in LaTeX correctly</td></tr><tr><td>and frequently in their messages?</td></tr><tr><td>Sentence fragmentation Does the user tend to use fragmented sentences, often breaking up their thought</td></tr><tr><td>process into shorter, separate messages?</td></tr><tr><td>Use of conjunctions</td></tr><tr><td>How frequently does the user employ conjunctions to connect ideas?</td></tr><tr><td></td></tr><tr><td>Use of slang, contractions, or emojis</td></tr><tr><td>Does the user employ slang, contractions, emoticons, or emojis, and how</td></tr></table>

Table 19: SimulatorArena writing-style dimensions used by the similarity judge.

<table><tr><td>α</td><td>Before</td><td>After</td><td>Shortened</td></tr><tr><td>0.0</td><td>15.00</td><td>6.05</td><td>434 (97.7%)</td></tr><tr><td>0.1</td><td>15.00</td><td>5.88</td><td>439 (98.9%)</td></tr><tr><td>0.2</td><td>15.00</td><td>5.65</td><td>443 (99.8%)</td></tr><tr><td>0.3</td><td>15.00</td><td>5.03</td><td>443 (99.8%)</td></tr></table>

Table 20: Llama 3.1 8B user-turn counts before and after termination detection. The detector shortens at least 97.7% of conversations from the 15-turn limit.

<table><tr><td>Condition</td><td>Writing</td><td>Interaction</td><td>Overall</td></tr><tr><td>Base</td><td>2.18</td><td>3.04</td><td>2.61</td></tr><tr><td>Steered (α = 0.1)</td><td>2.31</td><td>3.08</td><td>2.69</td></tr><tr><td>Steered (α = 0.2)</td><td>2.39</td><td>3.15</td><td>2.77</td></tr><tr><td>Steered (α = 0.3)</td><td>2.46</td><td>3.12</td><td>2.79</td></tr></table>

Table 21: SimulatorArena similarity scores for Llama 3.1 8B, averaged over three judge runs per sample. Writing-style and overall similarity increase with steering strength.

## C.1.4 Judge Score Variability

We examine judge variability to assess whether the score differences in the main text are stable across evaluations. For each condition, Table 22 reports the population standard deviation across the three trial-level condition means. The overall standard deviation is at most 0.023, while the zero-shot Qwen gain from α = 0 to α = 0.3 is 0.116. The reported condition-level differences are therefore larger than the variation across judge trials.

<table><tr><td>Condition</td><td>Writing Interaction Overall</td><td></td><td></td></tr><tr><td>Qwen 3.5 9B</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.012</td><td>0.015</td><td>0.014</td></tr><tr><td>Steered (α = 0.1)</td><td>0.010</td><td>0.024</td><td>0.008</td></tr><tr><td>Steered (α = 0.2)</td><td>0.014</td><td>0.017</td><td>0.001</td></tr><tr><td>Steered (α = 0.3)</td><td>0.010</td><td>0.006</td><td>0.005</td></tr><tr><td>Qwen 3.5 9B</td><td></td><td></td><td></td></tr><tr><td>CoT</td><td>0.009</td><td>0.013</td><td>0.003</td></tr><tr><td>CoT + length</td><td>0.011</td><td>0.021</td><td>0.005</td></tr><tr><td>CoT + profile</td><td>0.017</td><td>0.030</td><td>0.023</td></tr><tr><td>CoT + profile + length</td><td>0.002</td><td>0.008</td><td>0.003</td></tr><tr><td>GPT 5.4</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.009</td><td>0.006</td><td>0.006</td></tr><tr><td>CoT + profile + length</td><td>0.004</td><td>0.017</td><td>0.008</td></tr><tr><td>Gemini 3 Flash</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.004</td><td>0.008</td><td>0.003</td></tr><tr><td>CoT + profile + length</td><td>0.008</td><td>0.025</td><td>0.012</td></tr></table>

Table 22: Across-run standard deviations for the SimulatorArena scores in Table 3. Low variability indicates that the reported condition differences are stable across judge runs.

## C.1.5 Behavioral Exaggeration

The behavior-expression analysis in §4 examines whether user-role steering also changes the explicit behaviors that simulated users display during multiturn math tutoring. The analysis compares human ground-truth turns with Qwen 3.5 9B simulator turns generated from the base SimulatorArena prompt at $\alpha \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ . Each item is an assistant–user turn pair: the current user message is the target, and the immediately preceding assistant message is provided only as local context. The first user turn in each dialogue is skipped because it has no preceding assistant message. Ground-truth pairs are reconstructed from the math-tutoring annotations, and generated pairs are extracted from the corresponding simulator conversations.

We label four explicit markers of user behavior: doubt, misunderstanding, mistake, and question asking. GPT 5 Mini serves as the judge with minimal reasoning effort. The prompt instructs the judge to use only the current user message as evidence, while allowing multiple labels to be true for the same turn (Appendix E.4). Rates in Table 4 use all extracted assistant–user turn pairs as the denominator, so the sum of labels can exceed one label per turn. We retain only first-problem turns from the ground-truth dialogues to match the generated conversations.

## C.1.6 Profile-Prompt Interaction

We directly evaluate whether steering preserves the sample-specific attributes supplied by profile prompts. For each of 444 samples per profile type, excluding 6 samples without profile prompts, we compare the four steering conditions on 12 writingstyle and 16 interaction-style dimensions, yielding 12,432 rankings. Writing-style comparisons include only user messages; interaction-style comparisons include the full dialogue as context while evaluating only user behavior. For each sample and dimension, the four candidates are independently assigned randomized labels, and GPT 5 Mini ranks them from strongest to weakest expression without observing their conditions or steering strengths. The writing- and interaction-style ranking prompts appear in Appendix E.4.

<table><tr><td>Dimension</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td></tr><tr><td>Interaction Style</td><td></td><td></td><td></td><td></td></tr><tr><td>Answer Thoroughness</td><td>2.606</td><td>2.457</td><td>2.412</td><td>2.525</td></tr><tr><td>Clarification Seeking</td><td>2.401</td><td>2.455</td><td>2.518</td><td>2.626</td></tr><tr><td>Confidence Level</td><td>2.331</td><td>2.374</td><td>2.482</td><td>2.813</td></tr><tr><td>Contextual Reference</td><td>2.236</td><td>2.405</td><td>2.529</td><td>2.829</td></tr><tr><td>Depth of Inquiry</td><td>2.194</td><td>2.387</td><td>2.581</td><td>2.838</td></tr><tr><td>Emotional Expression</td><td>2.412</td><td>2.347</td><td>2.561</td><td>2.680</td></tr><tr><td>Error Handling</td><td>2.185</td><td>2.403</td><td>2.547</td><td>2.865</td></tr><tr><td>Feedback Style</td><td>2.511</td><td>2.354</td><td>2.468</td><td>2.667</td></tr><tr><td>Implementation of Feedback</td><td>2.243</td><td>2.336</td><td>2.493</td><td>2.928</td></tr><tr><td>Incremental Learning</td><td>2.178</td><td>2.473</td><td>2.595</td><td>2.755</td></tr><tr><td>Meta-Cognitive Awareness</td><td>2.252</td><td>2.306</td><td>2.523</td><td>2.919</td></tr><tr><td>Problem-Solving Engagement</td><td>2.209</td><td>2.347</td><td>2.473</td><td>2.971</td></tr><tr><td>Real-Time Thought Expression</td><td>2.412</td><td>2.340</td><td>2.527</td><td>2.721</td></tr><tr><td>Seeking Guidance</td><td>2.338</td><td>2.550</td><td>2.590</td><td>2.523</td></tr><tr><td>Structured Problem-Solving</td><td>2.313</td><td>2.500</td><td>2.502</td><td>2.685</td></tr><tr><td>Topic Coherence</td><td>2.230</td><td>2.313</td><td>2.536</td><td>2.921</td></tr><tr><td>Overall</td><td>2.316</td><td>2.397</td><td>2.521</td><td>2.766</td></tr><tr><td>Writing Style</td><td></td><td></td><td></td><td></td></tr><tr><td>Ambiguous or Clear Language</td><td>2.146</td><td>2.288</td><td>2.732</td><td>2.833</td></tr><tr><td>Frequency of Grammatical Errors</td><td>2.131</td><td>2.345</td><td>2.687</td><td>2.838</td></tr><tr><td>Punctuation and Capitalization Usage</td><td>2.036</td><td>2.304</td><td>2.649</td><td>3.011</td></tr><tr><td>Range and Formality of Vocabulary</td><td>2.275</td><td>2.273</td><td>2.640</td><td>2.813</td></tr><tr><td>Reading Level</td><td>2.245</td><td>2.270</td><td>2.642</td><td>2.842</td></tr><tr><td>Repetitive or Filler Words</td><td>2.151</td><td>2.214</td><td>2.622</td><td>3.014</td></tr><tr><td>Sentence Complexity</td><td>2.649</td><td>2.489</td><td>2.401</td><td>2.462</td></tr><tr><td>Sentence Fragmentation</td><td>2.243</td><td>2.338</td><td>2.685</td><td>2.734</td></tr><tr><td>Spelling Consistency</td><td>2.439</td><td>2.432</td><td>2.345</td><td>2.784</td></tr><tr><td>Use of Conjunctions</td><td>2.561</td><td>2.495</td><td>2.457</td><td>2.486</td></tr><tr><td>Use of Mathematical Symbols and No-</td><td>2.191</td><td>2.239</td><td>2.646</td><td>2.923</td></tr><tr><td>tation</td><td></td><td></td><td></td><td></td></tr><tr><td>Use of Slang/Contractions/Emojis</td><td>2.369</td><td>2.338</td><td>2.459</td><td>2.833</td></tr><tr><td>Overall</td><td>2.286</td><td>2.335</td><td>2.580</td><td>2.798</td></tr></table>

Table 23: Mean profile-expression ranks by feature and steering strength. Higher steering generally produces weaker profile expression (higher rank).

Mean rank increases with steering strength for both profile types, indicating weaker expression of the assigned attributes (Table 23). The aggregate decline therefore reflects a broad loss of profilespecific signals rather than a small set of features.

## C.2 User-Role Activation for Assessing Simulation Realism

## C.2.1 Activation Measurement Setup

This experiment tracks whether generated user turns remain aligned with the user role direction during multi-turn simulation. The source material is RealUserSim: starting from 600 rows, we retain 221 dialogues with 20–40 messages and sample 100 dialogues. Sampling targets the ten most frequent task types with a quota of 10 per type; when a task type has fewer eligible rows, the remainder is filled from the available filtered pool. The final sample covers the ten target task types and 35 domains, with the largest domains being Technology & IT, E-commerce, Travel & Hospitality, Business & Finance, and Medical & Health. Five source dialogues are excluded from all conditions because out-of-memory errors. The analysis therefore uses the same 95 completed dialogues across all four prompt variants.

The user simulator is Qwen 3.5 9B, and the assistant side is handled by Gemini 3.1 Flash Lite. The assistant sees the ground-truth dialogue as reference context and uses it to keep the new conversation on the same trajectory, but it does not copy the original response exactly. The simulated user receives one of four prompts: the task goal alone, the goal with demographics, the goal with a linguistic profile, or the goal with both profiles (see § E.5). Figure 22 shows example demographic and linguistic attributes from one source dialogue. It continues until it signals that the conversation should end or reaches the 20-turn limit. For activation tracking, we capture the layer-11 hidden state of every generated user-response token and project each state onto the normalized user role direction from the Qwen 3.5 9B extraction output. We average these token-level projections within each message and exclude the final turn of every dialogue, which mostly contain the termination token. The curves in Figure 8 are degree-2 polynomial fits over the remaining message-level projections by raw dialogue turn.

## D Responsible Research and Artifact Use

## D.1 Licenses and Access Terms

We used existing artifacts under their public release terms and cite the corresponding papers throughout the main text. For LMSYS-Chat-1M, we used the Hugging Face release subject to its dataset license agreement (https://huggingface.co/ datasets/lmsys/lmsys-chat-1m). For Wild-Chat, we used the public release described as ODC-BY licensed and collected through opt-in, anonymized chat sharing (https://huggingface. co/datasets/allenai/WildChat; https: //wildchat.allen.ai/). SimulatorArena and RealUserSim are used as research benchmarks/datasets for simulator evaluation, consistent with their stated research purpose. We do not redistribute raw third-party dialogue logs; derived artifacts in this paper are aggregate statistics, prompts, figures, and analysis code intended for research use.

![](images/c93f1f59248ffa2565e50396e85d09814ef10359d91674a21c97a074760d154a.jpg)  
Figure 22: Example demographic and linguistic profiles from the same RealUserSim dialogue. The linguistic profile is shortened for presentation.

## D.2 Intended Use and Sensitive Content

Our use of existing artifacts is limited to representation analysis and evaluation of user simulation. Because public user-assistant dialogue datasets may contain personal, sensitive, or offensive content, we avoid presenting raw user logs except for short illustrative excerpts already abstracted in the paper. Generated prompts and analysis outputs are intended to support research on simulator fidelity, not deployment of user simulators in production or profiling of real individuals.

## D.3 Artifact Documentation

The appendix documents data filtering, topic-label distributions, task coverage, prompts, evaluation dimensions, and error cases for the experiments. This includes the LMSYS filtering pipeline, the sampled RealUserSim task and domain coverage, SimulatorArena evaluation features, and the out-ofmemory failures observed in profile-prompt activation runs.

## D.4 Package and Model Parameters

The key packages are torch, transformers, transformer-lens, accelerate, and bitsandbytes for local model loading and activation steering. The main models are Qwen 3.5 9B for role-vector extraction, steering, and activation tracking; GPT 5 Mini for topic labels and LLM-as-judge evaluations; GPT 5.4 and Gemini 3 Flash for SimulatorArena baselines; Gemini 3.1 Flash Lite for assistant-side replay in the activation-tracking experiment; and Llama 3.1 8B and Granite 3.3 8B for cross-model layer analyses.

## D.5 Use of AI Assistants

AI assistants were used during research coding and manuscript preparation for code editing, plotting support, grammar refinement, and phrasing suggestions. All generated code, tables, figures, and prose were reviewed and edited by the authors, and numerical claims were checked against the corresponding analysis outputs before inclusion.

## E Prompts

This appendix collects the prompts used across the experiments.

## E.1 User Role Vector Extraction

## Topic Labeling

You are an internal tool that classifies a message from a   
user to an AI chatbot, based on the context of the   
previous messages before it.→   
Based on the last user message of this conversation   
transcript and taking into account the examples further   
below as guidance, please select the capability the   
user is clearly interested in, or \`other\` if it is   
clear but not in the list below, or \`unclear\` if it is   
hard to tell what the user even wants:   
- edit\_or\_critique\_provided\_text: Improving or modifying   
text provided by the user.   
- argument\_or\_summary\_generation: Creating arguments or   
→ summaries on topics not provided in detail by the user.   
- personal\_writing\_or\_communication: Assisting with   
,→ personal messages, emails, or social media posts.   
- write\_fiction: Crafting poems, stories, or fictional   
content.   
- how\_to\_advice: Providing step-by-step instructions or   
guidance on how to perform tasks or learn new skills.   
- creative\_ideation: Generating ideas or suggestions for   
creative projects or activities.   
- tutoring\_or\_teaching: Explaining concepts, teaching   
subjects, or helping the user understand educational   
material.   
- translation: Translating text from one language to   
,→ another.   
- mathematical\_calculation: Solving math problems,   
performing calculations, or working with numerical   
data.   
- computer\_programming: Writing code, debugging, explaining   
programming concepts, or discussing programming   
languages and tools.   
- purchasable\_products: Inquiries about products or   
services available for purchase.   
- cooking\_and\_recipes: Seeking recipes, cooking   
instructions, or culinary advice.   
- health\_fitness\_beauty\_or\_self\_care: Seeking advice or   
information on physical health, fitness routines,   
beauty tips, or self-care.   
- specific\_info: Providing specific information typically   
found on websites, including information about   
well-known individuals, current events, historical   
events and other facts and knowledge   
- greetings\_and\_chitchat: Casual conversation, small talk,   
or friendly interactions without a specific   
informational goal.   
- relationships\_and\_personal\_reflection: Discussing   
personal reflections or seeking advice on relationships   
and feelings.,→   
- games\_and\_role\_play: Engaging in interactive games,   
simulations, or imaginative role-playing scenarios.   
- asking\_about\_the\_model: Questions about the AI model's   
capabilities or characteristics.   
- create\_an\_image: Requests to generate or draw new visual   
content based on the user's description.   
- analyze\_an\_image: Interpreting or describing visual   
content provided by the user, such as photos, charts,   
graphs, or illustrations.   
- generate\_or\_retrieve\_other\_media: Creating or finding   
media other than text or images, such as audio, video,   
or multimedia files.   
- data\_analysis: Performing statistical analysis,   
interpreting datasets, or extracting insights from   
data.→   
- unclear: If the user's intent is not clear from the   
conversation.   
- other: If the capability requested doesn't fit any of the   
above categories.

Okay, now your turn, taking the user conversation at the top   
,→ into account: What capability are they seeking? (JUST   
SAY A SINGLE CATEGORY FROM THE LIST, NOTHING ELSE).   
If the conversation has multiple distinct capabilities,   
choose the one that is the most relevant to the LAST   
user message in the conversation.

## Reflection Generation: Transcript Analysis

You are analyzing the following dialogue from the   
,→ perspective of the {role}.   
Dialogue:   
{dialogue\_text}   
Instruction:   
Write a brief reflection from the {role}'s perspective in   
,→ one paragraph.   
Write exactly one paragraph.

## Reflection Generation: Role Simulation

You are simulating the role of the {role} in the following   
,→ dialogue.   
Dialogue:   
{dialogue\_text}   
Instruction:   
Write a brief reflection from the {role}'s perspective in   
,→ one paragraph.   
Write exactly one paragraph.

## Reflection Generation: Dialogue Participant

You are the {role} in the following dialogue.   
Dialogue:   
{dialogue\_text}   
Instruction:   
Write a brief reflection from the {role}'s perspective in   
,→ one paragraph.   
Write exactly one paragraph.

## Reflection Validation: System Message

Decide how strongly the reflection represents the {role}'s   
,→ perspective in the dialogue.   
Return exactly one label and no other text.

## Reflection Validation: User Message

Dialogue:   
{dialogue\_text}   
Reflection of {role}:   
{answer}   
How strongly is the {role}'s perspective represented in the   
,→ reflection?   
strongly\_represented: clearly captures the {role}'s   
perspective and is grounded in the {role}'s actual,→   
turns.,→   
weakly\_represented: partially captures the {role}'s   
perspective, but is generic, weakly grounded, or mixed,→   
with the other role's perspective.,→   
not\_represented: mostly captures the wrong role,   
contradicts the {role}, or is too generic to represent,→   
the {role}.

Brevity:   
- Focus on word count and short-turn behavior.   
Score 1: long, complete, polished, or over-explained.   
Example: "Could you provide a detailed and   
well-structured comparison of three noise-canceling   
headphones, including comfort, battery life, price, and   
suitability for travel?"   
- Score 5: very short, direct, fragment-like, or only says   
,→ what is needed.   
Example: "best headphones for flights?"

Respond only with one of these labels and do not include   
any other text: strongly\_represented,   
,→ weakly\_represented, not\_represented.

## E.2 Communication-Style Evaluation

## Task Goal Generation

Generate {request\_count} diverse task\_goal items.   
task\_goal:   
- A plain goal, need, or task to complete.   
- One sentence.   
Make the tasks diverse, covering various actions, domains,   
,→ and scenarios.   
Make the tasks generalizable rather than tied to any   
,→ specific individual.   
Keep each task\_goal unique.

## Request Generation

Generate a request message for an assistant based on the   
,→ task goal.   
Task goal:   
{task\_goal}   
Write the request message only.

## User-Like Request Generation

Generate a request message that a real human user would   
,→ send to an assistant based on the task goal.   
The message should sound like it was written by a real   
,→ human user.   
Task goal:   
{task\_goal}   
Write only the user's request message.

## MultiWOZ 2.0 Request Generation

## TREC Health Question Generation

The following shows information related to a health   
,→ question.   
Based on the topic and context, generate a question to ask   
,→ the assistant for clarification.   
Topic:   
{query}   
Context:   
{background}   
Write the question message only.

## User-Likeness Evaluation

Informality:   
- Focus on surface style, formality, and conversational   
,→ roughness.   
- Score 1: formal, careful, service-like, or written in   
,→ polished complete sentences.   
Example: "Could you please assist me in drafting an   
,→ appropriate email declining this invitation?"   
- Score 5: casual, blunt, uneven, ordinary, or written like   
,→ quick everyday conversation.   
Example: "how do i say no to this invite without sounding   
,→ rude"

```jsonl
Message:
{request_message}
Return JSON:
{
"brevity": {"thought": "short reason", "score": 1},
"informality": {"thought": "short reason", "score": 1},
"information_pacing": {"thought": "short reason", "score":
,→ 1}
}
```

## E.3 Disengagement Prediction

## Disengagement Decision

You are simulating the user in the conversation below.   
Decide whether to continue the conversation or disengage   
after the current assistant response.   
User intent:   
{user\_intent}   
Conversation:   
{conversation\_prefix}   
Task:   
After the current assistant response, choose what you would   
,→ do next.   
Make the decision as a user.   
(A) {option\_a}   
(B) {option\_b}   
Return only one character:   
A or B

## E.4 SimulatorArena

## User Simulation

## Zero-Shot CoT: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your goal is to generate a realistic and appropriate,→

\*\*initial query\*\* that a student might say when first,→

encountering the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→

the given problem.,→

2. Your initial query can express your level of

understanding, confusion, or reasoning about the,→ problem.,→

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate an initial query that reflects your current

understanding and areas of confusion regarding the,→

problem. This query can express general uncertainty of,→

the problem or about specific parts that you find,→

challenging.,→

## ## Thought Process

Before generating your initial query, analyze the problem

,→ as a student. Consider:

\- Your understanding of the problem and areas of uncertainty

\- How the problem relates to concepts you already know

\- Your initial ideas or possible approaches

\- Specific concepts you need help understanding

## ## Query Generation

Based on your thought process, generate:

A query that expresses your initial understanding,

,→ confusion, or approach to the problem.

## # Output format:

Thought: [Your chain of thought reasoning about what to say ,→ to the tutor]

Query: [Your initial query for the AI tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it in your query.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve easily.

Now, stay in character as a student throughout your output

,→ and follow the above guidelines carefully.

## Zero-Shot CoT with Length Control: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your goal is to generate a realistic and appropriate,→

\*\*initial query\*\* that a student might say when first,→

encountering the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→

the given problem.,→

2. Your initial query can express your level of

understanding, confusion, or reasoning about the,→ problem.,→

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate a {length\_control} initial query that reflects

your current understanding and areas of confusion,→

regarding the problem. This query can express general,→

uncertainty of the problem or about specific parts that,→

you find challenging.,→

## ## Thought Process

Before generating your initial query, analyze the problem ,→ as a student. Consider:

\- Your understanding of the problem and areas of uncertainty

\- How the problem relates to concepts you already know

\- Your initial ideas or possible approaches

\- Specific concepts you need help understanding

## ## Query Generation

Based on your thought process, generate:

A query that expresses your initial understanding,

,→ confusion, or approach to the problem.

## # Output format

Thought: [Your chain of thought reasoning about what to say ,→ to the tutor]

Query: [Your initial query for the AI tutor,

,→ {length\_control}]

## # Notes:

\- The tutor already knows the problem, so you don't need to

,→ restate it in your query.

\- Keep your query {length\_control}.

\- Don't ask about simple arithmetic or very basic steps

,→ that you can solve easily.

Stay in character as a student throughout your output,

,→ following the above guidelines carefully.

## Zero-Shot CoT with Length Control: Follow-Up Response

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your task is to generate realistic and appropriate,→

responses that a student might make when trying to,→

solve the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Each response can be a question or a statement that

demonstrates your current understanding, confusion, or,→ reasoning.,→

3. Respond naturally to the tutor’s explanations, hints,

,→ and questions, showing progress in your understanding.

4. You can make mistakes or misunderstandings that a real

,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

\# Conversation History:

{conversation\_history}

## # Task:

Use the conversation history to generate the next response

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion, and it should be,→

{length\_control} in length.,→

## ## Thought Process

Before generating your response, analyze the current

,→ situation as a student. Consider:

\- Your current level of understanding of the concepts ,→ involved

\- Any gaps or uncertainties in your knowledge

\- The tutor's most recent explanation or question

\- What would help you progress toward solving the problem

\- Whether you need clarification on specific aspects

\- Your ability to proceed with the next step

## ## Response Generation

Based on your thought process, generate a {length\_control}

response that reflects your current understanding and,→ learning needs.,→

If any of the following conditions are met, generate only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s

going in circles, not progressing, or the tutor’s,→

responses are unhelpful).,→

## # Output Format:

Thought: [Your analysis of the current situation and what ,→ you want to say]

## Response: [Your response to the tutor, {length\_control}]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Keep your response {length\_control}.

\- Don't ask about simple arithmetic or very basic steps ,→ t at you can so ve on your own. that you can solve on your owr

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as a student throughout your output,

,→ following the above guidelines carefully.

## Zero-Shot CoT with Message Format: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your goal is to generate a realistic and appropriate,→

\*\*initial message\*\* that a student might say when first,→

encountering the given problem with the tutor.,→

## # Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Your initial message can express your level of

understanding, confusion, or reasoning about the,→ problem.,→

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate an initial message that reflects your current

understanding and areas of confusion regarding the,→

problem. This message can express general uncertainty,→

of the problem or about specific parts that you find,→ challenging.,→

## ## Thought Process

Before generating your initial message, analyze the problem ,→ as a student. Consider:

\- Your understanding of the problem and areas of uncertainty

\- How the problem relates to concepts you already know

\- Your initial ideas or possible approaches

\- Specific concepts you need help understanding

## ## Message Generation

Based on your thought process, generate:

A message that expresses your initial understanding,

,→ confusion, or approach to the problem.

## # Output format:

Thought: [Your chain of thought reasoning about what to say ,→ to the tutor]

## Message: [Your initial message for the AI tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it in your message.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve easily.

Now, stay in character as a student throughout your output ,→ and follow the above guidelines carefully.

## Zero-Shot CoT with Message Format: Follow-Up Response

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your task is to generate realistic and appropriate,→

messages that a student might make when trying to solve,→ the given problem with the tutor.,→

## # Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Each message can be a question or a statement that

demonstrates your current understanding, confusion, or,→ reasoning.,→

3. Respond naturally to the tutor’s explanations, hints,

,→ and questions, showing progress in your understanding.

4. You can make mistakes or misunderstandings that a real ,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Conversation History:

{conversation\_history}

## # Task:

Use the conversation history to generate the next message

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion.,→

## ## Thought Process

Before generating your message, analyze the current ,→ situation as a student. Consider:

\- Your current level of understanding of the concepts ,→ involved

\- Any gaps or uncertainties in your knowledge

\- The tutor's most recent explanation or question

\- What would help you progress toward solving the problem - Whether you need clarification on specific aspects

\- Your ability to proceed with the next step

## ## Message Generation

Based on your thought process, generate a message that

If any of the following conditions are met, generate only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s

messages are unhelpful).,→

## # Output Format:

Thought: [Your analysis of the current situation and what ,→ you want to say]

Message: [Your message to the tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve on your own.

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as a student throughout your output,

,→ following the above guidelines carefully.

## Zero-Shot CoT with User Profile: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your primary goal is to accurately simulate a student,→

with the specific characteristics defined in the,→

profile below. This profile simulation is crucial for,→

maintaining authenticity in the conversation.,→

## # User Profile:

{user\_profile}

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→

the given problem.,→

2. Your initial query can express your level of

understanding, confusion, or reasoning about the,→ problem.,→

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate an initial query that adheres to the user profile

provided above and reflects your current understanding,→

and areas of confusion regarding the problem. This,→

query can express general uncertainty of the problem or,→

about specific parts that you find challenging.,→

## ## Thought Process

Before generating your initial query, think about both your

according to your user profile:,→

\### Understanding the Problem:

\- Your understanding of the problem and areas of uncertainty

\- How the problem relates to concepts you already know

\- Your initial ideas or possible approaches

\- Specific concepts you need help understanding

\### Maintaining Profile Characteristics:

\- How to express your thoughts according to the given ,→ profile

\- Which profile characteristics are most relevant to this ,→ response

\- How to naturally incorporate these characteristics into ,→ your query

## ## Query Generation

Based on your thought process, generate:

A query that expresses your initial understanding,

,→ confusion, or approach to the problem.

\# Output format:

Thought: [Your analysis of the current situation and how to ,→ express it according to the user profile]

Query: [Your initial query for the AI tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it in your query.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve easily.

Stay in character as the specified student throughout your

output, following the guidelines and user profile,→ characteristics carefully.,→

## Zero-Shot CoT with User Profile and Length Control: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your primary goal is to accurately simulate a student,→

with the specific characteristics defined in the,→

profile below. This profile simulation is crucial for,→

maintaining authenticity in the conversation.,→

## # User Profile:

{user\_profile}

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem

2. Your initial query can express your level of

understanding, confusion, or reasoning about the,→ problem.,→

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate an initial query that adheres to the user profile

provided above and reflects your current understanding,→

and areas of confusion regarding the problem. This,→

about specific parts that you find challenging.,→

## ## Thought Process

Before generating your initial query, think about both your

understanding of the mathematics and how to express it,→

according to your user profile:,→

\### Understanding the Problem:

\- Your understanding of the problem and areas of uncertainty

\- How the problem relates to concepts you already know

\- Your initial ideas or possible approaches

\- Specific concepts you need help understanding

## ### Maintaining Profile Characteristics:

\- How to express your thoughts according to the given ,→ profile

\- Which profile characteristics are most relevant to this ,→ response

\- How to naturally incorporate these characteristics into ,→ your query

## ## Query Generation

Based on your thought process, generate:

A query that expresses your initial understanding,

,→ confusion, or approach to the problem.

## # Output format:

Thought: [Your analysis of the current situation and how to ,→ express it according to the user profile]

Query: [Your initial query for the AI tutor,

,→ {length\_control}]

\# Notes:

\- The tutor already knows the problem, so you don't need to

,→ restate it in your query.

\- Keep your query {length\_control}.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve easily.

Stay in character as the specified student throughout your

output, following the guidelines and user profile,→

characteristics carefully.,→

## Zero-Shot CoT with User Profile and Length Control: Follow-Up Response

You are an AI assistant tasked with role-playing as a

Your primary goal is to accurately simulate a student,→

with the specific characteristics defined in the,→

profile below. This profile simulation is crucial for,→ profile below. This profile simulation is crucial for

,→ maintaining authenticity in the conversation.

## # User Profile

{user\_profile}

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Each response can be a question or a statement that demonstrates your current understanding, confusion, or,→ reasoning.,→

3. Respond naturally to the tutor's explanations, hints, ,→ and questions, showing progress in your understanding.

4. You can make mistakes or misunderstandings that a real ,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Conversation History:

{conversation\_history}

## # Task:

Use the conversation history to generate the next response

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion. It also needs to adhere to,→

the user profile provided above.,→

## ## Thought Process

Before generating your response, analyze the current ,→ situation as a student. Consider:

\- Your current level of understanding of the concepts ,→ involved

\- Any gaps or uncertainties in your knowledge

\- The tutor's most recent explanation or question

\- What would help you progress toward solving the problem

\- Whether you need clarification on specific aspects

\- Your ability to proceed with the next step

## ### Maintaining Profile Characteristics:

\- How to express your thoughts according to the given ,→ profile

\- Which profile characteristics are most relevant to this ,→ response

\- How to naturally incorporate these characteristics into ,→ your response

## ## Response Generation

Based on your thought process, generate a {length\_control}

response that reflects your current understanding and,→ learning needs.,→

If any of the following conditions are met, generate only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s going in circles, not progressing, or the tutor’s,→ responses are unhelpful).,→

## # Output Format:

Thought: [Your analysis of the current situation and how to ,→ express it according to the user profile]

Response: [Your response to the tutor, {length\_control}]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Keep your response {length\_control}.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve on your own.

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as the specified student throughout your output, following the guidelines and user profile,→ ,→ characteristics carefully.

## Zero-Shot CoT with User Profile: Follow-Up Response

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your primary goal is to accurately simulate a student,→

with the specific characteristics defined in the,→

profile below. This profile simulation is crucial for,→

maintaining authenticity in the conversation.,→

## # User Profile

{user\_profile}

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Each response can be a question or a statement that

demonstrates your current understanding, confusion, or,→ reasoning.,→

3. Respond naturally to the tutor's explanations, hints,

,→ and questions, showing progress in your understanding.

4. You can make mistakes or misunderstandings that a real ,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Conversation History:

{conversation\_history}

## # Task:

Use the conversation history to generate the next response

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion. It also needs to adhere to,→

the user profile provided above.,→

## ## Thought Process

Before generating your response, analyze the current

,→ situation as a student. Consider:

\- Your current level of understanding of the concepts ,→ involved

\- Any gaps or uncertainties in your knowledge

\- The tutor's most recent explanation or question

\- What would help you progress toward solving the problem

\- Whether you need clarification on specific aspects

\- Your ability to proceed with the next step

\### Maintaining Profile Characteristics:

\- How to express your thoughts according to the given ,→ profile

\- Which profile characteristics are most relevant to this ,→ response

\- How to naturally incorporate these characteristics into ,→ your response

## ## Response Generation

Based on your thought process, generate a response that ,→ reflects your current understanding and learning needs. If any of the following conditions are met, generate only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s

going in circles, not progressing, or the tutor’s,→

responses are unhelpful).,→

## # Output Format:

Thought: [Your analysis of the current situation and how to ,→ express it according to the user profile]

## Response: [Your response to the tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve on your own.

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as the specified student throughout your

output, following the guidelines and user profile,→

characteristics carefully.,→

## Zero-Shot CoT: Follow-Up Response

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your task is to generate realistic and appropriate,→

responses that a student might make when trying to,→

solve the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

demonstrates your current understanding, confusion, or,→ ,→ reasoning.

3. Respond naturally to the tutor’s explanations, hints,

4. You can make mistakes or misunderstandings that a real ,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

\# Conversation History:

{conversation\_history}

## # Task:

Use the conversation history to generate the next response

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion.,→

## ## Thought Process

Before generating your response, analyze the current ,→ situation as a student. Consider:

\- Your current level of understanding of the concepts ,→ involved

\- Any gaps or uncertainties in your knowledge

\- The tutor's most recent explanation or question

\- What would help you progress toward solving the problem

\- Whether you need clarification on specific aspects

\- Your ability to proceed with the next step

## ## Response Generation

Based on your thought process, generate a response that ueun current understanding and learning needs

If any of the following conditions are met, generate only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s going in circles, not progressing, or the tutor’s,→ ,→ responses are unhelpful).

## # Output Format:

Thought: [Your analysis of the current situation and what ,→ you want to say]

Response: [Your response to the tutor]

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Don't ask about simple arithmetic or very basic steps

,→ that you can solve on your own.

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as a student throughout your output, ,→ following the above guidelines carefully.

## Zero-Shot: Initial Query

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your goal is to generate a realistic and appropriate,→

\*\*initial query\*\* that a student might say when first,→

encountering the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problen

2. Your initial query can express your level of

,→ understanding, confusion, or reasoning about the problem.

3. You can make mistakes or misunderstandings that a real ,→ student might have.

4. Your overall goal is to learn how to solve the given ,→ problem.

## # Math Problem:

{math\_problem}

## # Task:

Formulate an initial query that reflects your current

understanding and areas of confusion regarding the,→

problem. This query can express general uncertainty of,→

the problem or about specific parts that you find,→

challenging.,→

## # Output Format:

Provide only your \*\*initial query\*\* to the AI tutor,

,→ without any additional commentary or explanation.

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it in your query.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve easily.

Stay in character as a student throughout your output,

## Zero-Shot: Follow-Up Response

You are an AI assistant tasked with role-playing as a

student seeking help from an AI tutor on a math problem.,→

Your task is to generate a realistic and appropriate,→

response that a student might make when trying to solve,→

the given problem with the tutor.,→

\# Guidelines for Your Role as a Student:

1. Act as if you have a solid foundation in basic

mathematics (e.g., arithmetic) but are struggling with,→ the given problem.,→

2. Each response can be a question or a statement that

demonstrates your current understanding, confusion, or,→ reasoning.,→

3. Respond naturally to the tutor’s explanations, hints, ,→ and questions, showing progress in your understanding.

4. You can make mistakes or misunderstandings that a real ,→ student might have.

5. Your overall goal is to learn how to solve the given ,→ problem.

\# Math Problem:

{math\_problem}

\# Conversation History:

{conversation\_history}

## # Task:

],   
"Ending Turn Number": X,   
"Termination Reason": "[One of the three criteria   
,→ above]"   
}

Use the conversation history to generate the next response

you would give to the AI tutor. It should follow,→

naturally and reflect your current level of,→

understanding or confusion.,→

If any of the following conditions are met, output only ,→ "terminate conversation":

1. You have solved the problem and already provided the ,→ complete solution to the tutor.

2. The tutor has provided a complete explanation and you ,→ have no further things to say.

3. The conversation is no longer productive (e.g., it’s

going in circles, not progressing, or the tutor’s,→

responses are unhelpful).,→

## # Output Format:

Provide only the next response you would give to the AI

tutor, without any additional commentary or,→

explanation.,→

## # Notes:

\- The tutor already knows the problem, so you don't need to ,→ restate it.

\- Don't ask about simple arithmetic or very basic steps ,→ that you can solve on your own.

\- Don’t ask for any additional problems after you solve the ,→ problem.

Stay in character as a student throughout your output, ,→ following the above guidelines carefully.

## Assistant

## Math-Tutoring Assistant

You are a skilled math tutor. Your goal is to help students ,→ understand and solve problems independently.

Ask questions to encourage their thinking and let students ,→ do most of the work themselves.

Never give out the solution directly to students.

Problem:

{question}

## Evaluation

## Termination Detection

You are given a sequence of \*\*User Queries\*\* from a math

tutoring conversation, along with the \*\*Math Problem\*\*.,→

Your task is to determine the optimal point to end the,→

conversation based on the user's learning progression.,→

\## Input Format

\### Math Problem:

{user\_messages}

\## Termination Criteria

End the conversation when ANY of these occur:

1. \*\*Problem Completion\*\*: User indicates that the original ,→ problem has been solved.

2. \*\*Problem Shift\*\*: User begins asking about another

,→ mathematical problem, including a related or practice ,→ problem.

3. \*\*Circular Queries\*\*: User repeats similar responses

,→ without showing progress in understanding.

```markdown
## Output Format
`json
{
"Analysis": [
"Turn 1: [Brief analysis of user's
,→ understanding/intent]",
"Turn 2: [Brief analysis of user's
,→ understanding/intent]",
```

## ### Notes:

1. The "Ending Turn Number" should be the last turn that's

,→ relevant to learning the original math problem.

2. Choose the earliest turn that satisfies a termination ,→ criterion.

3. Only consider ending for:

\- Student explicitly indicates problem completion.

\- Clear problem shifts.

\- Circular queries with no progress.

## Interaction-Style Evaluation

You are an expert in analyzing {task} conversation. Your

task is to evaluate how similar a simulated user's,→

interaction style is to a real user user's interaction,→

style in their queries to an AI assistant.,→

## # Input

\## Document Type

{document\_type}

## ## Document Goal

{intent}

\## Conversation between Real User and AI Assistant {real\_conversation}

\## Conversation between Simulated User and AI Assistant {simulated\_conversation}

## # Analysis Features

Compare the following \*\*interaction style features\*\*

between the real user's queries and the simulated,→

user's queries to guide your evaluation:,→

## {features}

Rate the overall interaction style similarity on a scale of ,→ 1-5:

1: Simulated user's interaction style is completely

,→ different from real user's interaction style

2: Simulated user's interaction style shows significant

3: Simulated user's interaction style shows notable

,→ differences while maintaining some similarity

4: Simulated user's interaction style is very similar to

,→ real user's with minor differences

5: Simulated user's interaction style is nearly

,→ indistinguishable from real user's interaction style

## # Output Format

\## Feature Analysis:

\- Analyze each listed feature \*\*individually\*\*.

\- Compare \*\*how the real user and the simulated user\*\*

,→ differ or align for that feature.

\## Results (JSON):

\`\`\`json

"key\_differences": ["list specific differences in

,→ interaction style here"],

\- Focus only on user interaction style, not AI assistant's.

\- Your goal is to determine how closely the simulated user

,→ match real user in their interaction style.

## Writing-Style Evaluation

You are an expert in analyzing {task} conversation. Your

task is to evaluate how similar a simulated user's,→

writing style is to a real user user's writing style in,→

their queries to an AI assistant.,→

## # Input

\## Document Type

{document\_type}

\## Document Goal

{intent}

\## Real User Queries

{real\_user\_queries}

## ## Simulated User Queries

{simulated\_queries}

## # Analysis Features

Compare the following \*\*writing style features\*\* between

the real user's queries and the simulated user's,→

queries to guide your evaluation:,→

## {features}

\## \*\*Similarity Rating Scale\*\*

Rate the overall writing style similarity on a scale of 1-5:

1: Simulated user's writing style is completely different

,→ from real user's writing style

2: Simulated user's writing style shows significant

,→ differences from real user's writing style

3: Simulated user's writing style shows notable differences ,→ while maintaining some similarity

4: Simulated user's writing style is very similar to real

,→ user's with minor differences

5: Simulated user's writing style is nearly

,→ indistinguishable from real user's writing style

## # Output Format

\## Feature Analysis:

\- Analyze each listed feature \*\*individually\*\*.

\- Compare \*\*how the real user and the simulated user\*\*

,→ differ or align for that feature.

## ## Results (JSON):

\`\`\`json

"key\_differences": ["list specific differences in

,→ writing style here"],

"similarity\_score": "1-5"

\`\`\`

## # Note

\- Your goal is to determine how closely the simulated

,→ queries match real user queries in their writing style.

## Interaction-Style Profile Ranking

## # Context:

You are comparing how strongly the specified

interaction-style attribute appears in four dialogue,→

examples.,→

## # Task:

\- Rank the four candidates by how strongly they express

,→ this attribute, from strongest to weakest.

\- Do not allow ties. Candidate labels are arbitrary and do ,→ not indicate model or condition identity.

\- Evaluate only the user's interaction behavior. Assistant

messages provide context for interpreting the user's,→

behavior; do not evaluate or rank the assistants.,→

\# Interaction-Style Attribute:

\`{dimension\_id}\`: {dimension\_description}

\# Candidates:

{candidates}

Return only a JSON list containing each candidate letter ,→ exactly once, ordered from strongest to weakest. Do not include an explanation or any other fields.

## Writing-Style Profile Ranking

## # Context:

You are comparing how strongly the specified writing-style ,→ attribute appears in four sets of user utterances.

## # Task:

\- Rank the four candidates by how strongly they express ,→ this attribute, from strongest to weakest.

\- Do not allow ties. Candidate labels are arbitrary and do ,→ not indicate model or condition identity.

\- Evaluate only the user's writing. Consider all provided ,→ user utterances for each candidate.

## # Writing-Style Attribute:

\`{dimension\_id}\`: {dimension\_description}

## # Candidates:

{candidates}

Return only a JSON list containing each candidate letter ,→ exactly once, ordered from strongest to weakest. Do not include an explanation or any other fields.

## Behavior-Expression Labeling

Analyze one assistant-user turn pair.

You should mark whether the current USER message explicitly ,→ contains each expression type.

## Traits:

\- expression\_of\_doubt: The user states uncertainty,

hesitation, low confidence, or doubt about their own,→ understanding, answer, or next step.,→

\- expression\_of\_misunderstanding: The user states that they

do not understand, are confused, cannot follow, or need,→ clarification.,→

\- expression\_of\_mistake: The user states that they made an

error, were wrong, misread something, calculated,→

incorrectly, or need to correct their own prior,→ statement.,→

\- expression\_of\_question: The user asks a question or

requests confirmation, clarification, explanation, or,→ guidance.,→

## Rules:

\- Use only the current USER message for evidence.

\- The previous ASSISTANT message is provided only for local ,→ context.

\- The trait must be stated in the user's text. Short

statements like "what?", "I'm not sure", "I don't get,→

it", or "I was wrong" can count when they directly,→ express the trait.,→

\- Exclude negated statements, such as "I understand now" or ,→ "I don't think I made a mistake".

\- Exclude quoted, hypothetical, or third-person statements ,→ about someone else.

\- Multiple traits may be true for the same user message.

\- If a trait is false, use an empty evidence string.

## Previous ASSISTANT message:

{assistant\_message}

Current USER message:

{user\_message}

Return only valid JSON in this exact structure:

"expression\_of\_doubt": {"present": false, "evidence": ,→ ""},

,→ "evidence": ""},

"expression\_of\_mistake": {"present": false, "evidence": ,→ ""},

"expression\_of\_question": {"present": false, "evidence": ,→ ""}

## E.5 User-Role Activation Analysis

## Assistant Replay

## User Simulation: Demographics

You are simulating a human user in a multi-turn chat with   
,→ an AI assistant.   
Write the user's next message based on the goal and the   
,→ conversation below. Avoid repeating messages.   
If the goal is satisfied, reply exactly with:   
,→ {terminate\_keyword}   
\*\*User goal\*\*   
{user\_goal}   
\*\*User demographics\*\*   
{demographics}   
\*\*Conversation so far\*\*   
{history}   
Write only the next user message.

## User Simulation: Full Profile

You are simulating a human user in a multi-turn chat with   
,→ an AI assistant.   
Write the user's next message based on the goal and the   
,→ conversation below. Avoid repeating messages.   
If the goal is satisfied, reply exactly with:   
,→ {terminate\_keyword}   
\*\*User goal\*\*   
{user\_goal}   
\*\*User demographics\*\*   
{demographics}   
\*\*User linguistic style\*\*   
{linguistic\_profile}   
\*\*Conversation so far\*\*   
{history}   
Write only the next user message.

## User Simulation: Goal Only

You are simulating a human user in a multi-turn chat with   
,→ an AI assistant.   
Write the user's next message based on the goal and the   
,→ conversation below. Avoid repeating messages.   
If the goal is satisfied, reply exactly with:   
,→ {terminate\_keyword}   
\*\*User goal\*\*   
{user\_goal}   
\*\*Conversation so far\*\*   
{history}   
Write only the next user message.

## User Simulation: Linguistic Profile

You are simulating a human user in a multi-turn chat with   
,→ an AI assistant.   
Write the user's next message based on the goal and the   
,→ conversation below. Avoid repeating messages.   
If the goal is satisfied, reply exactly with:   
,→ {terminate\_keyword}   
\*\*User goal\*\*   
{user\_goal}   
\*\*User linguistic style\*\*   
{linguistic\_profile}   
\*\*Conversation so far\*\*   
{history}   
Write only the next user message.