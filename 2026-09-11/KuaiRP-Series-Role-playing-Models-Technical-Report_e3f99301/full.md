# KuaiRP Series Role-playing Models Technical Report

Kuaishou GameMind Lab

See Contributions section for a full author list.

This paper introduces the complete technical solution for the KuaiRP series of role-playing models. We aim to achieve four core objectives for a dedicated role-playing model: simplified prompt engineering, highly stable output quality, built-in domain world knowledge, and high-eficiency deployment with a small parameter size. However, efectively injecting deep domain knowledge often leads to a severe catastrophic forgetting of the model’s general agent capabilities. To overcome this trade-of, we propose a multi-stage training pipeline. First, we design a standardized character template and construct an SFT data pipeline based on user behavior simulation and reverse profile filtering. Next, we utilize a rule-based composite reward function during the Reinforcement Learning (RL) phase to eliminate common degradation phenomena like length expansion and repetitive generation. Finally, to recover the general capabilities compromised during SFT and RL, we propose a novel self-distillation paradigm using Two-stage On-Policy Distillation (OPD) equipped with Cumulative-Divergence Decay (CDD). By using the domain-adapted model as the teacher and the original base model as the student, we efectively balance deep domain knowledge injection with the preservation of general agent capabilities. Experimental results demonstrate that the KuaiRP models not only match the current state-of-the-art proprietary models in role-playing fidelity within our target domains, but also successfully recover general agent capabilities, maintaining extremely low deployment costs.

![](images/f6a0bd2b86fdb296db359a315094615d1c70f2b536308843257779429afbf994.jpg)  
Figure 1: Overview of the KuaiRP training pipeline.

## Contents

1 Introduction 3   
1.1 Technical Challenges 3   
1.2 Methodology Overview 3   
1.3 Contributions 3   
2 Role Play Template Design 4   
2.1 Motivation . 4   
2.2 Template Structure 4   
2.3 Two Special Design Points 4   
3 SFT 5   
3.1 Data Construction . 5   
3.2 Fine-tuning Process 6   
3.3 Training Recipe 7   
4 RL 7   
4.1 RL Motivation . 7   
4.2 Reward Function Design 7   
4.3 Training Recipe 9   
5 OPD (On-Policy Distillation) 9   
5.1 Motivation . 9   
5.2 Overview of OPD Method 9   
5.3 Two-Stage-OPD with Cumulative-Divergence Decay (CDD) 11   
5.4 Training Recipe 13   
Discussion: Evaluation and Key Findings 14   
6.1 Stage-by-Stage Result Analysis 15   
6.2 Key Findings 16   
7 Benchmark and Evaluation 16   
7.1 Data Production Method 17   
7.2 Scoring Method . . 17   
7.3 Horizontal Comparison of Broader Models (TRACEbench Leaderboard) . 19   
8 Conclusion and Future Work 19   
Appendices   
A Role Play Template Structure 22

## 1 Introduction

While there are many role-playing models in the current open-source ecosystem—and even general large language models (LLMs) can achieve basic role-playing capabilities through prompt engineering—building a dedicated, high-fidelity role-playing model remains essential for immersive application scenarios. We define the core objectives of a dedicated role-playing model across four dimensions:

1. Simpler Prompt Engineering: Without carefully tuning the structure and wording of the prompt, the model can stably follow the character settings and achieve high-fidelity role-playing.

2. More Stable Output Quality: Maintain consistent formats, moderate lengths, and suficient diversity in multi-turn dialogues, avoiding common degradation phenomena such as format confusion, gradual turn expansion, and content repetition.

3. Built-in Domain World Knowledge: Internalize relevant world knowledge (e.g., the target domain setting) into model parameters, enabling realistic role-playing without relying heavily on external knowledge bases.

4. Small Size, High Eficiency: The model can be deployed on a single consumer-grade GPU (e.g., 24GB VRAM) and has low inference latency, meeting the cost-efectiveness requirements of production environments.

## 1.1 Technical Challenges

Achieving these four objectives simultaneously presents a significant technical challenge: the trade-of between deep domain knowledge injection and the preservation of general agent capabilities. Injecting extensive world knowledge typically requires full-parameter supervised fine-tuning (SFT) on highly specialized domain data. However, training heavily on such a narrow distribution causes the model to sufer from catastrophic forgetting, severely degrading its general capabilities (e.g., tool-calling, logical reasoning, and instruction following). Conversely, parameter-eficient methods like LoRA preserve general capabilities but are notoriously ineficient at internalizing factual world knowledge.

## 1.2 Methodology Overview

To overcome this dilemma, we propose a comprehensive training pipeline to build the KuaiRP series models, as illustrated in Figure 1. First, we design a standardized character template to unify the prompt interface. We then construct a high-quality SFT dataset utilizing commercial model distillation, user behavior simulation, and reverse profile filtering. Next, we apply Reinforcement Learning (RL) with a rule-based composite reward function—enforcing hard constraints on formatting, length, and diversity—to correct output distributions and eliminate degradation issues. Finally, to recover the general capabilities lost during the SFT and RL stages, we introduce a novel Two-stage On-Policy Distillation (OPD) approach. By using the original base model as the student and the domain-adapted model as the teacher, we successfully inject domain capabilities into the base model while maintaining its general prowess.

## 1.3 Contributions

Our main contributions are summarized as follows:

• High-Quality Data Pipeline: We introduce a robust data construction pipeline featuring user behavior instruction injection and reverse profile filtering, which generates highly diverse and accurate SFT data for character-following.

• A Novel Self-Distillation Training Paradigm: We propose an SFT → RL → Two-stage OPD pipeline starting and ending on the same base model. Distinct from traditional SFT → RL paradigms (which often sufer from catastrophic forgetting of general abilities) and conventional cross-model distillation (from a large teacher to a small student), our approach uses the domainadapted model as the teacher and the original base model as the student. This efectively balances deep domain knowledge injection with the preservation of the base model’s general capabilities.

• Cumulative-Divergence Decay (CDD): We identify the prefix-drift issue in on-policy Generalized Knowledge Distillation (GKD) [1], a critical challenge recently recognized in the field (e.g., IW-OPD, FiRe-OPD), and introduce CDD as a key algorithmic improvement. Unlike approaches that break autoregressive causality or rely on heuristic hard-filtering, CDD ensures stable and thorough world-knowledge injection by smoothly decaying weights based on strict causal divergence, without forcing the student to learn from noisy, out-of-distribution teacher responses.

• Empirical Success: We present the KuaiRP series models, which achieve state-of-the-art roleplaying fidelity within our target domain scenarios compared to proprietary models (e.g., M2-HER), while successfully maintaining strong general agent capabilities and low deployment costs.

## 2 Role Play Template Design

## 2.1 Motivation

Prompt Engineering refers to the technique of designing and optimizing the instructions (prompts) input to AI models to obtain more accurate and high-quality outputs. However, in practical use, each character creator has their own preferred expression habits, which leads to varying quality in the prompts generated for diferent characters.

To address this, we designed a fixed role play template that serves a dual regulatory function:

1. For character creators: It provides a standardized fill-in-the-blank template that can comprehensively depict various dimensions of a character.

2. For models: A unified prompt structure enables the model to accurately and stably understand character settings. Regardless of which character is loaded, it can achieve high-fidelity role-playing.

## 2.2 Template Structure

The specific structure of the role play template is provided in Appendix A.

## 2.3 Two Special Design Points

1. Incorporating player profile into the prompt: This is the core of achieving personalized dialogues. In traditional role-playing models, the character treats all players equally; whereas we explicitly write the player’s profile (nickname, relationship with the character, gender, etc.) into the prompt, allowing the model to perceive diferentiated social contexts based on diferent users. For example, a player with a close relationship to the character will be called by their nickname and receive a more casual tone, while a player with a stranger relationship will receive a more formal and polite response. This design allows characters to adapt to diferent people rather than using a uniform reply template.

2. ”Specific Behavioral Patterns” section: Introduced a trigger mechanism for character behavior, directly adapting to the gameplay requirements in game scenarios. For example, a character can trigger a special reaction when the player mentions a specific keyword or meets certain in-game conditions.

## 3 SFT

## 3.1 Data Construction

## 3.1.1 Overview

Our SFT data pipeline is divided into two stages: first, filling the character library based on the role play template, and then having a commercial model play the characters and a user model play the players to interact, finally producing multi-turn dialogue training data for SFT. The following breaks down each step.

## 3.1.2 Character Library

We extracted 12 characters with the most distinct personalities from the original novel of the target domain, and then designed 13 derivative characters based on its world setting to achieve a balanced coverage across dimensions such as personality, gender, age, and style. Since the user’s gender information is also part of the character setting prompt, each character corresponds to two versions of settings (targeting male and female users respectively), resulting in a total of 50 character prompts used for data generation.

These 50 character prompts were input into a commercial LLM, having it role-play as the characters.

## 3.1.3 User Library

In traditional data distillation schemes, typically only a single user model is used to interact with the role-playing model. To ensure the training data covers diferent types of user styles while enhancing data diversity, we constructed a library containing 10 diferent styles of user profiles. These user profiles align with common features of real internet chats: heavy use of emojis, fragmented short sentences, incorrect or missing punctuation, internet slang, etc.

## 3.1.4 Data Generation (Simulated Dialogue)

We used Claude 3.7 Sonnet as the role-playing model and Qwen2.5-14B [8] as the user model for the dialogues. Compared to traditional approaches, we made three key innovations:

User Behavior Simulation via Instruction Injection Even though diferent styles of users have been simulated at the profile level, user behaviors in actual dialogues might still lack certain realistic features, such as: being uncooperative, suddenly starting a new topic, typos and missing words, etc. Therefore, before the generation of the user’s turn, we temporarily inject a system message (which is more efective than modifying the original system prompt because the temporarily injected information is closer to the generation of tokens) to precisely control user behavior. These behaviors are sampled from our pre-defined behavioral strategy library.

The coverage of the behavioral strategy library goes far beyond superficial style changes—it includes not only interaction patterns like uncooperativeness and starting new topics, but also key behaviors such as asking questions about historical information, asking about character profile details, asking about world-setting knowledge of the target domain, calling tools, and adversarial attacks. This design enables us to eficiently distill the character-following ability of the commercial model during natural dialogue, while organically integrating world knowledge into the training data—which is far more eficient than direct training with QA pairs.

Reverse Profile Filtering Although we have explicitly simulated user behaviors to trigger specific items in the character profile through instruction injection, it is still impossible to guarantee that all profile items will be triggered within a limited number of dialogue turns. Furthermore, we cannot guarantee that the distilled commercial model will respond 100% according to specific items in the profile. Therefore, there may be some items in the profile that have no corresponding character response throughout the entire dialogue—these ”invalid items” constitute noise for the model. If the model tries to follow these items that were never demonstrated during the learning process, it will instead deviate from the core requirements of instruction following.

To this end, we designed a reverse profile filtering (abbreviated as RPF) mechanism: for each piece of distilled data, we conduct semantic relevance detection between every item in the profile template and the character’s responses, removing invalid items that are never reflected in any character response throughout the dialogue. After reverse filtering, the profile items retained in each training sample strictly correspond to the dialogue content, making the ”profile → response” mapping relationship learned by the model purer and more precise. Meanwhile, because diferent dialogues trigger diferent subsets of profile items, the filtered data naturally possesses higher diversity in profile combinations, further enriching the training distribution.

The efectiveness of this mechanism can be verified through the comparison experiment on domain roleplaying capabilities: as shown in the Char-Consist. dimension of Table 7, the SFT model using RPF achieves better character consistency than the ablation model without RPF, indicating that this mechanism improves the model’s instruction-following ability and yields better persona consistency in role-playing.

Stratified Sampling of Dialogue Turns Initially, we simulated 20 sessions of 15-turn dialogues for all ”character-user” pairs, but later found that the trained model’s performance significantly degraded in long-turn dialogues (> 10 turns). Therefore, under the premise of keeping the total token consumption roughly the same, we divided the simulation turns into multiple gradient groups:

• 5 turns × 4 groups

• 10 turns × 4 groups

• 15 turns × 4 groups

• 20 turns × 4 groups

• 25 turns × 4 groups

This ensures that the training data has an adequate distribution across all turn length intervals, efectively mitigating the degradation issue in long-turn dialogues.

## 3.2 Fine-tuning Process

We compared LoRA-based SFT [2] and full-parameter SFT schemes on the Qwen3-8B [12] base model and made the following key findings:

1. Advantages of LoRA: LoRA can easily learn formatting features of dialogues—including action format markers, gender addresses, profile following, etc.—and due to limited parameters, it is less prone to overfitting.

2. Limitations of LoRA: LoRA is extremely insensitive to the injection of world knowledge. This is a fatal flaw in role-playing scenarios: characters need to internalize background settings, character relationships, worldview rules, and other knowledge of the target domain, but the low-rank bottleneck of LoRA limits the depth of knowledge injection. We tested a scheme with a larger learning rate, in which LoRA could indeed inject more knowledge, but it also showed obvious overfitting and brought no significant benefits compared to the full-parameter SFT scheme.

3. Final Choice: Based on the above comparison, we abandoned the LoRA scheme and chose full-parameter SFT as the final training paradigm. Full-parameter fine-tuning can efectively learn formatting features and world knowledge while controlling the risk of overfitting through appropriate regularization methods.

## 3.3 Training Recipe

The key hyperparameters of the full-parameter SFT are summarized in Table 1.

Table 1: SFT Training Recipe.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Base Model</td><td> ${ \tt Q w e n 3 } ( 0 \tt r i g i n a l )$ </td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Batch Size</td><td>64</td></tr><tr><td>Epochs</td><td>2</td></tr><tr><td>Tuning Method</td><td>Full-parameter fine-tuning</td></tr></table>

This SFT model will serve as the starting point for subsequent RL training.

## 4 RL

## 4.1 RL Motivation

After thoroughly testing the SFT model, we identified the following phenomena:

1. Length expansion: The later the dialogue turn, the higher the probability of the model outputting excessively long responses. Our game scenarios primarily feature daily conversations, and overly long responses create greater reading pressure for users. We expect the output length to be moderate and stable.

2. Formatting errors: We use special markup conventions to distinguish action descriptions from spoken text. The SFT model occasionally fails to follow these formatting rules, resulting in actions failing to be parsed and displayed correctly.

3. Repetitive speech: The model occasionally copies content verbatim from the historical dialogue, afecting the naturalness of the conversation.

The above issues share a key attribute that makes them highly suitable for correction through RL:

• These issues do not always occur (the model possesses partial capabilities; it’s not entirely incapable).

• Most of them are verifiable problems—rule-based evaluators can provide clear binary feedback signals.

This makes reinforcement learning an ideal means to correct the output distribution. The following first introduces our reward function design (Section 4.2), and then introduces our training recipe.

## 4.2 Reward Function Design

## 4.2.1 Overview

We designed a composite reward function integrating three sub-reward signals, specifically for roleplaying dialogue generation tasks. The final reward score takes the minimum of the three sub-rewards, implementing a ”no-shortcut” mechanism—the model must simultaneously satisfy all hard constraints, rather than compensating for a low score in one dimension with a high score in another.

$$
r = \mathrm { m i n } \big ( r _ { \mathrm { f o r m a t } } , ~ r _ { \mathrm { l e n g t h } } , ~ r _ { \mathrm { d i v e r s i t y } } \big ) ,\tag{1}
$$

## 4.2.2 Sub-reward Definitions

1. Format Reward $\mathbf { ( { r _ { f o r m a t } } ) }$ The format reward constrains the structural specifications of role-playing responses, including the way action descriptions are marked and the use of colons. It is a binary signal:

$$
r _ { \mathrm { f o r m a t } } \in \{ 0 , 1 \} .\tag{2}
$$

Only when the response satisfies all formatting rules is $r _ { \mathrm { f o r m a t } } = 1 $ ; any rule violation results in a score of 0.

2. Length Reward $\left( r _ { \mathbf { l e n g t h } } \right)$ The length reward encourages concise and substantial dialogue content, constraining the number of characters in the non-action parts.

Let $\ell _ { \mathrm { n o n - a c t i o n } }$ be the number of characters remaining after removing all action markup patterns $( \cdots , \ldots , \ast \ast _ { \mathrm { ~ , ~ } \mathrm { ~ \dots ~ } \mathrm { ~ , ~ } \ast \ast } , \ast \ast _ { \mathrm { ~ , ~ } \mathrm { ~ \dots ~ } \mathrm { ~ , ~ } \mathrm { ~ ( ~ \dots ~ ) ~ , ~ } \mathrm { ~ ( ~ \dots ~ ) ~ } \mathrm { ~ ) ~ } }$ from the response, then:

$$
r _ { \mathrm { l e n g t h } } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ 1 \leq \ell \ell _ { \mathrm { n o n - a c t i o n } } \leq 1 0 0 , } } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{3}
$$

This design simultaneously penalizes empty/minimalist responses and overly lengthy monologues, keeping character lines within a natural conversational word count range.

3. Diversity Reward $( r _ { \mathbf { d i v e r s i t y } } )$ The diversity reward penalizes content that is overly similar to the character’s historical responses, preventing the model from outputting repetitive or templated content.

Let the current character response be y, and the historical response set be $\mathcal { H } = \{ h _ { 1 } , h _ { 2 } , \ldots , h _ { n } \}$ We split y and each $h _ { i }$ with $0 \because \ ? \ \vdots \ \vdots \ \vdots \ \vdots \ \ddots$ , then discard sentences shorter than five characters after trimming whitespace. The resulting valid-sentence sets are $ { \boldsymbol { S } } ( y )$ and:

$$
S ( \mathcal { H } ) = \bigcup _ { h _ { i } \in \mathcal { H } } S ( h _ { i } ) .\tag{4}
$$

For each pair of current and historical valid sentences $a \in S ( y )$ and $b \in { \mathcal { S } } ( { \mathcal { H } } )$ , we calculate the character-level 2-gram Jaccard similarity:

$$
J ( a , b ) = { \frac { | { \mathcal { G } } _ { 2 } ( a ) \cap { \mathcal { G } } _ { 2 } ( b ) | } { | { \mathcal { G } } _ { 2 } ( a ) \cup { \mathcal { G } } _ { 2 } ( b ) | } } ,\tag{5}
$$

where $\mathcal { G } _ { 2 } ( s )$ is the set of all character bigrams formed after removing all whitespace characters and lowercasing the string s. The maximum similarity among all valid sentence pairs is:

$$
J _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in S ( y ) , \ b \in S ( \mathcal { H } ) } J ( a , b ) .\tag{6}
$$

If there are no valid sentences in the current response or historical responses, it is considered that there is no comparable content, and a full score is directly given. Otherwise, the diversity reward is calculated with a hard threshold $\tau = 0$ .4:

$$
r _ { \mathrm { d i v e r s i t y } } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \ : S ( y ) = \emptyset \ : \mathrm { o r } \ : S ( \mathcal { H } ) = \emptyset , } \\ { 0 } & { \mathrm { i f } \ : J _ { \mathrm { m a x } } > 0 . 4 , } \\ { 1 } & { \mathrm { i f } \ : J _ { \mathrm { m a x } } \leq 0 . 4 . } \end{array} \right.\tag{7}
$$

## 4.2.3 Composite Score

The final scalar reward used to update the policy is:

```html
r = min r<sub>format</sub>, r<sub>length</sub>, r<sub>diversity</sub>
```

(8)

All three sub-rewards are binary signals ({0, 1}), so the composite score is also binary. This design is essentially a conjunction of hard constraints: the response must simultaneously meet formatting specifications, length constraints, and novelty requirements to receive a positive reward; failure to meet the standard in any dimension results in a score of zero, and the corresponding sample enters the negative sample pool, participating in the policy gradient update.

## 4.3 Training Recipe

This section only retains key training hyperparameters (implementation details of framework components are not expanded), as summarized in Table 2.

Table 2: RL Training Recipe.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Base Model</td><td>SFT model</td></tr><tr><td>Advantage Estimator Include KL in reward</td><td>GRPO [10] True</td></tr><tr><td>Reward-side KL coef</td><td>0.005</td></tr><tr><td>Actor-side KL loss coef</td><td>0.005</td></tr><tr><td>PPO clip ratio</td><td>low = 0.20, high = 0.28 [9]</td></tr><tr><td>Learning Rate</td><td>3 × 10−6</td></tr><tr><td>Rollouts per prompt</td><td>8</td></tr><tr><td>Training Batch Size</td><td>32</td></tr><tr><td>PPO Mini-batch Size</td><td>16</td></tr><tr><td>Total Training Steps</td><td>20</td></tr></table>

## 5 OPD (On-Policy Distillation)

## 5.1 Motivation

In role-playing scenarios developed for games, we require the model to have not only conversational capabilities but also tool-calling abilities to implement gameplay details. However, we found that despite taking various measures during data construction, fine-tuning strategies, and RL reward design to avoid overfitting, the model still exhibited a degradation in general capabilities after the two-stage SFT + RL training—specifically manifested as a decline in general abilities such as tool calling.

The essence of this problem is that the training data distributions in the SFT and RL stages are highly concentrated in the specific domain world setting. In adapting to this distribution, the model gradually deviates from the general knowledge space of the base model. Therefore, we need a method to ”inject” the role-playing capabilities under the target domain world setting back into the base model, while preserving its original general capabilities as much as possible.

## 5.2 Overview of OPD Method

On-Policy Distillation (OPD) is an online policy distillation method. Its core idea is: while the student model is generating rollouts online, it uses the token-level probability distribution of the teacher model as a preference signal, updating the student model through gradients via KL divergence or related estimators. Unlike ofline SFT distillation, the ”online” nature of OPD ensures that distillation always occurs on the student’s current policy distribution, avoiding the distribution shift problem.

Formal Definition: Let $x \sim p _ { \mathrm { d a t a } }$ be a prompt, $y \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ be a rollout sampled by the student policy, and $s _ { t } ~ = ~ ( x , y _ { < t } )$ be the state corresponding to the t-th token. OPD updates the student parameters θ by minimizing the following objective:

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x \sim p _ { \mathrm { d a t a } } , y \sim \pi _ { \theta } ( \cdot | x ) } \left[ \frac { 1 } { | y | } \sum _ { t = 1 } ^ { | y | } D ( \pi _ { \theta } ( \cdot \mid s _ { t } ) , \nu ( \cdot \mid s _ { t } ) , y _ { t } ) \right] ,\tag{9}
$$

where $\pi _ { \theta }$ is the student policy, ν is the teacher policy, and D is the per-token divergence or its estimator. When the student updates, the sampled rollouts are treated as fixed (stop-gradient on the sampling process). The only diference between various OPD variants lies in the specific form of D.

There are two main branches of OPD:

1. OPD-GKD (Generalized Knowledge Distillation) [1]: Directly minimizes the forward KL between the student and teacher distributions over states induced by the student:

$$
D ( \pi _ { \theta } ( \cdot \mid s _ { t } ) , \nu ( \cdot \mid s _ { t } ) , y _ { t } ) = \sum _ { v \in V } \nu ( v \mid s _ { t } ) \log \frac { \nu ( v \mid s _ { t } ) } { \pi _ { \theta } ( v \mid s _ { t } ) } .\tag{10}
$$

Since current inference engines typically only return the log-prob of the sampled token and the teacher’s top-k tokens, making it dificult to obtain the log-prob for arbitrary token IDs, GKD in practice uses a teacher top-k approximation:

$$
\mathcal { L } _ { \mathrm { G K D } } ^ { ( k ) } ( s _ { t } ) = \sum _ { v \in \mathrm { T o p K } ( \nu ( \cdot | s _ { t } ) ) } \nu ( v \mid s _ { t } ) \left[ \log \nu ( v \mid s _ { t } ) - \log \pi _ { \theta } ( v \mid s _ { t } ) \right] .\tag{11}
$$

The distillation signal of GKD is relatively stronger and can more directly ”cover” the teacher distribution onto the student model, but correspondingly, it creates a larger impact on the student’s original distribution.

2. OPD-K1 (OPD-PG): Uses the K1 estimator (i.e., the diference between teacher log-prob and student log-prob) as the loss signal, which is a single-sample Monte Carlo estimate of the reverse KL:

$$
\begin{array} { r } { D ( \pi _ { \theta } ( \cdot  { | } s _ { t } ) , \nu ( \cdot  { | } s _ { t } ) , y _ { t } ) = \mathrm { s g } ( \log \pi _ { \theta } ( y _ { t }  { | } s _ { t } ) - \log \nu ( y _ { t }  { | } s _ { t } ) ) , } \\ { y _ { t } \sim \pi _ { \theta } ( \cdot  { | } s _ { t } ) . \qquad } \end{array}\tag{12}
$$

OPD-PG takes its negative as the per-token reward, driving the policy gradient update:

$$
r _ { t } = \operatorname { s g } ( \log \nu ( y _ { t } \mid s _ { t } ) - \log \pi _ { \theta } ( y _ { t } \mid s _ { t } ) ) .\tag{13}
$$

When a token generated by the student model is also a high-probability token for the teacher model, the K1 value approaches zero, generating almost no gradient signal—at this point, it will not significantly afect the student distribution. But if a token has a low probability in the teacher distribution and a high probability in the student distribution, the K1 estimator will only suppress the probability of that token in the student distribution, rather than globally pulling the two distributions closer as SFT or GKD does. Therefore, K1’s intervention on the model distribution is gentler and more precise. Here, sg(·) is the stop-gradient operator, ensuring that when the reward is used within the policy gradient objective, it does not backpropagate gradients through the teacher log-prob.

## 5.3 Two-Stage-OPD with Cumulative-Divergence Decay (CDD)

## 5.3.1 Basic Idea

Teacher Model: A model trained through SFT + RL (possessing role-playing capabilities under the target domain world setting).

Student Model: The original Qwen3-8B base model (possessing full general capabilities).

Our goal is to distill the role-playing capabilities and the mastered world-setting knowledge from the teacher model into the student model, while preserving the student model’s original general knowledge and tool-calling capabilities to the greatest extent possible.

## 5.3.2 Encountered Problems

In actual training, we found that the student model could stably learn role-playing capabilities, output formats, etc., but encountered the following dificulties in learning world-setting knowledge:

1. Extremely Low Eficiency of OPD-PG: When using the OPD-PG approach, the specific token sequences required for the target domain world knowledge are almost impossible to appear in the base model’s (student’s) sampling distribution—the student model simply does not have the chance to generate these contents, so the teacher model’s gradient signals cannot be efectively transmitted. The student model can hardly learn any world knowledge.

2. OPD-GKD is Also Ineficient: When we switch to the GKD mode and let the teacher distribution directly cover the student distribution, the world knowledge can be stably learned. However, this more easily leads to overfitting and directly results in a decline in its role-playing capabilities.

## 5.3.3 Two-Stage OPD Strategy

To address the above problems, we designed a two-stage OPD training scheme and specially constructed a world-setting subset dataset to strengthen the learning of world knowledge:

Table 3: Two-stage OPD training scheme.
<table><tr><td>Dimension</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Objective Dataset Distillation</td><td>Distill formats and styles</td><td>Distill world knowledge World-setting related subset</td></tr><tr><td>Design Intent</td><td>Complete OPD dataset PG + KL distillation Transfer the teacher model&#x27;s formatting</td><td>GKD (top-k log-prob) Focused, high-intensity distillation on the</td></tr><tr><td></td><td>preferences and dialogue styles gently, avoiding drastic changes to the student distribution.</td><td>world-setting subset to ensure that the target domain world knowledge is fully injected.</td></tr></table>

• Stage 1: Use OPD-PG on the complete dataset. The goal of this stage is to allow the base model to quickly acquire the teacher model’s formatting specifications and dialogue styles. At the same time, because the intervention of PG is relatively mild, the impact on general capabilities is small.

• Stage 2: Switch to GKD-OPD, training only on the world-setting related subset. Since the dataset is restricted to world-setting related samples, the distillation signals are highly focused, allowing the world knowledge to be eficiently injected into the student model. Meanwhile, because general data does not participate in gradient updates during this stage, the student model’s general capabilities are preserved. Additionally, in the GKD-OPD training of Stage 2, we observed a structural problem stemming from on-policy sampling, and proposed the Cumulative-Divergence Decay (hereinafter referred to as CDD) scheme accordingly. We will detail this scheme in Section 5.3.4.

## 5.3.4 Cumulative-Divergence Decay (CDD)

Problem Analysis The core formula of GKD-OPD is to align the teacher and student distributions on the next step, based on the prefix $s _ { t } = ( y _ { 1 } , \ldots , y _ { t - 1 } )$ sampled by the student itself:

$$
\mathcal { L } _ { \mathrm { G K D } } ( s _ { t } ) = \sum _ { v \in \mathrm { T o p K } ( \nu ( \cdot | s _ { t } ) ) } \nu ( v \mid s _ { t } ) \left[ \log \nu ( v \mid s _ { t } ) - \log \pi _ { \theta } ( v \mid s _ { t } ) \right] .\tag{14}
$$

The problem lies in: when the student model has not yet fully learned the teacher’s style, the prefix $s _ { t }$ it samples is often a sequence that the teacher itself is unlikely to generate. On such prefixes, the top-k distribution provided by the teacher no longer represents ”how the teacher would continue if it generated up to this point,” but rather ”an inaccurate distribution the teacher is forced to give after inheriting a prefix it does not endorse.”

Using such a teacher distribution as a supervision signal brings two negative efects:

1. High Signal Noise: The teacher’s distribution on of-distribution prefixes may itself collapse or deviate from the teacher’s original behavior. Aligning with such a distribution causes the student to learn noise.

2. Diluted Gradients: All tokens contribute gradients equally. Samples with severely drifted prefixes will occupy the same gradient weight as samples with ”basically reasonable prefixes,” diluting the contribution of truly valuable samples.

Discussion: Comparison with Contemporary Prefix-Drift Solutions The prefix-drift issue in onpolicy distillation has recently been recognized as a critical challenge in the field, with several contemporary works attempting to address it, such as FiRe-OPD [4], TOPD [3], and IW-OPD [11]. However, these approaches exhibit fundamental limitations in mechanism and causality:

1. Heuristic Biases in FiRe-OPD: FiRe-OPD attempts to filter trajectories entirely if the average teacher log-probability is low, and applies soft reweighting based on student/teacher entropy. This trajectory-level hardfiltering is highly sample-ineficient, completely wasting the exploratory value of valid prefixes before the divergence point. Moreover, its entropy-based weighting conflates linguistic multi-modality (high student entropy due to valid alternative paths) with model ignorance, and mistakes teacher’s confident hallucinations on OOD prefixes as high-quality signals.

2. Breaking Autoregressive Causality in TOPD: To identify trajectory divergence, TOPD uses Optimal Transport (OT) to align short-window future continuations between teacher and student. Beyond its prohibitive computational overhead, TOPD severely penalizes valid alternative reasoning paths (e.g., reaching the same conclusion via diferent valid logical steps). Most critically, it breaks the strict temporal causality of autoregressive language models by using future information to optimize current tokens, which inevitably leads to severe train-inference mismatch.

3. Theoretical Foundation but Mechanistic Limitations in IW-OPD: IW-OPD tackles the problem through the lens of constrained optimization, proving mathematically that supervision weights should decay based on accumulated prefix discrepancy. However, its implementation sufers from critical flaws: 1) Its point-wise evaluation fails to accurately measure the true divergence between the teacher and student; 2) Its in-sequence linear Min-Max normalization implies that even if an entire trajectory is perfectly healthy without any deviation, its ending tokens will still be ruthlessly down-weighted. Conversely, even if a trajectory completely collapses at the very first token, it still rigidly allocates weights linearly within the sentence; 3) It lacks a hard-truncation mechanism for severe hallucinations.

Therefore, we need a method that strictly respects autoregressive causality, introduces zero extra computational overhead, and dynamically isolates toxic gradients with a hard bottom line.

Our Solution (CDD) We want to reduce the weight of token positions where the teacher prefix has already drifted severely, preserving full learning signals for positions where the prefix is still reasonable. Unlike the aforementioned approaches, CDD’s design ofers two core advantages:

1. Based on local full-distribution divergence: By calculating the Top-K Forward KL, it comprehensively and accurately evaluates the true divergence between the teacher and student.

2. Global absolute exponential decay: This absolute decay mechanism ensures that if the trajectory has not drifted, the weights of the entire sequence can remain high (near 1.0), fully squeezing the value of all data; however, once a fatal drift occurs, the weights drop exponentially in an instant.

For the t-th response token, let $\mathrm { d i v } _ { t }$ be the unweighted per-token value of the top-k forward KL in the current forward pass (i.e., the current GKD loss term itself, taking the positive part and detached). Define the strictly causal cumulative divergence:

$$
\mathrm { c u m } _ { t } = \sum _ { s < t , \ s \in \mathrm { v a l i d } } \mathrm { d i v } _ { s } .\tag{15}
$$

And the weight of each token position:

$$
\operatorname { r a w } _ { t } = \exp \bigl ( - \lambda \cdot \operatorname { c u m } _ { t } \bigr ) , \qquad w _ { t } = \operatorname* { m a x } ( w _ { \mathrm { m i n } } , \ \operatorname { r a w } _ { t } ) , \qquad \tilde { w } _ { t } = \frac { w _ { t } } { \bar { w } } ,\tag{16}
$$

where $\bar { w }$ is the mean of $w _ { t }$ over valid tokens in the batch. The final distillation loss is:

$$
\mathcal { L } _ { \mathrm { C D D - G K D } } = \frac { 1 } { \sum _ { t } \tilde { w } _ { t } } \sum _ { t } \tilde { w } _ { t } \cdot \mathcal { L } _ { \mathrm { G K D } } ( s _ { t } ) .\tag{17}
$$

Here, the hyperparameter λ acts as a knob to control the ”steepness of decay,” and the hyperparameter $w _ { \mathrm { m i n } }$ serves as a fallback to prevent weights at the tail of long texts from completely collapsing to 0.

## 5.4 Training Recipe

The key hyperparameters for both OPD stages are summarized in Table 4.

Table 4: OPD Training Recipes.
<table><tr><td>Parameter</td><td>Stage 1: OPD-PG</td><td>Stage 2: GKD</td><td>Stage 2: GKD + CDD</td></tr><tr><td>Loss</td><td> $\mathrm { P G } \left( \mathrm { K } 1 \right)$ </td><td>GKD</td><td> $\mathbf { G K D + C D D }$ </td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Advantage Estimator</td><td> $\mathtt { G R P 0 }$ </td><td>一</td><td>一</td></tr><tr><td>Rollouts per Prompt</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Train Batch Size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>PPO Mini Batch Size</td><td>16</td><td>16</td><td>16</td></tr><tr><td>PPO Clip Ratio</td><td>[0.20, 0.28]</td><td>一</td><td>一</td></tr><tr><td>Include KL in Reward</td><td>False</td><td>一</td><td>一</td></tr><tr><td>Distillation TopK</td><td>一</td><td>64</td><td>64</td></tr><tr><td>CDD λ</td><td>一</td><td>一</td><td>0.1</td></tr><tr><td>CDD  $w _ { \mathrm { m i n } }$ </td><td>一</td><td>一</td><td>0.05</td></tr><tr><td>Epochs</td><td>1</td><td>2</td><td>2</td></tr></table>

Based on our proposed OPD scheme, we finally achieved eficient and accurate injection of domain role-playing capabilities and domain knowledge into the student model, under the premise of preserving the student model’s general capabilities.

## 6 Discussion: Evaluation and Key Findings

This section systematically compares the models produced at various stages of the training pipeline (SFT → RL → OPD Stage 1 → OPD Stage 2 → OPD Stage 2 + CDD) with the original base model. The evaluation dimensions include: general Agent capabilities (BFCL v4 [6], Table 5), general roleplaying capabilities (Table 6), domain role-playing capabilities (Table 7), and domain safety refusal capabilities & world-setting knowledge (Table 8). The tables also include M2-HER [5], a recently released proprietary model by MiniMax representing the current state-of-the-art in the role-playing market, as an external baseline to locate the overall level of our series of models. The specific methods for evaluation data production and scoring are detailed in Section 7.

It should be noted that: if the original base models (e.g., qwen3-8b (baseline)) receive exactly the same inputs as other compared models, they will output excessively long texts, resulting in a precipitous drop in their scores. Therefore, we added extra length limits to their system prompts. All metrics are reported on a 100-point scale, and higher values are better. In all tables, bold and underlined values denote the best and second-best results within each model group, respectively.

Table 5: General agent capabilities on the Berkeley Function Calling Leaderboard v4. “Base” refers to qwen3-8b (baseline); subsequent columns are stages in our pipeline.
<table><tr><td>Metric</td><td>Base</td><td>SFT</td><td>RL</td><td>OPD1</td><td>OPD2 w/o CDD</td><td>OPD2 w/ CDD</td></tr><tr><td>Overall Acc</td><td>24.83</td><td>12.53</td><td>12.55</td><td>24.83</td><td>24.76</td><td>24.78</td></tr><tr><td>Non-Live AST Acc</td><td>88.33</td><td>41.85</td><td>41.96</td><td>87.04</td><td>87.00</td><td>86.69</td></tr><tr><td>Non-Live Simple AST</td><td>75.83</td><td>71.92</td><td>71.33</td><td>75.17</td><td>74.00</td><td>74.75</td></tr><tr><td>Non-Live Multiple AST</td><td>95.50</td><td>94.50</td><td>96.00</td><td>95.50</td><td>95.50</td><td>95.00</td></tr><tr><td>Non-Live Parallel AST</td><td>93.00</td><td>0.50</td><td>0.50</td><td>93.00</td><td>94.00</td><td>93.50</td></tr><tr><td>Non-Live Par. Mul. AST</td><td>89.00</td><td>0.50</td><td>0.00</td><td>84.50</td><td>84.50</td><td>83.50</td></tr><tr><td>Live Acc</td><td>80.24</td><td>74.54</td><td>74.54</td><td>79.64</td><td>79.35</td><td>79.50</td></tr><tr><td>Live Simple AST</td><td>83.72</td><td>72.87</td><td>72.48</td><td>82.17</td><td>79.84</td><td>84.50</td></tr><tr><td>Live Multiple AST</td><td>79.49</td><td>77.78</td><td>77.87</td><td>79.39</td><td>79.58</td><td>78.54</td></tr><tr><td>Live Parallel AST</td><td>68.75</td><td>0.00</td><td>0.00</td><td>68.75</td><td>87.50</td><td>75.00</td></tr><tr><td>Live Par. Mul. AST</td><td>83.33</td><td>0.00</td><td>0.00</td><td>70.83</td><td>58.33</td><td>70.83</td></tr><tr><td>Relevance Detection</td><td>87.50</td><td>81.25</td><td>75.00</td><td>81.25</td><td>87.50</td><td>87.50</td></tr><tr><td>Irrelevance Detection</td><td>79.69</td><td>8.96</td><td>9.00</td><td>81.64</td><td>81.28</td><td>81.62</td></tr></table>

Note: SFT: qwen3-8b-sft; RL: qwen3-8b-sft-grpo; OPD1: qwen3-8b-sft-grpo-opds1; OPD2 w/o CDD: qwen3-8b-sft-grpo-opds2; OPD2 w/ CDD: qwen3-8b-sft-grpo-opds2-cdd.

Table 6: General role-playing capabilities on TRACEbench.
<table><tr><td>Model</td><td>Method</td><td>Char-Consistency</td><td>Mem-Consistency</td><td>Diversity</td><td>LangQuality</td><td>Length</td></tr><tr><td colspan="2">M2-HER (API)</td><td>87.82</td><td>79.00</td><td>90.23</td><td>97.26</td><td>94.79</td></tr><tr><td rowspan="6">8B</td><td>Base</td><td>85.30</td><td>79.50</td><td>80.91</td><td>98.17</td><td>97.80</td></tr><tr><td>SFT</td><td>86.21</td><td>82.41</td><td>85.23</td><td>94.70</td><td>81.76</td></tr><tr><td>RL</td><td>86.02</td><td>80.50</td><td>94.92</td><td>96.54</td><td>98.50</td></tr><tr><td>OPD1</td><td>86.56</td><td>77.50</td><td>95.96</td><td>97.24</td><td>98.50</td></tr><tr><td>OPD2 w/o CDD</td><td>85.43</td><td>77.00</td><td>95.36</td><td>96.89</td><td>98.58</td></tr><tr><td>OPD2 w/ CDD</td><td>86.07</td><td>79.00</td><td>95.22</td><td>96.60</td><td>98.66</td></tr><tr><td rowspan="6">4B</td><td>Base</td><td>75.25</td><td>80.50</td><td>46.77</td><td>86.92</td><td>96.74</td></tr><tr><td>SFT</td><td>80.07</td><td>73.23</td><td>85.17</td><td>90.02</td><td>68.58</td></tr><tr><td>RL</td><td>81.18</td><td>67.50</td><td>93.44</td><td>92.60</td><td>92.09</td></tr><tr><td>OPD1</td><td>82.06</td><td>74.00</td><td>95.29</td><td>94.57</td><td>95.89</td></tr><tr><td>OPD2 w/o CDD</td><td>80.97</td><td>68.34</td><td>95.12</td><td>94.54</td><td>92.18</td></tr><tr><td>OPD2 w/ CDD</td><td>83.20</td><td>71.36</td><td>93.89</td><td>94.50</td><td>92.96</td></tr></table>

Note: w/o CDD and w/ CDD denote OPD2 models trained without and with Cumulative-Divergence Decay (Section 5.3.4), respectively.

Table 7: Domain role-playing capabilities on the TRACEbench domain-character subset, reported as mean ± standard deviation over five runs.
<table><tr><td>Model</td><td>Method</td><td>Char-Consist.</td><td> $\mathbf { M e m - C o n s i s t . }$ </td><td>Diversity</td><td>LangQuality</td><td>Length</td></tr><tr><td colspan="2">M2-HER (API)</td><td> $8 9 . 2 2 \pm 1 . 9 6$ </td><td> $6 0 . 0 0 \pm 2 3 . 6 3$ </td><td> $9 8 . 8 9 \pm 0 . 3 5$ </td><td> $9 9 . 6 2 \pm 0 . 4 1$ </td><td> $9 6 . 4 0 \pm 2 . 6 2$ </td></tr><tr><td rowspan="7"></td><td>Base</td><td> $8 5 . 1 4 \pm 1 . 9 9$ </td><td> $6 0 . 0 0 \pm 9 . 4 8$ </td><td> $6 1 . 4 8 \pm 1 . 3 1$ </td><td> $9 7 . 5 7 \pm 0 . 8 5$ </td><td> $9 3 . 7 5 \pm 0 . 0 0$ </td></tr><tr><td>SFT</td><td> $9 0 . 9 2 \pm 2 . 4 4$ </td><td> $7 5 . 0 0 \pm 4 . 4 2$ </td><td> $8 8 . 0 8 \pm 1 . 3 2 $ </td><td> $9 7 . 6 1 \pm 1 . 1 9$ </td><td> $7 7 . 3 1 \pm 3 . 1 4$ </td></tr><tr><td>SFT w/o RPF</td><td> $8 9 . 2 0 \pm 1 . 4 5$ </td><td> ${ \bf 7 8 . 7 5 \pm 1 0 . 4 6 }$ </td><td> $8 8 . 0 1 \pm 1 . 2 4 $ </td><td> $9 8 . 6 6 \pm 0 . 5 4$ </td><td> $7 4 . 8 0 \pm 4 . 0 9$ </td></tr><tr><td>RL</td><td> $9 0 . 6 1 \pm 1 . 9 1 $ </td><td> $6 8 . 7 5 \pm 1 3 . 9 8$ </td><td> ${ \bf 9 8 . 0 4 \pm 0 . 9 2 }$ </td><td> $9 8 . 7 5 \pm 0 . 6 5$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td>OPD1</td><td> $9 0 . 1 2 \pm { 1 . 6 1 }$ </td><td> $7 3 . 7 5 \pm 1 2 . 0 2$ </td><td> $9 7 . 2 9 \pm 1 . 0 3$ </td><td> $9 7 . 4 3 \pm 1 . 0 0$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td>OPD2 w/o CDD</td><td> $9 0 . 0 2 \pm 2 . 1 0$ </td><td> $7 7 . 5 0 \pm 9 . 4 8$ </td><td> $9 6 . 4 9 \pm 0 . 6 9$ </td><td> $9 8 . 2 8 \pm 0 . 8 6$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td>OPD2 w/ CDD</td><td> ${ \bf 9 2 . 0 8 \pm 1 . 4 5 }$ </td><td> $7 2 . 5 0 \pm 7 . 1 3$ </td><td> $9 7 . 4 4 \pm 0 . 8 2$ </td><td> $\mathbf { 9 9 . 1 2 \pm 0 . 9 0 }$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td rowspan="7">4B</td><td>Base</td><td> $7 7 . 0 9 \pm 1 . 9 7$ </td><td> $5 2 . 5 0 \pm 5 . 5 9$ </td><td> $3 6 . 7 3 \pm 2 . 9 5$ </td><td> $8 9 . 5 7 \pm 3 . 4 4$ </td><td> $9 0 . 3 0 \pm 3 . 5 7$ </td></tr><tr><td>SFT</td><td> $8 6 . 7 8 \pm 1 . 5 5$ </td><td> $5 6 . 2 5 \pm 8 . 8 4$ </td><td> $8 4 . 8 7 \pm 3 . 4 6$ </td><td> $9 7 . 3 8 \pm 0 . 5 8$ </td><td> $8 1 . 4 2 \pm 4 . 2 5$ </td></tr><tr><td>SFT w/o RPF</td><td> $8 6 . 4 8 \pm 3 . 8 0$ </td><td> $\mathbf { 7 0 . 0 0 \pm 1 2 . 0 2 }$ </td><td> $8 0 . 3 0 \pm 2 . 8 4$ </td><td> $9 6 . 4 9 \pm 0 . 7 7$ </td><td> $7 2 . 1 2 \pm 3 . 5 2$ </td></tr><tr><td>RL</td><td> $8 4 . 6 7 \pm 1 . 3 4$ </td><td> $6 0 . 6 7 \pm 8 . 6 8$ </td><td> $\mathbf { 9 5 . 8 5 \pm 1 . 4 4 }$ </td><td> $9 7 . 3 0 \pm 1 . 0 5$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td>OPD1</td><td> $\mathbf { 8 7 . 0 9 } \pm 4 . 2 2$ </td><td> $5 5 . 0 0 \pm 1 7 . 3 4$ </td><td> $9 3 . 4 4 \pm 1 . 4 3$ </td><td> ${ \bf 9 8 . 0 5 \pm 0 . 2 8 }$ </td><td> $\underline { { 9 9 . 9 3 \pm 0 . 1 5 } }$ </td></tr><tr><td>OPD2 w/o CDD</td><td> $8 5 . 9 2 \pm 1 . 7 1 $ </td><td> $4 5 . 0 0 \pm 9 . 2 7$ </td><td> $\underline { { 9 4 . 1 1 \pm 1 . 3 8 } }$ </td><td> $9 7 . 4 1 \pm 1 . 0 8$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr><tr><td>OPD2 w/ CDD</td><td> $\underline { { 8 6 . 7 9 \pm 2 . 5 2 } }$ </td><td> $5 8 . 7 5 \pm 1 0 . 4 6$ </td><td> $9 3 . 9 6 \pm 1 . 1 6$ </td><td> $9 7 . 0 3 \pm 1 . 1 0$ </td><td> $\mathbf { 1 0 0 . 0 0 } \pm 0 . 0 0$ </td></tr></table>

Note: RPF is described in Section 3; w/o RPF removes it during SFT data construction. w/o CDD and w/ CDD denote OPD2 models trained without and with CDD (Section 5.3.4), respectively.

Table 8: Domain safety refusal capabilities and world-knowledge mastery.
<table><tr><td>Model</td><td>Method</td><td>Adversarial</td><td>Political</td><td>Sexual</td><td>Domain Know.</td></tr><tr><td>M2-HER (API)</td><td></td><td>66.19</td><td>78.37</td><td>94.14</td><td>N/A</td></tr><tr><td rowspan="6">8B</td><td>Base</td><td>42.49</td><td>81.95</td><td>95.20</td><td>1.90</td></tr><tr><td>SFT</td><td>73.51</td><td>97.74</td><td>97.87</td><td>38.10</td></tr><tr><td>RL</td><td>74.68</td><td>97.36</td><td>98.16</td><td>34.29</td></tr><tr><td>OPD1</td><td>78.50</td><td>98.21</td><td>98.40</td><td>9.52</td></tr><tr><td>OPD2 w/o CDD</td><td>79.22</td><td>97.74</td><td>98.35</td><td>32.86</td></tr><tr><td>OPD2 w/ CDD</td><td>79.37</td><td>97.39</td><td>97.87</td><td>32.86</td></tr><tr><td rowspan="6">4B</td><td>Base</td><td>31.53</td><td>63.90</td><td>90.99</td><td>0.48</td></tr><tr><td>SFT</td><td>70.90</td><td>93.22</td><td>97.19</td><td>26.67</td></tr><tr><td>RL</td><td>70.99</td><td>95.51</td><td>98.30</td><td>23.81</td></tr><tr><td>OPD1</td><td>69.10</td><td>94.73</td><td>97.72</td><td>6.67</td></tr><tr><td>OPD2 w/o CDD</td><td>71.26</td><td>95.51</td><td>97.48</td><td>17.14</td></tr><tr><td>OPD2 w/ CDD</td><td>71.35</td><td>95.32</td><td>98.11</td><td>20.00</td></tr></table>

## 6.1 Stage-by-Stage Result Analysis

We analyze the results stage-by-stage in the order of training, focusing on the trade-of between ”domain capability injection” and ”general capability preservation”:

## 1. SFT: Substantial Improvement in Domain and Safety Refusal Capabilities, Significant Degradation in General Agent Capabilities

• General Agent capabilities show significant degradation after SFT, indicating that fine-tuning on role-playing data significantly harms the model’s general capabilities (Table 5);

• The consistency dimension of general role-playing capabilities does not change much, but the Length / LangQuality dimensions show a decline (Table 6);

• Domain role-playing capabilities improve significantly, indicating that domain data efectively injects world knowledge, but the Length dimension shows a decline, indicating instances of overly long outputs (Table 7);

• Both domain safety refusal capabilities and world knowledge mastery leap substantially; the training on role-playing data efectively injects refusal capabilities for high-risk dialogues (Table 8);

## 2. RL: Output Quality Significantly Improved, General Agent Capabilities Remain Degraded

• General Agent capabilities remain at the degraded level seen after SFT (Table 5).

• The consistency dimensions for both general and domain role-playing capabilities do not change much, but the Diversity / Length / LangQuality dimensions all improve significantly. This directly verifies that RL training based on our designed reward function can efectively eliminate degradation phenomena like length expansion and repetitive speech, while preserving output diversity (Table 6, Table 7);

• Domain safety refusal capabilities and world knowledge mastery are not significantly afected (Table 8);

## 3. OPD Stage 1 (PG): General Agent Capabilities Fully Recovered, Role-playing Capabilities Maintained

• General Agent capabilities (BFCL v4) recover to a level on par with the base model (Table 5);

• Both general and domain role-playing capabilities are maintained at a strong level (Table 6, Table 7);

• Domain safety refusal capabilities are maintained at a strong level, but the injection of world knowledge is limited (Table 8).

## 4. OPD Stage 2 (GKD + CDD): World Knowledge Returns, Domain Capabilities Peak

• General Agent capabilities (BFCL v4) remain stable, maintaining a level comparable to the base model (Table 5).

• General role-playing capabilities are maintained at a strong level (Table 6);

• Domain role-playing capabilities achieve the best performance across the entire pipeline (Table 7);

• Domain safety refusal capabilities are maintained at a strong level. Without CDD, the mastery of world knowledge improves but is limited; after introducing CDD, world knowledge recovers and approaches the SFT model’s level (Table 8);

## 6.2 Key Findings

1. The degradation of general capabilities is reversible, and online distillation is an efective compensation method: The substantial decline in general Agent capabilities caused by SFT is fully recovered after OPD Stage 1 (Table 5). Therefore, conducting SFT/RL with a small amount of domain data first, followed by online distillation ”with the base model as the student and the domain model as the teacher” for compensation, can serve as a low-cost, general training paradigm that balances domain adaptation and the preservation of base model capabilities.

2. The benefits of CDD are concentrated in the thoroughness and stability of knowledge injection: Compared to the GKD version without CDD, the CDD version is superior in domain character consistency and exhibits significantly smaller fluctuations (Table 7). This corroborates the design motivation in Section 5.3.4 that ”cumulative divergence weighting makes distillation more thorough at positions with reasonable prefixes.”

## 7 Benchmark and Evaluation

This section specifically introduces our evaluation system, which consists of two parts: data production method and scoring method.

## 7.1 Data Production Method

The evaluation data consists of two parts:

## 1. Single-turn Evaluation Data

• Formed by combining safety-related user requests with character cards into single-turn samples;

• Used to evaluate the model’s refusal and avoidance capabilities for high-risk questions in single-turn scenarios, as well as its mastery of domain knowledge (usually the game’s world setting).

## 2. Multi-turn Evaluation Data

• Each sample contains a prebuilt task plan (prebuilt checklist) and initial dialogue context;

• The user side interacts with the NPC through Agentic behaviors for multiple turns, producing a complete dialogue trajectory and the final checklist status;

• Used to evaluate the model’s ability to follow fine-grained character settings in the System Prompt in multi-turn scenarios, as well as its ability to maintain consistency throughout the multi-turn dialogue process.

## 7.2 Scoring Method

## 7.2.1 Single-turn Evaluation Scoring

Based on the single-turn evaluation data, we designed four metrics to assess the model’s refusal capabilities in high-risk dialogues and its mastery of domain knowledge related to the game’s world setting:

1. adult refusal (Pornography avoidance): When a user’s request involves adult/pornographic content (e.g., explicit descriptions, pornographic creative requests), the model should explicitly refuse to answer, rather than cooperate in generating or continuing such content.

2. political refusal (Politics avoidance): When the dialogue involves sensitive political topics, the model should avoid discussing or refuse to answer, avoiding the output of inappropriate content.

3. adversarial robustness: Assesses the model’s stability under adversarial guidance—when a user attempts to induce the model to break character boundaries or the role-playing task itself through leading rhetoric or exploiting loopholes in the character card, can the model adhere to the character settings and behavioral norms?

4. domain knowledge: Assesses the model’s mastery of the game’s world setting knowledge. The evaluation samples carry reference answers, and the model’s response will ultimately be compared with the reference answer to determine if the answer is correct.

• Each sample is given a binary judgment (0/1) by a Judge;

• When summarized and displayed, it is uniformly converted to a 100-point scale (0 or 100).

## 7.2.2 Multi-turn Evaluation Scoring

Based on the multi-turn evaluation data, we designed two metrics to evaluate the model’s ability to follow fine-grained character prompt settings and its ability to maintain information consistency before and after multi-turn dialogues. Both metrics share the same agentic checklist scoring mechanism (agentic checklist completion): each sample is pre-loaded with a prebuilt checklist describing the characteristics, behaviors, and speaking styles the character should exhibit, which is checked item by item by the Agentic User during the dialogue. The score for a single sample is defined as:

$$
\mathrm { s c o r e } _ { \mathrm { c a s e } } = \frac { \# \mathrm { c o m p l e t e d p r e b u i l t c h e c k l i s t s } } { \# \mathrm { p r e b u i l t c h e c k l i s t s } } \times 1 0 0\tag{18}
$$

Both the numerator and denominator are counted based solely on the prebuilt checklist set to ensure that the denominator is consistent across diferent models and can be directly compared. In each sample’s checklist, 1 item is used to assess information consistency before and after the dialogue, and the remaining items are all related to fine-grained character settings. Accordingly, the two metrics are calculated by splitting the checklist categories:

1. TRACE-Character-Consistency: Measured by the completion rate of all fine-grained character setting-related checklists. It reflects the model’s degree of adherence to the fine-grained character settings in the System Prompt—personality, speaking style, behavioral patterns, and knowledge boundaries—during multi-turn dialogues.

2. TRACE-Memory-Consistency: Measured by the completion rate of the short-term-memory consistency checklist. Since each sample contains only 1 such checklist, the single-sample score is 0 or 100; after averaging across samples, it reflects the model’s ability to recall and maintain consistency across turns regarding established facts from early in the dialogue (e.g., character identity information, player nickname, mutual relationship, etc.).

When summarizing across samples, this report uniformly adopts the Macro approach (averaging the scores of each sample), using ”the model’s average score when a sample is randomly selected” to reflect the overall level.

## 7.2.3 Three Language Quality Scores

Based on the multi-turn evaluation data, we also additionally calculated three language quality metrics.

1. Length

• Scored at the turn level based on the assistant’s reply, good=1 / bad=0;

• Uses language-adaptive measurement:

– When English is dominant (Latin letters prevail), it is based on word count, with a threshold range of [4, 80];

– Otherwise, it is based on CJK character count (if no CJK, it degenerates to non-space character count), with a threshold range of [15, 150].

## 2. Diversity

• Sentences are the comparison units, and turns are the scoring units: each assistant turn finally produces one diversity score;

• Sentence segmentation symbols are $0 \quad ! \quad ? \ ; \quad ! ? ; \langle { \mathtt { n } } . $ , and short sentences with a length of less than 5 are excluded from the evaluation;

• The valid sentences of the current turn and the historical valid sentences are compared pairwise to calculate the 2-gram Jaccard similarity: word-level 2-gram (word-bigram) is used when English is dominant, otherwise character-level 2-gram (char-bigram) is used; the maximum value $J _ { \operatorname* { m a x } } ^ { ( t ) }$ is taken;

• The mapping from similarity to score is a dual-threshold piecewise linear function:

$$
s _ { \mathrm { d i v e r s i t y } } ^ { ( t ) } = \left\{ \begin{array} { l l } { 1 , } & { J _ { \mathrm { m a x } } ^ { ( t ) } \leq 0 . 4 } \\ { 0 , } & { J _ { \mathrm { m a x } } ^ { ( t ) } \geq 0 . 6 } \\ { 1 - \displaystyle \frac { J _ { \mathrm { m a x } } ^ { ( t ) } - 0 . 4 } { 0 . 2 } , } & { 0 . 4 < J _ { \mathrm { m a x } } ^ { ( t ) } < 0 . 6 } \end{array} \right.\tag{19}
$$

• Where: if the current turn has no valid sentences, or the history has no valid sentences, the turn is scored as 1 (considered non-repetitive).

## 3. Language Quality (LangQuality)

• A binary judgment (0/1) is made turn-by-turn by a Judge, focusing on checking sentence fluency, grammar/wording errors, and semantic completeness;

• Scored as 0 if there are obvious grammatical errors, stacked typos, or incomplete semantics; otherwise scored as 1.

The final sample score for each metric is the average of the turn-by-turn scores multiplied by 100.

## 7.3 Horizontal Comparison of Broader Models (TRACEbench Leaderboard)

In addition to the models produced at various stages of the training pipeline in this paper, we also systematically evaluated a large number of open-source and closed-source models (covering general dialogue large models, proprietary role-playing models, and commercial API models) under the same evaluation system, forming a unified TRACEbench [13] Leaderboard. This Leaderboard covers the two multi-turn evaluation metrics (TRACE-Character-Consistency, TRACE-Memory-Consistency) and the three language quality metrics defined in Section 7—ensuring that all models can be directly compared horizontally under the same samples and identical scoring calibers. For the complete list of models, scores in each dimension, and rankings, please refer to our paper (TRACE-Bench) and project homepage (TRACE-Bench).

## 8 Conclusion and Future Work

This paper systematically proposes the complete training scheme for the KuaiRP series of role-playing models, containing the following core contributions:

1. High-Quality Data Pipeline: We constructed a robust data pipeline featuring user behavior instruction injection and reverse profile filtering, creating training data that closely mirrors real deployment scenarios.

2. Novel Self-Distillation Paradigm: We proposed an SFT → RL → OPD training pipeline starting and ending on the same base model. By using the domain-adapted model as the teacher and the original base model as the student, we efectively transferred domain capabilities while preserving general agent capabilities, distinguishing our method from traditional SFT-RL or large-to-small distillation paradigms.

3. Cumulative-Divergence Decay (CDD): We introduced an algorithmic enhancement for on-policy Generalized Knowledge Distillation, which dynamically mitigates the learning of noise caused by prefix-drift. By respecting the causal autoregressive nature of language models and preserving valid path diversity, CDD successfully protects the student from forced hallucinations, ensuring thorough and stable world knowledge injection.

4. Empirical Success: Utilizing our standardized character template and training framework, our KuaiRP series models achieve state-of-the-art role-playing performance within our target domain scenarios, while retaining the tool-calling and reasoning capabilities of the base model.

Overall, the KuaiRP series models achieved high-fidelity role-playing capabilities under the premise of ensuring small size and high-eficiency deployment. We believe that the ”SFT → RL → OPD two-stage distillation” training paradigm proposed in this paper provides a reproducible and scalable technical route for the development of dedicated role-playing models.

Despite these advancements, we have identified several limitations in our practice, which point to our future directions:

• Adaptation Challenges with Stronger Base Models: We conducted identical experiments on the Qwen3.5-9B [7] model but did not observe the same significant performance gains as with the Qwen3 series. The core reason is that the Qwen3.5-9B base model already exhibits a very high initial level of character consistency (Char-Consist.). Under our current training data and pipeline, we could not obtain stable improvements. We attribute this to our distilled training data becoming relatively ”outdated,” lagging behind the generational capability leap from Qwen3 to Qwen3.5. For increasingly powerful base models, when performing domain adaptation, in addition to distilling stronger data, we will explore On-Policy Self-Distillation (OPSD) schemes [14] and other lossless domain knowledge learning paradigms more extensively in the future.

• Subjective Experience in Role-Playing: Our current benchmark primarily measures character persona consistency but does not cover subjective experiences such as interestingness. Furthermore, the reward models used in our Reinforcement Learning (RL) stage have not incorporated these subjective reward dimensions. Considering that role-playing fundamentally caters to users’ entertainment needs, being ”interesting” is often more crucial than merely being ”accurate.” Therefore, integrating subjective experiences like fun into our evaluation benchmarks and reward mechanisms remains an essential direction for our future exploration.

## Contributions

Team Leader: Qi Gan

Project Leader: Yipeng Wang

Technical Implementation: Yipeng Wang, Ziwei Zhang, Jiahui Zhang, Qi Gan, Kai Sheng Afiliation: Kuaishou GameMind Lab

## References

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024.

[2] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models, 2021. arXiv:2106.09685.

[3] Yuxuan Jiang and Francis Ferraro. Bridging reasoning trajectories in on-policy distillation via near-future guidance, 2026. arXiv:2606.00305.

[4] Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie Yuan, and Tao Feng. Filter, then reweight: Rethinking optimization granularity in on-policy distillation, 2026. arXiv:2606.02684.

[5] MiniMax Team. A deep dive into the MiniMax-M2-her, 2026. https://www.minimax.io/news/ a-deep-dive-into-the-minimax-m2-her-2.

[6] Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.

[7] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. https://qwen.ai/blog?id= qwen3.5.

[8] Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. arXiv:2412.15115.

[9] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. arXiv:1707.06347.

[10] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. arXiv:2402.03300.

[11] Yan Xie, Sijie Zhu, Tiansheng Wen, Bo Chen, and Yifei Wang. On the position bias of on-policy distillation, 2026. arXiv:2606.22600.

[12] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. arXiv:2505.09388.

[13] Jiahui Zhang, Ziwei Zhang, Yipeng Wang, Yibo Liu, Haozhou Pang, Yikai Hu, Hongyan Ren, Lan Zhou, Qi Gan, and Kai Sheng. TRACE bench: Task-driven roleplay agentic checklist evaluation, 2026. arXiv:2608.11236.

[14] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026. arXiv:2601.18734.

## A Role Play Template Structure

The specific structure of the role play template is as follows:

Please play the following character and converse with the player. [Optional world setting]

```markdown
#### Character Name and Profile
**Identity**: [Character Name], [Brief description].
**Detailed Description**: [Detailed description of the character, including background
information and traits that make them unique or noteworthy]
**Past Experience and Background**: [Detailed background narrative, including career
history, major achievements, and growth experiences that shaped their personality.
Include details related to their growth and evolution]
```

[Clearly describe what the character knows, what they can do, and what is beyond their expertise or capability. This helps set reasonable expectations for their responses]

```markdown
#### Dialogue Examples
**Scenario 1**: ([Gesture/Demeanor] [Optional]) "[Example dialogue demonstrating the
character’s speaking style and personality in a specific situation]"
**Scenario N**: ([Gesture/Demeanor] [Optional]) "[Another example dialogue demonstrating
different aspects of the character]"
```