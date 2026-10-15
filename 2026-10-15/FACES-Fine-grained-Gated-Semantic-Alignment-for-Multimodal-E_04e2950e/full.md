# FACES: Fine-grained Gated Semantic Alignment for Multimodal Emotional Support Conversation

Hang Ren, Peidong Wang, Yichen Gao, Wen Zhang Jiayang Yu, Xiaocui Yang, Shi Feng<sup>†</sup>

School of Computer Science and Engineering, Northeastern University, China {renh2, gaoyc3, zhangw6}@mails.neu.edu.cn {pdongwang, yujiayang}@stumail.neu.edu.cn {yangxiaocui, fengshi}@cse.neu.edu.cn

## Abstract

In recent years, with advances in multimodal interaction technologies and the growing demand for intelligent emotional support, Multimodal Emotional Support Conversation (MESC) has emerged as an important task. However, existing systems still suffer from shallow responses and inappropriate support strategies due to weak semantic feedback, cross-dimension reward compensation, and coarse reasoning supervision. To address these challenges, we propose FACES, a fine-grained gated semantic alignment framework for MESC. Specifically, FACES integrates multi-source rewards to construct multidimensional semantic and functional alignment signals, employs a noncompensatory gated fusion mechanism to reduce cross-dimension reward offset, and leverages structured Chain-of-Thought (CoT) anchors to perform token-level credit assignment. Under a controlled response-length protocol, FACES improves the average MLLM-based evaluation score from 3.61 to 4.04 and achieves an 80.22% pairwise win rate over SOTA. Human evaluation under the same protocol yields a 75.3% preference rate. Results on three external datasets further support transferabil ity. The code and dataset are available at https://github.com/OCT-ARR/arr.

## 1 Introduction

Multimodal Emotional Support Conversation (MESC) (Chu et al., 2025) extends ESC (Liu et al., 2021) by generating empathetic, appropriate, and supportive responses from dialogue context and multimodal cues such as facial expressions, postures, and interaction context (Chaubey et al., 2026; Lu et al., 2025). Beyond producing a superficially natural response, MESC requires models to infer users’ emotions, underlying causes, and support needs while grounding these in visual evidence. Correct emotion recognition alone does not ensure appropriate support (Liu et al., 2021; Wang et al., 2024a). In Figure 1, the baseline identifies anger and relationship conflict but merely paraphrases the user, illustrating the gap between understanding and supportive action.

![](images/3bbe5c71ac2af62a7e57e471079149a9fc0c27a0a5bad2b88ec17362a28efb7d.jpg)  
Figure 1: A MESC example. Both methods use flexible thinking to adaptively select contextually relevant reasoning aspects, while FACES adds reward alignment, recognizing the user’s feeling of being misunderstood and asking a targeted follow-up question.

To better understand users and select support strategies, prior work has advanced MESC through dataset construction (Chu et al., 2025), multimodal modeling (Le et al., 2026), and reasoning-based training for support generation (Wang et al., 2025a). Complementary work personalizes multimodal support through memory (Tu et al., 2025) and links emotion recognition and cause inference to support generation (Tu et al., 2026). Reasoning-based methods use structured CoT (Wei et al., 2022) to make explicit how inferences about user emotions and underlying causes guide support strategy selection. In particular, FIRES (Wang et al., 2025a) introduces flexible thinking and shows that reinforcement learning (RL) objectives improve structured reasoning and final response quality in MESC.

Despite this progress, existing RL objectives for MESC still face three limitations in translating intermediate understanding into effective support. First, surface-overlap metrics such as BLEU and ROUGE-L (Papineni et al., 2002; Lin, 2004) poorly capture cross-modal grounding, semantic consistency, and supportive intent, providing weak feedback on whether a response meets the user’s support needs. Second, additive reward fusion can let strong understanding scores mask a poor support strategy or ineffective response (Van Moffaert et al., 2013). Accurate emotion or situation understanding alone cannot compensate for these support failures. Third, existing MESC methods relying on completion-level rewards (Wang et al., 2025a) lack stage-specific credit for visual grounding, emotion and situation understanding, strategy planning, or response generation. This coarse feedback makes it difficult to identify and correct the reasoning errors underlying an inappropriate support response.

To address these issues, we propose FACES, an MESC framework that uses gated multidimensional rewards to guide fine-grained token updates. Specifically, FACES uses CLIP (Radford et al., 2021) for visual perception, MiniLM (Wang et al., 2020) for emotion understanding, and a multimodal large language model (MLLM) judge (Zheng et al., 2023; Chen et al., 2024) for situation understanding, support strategy, and response outcome. After a format-and-outcome warm-up, non-compensatory gated reward fusion limits compensation for weak strategy or response quality. Reasoning-stage markers route dimension-specific token-level advantages to the corresponding CoT and response spans, jointly optimizing intermediate reasoning and final response generation.

Our main contributions are as follows:

• We propose a multi-source semantic reward scheme that provides more reliable semanticlevel supervision than surface-overlap metrics.

• We introduce format-and-outcome warm-up with non-compensatory gated reward fusion to reduce cross-dimension compensation.

• We propose fine-grained semantic alignment rewards for structured CoT and outcome generation, aligning dimension-specific feedback with token spans to jointly optimize intermediate reasoning and the final response.

Under a 15-word response cap (cap-15) on MESC-FT, FACES raises the average MLLMbased evaluation score from 3.61 to 4.04, achieves an 80.22% pairwise win rate, and obtains a 75.3% human win rate over the SOTA baseline FIRES. Results on MELD-ESC300, StickerConv, and AvaMERG support cross-dataset transferability.

## 2 Related Work

## 2.1 MESC Research

Compared with traditional ESC (Liu et al., 2021), MESC uses multimodal cues alongside dialogue, providing additional evidence about the helpseeker’s emotion and situation when words are ambiguous (Chu et al., 2025; Park et al., 2024). Two representative approaches have emerged. The first fuses multimodal signals to infer emotion and guide support generation, through sequential emotion–strategy–response modeling (Chu et al., 2025) or psychologically informed optimization (Le et al., 2026). The second makes intermediate decisions explicit, using flexible reasoning over visual context, emotion, situation, and strategy (Wang et al., 2025a), or appraisal-guided reasoning and retrieval (Chu et al., 2026). But remaining challenges include weak semantic feedback, compensation across reward dimensions, and insufficient supervision for intermediate reasoning; these are the problems FACES aims to address.

## 2.2 Reward Evaluation and Fusion

Reward evaluation and fusion aim to build reliable supervision and coordinate multiple objectives. Preference-based learning can optimize a policy through a learned reward model (Christiano et al., 2017; Ouyang et al., 2022; Stiennon et al., 2020; Bai et al., 2022). Later work develops psychologygrounded, multi-perspective reward models (Wang et al., 2026a). MultiMood (Le et al., 2026) combines psychological criteria with PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024); G-Eval (Liu et al., 2023), Prometheus (Kim et al., 2024), PandaLM (Wang et al., 2024b), STICKERCONV (Zhang et al., 2024), and related studies (Zhao et al., 2024) demonstrate the utility of model-based evaluation. However, linear fusion can allow strong dimensions to mask weak ones (Van Moffaert et al., 2013), motivating non-compensatory reward fusion in multimodal emotional support conversations.

## 2.3 Fine-grained Reward Optimization

Fine-grained reward optimization moves beyond outcome-only feedback by supervising intermediate reasoning steps or local response segments. Prior work shows that process-level feedback improves reasoning reliability (Lightman et al., 2024), and dense local signals are useful for multimodal alignment (Yu et al., 2024). In ESC, FIRES optimizes flexible multimodal reasoning via completion-level rewards; RLVER derives emotion rewards from simulated users (Wang et al., 2025a, 2026b). Other methods provide finer-grained supervision at different levels: MindChat-R0 applies GRPO to structured empathetic reasoning; AFlow supervises dialogue prefixes; PEER trains a unified process–outcome reward model (She et al., 2025; Zou et al., 2026; Wang et al., 2025c). In MESC, FACES uses non-compensatory fusion for heterogeneous dimension-level signals, routing their grouprelative advantages to aligned token spans.

## 3 Methodology

To address the challenges outlined above, FACES optimizes structured candidates through the finegrained reward workflow in Figure 2, following supervised fine-tuning and a format-and-outcome RL warm-up. CLIP, MiniLM, and an MLLM judge provide five dimension-specific scores; noncompensatory fusion yields an overall reward that gates group-relative advantages, which are then assigned to aligned reasoning and response spans for targeted token-level updates to the policy.

## 3.1 Multi-source Semantic Rewards

For reward construction, BLEU and ROUGE-L primarily measure lexical overlap with reference responses (Papineni et al., 2002; Lin, 2004) and may assign low scores to semantically valid empathetic replies with different wording. To provide more reliable supervision, we construct multi-source rewards that extend beyond surface matching.

Given input x, the policy produces $y ( x ) \ =$ $\{ y _ { i } \} _ { i = 1 } ^ { N } .$ We score five dimensions, D = {Vis, Emo, Sit, Str, Out}: visual perception, emotion understanding, situation understanding, support strategy, and response outcome. The first four dimensions follow the intermediate reasoning chain of visual perception, emotion recognition, situation attribution, and strategy planning. The outcome dimension separately captures final-response failures that may persist even when the intermediate reasoning appears plausible. This decomposition is guided by three criteria: it covers distinct MESC failure modes, aligns with explicit reasoning spans in the MESC-FT/FIRES flexible-thinking format (Wang et al., 2025a), and supports both automatic scoring and model-based evaluation.

For visual perception and emotion understanding, we use CLIP and MiniLM. Let $z _ { i } ^ { \mathrm { V i s } }$ and $z _ { i } ^ { \mathrm { E m o } }$ denote the visual-perception and emotionunderstanding spans extracted from $y _ { i } ,$ , let I denote the input visual evidence, and let $z _ { * } ^ { \mathrm { E m o } }$ denote the reference emotion span. The visual score is CLIPbased and binarized using a validation-selected threshold $( \tau _ { v } = 0 . 6 0$ by default; sensitivity is analyzed in Appendix F):

$$
r _ { i } ^ { \mathrm { V i s } } = \mathbb { I } \bigl [ \mathrm { C L I P } \bigl ( z _ { i } ^ { \mathrm { V i s } } , I \bigr ) \geq \tau _ { v } \bigr ]\tag{1}
$$

The emotion score is computed with MiniLM as:

$$
r _ { i } ^ { \mathrm { E m o } } = \frac { 1 + \cos \left( e \left( z _ { i } ^ { \mathrm { E m o } } \right) , e \left( z _ { * } ^ { \mathrm { E m o } } \right) \right) } { 2 }\tag{2}
$$

where $e ( \cdot )$ denotes the MiniLM sentence encoder used for emotion representation; further implementation details are provided in Appendix A.3.

For the remaining semantic dimensions, $\mathcal { D } _ { \mathrm { M } } =$ {Sit, Str, Out}, we use Qwen3-VL-8B-Instruct as the MLLM judge (Bai et al., 2025a). To reduce prompt sensitivity, we use three semantically equivalent prompts and score each candidate three times; a representative training prompt is provided in Appendix B.1. The prompts jointly assess semantic consistency, functional consistency, and evidenceaware contextual fit, while penalizing overly verbose, repetitive, or formulaic outputs.

For each $d \in \mathcal { D } _ { \mathrm { M } }$ , the judge outputs a discrete score $s _ { i , k } ^ { d } \in \{ 1 , 2 , 3 , 4 , 5 \}$ for candidate $y _ { i }$ under prompt variant k. We linearly rescale each score from the 1–5 range to [0, 1] via min–max normalization and then average across the three prompt variants to obtain the final score $r _ { i } ^ { d } .$ . The reward system comprises a visual signal based on image– text alignment and four reference-conditioned textbased signals: Emotion uses the annotated emotion span, whereas Situation, Strategy, and Outcome use their corresponding reference annotations. Together, these signals form the basis for non-compensatory reward fusion and fine-grained credit assignment during policy optimization.

![](images/c81e1e5f1d9d58027f1df3ceb22447872bdf2c1e070688beec4d367abe2ac49c.jpg)  
Figure 2: Workflow of FACES. After SFT and format and outcome warm-up, the policy samples structured CoT candidates $y _ { i } \in Y ( x )$ . CLIP and MiniLM score visual and emotion spans, while an MLLM judge (Qwen3-VL-8B-Instruct) scores situation, strategy, and outcome. Candidates failing the format check receive zero reward; valid candidates undergo non-compensatory fusion. Group ranking, positive-only advantage scaling, and token-level credit drive clipped GRPO updates for supportive responses.

## 3.2 Non-compensatory Gated Reward Fusion

To reduce early optimization noise, we use a twostage reward design. In warm-up, we optimize only format compliance and final response quality. Let $r _ { i } ^ { \mathrm { F m t } } \in \{ 0 , 1 \bar { \} }$ indicate whether $y _ { i }$ satisfies the predefined structured format, and let $r _ { i } ^ { \mathrm { O u t } } \in [ 0 , 1 ]$ denote its response-quality score. The warm-up reward is defined as:

$$
R _ { i } ^ { \mathrm { { W a r m } } } = r _ { i } ^ { \mathrm { { F m t } } } + r _ { i } ^ { \mathrm { { O u t } } }\tag{3}
$$

This stage aims to build stable structural control and basic response quality before full RL.

Building on this initialization, we introduce noncompensatory gated fusion. Additive fusion provides a smooth signal but can let strong understanding scores mask poor strategy choices or ineffective responses, while multiplying all dimensions can make optimization overly sensitive to noise. We therefore separate cognitive dimensions, which offer complementary evidence about the user’s state, from response-critical dimensions that directly affect support appropriateness and helpfulness.

Let the five dimension scores of candidate $y _ { i }$ be $r _ { i } ^ { \mathrm { V i s } } , r _ { i } ^ { \mathrm { E m o } } , r _ { i } ^ { \mathrm { S i t } } , r _ { i } ^ { \mathrm { S t r } } , r _ { i } ^ { \mathrm { O u t } }$ , all normalized to [0, 1]. This normalization aligns their numerical ranges but does not make their heterogeneous scales semantically equivalent; accordingly, we treat the fusion rule as an empirically motivated design choice.

We first combine visual perception, emotion understanding, and situation understanding into a cognitive reward that reflects the inferred user state:

$$
R _ { i } ^ { \mathrm { C o g } } = \frac { r _ { i } ^ { \mathrm { V i s } } + r _ { i } ^ { \mathrm { E m o } } + r _ { i } ^ { \mathrm { S i t } } } { 3 }\tag{4}
$$

We use mean fusion for the cognitive dimensions to reduce sensitivity to local uncertainty and avoid overly sparse supervision. In contrast, appropriate support strategy selection is central to effective emotional support (Liu et al., 2021; Tu et al., 2022; Cheng et al., 2022), while response outcome directly reflects the final supportive quality of the reply. We thus use both as multiplicative gates.

The final reward is defined as:

$$
R _ { i } = \left\{ \begin{array} { l l } { 0 , } & { r _ { i } ^ { \mathrm { F m t } } = 0 , } \\ { R _ { i } ^ { \mathrm { C o g } } \cdot r _ { i } ^ { \mathrm { S t r } } \cdot r _ { i } ^ { \mathrm { O u t } } , } & { r _ { i } ^ { \mathrm { F m t } } = 1 } \end{array} \right.\tag{5}
$$

The fusion averages the three cognitive scores and applies non-compensatory vetoes for zero Strategy or Outcome scores. For format-valid candidates, $R _ { i } \leq \operatorname* { m i n } ( R _ { i } ^ { \mathrm { { C o g } } } , r _ { i } ^ { \mathrm { { S t r } } } , r _ { i } ^ { \mathrm { { O u t } } } )$ , so each factor bounds the total reward. Format-invalid candidates receive zero reward and zero token advantages.

## 3.3 Fine-grained Rewards

For each candidate response $y _ { i } .$ , the preceding subsections yield dimension-wise scores $r _ { i } ^ { d }$ and an overall reward $R _ { i }$ . Based on these signals, we convert candidate-level and segment-level supervision into token-level advantages aligned with the reasoning process, enabling joint optimization of intermediate reasoning and final responses.

As shown in Figure 2, for each format-valid candidate $y _ { i } .$ , we parse CoT and response spans $\mathcal { T } _ { i } ^ { d } = [ b _ { i } ^ { d } , e _ { i } ^ { d } )$ , aligning the visual, emotion, situation, and strategy rewards with their respective CoT spans and $r _ { i } ^ { \mathrm { { { \dot { O u t } } } } }$ with the final response.

Since absolute scores across dimensions may exhibit scale drift, we first convert them into centered rank-based advantages within each group. For any dimension $d \in \mathcal { D }$ , we define:

$$
\widetilde { A } _ { i } ^ { d } = \frac { N + 1 } { 2 } - \mathrm { r a n k } ( r _ { i } ^ { d } ) ,\tag{6}
$$

where rank(·) denotes the descending rank within the group, with ties assigned the average rank.

To better align local updates with overall reward quality, we apply a simple positive gate based on the total reward. Let $R _ { \mathrm { m a x } }$ denote the maximum total reward within the current group, and let $\gamma$ denote a validation-selected floor value (default 0.10; sensitivity analyzed in Appendix F). The gate coefficient for candidate i is defined as:

$$
g _ { i } = \left\{ \begin{array} { l l } { \gamma , } & { R _ { \mathrm { m a x } } \leq \epsilon , } \\ { \gamma + ( 1 - \gamma ) \displaystyle \frac { R _ { i } } { R _ { \mathrm { m a x } } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{7}
$$

where ϵ is a small constant. We then apply this gate only to positive segment advantages while leaving negative ones unchanged:

$$
A _ { i } ^ { d } = \left\{ { \begin{array} { l l } { \displaystyle g _ { i } { \widetilde { A } } _ { i } ^ { d } , } & { \displaystyle { \widetilde { A } } _ { i } ^ { d } > 0 , } \\ { \displaystyle { \widetilde { A } } _ { i } ^ { d } , } & { \displaystyle { \widetilde { A } } _ { i } ^ { d } \leq 0 . } \end{array} } \right.\tag{8}
$$

This gate attenuates positive advantages from favorable local segments when a candidate’s total reward is low relative to others in the same group.

For token-level credit assignment, we use an endpoint-anchored scheme. Segment endings often contain the semantic conclusion, while earlier tokens within the same span may encode the evidence, emotion judgment, and situation reasoning needed to reach that conclusion. We therefore avoid continuously decaying the weight with distance from the endpoint. Instead, we scale all earlier tokens by the same factor while retaining the full advantage for the final anchor tokens. Let m be the anchor length and λ the decay factor (defaults: $m = 5 , \lambda = 0 . 8$ ; sensitivity is analyzed in Appendix F). For any token $t \in \mathcal { I } _ { i } ^ { d }$ , we set:

$$
A _ { i , t } = \left\{ { \begin{array} { l l } { A _ { i } ^ { d } , } & { e _ { i } ^ { d } - m \leq t < e _ { i } ^ { d } , } \\ { \lambda A _ { i } ^ { d } , } & { b _ { i } ^ { d } \leq t < e _ { i } ^ { d } - m . } \end{array} } \right.\tag{9}
$$

Thus, the last m tokens receive the full advantage, while earlier tokens receive a decayed one. For format-invalid candidates, whose segment boundaries cannot be reliably parsed, we simply set all token-level advantages to zero and do not apply reward-driven fine-grained supervision.

For policy optimization, FACES builds on the standard clipped GRPO update (Shao et al., 2024), replacing the completion-level advantage with our fine-grained token advantage $A _ { i , t }$ Let $\begin{array} { r l } { s _ { i , t } } & { { } = } \end{array}$ $( x , y _ { i , < t } )$ and $T _ { i } ~ = ~ | y _ { i } |$ . We denote by $\mathcal { S } _ { i , t } ^ { \mathrm { c l i p } } ( \theta )$ the clipped GRPO surrogate computed using the token-level probability ratio between the current policy and the rollout policy for $y _ { i , t }$ at $s _ { i , t } , \epsilon _ { \mathrm { c l i p } } ,$ and $A _ { i , t }$ . The objective is:

$$
\mathcal { L } _ { \mathrm { f g } } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \left( S _ { i , t } ^ { \mathrm { c l i p } } ( \theta ) - \beta \Delta _ { i , t } ^ { \mathrm { K L } } \right) ,\tag{10}
$$

where $\beta$ is the KL regularization coefficient, and $\Delta _ { i , t } ^ { \mathrm { K L } }$ denotes the token-level KL penalty between the current policy $\pi _ { \theta }$ and the reference policy $\pi _ { \mathrm { r e f } }$ In practice, $\pi _ { \theta _ { \mathrm { o l d } } }$ is kept fixed within the rollout batch, while $\pi _ { \mathrm { r e f } }$ is used only for KL regularization.

Overall, FACES combines group-relative advantages, total-reward-based gating, span-aligned token assignment, and clipped GRPO to optimize reasoning and response at the token level.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. We conduct training and in-domain evaluation on MESC-FT, a FIRES-built dataset (Wang et al., 2025a) on top of MESC (Chu et al., 2025) for multimodal emotional support conversations, which provides instance-level flexible-thinking supervision. We use an approximately 8:1:1 split for training, validation, and testing. For cross-dataset evaluation, we construct MELD-ESC300, a 300- sample external test set derived from MELD (Poria et al., 2019), and evaluate zero-shot transfer on the unseen multimodal benchmarks StickerConv (Zhang et al., 2024) and AvaMERG (Zhang et al., 2025). These datasets are used only for evaluation, without fine-tuning. Details of MELD-ESC300 construction and the StickerConv/AvaMERG evaluation protocols are provided in Appendix G.

Evaluation Metrics. We evaluate responses using automatic metrics, MLLM judgments, and human preferences. Automatic metrics include

<table><tr><td>Model</td><td>B-2↑</td><td>R-L ↑</td><td>M.↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E. ↓</td></tr><tr><td>BlenderBot-Joint*</td><td>4.85</td><td>15.25</td><td>一</td><td>85.50</td><td></td><td></td></tr><tr><td>BBMHR*</td><td>1.31</td><td>15.38</td><td></td><td>86.60</td><td></td><td></td></tr><tr><td>SMES*</td><td>5.13</td><td>15.42</td><td></td><td>86.80</td><td></td><td></td></tr><tr><td>Qwen2.5-VL*</td><td>9.77</td><td>20.30</td><td>14.05</td><td>86.35</td><td>84.01</td><td>0.8744</td></tr><tr><td>SMES_Qwen2.5*</td><td>9.24</td><td>19.88</td><td>13.16</td><td>85.62</td><td>92.61</td><td>0.7577</td></tr><tr><td>FIRES (Baseline)*</td><td>10.18</td><td>22.76</td><td>19.54</td><td>86.26</td><td>98.93</td><td>0.7209</td></tr><tr><td>GPT-40</td><td>2.84</td><td>9.97</td><td>16.56</td><td>95.75</td><td>59.36</td><td>0.5300</td></tr><tr><td>Claude-Haiku-4.5</td><td>3.45</td><td>10.01</td><td>17.67</td><td>95.73</td><td>70.31</td><td>0.4753</td></tr><tr><td>Gemini-2.5-Flash-Lite</td><td>4.81</td><td>12.86</td><td>15.90</td><td>96.41</td><td>67.69</td><td>0.4152</td></tr><tr><td>GLM-4.1V-Thinking-FlashX</td><td>3.03</td><td>9.56</td><td>13.47</td><td>96.09</td><td>78.38</td><td>0.5316</td></tr><tr><td>Doubao-1.6-Vision</td><td>2.86</td><td>10.81</td><td>14.86</td><td>96.01</td><td>77.08</td><td>0.4747</td></tr><tr><td>Kimi-K2.5</td><td>6.00</td><td>14.10</td><td>20.10</td><td>96.33</td><td>76.59</td><td>0.4417</td></tr><tr><td>GRPO (Format + Outcome)</td><td>9.93</td><td>21.48</td><td>18.05</td><td>86.33</td><td>95.17</td><td>0.4406</td></tr><tr><td>Ours (Full)</td><td>9.68</td><td>21.09</td><td>20.62</td><td>87.14</td><td>99.63</td><td>0.3643</td></tr></table>

Table 1: Automatic evaluation results on the MESC-FT test set. The superscript <sup>∗</sup> denotes results from prior work or corresponding baseline reports. “–” denotes unavailable results. Best results are shown in bold.

BLEU-2 (B-2) (Papineni et al., 2002), ROUGE-L (R-L) (Lin, 2004), METEOR (M.) (Banerjee and Lavie, 2005), BERTScore-F1 (F<sub>BERT</sub>) (Zhang et al., 2020), Distinct-3 (Dist-3) (Li et al., 2016), and DIFF-EPITOME (DIFF-E.) (Lee et al., 2022); higher is better except for DIFF-E. Following StickerConv’s MLLM evaluation protocol (Zhang et al., 2024), we use Qwen2.5-VL-32B-Instruct-AWQ as the primary post-hoc judge (Bai et al., 2025b), distinct from the training-time reward judge. As a cross-family robustness check, GLM-4.6V-Flash judges the same fixed candidate pairs given identical dialogue context and visual evidence. For MESC-FT and MELD-ESC300, we report pointwise and blind pairwise MLLM evaluations with human pairwise preferences. For StickerConv and AvaMERG, we report automatic metrics and blind MLLM pairwise preferences without human annotation; protocols appear in Appendix G.4.

Compared Methods. We benchmark FACES against prior ESC/MESC systems (Liu et al., 2021; Zhang et al., 2023; Chu et al., 2025) and FIRES (Wang et al., 2025a), as well as six strong, directly prompted MLLMs listed in Table 1. All six receive the same dialogue context and visual evidence. To assess the three main components, we conduct contribution-level ablations and train DPO (Rafailov et al., 2023), SimPO (Meng et al., 2024), and RLOO (Ahmadian et al., 2024) as same-backbone baselines under identical SFT initialization and training prompts. Appendix E reports dimension-level reward ablations and offline diagnostics on gated reward fusion versus linear, geometric-mean, and Tchebycheff aggregation; Appendix F tests endpoint-anchored token credit against linear and exponential decay; and Appendix A details baseline sources and implementation settings for reproducibility.

## 4.2 Main Results

## 4.2.1 Automatic Evaluation

Table 1 shows that FACES leads on METEOR, Distinct-3, and DIFF-EPITOME, while FIRES leads on BLEU-2 and ROUGE-L and Gemini-2.5- Flash-Lite on BERTScore-F1. Although FACES trails FIRES on BLEU-2 and ROUGE-L, these metrics emphasize lexical overlap, and a supportive response need not closely match the reference wording. In contrast, its gains over FIRES on METEOR, BERTScore-F1, Distinct-3, and DIFF-EPITOME suggest stronger semantic correspondence, greater response diversity, and better support-strategy alignment. This pattern is consistent with semantic rewards favoring meaning over wording, while strategy and outcome gates discourage plausible but unhelpful replies.

## 4.2.2 MLLM Evaluation

To complement automatic metrics, we assess supportive quality with the MLLM judge using dialogue context and visual evidence. For brevity and symmetry, we remove reasoning traces and prefix-truncate FACES and baseline responses to 15 words; greedy decoding reduces sampling randomness. Details are in Appendix C.1.

Under length control, the judge scores Empathy, Multimodal Appropriateness, and Context-Response Consistency on a 1–5 Likert scale, measuring overall emotional support quality, multimodal grounding, and contextual faithfulness, respectively. We average three judge prompts per metric to reduce prompt bias; detailed evaluation rules and prompts are in Appendix B.2.

(a) Pointwise MLLM evaluation results on the MESC-FT test set. Scores are on a 1–5 scale, and higher is better.
<table><tr><td>Model</td><td>Empathy ↑</td><td>Multimodal App. ↑</td><td>Context-Response Cons. ↑</td><td>Avg. ↑</td></tr><tr><td>FIRES (Baseline)</td><td>3.06</td><td>3.88</td><td>3.89</td><td>3.61</td></tr><tr><td>Ours (Full)</td><td>3.98</td><td>4.03</td><td>4.11</td><td>4.04</td></tr></table>

(b) Pairwise MLLM preference evaluation results on the MESC-FT test set. Win rates are computed over 450 test instances.
<table><tr><td>Model A</td><td>Model B</td><td>A Win</td><td>B Win</td><td>Tie</td><td>A Win Rate</td><td>B Win Rate</td><td>Tie Rate</td></tr><tr><td>Ours (Full)</td><td>FIRES (Baseline)</td><td>361</td><td>53</td><td>36</td><td>80.22%</td><td>11.78%</td><td>8.00%</td></tr></table>

Table 2: Length-controlled MLLM evaluation results on the MESC-FT test set. Panel (a) reports pointwise scores across three evaluation dimensions, and panel (b) reports blind pairwise preferences over 450 test instances. Pairwise results are significant under a two-sided sign test with ties excluded $( p < 0 . 0 0 1 )$ ).

<table><tr><td>Dataset</td><td>Judge</td><td>FACES Win</td><td>GPT-40 Win</td><td>Tie</td></tr><tr><td>MESC-FT</td><td>Qwen2.5-VL-32B</td><td>72.0%</td><td>16.0%</td><td>12.0%</td></tr><tr><td>MESC-FT</td><td>GLM-4.6V-Flash</td><td>69.8%</td><td>17.8%</td><td>12.4%</td></tr><tr><td>MESC-FT</td><td>Human</td><td>76.0%</td><td>14.0%</td><td>10.0%</td></tr><tr><td>MELD-ESC300</td><td>Qwen2.5-VL-32B</td><td>66.0%</td><td>14.0%</td><td>20.0%</td></tr><tr><td>MELD-ESC300</td><td>GLM-4.6V-Flash</td><td>63.0%</td><td>16.7%</td><td>20.3%</td></tr><tr><td>MELD-ESC300</td><td>Human</td><td>72.0%</td><td>13.3%</td><td>14.7%</td></tr></table>

Table 3: Cap-15 pairwise comparison between FACES and directly prompted GPT-4o.

Table 2(a) shows that Ours (Full) improves all three pointwise dimensions over FIRES (Baseline) under cap-15, raising the average from 3.61 to 4.04. The largest gain is in Empathy, suggesting more supportive responses even with reasoning traces removed and responses truncated; other gains suggest better visual grounding and contextual consistency.

We further conduct blind pairwise preference evaluation under the same length-controlled setting. For each sample, the judge compares two anonymized responses given the same dialogue and visual context, selecting the better overall support response or a tie. We use three prompt variants with bidirectional candidate swapping, and majority voting reduces prompt and position bias.

Table 2(b) reports an 80.22% cap-15 win rate for Ours (Full) over FIRES (Baseline). Together with the pointwise gains, this suggests that response length alone does not explain the preference. Appendix C.4 reports an 84.9% win rate for Ours (Full) under an independent GLM judge, indicating that the preference is not specific to the Qwen judge. Comparisons with directly prompted GPT-4o and five other strong MLLMs appear in Table 3 and Appendix C.5, respectively.

## 4.2.3 Human Preference Evaluation

To validate the cap-15 model-based evaluation, three annotators compared anonymized FACES and FIRES responses on all 450 MESC-FT and

<table><tr><td>Dataset</td><td>Exact Agreement</td><td>Cohen&#x27;s κ</td></tr><tr><td>MESC-FT</td><td>74.7%</td><td>0.316</td></tr><tr><td>MELD-ESC300</td><td>71.7%</td><td>0.441</td></tr></table>

Table 4: Sample-level agreement between final human preferences and Qwen judge majority preferences for the FACES and FIRES comparison under cap-15.

300 MELD-ESC300 instances, assessing empathy, multimodal appropriateness, strategy suitability, and helpfulness using the same context and visual evidence. Candidate order was randomized and ties were allowed; majority vote determined labels, with a fourth adjudicator resolving three-way disagreements. Fleiss’ κ was computed from preadjudication labels. Appendix D details annotator qualifications, protocol, and preference counts.

On MESC-FT, the FACES, FIRES, and tie rates are 75.3%, 14.0%, and 10.7%; on MELD-ESC300, they are 70.3%, 14.7%, and 15.0%. Both FACES advantages are statistically significant under twosided sign tests excluding ties $( p < 0 . 0 0 1 )$ . Fleiss’ κ values are 0.62 and 0.58, respectively.

Table 3 shows that the same annotators also prefer FACES to directly prompted GPT-4o in 76.0% of MESC-FT and 72.0% of MELD-ESC300 cases. Table 4 reports exact human–Qwen agreement rates of 74.7% and 71.7%, with Cohen’s κ values of 0.316 and 0.441, respectively. The two evaluations agree beyond chance but still differ on some individual cases, as detailed in Appendix D.3.

## 4.3 Ablation Study

To analyze the three main designs, we ablate multisource rewards, non-compensatory gating, and finegrained credit assignment from FACES using the same test set and evaluation protocol.

Table 5 shows that Ours (Full) leads on ME-TEOR, BERTScore-F1, Distinct-3, and DIFF-EPITOME, reflecting stronger semantic quality, diversity, and support-strategy alignment. Distinct metric declines across ablations reveal the components’ complementary roles. Omitting multisource rewards boosts BLEU-2 and ROUGE-L but lowers semantic and support-oriented metrics, consistent with overlap-based rewards favoring lexical matching. The overall profile weakens without noncompensatory gating; additive fusion allows crossdimension compensation. Fine-grained credit assignment matters: its removal causes broad degradation, supporting span-aligned token-level supervision. Appendices E and F detail dimension-level ablations and hyperparameter sensitivity analyses.

<table><tr><td>Model</td><td>B-2↑</td><td>R-L ↑</td><td>M.↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E.↓</td></tr><tr><td>w/o multi-source reward</td><td>10.46</td><td>22.31</td><td>19.67</td><td>86.55</td><td>98.62</td><td>0.4189</td></tr><tr><td>w/o non-comp. gating</td><td>9.82</td><td>20.84</td><td>20.02</td><td>86.79</td><td>99.18</td><td>0.4057</td></tr><tr><td>w/o fine-grained credit</td><td>8.56</td><td>18.37</td><td>18.59</td><td>86.02</td><td>98.95</td><td>0.4902</td></tr><tr><td>Ours (Full)</td><td>9.68</td><td>21.09</td><td>20.62</td><td>87.14</td><td>99.63</td><td>0.3643</td></tr></table>

Table 5: Contribution-level ablation results on the MESC-FT test set. All metrics are higher-is-better except DIFF-E.

<table><tr><td>Model</td><td>B-2↑</td><td>R-L↑</td><td>M.↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E. ↓</td></tr><tr><td>FIRES (Baseline)</td><td>6.32</td><td>17.39</td><td>11.43</td><td>85.39</td><td>90.72</td><td>0.8192</td></tr><tr><td>Ours (Full)</td><td>5.59</td><td>16.57</td><td>15.16†</td><td>86.23†</td><td>94.07†</td><td>0.7664†</td></tr></table>

Table 6: Automatic evaluation results on the MELD-ESC300 test set. All metrics are higher-is-better except DIFF-E. <sup>†</sup> indicates a statistically significant improvement over FIRES under paired bootstrap resampling $( p < 0 . 0 5 )$ .

## 4.4 Generalization Analysis

We assess transfer without further training on MELD-ESC300, StickerConv, and AvaMERG. MELD-ESC300 contains 300 samples from the MELD test split using EMOTyDA resources (Poria et al., 2019; Saha et al., 2020); a manual audit found that 274 (91.3%) met the overall validity criterion. We evaluate all 300, treating the set as a controlled external test rather than a dedicated ESC benchmark. Appendix G details its construction.

Table 6 shows that Ours (Full) trails FIRES (Baseline) on BLEU-2 and ROUGE-L but improves METEOR, BERTScore-F1, Distinct-3, and DIFF-EPITOME, with all four gains significant under paired bootstrap resampling $( p < 0 . 0 5 )$ . This pattern suggests stronger semantic matching, diversity, and strategy alignment despite lower lexical overlap. Under cap-15, the MLLM average rises from 3.35 to 3.73 and the Qwen pairwise win rate reaches 62.00%. Human preference reaches 70.3% vs 14.7% for FIRES $( p < 0 . 0 0 1$ , Fleiss’ $\kappa = 0 . 5 8 )$ . Appendices G.3, G.2, and D.2 provide details.

On 300 StickerConv and 300 AvaMERG samples, Ours (Full) wins 63.7% and 79.3% of Qwen comparisons with FIRES under cap-15; the corresponding GLM rates are 64.3% and 65.7%. Appendices G.4 and C.4 report the full results. Together, these tests support transfer across the evaluated domains, while Appendix H shows a failure caused by incorrect visual grounding.

<table><tr><td>Method</td><td>MLLM Avg. ↑</td></tr><tr><td>FIRES (Baseline)</td><td>3.61</td></tr><tr><td>DPO</td><td>3.62</td></tr><tr><td>SimPO</td><td>3.77</td></tr><tr><td>RLOO</td><td>3.83</td></tr><tr><td>Ours (Full)</td><td>4.04</td></tr></table>

Table 7: Comparison of optimization methods on MESC-FT under the same backbone, SFT initialization, prompts, and length-controlled MLLM protocol.

## 4.5 Optimization

To assess whether alternative alignment objectives yield comparable gains, we compare FACES with DPO, SimPO, and RLOO under the same backbone, SFT initialization, prompts, and evaluation protocol. Table 7 shows that all three improve over FIRES but remain below FACES.

## 5 Conclusion

We present FACES, a fine-grained gated semantic alignment framework for MESC. It addresses weak semantic supervision, compensatory reward fusion, and coarse credit assignment through multidimensional rewards, non-compensatory gating, and spanaligned token-level advantages. On MESC-FT, FACES improves several automatic metrics and is preferred to FIRES in matched cap-15 MLLM and human evaluations. This preference advantage persists under an independent GLM judge, while ablations support the contributions of the three main designs. Evaluations on three external datasets provide further evidence of transfer without additional training. These findings highlight fine-grained semantic alignment with non-compensatory gating as a promising direction for MESC generation.

## Limitations

Although FACES shows consistent gains on multimodal emotional support generation, this study still leaves room for further extensions in personalization and evaluation. It focuses on instance-level semantic alignment and therefore does not explicitly model user-specific preferences. Moreover, our experimental setup prioritizes reproducibility and controlled comparison by relying on automatic metrics, MLLM-based assessment, pairwise human preference evaluation, and keyframe-based visual inputs. While these design choices are sufficient to establish the effectiveness of FACES in the present setting, future work could further strengthen the framework through finer-grained human evaluation, more diverse annotators, and richer multimodal signals such as vocal prosody and physiological signals in realistic support settings.

## Ethical Considerations

This study uses the MESC-FT dataset and constructs the MELD-ESC300 external generalization test set from publicly available MELD/EMOTyDA resources. All data are used solely for research purposes, and we strictly comply with the licensing, citation, and redistribution requirements of the original datasets. For released code, datasets, or derived resources, we avoid redistributing raw audio, video, or image materials without proper authorization; instead, we provide guidance for obtaining the original datasets, along with data processing scripts and reproduction instructions.

This work focuses on multimodal emotional support conversation generation, but the proposed model remains a research prototype. It should not be used as a tool for clinical diagnosis, psychotherapy, or crisis intervention without professional supervision. The model may inherit biases from the training data and foundation models, which could lead to inappropriate judgments in emotion recognition, support strategy selection, or response generation. For acute psychological crises, personal safety risks, or other scenarios requiring professional intervention, any real-world deployment should incorporate human oversight, safety screening, and professional referral protocols to determine whether and how the system should respond.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ustun, and Sara Hooker. 2024. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, and 32 others. 2022. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073.

Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measuresfor machine translation and/or summarization, pages 65–72.

Ashutosh Chaubey, Xulang Guan, and Mohammad Soleymani. 2026. Face-llava: Facial expression and attribute understanding through instruction tuning. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2648–2660.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. 2024. Mllm-asa-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning.

Yi Cheng, Wenge Liu, Wenjie Li, Jiashuo Wang, Ruihui Zhao, Bang Liu, Xiaodan Liang, and Yefeng Zheng. 2022. Improving multi-turn emotional support dialogue generation with lookahead strategy planning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Yuqi Chu, Lizi Liao, Jinggui Liang, Boyang Li, and Richang Hong. 2026. Beyond semantic similarity: Appraisal-guided Chain-of-Thought reasoning and retrieval for multimodal emotional support conversations. In Findings of the Association for Computational Linguistics: ACL 2026.

Yuqi Chu, Lizi Liao, Zhiyuan Zhou, Chong-Wah Ngo, and Richang Hong. 2025. Towards multimodal emotional support conversation systems. IEEE Transactions on Multimedia.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. 2024. Prometheus: Inducing finegrained evaluation capability in language models. In The Twelfth International Conference on Learning Representations.

Huy M Le, Dat Tien Nguyen, Ngan TT Vo, Tuan DQ Nguyen, Nguyen Le Binh, Duy Minh Ho Nguyen, Daniel Sonntag, Lizi Liao, and Binh T Nguyen. 2026. Reinforce trustworthiness in multimodal emotional support system. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 31474–31482.

Young-Jun Lee, Chae-Gyun Lim, and Ho-Jin Choi. 2022. Does gpt-3 generate empathetic dialogues? a novel in-context example selection method and automatic evaluation metric for empathetic dialogue generation. In Proceedings ofthe 29th international conference on computational linguistics, pages 669– 683.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and William B Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings ofthe 2016 conference ofthe North American chapter of the association for computational linguistics: human language technologies, pages 110–119.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. 2021. Towards emotional support dialog systems. In Proceedings ofthe 59th annual meeting ofthe associationfor computational linguistics and the 11th international joint conference on natural language processing (volume 1: Long papers), pages 3469–3483.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval:

Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522.

Haifeng Lu, Jiuyi Chen, Feng Liang, Mingkui Tan, Runhao Zeng, and Xiping Hu. 2025. Understanding emotional body expressions via large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 1447–1455.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Se Jin Park, Yeonju Kim, Hyeongseop Rha, Bella Godiva, and Yong Man Ro. 2024. Av-emodialog: Chat with audio-visual users leveraging emotional cues. arXiv preprint arXiv:2412.17292.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. Meld: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings ofthe 57th annual meeting ofthe association for computational linguistics, pages 527–536.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Tulika Saha, Aditya Patra, Sriparna Saha, and Pushpak Bhattacharyya. 2020. Towards emotion-aided multimodal dialogue act classification. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4361–4372.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Dong She, Chenxu Zhang, Xianrong Yao, Yang Gao, and Zhanpeng Jin. 2025. MindChat-R0: A large language model for emotionally supportive dialogue through reinforcement learning. In Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008– 3021.

Geng Tu, Taiyu Niu, Xi Zeng, Ruifeng Xu, and Min Zhang. 2026. A graph-enhanced MLLM for hierarchical multimodal emotion understanding and support in conversations. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Geng Tu, Bingbing Wang, Erik Cambria, Wenjie Li, and Ruifeng Xu. 2025. SupportPlay: A multi-agent role-playing system for personalized and sustained multimodal emotional support conversation. In Companion Proceedings ofthe ACM on Web Conference 2025.

Quan Tu, Yanran Li, Jianwei Cui, Bin Wang, Ji-Rong Wen, and Rui Yan. 2022. Misc: A mixed strategyaware model integrating comet for emotional support conversation. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 308–319.

Kristof Van Moffaert, Madalina M Drugan, and Ann Nowé. 2013. Scalarized multi-objective reinforcement learning: Novel design techniques. In 2013 IEEE symposium on adaptive dynamic programming and reinforcement learning (ADPRL), pages 191– 199. IEEE.

Chengbing Wang, Wuqiang Zheng, Yang Zhang, Fengbin Zhu, Junyi Cheng, Yi Xie, Wenjie Wang, and Fuli Feng. 2026a. PERM: Psychology-grounded empathetic reward modeling for large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 7350–7373.

Fanfan Wang, Xiangqing Shen, Jianfei Yu, and Rui Xia. 2025a. Flexible thinking for multimodal emotional support conversation via reinforcement learning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1341–1356.

Jiashuo Wang, Chunpu Xu, Chak Tou Leong, Wenjie Li, and Jing Li. 2024a. Muffin: Mitigating unhelpfulness in emotional support conversations with multifaceted ai feedback. In Findings of the Association for Computational Linguistics: ACL 2024, pages 567–585.

Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Cixing Li, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Jia Li, Zhaopeng Tu, and Xiaolong Li. 2026b. RLVER: Reinforcement learning with verifiable emotion rewards for empathetic agents. In The Fourteenth International Conference on Learning Representations.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33:5776–5788.

Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Yong Chen, and Luo Ji. 2025b. Convert language model into a value-based strategic planner. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 6: Industry Track).

Yidong Wang, Zhuohao Yu, Wenjin Yao, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. 2024b. PandaLM: An automatic evaluation benchmark for LLM instruction tuning optimization. In The Twelfth International Conference on Learning Representations.

Yunxiao Wang, Meng Liu, Sicheng Zhao, Lizi Liao, and Liqiang Nie. 2025c. PEER: Unified process– outcome reinforcement learning for structured empathetic reasoning. arXiv preprint arXiv:2508.09521.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, and Tat-Seng Chua. 2024. RLHF-V: Towards trustworthy MLLMs via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807–13816.

Han Zhang, Zixiang Meng, Meng Luo, Hong Han, Lizi Liao, Erik Cambria, and Hao Fei. 2025. Towards multimodal empathetic response generation: A rich text-speech-vision avatar-based benchmark. In Proceedings of the ACM on Web Conference 2025, pages 2872–2881.

Qiang Zhang, Jason Naradowsky, and Yusuke Miyao. 2023. Ask an expert: Leveraging language models to

improve strategic reasoning in goal-oriented dialogue models. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 6665–6694.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

Yiqun Zhang, Fanheng Kong, Peidong Wang, Shuang Sun, Lingshuai Wang, Shi Feng, Daling Wang, Yifei Zhang, and Kaisong Song. 2024. STICKERCONV: Generating multimodal empathetic responses from scratch. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7707–7733. Association for Computational Linguistics.

Haiquan Zhao, Lingyu Li, Shisong Chen, Shuqi Kong, Jiaan Wang, Kexin Huang, Tianle Gu, Yixu Wang, Jian Wang, Dandan Liang, Zhixu Li, Yan Teng, Yanghua Xiao, and Yingchun Wang. 2024. ESC-eval: Evaluating emotion support conversations in large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15785–15810.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Chenghui Zou, Ning Wang, Tiesunlong Shen, Luwei Xiao, Chuan Ma, Xiangpeng Li, Rui Mao, and Erik Cambria. 2026. Affective flow language model for emotional support conversation. arXiv preprint arXiv:2602.08826.

## A Experimental Setup

This section provides the core implementation details required to reproduce our experiments. We report the main model, training, decoding, evaluation, reward, and hardware settings, while omitting unchanged optimizer defaults such as AdamW betas and other non-critical implementation details.

## A.1 Training Configuration

Our policy is initialized through supervised finetuning, followed by a format-and-outcome GRPO warm-up and full FACES training with gated rewards and a clipped GRPO objective over finegrained token-level advantages. Table 8 reports the core FACES configuration, including its GRPOspecific hyperparameters.

DPO, SimPO, and RLOO use the same backbone, SFT initialization, training prompts, and evaluation protocol, while retaining their respective optimization objectives and update rules.

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Policy backbone</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Reward critic</td><td>Qwen3-VL-8B-Instruct</td></tr><tr><td>SFT learning rate</td><td>5 × 10−5</td></tr><tr><td>GRPO learning rate</td><td>1 × 10−5</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>SFT / warm up / FACES epochs</td><td>3/3/3</td></tr><tr><td>Global batch size</td><td>32</td></tr><tr><td>Learning rate warmup ratio</td><td>0.05</td></tr><tr><td>GRPO group size G</td><td>4</td></tr><tr><td>GRPO clip coefficient €clip</td><td>0.2</td></tr><tr><td>Policy updates per rollout</td><td>4</td></tr><tr><td>KL coefficient β</td><td>0.001</td></tr><tr><td>GRPO rollout decoding</td><td>Sampling with tempera- ture 1.3 and top-p 0.95</td></tr><tr><td>Maximum rollout length</td><td>192 tokens</td></tr><tr><td>Maximum sequence length</td><td>1024 tokens</td></tr><tr><td>Maximum image side length</td><td>384 pixels</td></tr><tr><td>Maximum video frames</td><td>16</td></tr><tr><td>LoRA rank / alpha / dropout</td><td>8 / 32 / 0.05</td></tr><tr><td>LoRA target modules</td><td>All linear layers</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Training hardware</td><td>2 NVIDIA A6000 GPUs</td></tr></table>

Table 8: Core training configuration.

## A.2 Inference and Evaluation Settings

The FIRES automatic scores in Table 1 are taken from Wang et al. (2025a). For local response-level evaluation, FIRES responses are generated using the inference settings in Table 9, which also lists the primary Qwen and independent GLM judge configurations. Test-time generation is greedy, whereas GRPO rollouts sample at temperature 1.3 and top-p 0.95 to produce diverse training candidates.

The primary Qwen judge also uses deterministic decoding with temperature 0 and sampling disabled. For cross-judge evaluation, GLM-4.6V-Flash evaluates exactly the same fixed candidate responses as the primary Qwen judge. Judge-specific output budgets affect only the evaluation outputs and do not alter the candidate responses. We additionally disable GLM thinking mode to reduce variation arising from judge-side reasoning.

Both judges receive the same dialogue context, visual evidence, length-controlled responses, and candidate ordering, with up to eight images. Bootstrap analyses use 5,000 resamples and seed 42. Judge prompts and controlled protocols appear in Appendices B.2 and C.

For directly prompted MLLMs, metrics use complete responses and references. Distinct-3 uses corpus-level aggregation. BERTScore-F1 uses final-layer RoBERTa-large without IDF or baseline rescaling. DIFF-EPITOME uses mean squared error between candidate and reference scores.

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Primary evaluation judge</td><td>Qwen2.5-VL-32B- Instruct-AWQ</td></tr><tr><td>Cross-family evaluation judge</td><td>GLM-4.6V-Flash</td></tr><tr><td>Local generation and Qwen judge</td><td></td></tr><tr><td>Decoding strategy</td><td>Greedy decoding</td></tr><tr><td>Maximum new tokens</td><td>256</td></tr><tr><td>Temperature</td><td>0</td></tr><tr><td>Sampling</td><td>False</td></tr><tr><td>Top-p</td><td>Not used</td></tr><tr><td>Maximum input images</td><td>8</td></tr><tr><td>GLM judge</td><td></td></tr><tr><td>Maximum output tokens</td><td>768</td></tr><tr><td>Temperature</td><td>0</td></tr><tr><td>Thinking mode</td><td>Disabled</td></tr><tr><td>Maximum input images</td><td>8</td></tr><tr><td>Statistical evaluation</td><td></td></tr><tr><td>Bootstrap samples</td><td>5000</td></tr><tr><td>Bootstrap seed</td><td>42</td></tr></table>

Table 9: Inference-time generation and automatic evaluation settings, with judge-specific configurations reported separately.

## A.3 Reward Configuration

FACES uses a non-compensatory reward fusion strategy. All continuous reward components are normalized to [0, 1], while the format score $r _ { i } ^ { \mathrm { F m t } }$ is binary. We first aggregate visual grounding, emotion recognition, and situation understanding into a cognitive reward:

$$
R _ { i } ^ { \mathrm { C o g } } = \frac { r _ { i } ^ { \mathrm { V i s } } + r _ { i } ^ { \mathrm { E m o } } + r _ { i } ^ { \mathrm { S i t } } } { 3 } .\tag{11}
$$

The final reward is then computed with format validity as a hard gate, and strategy and outcome as multiplicative gates:

$$
R _ { i } = \left\{ \begin{array} { l l } { 0 , } & { r _ { i } ^ { \mathrm { F m t } } = 0 , } \\ { R _ { i } ^ { \mathrm { C o g } } \cdot r _ { i } ^ { \mathrm { S t r } } \cdot r _ { i } ^ { \mathrm { O u t } } , } & { r _ { i } ^ { \mathrm { F m t } } = 1 . } \end{array} \right.\tag{12}
$$

Table 10 summarizes validation-selected reward settings. The critic decodes deterministically with up to 512 new tokens. CLIP averages the four highest framewise cosine similarities, rescales the mean by $( 1 + s ) / 2$ , then applies the 0.60 threshold. MiniLM uses masked mean pooling and L2- normalized embeddings. Parsing retains the first think/response blocks and last nonempty repeated stage. Missing visual/emotion inputs and invalid critic fields score zero; missing Situation/Strategy spans are passed as empty text.

## A.4 Reward Reliability and Cost

For 1,829 format-valid offline-diagnostic candidates, MLLM-scored dimensions use absoluteagreement ICC(2,1) per prompt and ICC(2,3) for three-prompt means. The same three annotators in Appendix D.1 independently rate 150 candidates stratified by gated-reward quintile from 1 to 5 on five reward dimensions and Overall Quality. Table 11 reports Spearman’s $\rho$ for each dimensional reward against its mean human score and for gated reward against mean human Overall Quality.

<table><tr><td>Component</td><td>Range</td><td>Role in reward fusion</td></tr><tr><td>Format  $r _ { i } ^ { \mathrm { F m t } }$ </td><td>{0, 1}</td><td>Hard gate in the full FACES reward stage; invalid outputs receive 0 in</td></tr><tr><td>Visual r ν Vis</td><td>[0, 1]</td><td>final reward fusion. 1/3 in cognitive reward</td></tr><tr><td>Emotion Emo</td><td>[0, 1]</td><td>1/3 in cognitive reward</td></tr><tr><td>Situation Sit</td><td>[0, 1]</td><td>1/3 in cognitive reward</td></tr><tr><td>Strategy i</td><td>[0, 1]</td><td>Multiplicative gate</td></tr><tr><td>Outcome r Out</td><td>[0, 1]</td><td>Multiplicative gate</td></tr><tr><td>Cognitive  $\overset { \underset { i } { \operatorname { \sum } } \operatorname { c o g } } { \operatorname { \sum } }$ </td><td>[0, 1]</td><td>Mean of visual, emotion, and situa-</td></tr><tr><td>Final reward  $R _ { i }$ </td><td>[0, 1]</td><td>tion Product of cognitive, strategy, and</td></tr></table>

Table 10: Reward components and non-compensatory fusion rule.

Three-prompt averaging raises ICC from 0.74– 0.79 to 0.90–0.92; gated reward correlates with human Overall Quality at $\rho = 0 . 7 2$ , supporting its use in training.

To quantify reward computation cost, we benchmark 100 GRPO steps with the same Qwen2.5-VL-7B policy, batch and group sizes, generation limit, and two NVIDIA A6000 GPUs. FIRES uses Format, ROUGE $\scriptstyle - \mathrm { { L , } }$ and DIFF-EPITOME/RoBERTa rewards; FACES uses CLIP, MiniLM, and the Qwen3-VL reward critic. Table 12 shows that FACES uses 2.45 times as many GPU hours as FIRES. Neither uses reward models at deployment; Table 13 summarizes inference measurements for FACES and FIRES on identical hardware.

<table><tr><td>Dimension</td><td>ICC(2,1) ICC(2,3)</td><td></td><td>Human  $\rho$ </td></tr><tr><td>Visual</td><td>一</td><td>一</td><td>0.61</td></tr><tr><td>Emotion</td><td></td><td></td><td>0.68</td></tr><tr><td>Situation</td><td>0.74</td><td>0.90</td><td>0.66</td></tr><tr><td>Strategy</td><td>0.79</td><td>0.92</td><td>0.71</td></tr><tr><td>Outcome</td><td>0.77</td><td>0.91</td><td>0.74</td></tr><tr><td>Gated reward vs. Overall Quality</td><td></td><td></td><td>0.72</td></tr><tr><td>Pipeline</td><td>Wall-time (h)</td><td>GPU-h</td><td>Relative cost</td></tr><tr><td>FIRES</td><td>17.2</td><td>34.4</td><td>1.00×</td></tr><tr><td>FACES</td><td>42.2</td><td>84.4</td><td>2.45×</td></tr></table>

Table 11: Reliability of the training-time reward system. Visual and Emotion are deterministic CLIP and MiniLM scores, so prompt-level ICC is not applicable. The last column reports reward–human correlations, all significant with $p < 0 . 0 0 1$ . Higher values indicate stronger reliability or alignment.

Table 12: Training cost over 100 stable GRPO steps. Lower values are better for all three measures.
<table><tr><td>Metric</td><td>FACES</td><td>FIRES</td></tr><tr><td>Latency (s) ↓</td><td> $1 5 . 8 2 8 \pm 2 . 8 2 0$ </td><td> $1 3 . 7 5 2 \pm 2 . 1 4 5$ </td></tr><tr><td>Output length</td><td>184.7 tokens</td><td>128.9 tokens</td></tr><tr><td>Throughput</td><td>11.67 tokens/s</td><td>9.37 tokens/s</td></tr><tr><td>Peak memory</td><td>20.130 GiB</td><td>20.130 GiB</td></tr></table>

Table 13: Deployment-time inference cost measured on 100 samples using one NVIDIA RTX A6000. Latency is reported per sample. Reward models are used only during training.

## B Prompt Templates

This section reports the prompt templates used in FACES. The prompt design follows three principles. First, training-time reward construction and post-hoc evaluation use separate prompt families because they serve different purposes: the former provides optimization signals for the policy, whereas the latter measures the quality of fixed model outputs under independently specified evaluation criteria. Second, for every scoring task, we use three semantically equivalent prompt variants that preserve the same input fields, scoring criteria, score range, and output schema while varying their surface realization. This design reduces sensitivity to any single prompt formulation without changing the underlying evaluation objective. Third, all prompts require machine-readable JSON outputs, enabling deterministic parsing and consistent aggregation across instances.

For training-time reward scoring, Table 14 presents one representative template. For post-hoc MLLM evaluation, we similarly use three variants for each pointwise metric and pairwise preference setting. The shared evaluation rules and representative templates are reported in Table 15, Table 16, Table 17, and Table 18. The complete prompt variants used in the experiments, together with the corresponding parsing and aggregation code, are available in the code repository.

## B.1 Training-time Reward Prompts

The training-time MLLM reward prompts are used to score situation understanding, support strategy, and response outcome during reward construction. Candidate reasoning and response spans, together with their corresponding reference annotations, are dynamically inserted through placeholders such as {context} and {candidate\_response}. The judge assesses semantic consistency, functional alignment, and contextual appropriateness, while penalizing unnecessary verbosity, and returns structured scores from 1 to 5 for the three dimensions. Visual perception and emotion understanding are evaluated separately using CLIP and MiniLM rather than the MLLM prompt. Scores obtained from the three prompt variants are normalized and then averaged before being used in reward fusion.

## B.2 MLLM Evaluation Prompts

For automatic MLLM evaluation, we use complementary pointwise scoring and blind pairwise preference comparison. Pointwise evaluation measures each response independently along Empathy, Multimodal Appropriateness, and Context–Response Consistency, whereas pairwise evaluation asks the judge to select the better overall emotional-support response or return a tie. The former provides interpretable dimension-level measurements, while the latter captures holistic quality when different strengths and weaknesses must be considered jointly. In pairwise comparisons, model identities are hidden, and both candidates are evaluated using the same dialogue context and visual evidence.

Unlike training-time reward scoring, post-hoc MLLM judges receive no reference situation, strategy, or response annotations. Consequently, a candidate can be assessed for contextual appropriateness without requiring its wording or support strategy to match a single dataset reference.

For each pointwise metric, we apply three semantically equivalent prompt variants and average the resulting scores. Pairwise evaluation uses the same number of prompt variants together with bidirectional candidate swapping and majority voting, reducing sensitivity to prompt wording and presentation order. For the length-controlled evaluations reported in the main results and Appendix C, all think-related fields and think-specific criteria are removed from the judge input and prompt. The judge therefore evaluates only the visible, lengthcontrolled final responses rather than explicit reasoning traces. All prompts require structured JSON outputs; invalid or unparsable pairwise outputs are conservatively treated as ties. Table 15 summarizes the shared evaluation rules, Table 16 defines the pointwise dimensions, and Tables 17 and 18 present the representative prompts.

```jsonl
Representative Training-time Reward Prompt
You are a strict evaluator for multimodal empathetic dialogue.
Evaluate the candidate using the dialogue context and the corresponding reference annotations. Your judgment must
consider all of the following:
1. semantic consistency with the corresponding reference part;
2. functional alignment with the intended role of the reference part;
3. appropriateness to the current dialogue context and available evidence.
Score the candidate on the following three dimensions using integer scores from 1 to 5:
- Situation: whether the candidate situation inference is semantically consistent with the reference situation, functionally
appropriate for understanding the user’s condition, and grounded in the available evidence.
- Strategy: whether the candidate support strategy is semantically aligned with the reference strategy, functionally
appropriate for the user’s needs, and psychologically safe in the current context.
- Outcome: whether the final response is semantically and functionally aligned with the reference response, while remaining
empathetic, helpful, and contextually appropriate.
Scoring rule:
- 5 = great
- 4 = good
- 3 = acceptable
- 2 = weak
- 1 = poor
- Penalize overly long, verbose, repetitive, or formulaic candidates with redundant or padded wording. If a candidate is
substantially longer than the reference and the extra content does not improve semantic or functional alignment, lower the
corresponding score(s).
[CONTEXT]
{context}
[CANDIDATE_SITUATION]
{candidate_situation}
[REFERENCE_SITUATION]
{reference_situation}
[CANDIDATE_STRATEGY]
{candidate_strategy}
[REFERENCE_STRATEGY]
{reference_strategy}
[CANDIDATE_RESPONSE]
{candidate_response}
[REFERENCE_RESPONSE]
{reference_response}
Return JSON only in the following format:
{"situation_score": 1, "strategy_score": 1, "outcome_score": 1}
```  
Table 14: Representative prompt template for training-time MLLM reward scoring. We use three semantically equivalent prompt variants in the actual reward-scoring pipeline

<table><tr><td>Rule</td><td>Description</td></tr><tr><td>JSON-only output</td><td>The judge must return a valid JSON object and no extra explanation.</td></tr><tr><td>Score range</td><td>Pointwise scores are integers from 1 to 5, where 5 indicates the best quality.</td></tr><tr><td>Pairwise labels</td><td>Pairwise comparison returns one of A, B, or TIE.</td></tr><tr><td>Model anonymity</td><td>Candidate names are hidden from the judge and replaced with Candidate A and Candi- date B.</td></tr><tr><td>Think section</td><td>Present in the base templates; removed in length-controlled variants.</td></tr><tr><td>Verbosity penalty</td><td>The judge is instructed to penalize overly long, repetitive, verbose, or formulaic re- sponses when the extra content does not improve helpfulness, empathy, or contextual</td></tr><tr><td>Pointwise aggregation</td><td>alignment. Each response is scored from 1 to 5 under three prompt variants, and the three scores are averaged for each evaluation dimension.</td></tr><tr><td>Pairwise aggregation</td><td>For each of the three prompt variants, the judge is queried twice, once with FACES as Candidate A and FIRES as Candidate B, and once with the candidate order reversed. After mapping each A, B, or Tie judgment back to the corresponding model identity, the two order-specific judgments are combined as follows: agreement is retained; if one judgment is Tie, the other is retained; and conflicting non-tie judgments produce Tie. The resulting three variant-level decisions are aggregated by comparing the FACES and FIRES vote counts, with equal counts producing Tie. An invalid or unparsable output is</td></tr></table>

Table 15: Common rules for MLLM-based automatic evaluation.

<table><tr><td>Metric</td><td>Evaluation focus</td><td>JSON key</td></tr><tr><td>Empathy</td><td>Whether the final response expresses appropriate empathy, emotional validation, and supportive wording.</td><td>empathy_text</td></tr><tr><td>Multimodal appropriateness</td><td>Whether the response is consistent with the dialogue context and multimodal available visual evidence.</td><td>appropriateness</td></tr><tr><td>Context-Response Cons.</td><td>Whether the final response is consistent with the dialogue context, context_response visual evidence, and support direction.</td><td>consistency</td></tr></table>

Table 16: Pointwise MLLM evaluation metrics.

Representative Pointwise Evaluation Prompt   
You are an impartial evaluator for multimodal emotional support dialogue.   
Evaluate the candidate output according to the specified metric. Use the dialogue context, any attached visual evidence, the   
model’s think section if available, and the final response. If the think section is missing, it will be shown as Not provided.   
Your judgment must consider:   
1. whether the candidate is appropriate to the user’s emotional state and dialogue context;   
2. whether it is grounded in the available multimodal evidence;   
3. whether it avoids unsupported, unsafe, overly verbose, repetitive, or formulaic content;   
4. whether it satisfies the specific evaluation metric.   
[EVALUATION\_METRIC]   
{metric\_name}   
[METRIC\_DEFINITION]   
{metric\_definition}   
[CONTEXT]   
{context}   
[MODEL\_THINK]   
{think}   
[CANDIDATE\_RESPONSE]   
{response}   
Score the candidate from 1 to 5. Return JSON only in the following format:   
{json\_score\_format}  
Table 17: Representative pointwise MLLM evaluation prompt. We use three semantically equivalent prompt variants for each pointwise metric.

Representative Pairwise Preference Prompt   
You are an impartial judge for multimodal emotional support dialogue.   
Compare Candidate A and Candidate B based on the dialogue context, any attached visual evidence, the provided think   
sections if available, and the final responses. The model identities are hidden. If a think section is missing, it will be shown   
as Not provided.   
Choose the better candidate according to the following criteria:   
1. empathy and emotional validation;   
2. contextual relevance to the user’s situation;   
3. consistency with the visual evidence when available;   
4. appropriateness and safety of the support strategy;   
5. coherence between the think section and the final response;   
6. avoidance of unnecessarily long, repetitive, verbose, or formulaic wording.   
If both candidates are comparable in overall quality, choose TIE.   
[CONTEXT]   
{context}   
[CANDIDATE A THINK]   
{think\_a}   
[CANDIDATE A RESPONSE]   
{response\_a}   
[CANDIDATE B THINK]   
{think\_b}   
[CANDIDATE B RESPONSE]   
{response\_b}   
Return JSON only in the following format:   
{"winner": "A"}   
The value of winner must be one of A, B, or TIE.  
Table 18: Representative pairwise MLLM preference prompt. We use three semantically equivalent prompt variants, bidirectional candidate swapping, and majority voting to reduce prompt-order bias.

## C Controlled Evaluation

To examine whether the observed pairwise preference trend persists after restricting the information visible to the judge, we reuse the same underlying Qwen2.5-VL-32B-Instruct-AWQ judge model under controlled views. The overall evaluation setup remains unchanged, including three semantically equivalent prompt variants, bidirectional candidate swapping, and majority vote. Each controlled view exposes only the permitted candidate fields; when reasoning traces are hidden, the corresponding fields and evaluation criteria are removed from the prompt.

## C.1 Controlled Views

We consider a response-only view without a word limit and three length-controlled views. In the response-only no-cap view, the judge sees the dialogue context, attached images, and the final response only; the prompt excludes the think section and no think-specific evaluation criterion is used. This view shows the same preference trend, but it may still be affected by response-length differences. Therefore, we focus the reported controlled results on the length-controlled cap-30/cap-25/cap-15 views. In these views, each candidate response is prefix-truncated to at most N words, where $N \in \{ 3 0 , 2 5 , 1 5 \}$

## C.2 Controlled Pairwise Results

Table 19 reports blind pairwise preference outcomes under the three length-controlled views (cap-30, cap-25, and cap-15) on the MESC-FT test set and MELD-ESC300.

<table><tr><td>Dataset</td><td>View</td><td>A Win</td><td>B Win</td><td>Tie</td></tr><tr><td>MESC-FT test set</td><td>cap-30</td><td>426</td><td>11</td><td>13</td></tr><tr><td>MESC-FT test set</td><td>cap-25</td><td>411</td><td>18</td><td>21</td></tr><tr><td>MESC-FT test set</td><td>cap-15</td><td>361</td><td>53</td><td>36</td></tr><tr><td>MELD-ESC300</td><td>cap-30</td><td>287</td><td>7</td><td>6</td></tr><tr><td>MELD-ESC300</td><td>cap-25</td><td>279</td><td>11</td><td>10</td></tr><tr><td>MELD-ESC300</td><td>cap-15</td><td>186</td><td>29</td><td>85</td></tr></table>

Table 19: Controlled pairwise evaluation results. Each cell reports majority-vote decisions. Model A denotes Ours (Full), and Model B denotes FIRES (Baseline).

Across both datasets, Model A remains consistently preferred under all length-controlled views. As the word cap becomes stricter, the win rate of Model A gradually decreases, but Model A still clearly outperforms Model B even under the most restrictive cap-15 setting. Since cap-15 removes explicit reasoning content and imposes the strongest constraint on visible response length, it provides the most conservative evidence in this appendix.

## C.3 Statistical Significance

To assess whether the observed preference advantages in the controlled pairwise evaluation are statistically reliable, we further perform significance testing on the main comparisons. For each dataset and each reported length-controlled view, we apply a two-sided exact binomial sign test over decisive (i.e., non-tie) comparisons only, and estimate 95% confidence intervals for the FACES preference share via bootstrap resampling. This analysis does not introduce any new evaluation criterion; rather, it provides a statistical summary, including confidence intervals, of the majority-vote pairwise results reported in the main text and appendix.

The results show that FACES remains significantly preferred over the baseline under all reported length-controlled views on both the MESC-FT test set and MELD-ESC300. Notably, even under the most conservative cap-15 condition, where explicit reasoning traces are hidden and the judge can access only length-truncated final responses, FACES still attains a decisive preference share of 87.2% on the MESC-FT test set (95% CI: 83.9%–90.4%, $p \approx 2 . 0 \times 1 0 ^ { - 5 7 } )$ and 86.5% on MELD-ESC300 $( 9 5 \% \thinspace \mathrm { C I } ; 8 1 . 9 \% - 9 1 . 2 \% , p \approx 3 . 1 \times 1 0 ^ { - 2 9 } )$ . These results suggest that the observed preference gains are not merely driven by visible CoT content or longer responses, but remain robust under substantially restricted judging conditions.

## C.4 Cross-Judge Robustness

To examine whether the preference results are specific to a Qwen-family evaluator, we add GLM-4.6V-Flash as an independent judge. We retain the same three prompt variants, bidirectional candidate swapping, length control, and majority voting. Each result reports FACES win/FIRES win/tie.

Table 20 reports the results obtained with the independent GLM judge. On MESC-FT, GLM assigns FACES win rates of 84.9%, 97.1%, and 99.6% under cap-15, cap-25, and unrestricted response-only evaluation, respectively. Under cap-15, the corresponding win rates are 64.3% on StickerConv and 65.7% on AvaMERG. After ties are excluded, the cap-15 preference shares on MESC-FT, StickerConv, and AvaMERG are 93.4%, 76.8%, and 80.4%, respectively, all significantly above chance under the reported sign tests $( p < 0 . 0 0 1 )$ These results show a consistent length-controlled preference across the two judge families.

<table><tr><td>Dataset</td><td>View</td><td>n</td><td>Qwen2.5-VL-32B</td><td>GLM-4.6V-Flash</td><td>GLM FACES Share Excl. Ties</td></tr><tr><td>MESC-FT</td><td>cap-15</td><td>450</td><td>80.22/11.78/8.00</td><td>84.9/6.0/9.1</td><td> $9 3 . 4 \% ( p = 2 . 0 { \times } 1 0 ^ { - 8 1 } )$ </td></tr><tr><td>MESC-FT</td><td>cap-25</td><td>450</td><td>91.3/4.0/4.7</td><td>97.1/1.3/1.6</td><td>一</td></tr><tr><td>MESC-FT</td><td>response-only</td><td>450</td><td>100.0/0.0/0.0</td><td>99.6/0.2/0.2</td><td></td></tr><tr><td>StickerConv</td><td>cap-15</td><td>300</td><td>63.7/18.3/18.0</td><td>64.3/19.4/16.3</td><td>76.8%  $( p = 4 . 0 { \times } 1 0 ^ { - 1 8 } )$ </td></tr><tr><td>AvaMERG</td><td>cap-15</td><td>300</td><td>79.3/11.3/9.4</td><td>65.7/16.0/18.3</td><td> $8 0 . 4 \% \overset { \cdot } { ( p = 1 . 3 \times 1 0 ^ { - 2 2 } ) }$ </td></tr></table>

Table 20: Cross-judge pairwise preferences (%). Both judges evaluate the same candidate pairs. Cap-N views truncate final responses to at most N words; response-only uses no word cap.

<table><tr><td>Dataset</td><td>Opponent</td><td>n</td><td>Qwen2.5-VL-32B</td><td>GLM-4.6V-Flash</td></tr><tr><td>MESC-FT</td><td>GPT-40</td><td>450</td><td>72.0/16.0/12.0</td><td>69.8/17.8/12.4</td></tr><tr><td>MESC-FT</td><td>Claude-Haiku-4.5</td><td>450</td><td>74.0/14.4/11.6</td><td>71.3/16.3/12.4</td></tr><tr><td>MESC-FT</td><td>Gemini-2.5-Flash-Lite</td><td>450</td><td>72.7/15.3/12.0</td><td>70.2/17.1/12.7</td></tr><tr><td>MESC-FT</td><td>GLM-4.1V-Thinking-FlashX</td><td>450</td><td>76.0/12.7/11.3</td><td>72.9/14.7/12.4</td></tr><tr><td>MESC-FT</td><td>Doubao-1.6-Vision</td><td>450</td><td>76.7/12.0/11.3</td><td>74.0/13.8/12.2</td></tr><tr><td>MESC-FT</td><td>Kimi-K2.5</td><td>450</td><td>71.6/16.2/12.2</td><td>69.3/18.0/12.7</td></tr><tr><td>MELD-ESC300</td><td>GPT-40</td><td>300</td><td>66.0/14.0/20.0</td><td>63.0/16.7/20.3</td></tr><tr><td>MELD-ESC300</td><td>Claude-Haiku-4.5</td><td>300</td><td>68.0/12.3/19.7</td><td>65.0/14.7/20.3</td></tr><tr><td>MELD-ESC300</td><td>Gemini-2.5-Flash-Lite</td><td>300</td><td>67.0/13.3/19.7</td><td>64.0/15.7/20.3</td></tr><tr><td>MELD-ESC300</td><td>GLM-4.1V-Thinking-FlashX</td><td>300</td><td>69.3/11.0/19.7</td><td>65.7/14.0/20.3</td></tr><tr><td>MELD-ESC300</td><td>Doubao-1.6-Vision</td><td>300</td><td>70.0/10.7/19.3</td><td>67.0/13.0/20.0</td></tr><tr><td>MELD-ESC300</td><td>Kimi-K2.5</td><td>300</td><td>65.0/15.0/20.0</td><td>62.0/17.7/20.3</td></tr></table>

Table 21: Cap-15 blind pairwise comparisons between FACES and directly prompted strong MLLMs. Each cell reports FACES win/opponent win/tie percentages.

## C.5 Strong Direct-Prompted MLLMs

We compare FACES with six directly prompted strong MLLMs on MESC-FT (450 test instances) and MELD-ESC300 (300 external test instances). MESC-FT assesses in-domain performance, while MELD-ESC300 assesses cross-dataset transfer without additional fine-tuning on the external dataset. None of the six comparison models is taskfine-tuned, and each receives the same dialogue context, visual evidence, and response instruction.

For both datasets, we apply the cap-15 responseonly blind pairwise protocol, removing reasoning traces and prefix-truncating each candidate response to at most 15 words. We use three semantically equivalent judge prompts, bidirectional candidate swapping to reduce presentation order bias, and majority voting. The same fixed candidate responses are evaluated by Qwen2.5-VL-32B-Instruct-AWQ and GLM-4.6V-Flash.

Table 21 shows that FACES is preferred over all six opponents by both judges on both datasets. On MESC-FT, its win rates range from 71.6% to 76.7% under Qwen and from 69.3% to 74.0% under GLM. On MELD-ESC300, the corresponding ranges are 65.0%–70.0% and 62.0%–67.0%. The lower win rates on MELD-ESC300 indicate a smaller preference advantage under cross-dataset evaluation, but the direction of preference remains consistent for every opponent. GLM also assigns lower FACES win rates than Qwen throughout, while preserving the same overall conclusion. This consistency provides evidence that the observed advantage is not confined to evaluation by a Qwen-family judge.

<table><tr><td>Dataset</td><td>FACES Win</td><td>GPT-40 Win</td><td>Tie</td></tr><tr><td>MESC-FT</td><td>342 (76.0%)</td><td>63 (14.0%)</td><td>45 (10.0%)</td></tr><tr><td>MELD-ESC300</td><td>216 (72.0%)</td><td>40 (13.3%)</td><td>44 (14.7%)</td></tr></table>

Table 22: Human pairwise preference between FACES and GPT-4o under the cap-15 blind protocol.

Human evaluation provides complementary evidence for the comparison with GPT-4o. Among the six directly prompted MLLMs, only GPT-4o is included in this human comparison. Table 22 reports the complete win, loss, and tie counts under the cap-15 blind protocol. Human annotators prefer FACES in 342 of 450 MESC-FT instances (76.0%) and 216 of 300 MELD-ESC300 instances (72.0%). These preferences agree in direction with both MLLM judges on both datasets. The human results support the comparison with GPT-4o specifically; comparisons with the other five MLLMs rely on the two automatic judges.

## D Detailed Preference Evaluation

## D.1 Human Annotation Protocol

The three primary annotators comprised two PhD students with experience in affective computing or emotional-support modeling and one lecturer in a related field. Each annotator evaluated all 450 MESC-FT test instances and all 300 MELD-ESC300 instances for the FACES–FIRES comparison. The additional FACES–GPT-4o comparison used the same annotators, complete evaluation sets, and annotation procedure; its results are reported in Appendix C.5.

<table><tr><td>Criterion</td><td>Guideline</td></tr><tr><td>Empathy</td><td>Whether the response recognizes and appropriately addresses the user&#x27;s emotion.</td></tr><tr><td>Multimodal</td><td>Whether the response is consistent with the dialogue context and visual evidence.</td></tr><tr><td>appropriateness</td><td>Whether the support strategy fits the user&#x27;s current needs without evident risk.</td></tr><tr><td>Strategy suitability</td><td>Whether the response provides practical supportive value.</td></tr><tr><td>Helpfulness</td><td>Select a winner only when one response is clearly better overall. Select Tie when neither</td></tr><tr><td>Tie rule</td><td>response has a clear overall advantage, including cases where the responses have different strengths and weaknesses.</td></tr></table>

Table 23: Criteria and decision rule for human pairwise preference annotation.

<table><tr><td>Dataset</td><td>FACES Win</td><td>FIRES Win</td><td>Tie</td><td>κ</td></tr><tr><td>MESC-FT</td><td>339 (75.3%)</td><td>63 (14.0%)</td><td>48 (10.7%)</td><td>0.62</td></tr></table>

Table 24: Final human preferences between FACES and FIRES on the 450-instance MESC-FT test set. κ denotes Fleiss’ kappa.

Each item presented the dialogue context, visual evidence, and two anonymized candidate responses under the cap-15 view defined in Appendix C.1. Candidate identities were hidden and their presentation order was randomized. Annotators selected Candidate A, Candidate B, or Tie based on the criteria in Table 23. The four criteria jointly guided a single overall preference decision rather than separate numerical scores for each criterion.

The three annotators evaluated each item independently without seeing one another’s labels. A label supported by at least two annotators became the final preference label. When all three labels differed, a fourth externally recruited psychology expert adjudicated the case. All reported preference counts include these adjudicated decisions where applicable. Fleiss’ κ was computed from the three primary annotators’ raw labels before adjudication, measuring agreement among human annotators.

The primary annotators were recruited internally at their university and compensated according to the actual annotation workload. The fourth psychology expert was externally recruited and compensated for adjudication.

## D.2 Human Preference Counts

Tables 24 and 25 report the final human preference counts for FACES versus FIRES. Percentages use all evaluated instances as the denominator, including ties. The reported Fleiss’ κ values measure agreement before adjudication, following the procedure in Appendix D.1.

<table><tr><td>Dataset</td><td>FACES Win</td><td>FIRES Win</td><td>Tie</td><td>κ</td></tr><tr><td>MELD-ESC300</td><td>211 (70.3%)</td><td>44 (14.7%)</td><td>45 (15.0%)</td><td>0.58</td></tr></table>

Table 25: Final human preferences between FACES and FIRES on the 300-instance MELD-ESC300 benchmark. κ denotes Fleiss’ kappa.

## D.3 Human–MLLM Agreement

We assess sample-level agreement for the FACES– FIRES comparison by pairing the final human preference label with the majority-vote label from Qwen2.5-VL-32B-Instruct-AWQ on the same cap-15 candidate pair. Both labels take one of three nominal values: FACES win, FIRES win, or Tie. We retain ties as a category and report exact agreement and unweighted Cohen’s κ in Table 26.

Exact agreement is the proportion of instances with identical human and MLLM labels. Cohen’s κ additionally adjusts for agreement expected from their marginal label distributions:

$$
\kappa = { \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } } ,\tag{13}
$$

$$
p _ { e } = \sum _ { c \in \mathcal { C } } p _ { H } ( c ) p _ { M } ( c ) ,\tag{14}
$$

where $p _ { o }$ is the observed exact agreement rate, ${ \mathcal { C } } =$ {FACES win, FIRES win, Tie}, and $p _ { H } ( c )$ and $p _ { M } ( { \boldsymbol { c } } )$ are the human and MLLM marginal label proportions.

MESC-FT has higher exact agreement but lower Cohen’s κ because both label distributions are more concentrated on FACES. The expected chance agreement is approximately 0.629 on MESC-FT, compared with 0.493 on MELD-ESC300. Both datasets therefore show agreement beyond chance, although substantial sample-level disagreement remains. These results support consistency with human preferences without establishing that the MLLM judge can replace human evaluation.

This analysis concerns the post-hoc Qwen judge on FACES–FIRES pairs. It does not measure agreement with GLM or on FACES–GPT-4o pairs. The separate analysis of training rewards against human numerical scores is reported in Appendix A.4.

<table><tr><td>Dataset</td><td>n</td><td>Human F/FIRES/Tie</td><td>MLLM F/FIRES/Tie</td><td>Exact Agreement</td><td>Cohen&#x27;s κ</td></tr><tr><td>MESC-FT</td><td>450</td><td>339/63/48</td><td>361/53/36</td><td>74.7% (336/450)</td><td>0.316</td></tr><tr><td>MELD-ESC300</td><td>300</td><td>211/44/45</td><td>186/29/85</td><td>71.7% (215/300)</td><td>0.441</td></tr></table>

Table 26: Agreement between final human preferences and Qwen2.5-VL-32B-Instruct-AWQ majority preferences for the FACES–FIRES comparison under cap-15. F denotes FACES; preference distributions are reported as counts.

<table><tr><td>Model</td><td>B-2↑</td><td>R-L ↑</td><td>M.↑</td><td> $\mathbf { F _ { B E R T } } \uparrow$ </td><td>Dist-3 ↑</td><td>DIFF-E.↓</td></tr><tr><td>w/o format</td><td>10.91</td><td>21.84</td><td>15.15</td><td>85.03</td><td>80.47</td><td>0.4293</td></tr><tr><td>w/o visual</td><td>8.89</td><td>20.59</td><td>19.92</td><td>86.51</td><td>98.74</td><td>0.4073</td></tr><tr><td>w/o emotion</td><td>9.44</td><td>19.89</td><td>19.37</td><td>86.95</td><td>97.92</td><td>0.6108</td></tr><tr><td>w/o situation</td><td>9.97</td><td>19.62</td><td>19.04</td><td>86.24</td><td>97.74</td><td>0.5918</td></tr><tr><td>w/o strategy</td><td>9.23</td><td>20.27</td><td>17.92</td><td>84.86</td><td>97.05</td><td>0.5251</td></tr><tr><td>Ours (Full)</td><td>9.68</td><td>21.09</td><td>20.62</td><td>87.14</td><td>99.63</td><td>0.3643</td></tr></table>

Table 27: Additional dimension-level ablation results on the MESC-FT test set. All metrics are higher-is-better except DIFF-E.

![](images/66e400d031d0e7b440f6253c9e9489113adc57985f32ce08257c365ddc265320.jpg)  
Figure 3: Normalized radar chart of dimension-level ablations. Larger values indicate better performance; DIFF-E. is reversed. FACES remains most balanced across metrics.

## E Additional Ablation Analysis

Table 27 reports ablations of individual reward dimensions and format control on MESC-FT. We ablate the dimensions used for intermediate reasoning and structural control, but do not ablate the outcome reward because it provides the shared final-response supervision needed for comparable training. Table 28 summarizes the function, responsible semantic span, and scoring signal of each reward dimension. The normalized trends are shown in Figure 3. These results complement the contribution-level ablations in the main text by showing how the ablated reward dimensions and format control affect performance across the six automatic evaluation metrics.

Ours (Full) achieves the best METEOR, BERTScore-F1, Distinct-3, and DIFF-EPITOME, indicating the most balanced overall profile. Removing the format constraint increases BLEU-2 and ROUGE-L, but substantially hurts semantic quality and diversity, suggesting that lexical overlap alone does not reflect better support quality. Removing emotion or situation rewards leads to larger DIFF-EPITOME degradation, while removing strategy reward lowers semantic metrics, showing that these dimensions provide complementary supervision for strategy-aligned responses.

## E.1 Reward Aggregation Diagnostics

We compare four aggregation rules using the same candidate responses from 369 prompts, with five responses initially sampled per prompt. Excluding responses with format scores below 0.5 leaves 1,829 candidates. Each rule selects the 457 candidates with the highest rewards, approximately 25% of the retained set. A critical failure is defined as a candidate whose Strategy or Outcome score falls in the bottom quartile of the corresponding score distribution across all retained candidates.

Table 29 shows that geometric mean, Tchebycheff, and gated aggregation each yield a critical failure rate of 0.0%, compared with 2.4% for linear aggregation. Among the four methods, gated aggregation achieves the highest mean Strategy and Outcome scores. Because none of the candidates ranked in the top quartile by the other three aggregation rules meets the critical failure criterion, our additional diagnostic focuses on the 11 candidates that meet this criterion under linear aggregation. We compare these candidates with the gated top quartile using three signals that are not produced by the Qwen judge. Relative to the mean scores of the gated top quartile, these linear critical failures score

<table><tr><td>Dimension</td><td>Targeted failure mode</td><td>Responsible span</td><td>Scoring signal</td></tr><tr><td>Visual</td><td>Incorrect or missing use of visual evidence</td><td>Visual-perception span</td><td>CLIP similarity</td></tr><tr><td>Emotion</td><td>Missing or incorrect affect recognition</td><td>Emotion-understanding span</td><td>MiniLM similarity</td></tr><tr><td>Situation</td><td>Incorrect attribution of the user&#x27;s condition</td><td>Situation-understanding span</td><td>MLLM judgment</td></tr><tr><td>Strategy</td><td>Inappropriate or unsafe support strategy</td><td>Strategy-planning span</td><td>MLLM judgment; multiplicative gate</td></tr><tr><td>Outcome</td><td>Unhelpful or misaligned final response</td><td>Final-response span</td><td>MLLM judgment; multiplicative gate</td></tr></table>

Table 28: Roles of the five reward dimensions. The taxonomy follows the MESC-FT/FIRES flexible-thinking structure and adds Outcome to capture final-response failures.

<table><tr><td>Aggregation</td><td>Definition</td><td>Top-Str. ↑</td><td>Top-Out. ↑</td><td>Critical-Fail ↓</td></tr><tr><td>Linear</td><td>Equal-weight mean over five dimensions</td><td>0.325</td><td>0.328</td><td>2.4%</td></tr><tr><td>Geometric mean</td><td>Geometric mean over five dimensions</td><td>0.313</td><td>0.317</td><td>0.0%</td></tr><tr><td>Tchebycheff</td><td>Minimum over five dimensions</td><td>0.314</td><td>0.311</td><td>0.0%</td></tr><tr><td>Gated (Ours)</td><td>mean(Visual, Emotion, Situation) × Strategy × Outcome</td><td>0.343</td><td>0.346</td><td>0.0%</td></tr></table>

Table 29: Offline comparison of reward aggregation schemes. Top-Str. and Top-Out. denote mean Strategy and Outcome scores among top-25% candidates.

<table><tr><td>Diagnostic</td><td>Value</td><td>Linear</td><td>Gated (Ours)</td></tr><tr><td>High-risk region fraction</td><td>65.0%</td><td></td><td></td></tr><tr><td>Mean reward in QHL</td><td></td><td>0.4465</td><td>0.0096</td></tr><tr><td>Global mean reward</td><td></td><td>0.4320</td><td>0.0349</td></tr><tr><td> ${ \mathcal { Q } } _ { \mathrm { H L } }$  / global mean</td><td></td><td>103.4%</td><td>27.6%</td></tr><tr><td>Groups assigning maximum advantage to a critical failure</td><td></td><td>14.9%</td><td>11.1%</td></tr></table>

Table 30: Reward-geometry diagnostics for candidates with strong cognitive signals but weak Strategy or Outcome quality.

9.4% lower on MiniLM similarity, 2.9% lower on EPITOME, and 6.2% lower on ROUGE-L. This diagnostic provides additional evidence that linear aggregation can assign high rewards to candidates that appear weaker on average under signals independent of the Qwen judge.

We further examine the reward geometry in the high-risk region

$$
\mathcal { Q } _ { \mathrm { H L } } = \left\{ R ^ { \mathrm { C o g } } \geq 0 . 6 , { r ^ { \mathrm { S t r } } } { r ^ { \mathrm { O u t } } } < 0 . 1 \right\} ,\tag{15}
$$

which contains 65.0% of the format-valid candidates. This region represents candidates with strong cognitive signals but critically weak strategy or outcome quality.

Table 30 shows that the mean reward within ${ \mathcal { Q } } _ { \mathrm { H L } }$ is 103.4% of the global mean under linear aggregation, compared with 27.6% under gated aggregation. Across the 369 prompt groups, the proportion of groups in which a critical failure receives the maximum advantage decreases from 14.9% under linear aggregation to 11.1% under gated aggregation. These results show that gated aggregation suppresses rewards for candidates with strong cognitive signals but weak Strategy or Outcome quality. It also reduces, but does not eliminate, the selection of critical failures within prompt groups.

## E.2 Fine-grained Credit Diagnostic

To further examine span level credit assignment, we use a supplementary five dimensional diagnostic on 100 fixed MESC-FT validation instances sampled with seed 42. For each response, CLIP scores Visual consistency, MiniLM scores Emotion alignment, and Qwen2.5-VL-32B-Instruct-AWQ scores Situation, Strategy, and Response. Let $q _ { i } ^ { d }$ denote the normalized score assigned to response i on dimension d. For d = Vis and $d =  { \mathrm { E m o } } , q _ { i } ^ { d }$ is the corresponding CLIP or MiniLM score. For each Qwen evaluated dimension d ∈ {Sit, Str, Resp}, the judge produces $s _ { i , k } ^ { d } ~ \in ~ \{ 1 , 2 , 3 , 4 , 5 \}$ under prompt variant k. We map these scores to [0, 1] and average them across the three variants:

$$
q _ { i } ^ { d } = \frac { 1 } { 3 } \sum _ { k = 1 } ^ { 3 } \frac { s _ { i , k } ^ { d } - 1 } { 4 } .\tag{16}
$$

All five scores are in [0, 1], and the composite score is

<table><tr><td>Variant</td><td>Composite ↑ Change</td><td></td></tr><tr><td>Full model</td><td>50.6</td><td></td></tr><tr><td>Without span alignment</td><td>45.2</td><td> $- 1 0 . 7 \%$ </td></tr></table>

Table 31: Supplementary diagnostic for credit assignment to semantic spans.

$$
\begin{array} { c } { C _ { i } = 1 5 q _ { i } ^ { \mathrm { V i s } } + 2 0 q _ { i } ^ { \mathrm { E m o } } + 2 0 q _ { i } ^ { \mathrm { S i t } } } \\ { + 2 0 q _ { i } ^ { \mathrm { S t r } } + 2 5 q _ { i } ^ { \mathrm { R e s p } } , } \end{array}\tag{17}
$$

which ranges from 0 to 100. This supplementary diagnostic differs from the three dimensional 1–5 MLLM evaluation used for the main results.

Table 31 shows a 10.7% decrease in the composite score, dropping from 50.6 to 45.2, when feedback is decoupled from its corresponding spans, suggesting that broadcasting feedback across the entire generation hinders the targeted learning of critical empathetic behaviors.

## F Hyperparameter Sensitivity Analysis

We vary $\tau _ { v } , \gamma , m$ , and λ individually to assess FACES sensitivity. These set visual reward sparsity, the minimum scale for positive advantages, the number of final tokens receiving full advantage, and the weight of earlier tokens, respectively. $\mathsf { A p - }$ pendix A gives the fixed training, decoding, and evaluation settings. Defaults and sensitivity results use the MESC-FT validation set; we compare endpoint anchoring with linear and exponential decay using validation rewards and MESC-FT test-set response evaluations.

Table 32 shows moderate robustness to changes in the visual threshold $\tau _ { v } .$ . Lower thresholds make visual rewards more permissive and slightly improve lexical overlap but weaken semantic discrimination; higher thresholds make rewards sparse and reduce semantic and MLLM performance. The default $\tau _ { v } = 0 . 6 0$ gives the best DIFF-E. and MLLM average, though not every metric.

Table 33 shows that FACES is stable around the default gate floor. Since $g _ { i }$ is dynamically computed from $\gamma$ and $R _ { i } / R _ { \operatorname* { m a x } }$ , we analyze $\gamma$ rather than treating $g _ { i }$ as an independent hyperparameter. When $\gamma$ is too small, positive local advantages are overly suppressed. $\mathbf { A s } \ \gamma$ increases, the gate becomes weaker. Setting $\gamma = 1 . 0 0$ improves B-2 and R-L but worsens DIFF-E. and the MLLM average, supporting the use of the gate to limit positive updates for candidates whose total rewards are low relative to the best candidate in the same group.

Table 34 analyzes the endpoint-anchored creditassignment parameters on the held-out validation set. A shorter anchor window or a smaller λ overemphasizes segment endings and weakens supervision for earlier reasoning tokens. A larger anchor window is still stable but slightly less balanced. When $\lambda = 1 . 0 0$ , all tokens in a segment receive the same advantage, which improves lexical-overlap metrics but reduces semantic and MLLM-based quality. The default setting $( m = 5 , \lambda = 0 . 8 0 )$ provides a good balance between endpoint emphasis and full-span supervision.

## F.1 Credit Assignment Alternatives

We compare endpoint anchoring with linear and exponential decay using the same data, training budget, and evaluation protocol. Let $j \in \{ 0 , \ldots , L -$ 1} index a token in a span of length L. The unnormalized linear and exponential schedules are

$$
\widetilde w _ { j } ^ { \mathrm { l i n } } = \lambda + ( 1 - \lambda ) \frac { j } { L - 1 } ,\tag{18}
$$

$$
\widetilde { w } _ { j } ^ { \mathrm { e x p } } = 0 . 7 ^ { L - 1 - j } ,\tag{19}
$$

where $\lambda \ : = \ : 0 . 8$ and $\widetilde { w } _ { 0 } ^ { \mathrm { l i n } } = 1$ when $L \ = \ 1$ To isolate the positional distribution of credit, we rescale both alternatives to match the total token weight of endpoint anchoring within each span. We define the target total weight as

$$
S _ { L } = \operatorname* { m i n } ( m , L ) + \lambda \operatorname* { m a x } ( L - m , 0 ) .\tag{20}
$$

We use

$$
w _ { j } ^ { q } = S _ { L } \frac { \widetilde { w } _ { j } ^ { q } } { \sum _ { k = 0 } ^ { L - 1 } \widetilde { w } _ { k } ^ { q } } \quad q \in \{ \mathrm { l i n } , \mathrm { e x p } \} ,\tag{21}
$$

with $m = 5$ and $\lambda \ : = \ : 0 . 8$ . Hence, all three mechanisms assign the same total weight $S _ { L }$ to each span and differ only in how that weight is distributed across token positions. Each weight scales the corresponding span advantage.

We evaluate the three trained models using two protocols. Table 35 reports scores on the same fixed set of 100 MESC-FT validation instances using the reward functions employed during training. Table 36 reports pointwise evaluation on all 450 MESC-FT test instances using Qwen2.5-VL-32B-Instruct-AWQ. For this evaluation, reasoning traces are removed and responses are truncated to their first 15 words. The judge rates three dimensions from 1 to 5 using the three semantically equivalent prompts described in Appendix B.2.

<table><tr><td> $\scriptstyle \tau _ { v }$ </td><td>B-2↑</td><td>R-L ↑</td><td>M.↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E. ↓</td><td>MLLM Avg. ↑</td></tr><tr><td>0.50</td><td>9.91</td><td>21.36</td><td>20.19</td><td>86.84</td><td>99.18</td><td>0.3984</td><td>3.94</td></tr><tr><td>0.60</td><td>9.74</td><td>21.16</td><td>20.48</td><td>87.03</td><td>99.42</td><td>0.3726</td><td>4.01</td></tr><tr><td>0.65</td><td>9.65</td><td>21.02</td><td>20.35</td><td>86.99</td><td>99.46</td><td>0.3795</td><td>3.98</td></tr><tr><td>0.70</td><td>9.56</td><td>20.84</td><td>20.07</td><td>86.83</td><td>99.29</td><td>0.4017</td><td>3.93</td></tr></table>

Table 32: Sensitivity analysis of the CLIP visual threshold $\tau _ { v }$ on the MESC-FT validation set.

<table><tr><td>γ</td><td>B-2↑</td><td>R-L ↑</td><td>M.↑</td><td> $\mathbf { F _ { B E R T } } \uparrow$ </td><td>Dist-3 ↑</td><td>DIFF-E. ↓</td><td>MLLM Avg. ↑</td></tr><tr><td>0.00</td><td>9.55</td><td>20.87</td><td>20.27</td><td>86.89</td><td>99.20</td><td>0.3911</td><td>3.95</td></tr><tr><td>0.10</td><td>9.74</td><td>21.16</td><td>20.48</td><td>87.03</td><td>99.42</td><td>0.3726</td><td>4.01</td></tr><tr><td>0.30</td><td>9.96</td><td>21.48</td><td>20.15</td><td>86.82</td><td>99.21</td><td>0.3995</td><td>3.94</td></tr><tr><td>1.00</td><td>10.13</td><td>21.88</td><td>19.88</td><td>86.61</td><td>98.96</td><td>0.4387</td><td>3.86</td></tr></table>

Table 33: Sensitivity analysis of the positive gate floor γ on the MESC-FT validation set. When $\gamma = 1 . 0 0 \mathrm { { : } }$ , positive gating is effectively removed.

<table><tr><td>m</td><td>λ</td><td>B-2↑</td><td>R-L↑</td><td>M. ↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E.↓</td><td>MLLM Avg. ↑</td></tr><tr><td>3</td><td>0.80</td><td>9.61</td><td>20.98</td><td>20.26</td><td>86.91</td><td>99.30</td><td>0.3878</td><td>3.96</td></tr><tr><td>5</td><td>0.60</td><td>9.58</td><td>20.90</td><td>20.18</td><td>86.88</td><td>99.24</td><td>0.3946</td><td>3.94</td></tr><tr><td>5</td><td>0.80</td><td>9.74</td><td>21.16</td><td>20.48</td><td>87.03</td><td>99.42</td><td>0.3726</td><td>4.01</td></tr><tr><td>7</td><td>0.80</td><td>9.83</td><td>21.25</td><td>20.34</td><td>86.96</td><td>99.46</td><td>0.3819</td><td>3.98</td></tr><tr><td>5</td><td>1.00</td><td>10.02</td><td>21.52</td><td>20.08</td><td>86.80</td><td>99.21</td><td>0.4135</td><td>3.91</td></tr></table>

Table 34: Sensitivity to the credit assignment parameters on the MESC-FT validation set. The final m tokens of each span receive the full advantage, while earlier tokens receive λ times that advantage.

<table><tr><td>Token Credit</td><td>n</td><td>Format</td><td>Visual</td><td>Emotion</td><td>Situation</td><td>Strategy</td><td>Outcome</td><td>Overall</td></tr><tr><td>Endpoint-anchored (Ours)</td><td>100</td><td>0.9700</td><td>0.9300</td><td>0.8626</td><td>0.2608</td><td>0.1692</td><td>0.1558</td><td>0.0413</td></tr><tr><td>Linear decay</td><td>100</td><td>0.9700</td><td>0.9000</td><td>0.8627</td><td>0.2558</td><td>0.1500</td><td>0.1417</td><td>0.0360</td></tr><tr><td>Exponential decay</td><td>100</td><td>1.0000</td><td>0.9300</td><td>0.8911</td><td>0.2642</td><td>0.1525</td><td>0.1392</td><td>0.0347</td></tr></table>

Table 35: Training-reward comparison of token-credit mechanisms on 100 validation samples.

<table><tr><td>Token Credit</td><td>n</td><td>Empathy</td><td>Multimodal App.</td><td>Context-Response Cons.</td><td>Avg.</td></tr><tr><td>Endpoint-anchored (Ours)</td><td>450</td><td>3.98</td><td>4.03</td><td>4.11</td><td>4.04</td></tr><tr><td>Linear decay</td><td>450</td><td>3.87</td><td>3.96</td><td>4.00</td><td>3.94</td></tr><tr><td>Exponential decay</td><td>450</td><td>3.80</td><td>3.91</td><td>3.96</td><td>3.89</td></tr></table>

Table 36: Independent post-hoc MLLM evaluation of alternative token-credit mechanisms on MESC-FT.

Table 35 shows that endpoint anchoring achieves the highest Strategy, Outcome, and Overall rewards. Its Overall reward reaches 0.0413, exceeding linear and exponential decay by 14.7% and 19.0%, respectively. Although exponential decay obtains higher Format, Emotion, and Situation scores and matches endpoint anchoring on Visual, these advantages do not result in a higher Overall reward. Endpoint anchoring performs better on Strategy and Outcome, which serve as multiplicative gates in the final reward. The results indicate that endpoint anchoring achieves a better balance between intermediate reasoning and final response quality, even though it does not lead on every individual reward dimension.

Table 36 shows that endpoint anchoring obtains the highest observed scores across all three dimensions, averaging 4.04 compared with 3.94 for linear decay and 3.89 for exponential decay. Together with the validation-reward results, this pattern suggests that retaining supervision over earlier tokens while emphasizing span endings is beneficial in the evaluated setting.

## G External Dataset

We use MESC-FT as the sole training set because its multimodal emotional support setting and supervision structure align with the structured reasoning and multidimensional reward design of FACES. Adding other datasets during training would change both the amount and distribution of training data, making it difficult to distinguish the effect of the method from the effect of additional data. Accordingly, we reserve MELD-ESC300, StickerConv, and AvaMERG for cross-dataset evaluation without additional training. This choice follows ESC evaluations on datasets not strictly dedicated to ESC: FIRES evaluates on an ESC-oriented EMOTyDA subset (Wang et al., 2025a), while straQ\* evaluates on EmpatheticDialogues (Wang et al., 2025b).

## G.1 Benchmark Construction

To evaluate out-of-domain generalization, we construct an external benchmark, MELD-ESC300, from the MELD test set using the publicly available EMOTyDA resources (Poria et al., 2019; Saha et al., 2020), and use it for cross-dataset evaluation without any additional fine-tuning. Rather than directly reusing the original MELD annotations, we filter dialogue segments that better approximate emotional-support scenarios, allowing us to examine whether the model’s gains transfer to a nontraining-domain multimodal dialogue setting.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total samples</td><td>300</td></tr><tr><td>strict samples</td><td>208</td></tr><tr><td>relaxed samples</td><td>92</td></tr><tr><td>Frames per sample</td><td>5</td></tr><tr><td>Maximum context turns</td><td>6</td></tr><tr><td>Emotion: neutral</td><td>232</td></tr><tr><td>Emotion: joy</td><td>51</td></tr><tr><td>Emotion: surprise</td><td>17</td></tr></table>

Table 37: Summary statistics of MELD-ESC300. Strict denotes samples whose immediately preceding utterance already exhibits negative affect; relaxed denotes samples where the immediately preceding utterance does not, but one of the three most recent preceding utterances does. For all samples, the visual input is taken from the video of the immediately preceding utterance, from which five frames are uniformly extracted.

Concretely, we treat the i-th utterance in a dialogue as a candidate target response and retain only samples satisfying the following conditions: (1) at least two preceding turns are available; (2) the target response length is between 4 and 40 words; (3) the target sentiment is not negative and its emotion label belongs to {neutral, joy, surprise}; this condition is used only to exclude clearly negative target replies and is not treated as a proxy for empathy; (4) the target speaker differs from the speaker in the immediately preceding utterance; and (5) the video clip corresponding to the immediately preceding utterance is available, so that stable visual context can be provided.

We then apply a two-stage filtering strategy to identify ESC-style samples. A sample is marked as strict if the immediately preceding utterance already exhibits negative affect, i.e., its sentiment is negative or its emotion label belongs to {anger, sadness, fear, disgust}. If this condition is not met, we further inspect the three most recent preceding utterances; if any of them contains negative affect, the sample is retained and marked as relaxed. Importantly, regardless of whether a sample is selected by the strict or relaxed rule, the visual input is always taken from the video of the immediately preceding utterance, from which we uniformly extract five frames. This design avoids directly leaking visual information associated with the target response itself. The final model input contains at most six preceding turns, and the reference output is the original target utterance.

<table><tr><td>Subset</td><td>n</td><td>Emotional Need</td><td>Relevant Response</td><td>Supportive Function</td><td>Overall Validity</td></tr><tr><td>Strict</td><td>208</td><td>204 (98.1%)</td><td>201 (96.6%)</td><td>196 (94.2%)</td><td>193 (92.8%)</td></tr><tr><td>Relaxed</td><td>92</td><td>89 (96.7%)</td><td>87 (94.6%)</td><td>83 (90.2%)</td><td>81 (88.0%)</td></tr><tr><td>Overall</td><td>300</td><td>293 (97.7%)</td><td>288 (96.0%)</td><td>279 (93.0%)</td><td>274 (91.3%)</td></tr></table>

Table 38: Expert audit of MELD-ESC300 construction validity.

(a) Pointwise MLLM evaluation results on MELD-ESC300. Scores are on a 1–5 scale, and higher is better.
<table><tr><td>Model</td><td>Empathy ↑</td><td>Multimodal App. ↑</td><td>Context-Response Cons. ↑</td><td>Avg. ↑</td></tr><tr><td>FIRES (Baseline)</td><td>2.59</td><td>3.61</td><td>3.84</td><td>3.35</td></tr><tr><td>Ours (Full)</td><td>3.27</td><td>3.85</td><td>4.06</td><td>3.73</td></tr></table>

(b) Pairwise MLLM preference evaluation results on MELD-ESC300. Win rates are computed over 300 test instances.
<table><tr><td>Model A</td><td>Model B</td><td>A Win</td><td>B Win</td><td>Tie</td><td>A Win Rate</td><td>B Win Rate</td><td>Tie Rate</td></tr><tr><td>Ours (Full)</td><td>FIRES (Baseline)</td><td>186</td><td>29</td><td>85</td><td>62.00%</td><td>9.67%</td><td>28.33%</td></tr></table>

Table 39: Length-controlled MLLM evaluation results on MELD-ESC300. Panel (a) reports pointwise scores across three evaluation dimensions, and panel (b) reports blind pairwise preferences over 300 test instances. Pairwise results are significant under a two-sided sign test with ties excluded $( p < 0 . 0 0 1 )$ .

Following this procedure with a fixed random seed of 42, we construct a 300-instance benchmark, MELD-ESC300. Of these, 208 satisfy the strict criterion and 92 are selected by the relaxed criterion. As shown in Table 37, the target responses are dominated by neutral cases, with a smaller yet non-trivial number of joy and surprise instances. We report only the emotion distribution, as the corresponding sentiment labels are largely implied by the retained target emotion categories in this subset and therefore add limited additional information. These statistics make the composition of the benchmark explicit and reproducible. We also conduct an expert audit of all 300 samples to assess emotional need, response relevance, supportive function, and overall validity.

A sample is considered valid only if the preceding context conveys an emotional need and the target response is relevant and plausibly supportive. The same three annotators described in Appendix D.1 independently audited all samples, with the final label for each criterion determined by majority vote. The non-negative target-emotion filter is intended to exclude clearly negative replies, rather than to serve as a proxy for empathy. As shown in Table 38, 274 of the 300 samples pass the overall validity criterion, yielding validity rates of 91.3% overall, 92.8% for the strict subset, and 88.0% for the relaxed subset. For example, when Phoebe expresses anger about a bad date, Monica responds, “Don’t worry Phoebe, you’re gonna meet someone.” In another case, Joey criticizes himself for being unable to finish his writing, and Ross responds, “I’ll help you.” These examples illustrate that the filtering procedure captures a negative context followed by a relevant and plausibly supportive response, rather than treating the target emotion label as evidence of empathy. Overall, the benchmark centers on mildly supportive or nonnegative target replies while still covering a modest range of more explicitly positive feedback.

We emphasize that MELD-ESC300 is derived from scripted TV dialogue rather than professionally curated emotional-support data, so it is intended as a controlled external generalization test set rather than a replacement for dedicated ESC benchmarks. By combining recent negative-affect context, neutral-to-positive target replies, and visual evidence from the immediately preceding turn, it provides a practical approximation of crossdomain emotional-support scenarios for evaluating transfer beyond the MESC-FT test set.

## G.2 MELD-ESC300 MLLM Evaluation

Table 39 shows that the pointwise average increases from 3.35 for FIRES to 3.73 for FACES, with higher scores on all three dimensions. In the blind pairwise comparison, FACES is preferred in 62.0% of the 300 instances, compared with 9.7% for FIRES; the preference share differs from chance when ties are excluded $( p < 0 . 0 0 1 )$ . These results provide evidence that the advantage observed on MESC-FT transfers to MELD-ESC300 under the tested cap-15 protocol without additional training.

<table><tr><td>Model</td><td>B-2 ↑</td><td>R-L ↑</td><td>M. ↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E. ↓</td></tr><tr><td>GPT-40</td><td>1.41</td><td>5.73</td><td>13.04</td><td>95.35</td><td>54.53</td><td>0.5053</td></tr><tr><td>Claude-Haiku-4.5</td><td>1.68</td><td>6.81</td><td>15.12</td><td>95.56</td><td>64.18</td><td>0.7507</td></tr><tr><td>Gemini-2.5-Flash-Lite</td><td>2.47</td><td>7.82</td><td>13.47</td><td>96.02</td><td>56.19</td><td>0.2994</td></tr><tr><td>GLM-4.1V-Thinking-FlashX</td><td>1.82</td><td>6.23</td><td>13.60</td><td>95.74</td><td>69.25</td><td>0.8126</td></tr><tr><td>Doubao-1.6-Vision</td><td>1.70</td><td>6.67</td><td>11.80</td><td>95.65</td><td>74.42</td><td>0.7057</td></tr><tr><td>Kimi-K2.5</td><td>1.63</td><td>6.06</td><td>14.47</td><td>95.60</td><td>73.10</td><td>0.8056</td></tr><tr><td>Ours (Full)</td><td>5.59</td><td>16.57</td><td>15.16</td><td>86.23</td><td>94.07</td><td>0.7664</td></tr></table>

Table 40: Automatic evaluation of FACES and six directly prompted MLLMs on MELD-ESC300. Higher values are better for all metrics except DIFF-E. The best result for each metric is shown in bold.

<table><tr><td>Dataset</td><td>View</td><td>n</td><td>FACES Win</td><td>FIRES Win</td><td>Tie</td><td>FACES Share Excl. Ties</td><td>Sign-test p</td></tr><tr><td>StickerConv</td><td>cap-15</td><td>300</td><td>63.7%</td><td>18.3%</td><td>18.0%</td><td>77.6% (72.4–82.9%)</td><td> $8 . 9 \times 1 0 ^ { - 1 9 }$ </td></tr><tr><td>AvaMERG</td><td>cap-15</td><td>300</td><td>79.3%</td><td>11.3%</td><td>9.4%</td><td>87.5% (83.5–91.5%)</td><td> $7 . 2 \times 1 0 ^ { - 3 9 }$ </td></tr><tr><td>StickerConv</td><td>cap-25</td><td>300</td><td>75.7%</td><td>15.0%</td><td>9.3%</td><td></td><td></td></tr><tr><td>AvaMERG</td><td>cap-25</td><td>300</td><td>85.0%</td><td>8.7%</td><td>6.3%</td><td></td><td></td></tr><tr><td>StickerConv</td><td>response-only</td><td>300</td><td>82.3%</td><td>12.0%</td><td>5.7%</td><td></td><td></td></tr><tr><td>AvaMERG</td><td>response-only</td><td>300</td><td>90.7%</td><td>5.7%</td><td>3.6%</td><td></td><td></td></tr></table>

Table 41: External zero-shot blind pairwise evaluation on StickerConv and AvaMERG.

## G.3 Additional Automatic Baselines

Table 40 reports additional automatic evaluation results for six strong, directly prompted MLLM baselines on MELD-ESC300 under the same protocol as in the main text.

## G.4 Additional External Generalization

Beyond MELD-ESC300, we evaluate transfer to StickerConv and AvaMERG without additional training. Table 41 reports the blind pairwise results, while Table 42 reports the corresponding automatic evaluation. We sample 300 test instances from each dataset with seed 42. StickerConv uses textual sticker cues in the dialogue context, while AvaMERG uses dialogue text and five uniformly sampled facial frames per instance, with audio excluded. For each dataset, responses are generated greedily and evaluated using the same blind pairwise protocol as the main experiments: three semantically equivalent judge prompts, bidirectional candidate swapping, prefix truncation under cap-15 and cap-25 settings, and majority voting. Statistical significance is assessed with a two-sided exact binomial sign test over non-tied comparisons, and confidence intervals are estimated using 5,000 bootstrap resamples.

The automatic results are complementary rather than uniformly one-sided. FACES mainly improves supportive semantics, diversity, and DIFF-EPITOME, while FIRES remains competitive on some lexical or embedding-based metrics.

## H Case Study

To complement the controlled pairwise results, we provide qualitative examples that illustrate where FACES improves over the baseline and where it can still fail. Across the success cases, FIRES often paraphrases the user’s surface expression, whereas FACES more consistently follows a chain of specific emotion labeling, causal situation interpretation, and targeted exploration. Under aggressive cap-15 truncation, the higher tie rate may reflect the removal of explanatory and strategy-related content, but the FIRES win rate remains low, suggesting that truncation reduces separability rather than reversing the preference. The failure case illustrates an important boundary: once the visual cue is misread, the error can propagate through situation inference and strategy selection. In panel (F) of Figure 4, Mona holds a photograph, which FIRES correctly describes as a picture taken of her. FACES, however, mistakes it for a phone and responds as though she were showing something to Ross and Joey. The result is a fluent but visually ungrounded follow-up. Gated reward fusion can suppress poor final behavior during training, but it cannot fully repair incorrect upstream perception. The figure compares the dialogue context, visual cue, baseline response, FACES response, and a brief analysis for six representative cases, covering in-domain success, length-controlled success, cross-dataset transfer, and a failure case.

<table><tr><td>Dataset</td><td>Model</td><td>B-2↑</td><td>R-L↑</td><td>M.↑</td><td>FBERT ↑</td><td>Dist-3 ↑</td><td>DIFF-E.↓</td></tr><tr><td>StickerConv</td><td>FACES</td><td>20.13</td><td>25.46</td><td>34.08</td><td>95.99</td><td>68.19</td><td>0.4382</td></tr><tr><td>StickerConv</td><td>FIRES</td><td>8.55</td><td>22.02</td><td>17.94</td><td>96.24</td><td>60.51</td><td>0.4715</td></tr><tr><td>AvaMERG</td><td>FACES</td><td>5.80</td><td>11.64</td><td>19.42</td><td>96.11</td><td>74.55</td><td>0.5127</td></tr><tr><td>AvaMERG</td><td>FIRES</td><td>5.28</td><td>14.48</td><td>13.71</td><td>96.62</td><td>67.11</td><td>0.5406</td></tr></table>

Table 42: Automatic evaluation on StickerConv and AvaMERG.

![](images/d4a6bae46939a1789de6d217b7ba90e8874f3aa1ab33dba5425e562a790375a8.jpg)  
Figure 4: Qualitative comparison of FACES and FIRES. Panels (A)–(B) show MESC-FT successes in standard and cap-15 settings; (C)–(E) show transfer successes on MELD-ESC300, StickerConv, and AvaMERG, respectively; and (F) shows a failure caused by a misread visual cue.