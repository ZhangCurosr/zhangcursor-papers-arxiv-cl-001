# EVERY TOKEN NEEDS ITS OWN TEACHER: MULTI-TEACHER ADAPTIVE ON-POLICY SELF-DISTILLATION

Yifu Luo <sup>1,2,3</sup>, Chuang Ma <sup>4,5</sup>, Zeyu Chen <sup>1,2</sup>,

Enkelejda Kasneci <sup>1,2</sup>, Qi Chen <sup>6,7</sup>, Gjergji Kasneci <sup>1,2</sup>

<sup>1</sup>TU Munchen¨ <sup>2</sup>Munich Center for Machine Learning (MCML)

<sup>3</sup>relAI – Konrad Zuse School of Excellence in Reliable AI <sup>4</sup>Kyoto University <sup>5</sup>NII LLMC <sup>6</sup>ELLIS Institute Finland <sup>7</sup>Aalto University

## ABSTRACT

On-policy Self-distillation (OPSD) has proven effective for post-training large language models by constructing a teacher conditioned on privileged references. However, existing approaches keep the teacher fixed either throughout training by consistently exposing it to the full reference, or within each response by adaptively varying partial reference exposure only across responses. In this paper, we argue that neither of them is sufficient. Instead, every token needs its own teacher, as the teacher preference can vary token by token within the same response. Through a teacher-intervention pilot study, we validate that the teacher preference differs across tokens, and reveal that a preferred teacher can be identified for each token through majority voting and KL-based selection. Motivated by these findings, we propose Multi-teacher Adaptive On-policy Self-Distillation (MTA-OPSD), the first token-level adaptive OPSD method. Our method constructs a multi-teacher pool consisting of candidate teachers, and enables every token to adaptively select its preferred teacher during training. Experiments with Qwen3 models on mathematical reasoning tasks show that MTA-OPSD consistently outperforms existing OPSD approaches with an average gain of up to +5.9 over the base model, highlighting the importance of adapting teacher supervision at the token level. The code is available at https://github.com/xingzhejun/MTA-OPSD.

## 1 INTRODUCTION

On-policy Self-distillation (OPSD) (Zhao et al., 2026; Hubotter et al., 2026; Shenfeld et al., 2026;¨ Cui et al., 2026) has emerged as a promising paradigm for post-training large language models (LLMs), showing superiority over other post-training paradigms like Supervised Fine-tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR) (e.g., GRPO (Guo et al., 2025)). In OPSD, the same model serves as both student and teacher. The student generates rollout responses, while the teacher provides supervision with access to privileged references, such as reference so lutions. Compared with RLVR, OPSD provides dense token-level learning signals beyond sparse outcome rewards. Compared with SFT, it trains on trajectories sampled by the student itself, mitigating the exposure bias (Bengio et al., 2015).

Despite these advantages, recent studies (Kim et al., 2026; Kaur et al., 2026; Harne et al., 2026; Armandpour et al., 2026) reveal that OPSD can also suffer from degraded reasoning performance, which stems from the highly informative references exposed to the teacher. It may induce overly sharp teacher distributions, producing supervision that is poorly aligned with the student’s learning dynamics (e.g., suppressing uncertainty (Kim et al., 2026; Zhang et al., 2026)). A line of work (Tan & Hong, 2026; Han et al., 2026b; Yang et al., 2026c) has attempted to address this student-teacher mismatch through adaptive reference exposure. Rather than providing the teacher with the complete reference, it adjusts how much of the reference the teacher receives based on the student’s learning dynamics, such as the student-teacher gap. However, these methods adapt at the response level. Although reference exposure may vary across responses, it remains fixed across tokens within each response. In other words, all tokens in a rollout are supervised by the same teacher.

In this paper, we challenge this consensus by arguing:

The adaptive teacher should operate at the token level, because every token requires its own dedicated teacher.

A teacher that provides useful guidance at one token may be less suitable at another, even within the same response. Rather than treating it with the same teacher, each token in the response merits consideration with its own preferred teacher. As illustrated in Figure 1, when different teachers are applied to the same response, they can provide similar learning signals for some tokens but disagree on the direction of supervision for others. This suggests that not only each response, but also each token, may have distinct teacher preferences.

While some prior work (Wang et al., 2026c; Hao et al., 2026; Xu et al., 2026b; Ke et al.,

![](images/b88287a39a7af48713b494f5f6b614c61988972a19f6ee7efecf403fe1df9cac.jpg)  
Figure 1: Token-level learning signals induced by different teachers for the same response. Teachers agree at some positions but provide opposing signals at others, motivating token-level teacher selection. Additional details on this visualization are provided in Section A.

2026; Wei et al., 2026; Wang et al., 2026a; Li et al., 2026; Han et al., 2026a) has explored token selection for post-training (e.g., focusing solely on high-entropy tokens (Wang et al., 2026c)), how to validate our claim is challenging, and how to adaptively identify the preferred teacher for each token remains unexplored. To answer these, we conduct a teacher-intervention pilot study (Section 4) to quantify the token-level teacher preference. As shown in Figure 2, we preserve a student-generated prefix, allow a candidate teacher to generate a short continuation at a selected token position, and then return control to the student to complete the response. The accuracy gain from this “student teacher-student” relay over pure student generation serves as a proxy for the teacher preference at that token position. A large gain implies a compatible teacher for that position, while a small gain suggests the opposite. Interestingly, the results are simple, yet clear and compelling. First, teacher preferences differ across tokens, even within the same response. Second, at the token level, majority voting yields a subset of reliable teachers, and the preferred teacher always exhibits a large student-teacher KL distance. The first finding validates our claim, and the second finding motivate a practical rule for token-level teacher selection.

Based on these findings, we propose Multi-teacher Adaptive On-policy Self-Distillation (MTA-OPSD), the first token-level adaptive OPSD method. As shown in Figure 3, MTA-OPSD constructs a multi-teacher pool consisting of candidate teachers, by varying the truncation strategy and exposure ratio of the privileged reference. During training, rather than using a fixed teacher for all tokens in the same response, each token adaptively selects its own teacher from the multi-teacher pool based on its preference, through majority voting followed by KL-based selection. Our token-level adaptive OPSD design is simple yet effective, precisely addressing the student-teacher mismatch. By assigning each token its preferred teacher, MTA-OPSD provides a more balanced learning signal rather than an overly sharp, rigid teacher distribution from a single teacher. We evaluate MTA-OPSD on mathematical reasoning benchmarks using models from the Qwen3 family (Yang et al., 2025), and it consistently outperforms existing OPSD approaches with an average gain of up to +5.9 over the base model.

Our contributions are summarized as follows:

• We quantify the teacher preference across tokens via a teacher-intervention pilot study, validating that preferred teachers can differ from token to token within the same response. We further reveal that the token-level preferred teacher can be identified through majority voting and KL-based selection.

• We propose MTA-OPSD, the first token-level adaptive OPSD method. By constructing a multi-teacher pool as the supervision source, our approach enables each token to adaptively select its preferred teacher under the guidance of majority voting and KL divergence, effectively addressing the student-teacher mismatch.

• We conduct extensive experiments across various models on mathematical reasoning tasks, demonstrating that our approach achieves superior reasoning performance compared to existing OPSD methods. Furthermore, we empirically analyze comprehensive ablation studies, paving the way for future advances in this field.

## 2 RELATED WORK

## 2.1 ON-POLICY DISTILLATION AND ON-POLICY SELF-DISTILLATION

Knowledge distillation (Hinton et al., 2015) transfers knowledge from a large teacher model to a student model by training on the teacher’s soft output distributions. Early approaches (Kim & Rush, 2016; Jiao et al., 2020; Wang et al., 2020) leveraged it to sequence-level distillation, establishing the dominant off-policy distillation approaches. (Gu et al., 2024; Agarwal et al., 2024; Lu & Lab, 2025; Yang et al., 2026b) extended it to On-policy Distillation (OPD), addressing the exposure bias (Bengio et al., 2015) mismatch by shifting the training distribution to the student’s own generations.

Despite the rapid progress of OPD, the paradigm relies on a separate, typically stronger teacher model, which may be unavailable or expensive to maintain. To address this challenge, OPSD (Zhao et al., 2026; Hubotter et al., 2026; Shenfeld et al., 2026; Luo et al., 2026; Yang et al., 2026a) removes¨ this requirement by instantiating the teacher and student from the same model. Under this setting, the teacher and student share the same model, but the teacher is granted access to a privileged reference unavailable to the student.

However, recent studies (Kim et al., 2026; Wang et al., 2026b; Kaur et al., 2026; Harne et al., 2026; Shen et al., 2026) reveal a persistent mismatch in OPSD. In particular, a highly informative reference can induce an overly sharp teacher distribution and create a mismatch between the teacher’s supervision and the student’s learning dynamics. These observations motivate the line of research for adaptive OPSD methods.

## 2.2 ADAPTIVE ON-POLICY SELF-DISTILLATION

Adaptive OPSD seeks to align the teacher supervision with the student’s learning dynamics. Rather than exposing the complete privileged reference, existing methods (Tan & Hong, 2026; Han et al., 2026b; Yang et al., 2026c) control its content or exposure ratio using mechanisms such as learned reference-exposure schedules. By varying the amount of privileged references available to the teacher, these methods aim to avoid overly sharp supervision while retaining the benefits of dense learning signals.

However, these methods adapt at the response level, where all tokens in the response share the same teacher and the same privileged reference. In contrast, we identify that each token needs its own teacher as tokens have distinct teacher preferences (Section 4), and further propose our token-level multi-teacher adaptive OPSD method.

## 3 PRELIMINARIES

OPSD is a special case of OPD. OPD transfers knowledge from a stronger teacher model $p _ { T }$ to a weaker student model $p _ { \theta }$ by enforcing dense supervision over trajectories sampled by the student. Given a prompt $x ,$ the student samples a response $y = \{ y ^ { 1 } , y ^ { 2 } , \cdot \cdot \cdot , y ^ { L } \}$ , where $\dot { L }$ denotes the generation length and $y ^ { i }$ refers to the i-th token in the response. The learning objective is to minimize the token-level KL between the teacher’s and the student’s next-token distributions:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x } [ \sum _ { i = 1 } ^ { L } \mathcal { D } _ { \mathrm { K L } } ( p _ { \theta } ( \cdot  { | \~ \boldsymbol { y } ^ { < i } , \boldsymbol { x } ) }  { | ~ | } p _ { T } ( \cdot  { | ~ \boldsymbol { y } ^ { < i } , \boldsymbol { x } ) ) ] } } \\ { \displaystyle = \mathbb { E } _ { x } [ \sum _ { i = 1 } ^ { L } \mathcal { D } _ { \mathrm { K L } } ( p _ { \theta } ( \cdot  { | ~ \boldsymbol { z } _ { i } ) }  { | | } p _ { T } ( \cdot  { | ~ \boldsymbol { c } _ { i } ) } ) ] } \end{array}\tag{1}
$$

where $p ( \cdot \mid y ^ { < i } , x )$ denotes the distribution over the next token $y ^ { i }$ , and we let $c _ { i }$ refer to $( y ^ { < i } , x )$ for simplicity. While forward KL and reverse KL are most used, other distribution divergence measures like generalized Jensen-Shannon divergence (Agarwal et al., 2024) can also be employed.

Recent advances have extended OPD to OPSD, where the student and teacher are instantiated from the same model $p _ { \theta }$ . The difference lies entirely in their conditioning contexts. Privileged reference $r ,$ such as reference solutions (Zhao et al., 2026) or environment feedback (Hubotter et al., 2026), is¨ appended to the original prompt to construct a teacher-specific prompt. Thus, the teacher distribution is:

$$
p _ { T } ( \cdot \mid c _ { i } ) = p _ { \theta } ( \cdot \mid c _ { i } , r ) .\tag{2}
$$

Consequently, the learning objective in Equation (1) adapts into the following:

$$
\mathcal { L } _ { \mathrm { O P S D } } ( \theta ) = \mathbb { E } _ { x } [ \sum _ { i = 1 } ^ { L } \mathcal { D } _ { \mathrm { K L } } ( p _ { \theta } ( \cdot  { | \mathbf { \sigma } | } \ c _ { i } )   | \mathbf { \sigma } p _ { \theta } ( \cdot  { | \mathbf { \sigma } | } \ c _ { i } , r ) ) ] .\tag{3}
$$

In this setting, both the teacher and student share the same model but differ only in the conditioning contexts, and the response is solely generated from the student.

## 4 PILOT STUDY: TOKEN MATTERS

Before diving into a detailed design, a critical question must first be answered:

Given $c _ { i } = ( y ^ { < i } , x )$ , how can we determine ifone teacher $p _ { T } ( c _ { i } )$ is more suitable than another?

A direct approach would be to train on every token separately with specific teachers, covering enough token-teacher combinations. However, it is obviously not practical.

We instead answer this question from another view. Motivated by (Xu et al., 2025; 2026a), we estimate teacher suitability through teacher intervention. Specifically, given a prompt x and a student’s response $\overline { { y } } = \left\{ y ^ { 1 } , y ^ { 2 } , \cdot \cdot \cdot , y ^ { L } \right\}$ , we select a to-

![](images/f182ab39bcc7d56885ca439096dc8c5046103f0d1678ae2d0c0b30e7aa152622.jpg)  
Figure 2: Illustration of the teacher-intervention pilot study.

ken position $1 < i < L$ and preserve the student-generated prefix before that position. Starting from position i, a candidate teacher takes over the generation process and produces a short subsequence:

$$
\begin{array} { r l } & { z _ { 1 : i - 1 } = y _ { 1 : i - 1 } = p _ { \theta } ( x ) } \\ & { z _ { t : t + k } = p _ { T } ( y ^ { < i } , x ) = p _ { \theta } ( c _ { i } , r ) , } \end{array}\tag{4}
$$

where k denotes the teacher-intervention length. After this intervention, control is returned to the student, which completes the remainder of the response:

$$
z _ { i + k + 1 : L } = p _ { \theta } ( z ^ { < i + k + 1 } , x ) .\tag{5}
$$

Thus, we obtain a “student-teacher-student” relay response $z = \{ z _ { 1 } , z _ { 2 } , \cdot \cdot \cdot , z _ { L } \}$ . We compare its accuracy with the pure student response, which serves as a proxy for the candidate teacher preference at the selected token position. A larger gain indicates a more suitable teacher, whereas a smaller gain indicates the opposite.

The main purpose of this pilot study is that, by varying both the intervention token position and the candidate teacher, this study separates variation along the token and teacher axes. This allows us to examine whether teacher preferences change across token positions and to identify their patterns. Interestingly, the pattern is simple but clear. We next describe the experimental setup and then present our findings, as highlighted in Section 4.2.

## 4.1 SETUP

Our study varies two principal factors: tokens at which the intervention begins, and teachers which are used for the intervention.

For the token axis, we consider both regular-spaced and statistic-extreme token positions. For the former, we select tokens located at 25%, 50%, and 75% of the original student rollout. For the latter, we select tokens associated with the highest and lowest student entropy, and the largest and smallest student-teacher KL distance <sup>0</sup>. In total, we select 7 intervention positions from each rollout.

For the teacher axis, we vary both the reference-truncation strategy<sup>1</sup> and the reference-exposure ratio. We consider prefix, suffix, and random-span for truncation strategy, each with exposure ratios of 0.5 and 0.75, together with a full-reference teacher. This produces 7 candidate teachers.

We evaluate intervention lengths across $k = 6 4 , 1 2 8 ,$ and 256. We sample 64 prompts of at least medium difficulty<sup>2</sup> from DeepMath-103k (He et al., 2026). We conduct the study using Qwen3-1.7B and Qwen3-4B as student models. To reduce sampling variance, for every experimental configuration, we sample 2 teacher continuations and then 2 student relays following each continuation. In total, $6 4 * 7 * 7 * 3 * 2 * 2 * 2 = 7 5 2 6 4$ generations are evaluated.

## 4.2 RESULTS AND FINDINGS

Interestingly, the results show clear and meaningful findings.

Findings1: Teacher preference varies across tokens.   
Findings2: Majority voting provides a subset of reliable teachers.   
Findings3: The preferred teacher always exhibits a large KL distance from the student.

Teacher preference varies by tokens. We define the preferred teacher at a token position as the candidate that produces the largest intervention gain. To determine whether this preference changes within a response, we measure how often the preferred teacher is shared by different regular-spaced tokens in the same

Table 1: Overlap of preferred teachers between tokens within the same rollout.
<table><tr><td>Token Position</td><td>0.25</td><td>0.50</td><td>0.75</td></tr><tr><td>0.25</td><td>1</td><td>48.44%</td><td>50.00%</td></tr><tr><td>0.50</td><td>48.44%</td><td></td><td>51.56%</td></tr><tr><td>0.75</td><td>50.00%</td><td>51.56%</td><td>-</td></tr></table>

rollout. As shown in Table 1, for Qwen3-1.7B with $k = 2 5 6 .$ , the overlap is only approximately 50%, which indicates that the teacher preference differs by tokens. These results directly support our claim: every token needs its own teacher.

Majority voting identifies reliable teachers. By comparing the teacher distribution with the student one over a student-generated token $y ^ { i } .$ , we can obtain the advantage signal:

$$
a _ { T } ( y ^ { i } \mid c _ { i } ) = \log \left( p _ { T } \left( y ^ { i } \mid c _ { i } \right) \right) - \log \left( p _ { \theta } \left( y ^ { i } \mid c _ { i } \right) \right) = \log \frac { p _ { T } ( y ^ { i } \mid c _ { i } ) } { p _ { \theta } ( y ^ { i } \mid c _ { i } ) } .\tag{6}
$$

$a _ { T } ( \cdot \ | \ c _ { i } )$ indicates the teacher’s relative support for the student’s choice $y ^ { i } .$ . Specifically, If $a _ { T } ( \cdot \mid c _ { i } ) > 0 .$ the teacher favors the sampled token; otherwise, it indicates that the student’s choice is relatively opposed by the teacher. Thus, at each token position, we can divide the candidate teachers into two groups based on this sign, and obtain the majority subset.

Table 2: Fraction of preferred teachers contained in the majority subset.

We then examine whether the teacher producing the largest gain belongs to

<table><tr><td>Model \Token Position</td><td>Regular</td><td>Extreme</td></tr><tr><td>Qwen3-1.7B</td><td>77.94%</td><td>88.09%</td></tr><tr><td>Qwen3-4B</td><td>87.11%</td><td>90.68%</td></tr></table>

Table 3: Average gains of teachers in the majority and minority subsets.
<table><tr><td>Group</td><td>Majority</td><td>Minority</td><td>Overall</td></tr><tr><td>Gain</td><td>+3.34%</td><td>+1.88%</td><td>+2.34%</td></tr></table>

the majority subset. As shown in Table 2, in around 80%−90% of cases it falls within this subset. We further compare the average gain of teachers in both the majority and minority subsets. For Qwen3- 4B with $k = 2 5 6$ , as shown in Table 3, the majority subset always achieves larger gains per teacher. These results indicate that majority voting provides a reliable first-stage filter for teacher selection.

Large KL distinguishes the preferred teacher. We also found an interesting correlation between the teacher preference and the studentteacher KL distance. For each token, we select the teachers with the largest and smallest student-teacher KL distance from the majority subset and compare their gains. As shown in Ta-

Table 4: Average gains from the farthest and nearest teachers.
<table><tr><td>Model \Teacher</td><td>Farthest</td><td>Nearest</td><td> $\Delta$ </td></tr><tr><td>Qwen3-1.7B</td><td>+2.42%</td><td>+0.59%</td><td>+1.83%</td></tr><tr><td>Qwen3-4B</td><td>+2.78%</td><td>+1.21%</td><td>+1.57%</td></tr></table>

ble $^ { 4 , }$ in regular token positions, the farthest teacher consistently produces greater average gains. This observation suggests that, student-teacher KL distance provides a useful criterion for teacher preference, for each token.

## 5 METHOD

Based on the findings in Section 4.2, we propose Multi-Teacher Adaptive OPSD (MTA-OPSD), a simple yet effective framework for token-level adaptive OPSD. As illustrated in Figure 3, rather than using a single teacher, MTA-OPSD constructs a multi-teacher pool consisting of candidate teachers conditioned on different variants of the privileged reference. Through majority voting followed by KL-based selection, it assigns each token its preferred teacher. This ensures that different tokens within the same response receive supervision from their respective teachers.

## 5.1 MULTI-TEACHER POOL

To support token-level adaptive teacher selection, we construct a multi-teacher pool M, consisting of all candidate teachers available for each token:

$$
M = \{ p _ { T _ { 1 } } , p _ { T _ { 2 } } , \cdot \cdot \cdot , p _ { T _ { K } } \} ,\tag{7}
$$

where K is the number of candidate teachers and $p _ { T _ { k } }$ denotes the k-th teacher. All teachers share the same model $p _ { \theta }$ with the student. They differ from each other only in the privileged references on which they are conditioned:

$$
R = \{ r _ { 1 } , r _ { 2 } , \cdot \cdot \cdot , r _ { K } \} ,\tag{8}
$$

which leads to the next-token distribution of each teacher as:

$$
p _ { T _ { k } } = p _ { \theta } ( \cdot \mid x , r _ { k } ) .\tag{9}
$$

Each $r _ { k }$ differs from each other by its truncation way and exposure ratio of the full reference. An example of our multi-teacher pool is provided in Section F.

While the multi-teacher pool supplies a diverse set of supervision signals, it does not determine which teacher should supervise each token. Guided by the pilot study in Section 4, MTA-OPSD performs token-level adaptive teacher assignment in two steps: (1) majority voting identifies a reliable subset of teachers, and (2) KL-based selection chooses the preferred teacher from that subset.

## 5.2 MAJORITY VOTING

As shown in Section 4.2, majority voting provides a reliable teacher subset by identifying the direction of the advantage signals. We therefore use majority voting as the first step of teacher selection.

Specifically, for a token $y ^ { i }$ in the student response, the multi-teacher pool produces the following set of its teacher token distributions:

$$
\begin{array} { r l } & { P _ { M } ( c _ { i } ) = \left\{ p _ { T _ { 1 } } ( \cdot \mid c _ { i } ) , p _ { T _ { 2 } } ( \cdot \mid c _ { i } ) , \cdot \cdot \cdot , p _ { T _ { K } } ( \cdot \mid c _ { i } ) \right\} } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x } = \left\{ p _ { \theta } ( \cdot \mid c _ { i } , r _ { 1 } ) , p _ { \theta } ( \cdot \mid c _ { i } , r _ { 2 } ) , \cdot \cdot \cdot , p _ { \theta } ( \cdot \mid c _ { i } , r _ { K } ) \right\} . } \end{array}\tag{10}
$$

![](images/a1cd83e84c20ea89409f7d41289f1f7e550deaa37921896c2065559fd3ff884e.jpg)  
Figure 3: The framework of our approach, MTA-OPSD. With the multi-teacher pool, it adaptively assigns each token its preferred teacher, by majority voting followed by KL-based selection.

For each teacher token distribution in $P _ { M }$ , we compare it with the student token distribution $p _ { \theta } ( \cdot \mid$ $y ^ { < i } , x )$ over $y ^ { i }$ , to obtain the following set:

$$
A _ { M } ( c _ { i } ) = \{ a _ { T _ { 1 } } ( y ^ { i } \mid c _ { i } ) , a _ { T _ { 2 } } ( y ^ { i } \mid c _ { i } ) , \cdot \cdot \cdot , a _ { T _ { K } } ( y ^ { i } \mid c _ { i } ) \} ,\tag{11}
$$

where advantage $a _ { T _ { k } } ( y ^ { i } \mid c _ { i } )$ is computed using Equation (6). The sign of $a _ { T _ { k } } \left( y ^ { i } \mid c _ { i } \right)$ indicates the k-th teacher’s relative support for the student’s choice at $y ^ { i }$ . We denote it as the supporting binary vote:

$$
\begin{array} { r } { { v _ { T _ { k } } } ( c _ { i } ) = \mathrm { s g n } \left( { a _ { T _ { k } } } \left( { y ^ { i } } \mid { c _ { i } } \right) \right) = \left\{ \begin{array} { l l } { + 1 , } & { { a _ { T _ { k } } } \left( { y ^ { i } } \mid { c _ { i } } \right) > 0 , } \\ { - 1 , } & { { a _ { T _ { k } } } \left( { y ^ { i } } \mid { c _ { i } } \right) < 0 . } \end{array} \right. } \end{array}\tag{12}
$$

We then determine the majority voting direction at token position i:

$$
V _ { M } ( c _ { i } ) = \mathrm { s g n } \left( \sum _ { k = 1 } ^ { K } v _ { T _ { k } } \left( c _ { i } \right) \right) ,\tag{13}
$$

and the corresponding token-level majority teacher subset<sup>3</sup>:

$$
M _ { \mathrm { m a j } } ( c _ { i } ) = \left\{ p _ { T _ { k } } \in M \ | \ v _ { T _ { k } } \left( c _ { i } \right) = V _ { M } \left( c _ { i } \right) \right\} .\tag{14}
$$

Note that this procedure is token-level, performed independently at every token position.

## 5.3 KL-BASED SELECTION

Majority voting filters out teachers whose supervision direction disagrees with most candidates, but it does not distinguish among teachers within the majority subset. As shown in Section 4.2, the preferred teacher always has large KL distance from the student. We therefore use KL distance as the second-stage selection criterion.

Specifically, for token $y ^ { i }$ in the student response and its majority teacher subset $M _ { \mathrm { m a j } } ( c _ { i } )$ , the tokenlevel preferred teacher is defined as:

$$
p _ { T } ^ { * } ( c _ { i } ) = p _ { \theta } ( \cdot \mid c _ { i } , r ^ { * } ) = \arg \operatorname* { m a x } _ { p _ { T _ { k } } \in M _ { \operatorname* { m a j } } ( c _ { i } ) } { \cal D } _ { \mathrm { K L } } \left( p _ { \theta } \left( \cdot \mid c _ { i } \right) \mid \mid p _ { T _ { k } } \left( \cdot \mid c _ { i } \right) \right) .\tag{15}
$$

Note that since the preferred teacher is determined separately for every token, it can vary across tokens within the same response. Consequently, substituting this token-level teacher into the learning objective in Equation (3) gives:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { M T A - O P S D } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { x } } \left[ \displaystyle \sum _ { i = 1 } ^ { L } \mathcal { D } _ { \mathrm { K L } } \left( p _ { \boldsymbol { \theta } } \left( \cdot \mid c _ { i } \right) \mid \mid p _ { T } ^ { * } \left( \cdot \mid c _ { i } \right) \right) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { \boldsymbol { x } } \left[ \displaystyle \sum _ { i = 1 } ^ { L } \mathcal { D } _ { \mathrm { K L } } \left( p _ { \boldsymbol { \theta } } \left( \cdot \mid c _ { i } \right) \mid \mid p _ { \boldsymbol { \theta } } \left( \cdot \mid c _ { i } , r ^ { * } \right) \right) \right] , } \end{array}\tag{16}
$$

Thus, MTA-OPSD assigns each token its own teacher.

## 5.4 DISCUSSION

The mechanism of our method can be understood as decoupling the reliability of the supervision direction from the informativeness of the supervision signal. Majority voting first determines whether the teacher pool collectively supports or opposes the student-generated token $y ^ { i }$ , KL-based selection then identifies the most informative teacher aligned with that direction. This ordering is crucial: KL distance is not used as a standalone measure of teacher quality, but only after directional consensus has filtered out potentially unreliable teachers.

Specifically, when $V _ { M } ( c _ { i } ) = 1$ , the majority teachers in $M _ { \mathrm { m a j } } ( c _ { i } )$ assign a higher probability to $y ^ { i }$ than the student itself, and selecting the teacher with the largest KL therefore reinforces the student’s current choice. Meanwhile, the selected teacher can also provide a substantially different overall probability distribution. Consequently, the supervision can strengthen the sampled token without being limited to simply imitating it, thereby retaining informative probabilities for other plausible predictions and potentially reducing overfitting to the student’s current trajectory.

When $V _ { M } ( c _ { i } ) = - 1$ , the majority teachers assign a lower probability to $y ^ { i }$ , and the sampled token is collectively regarded as a less preferred choice. In this case, a teacher with a small KL distance may only weakly downweight this token or remain too close to the student distribution. In contrast, the farthest teacher is more likely to redistribute probability mass toward different candidate tokens, providing a stronger corrective signal together with more distinct alternative prediction directions. Such supervision may therefore help the student move away from a potentially erroneous local choice rather than merely making a marginal adjustment to it.

## 6 EXPERIMENTS

## 6.1 EXPERIMENT SETUP

Additional experiment setup details are provided in Section C.

Models and Datasets. We evaluate models from the Qwen3 family (Yang et al., 2025) at three scales: Qwen3-1.7B, Qwen3-4B, and Qwen3-8B, all with thinking mode enabled during both training and validation. We conduct experiments across mathematical reasoning tasks. The training data is drawn from DeepMath-103K (He et al., 2026), and we evaluate on AMC23 (MAA, 2023), AIME24 (Zhang & Math-AI, 2024), and AIME25 (Zhang & Math-AI, 2025).

Baselines. We compare our method against two fixed-teacher OPSD methods and one adaptive OPSD method: (1) OPSD (Zhao et al., 2026), which performs dense-logit distillation using forward KL with token-wise pointwise KL clipping; (2) SDPO (Hubotter et al., 2026), which performs¨ dense-logit distillation using Jensen–Shannon divergence; and (3) Paint (Tan & Hong, 2026), which adaptively determines the reference-exposure ratio according to the student-teacher gap.

Implementation Details. All baselines and our method use dataset-provided ground-truth reference solutions as the privileged information, and all contexts include both the reasoning trace and the final answer. We use identical training and evaluation settings across all methods, with maximum generation lengths of 4096 tokens during training and 38912 tokens during validation. The multi-teacher pool contains seven teacher configurations: {full reference, prefix-0.5, prefix-0.75, suffix-0.5, suffix-0.75, random-span-0.5, random-span-0.75}<sup>4</sup>. We train each method for 300 steps, evaluate checkpoints every 50 steps on the test set, and report its best Avg@12 results.

## 6.2 MAIN RESULTS

Table 5 presents a comprehensive reasoning performance comparison between existing OPSD methods and our approach. For average performance, MTA-OPSD achieves the highest average performance at every model scale, yielding an average improvement of +5.9 for Qwen3-1.7B, +1.5 for Qwen3-4B, and +2.8 for Qwen3-8B, over the base model. These results outperform existing OPSD methods. For individual benchmarks, MTA-OPSD also achieves or matches the best result on nearly every model-benchmark combination. Note that the improvements are particularly consistent and strong on the more challenging AIME25 benchmark. These results highlight the importance of token-level adaptive teacher assignments. A qualitative comparison is provided in Section D.

Table 5: Mathematical reasoning performance across model scales. MTA-OPSD achieves the best average performance at all three model scales.
<table><tr><td>Model</td><td>Method</td><td>AMC23</td><td>AIME24</td><td>AIME25</td><td>Average</td></tr><tr><td rowspan="5">Qwen3-1.7B</td><td>Base Model</td><td>74.1</td><td>45.6</td><td>33.6</td><td>51.1</td></tr><tr><td>OPSD</td><td>75.3</td><td>49.2</td><td>37.5</td><td>54.0</td></tr><tr><td>SDPO</td><td>73.5</td><td>43.3</td><td>31.4</td><td>49.4</td></tr><tr><td>Paint</td><td>76.0</td><td>47.5</td><td>38.6</td><td>54.0</td></tr><tr><td>MTA-OPSD</td><td>78.5</td><td>51.4</td><td>41.1</td><td>57.0+5.9</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Base Model</td><td>88.6</td><td>73.1</td><td>65.3</td><td>75.7</td></tr><tr><td>OPSD</td><td>89.1</td><td>74.4</td><td>65.6</td><td>76.4</td></tr><tr><td>SDPO</td><td>87.7</td><td>71.4</td><td>61.7</td><td>73.6</td></tr><tr><td>Paint</td><td>89.1</td><td>71.7</td><td>66.1</td><td>75.6</td></tr><tr><td>MTA-OPSD</td><td>89.0</td><td>74.4</td><td>68.1</td><td>77.2+1.5</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>Base Model</td><td>89.1</td><td>73.1</td><td>66.9</td><td>76.4</td></tr><tr><td>OPSD</td><td>89.9</td><td>75.0</td><td>67.8</td><td>77.6</td></tr><tr><td>SDPO</td><td>88.9</td><td>73.9</td><td>65.6</td><td>76.1</td></tr><tr><td>Paint</td><td>90.0</td><td>76.1</td><td>68.3</td><td>78.1</td></tr><tr><td>MTA-OPSD</td><td>89.8</td><td>77.5</td><td>70.3</td><td>79.2+2.8</td></tr></table>

## 6.3 ABLATION STUDIES

All ablation studies use Qwen3-1.7B as the base model and follow the same experiment settings.

Effect of the Multi-Teacher Pool. We first investigate whether the multi-teacher pool provides more effective supervision than any individual teacher. In Table 6, we compare our method with training using each teacher configuration from the pool independently. Although the effectiveness of individual teachers varies from each other, MTA-OPSD achieves the highest score on both average and individual benchmarks. These results suggest that the multiteacher pool offers diverse, sufficient, and complementary learning signals.

Effect of Token-Level Teacher Assignment. The multi-teacher pool is accompanied by the token-level adaptive teacher assignment. The previous ablation establishes the value of the former, but does not isolate the contribution of the latter.

Table 6: Comparison of MTA-OPSD with single-teacher variants.
<table><tr><td>Teacher</td><td>AMC23</td><td>AIME24</td><td>AIME25</td><td>Average</td></tr><tr><td>Base Model</td><td>74.1</td><td>45.6</td><td>33.6</td><td>51.1</td></tr><tr><td>Full (OPSD)</td><td>75.3</td><td>49.2</td><td>37.5</td><td>54.0</td></tr><tr><td>Prefix-0.5</td><td>76.8</td><td>50.8</td><td>40.0</td><td>55.9</td></tr><tr><td>Prefix-0.75</td><td>78.2</td><td>49.4</td><td>41.1</td><td>56.2</td></tr><tr><td>Suffix-0.5</td><td>75.9</td><td>48.6</td><td>34.7</td><td>53.1</td></tr><tr><td>Suffix-0.75</td><td>75.0</td><td>50.3</td><td>40.0</td><td>55.1</td></tr><tr><td>Random-span-0.5</td><td>76.8</td><td>49.4</td><td>36.9</td><td>54.4</td></tr><tr><td>Random-span-0.75</td><td>76.4</td><td>50.6</td><td>35.8</td><td>54.3</td></tr><tr><td>MTA-OPSD</td><td>78.5</td><td>51.4</td><td>41.1</td><td>57.0</td></tr></table>

Table 7: Comparison of token-level teacher-assignment strategies using the same multi-teacher pool.
<table><tr><td>Strategy</td><td>AMC23</td><td>AIME24</td><td>AIME25</td><td>Average</td></tr><tr><td>Base Model</td><td>74.1</td><td>45.6</td><td>33.6</td><td>51.1</td></tr><tr><td>Random-pick</td><td>76.1</td><td>49.2</td><td>35.5</td><td>53.6</td></tr><tr><td>Mean-loss</td><td>75.7</td><td>48.9</td><td>36.4</td><td>53.7</td></tr><tr><td>MTA-OPSD</td><td>78.5</td><td>51.4</td><td>41.1</td><td>57.0</td></tr></table>

Table 8: Ablation of majority voting and KL-based teacher selection.
<table><tr><td>Majority Voting</td><td>KL Selection</td><td>AMC23</td><td>AIME24</td><td>AIME25</td><td>Average</td></tr><tr><td>No</td><td>Farthest</td><td>75.8</td><td>48.9</td><td>33.9</td><td>52.9</td></tr><tr><td rowspan="3">Yes</td><td>Farthest</td><td>78.5</td><td>51.4</td><td>41.1</td><td>57.0</td></tr><tr><td>Random</td><td>78.0</td><td>51.1</td><td>40.8</td><td>56.6</td></tr><tr><td>Nearest</td><td>75.7</td><td>48.6</td><td>35.6</td><td>53.3</td></tr></table>

We therefore retain the same multi-teacher pool and compare MTA-OPSD with two alternative

assignment strategies: (1) Random teacher selection from the pool for each token, and (2) averaging the distillation losses from all teachers. As shown in Table 7, MTA-OPSD consistently outperforms both alternatives on every benchmark. These show that the gains do not arise solely from multiple teachers; selecting a suitable teacher for each token is critical to using the teacher pool effectively.

Teacher-Assignment Dynamics. We also investigate how frequently each teacher is selected throughout training. The assignments are distributed relatively evenly across the teacher pool, indicating that the selection mechanism does not collapse to a single dominant teacher. We also observe that, under the same truncation strategy, a higher reference-exposure ratio does not necessarily produce a larger student-teacher KL distance, as discussed in Section E. This observation further suggests that existing adaptive methods of adjusting reference exposure alone cannot determine teacher suitability.

Effect of Majority Voting and KL-Based Selection. MTA-OPSD selects a teacher in two steps: majority voting and KL-based selection. Table 8 presents the contribution of each step, where the second row is our default setting. MTA-OPSD consistently outperforms all variants, further validating the effectiveness of our design: majority voting first identifies a reliable teacher subset, and KL-based selection chooses a teacher from that subset.

## 7 CONCLUSION

We present MTA-OPSD, the first token-level adaptive OPSD method. A teacher-intervention study shows that teacher preferences vary across tokens and motivates our two-step token-level adaptive teacher selection strategy: majority voting with KL-based selection. Future work may explore selfevolving teacher pools and more stable multi-teacher training.

## ACKNOWLEDGMENTS

This paper is supported by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Research, Technology and Space.

Qi Chen is supported by research funding from the ELLIS Institute Finland and the Department of Computer Science at Aalto University.

We acknowledge CSC-IT Center for Science, Finland, for providing access to the supercomputer LUMI, owned by the European High Performance Computing Joint Undertaking (EuroHPC JU) and hosted by CSC Finland in collaboration with the LUMI consortium. We also acknowledge the computational resources provided by the Aalto Science-IT project through the Triton cluster.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024.

Mohammadreza Armandpour, Fatih Ilhan, David Harrison, Ajay Jaiswal, Duc NM Hoang, Fartash Faghri, Yizhe Zhang, Minsik Cho, and Mehrdad Farajtabar. Unmasking on-policy distillation: Where it helps, where it hurts, and why. arXiv preprint arXiv:2605.10889, 2026.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Fangming Cui, Sunan Li, and Jiahong Li. A brief overview: On-policy self-distillation in large language models. arXiv preprint arXiv:2605.18141, 2026.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In The twelfth international conference on learning representations, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Zhuowen Han, Jinwei Xiao, Zhengxi Lu, Renren Jin, Zhiyuan Yao, Yuxin Liu, Hongyan Hao, Yueqing Sun, Yu Yang, Qi Gu, et al. Distill where you fail: Recovering learning signals of negative rl-groups from adaptive teacher guidance. arXiv preprint arXiv:2608.00782, 2026a.

Zihao Han, Tiangang Zhang, Huaibin Wang, and Yilun Sun. Adaptive teacher exposure for selfdistillation in llm reasoning. arXiv preprint arXiv:2605.11458, 2026b.

Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, and Jiawei Chen. Rethinking entropy interventions in rlvr: An entropy change perspective. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 31105–31133, 2026.

Sarthak Harne, Chinmay Karkar, Yash Pandya, Ahmed Awadallah, and Akshay Nambi. Privileged, but biased: How pi-conditioned teachers break self-distillation. arXiv preprint arXiv:2608.04794, 2026.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. In International Conference on Learning Representations, volume 2026, pp. 138306–138322, 2026.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding. In Findings of the association for computational linguistics: EMNLP 2020, pp. 4163–4174, 2020.

Simran Kaur, Narutatsu Ri, Yinghui He, Liam Fowl, and Sanjeev Arora. Rethinking on-policy self-distillation for thinking models. arXiv preprint arXiv:2607.05184, 2026.

Junlong Ke, Zichen Wen, Weijia Li, Conghui He, and Linfeng Zhang. Respecting self-uncertainty in on-policy self-distillation for efficient llm reasoning. arXiv preprint arXiv:2605.13255, 2026.

Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472, 2026.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 1317–1327, 2016.

Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. Unifying group-relative and self-distillation policy optimiza tion via sample routing. arXiv preprint arXiv:2604.02288, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Yifu Luo, Zeyu Chen, Haoyu Wang, Xinhao Hu, Yuxuan Zhang, Zhizhou Sha, and Shiwei Liu. Learning from the self-future: On-policy self-distillation for dllms. arXiv preprint arXiv:2606.18195, 2026.

MAA. Amc 2023. https://huggingface.co/datasets/math-ai/amc23, 2023. Dataset compiled from the 2023 American Mathematics Competitions.

Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, and Lijie Wen. Rlcsd: Reinforcement learning with contrastive on-policy self-distillation. arXiv preprint arXiv:2606.11709, 2026.

Zhanming Shen, Jintao Tong, Shaotian Yan, Chen Shen, Hao Chen, Wentao Ye, Xiaomeng Hu, Rui Miao, Haobo Wang, Junbo Zhao, et al. Purified opsd: On-policy self-distillation without losing how to think. arXiv preprint arXiv:2607.02234, 2026.

Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-distillation enables con-¨ tinual learning. arXiv preprint arXiv:2601.19897, 2026.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Zhiquan Tan and Yinrong Hong. Paint: Partial-solution adaptive interpolated training for selfdistilled reasoners. arXiv preprint arXiv:2604.26573, 2026.

Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, and Lan-Zhe Guo. Trace: Distilling where it matters via token-routed self on-policy alignment. arXiv preprint arXiv:2605.10194, 2026a.

Meng Wang, Haohan Zhao, Wenzhuo Liu, Lu Yang, Geng Liu, Haiyang Guo, Guo-Sen Xie, Gaofeng Meng, Hongbin Liu, and Fei Zhu. Denser ̸= better: Limits of on-policy self-distillation for continual post-training. arXiv preprint arXiv:2607.01763, 2026b.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. Advances in Neural Information Processing Systems, 38:115452–115486, 2026c.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33:5776–5788, 2020.

Zhenlin Wei, Pu Jian, Yingzhuo Deng, Xiaohan Wang, Jiajun Chai, Zhexin Hu, Wei Lin, Shanbin Zhang, and Guojun Yin. Amr-sd: Asymmetric meta-reflective self-distillation for token-level credit assignment. arXiv preprint arXiv:2605.18529, 2026.

Haolei Xu, Xiaowen Xu, Haiwen Hong, Zixuan Ni, Hongxing Li, Yiwen Qiu, Weiming Lu, and Yongliang Shen. Pass the baton: Trajectory-relayed on-policy distillation. arXiv preprint arXiv:2607.26057, 2026a.

Wenda Xu, Rujun Han, Zifeng Wang, Long Le, Dhruv Madeka, Lei Li, William Wang, Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative knowledge distillation: Bridging the teacher-student gap through interleaved sampling. In International Conference on Learning Representations, volume 2025, pp. 64616–64646, 2025.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. Tip: Token importance in on-policy distillation. arXiv preprint arXiv:2604.14084, 2026b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026a.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026b.

Yongkang Yang, Zhezheng Hao, Hong Zhang, Yi Liu, Xiankun Lin, Wence Ji, Fanjunduo Wei, Jiarui Yu, Qiang Lin, Xiaoyun Liang, et al. Matching supervision to the student’s learning capacity: A unified framework for on-policy self-distillation. arXiv preprint arXiv:2608.08176, 2026c.

Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, and Yizhe Zhang. Embarrassingly simple self-distillation improves code generation. arXiv preprint arXiv:2604.01193, 2026.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

## A VISUALIZATION DETAILS

The three teachers in Figure 1 use the configurations of prefix-0.5, suffix-0.5, and random-span-0.5.   
The entire student response is shown in Figure 4.

![](images/5a6fb4f05f2a5ac8aca85246dd7fc32df63468ad65b4762e2f7faa21ecf3efac.jpg)  
Figure 4: Entire student response for Figure 1.

## B ADDITIONAL MAJORITY VOTING DETAILS

As discussed in Section 5.2, a tie may occur when the multi-teacher pool contains an even number of teachers, with equal numbers of positive and negative votes. In this case, we break the tie according to the magnitudes of the corresponding advantage signals. Specifically, the majority direction is defined as

$$
V _ { M } ( c _ { i } ) = \left\{ \begin{array} { l l } { \mathrm { s g n } \left( \displaystyle \sum _ { k = 1 } ^ { K } v _ { T _ { k } } ( c _ { i } ) \right) , \mathrm { ~ i f ~ } \displaystyle \sum _ { k = 1 } ^ { K } v _ { T _ { k } } ( c _ { i } ) \neq 0 , } \\ { \mathrm { s g n } \left( \displaystyle \sum _ { k = 1 } ^ { K } a _ { T _ { k } } ( y ^ { i } \mid c _ { i } ) \right) , \mathrm { ~ o t h e r w i s e . } } \end{array} \right.\tag{17}
$$

Thus, when the numbers of positive and negative votes are equal, we select the group with the larger total absolute advantage. The majority teacher subset is then still constructed using Equation (14). This tie-breaking rule retains the direction supported by the stronger aggregate teacher signal.

## C ADDITIONAL EXPERIMENT SETTINGS

All experiments are conducted on 4 H200 GPUs. We use a learning rate of $1 \times 1 0 ^ { - 6 }$ . All methods are implemented on top of the veRL (Sheng et al., 2025) PPO trainer with FSDP2 actor sharding and vLLM rollouts. All runs use full-parameter training (no LoRA (Hu et al., 2021)). The global batch size is 32. Following (Zhao et al., 2026), we restrict the token-level distillation objective to the top-100 entries of the teacher distribution.

Note that following (Pan et al., 2026), we use training problems at difficulty levels 5–7 for Qwen3- 1.7B, 6–8 for Qwen3-4B, and levels 7–10 for Qwen3-8B.

## D QUALITY VISUALIZATION

We provide a qualitative comparison in Figure 5.

![](images/c8a7d51d9a7f2334e15d13cd44f277e3cdf7a6dbb3d573c63092b0de663b63f7.jpg)  
Figure 5: Quality visualization.

## E ADDITIONAL DETAILS ABOUT KL-DISTANCE

As discussed in Section 6.3, even under same truncation strategy, larger reference-exposure ratio does not lead to larger student-teacher KL distance. In our pilot study Section 4.2, for example, for Qwen3-4B model with suffix truncation, only 49.1% cases that 0.75 exposure ratio has larger KL distance from the student. In contrast, the statistic for 0.5 exposure ratio is 50.9%.

## F MULTI-TEACHER POOL VISUALIZATION

Figures below demonstrate a brief visualization of our multi-teacher pool.

![](images/775e051ee438e4e69559c22f4309cd38fcda51030eea5853d209e510ce5d3e0c.jpg)  
Figure 6: Multi-teacher pool visualization.

![](images/0e607f8a96ed7b311096f81da08a8df2444494ddfd8025db061cb694382665e5.jpg)  
Figure 7: Multi-teacher pool visualization.

![](images/644ce00cc1c4bdcbad18c556de7e6ccde8aff3702832a503fdf6f69900644d5b.jpg)  
Figure 8: Multi-teacher pool visualization.

![](images/2244220523a3b00ffdeca8c29148564d5eae455e65be32e768fa99669e6702cd.jpg)  
Figure 9: Multi-teacher pool visualization.