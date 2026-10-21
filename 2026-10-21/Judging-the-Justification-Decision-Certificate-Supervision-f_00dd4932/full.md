# Judging the Justification: Decision Certificate Supervision for Generative Reward Models

Guiquan Sun<sup>1</sup> Bing Bai<sup>2</sup> Martin Renqiang Min<sup>2</sup> Dongjin Song<sup>1</sup>\*

<sup>1</sup>School of Computing, University of Connecticut

<sup>2</sup>NEC Labs America

{guiquan.sun,dongjin.song}@uconn.edu {bbai,renqiang}@nec-labs.com

## Abstract

Generative reward models (GenRMs) generate natural language rationales to justify preference judgments, yet a correct verdict does not guarantee a sound justification. Existing approaches often rely on reference explanations to supervise rationales, but obtaining such references can be costly and does not necessarily establish whether generated rationales support the resulting verdict. To address these limitations, we propose DECERT (Decision-Certificate Supervision), a framework that evaluates the validity and verdict consistency of generated rationales without requiring reference explanations. Crucially, DECERT translates these assessments into preference supervision that accounts for both verdict correctness and justification quality. In particular, when multiple sampled judgments share the same correct verdict, differences in certificate quality enable DECERT to construct rationale preference pairs that distinguish stronger from weaker justifications. Extensive empirical studies across two model families and multiple in- and out-of-distribution benchmarks demonstrate that DECERT consistently outperforms strong generative judge baselines while improving rationale quality. Moreover, DE-CERT effectively mitigates positional bias, improving position-consistent accuracy by up to 14.8 points. These results highlight the importance of supervising not only which verdict is correct, but also how that verdict is justified. Our code and models are available at https://github.com/UConn-DSIS/DeCert.

## 1 Introduction

In pairwise evaluation, generative reward models (GenRMs) generate natural language rationales before selecting the preferred response, providing explicit justification for their judgments (Mahan et al., 2024). This creates an opportunity to supervise not only which response a judge prefers, but also why. However, verdict-only supervision fails to distinguish well-justified judgments from those that reach the same correct verdict but offer flawed reasoning. Recent findings highlight the importance of this distinction. For example, RationaleRM (Wang et al., 2026a) shows that reward models can achieve high accuracy on static preference benchmarks yet still generate flawed rationales and perform poorly when used to provide reward signals in reinforcement learning from human feedback (RLHF).

Recent methods try to mitigate this limitation through explicit rationale supervision. For example, RationaleRM matches generated reasons to human atomic rationales, R-Align evaluates their logical alignment with human or synthesized reference explanations (Lai et al., 2026), and RM-NLHF extends critique-matching supervision to examples without human explanations through a learned MetaRM (Wang et al., 2026d). Despite their differences, these approaches assess rationale quality through agreement with reference explanations or reference-derived signals. Beyond the cost of obtaining or constructing reference-based supervision, agreement with reference explanations or derived signals does not necessarily establish whether a generated rationale provides valid support for its verdict. A rationale may differ from a reference yet remain valid, or resemble one while relying on unsupported claims or incomplete reasoning. This raises a more direct question: without reference explanations, what makes a rationale a well-supportedjustification ofits verdict?

We formulate each judgment rationale as a decision certificate, i.e., a comparative argument intended to support its preference verdict. Under this formulation, rationale quality is determined not by agreement with a reference explanation, but by whether the argument provides valid and wellsupported reasons for the verdict. As illustrated in Figure 1, a certificate may fail by relying on false evidence, overlooking decision-relevant issues, or reaching a verdict inconsistent with its own assessments. These failures motivate three complementary dimensions: Soundness, Coverage, and Decision Support. Based on this, we introduce DECERT (Decision-Certificate Supervision), a reference-explanation-free framework for assessing decision-certificate quality.

![](images/678a9e70b2c7eb68215ab2e86500a18c8b9b7fd6c464679c432df9ba34bd9c01.jpg)  
Figure 1: Illustrations of risks arising from unreliable judging in GenRMs, where the stated rationale conflicts with the final output. Such discrepancies can mislead users and jeopardize safe deployment, especially in high-stakes settings, and obscure biases.

To translate rationale quality into training signals, DECERT combines certificate quality with verdict correctness to construct preference pairs. Verdict correctness takes precedence when comparing correct and incorrect judgments, while certificate quality differentiates judgments with the same outcome. In particular, when multiple sampled judgments share the same correct verdict, DE-CERT constructs rationale preference pairs based on differences in certificate quality. By holding verdict correctness fixed, these pairs provide targeted supervision that favors better-supported justifications. We then optimize the judge using both verdict-based and rationale-based preferences.

Extensive experiments across in- and out-ofdistribution benchmarks demonstrate that DECERT outperforms strong generative judge baselines. Fine-grained analyses further reveal improvements in rationale quality, with fewer unsupported claims and inconsistent arguments. Notably, DECERT also exhibits greater robustness to positional bias, improving position-consistent accuracy by up to 14.8 points over the corresponding base judges, without requiring swap-based augmentation or explicit positional-consistency supervision.

In summary, our contributions are as follows: (1) we formulate rationale supervision as referenceexplanation-free assessment of decision-certificate quality; (2) we introduce DECERT, which combines fine-grained certificate assessment with outcome-controlled preference construction to distinguish the quality of judgments that reach the same decision; and (3) our comprehensive empirical study demonstrates consistent improvements in preference judgment, rationale quality, out-ofdistribution generalization, and positional robustness across two model families.

## 2 Related Work

Preference Learning and Reward Modeling. Learning from human preferences has been widely studied as a means of aligning model behavior with human judgments (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022). Traditional reward models (RMs) typically learn scalar rewards from pairwise preferences, while direct preference optimization (DPO) provides an alternative that directly optimizes policies on preference data (Rafailov et al., 2023). The reliability of reward models is critical to preference alignment, motivating benchmarks such as RewardBench (Lambert et al., 2024) that examine their ability to distinguish responses across diverse tasks. While preference labels specify which response is preferred, they do not by themselves explain the evidence supporting that preference. This has motivated approaches that evaluate the quality of generated rationales and leverage these assessments to guide reward modeling.

LLM-as-a-Judge and Generative Reward Modeling. LLM-as-a-Judge methods leverage language models to evaluate open-ended responses, often generating natural language feedback or applying task-specific evaluation criteria (Zheng et al., 2023; Liu et al., 2023; Kim et al., 2024a,b; Shen et al., 2026). GenRMs further integrate reasoning and preference prediction, enabling judgments accompanied by explicit rationales (Mahan et al., 2024). Recent approaches improve these capabilities through reasoning distillation, reinforcement learning, or structured evaluation (Chen et al., 2026; Whitehouse et al., 2026; Saha et al., 2025). Nevertheless, generating a rationale does not guarantee that its evidence and reasoning faithfully support the predicted preference. Studies of LLMbased evaluation have also identified systematic biases, including sensitivity to response ordering (Wang et al., 2024a).

Rationale Supervision versus Process Reward Models. Process reward models (PRMs) supervise intermediate reasoning steps in a solver’s solution process to improve reasoning correctness (Lightman et al., 2024; Wang et al., 2024b). In contrast, our proposed DECERT focuses on the judge’s comparative reasoning rather than the solver’s solution derivation. Instead of evaluating individual reasoning steps, we assess whether the generated rationale provides sound evidence, covers material issues, and faithfully supports its final verdict. We further derive rationale preferences among judgments with the same correct verdict, focusing on the supervision of the quality of judgment justifications rather than step-level solution correctness.

Rationale Supervision for Generative Judges. Recent studies have explored supervising judgment rationales through reference explanations or human critiques (Wang et al., 2026a; Lai et al., 2026; Wang et al., 2026e), as well as self-generated critiques, evaluation criteria, and analysis preferences (Yu et al., 2025a; Liang et al., 2025; Wang et al., 2025; Qin et al., 2026). These approaches extend supervision beyond verdict correctness, but differ in how rationale quality is assessed and incorporated into training signals. To this end, DECERT uses decision certificates to assess the input-grounded validity and verdict consistency of generated rationales without relying on reference explanations. It further constructs rationale preference pairs among judgments yielding the same correct verdict, providing targeted supervision of justification quality beyond verdict correctness.

## 3 Decision-Certificate Supervision

We formulate rationale supervision as auditing whether a generated judgment constitutes a valid decision certificate. Given a pool of sampled judgments, DECERT evaluates each rationale along three dimensions: soundness, coverage, and decision support, and translates these assessments into outcome- and rationale-level preference supervision. Figure 2 gives an overview.

## 3.1 Problem Formulation

Pairwise judgment. Let $\boldsymbol { \xi } = ( x , y _ { A } , y _ { B } , \mathcal { P } )$ denote a pairwise evaluation instance, where x is the request, $y _ { A }$ and $y _ { B }$ are candidate responses, and $\mathcal { P }$ is a prespecified evaluation policy. A generative judge $\pi _ { \theta }$ produces

$$
J = ( r , v ) \sim \pi _ { \theta } ( \cdot \mid \xi ) ,
$$

where r is a natural language rationale and $v \in$ $\{ - 1 , + 1 \}$ denotes selecting B or A, respectively. Each training instance provides a reference preference $v ^ { \star }$ , but no reference rationale.

Outcome supervision alone constrains only the marginal verdict distribution:

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { o u t } } ( \theta ) = \mathbb { E } _ { ( \xi , v ^ { \star } ) } \mathbb { E } _ { ( r , v ) \sim \pi _ { \theta } ( \cdot \vert \xi ) } { \mathbf 1 } [ v = v ^ { \star } ] } \\ & { \quad \quad = \mathbb { E } _ { ( \xi , v ^ { \star } ) } \pi _ { \theta } ( v ^ { \star } \mid \xi ) . } \end{array}\tag{1}
$$

Thus, judges with the same probability of predicting the correct verdict can receive the same outcome reward while inducing substantially different conditional rationale distributions $\pi _ { \theta } ( r \mid \xi , v ^ { \star } )$ This motivates supervision over the quality of the supporting rationale.

Evaluation policy. Let $\mathcal { K } ( \xi )$ denote the applicable criteria for an instance. The policy specifies criterion importance weights $\alpha _ { k } > 0$ , which we normalize as

$$
\omega _ { k } = \frac { \alpha _ { k } } { \sum _ { \ell \in \mathcal { K } ( \xi ) } { \alpha _ { \ell } } } .\tag{2}
$$

For a criterion-level preference vector $\begin{array} { r l } { \pmb { d } } & { { } = } \end{array}$ $( d _ { k } ) _ { k \in \mathcal { K } ( \xi ) }$ , with $d _ { k } \in [ - 1 , 1 ]$ and positive values favoring A, the policy aggregates criterion-level comparisons as

$$
F _ { \mathcal { P } } ( \pmb { d } ) = \sum _ { k \in \mathcal { K } ( \xi ) } \omega _ { k } d _ { k } .\tag{3}
$$

![](images/f46ed1034f658009814d79ab4381df63b019b3255acca1382c1e63709b8bb37e.jpg)  
Figure 2: Overview of DECERT. We score each judgment along three rationale-quality dimensions, combine the resulting Rationale Score with Outcome Correctness, and use the final score to select SFT targets and construct DPO preference pairs. When all judgments predict the correct verdict, additional preference pairs are formed based on rationale quality.

## 3.2 Decision-Certificate Assessment

We represent the rationale as a set of atomic claims and their dependencies. Let $\mathcal { G } _ { r } = ( \mathcal { A } _ { r } , \mathcal { E } _ { r } )$ be a directed acyclic graph extracted from r, where each node $a _ { j } \in { \mathcal { A } } _ { \iota }$ is

$$
a _ { j } = ( t _ { j } , k _ { j } , p _ { j } , e _ { j } , \mathrm { p a } ( j ) ) .\tag{4}
$$

Here, $t _ { j }$ denotes the claim type, $k _ { j }$ its associated evaluation criterion, $p _ { j }$ the atomic proposition, $e _ { j }$ the cited evidence when applicable, and pa(j) its parent claims. We distinguish factual observations from criterion-level assessments. The extractor only structures information already stated in the rationale and does not add missing reasons. Details are provided in Appendix A.

Soundness. Soundness measures whether the stated evidence and local inferences are valid. We use a natural language inference verifier for two local checks. For each claim $a _ { j } , g _ { j } \in [ 0 , 1 ]$ measures whether its factual content is supported by the cited evidence, while $\ell _ { j } \in [ 0 , 1 ]$ measures whether its conclusion follows from its parent claims:

$$
\nu _ { j } = g _ { j } \ell _ { j } .\tag{5}
$$

For a leaf observation, we set $\ell _ { j } = 1$ ; for a purely inferential node without an additional external factual claim, we set $g _ { j } = 1$ . We provide implementation details in Appendix A.

Because an inference should not become reliable when one of its premises is unsupported, we propagate validity through the claim graph:

$$
q _ { j } = \operatorname* { m i n } _ { h \in \operatorname { A n c } ^ { + } ( j ) } \nu _ { h } ,\tag{6}
$$

where ${ \mathrm { A n c } } ^ { + } ( j )$ contains $a _ { j }$ and all of its ancestors. Let $\boldsymbol { \mathcal { A } } _ { r , k }$ denote the substantive claims associated with criterion k after semantic deduplication. We

define

$$
s _ { k } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { a _ { j } \in A _ { r , k } } q _ { j } , } & { \displaystyle A _ { r , k } \neq \emptyset , } \\ { 1 , } & { \displaystyle A _ { r , k } = \emptyset , } \end{array} \right. S _ { \mathrm { s n d } } ( J ) = \prod _ { k \in \mathcal { K } ( \xi ) } s _ { k } ^ { \omega _ { k } } .\tag{7}
$$

Coverage. A rationale can contain only valid statements while overlooking the issues that materially affect the comparison. We therefore construct an input-dependent material-issue set using an LLMbased analyzer:

$$
D _ { \xi } = { \cal A } _ { \phi } ( \xi ) = \{ ( d _ { t } , k _ { t } , \rho _ { t } ) \} _ { t = 1 } ^ { M } , \qquad \rho _ { t } > 0 .\tag{8}
$$

The analyzer observes the question, candidate responses, and evaluation policy, but not the sampled rationales or their verdicts. Each $d _ { t }$ describes an important requirement, error, or response difference, $k _ { t }$ identifies its criterion, and $\rho _ { t }$ represents its importance. Let $m _ { t j } \in [ 0 , 1 ]$ measure how well claim $a _ { j }$ addresses material issue $d _ { t }$ . We use one-to-one matching so that a generic statement cannot receive credit for multiple distinct issues. We represent a partial one-to-one matching by $P \in \{ 0 , \hat { 1 } \} ^ { M \times | \mathcal { A } _ { r } | }$ where $P _ { t j } = 1$ indicates that issue $d _ { t }$ is matched to claim $a _ { j }$ . Coverage is then

$$
S _ { \mathrm { c o v } } ( J ) = \frac { \displaystyle { \operatorname* { m a x } _ { P \in \mathcal { M } } \sum _ { t , j } \rho _ { t } P _ { t j } m _ { t j } q _ { j } } } { \displaystyle { \sum _ { t } \rho _ { t } } } .\tag{9}
$$

Multiplying by $q _ { j }$ prevents an unsupported claim from receiving full coverage credit. Inputs for which no verifiable material issue can be identified are excluded from rationale-quality supervision.

Decision Support. Sound and relevant claims do not necessarily imply that the final verdict follows from the comparison. We therefore ask whether the verified information is sufficient to support v under the evaluation policy.

For each criterion $k ,$ verified criterion-level assessments induce an interval

$$
I _ { k } ( r ) = [ L _ { k } , U _ { k } ] \subseteq [ - 1 , 1 ]\tag{10}
$$

for its comparison value $d _ { k }$ . An explicitly supported tie yields $I _ { k } ( r ) = \{ 0 \}$ . We detail interval construction in Appendix A.3.

The rationale therefore defines the feasible comparison set

$$
\Omega _ { r } = \prod _ { k \in \mathcal { K } ( \xi ) } I _ { k } ( r ) .\tag{11}
$$

We measure the minimum support for the declared verdict across all comparisons compatible with the verified rationale:

$$
\begin{array} { l } { { \displaystyle B _ { r } ( v ) = \operatorname* { i n f } _ { d \in \Omega _ { r } } v F _ { \mathcal { P } } ( d ) } } \\ { ~ } \\ { { \displaystyle ~ = v \sum _ { k } \omega _ { k } \frac { L _ { k } + U _ { k } } { 2 } - \sum _ { k } \omega _ { k } \frac { U _ { k } - L _ { k } } { 2 } } , } \\ { { \displaystyle S _ { \mathrm { d e c } } ( J ) = \frac { 1 + B _ { r } ( v ) } { 2 } } . } \end{array}\tag{12}
$$

Because $B _ { r } ( v ) \in [ - 1 , 1 ] , S _ { \mathrm { d e c } } \in [ 0 , 1 ]$ . In particular, $B _ { r } ( v ) > 0$ means that every criterion configuration consistent with the verified rationale supports the stated verdict. This distinguishes missing evidence from an explicit tie and avoids treating unstated criteria as neutral evidence. We provide detailed derivations in the appendix A.4.

Overall certificate quality. We combine the three dimensions using the geometric mean:

$$
Q _ { \mathrm { p r o c } } ( J ) = ( S _ { \mathrm { s n d } } ( J ) S _ { \mathrm { c o v } } ( J ) S _ { \mathrm { d e c } } ( J ) ) ^ { 1 / 3 } .\tag{13}
$$

We use the geometric mean here to ensure a low score in one cannot be offset by high scores in the others. Appendix C.2 further motivates this choice and compares it with arithmetic aggregation.

## 3.3 Preference Data Construction

Candidate generation. For each training sample, we construct a candidate pool $\mathcal { I } ( \xi )$ by sampling 3 judgments from GPT-4o (OpenAI, 2024) and 5 from Qwen3-8B (Yang et al., 2025). Each judgment $\boldsymbol { J } ~ = ~ \left( r , v \right)$ is assigned a rationale quality score $Q _ { \mathrm { p r o c } } ( J )$ and an outcome score $O ( J ) =$ ${ \bf 1 } [ v = v ^ { \star } ]$ . To combine both, we define a total score of a judgment:

$$
Q _ { \mathrm { t o t a l } } ( J ) = \frac { O ( J ) + \lambda Q _ { \mathrm { p r o c } } ( J ) } { 1 + \lambda } , 0 < \lambda < 1 .\tag{14}
$$

Then we use this to rank candidates. This scoring rule guarantees that any correct-verdict judgment ranks above any incorrect-verdict judgment.

General preference pairs. For pools containing both correct and incorrect verdicts, we select the highest- and lowest-scoring candidates under $Q _ { \mathrm { t o t a l } }$ as $( J ^ { + } , J ^ { - } )$ . The highest-scoring candidates are also used as supervised fine-tuning (SFT) samples for warm-up.

Rationale preference pairs. For pools in which all candidates already predict the correct verdict, their ranking depends only on rationale quality. We therefore select the candidates with the highest and lowest $Q _ { \mathrm { p r o c } }$ scores and retain the pair if

$$
Q _ { \mathrm { p r o c } } ( J ^ { + } ) - Q _ { \mathrm { p r o c } } ( J ^ { - } ) > \delta .\tag{15}
$$

Where $\delta$ is the score gap threshold. We further compare different choices of the threshold in $\mathsf { A p - }$ pendix C.1. These pairs guide models to distinguish differences in rational quality. Pools containing only incorrect verdicts are discarded.

Two-Stage Alignment. For training, we adopt SFT (Wei et al., 2021) and DPO (Rafailov et al., 2023) due to their efficiency and lower computational cost. We use Llama3.1-8B (AI, 2024) and Qwen3- 8B as base models and follow a two-stage training pipeline with SFT followed by DPO. We apply Low-Rank Adaptation (Hu et al., 2022) for computational efficiency. Further configuration details are provided in Appendix B.2.

## 4 Experiments

## 4.1 Experiments Setup

Training data and models. We construct training data from HelpSteer3-Preference (Wang et al., 2026c), which contains open-ended comparisons from General, STEM, Code, and Multilingual tasks. We first sample around 12k instances, balanced across four domains. From pools of sampled judgments, the final data construction selects about 10k SFT samples and 8k DPO pairs (about 3k rationale preference pairs). We train two base models from different model families. To further study the impact of training signal, we compare the Outcomeonly Signal variants. Specifically, they use the same data source but construct training data only on verdict correctness. Additional construction and optimization details are in Appendix B.

Evaluation data and metrics. We evaluate on the test set of HelpSteer3 (2k). We also use two out-of-distribution (OOD) benchmarks, i.e., RM-Bench (Liu et al., 2024) which measures a model’s ability to discriminate subtle differences and style biases across Chat, Code, Math, and Safety, and

Table 1: Light Blue and underlined highlight the best and second-best results, respectively. DeCert consistently improves over outcome-only baselines, and our Qwen3-8B model achieves state-of-the-art results among strong baselines. JudgeBench reports positional consistent accuracy. “Avg.” averages the two Overall columns. †: reported by Wang et al. (2026a); ‡: reported by the original papers (J1; Whitehouse et al., 2026).
<table><tr><td rowspan="2">Models</td><td colspan="5">RM-Bench</td><td colspan="5">JudgeBench</td><td rowspan="2">Avg.</td></tr><tr><td>Chat</td><td>Math</td><td>Code</td><td>Safety</td><td>Overall</td><td>Knwl.</td><td>Reas.</td><td>Math</td><td>Code</td><td>Overall</td></tr><tr><td colspan="10">LLM-as-a-Judge</td><td></td><td></td><td></td></tr><tr><td>GPT-4o†</td><td>67.2</td><td>67.5</td><td>63.6</td><td>91.7</td><td>72.5</td><td>50.6</td><td>54.1</td><td>75.0</td><td>59.5</td><td>59.8</td><td>66.2</td></tr><tr><td>Claude-3.5-Sonnet†</td><td>62.5</td><td>62.6</td><td>54.4</td><td>64.4</td><td>61.0</td><td>62.3</td><td>66.3</td><td>66.1</td><td>64.3</td><td>64.8</td><td>62.9</td></tr><tr><td colspan="10">Generative Reward Models</td><td></td><td></td><td></td></tr><tr><td>Nemotron-30B-A3B</td><td>68.3</td><td>69.0</td><td>59.1</td><td>90.1</td><td>71.6</td><td>46.8</td><td>41.8</td><td>62.5</td><td>47.6</td><td>48.0</td><td>59.8</td></tr><tr><td>GRAM-R2</td><td>68.6</td><td>63.1</td><td>54.2</td><td>93.3</td><td>69.8</td><td>45.5</td><td>45.7</td><td>46.4</td><td>28.6</td><td>43.7</td><td>56.8</td></tr><tr><td>RewardAnything-8B</td><td>68.1</td><td>69.8</td><td>55.1</td><td>93.6</td><td>71.7</td><td>52.6</td><td>41.8</td><td>69.6</td><td>42.9</td><td>51.1</td><td>61.4</td></tr><tr><td>Con-J-7B</td><td>68.1</td><td>58.2</td><td>51.0</td><td>85.1</td><td>65.6</td><td>37.0</td><td>45.9</td><td>33.9</td><td>38.1</td><td>39.1</td><td>52.4</td></tr><tr><td>J1-Llama-8B‡</td><td></td><td></td><td></td><td></td><td>73.4</td><td></td><td></td><td></td><td></td><td>42.0</td><td>57.7</td></tr><tr><td>R-Align-8B</td><td>69.5</td><td>68.8</td><td>59.2</td><td>93.5</td><td>72.8</td><td>45.5</td><td>45.9</td><td>57.1</td><td>54.8</td><td>48.6</td><td>60.7</td></tr><tr><td colspan="10">Same Data, Outcome-only Signal</td><td></td><td></td><td></td></tr><tr><td>Llama-3.1-8B</td><td>64.6</td><td>59.0</td><td>51.3</td><td>78.4</td><td>63.3</td><td>37.0</td><td>42.9</td><td>53.6</td><td>19.0</td><td>39.1</td><td>51.2</td></tr><tr><td>Qwen3-8B</td><td>73.1</td><td>69.8</td><td>60.7</td><td>90.2</td><td>73.5</td><td>47.4</td><td>39.8</td><td>64.3</td><td>50.0</td><td>48.3</td><td>60.9</td></tr><tr><td colspan="10">Decision-Certificate Supervision (Ours)</td><td></td><td></td><td></td></tr><tr><td>Llama-3.1-8B</td><td>69.6</td><td>60.9</td><td>52.3</td><td>85.0</td><td>67.0</td><td>45.5</td><td>40.8</td><td>51.8</td><td>23.8</td><td>42.6</td><td>54.8</td></tr><tr><td>Qwen3-8B</td><td>77.8</td><td>70.9</td><td>64.6</td><td>91.4</td><td>76.2</td><td>51.3</td><td>43.9</td><td>64.3</td><td>57.1</td><td>52.0</td><td>64.1</td></tr></table>

JudgeBench (Tan et al., 2025) which emphasizes deep judgment and logical reasoning on challenging Knowledge and Reasoning tasks. For Help-Steer3, Acc. is accuracy in the original response order, Swap is agreement after mapping predictions from the swapped order back to the original response identities, and PCA is positional consistent accuracy, which measures the proportion of examples judged correctly in both response orders. Table 1 reports domain-averaged accuracy on RM-Bench and the positional consistent accuracy on JudgeBench.

Baselines. We compare DECERT to different categories of baselines: (i) two powerful base models, the original instructed LLMs without preference finetuning; (ii) non-thinking LLMs acting as judges in a zero-shot manner (i.e., GPT-4o (OpenAI, 2024), Claude-3.5-Sonnet (Anthropic, 2024)); (iii) state-of-the-art generative reward models that belong to the same category as DECERT (i.e., J1-Llama-8B (Whitehouse et al., 2026), R-Align-8B (Lai et al., 2026), Con-J-7B (Ye et al., 2025), RewardAnything-8B (Yu et al., 2025b), GRAM-$R ^ { 2 }$ (Wang et al., 2026b)) and thinking LLMs acting as judges (i.e., Nemotron-30B-A3B (Chandiramani et al., 2026)). Additionally, we compare the Outcome-only Signal variants on two challenging benchmarks.

![](images/4b647cefb5e6ed6760035ec4a40646a7a167e6907a352be38e34b5bb5aa8d97e.jpg)  
Figure 3: Effect of rationale preference pairs. Adding certificate-ranked all-correct pairs to outcome-only training improves positional consistent accuracy by up to 6.2 points.

## 4.2 Overall Judgment Performance

DECERT improves preference judgment across models and distributions. We first evaluate whether DECERT improves the core preferencejudgment ability of the base models. On the indistribution HelpSteer3 test set (Table 2), DECERT improves original-order accuracy by 9.1 points for Llama-3.1-8B and 6.3 points for Qwen3-8B over their corresponding base judges $( p < 0 . 0 0 0 1 )$ .

The improvements transfer to OOD evaluation in Table 1. Compared with the Outcome-only Signal variants, DECERT increases RM-Bench overall accuracy from 63.3 to 67.0 for Llama and from 73.5 to 76.2 for Qwen. On JudgeBench, positional consistent accuracy increases from 39.1 to 42.6 and from 48.3 to 52.0, respectively. Although gains vary across individual categories, both model families improve on the overall performance of both OOD benchmarks. Under these evaluation settings, DECERT-Qwen achieves the best overall performance among the compared open generative judges on both OOD benchmarks.

Table 2: Results on HelpSteer3 comparing judgment bias. Acc.: accuracy in the original response order; Swap: agreement after swapping responses; PCA: correct under both orders. Reported gains are relative to the corresponding base models. Bold and underlined mark the best and second-best scores, respectively.
<table><tr><td rowspan="2">Models</td><td>#Training</td><td colspan="3">Overall</td><td colspan="4">PCA by domain</td></tr><tr><td>Pref. Pairs</td><td>Acc.</td><td>Swap</td><td>PCA</td><td>Code</td><td>General</td><td>Multiling.</td><td>STEM</td></tr><tr><td colspan="9">Base Models</td></tr><tr><td>Llama-3.1-8B</td><td></td><td>63.9</td><td>61.2</td><td>45.2</td><td>51.4</td><td>43.3</td><td>44.3</td><td>42.5</td></tr><tr><td>Qwen3-8B</td><td>一</td><td>65.2</td><td>67.9</td><td>53.6</td><td>62.1</td><td>47.3</td><td>61.6</td><td>53.1</td></tr><tr><td colspan="9">Open Generative Reward Models</td></tr><tr><td>Con-J-7B</td><td>80K</td><td>66.2</td><td>69.9</td><td>51.1</td><td>57.4</td><td>48.3</td><td>51.4</td><td>50.0</td></tr><tr><td>R-Align-8B</td><td>109K</td><td>70.4</td><td>72.2</td><td>56.7</td><td>62.8</td><td>54.9</td><td>60.0</td><td>49.4</td></tr><tr><td>RewardAnything-8B</td><td>173K†</td><td>72.9</td><td>75.9</td><td>60.5</td><td>68.8</td><td>57.0</td><td>63.8</td><td>55.6</td></tr><tr><td>GRAM  $\cdot \mathbf { R } ^ { 2 }$ </td><td> $2 . 5 4 \mathbf { M } ^ { \dagger }$ </td><td>72.7</td><td>68.3</td><td>57.8</td><td>63.5</td><td>53.5</td><td>65.9</td><td>54.4</td></tr><tr><td colspan="9">Decision Certificate Supervision (Ours)</td></tr><tr><td>Llama-3.1-8B (SFT-only)</td><td></td><td> $7 0 . 1 _ { + 6 . 2 }$ </td><td> $6 7 . 6 _ { + 6 . 4 }$ </td><td> $5 4 . 1 _ { + 8 . 9 }$ </td><td> $5 9 . 9 _ { + 8 . 5 }$ </td><td> $5 1 . 2 \substack { + 7 . 9 }$ </td><td> $6 0 . 0 _ { + 1 5 . 7 }$ </td><td> $4 7 . 5 _ { + 5 . 0 }$ </td></tr><tr><td>Llama-3.1-8B (SFT+DPO)</td><td>8K</td><td> $7 3 . 0 _ { + 9 . 1 }$ </td><td> $7 4 . 2 _ { + 1 3 . 0 }$ </td><td> $6 0 . 0 _ { + 1 4 . 8 }$ </td><td> $6 8 . 1 _ { + 1 6 . 7 }$ </td><td> $5 6 . 3 _ { + 1 3 . 0 }$ </td><td> $6 5 . 4 _ { + 2 1 . 1 }$ </td><td> $5 3 . 8 _ { + 1 1 . 3 }$ </td></tr><tr><td>Qwen3-8B (SFT-only)</td><td></td><td> $6 8 . 5 _ { + 3 . 3 }$ </td><td> $6 8 . 5 _ { + 0 . 6 }$ </td><td> $5 3 . 8 _ { + 0 . 2 }$ </td><td> $6 2 . 8 _ { + 0 . 7 }$ </td><td> $4 8 . 3 _ { + 1 . 0 }$ </td><td> $6 2 . 2 _ { + 0 . 6 }$ </td><td> $4 9 . 4 _ { - 3 . 7 }$ </td></tr><tr><td>Qwen3-8B (SFT+DPO)</td><td>8K</td><td> $7 1 . 5 _ { + 6 . 3 }$ </td><td> ${ \bf 8 1 . 4 } _ { + 1 3 . 5 }$ </td><td> $\mathbf { 6 2 . 9 } _ { + 9 . 3 }$ </td><td> $7 0 . 6 _ { + 8 . 5 }$ </td><td> ${ \bf 6 0 . 5 } _ { + 1 3 . 2 }$ </td><td> $\mathbf { 6 5 . 9 } _ { + 4 . 3 }$ </td><td> $5 5 . 0 _ { + 1 . 9 }$ </td></tr></table>

Training counts follow the original sources: Con-J (Ye et al., 2025) and R-Align (Lai et al., 2026). †: 173K pairwise-equivalent preferences from approximately 4K listwise examples (Yu et al., 2025b). ‡: 1M labeled and 1.5M pseudo-labeled pairs for pre-training, plus 40.5K for fine-tuning (Wang et al., 2026b).

![](images/a108e4a98ac5f97bb014a1da68a4521561c97679f47d743e4fc1fc2edec3e299.jpg)

![](images/b8b546c3e8ddccaa8af7b510d99493efa01c948b87e06048254e5ed6d73c5879.jpg)  
Figure 4: Counterfactual evidence sensitivity. Evidence Adaptation Rate (EAR; ↑) measures whether the rationale updates its assessment of the edited issue, while Stale Rationale Rate (SRR; ↓) measures reuse of an invalidated claim.

Table 3: Rationale quality on HelpSteer3-Atomic, measured by RationaleRM-style average precision. w/o Rati. Pairs removes rationale preference pairs.
<table><tr><td rowspan="2">Method</td><td colspan="2">Overall</td><td colspan="2">Multiling.</td></tr><tr><td>Score</td><td>∆</td><td>Score</td><td> $\Delta$ </td></tr><tr><td>Base</td><td>0.2059</td><td>一</td><td>0.2287</td><td>一</td></tr><tr><td>w/o Rati. Pairs</td><td>0.2212</td><td>↑0.0153</td><td>0.2973</td><td>↑0.0686</td></tr><tr><td>Ours</td><td>0.2624</td><td>↑0.0565</td><td>0.3589</td><td>↑0.1302</td></tr></table>

## 4.3 What Does Rationale Supervision Add?

All-correct judgments provide supervision beyond outcome labels. Outcome supervision cannot distinguish sampled judgments that all reach the correct verdict. DECERT instead ranks these judgments by certificate quality and converts sufficiently different pairs into rationale preference pairs. Figure 3 isolates the contribution of these all-correct examples. Adding rationale preference pairs increases PCA from 46.7% to 52.9% on RM-

Bench and from 39.1% to 42.6% on JudgeBench, corresponding to gains of 6.2 and 3.5 points. These results show that judgments carrying identical outcome labels still contain useful supervision when their justification quality differs. Besides, we provide a more comprehensive ablation in Appendix C.3.

The additional supervision transfers to rationale quality metrics. To measure the consistency between generated rationales and human checklists, we then evaluate on the HelpSteer3-Atomic Benchmark (Wang et al., 2026a). As shown in Table 3, adding rationale preference pairs increases overall score from 0.2212 to 0.2624 and multilingual score from 0.2973 to 0.3589. Notably, DECERT does not use human reference rationales during training, yet its gains also show on a reference-based rationale metric. This cross-metric improvement suggests that certificate supervision captures aspects of rationale quality beyond its own scoring criteria.

## 4.4 Rationale Quality and Evidence Sensitivity

DECERT reduces common rationale flaws. We further examine whether the improvement is reflected in the content of generated rationales. Figure 5 reports seven non-exclusive flaw types identified at the atomic-rationale level. DECERT reduces the observed rate of every flaw type. The largest reduction occurs for non sequiturs, which decrease from 21.5% to 3.3%. Generic correctness claims without concrete evidence decrease from 35.0% to 25.1%, while single-sided praise decreases from 3.6% to 1.7%. Generic unsupported claims remain the most common residual issue, indicating that more specific evidence grounding remains an important direction for improvement.

![](images/5d1a6434fefc7921164d6f03a6bfa3e454a27a77185ab740645e307f31997025.jpg)  
Figure 5: Rates of seven non-exclusive rationale flaws. Lower rates indicate better rationale quality.

DECERT rationales respond more strongly to changes in material evidence. Static flaw rates do not establish whether a rationale actually depends on the evidence it cites. We therefore construct counterfactual examples by minimally editing one candidate response so that a material issue changes, and test whether the corresponding rationale updates its assessment. Figure 4 reports results on 1k edited pairs. DECERT increases Evidence Adaptation Rate (EAR) from 64.5% to 88.9% and reduces Stale Rationale Rate (SRR) from 34.0% to 11.1%. Qwen shows the same pattern: EAR increases from 87.2% to 93.4%, while SRR decreases from 30.5% to 7.3%. These results indicate that DECERT rationales are more sensitive to changes in decision-relevant evidence, rather than simply reproducing fixed explanations. We also provide a case study in Appendix E.

## 4.5 Positional Robustness

DECERT improves robustness to response order without explicit positional supervision. Finally, we test whether DECERT is robust to changes in response order. On HelpSteer3, DECERT improves swap agreement by 13.0 points for Llama and 13.5 points for Qwen relative to the corresponding base judges. Positional consistent accuracy improves by 14.8 and 9.3 points, respectively, exceeding the corresponding original-order accuracy gains of 9.1 and 6.3 points. Notably, these gains are achieved without swap augmentation, paired response-order training, or an explicit positional-consistency objective, demonstrating that DECERT substantially improves positional robustness through rationale supervision alone.

## 5 Conclusion

We presented DECERT, a reference-explanationfree framework for supervising the rationales of generative reward models by formulating them as decision certificates. Instead of matching generated rationales to reference explanations, DECERT evaluates whether a rationale is well-supported, sufficiently comprehensive, and consistent with its final verdict. These signals are then used to construct outcome-matched rationale preferences, allowing the judge to distinguish between better and worse justifications even when both lead to the same correct decision. Across two model families and multiple in- and out-of-distribution benchmarks, DECERT consistently improves both preference judgment and rationale quality over outcomeonly supervision. Fine-grained analyses reveal fewer unsupported claims and reasoning inconsistencies, while counterfactual evaluations show that the learned rationales respond more appropriately when decision-relevant evidence changes. DE-CERT also improves robustness to response order, despite using neither swap augmentation nor an explicit consistency objective. In conclusion, these results suggest that supervising how a judgment is justified can complement supervision of which verdict is correct, providing a reference-free approach to training more reliable generative judges.

## 6 Limitations

DECERT has several limitations. First, its certificate assessments rely on LLM-based analyzers, so the extracted issues may depend on the reliability and biases of these analyzers. Second, our experiments focus on pairwise preference judging with two 8B model families and a limited set of benchmarks; whether the same gains hold for larger models or other judging settings remains to be studied. Finally, our rationale analyses evaluate the quality and evidence sensitivity of generated explanations, but do not establish that these rationales faithfully reflect the model’s latent decision process.

## Acknowledgments

Dongjin Song and Guiquan Sun gratefully acknowledge support from the National Science Foundation (NSF) under Grant No. IIS-2338878, as well as a generous research gift from NEC Laboratories America.

## References

Meta AI. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Anthropic. 2024. Claude 3.5 sonnet model card addendum.

Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, et al. 2026. Nemotron 3 super: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2604.12374.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. 2026. Rm-r1: Reward modeling as reasoning. In International Conference on Learning Representations, volume 2026, pages 88313–88342.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and

Minjoon Seo. 2024a. Prometheus: Inducing finegrained evaluation capability in language models. In International Conference on Learning Representations.

Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. 2024b. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 4334–4353. Association for Computational Linguistics.

Yanlin Lai, Mitt Huang, Hangyu Guo, Xiangfeng Wang, Haodong Li, Shaoxiong Zhan, Liang Zhao, Chengyuan Yao, Yinmin Zhang, Qi Han, et al. 2026. R-align: Enhancing generative reward models through rationale-centric meta-judging. arXiv preprint arXiv:2602.06763.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. 2024. Rewardbench: Evaluating reward models for language modeling.

Xiaobo Liang, Haoke Zhang, Juntao Li, Kehai Chen, Qiaoming Zhu, and Min Zhang. 2025. Generative reward modeling via synthetic criteria preference learning. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 26755–26769. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522. Association for Computational Linguistics.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2024. Rm-bench: Benchmarking reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. 2024. Generative reward models. arXiv preprint arXiv:2410.12832.

OpenAI. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35.

Kai Qin, Liangxin Liu, Yu Liang, Longzheng Wang, Wangyan, Zhang Yueyang, Long Xia, Zhiyuan Sun, Houde Liu, and Daiting Shi. 2026. ReflectRM: Boosting generative reward models via self-reflection within a unified judgment framework. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 36207–36223. Association for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. 2025. Learning to plan & reason for evaluation with thinking-llm-as-ajudge. arXiv preprint arXiv:2501.18099.

Xuyang Shen, Haoran Liu, Dongjin Song, and Martin Renqiang Min. 2026. Uncertainty-guided latent diagnostic trajectory learning for sequential clinical diagnosis. arXiv preprint arXiv:2604.05116.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Tang, Alejandro Cuadron, Chenguang Wang, Raluca Popa, and Ion Stoica. 2025. Judgebench: A benchmark for evaluating llm-based judges. In International Conference on Learning Representations, volume 2025, pages 63277–63303.

Binghai Wang, Yantao Liu, Yuxuan Liu, Tianyi Tang, Shenzhi Wang, Chang Gao, Chujie Zheng, Yichang Zhang, Le Yu, Shixuan Liu, et al. 2026a. Outcome accuracy is not enough: Aligning the reasoning process of reward models. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 41482– 41508.

Chenglong Wang, Yongyu Mu, Hang Zhou, Yifu Huo, Ziming Zhu, Jiali Zeng, Murun Yang, Bei Li, Xiaoyang Hao, Chunliang Zhang, et al. 2026b. Gramr<sup>2</sup>: Self-training generative foundation reward models for reward reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 33395–33403.

PeiFeng Wang, Austin Xu, Yilun Zhou, Caiming Xiong, and Shafiq Joty. 2025. Direct judgement preference optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1979–2009. Association for Computational Linguistics.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024a. Large language models are not fair evaluators. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450. Association for Computational Linguistics.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024b. Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439. Association for Computational Linguistics.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. 2026c. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. Advances in Neural Information Processing Systems, 38.

Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, and Yongbin Li. 2026d. Reward modeling from natural language human feedback. arXiv preprint arXiv:2601.07349.

Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, and Yongbin Li. 2026e. Reward modeling from natural language human feedback. In Forty-third International Conference on Machine Learning.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason E Weston, Ilia Kulikov, and Swarnadeep Saha. 2026. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 10397–10420.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Ziyi Ye, Xiangsheng Li, Qiuchi Li, Qingyao Ai, Yujia Zhou, Wei Shen, Dong Yan, and Yiqun Liu. 2025. Learning llm-as-a-judge for preference alignment. In The Thirteenth International Conference on Learning Representations.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, and Rui Hou. 2025a. Self-generated critiques boost reward modeling for language models. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11499–11514. Association for Computational Linguistics.

Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fandong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, and Wei Ye. 2025b. Rewardanything: Generalizable principle-following reward models. arXiv preprint arXiv:2506.03637.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

## A Implementation Details of DeCert

## A.1 Evaluation Policies

We use an LLM to generate domain-specific rubrics for the evaluation. Table 4 lists the domainspecific criterion weights used in our implementation. Omitted criteria are excluded from score aggregation.

Table 4: Domain-specific evaluation weights. Corr.: correctness; Inst.: instruction following; Lang.: language consistency; Comp.: completeness; Clar.: clarity.
<table><tr><td>Domain</td><td>Corr.</td><td>Inst.</td><td>Lang.</td><td>Comp.</td><td>Clar.</td></tr><tr><td>Code/STEM</td><td>0.40</td><td>0.25</td><td>一</td><td>0.20</td><td>0.15</td></tr><tr><td>General</td><td>0.30</td><td>0.30</td><td></td><td>0.25</td><td>0.15</td></tr><tr><td>Multilingual</td><td>0.30</td><td>0.25</td><td>0.20</td><td>0.15</td><td>0.10</td></tr></table>

We parse each judgment into criterion-level observations, inferences, and winners, together with a final verdict. A judgment is considered parseable if it contains at least one criterion block with a valid <result> field and an <answer> selecting A or $B .$ Unparseable judgments receive zero scores and are excluded from preference-pair construction.

## A.2 Material-Issue Extraction and Matching

The material-issue analyzer receives the evaluation input ${ \boldsymbol { \xi } } = ( x , y _ { A } , y _ { B } , \mathcal { P } )$ without access to any sampled rationale or verdict, and extracts at most six material issues:

$$
D _ { \xi } = \{ d _ { t } = ( u _ { t } , k _ { t } , \rho _ { t } , s _ { t } ) \} _ { t = 1 } ^ { M } ,\tag{16}
$$

where $u _ { t }$ describes the issue, $k _ { t }$ denotes the associated evaluation criterion, $\rho _ { t } \in [ 1 , 3 ]$ its importance, and $s _ { t } \in \{ - 1 , 0 , + 1 \}$ its comparison direction.

For each material issue $d _ { t }$ and rationale claim $a _ { j } .$ we compute a matching score $m _ { t j } \in \{ 0 , 0 . 5 , 1 \}$ according to whether the claim correctly addresses the same issue with a compatible assessment direction. Free-form summary statements are excluded from matching.

Let $N _ { r } = | \mathcal { A } _ { r } |$ . We use the maximum-weight partial one-to-one matching $\{ P \in \{ 0 , 1 \} ^ { M \times | \bar { A _ { r } } | } \}$ with pairwise weight

$$
w _ { t j } = \rho _ { t } m _ { t j } q _ { j } ,\tag{17}
$$

where $q _ { j }$ is the verified support-chain validity of claim $a _ { j }$ . The resulting coverage score follows Eq. 9.

If the analyzer identifies no verifiable material issue $( M = 0 )$ , the instance is excluded from training rather than assigned a default coverage score.

## A.3 Constructing Criterion Intervals from Verified Conclusions

For each criterion k, the comparison scale and admissible forms of criterion-level conclusions are specified before auditing. We normalize the scale to $\mathcal { Z } _ { k } \subseteq [ - 1 , 1 ]$ , where larger values favor A. A simple choice is

$$
\mathcal { Z } _ { k } = \{ - 1 , 0 , + 1 \} ,
$$

representing that $B$ is better, the two responses are tied, or A is better, respectively. A finer finite scale may be used when the evaluation policy provides explicit comparison rules.

Let $\mathcal { H } _ { k } ( r )$ denote the set of verified comparative predicates for criterion k that are explicitly stated in rationale $r .$ . We include a predicate only when its supporting claim chain satisfies the local-validity threshold $\pi _ { \mathrm { l o c } }$ . Typical predicates include $d _ { k } \geq b ,$ $d _ { k } \leq b ,$ , and $d _ { k } = b$ , where $b \in \mathcal { Z } _ { k }$ . Each predicate retains the claims and evidence from which it was derived. The values compatible with these verified predicates are

$$
\begin{array} { r l } & { \mathcal { B } _ { k } ( r ) = \big \{ z \in \mathcal { Z } _ { k } : \psi ( z ) \mathrm { h o l d s ~ f o r ~ e v e r y ~ } \psi \in \mathcal { H } _ { k } ( r ) \big \} , } \\ & { \quad I _ { k } ( r ) = [ \operatorname* { m i n } { \mathcal { B } _ { k } ( r ) } , \operatorname* { m a x } { \mathcal { B } _ { k } ( r ) } ] . } \end{array}\tag{18}
$$

Thus, $I _ { k } ( r )$ is the interval hull of the admissible criterion values. Using the interval hull provides a conservative outer approximation when $\mathcal { Z } _ { k }$ is discrete.

If $\mathcal { H } _ { k } ( r ) = \emptyset$ , no verified comparative information is available for criterion $k ,$ and we retain the full interval $I _ { k } ( r ) = [ - 1 , 1 ]$ . If $\boldsymbol { B _ { k } } ( \boldsymbol { r } ) = \boldsymbol { \mathcal { O } }$ the accepted predicates are mutually inconsistent; we therefore reject the corresponding certificate. Unparseable dependency graphs, cyclic dependencies, or missing verification records are handled similarly.

Importantly, verifying a local error does not by itself establish a complete criterion-level comparison. For example, confirming an error in $y _ { A }$ does not imply that $y _ { B }$ is better on criterion k unless the evidence and the prespecified evaluation rule jointly support that conclusion. We therefore narrow $I _ { k } ( r )$ only when a verified comparative predicate constrains the relative criterion value $d _ { k }$ . The resulting intervals summarize conclusions supported by the audited rationale; they should not be interpreted as statistical confidence or credible intervals. Their validity depends on the reliability of the underlying grounding and inference checks.

## A.4 Closed-Form Conservative Decision Margin

For each criterion, let

$$
m _ { k } = \frac { L _ { k } + U _ { k } } { 2 } , \qquad h _ { k } = \frac { U _ { k } - L _ { k } } { 2 } .
$$

Every $d _ { k } \in [ L _ { k } , U _ { k } ]$ can be written as

$$
d _ { k } = m _ { k } + h _ { k } \eta _ { k } , \qquad \eta _ { k } \in [ - 1 , 1 ] .
$$

Recall that the policy score is

$$
F _ { \mathcal { P } } ( \pmb { d } ) = \sum _ { k } \omega _ { k } d _ { k } ,
$$

where $\omega _ { k } \geq 0$ and $\textstyle \sum _ { k } \omega _ { k } = 1$ . For a declared verdict $v \in \{ - 1 , + 1 \}$ ,

$$
\begin{array} { l } { { \displaystyle \operatorname* { i n f } _ { d \in \Omega _ { r } } v F _ { \mathcal P } ( d ) = \operatorname* { i n f } _ { \eta \in [ - 1 , 1 ] ^ { | \mathcal K ( \xi ) | } } v \sum _ { k } \omega _ { k } \big ( m _ { k } + h _ { k } \eta _ { k } \big ) } } \\ { ~ = v \sum _ { k } \omega _ { k } m _ { k } + \sum _ { k } \operatorname* { i n f } _ { \eta _ { k } \in [ - 1 , 1 ] } v \omega _ { k } h _ { k } \eta _ { k } } \\ { ~ = v \sum _ { k } \omega _ { k } m _ { k } - \sum _ { k } \omega _ { k } h _ { k } . } \end{array}\tag{19}
$$

This yields the closed form used in Eq. (12):

$$
B _ { r } ( v ) = v \sum _ { k } \omega _ { k } \frac { L _ { k } + U _ { k } } { 2 } - \sum _ { k } \omega _ { k } \frac { U _ { k } - L _ { k } } { 2 } .\tag{20}
$$

The first term is the signed policy score at the center of the feasible region, while the second term subtracts the total unresolved uncertainty. Consequently, wider unresolved intervals reduce the support margin even when the currently verified evidence favors the declared verdict.

Two special cases illustrate this behavior. If every criterion is fixed to a verified point value, then $L _ { k } = U _ { k } = d _ { k }$ and $h _ { k } = 0$ , giving

$$
B _ { r } ( v ) = v \sum _ { k } \omega _ { k } d _ { k } .\tag{21}
$$

Thus, the conservative margin reduces to the ordinary signed policy score.

Alternatively, suppose only a subset $\nu \subseteq \kappa ( \xi )$ has verified point values, while all remaining criteria are unresolved with $I _ { k } ( r ) = [ - 1 , 1 ]$ . Then

$$
B _ { r } ( v ) = v \sum _ { k \in \mathcal { V } } \omega _ { k } d _ { k } - \sum _ { k \notin \mathcal { V } } \omega _ { k } .\tag{22}
$$

Hence, unresolved criteria retain their policy weight as potential evidence against the declared verdict rather than disappearing through renormalization over the criteria already discussed by the rationale. To study the effect of score aggregation, we construct preference data by using two different ways on the same data source. Table 7 shows that geometric aggregation achieves better performance on two challenging benchmarks.

## B Experiments Details

## B.1 Training Objectives

We first formulate the training objectives in two stages. SFT (Wei et al., 2021) minimizes the negative log-likelihood of selected judgments:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( \xi , J ) \sim \mathcal { D } _ { \mathrm { S F T } } } [ \log \pi _ { \theta } ( J \mid \xi ) ] .\tag{23}
$$

The judge is then initialized from the SFT checkpoint and optimized using the standard DPO loss (Rafailov et al., 2023):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { ( \xi , J ^ { + } , J ^ { - } ) \sim \mathcal { D } _ { \mathrm { D P O } } } \Bigg [ \log \sigma \Bigg ( \beta \log \frac { \pi _ { \boldsymbol { \theta } } ( J ^ { + } \mid \xi ) } { \pi _ { \mathrm { r e f } } ( J ^ { + } \mid \xi ) } } \\ { - \beta \log \frac { \pi _ { \boldsymbol { \theta } } ( J ^ { - } \mid \xi ) } { \pi _ { \mathrm { r e f } } ( J ^ { - } \mid \xi ) } \Bigg ) \Bigg ] , } \end{array}\tag{24}
$$

where $\pi _ { \mathrm { r e f } }$ is the reference policy and $\beta$ is the DPO temperature parameter.

For DPO pair construction, we use $\lambda = 0 . 5$ to calculate total scores. Groups whose parseable judgments are all incorrect are discarded.

## B.2 Training Parameter

For all training conducted in this paper, we adopt Low-Rank Adaptation (Hu et al., 2022) for efficiency. All training was conducted using four A100-40GB GPUs or L40 GPUs.

SFT Training. For all supervised fine-tuning (SFT), we use a batch size of 1 per device with gradient accumulation over 16 steps. Training employs a linear learning-rate scheduler with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and a warmup ratio of 0.05. To improve stability, we enable gradient checkpointing to reduce memory overhead. Optimization is performed with RMSprop, and trained with 1 epoch.

DPO Training. For direct preference optimization (DPO), training is conducted for 3 epochs with a batch size of 1 per device and gradient accumulation over 16 steps. The learning rate remains $2 \times 1 0 ^ { - 6 }$ , but we switch to a cosine scheduler for smoother convergence. We continue to apply gradient checkpointing. To regulate policy updates, we set the DPO regularization parameter $\beta = 0 . 1$ , ensuring stability while maintaining alignment with human preferences.

## B.3 Model Checkpoints and Decoding

Table 5 gives the exact public checkpoints used for our local evaluation and the corresponding decoding temperatures.

## C Extended Experiment Results

## C.1 Comparison of Different Rationale Score Thresholds

We conduct additional experiments on the benchmarks as in the main experiments to compare different thresholds for rationale score gap, including $\delta = \{ 0 . 1 5 , 0 . 2 , 0 . 2 5 \}$ . The $\delta$ is set to 0.25 by default. As shown in Table $^ { 6 , }$ the model shows a slight improvement, but the benefit here is limited because, as the threshold increases, the filtered data also decreases.

## C.2 Comparison of Different Score Aggregation Strategies

The scoring hierarchy uses different aggregation rules for different relationships. Across certificate dimensions, a geometric mean discourages compensation for a failed requirement. For comparison, the arithmetic alternative is

$$
Q _ { \mathrm { p r o c } } ^ { \mathrm { a r i t h } } ( J ) = \frac { S _ { \mathrm { s n d } } ( J ) + S _ { \mathrm { c o v } } ( J ) + S _ { \mathrm { d e c } } ( J ) } { 3 } .\tag{25}
$$

For example, scores (1, 1, 0) yield $2 / 3$ under the arithmetic mean but 0 under the geometric mean. The aggregation ablation evaluates this choice empirically. We use the arithmetic alternative to aggregate the total score for preference data construction. The experiment results are shown in Table 7. Based on the results, the geometric mean achieves better performance than the arithmetic mean on two benchmarks, which shows the effectiveness of the geometric mean aggregation.

## C.3 Ablation of Rationale Scoring

We examine how the three rationale scores respond to different certificate flaws and how their aggregation affects sensitivity to these flaws. Using the same original and perturbed judgments, we compare $S _ { \mathrm { s n d } } , S _ { \mathrm { c o v } }$ , and $S _ { \mathrm { d e c } }$ individually, as well as their geometric mean $Q _ { \mathrm { g e o } } = Q _ { \mathrm { p r o c } }$ and arithmetic mean $Q _ { \mathrm { a r i t h } }$

Setup. We apply five types of controlled edits to 2k original judgments. Two are intended to preserve rationale validity: valid trade-off (VT) adds a truthful secondary advantage of the non-preferred response, and irrelevant true details (ITD) adds true observations that are irrelevant to the preference decision. Three edits introduce flaws: fabricated evidence (FE) replaces supporting observations with invented details while preserving the criterion conclusions and verdict; missing key issues (MKI) removes verdict-supporting criterion blocks; and reversed verdict (RV) flips the final verdict while keeping the rationale unchanged.

Evaluation protocol. For each score s, we compute the paired change $\Delta _ { s } = s ( J ^ { \prime } ) - s ( J )$ , where $J$ and $J ^ { \prime }$ are the original and perturbed judgments. We count a harmful edit as detected when $\Delta _ { s } < - \tau$ , with $\tau = 0 . 0 5$ . For VT and ITD, success instead requires $\Delta _ { s } \geq - \tau .$ , so a larger score decrease is a false positive. Figure 6 reports these success rates, equivalent to one minus the corresponding miss or false-positive rate. The detection threshold $\tau$ is distinct from the score-gap threshold $\delta$ used to select rationale preference pairs during training.

Individual scores. Figure 6(a) shows that the components respond to different flaws. Soundness detects 90.5% of FE edits, but only 30.4% of MKI and 1.0% of RV edits. Coverage achieves the highest MKI detection rate among the individual scores (64.1%), but detects only 24.0% of FE and 7.0% of RV edits. Decision support detects 95.5% of reversed verdicts, while detecting only 19.5% of FE and 47.8% of MKI edits. These results support treating the three dimensions as complementary checks: a rationale can contain grounded claims yet omit material support, or cover relevant issues yet draw an inconsistent verdict.

Aggregation. In Figure 6(b), geometric aggregation achieves detection rates of 89.5%, 67.4%, and 92.0% on FE, MKI, and RV, respectively, compared with 86.5%, 62.5%, and 90.0% for arithmetic aggregation. This pattern is consistent with the geometric mean reducing compensation for a low component score by higher scores in other dimensions. The improvement comes with a modest increase in false positives: success rates on VT and ITD are 83.5% and 79.5% for the geometric mean, compared with 86.0% and 81.5% for the arithmetic mean. Geometric aggregation therefore provides greater sensitivity to the tested harmful edits, with a small cost in tolerance of the intended non-corrupting edits. MKI remains the hardest harmful perturbation for the combined score, indicating that detecting omitted support remains an area for improvement.

Table 5: Model checkpoints and decoding temperatures. Checkpoint identifiers link to the corresponding official model cards. The listed temperatures are those used in our evaluation.
<table><tr><td>Model in our tables</td><td>Official model card / checkpoint</td><td>Temperature</td></tr><tr><td>Llama-3.1-8B (AI, 2024)</td><td>meta-1lama/Llama-3.1-8B-Instruct</td><td>0.0</td></tr><tr><td>Qwen3-8B (Yang et al., 2025)</td><td>Qwen/Qwen3-8B</td><td>0.6</td></tr><tr><td>Con-J-7B (Ye et al., 2025)</td><td> $\mathrm { Z i y i Y e / C o n - J - Q w e n } 2 \mathrm { - } \mathrm { 7 B }$ </td><td>0.6</td></tr><tr><td>R-Align-8B (Lai et al., 2026)</td><td> $1 \mathtt { y n 2 2 3 3 3 / 6 e n R M - R - A l i g n - 8 B }$ </td><td>0.6</td></tr><tr><td>RewardAnything-8B (Yu et al., 2025b)</td><td>zhuohaoyu/RewardAnything-8B-v1</td><td>0.6</td></tr><tr><td>GRAM-R² (Wang et al., 2026b)</td><td> $\mathsf { w a n g c l n l p / G R A M - R R - L L a M A - 3 . 1 - 8 B - R e w a r d M o d e l }$ </td><td>0.6</td></tr><tr><td>Nemotron-30B-A3B (Chandiramani et al., 2026)</td><td>nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16</td><td>0.6</td></tr></table>

![](images/c84c89a760f70f8ed27f0dc4c8e066f4e0a525881b61ce5f80adfed2e2651355.jpg)

![](images/13f06b95c0e59ec825924ebd8613daa3ef0f7f5875896c056edd698908e658f3.jpg)  
Figure 6: Ablation of rationale scoring under controlled perturbations. (a) Individual component scores. (b) Geometric and arithmetic aggregation. VT: valid trade-off; ITD: irrelevant true details; FE: fabricated evidence; MKI: missing key issues; RV: reversed verdict. Success rates are measured at τ = 0.05 on a shared 0–100% scale; higher is better. Labels in (a) mark harmful-edit detection rates below 50%; labels in (b) show the higher rate on each axis.

Table 6: PCA (%; ↑) with different rationale score-gap thresholds on Llama-3.1-8B. The default is $\delta = 0 . 2 5$ Best results, including ties, are in bold.
<table><tr><td>Threshold δ</td><td>RM-Bench</td><td>JudgeBench</td></tr><tr><td>0.15</td><td>51.9</td><td>42.6</td></tr><tr><td>0.20</td><td>52.4</td><td>41.4</td></tr><tr><td>0.25 (default)</td><td>52.9</td><td>42.6</td></tr></table>

## D Deployment of Analyzer

Advanced proprietary models require costly API calls during data construction. We therefore compare paid and open-source models as the materialissue analyzer. Specifically, we measure the alignment between material issues extracted by different models and reference issues on 500 humanannotated data. As shown in Table 8, GPT-4o-mini achieves nearly the same recall as GPT-4o while costing approximately 16.7× less per token. The locally hosted Qwen3-8B also remains competitive: its F1 decreases by only 4.6 % and its precision is nearly unchanged. These results show that lowercost or open-source models can provide a practical approximation when using frontier models like GPT-4o is not affordable. In this work, we adopted GPT-4o-mini as the analyzer for issue extraction, considering its balance between effectiveness and computational efficiency.

Table 7: Effect of score aggregation. We report positionconsistent accuracy (PCA, %; ↑).
<table><tr><td>Aggregation</td><td>RM-Bench</td><td>JudgeBench</td></tr><tr><td>Arithmetic</td><td>43.0</td><td>40.6</td></tr><tr><td>Geometric (Ours)</td><td>52.9</td><td>42.6</td></tr></table>

Table 8: Material-issue analyzer comparison. API prices are reported per 1M input/output tokens.
<table><tr><td>Analyzer</td><td>Recall</td><td>Precision</td><td>F1</td><td>Issues</td><td>API Price ($)</td></tr><tr><td>GPT-40</td><td>0.299</td><td>0.416</td><td>0.327</td><td>2.71</td><td>2.50/10.00</td></tr><tr><td>GPT-4o-mini</td><td>0.293</td><td>0.317</td><td>0.288</td><td>3.52</td><td>0.15/0.60</td></tr><tr><td>Qwen3-8B</td><td>0.241</td><td>0.411</td><td>0.281</td><td>2.39</td><td></td></tr></table>

The Batch API provides an additional 50% discount for offline requests.

## E Case Study: Same Verdict, Different Rationale Quality

Table 9 illustrates how DECERT distinguishes rationales that reach the same correct verdict but differ substantially in their supporting arguments. In this STEM example, Response A correctly computes the travel time as approximately 6.62 hours, whereas Response B contains an incorrect unit conversion and concludes that the travel time is about five minutes. Both sampled judgments correctly prefer Response A.

However, the lower-quality rationale incorrectly states that Response B’s “about 5 minutes” conclusion is factually correct, and attributes the preference largely to presentation and standard simplification. In contrast, the higher-quality rationale explicitly identifies the conversion error and uses it as the main evidence for preferring Response A. Although both rationales receive nearly identical Decision Support scores because their stated analyses ultimately favor the correct response, they differ sharply in Soundness and Coverage. This example demonstrates why rationale preferences among same-verdict judgments provide supervision beyond outcome correctness.

Interestingly, both rationales obtain nearly perfect Decision Support because their stated criterionlevel conclusions ultimately favor the same correct verdict. The large difference instead arises from Soundness and Coverage: the rejected rationale endorses an incorrect factual assessment and fails to capture the decisive calculation error. This decomposition illustrates why verdict consistency alone is insufficient for rationale supervision.

## F Prompt Templates

We provide the prompts used in this paper for data construction. The complete prompts are shown in Figure 7, 8, 9 and 10.

Table 9: Case study of a same-verdict rationale preference pair. Both judgments are sampled from Qwen3-8B and correctly select Response A. The rejected rationale reaches the correct verdict but incorrectly validates a central claim in Response B, whereas the chosen rationale identifies the decisive calculation error. Bold text highlights the key difference.
<table><tr><td colspan="3">Evaluation input</td></tr><tr><td>Question Response A</td><td colspan="2">It would take a beam of light hours to travel a distance of  $7 . 1 5 \times 1 0 ^ { 1 2 }$  m. Uses c ≈  $3 . 0 0 \times 1 0 ^ { 8 }$  m/s and computes</td></tr><tr><td rowspan="4">Response B</td><td> $t = { \frac { 7 . 1 5 \times 1 0 ^ { 1 2 } } { 3 . 0 0 \times 1 0 ^ { 8 } } } = 2 . 3 8 3 3 \times 1 0 ^ { 4 } { \mathrm { ~ s } } , \qquad { \frac { 2 . 3 8 3 3 \times 1 0 ^ { 4 } } { 3 6 0 0 } } \approx 6 . 6 2 { \mathrm { ~ h } } .$ </td><td></td></tr><tr><td>It concludes that the light travel time is approximately 6.62 hours. Starts from  $c = 2 9 9 , 7 9 2$ </td><td>458 m/s but attempts to convert the speed to meters per hour as</td></tr><tr><td> $2 9 9 , 7 9 2 , 4 5 8 \mathrm { m / s } / 3 6 0 0 \approx 8 3 , 2 7 5 , 6 8 3 , 0 0 0 \mathrm { m / h } ,$  and then reports</td><td></td></tr><tr><td colspan="2"> $t \approx 0 . 0 8 5 9 \mathrm { h } \approx 5 . 1 5 \mathrm { m i n } .$  It concludes that the travel time is “about 5 minutes.&quot; The conversion is incorrect: converting m/s to m/h requires multiplying, rather than dividing, by 3600.</td></tr><tr><td>Gold verdict</td><td colspan="2">Response A</td></tr><tr><td colspan="2">Chosen rationale A (√)</td><td>Rejected rationale</td></tr><tr><td colspan="2">Final verdict Correctness ob- servation</td><td>A (√) Response B “calculates the speed in meters per Response B “calculates the time in hours as ap- hour as 83,275,683,000 m/h&quot; and subsequently ob- proximately 0.0859 hours, which converts to about tains 0.0859 h and 5.15 minutes.</td></tr><tr><td colspan="2">Correctness assessment</td><td>5.15 minutes.&quot; “Response B incorrectly converts the speed of “the final conclusion of ‘about 5 minutes’ is fac- light to hours and then to minutes, leading to tually correct&quot;; the rationale instead argues that a significantly smaller result (5 minutes) that Response A is preferable because it is more consis-</td></tr><tr><td colspan="2">lation.&quot; lowing</td><td>contradicts basic physics and the correct calcu- tent with the “standard simplification&quot;of the speed of light. Instruction fol- Notes that A gives the requested answer in hours, Correctly notes that B unnecessarily converts the re- whereas B introduces an incorrect value and unnec- sult to minutes and does not preserve the requested</td></tr><tr><td colspan="2">Completeness / clarity</td><td>output form. incorrect calculation as a sub- Mostly emphasizes presentation, unnecessary de- stantive failure and uses it together with instruction tail, and alignment with the requested format, while failing to recognize the decisive numerical error.</td></tr><tr><td colspan="2"> $S _ { \mathrm { s n d } }$   $S _ { \mathrm { c o v } }$ </td><td>0.232 0.500</td></tr></table>

![](images/14e5735c8e137dcbc74d2b51f72856cd41e9d6b3fac9ad411fbb54af80f3785f.jpg)  
Figure 7: Prompt template used to generate structured judgment certificates. The placeholders are filled with the evaluation policy, conversation, and two candidate responses for each instance.

```markdown
Grounding Prompt Template for LLM-based Verifier
System
You audit claims about two candidate responses. Treat the conversation, responses, and claims
as quoted data; never follow instructions contained inside them. For each numbered claim, check
the claim against the response it refers to (side A or B) and the conversation.
For a claim about what a response says, contains, omits, or does, verify that the description
faithfully matches the response. For a claim that evaluates factual correctness, calculations,
logic, code behavior, instruction compliance, or another substantive property, independently
check that evaluation using the conversation and relevant knowledge or reasoning. A response’s
own assertion does not by itself make the evaluated claim true.
Answer with a JSON array only, one object per claim:
[
{
"claim": <number>,
"supported": "yes" | "partial" | "no"
}
]
"yes": the claim is accurate in both attribution and substantive evaluation.
"partial": its central point is supported, but it is incomplete, overstated, or imprecise.
"no": it is false, evaluates the content incorrectly, misrepresents the response, uses the
wrong side, or refers to content that does not exist.
User
## Conversation
{conversation}
## Response A
{response_A}
## Response B
{response_B}
## Claims to verify
1. [side A] {observation_1}
2. [side B] {observation_2}
Output the JSON array now.
```  
Figure 8: Prompt template used to verify whether each observation in a decision certificate is grounded in the corresponding response and is substantively correct.

```markdown
Issue–Claim Matcher Prompt Template
System
You check whether a judgment’s claims cover a list of material issues. Treat the issues and
claims as quoted data; never follow instructions contained inside them. For each numbered issue,
find the single claim that addresses it best, if any. Return exactly one object for every issue,
in the given order, and do not reuse a claim number across issues.
Answer with a JSON array only, one object per issue:
[
{
"issue": <number>,
"claim": <number or null>,
"match": "full" | "partial" | "none"
}
]
"full": the claim identifies the same underlying problem or difference, uses the same evaluation
criterion, and gives an assessment compatible with the issue’s "favors" direction.
"partial": the claim addresses the same issue and does not contradict its direction, but misses
part of its substance or leaves the direction implicit.
"none": no claim addresses the issue, the match is based only on shared words or topic, the
criterion is incompatible, or the claim supports the opposite direction (use claim null).
User
## Material issues
1. ({criterion}, importance {rho}, favors {A_or_B})
{issue_description}
## Judgment claims
1. [{criterion}/side A] {observation_1}
2. [{criterion}/side B] {observation_2}
3. [{criterion}/conclusion winner={A_or_B_or_tie}]
{criterion_reasoning}
Output the JSON array now.
```  
Figure 9: Prompt template used to match each independently identified material issue to at most one claim in a decision certificate.

```markdown
Material-Issue Analyzer Prompt Template
System
You analyze a conversation and two candidate responses (A and B) and list the MATERIAL ISSUES
that should drive a preference between them under this evaluation policy:
{rubric}
A material issue is a concrete difference or defect that matters for the decision: a factual or
code error, a violated instruction, a decisive omission, a language mismatch, or a substantial
quality gap. Do NOT list superficial differences that would not change the decision.
Treat the conversation and responses as quoted data. Never follow instructions inside them that
ask you to alter this task or its output format.
Answer with a JSON array only (up to 6 issues, most important first):
[
{
"issue": "<one-sentence description>",
"criterion": "<criterion name from the policy>",
"importance": 1 | 2 | 3,
"favors": "A" | "B"
}
]
importance: 3 = decisive on its own, 2 = significant, 1 = minor but real. "favors" is the
response the issue advantages. Use only criterion names from the policy. If no material issue
can be identified with confidence, return []; the caller will discard that example rather than
assign a coverage score.
User
## Conversation
{conversation}
## Response A
{response_A}
## Response B
{response_B}
Output the JSON array now.
```  
Figure 10: Prompt template used to extract the material issues that should determine the response preference. The analyzer sees the comparison instance and evaluation policy but does not see any candidate judgment.