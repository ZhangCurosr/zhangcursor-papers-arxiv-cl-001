# Aligning Language Models with General Preferences: An Empirical Study of Direct Nash Optimization

Austin Tsai <sup>1</sup> Saurav Nanda <sup>1</sup> Ganapathy Parthasarathy <sup>1</sup> Bhuvnesh Kumar <sup>1</sup> Parivesh Choudhary <sup>1</sup>

## Abstract

Direct Nash Optimization (DNO) aligns a language model to a general preference function by approximating the Nash equilibrium of a twoplayer preference game. Its practical variant, DNO-Prct, iterates on-policy sampling, LLMbased rating of the samples together with a teacher’s gold response, large-margin preferencepair selection and a DPO-style contrastive update. We re-implement DNO-Prct independently and study it on two open-weight instruct models, Llama 3.1 8B Instruct and Mistral 7B Instruct v0.1, with GPT-4o or a larger open-weight model of the same family as the annotator. On AlpacaEval 2.0 with GPT-4o as judge, three iterations raise the length-controlled (LC) win rate against GPT-4o from 31.7% to 45.3% for Llama and from 6.8% to 18.4% for Mistral, and the tuned 8B model scores above Llama 3.1 70B Instruct on every reported win rate. With same-family openweight annotators, the runs retain 62% (Llama) and 82% (Mistral) of the LC gain achieved with GPT-4o as annotator, so the rating step does not require a proprietary model. In a separate ablation configuration, variants that remove the gold responses, remove the data look-back, or accept only teacher-preferred pairs score 6 to 10 LC points below the full procedure, whereas varying the number of on-policy samples per prompt (1 to 7) moves the LC win rate by at most 2.1 points, and a margin sweep (1 to 5) run in its own pipeline by at most 2.4 points. We also report the cost of the pipeline and where the practical algorithm departs from the theory it approximates. All results are single runs, and GPT-4o is both the teacher and the evaluation judge; we discuss what this implies for the reported gains.

## 1. Introduction

Pre-trained large language models (LLMs) have advanced rapidly (Radford et al., 2018; 2019; Brown et al., 2020; Rae et al., 2021). Trained only to predict the next token, they are not by themselves aligned with what humans prefer. Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023; OpenAI, 2023) bridges this gap by collecting human preferences, training a reward model on them, and optimizing the policy with reinforcement learning. Single-stage contrastive methods such as Direct Preference Optimization (DPO; Rafailov et al., 2023) remove the explicit reward model. In their standard formulations, both assume a pointwise (Bradley– Terry) reward and therefore cannot represent intransitive preferences, which Liu et al. (2025) show arise with high probability under a Luce-type preference model; they are also vulnerable to reward hacking (Amodei et al., 2016), and a reward model or preference dataset fixed in advance becomes stale as the policy distribution shifts during training (Ross et al., 2011; Cheng et al., 2024; Azar et al., 2024; Munos et al., 2024).

To address these limitations, a line of work optimizes the general preference function directly by finding the Nash equilibrium of a two-player game, solving that game with single-player (self-play) algorithms that exploit the symmetry of the preference function (Munos et al., 2024; Swamy et al., 2024). For a preference function with $\mathcal { P } ( y \succ y ^ { \prime } \mid x ) + \mathcal { P } ( y ^ { \prime } \succ y \mid x ) = 1$ and $\begin{array} { r } { \mathcal { P } ( \pi \succ \pi ^ { \prime } ) : = } \end{array}$ $\mathbb { E } _ { x \sim \rho , y \sim \pi ( \cdot | x ) , y ^ { \prime } \sim \pi ^ { \prime } ( \cdot | x ) } [ \mathcal { P } ( y ~ \succ ~ y ^ { \prime } ~ | ~ x ) ]$ for a prompt distribution $\rho , \ \mathrm { a }$ Nash equilibrium is a policy $\pi ^ { \star }$ with $\begin{array} { r } { \mathcal { P } ( \pi ^ { \star } \succ \pi ) \ge \frac { 1 } { 2 } } \end{array}$ for every competing policy π in the policy class Π; this policy is the Minimax Winner of the game. Rosset et al. (2024) propose Direct Nash Optimization (DNO): a provable algorithm whose iterates converge to the Nash equilibrium on average, together with a practical variant, DNO-Prct, which in their largest run, on about ten times more prompts over six iterations, brought a 7B model to a 33% length-controlled AlpacaEval 2.0 win rate against GPT-4 Turbo, with GPT-4 Turbo as annotator and gold-response source. DNO-Prct bundles several design choices: batched on-policy sampling, a teacher’s gold response among the candidates, LLM-based rating against a rubric, a minimum rating margin, and a decaying look-back over earlier pairs.

To our knowledge no independent re-implementation of DNO-Prct has been published: its published results, including its ablations, use one base model and a proprietary annotator, so it is not known whether its gains and design choices transfer to other base models and annotators. This paper is an empirical study of DNO-Prct. We re-implemented the algorithm from the description in Rosset et al. (2024), ran it on two open-weight instruct models with GPT-4o and with same-family open-weight models as annotators, and examined its main design choices. Throughout, “DNO” refers to DNO-Prct unless stated otherwise. Our contributions are as follows.

(1) An independent re-implementation with a procedural description: We give the pair-construction procedure as pseudo-code (Appendix C), the rating prompt verbatim (Appendix A) and the hyperparameters and implementation settings (Appendix B). Together with Rosset et al. (2024), this describes the pipeline in enough detail to re-implement it.

(2) Gains on two model families: Over three iterations the student’s length-controlled (LC) AlpacaEval 2.0 win rate against GPT-4o rises from 31.71% to 45.32% for Llama 3.1 8B Instruct (Grattafiori et al., 2024) and from 6.77% to 18.44% for Mistral 7B Instruct v0.1 (Jiang et al., 2023). Under the same protocol, the tuned 8B model scores above Llama 3.1 70B Instruct, which has nearly nine times as many parameters, on all four reported win rates, and the tuned 7B model narrows its gap to Mixtral 8x7B Instruct v0.1 (Jiang et al., 2024) on all four, to between 0.2 and 12.8 points (Section 5.3).

(3) Open-weight annotators: Replacing GPT-4o with Llama 3.1 70B Instruct or Mistral Large 2 as the annotator retains 62% (Llama) and 82% (Mistral) of the LC gain against the GPT-4o opponent; against GPT-4-0613 the Llama run retains 45% and the Mistral run matches the GPT-4o-annotated run, so the rating step does not require a proprietary model (Section 5.4).

(4) Which components matter: In a separate ablation configuration, variants that remove the gold responses, remove the data look-back, or restrict pair selection to teacherpreferred pairs score 6 to 10 LC points below the full procedure against GPT-4o (9 to 19 points against GPT-4-0613), whereas the number of on-policy samples per prompt (1 to 7) moves the LC win rate against GPT-4o by at most 2.1 points, and a margin sweep (1 to 5) run in its own pipeline by at most 2.4 points (Section 6). We also document the cost of the pipeline (Section 7) and the places where DNO-Prct departs from the theory it approximates (Section 6.6).

Rosset et al. (2024, Section 5.2) compared large-margin filtering, on-policy sampling and the SPIN and DNO-Restrictive pair rules in their own setting, and their Table 2 tracks teacher- and student-preferred pairs and the average teacher and student ratings across iterations. We extend these analyses beyond that setting, to two further base models and to open-weight annotators, add sweeps of the rating margin and the number of samples, and repeat their teacher– student analysis (Section 6.5). The experiments were run in 2024, and Section 3 relates them to work that appeared during or after them, notably on preference leakage between data generator and judge (Li et al., 2026) and on the exploitability of AlpacaEval (Zheng et al., 2025).

## 2. Preliminaries: Direct Nash Optimization

There are two variants of Direct Nash Optimization: the theoretical DNO (Algorithm 1) and the practical DNO-Prct. Rosset et al. (2024) define the reward of a response at iteration t as its expected win rate against the current policy’s own samples under the preference function $\mathcal { P } \colon$

$$
r _ { t } ( x , y ) : = \mathbb { E } _ { y ^ { \prime } \sim \pi _ { t } ( \cdot | x ) } \big [ \mathcal { P } ( y \succ y ^ { \prime } \mid x ) \big ] ,\tag{1}
$$

for prompts $x \in \mathcal { X }$ and responses $y \in \mathcal { V }$ . DNO aligns the implicit reward used by DPO with this preference-based reward. The implicit reward of a policy π relative to $\pi _ { t }$ is

$$
\begin{array} { r } { r _ { \pi , t } ( x , y ) : = \eta \log \frac { \pi ( y \mid x ) } { \pi _ { t } ( y \mid x ) } + \eta \log Z _ { t } ( x ) , } \\ { Z _ { t } ( x ) : = \displaystyle \sum _ { y \in \mathcal { V } } \pi _ { t } ( y \mid x ) \exp \Bigl ( \frac { r _ { t } ( x , y ) } { \eta } \Bigr ) , } \end{array}\tag{2}
$$

where $\eta > 0$ is the coefficient of the KL penalty in the objective $\mathcal { P } ( \pi \succ \pi _ { t } ) - \eta \mathbb { E } _ { x \sim \rho } [ D _ { \mathrm { K L } } ( \pi \| \pi _ { t } ) ]$ ], whose maximizer is the soft-policy-iteration update

$$
\pi _ { t + 1 } ^ { \star } ( y \mid x ) = \pi _ { t } ( y \mid x ) \exp \big ( r _ { t } ( x , y ) / \eta \big ) / Z _ { t } ( x ) ;
$$

Rosset et al. (2024) call $\eta$ the learning rate. In Eq. (5) below it plays the role of $\beta$ in DPO (in DNO-Prct, $\beta$ corresponds to a rescaled $\tilde { \eta } ;$ Section $6 . 6 ) $ , and it is distinct from the optimizer’s learning rate. DNO minimizes a cross-entropy loss between the predicted preference $\sigma ( r _ { \pi , t } ( x , y _ { 1 } ) - r _ { \pi , t } ( x , y _ { 2 } ) )$ and the target preference $\sigma ( r _ { t } ( x , y _ { 1 } ) - r _ { t } ( x , y _ { 2 } ) )$ , with $\sigma ( z ) : = 1 / ( 1 { + } e ^ { - z } )$ . Writing

$$
\Delta _ { \pi , t } ( x , y _ { 1 } , y _ { 2 } ) : = r _ { \pi , t } ( x , y _ { 1 } ) - r _ { \pi , t } ( x , y _ { 2 } )
$$

$$
\begin{array} { r } { = \eta \log \frac { \pi ( y _ { 1 } | x ) } { \pi _ { t } ( y _ { 1 } | x ) } - \eta \log \frac { \pi ( y _ { 2 } | x ) } { \pi _ { t } ( y _ { 2 } | x ) } , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \Delta _ { t } ^ { \star } ( x , y _ { 1 } , y _ { 2 } ) : = r _ { t } ( x , y _ { 1 } ) - r _ { t } ( x , y _ { 2 } ) , } \end{array}\tag{4}
$$

in which the partition function cancels, the update at iteration t is

$$
\begin{array} { r l } & { \pi _ { t + 1 } = \underset { \pi \in \Pi } { \arg \operatorname* { m a x } } ~ \mathbb { E } _ { \mathcal { D } _ { t } } \Big [ \sigma ( \Delta _ { t } ^ { \star } ) \log \sigma ( \Delta _ { \pi , t } ) } \\ & { \qquad + \sigma ( - \Delta _ { t } ^ { \star } ) \log \sigma ( - \Delta _ { \pi , t } ) \Big ] , } \end{array}\tag{5}
$$

Algorithm 1 Direct Nash Optimization (adapted from Ros  
set et al., 2024)   
Require: preference function ${ \mathcal P } ,$ KL coefficient $\eta ,$ itera  
tions $N _ { \ast }$ , prompt distribution $\rho$   
1: Initialize $\pi _ { 1 }  \mathrm { u n i f } ( \mathcal { V } )$   
2: for $t = 1 , \ldots , N$ do   
3: Compute $r _ { t } ( x , y )$ by Eq. (1) for all $( x , y ) \in \mathcal { X } \times \mathcal { Y } .$   
4: Obtain $\pi _ { t + 1 }$ by $\operatorname { E q . } ( 5 ) ,$ , where $\mathcal { D } _ { t }$ is drawn as $x \sim \rho ,$   
$y _ { 1 } \sim \mu _ { 1 , t } ( \cdot \mid x )$ , y<sub>2</sub> $\sim \mu _ { 2 , t } ( \cdot \mid x )$ with $\mu _ { 1 , t } , \mu _ { 2 , t }$   
off-policy or on-policy (π<sub>t</sub>).   
5: end for   
6: return ${ \bar { \pi } } = \operatorname { u n i f } \left( \pi _ { 1 : N } \right)$

where the arguments $( x , y _ { 1 } , y _ { 2 } )$ are omitted, the expectation is over $( x , y _ { 1 } , y _ { 2 } ) \sim \mathcal { D } _ { t } ,$ , and the second term is the first with $y _ { 1 }$ and $y _ { 2 }$ exchanged.

Guarantees: Each exact iteration is one step of soft policy iteration, a no-regret update, so the uniform mixture π¯ of the iterates converges to the Nash equilibrium of the two-player preference game, the Minimax Winner (Kreweras, 1965; Simpson, 1969; Kramer, 1973; Fishburn, 1984); the mixture argument is due to Swamy et al. (2024), and Munos et al. (2024) establish last-iterate convergence for the regularized game. Rosset et al. (2024) also establish two further properties (their Section 3.1 and Theorem 1). The exact soft-policy-iteration update improves monotonically, $\begin{array} { r } { \mathcal { P } ( \pi _ { t + 1 } ^ { \star } \succ \pi _ { t } ) \ge \frac { 1 } { 2 } + \eta \mathbb { E } _ { x \sim \rho } [ D _ { \mathrm { K L } } ( \pi _ { t + 1 } ^ { \star } \| \pi _ { t } ) ] } \end{array}$ , and their Theorem 1 bounds the expected squared total-variation distance between the learned $\pi _ { t + 1 }$ and $\pi _ { t + 1 } ^ { \star }$ in terms of the sample size, the size of the policy class, a boundedness constant $R _ { \mathrm { m a x } }$ and a data-dependent concentrability coefficient, so the improvement carries over up to that error. These statements hold under the assumptions of the cited results (realizability of the soft-policy-iteration target in the policy class, and boundedness), and the bound is informative only when the concentrability coefficient is small; neither condition is guaranteed for a neural policy class. The improvement is against the previous iterate under $\mathcal { P }$ and $\rho ;$ it is not a statement that benchmark scores rise at every iteration or that the student surpasses a fixed teacher. The practical procedure below, with finite samples, LLM ratings, margin filtering, look-back and a single selected checkpoint, approximates the exact update and does not inherit these guarantees.

DNO-Prct: The main obstacle in practice is the expectation in Eq. (1). Rosset et al. (2024) therefore propose DNO-Prct, each iteration of which has three steps:

1. Batched on-policy sampling and rating: K outputs per prompt are sampled from the current policy $\pi _ { t }$ and, together with a gold response $y ^ { \mathrm { g o l d } }$ from a stronger teacher model, are scored by an annotator; the scores serve as a proxy for the pairwise win-rate ranking under $\mathcal { P }$

2. Preference pair construction: Only pairs whose scores differ by a sufficiently large margin are kept, with the intent that the target $\sigma ( \Delta _ { t } ^ { \star } )$ , after a rescaling of $r _ { t } .$ , be close to one so that Eq. (5) reduces approximately to the pure contrastive (DPO) loss (an approximation discussed in Section $6 . 6 ) ;$ a decaying fraction of pairs from earlier iterations is mixed in (data look-back, described in Section 5.1 of Rosset et al. (2024) rather than in their Algorithm 2).

3. Contrastive update: The policy is trained with the DPO objective on the constructed pairs, with $\pi _ { t }$ as the reference policy.

DNO-Prct is thus an iterative, on-policy form of DPO with the same contrastive objective; the difference lies in how the preference pairs are constructed (Figure 1 in Section 4.2). We refer readers to Rosset et al. (2024) for the derivation.

## 3. Related Work

RLHF and offline contrastive alignment: The rewardmodel-plus-RL recipe (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022) is a standard route to aligned chat models. Offline contrastive methods such as DPO (Rafailov et al., 2023), IPO (Azar et al., 2024) and SimPO (Meng et al., 2024) fit the policy directly to a preference dataset; Zephyr (Tunstall et al., 2023) popularized distilled supervised fine-tuning (SFT) on UltraChat followed by DPO on UltraFeedback (Cui et al., 2024). DPO and SimPO inherit the transitivity of a pointwise (Bradley–Terry) reward, which IPO avoids, and DPO in particular is known to exploit response length (Park et al., 2024). Meng et al. (2024) report that Llama-3-8B-Instruct trained with SimPO reaches 44.7% LC on the public AlpacaEval 2.0 leaderboard, above Llama-3-70B-Instruct (34.4%), so a small model scoring above a larger sibling on this metric is not by itself evidence of a better algorithm.

Iterative, online and self-play alignment: Iterative DPO (Xu et al., 2023; Xiong et al., 2024; Dong et al., 2024), online AI feedback (Guo et al., 2024) and Self-Rewarding LMs (Yuan et al., 2024) regenerate preference data from the current policy each round. Tang et al. (2024) find that the advantage of online over offline alignment is not explained by data coverage or quality alone and point to on-policy sampling as the deciding factor; Lanchantin et al. (2025) find that semi-online training, in which the response generator is synchronized with the policy only every few updates, as in DNO-Prct’s per-iteration sampling, performs on par with fully online training and well above offline methods. SPIN (Chen et al., 2024) treats the SFT targets as always preferred over self-generated responses. Game-theoretic methods seek the Nash equilibrium of the preference game: Nash-MD (Munos et al., 2024), online IPO (Calandriello et al., 2024), SPO (Swamy et al., 2024), SPPO (Wu et al.,

2025), INPO (Zhang et al., 2025b), DNO (Rosset et al., 2024) and, concurrently with or after our experiments, MPO (Wang et al., 2025), optimistic online mirror descent (Zhang et al., 2025a) and RSPO (Tang et al., 2026). REBEL (Gao et al., 2024) regresses relative rewards and extends to intransitive preferences, and Zhang et al. (2025c) learn general preference models beyond Bradley–Terry for such methods. INPO reaches 42.6% LC on AlpacaEval 2.0 from a Llama-3-8B SFT model with an open-weight pairwise preference model and no proprietary model in the loop. We have not found a published independent re-implementation of DNO-Prct on other base models, nor a public implementation of it.

Where do the gains of AI feedback come from? Sharma et al. (2024) show that improvements attributed to reinforcement learning from AI feedback (RLAIF) are largely explained by using a weaker teacher for SFT than the critic used for feedback, and that SFT directly on GPT-4 completions outperforms the pipelines they study. Our goldresponse ablation (Section 6.3) is related but not a replication of their comparison: we did not run a matchedteacher SFT control. Ivison et al. (2024) disentangle preference data, learning algorithm, reward model and policy prompts in a comparable empirical study of DPO and PPO. Open-weight models trained specifically as evaluators can match or exceed GPT-4 on reward-model benchmarks (Wang et al., 2024); our open-weight annotators were not evaluator-trained and retained 62–82% of the LC gain of the GPT-4o-annotated runs against GPT-4o (Section 5.4).

LLM-as-a-judge evaluation and its failure modes: We evaluate with AlpacaEval 2.0 (Li et al., 2023) and its lengthcontrolled win rate (Dubois et al., 2024), which with the default judge and reference correlates strongly with the rankings of Chatbot Arena (Chiang et al., 2024); our judge and references are not the default (Section 4.1). LLM judges exhibit position, verbosity and self-preference biases (Zheng et al., 2023; Panickssery et al., 2024; Wataoka et al., 2024). Li et al. (2026) find that a judge favors students trained on data from a related model: their average preference-leakage score is about 24% when judge and data generator are the same model, as GPT-4o is in our main runs, and about 3–9% when they share a model family. Zheng et al. (2025) show that a null model that always returns the same, adversarially crafted response reaches 86.5% LC on AlpacaEval 2.0. Both appeared during or after our experiments, and Section 8 uses them to interpret our results.

## 3.1. Relation to Online DPO

DNO is conceptually close to online DPO (Guo et al., 2024). In each round of online DPO the model generates two candidate responses to a prompt, an annotator decides which is better, and the resulting pair is used for a DPO-style update.

DNO likewise samples candidates, rates them and learns from the resulting preferences, but differs in three ways.

(1) Rating guideline and margin: DNO rates responses against an explicit rubric and forms a pair only when the positive–negative rating margin is sufficiently large. The margin filter is what allows the regression objective of Eq. (5) to be approximated by the pure contrastive loss without an excessively large KL coefficient (Rosset et al., 2024, Section 4); it is an approximation device rather than a condition for Nash convergence. Its empirical effect is studied in Section 6.2.

(2) Multiple on-policy candidates: Instead of a single pair per prompt, DNO samples K candidates per prompt, intended to cover more of the policy’s output distribution than a single pair. Section 6.1 varies K.

(3) Teacher responses as candidates: DNO adds the teacher’s response $y ^ { \mathrm { g o l d } }$ to the candidate set. Rating teacher and student responses together captures cases where the student outperforms the teacher, which SPIN-style self-play ignores (Sections 6.3 and 6.4).

Online DPO as instantiated by Guo et al. (2024) additionally keeps a fixed reference policy and samples pairs per minibatch, whereas DNO-Prct collects a large batch per iteration, resets the reference to π , and mixes in look-back data.

## 4. Experimental Setup

## 4.1. General Setup

Timeline: All experiments were run between the release of Llama 3.1 and Mistral Large 2 in late July 2024 and the end of December 2024, with the model and API versions available then. All reported results come from these 2024 runs; a few descriptive statistics, such as the prompt overlap and the truncation rate, were computed afterwards from their stored outputs.

Dataset: We use the UltraFeedback dataset (Cui et al., 2024; OpenBMB, 2023) as the fine-tuning and preferencealignment dataset. UltraFeedback comprises 63,967 prompts, four responses per prompt from a pool of seventeen models (including GPT-4, GPT-3.5 Turbo, Bard, Llama-2-70B-chat, Vicuna-33B and Falcon-40B-Instruct), and GPT-4 annotations along four aspects (instruction following, truthfulness, honesty and helpfulness), each on a 1–5 scale, together with an overall score on a 1–10 scale. The prompts are sourced from UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA and FLAN. The SFT and offline-DPO thresholds below are applied to the average of the four per-aspect ratings. In addition to the original content, we generate one response from GPT-4o (OpenAI, 2024) for every prompt to serve as the gold response $y ^ { \mathrm { g o l d } }$ in the GPT-4o-annotated runs.

Base models and annotators: We experiment with two base models, Llama 3.1 8B Instruct and Mistral 7B Instruct v0.1. The base model provides the initial weights for every run except the two-stage runs of Section 5.5 and the look-back ablation of Section 6.4, which start from an SFT checkpoint. GPT-4o is the annotator in the DNO experiments except those of Section 5.4, where the annotator is Llama 3.1 70B Instruct or Mistral Large 2 (Mistral-Large-Instruct-2407, 123B parameters; hereafter Mistral Large; Mistral AI, 2024).

Baselines: We compare DNO with our own implementations of supervised fine-tuning (SFT), offline DPO (Rafailov et al., 2023), a SPIN-style baseline (Chen et al., 2024), DNO-Restrictive, SPPO (Wu et al., 2025) and online DPO (Guo et al., 2024; Dong et al., 2024). Our SPIN-style baseline marks the teacher’s gold response as always preferred over the on-policy samples, which is the “mark the teacher (T) always positive” rule of Section 6.4; it is not the original SPIN recipe, which uses the SFT targets as winners. DNO-Restrictive accepts only pairs whose positive is the teacher’s response. SPPO and online DPO were run on UltraFeedback with only the learning-rate search described below; both ended below the base model, and we therefore draw no conclusions about those methods from our runs.

Pre-processing per algorithm: For SFT, we keep as targets the UltraFeedback responses whose average rating is five, the maximum, forming a question-answering dataset in OpenAI’s conversation format with the “user” role for the prompt and the “assistant” role for the response. For offline DPO, we mark responses with an average rating of five as positive and responses rated at least two points lower as negative, and collect all positive–negative pairs per prompt; the prompt takes the “user” role followed by a generation token, and each chosen or rejected response is followed by an end-of-sequence or end-of-turn token. DNO uses only the prompts and the gold responses; on-policy responses are generated on the fly. We split the prompts into three disjoint partitions, one per DNO iteration, so that no prompt is used for new candidate generation in more than one iteration; pairs built in earlier iterations may be replayed through look-back (Section 4.2).

Training setup: We follow most hyperparameters of the Llama recipes (Meta AI, 2024). The batch size is 2 per device with 4 gradient-accumulation steps on four NVIDIA A100-SXM 80 GB GPUs, an effective batch size of 32 for all experiments, and training uses low-rank adaptation (LoRA; Hu et al., 2022) (rank 16, α = 32, dropout 0.05; Appendix B). We train SFT for 20 epochs (reporting the epoch-3 and epoch-10 checkpoints), DPO for 3 epochs, and DNO for three iterations, each a single epoch over that iteration’s training pairs, to limit overfitting to them. Each iteration consists of on-policy generation, rating, large-margin pair selection and the contrastive update with the DPO loss $( \beta = 0 . 1 )$ . It starts from the merged checkpoint of the previous iteration, and the reference policy is that checkpoint with the adapters disabled, i.e., π , as DNO-Prct prescribes. All algorithms are implemented with the Hugging Face TRL library (von Werra et al., 2020). All runs use seed 2024.

Learning rate: We search over $\{ 5 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 6 } \}$ during the first epoch (first iteration for DNO) of every experiment and keep the better rate for the remaining epochs or iterations; the rate is $5 \times 1 0 ^ { - 5 }$ for SFT and $5 \times 1 0 ^ { - 6 }$ for DPO and most DNO experiments. The two rates differed by 5 to 10 points in win rate against GPT-4-0613, so in our runs the choice of learning rate had a large effect.

Evaluation: One objective of this study is to see how far a small model can be pushed relative to a larger model of its family, so we evaluate all models, including the larger ones, under one pipeline: AlpacaEval 2.0 (v0.6.5; Li et al., 2023) on its 805 prompts, reporting the raw and the length-controlled (LC) win rate (Dubois et al., 2024). The protocol departs from the public leaderboard in three ways. (i) Opponents: instead of gpt4 1106 preview, every model is compared with two opponents, GPT-4o and GPT-4-0613, whose reference outputs we generated through Azure OpenAI (temperature 0, at most 4,096 tokens). (ii) Judge: the judge is GPT-4o, prompted with AlpacaEval’s alpaca eval gpt4 pairwise-ranking template at temperature 0, rather than the leaderboard’s default GPT-4 Turbo annotator; GPT-4o is therefore also the teacher and one of the opponents (Section 8). (iii) Decoding: each evaluated model answers with vLLM (Kwon et al., 2023), its chat template, temperature 0.001 and at most 4,096 new tokens. AlpacaEval computes the length-controlled win rate relative to whichever reference outputs it is given, so LC values are defined for both opponents, but neither LC nor raw values are comparable to the public leaderboard or to numbers in other papers, and the reported correlation of LC with Chatbot Arena (Spearman 0.98; Dubois et al., 2024) was established for the default judge and reference, not for this protocol.

Uncertainty: The standard error of the raw win rate reported by AlpacaEval is at most 1.9 points per evaluation (1.76 for the Llama Iteration-3 model against GPT-4o). Its standard error for the LC win rate (at most 0.26) is an error of the regression fit, not a sampling error, so we do not use it to compare models. These are evaluation uncertainties for single models; a comparison between two models on the shared prompts would need a paired analysis, which we have not done, and where we relate a difference to standard errors we treat the two evaluations as independent. The variation between training runs is unknown, because each configuration was trained once. We therefore read differences of a few points between reported runs as suggestive rather than established, and we do not claim that two configurations are equivalent when their scores are close.

## 4.2. On-Policy Data Construction in DNO Iterations

Throughout the paper we call the annotator the teacher (T) and the model being fine-tuned the student (S). In the GPT-4o runs the teacher also supplies the gold response $y ^ { \mathrm { g o l d } }$ . Figure 1 summarizes one iteration. In every block of experiments the candidates are sampled from the student $\pi _ { t }$ and the evaluation judge is GPT-4o; in the open-weightannotator runs (Section 5.4), Llama 3.1 70B Instruct or Mistral Large replaces GPT-4o as the annotator. A pair whose positive is the gold response $y ^ { \mathrm { g o l d } }$ (the teacher side, T) and whose negative is a student response is written $T \succ$ S, and likewise $S \succ T$ and $S \succ S$

On-policy response generation: At the start of each iteration we sample $K = 5$ responses per prompt (varied in Section 6.1) from the current model, the initial checkpoint in the first iteration and the model from the previous iteration thereafter, with temperature 0.7 and nucleus sampling $( \mathrm { t o p } { - } p , p = 0 . 9 5 ;$ Holtzman et al., 2020).

Rating: For the $K + 1$ responses per prompt (K on-policy responses and $y ^ { \mathrm { g o l d } } { \mathrm { ; } }$ six in the main runs), the teacher rates quality following the preference-ranking approach of Rosset et al. (2024), with an additive rubric of six criteria whose integer scores range from 0 to 6 (Appendix A). Our prompt modifies the original, which was written for GPT-family annotators, so that it also works with Mistral Large and Llama 3.1 70B Instruct. Each response is evaluated along dimensions such as relevance, correctness and conciseness. For each prompt all candidates are sent to the annotator in one query, in a fixed order with $y ^ { \mathrm { g o l d } }$ last, and scored side by side; the annotator returns one score per candidate. Candidates containing certain degenerate character sequences are dropped before rating, and a reply that cannot be parsed or does not contain one score per candidate is discarded.

Preference pair construction: We select pairs $( x , y ^ { + } , y ^ { - } )$ whose rating margin rating(y<sup>+</sup>)−rating(y−) is sufficiently large. We mark all responses with a score of at least five as positives. For each positive, responses rated at least two points lower are negatives, and we form all positive– negative pairs. If the positive is from the student and the negative is the teacher’s response, we relax the margin to one point, because GPT-4o tends to rate its own responses highly (Panickssery et al., 2024; Wataoka et al., 2024); whether such a gap reflects judge bias or a real quality difference cannot be determined from the scores alone (Chen et al., 2025). This is a deviation from Rosset et al. (2024), who relax the margin whenever the positive is from the student, including $S \succ S$ pairs, on the grounds that their teacher’s responses rarely scored below five. Following the original paper, we include part of the data from past iterations to preserve earlier preference behavior (data look-back): at iteration t we sample at most 30% of the training pairs of iteration $t - 1$ , 15% of those of iteration t − 2, and so on. These pairs are reused as they are, without re-rating. Appendix C gives the procedure as pseudo-code.

## 5. Main Results

We report DNO and the baselines on two base models. Unless stated otherwise, GPT-4o is both the annotator and the source of $y ^ { \mathrm { g o l d } }$ . The DNO rows of Tables 1 and 2 are the canonical runs that the direct-DNO numbers in this section refer to; the ablations of Section 6 were run in a separate configuration and are compared with other runs only where stated; in particular, the SPIN-style and DNO-Restrictive rows of Table 1 are those ablation runs, and the two-stage runs of Section 5.5 were trained in the ablation configuration.

## 5.1. Llama 3.1 8B Instruct

Table 1 presents the main results with Llama 3.1 8B Instruct as the base model. DNO Iteration 3 performs best, with an LC win rate against GPT-4o that is 8.30 points above offline DPO (45.32% vs. 37.02%) and 11.59 points above SFT at epoch 3 (33.73%). SFT degrades with more epochs (24.01% at epoch 10, below the base model at 31.71%); a plausible explanation, which we have not tested with a matched-data control, is that its targets are UltraFeedback responses from 2023-era models, many of them weaker than the instruct base model. A single DNO iteration (39.01%) already scores 1.99 points above three epochs of offline DPO (37.02%), and the second (41.57%) is 4.55 points above it. Against GPT-4-0613 the three DNO iterations keep the same monotone ordering on both metrics, but offline DPO has a higher LC win rate (66.40%) than DNO Iterations 1 and 2 (61.44% and 64.87%), and it is the only model in Table 1 whose LC win rate against this opponent exceeds its raw win rate (54.12%). The baselines are also not consistently ordered: offline DPO is above SFT (epoch 3) on LC win rate against GPT-4o (37.02% vs. 33.73%) but below it on raw win rate against GPT-4-0613 (54.12% vs. 63.35%).

Responses grow longer with every DNO iteration (3,281 to 3,786 characters on average). At Iteration 3 the median is 2,796 characters, and about 5% of the responses reach or nearly reach the 4,096-token generation limit, so the average is raised by a small share of very long outputs, some of them truncated. The contrastive objective can widen the margin between positive and negative responses by lengthening outputs (Park et al., 2024), iterative DPO with preferences labeled by a trained reward model increases verbosity (Liu et al., 2024), and Rosset et al. (2024, Section 5.2) report a length increase of at least a factor of two in their own DNO runs. Because our rating step has no explicit length control, part of the raw-win-rate gain may be due to length. The LC win rate also rises with every iteration. This is consistent with more informative responses but does not establish it, because length control is a regression adjustment. For Mistral the length growth is far larger (Section 5.2); we return to length in Section 8.

![](images/1ebd96788f1c87e1abe45313e83d6dd0efecc5534628fa1180b470dfe630b915.jpg)  
Figure 1. One DNO-Prct iteration (Algorithm 2). Blue: training-time steps performed by GPT-4o in the main runs. Look-back pairs are reused without re-rating; LoRA adapters trained on π<sub>t</sub> are merged to give π<sub>t+1</sub>.

Table 1. Llama 3.1 8B Instruct: DNO versus our reproduced baselines, with GPT-4o as annotator and gold-response source. AlpacaEval 2.0 LC / raw win rate (%) against two opponents, judged by $\mathrm { G P T - 4 o } ,$ and average length. The SPIN-style and DNO-Restrictive rows are ablation runs (Table 7; full-DNO reference 43.90% LC). Bold: best DNO value per column.
<table><tr><td>Algorithm</td><td>Epoch or iter.</td><td>LC / raw win rate (%) vs. GPT-40</td><td>LC / raw win rate (%) vs. GPT-4-0613</td><td>Avg. length (chars)</td></tr><tr><td>DNO</td><td>Iter. 1</td><td>39.01 / 43.04</td><td>61.44 / 68.45</td><td>3281</td></tr><tr><td>DNO</td><td>Iter. 2</td><td>41.57 / 44.66</td><td>64.87 / 71.37</td><td>3626</td></tr><tr><td>DNO</td><td>Iter. 3</td><td>45.32 / 48.26</td><td>70.18 / 74.60</td><td>3786</td></tr><tr><td>DNO-Restrictive</td><td>Iter. 3</td><td>37.98 / 38.62</td><td>60.16 / 60.51</td><td>4802</td></tr><tr><td>SPIN-style</td><td>Epoch 3</td><td>35.17 / 41.99</td><td>50.30 / 66.81</td><td>3207</td></tr><tr><td>SPPO</td><td>Epoch 3</td><td>31.28 / 34.22</td><td>45.85 / 59.44</td><td>2241</td></tr><tr><td>Online DPO</td><td>Epoch 3</td><td>29.92 / 32.55</td><td>42.21 / 58.14</td><td>2076</td></tr><tr><td>Offline DPO</td><td>Epoch 3</td><td>37.02 / 37.58</td><td>66.40 / 54.12</td><td>2173</td></tr><tr><td>SFT</td><td>Epoch 3</td><td>33.73 /35.47</td><td>50.86 / 63.35</td><td>2417</td></tr><tr><td>SFT</td><td>Epoch 10</td><td>24.01 / 23.98</td><td>39.64 / 49.88</td><td>2076</td></tr><tr><td>Llama 3.1 8B Instruct (base model)</td><td></td><td>31.71 /35.51</td><td>49.64 / 62.19</td><td>2237</td></tr><tr><td>Llama 3.1 70B Instruct (larger model)</td><td></td><td>41.40 / 39.94</td><td>55.23 / 67.64</td><td>1873</td></tr></table>

## 5.2. Mistral 7B Instruct v0.1

Table 2 presents the comparison with Mistral 7B Instruct v0.1. The raw win rates that Rosset et al. (2024) report for their Mistral-based run are listed alongside for reference, but they are not comparable with ours: they were measured against GPT-4 Turbo with a GPT-4 Turbo annotator, starting from Orca-2.5, an SFT variant of Mistral 7B.

Their Iteration 3, offline DPO and SFT-on-positives values are from their Table 1, where the baselines are reported at epoch 4; the values for Iterations 1 and 2, which their tables do not list, are approximate. As with Llama 3.1, the LC win rate against GPT-4o rises with every iteration, and Iteration 3 scores above offline DPO by 11.39 points and above SFT by 13.07 points in LC win rate against GPT-4o (18.44% vs. 7.05% and 5.37%), an 11.67-point gain over the base model. Against GPT-4-0613, both metrics decline slightly after Iteration 2 (LC 38.67% to 38.20%; raw 40.17% to 37.85%), whereas against GPT-4o they keep rising; whether this is saturation or run noise cannot be determined from single runs. One untested hypothesis is that the GPT-4o gold responses pull the student toward GPT-4o-like outputs, which the judge may score differently when the opponent is GPT-4o itself than when it is GPT-4-0613. The average response length grows to 7,506 characters, more than five times the base model’s.

Table 2. Mistral 7B Instruct v0.1: DNO versus our reproduced baselines. Columns as in Table 1; the Rosset et al. column lists raw win rates from their own Mistral-based run, for reference only (∼: approximate; see Section 5.2). Bold: best DNO value per column.
<table><tr><td>Algorithm</td><td>Epoch or iter.</td><td>LC / raw win rate (%) vs. GPT-40</td><td>LC / raw win rate (%) vs. GPT-4-0613</td><td>Reported by Rosset et al. (%)</td><td>Avg. length (chars)</td></tr><tr><td>DNO</td><td>Iter. 1</td><td>9.94 / 10.64</td><td>33.62 / 35.12</td><td>~10</td><td>2424</td></tr><tr><td>DNO</td><td>Iter. 2</td><td>13.62 / 13.45</td><td>38.67 / 40.17</td><td>~21</td><td>5365</td></tr><tr><td>DNO</td><td>Iter. 3</td><td>18.44 / 18.53</td><td>38.20 / 37.85</td><td>24.97</td><td>7506</td></tr><tr><td>Offline DPO</td><td>Epoch 3</td><td>7.05 / 5.85</td><td>23.27 / 24.63</td><td>18.22</td><td>1465</td></tr><tr><td>SFT</td><td>Epoch 3</td><td>5.37 / 4.98</td><td>14.76 / 15.69</td><td>7.96</td><td>1804</td></tr><tr><td>SFT</td><td>Epoch 10</td><td>4.28 / 4.11</td><td>13.77 / 14.82</td><td></td><td>1808</td></tr><tr><td>Mistral 7B Instruct v0.1 (base model)</td><td></td><td>6.77 / 5.47</td><td>18.40 / 18.66</td><td></td><td>1375</td></tr><tr><td>Mixtral 8x7B Instruct v0.1 (larger model)</td><td></td><td>23.09 / 18.76</td><td>41.78 / 50.68</td><td></td><td>1464</td></tr></table>

## 5.3. Comparison with Larger Models of the Same Family

To probe the limits of the student, we compare each DNOtuned model with a larger model from the same family and release period, evaluated under the identical pipeline: Llama 3.1 70B Instruct and Mixtral 8x7B Instruct v0.1, respectively. In Table 1, Llama 3.1 8B Instruct after three DNO iterations scores above its 70B variant, which has nearly nine times as many parameters, on all four metrics; against GPT-4-0613 the LC difference is 14.95 points (70.18% vs. 55.23%), and against GPT-4o it is 3.92 points (45.32% vs. 41.40%). The raw win rates against GPT-4o differ by 8.32 points (48.26% vs. 39.94%, standard errors 1.76 and 1.72), about 3.4 standard errors of the difference between two independent evaluations. Two qualifications apply: the two models were not compared head-to-head but each against the same opponents; and GPT-4o, which produced the gold responses and rated every candidate, is also one of those opponents and the judge. The GPT-4-0613 column uses an opponent from an earlier model generation, but the judge is still GPT-4o, so it does not remove this exposure. Mistral 7B Instruct v0.1 with DNO (Iteration 3) comes close to Mixtral 8x7B on raw win rate against GPT-4o (18.53% vs. 18.76%) but remains behind on LC win rate against GPT-4o (18.44% vs. 23.09%) and on both metrics against GPT-4-0613 (38.20% vs. 41.78% LC; 37.85% vs. 50.68% raw). Mixtral has roughly 6.5 times the total parameters of the 7B model (46.7B vs. 7.2B; 12.9B active per token) in a mixture-of-experts architecture; whether that, the base model’s capacity, or the data accounts for the remaining gap we cannot tell from these runs.

## 5.4. Open-Weight Annotators from the Same Model Family

In this section we switch the annotator from the proprietary GPT-4o to a larger open-weight model from the student’s own family: Llama 3.1 70B Instruct annotates for Llama 3.1 8B Instruct, and Mistral Large annotates for Mistral 7B Instruct v0.1. These experiments show that the rating step of DNO can be run with an open-weight model; we make this claim for the rating step only, because the source of $y ^ { \mathrm { g o l d } }$ in these runs was not recorded. In this setting DNO can also be read as preference distillation from the larger model: the student learns from the teacher’s preference signal and, where $y ^ { \mathrm { g o l d } }$ comes from the teacher, from one of its responses.

The left and middle blocks of Table 3 compare the two annotators for Llama. Raw win rates against GPT-4o are similar (47.45% with the Llama annotator vs. 48.26% with GPT-4o), while the LC win rate is 5.19 points lower (40.13% vs. 45.32%); against GPT-4-0613 the LC gap is 11.27 points (58.91% vs. 70.18%), which may reflect more accurate ratings from GPT-4o or a judge preference for GPT-4o-like outputs (Section 8). Relative to the base model, the run with the open-weight annotator retains 8.4 of the 13.6 LC points gained against GPT-4o (62%) and 9.3 of the 20.5 points gained against GPT-4-0613 (45%). Even with this annotator, the 8B student scores above the 70B model on three of four metrics and 1.27 LC points below it against GPT-4o.

With Mistral Large as teacher (middle block of Table 4), the student improves over the base model: the LC win rate rises from $6 . 7 7 \%$ to 16.30% against GPT-4o (9.5 of the 11.7 points gained with GPT-4o as annotator, 82%) and from 18.40% to 38.71% against GPT-4-0613, matching the GPT-4o-annotated run on that metric (38.20%; 0.5 points higher). Unlike the GPT-4o-annotated runs, the LC win rate against GPT-4o reaches its highest value in the first iteration (16.40%) and does not exceed it afterwards (14.29% and 16.30%), and against GPT-4-0613 it stays between 38.71% and 39.16%; response length varies widely across iterations (8,264, 5,365 and 7,699 characters). On inspection, the annotations from Mistral Large were less consistent in formatting than those of GPT-4o or Llama 3.1 70B Instruct; we have not quantified the rating-failure rate or the number of usable pairs per iteration, so we do not attribute the flat trajectory to a specific cause.

Table 3. Llama 3.1 8B Instruct under three DNO settings: direct DNO annotated by GPT-4o (the run of Table 1), direct DNO annotated by Llama 3.1 70B Instruct, and SFT followed by DNO annotated by GPT-4o. Cells: LC / raw win rate (%); Len.: average length (chars). Bold: best DNO value per column within each block.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">Direct DNO, annotator GPT-4o</td><td colspan="3">Direct DNO, annotator Llama 3.1 70B</td><td colspan="3">SFT + DNO, annotator GPT-4o</td></tr><tr><td>vs. GPT-40</td><td>vs. GPT-4-0613</td><td>Len.</td><td>Vs. GPT-40</td><td>vs. GPT-4-0613</td><td>Len.</td><td>vs. GPT-40</td><td>vs. GPT-4-0613</td><td>Len.</td></tr><tr><td>DNO Iter. 1</td><td>39.01 / 43.04</td><td>61.44 / 68.45</td><td>3281</td><td>37.35 / 44.34</td><td>56.95 / 71.08</td><td>2763</td><td>36.85 / 38.20</td><td>50.52 / 65.53</td><td>2092</td></tr><tr><td>DNO Iter. 2</td><td>41.57 / 44.66</td><td>64.87 / 71.37</td><td>3626</td><td>40.01 / 47.45</td><td>58.55 / 72.55</td><td>3025</td><td>38.90 / 44.22</td><td>55.06 / 72.42</td><td>2285</td></tr><tr><td>DNO Iter. 3</td><td>45.32 / 48.26</td><td>70.18 / 74.60</td><td>3786</td><td>40.13 / 47.45</td><td>58.91 / 71.80</td><td>3185</td><td>44.92 / 45.14</td><td>68.96 / 69.30</td><td>4342</td></tr><tr><td>Llama 3.1 8B Instruct (base)</td><td>31.71 / 35.51</td><td>49.64 / 62.19</td><td>2237</td><td>一</td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>Llama 3.1 70B Instruct (larger)</td><td>41.40 /39.94</td><td>55.23 / 67.64</td><td>1873</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4. Mistral 7B Instruct v0.1 under the three DNO settings of Table 3, with Mistral Large as the open-weight annotator.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">Direct DNO, annotator GPT-4o</td><td colspan="3">Direct DNO, annotator Mistral Large</td><td colspan="3">SFT + DNO, annotator GPT-4o</td></tr><tr><td>vs. GPT-40</td><td>vs. GPT-4-0613</td><td>Len.</td><td> ${ \mathrm { v s . G P T } } { \mathrm { 4 o } }$ </td><td> $\mathbf { V S . \ G P T } \mathbf { - } 4 { - } 0 6 1 3$ </td><td>Len.</td><td>vs. GPT-40</td><td>vs. GPT-4-0613</td><td>Len.</td></tr><tr><td>DNO Iter. 1</td><td>9.94 / 10.64</td><td>33.62 / 35.12</td><td>2424</td><td>16.40 / 17.36</td><td>38.72 / 40.21</td><td>8264</td><td>10.18 / 10.43</td><td>30.83 / 34.53</td><td>2078</td></tr><tr><td>DNO Iter. 2</td><td>13.62 / 13.45</td><td>38.67 / 40.17</td><td>5365</td><td>14.29 / 15.50</td><td>39.16 / 38.81</td><td>5365</td><td>11.43 / 10.68</td><td>29.29 / 32.92</td><td>1768</td></tr><tr><td>DNO Iter. 3</td><td>18.44 / 18.53</td><td>38.20 / 37.85</td><td>7506</td><td>16.30 / 17.81</td><td>38.71 / 39.09</td><td>7699</td><td>12.58 / 13.29</td><td>30.56 / 37.27</td><td>2030</td></tr><tr><td>Mistral 7B Instruct v0.1 (base)</td><td>6.77 / 5.47</td><td>18.40 / 18.66</td><td>1375</td><td></td><td></td><td></td><td>一</td><td></td><td>一</td></tr><tr><td>Mixtral 8x7B Instruct v0.1 (larger)</td><td>23.09 / 18.76</td><td>41.78 / 50.68</td><td>1464</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.5. Two-Stage Training: SFT Followed by DNO

To compare with the conventional two-stage recipe (Ouyang et al., 2022; Tunstall et al., 2023), we first fine-tune each base model with SFT and then apply DNO on UltraFeedback, with the pre-processing of Section 4.1; DNO then starts from the SFT checkpoint.

The right-hand blocks of Tables 3 and 4 compare two-stage training with direct DNO on the instruct model. For Llama the two recipes end close to each other in LC win rate (44.92% vs. 45.32% against GPT-4o; 68.96% vs. 70.18% against GPT-4-0613), while direct DNO keeps a lead in raw win rate (3.1 and 5.3 points). These comparisons use the canonical run of Table 1; the two-stage runs were trained in the configuration of the ablation reference of Table 7 (43.90% and 69.46%), against which two-stage training is 1.0 point higher on LC win rate against GPT-4o and 0.5 points lower against GPT-4-0613. Two-stage responses are shorter than direct DNO’s in the first two iterations (2,092 and 2,285 vs. 3,281 and 3,626 characters) but longer at Iteration 3 (4,342 vs. 3,786); in both recipes the LC win rate peaks at the third iteration. For Mistral the conclusion does not transfer: two-stage training has lower point estimates on all four metrics (LC 12.58% vs. 18.44% against GPT-4o and 30.56% vs. 38.20% against GPT-4-0613; raw 13.29% vs. 18.53% and 37.27% vs. 37.85%), and its responses are about 73% shorter at Iteration 3 (2,030 vs. 7,506 characters). The SFT stage is a plausible cause: the epoch-3 SFT checkpoint lowers Mistral’s LC win rate against GPT-4-0613 by 3.6 points (14.76% vs. 18.40% in Table 2; by 1.4 points against GPT-4o). We have not tested this explanation with a different SFT initialization.

## 6. Ablation Studies and Analysis

All ablations use Llama 3.1 8B Instruct with GPT-4o as annotator and three iterations, in configurations separate from the canonical runs, and ablation runs are compared with runs outside their own table only where stated. The full-DNO reference row of Tables 5 and 7 (43.90% LC against GPT-4o, 5,324 characters) is a different run from the canonical run of Table 1 (45.32%, 3,786 characters). The margin sweep of Table 6 was run with its own pairconstruction and training scripts and at most 2,000 training steps per iteration. Within each sweep, the hyperparameters other than the swept one are fixed, but the number and mix of selected pairs, and hence of updates, change with it, so a sweep does not isolate one factor.

## 6.1. Number of On-Policy Samples

We vary the number of on-policy samples per prompt over 1, 3, 5 and 7 (Table 5). The LC values lie within a 2.1-point range (43.90% to 45.96%) with no trend in the number of samples, so within this range the sample count changes the LC win rate little; the main experiments use five samples, the default of Rosset et al. (2024). Two observations hold regardless. First, the settings differ in more than the sample count: the reported number of selected pairs spans an order of magnitude (6,733 for one sample; 68,524 for five; 68,819 for seven), and with one epoch per iteration more pairs mean more updates. Second, the average ratings of positives and negatives are similar across settings (5.8–5.9 and 3.4); averages do not capture the diversity or coverage of the selected pairs, which we did not measure.

Table 5. Number of on-policy samples per prompt (ablation configuration). Selected pairs by type (T: teacher’s gold response; S: student response); ratings on the 0–6 scale.
<table><tr><td>Samples per prompt</td><td>LC / raw win rate (%) vs. GPT-40</td><td>Avg. length (chars)</td><td> $T \succ S$ </td><td> $S \succ T$ </td><td> $S \succ S$ </td><td>Total</td><td>Avg. rating pos. / neg.</td></tr><tr><td>1</td><td>44.90 / 39.68</td><td>3251</td><td>4,025</td><td>2,708</td><td>0</td><td>6,733</td><td>5.87 / 3.39</td></tr><tr><td>3</td><td>45.96 / 41.37</td><td>2700</td><td>14,984</td><td>5,569</td><td>7,793</td><td>28,346</td><td>5.84 / 3.41</td></tr><tr><td>5</td><td>43.90 / 41.77</td><td>5324</td><td>20,978</td><td>14,167</td><td>33,379</td><td>68,524</td><td>5.92 / 3.44</td></tr><tr><td>7</td><td>44.15 / 38.75</td><td>3116</td><td>14,145</td><td>9,060</td><td>45,614</td><td>68,819</td><td>5.79 / 3.35</td></tr></table>

## 6.2. Rating Margin

A large calibrated reward gap motivates the contrastive approximation to Eq. (5); a large rubric-score gap is only a proxy for that condition (Section 6.6), but a margin that is too large discards most pairs and shifts the data mix toward teacher positives. Table 6 evaluates margins of one to five with all other hyperparameters fixed; the number and mix of selected pairs, and hence of updates, change with the margin. The LC win rate against GPT-4o lies between 36.60% and 38.99% for all five margins, with the highest value at margin three; over this range the LC win rate changes by only 2.4 points, and the main experiments keep the default margin of two from Rosset et al. (2024). Rosset et al. (2024, Section 5.2) compared margin filtering with training on all pairs and found filtering beneficial; our sweep varies the margin but has no unfiltered run. Three observations follow from the pair statistics. First, the share of pairs whose positive is the teacher’s response is about 30% at margins one and two and 35–38% at margins three to five; it is not monotonic in the margin. Second, the average rating of positives stays at 5.8–6.0 while that of negatives falls from 3.70 to 0.92, so large margins select very poor negatives. We did not measure whether such pairs are less informative per sample. Under the DPO loss the per-pair gradient weight is $\sigma ( - z )$ where z is the model’s implicit log-ratio margin scaled by $\beta ;$ because the reference policy is the starting checkpoint of each iteration (Section 4.1), z = 0 for every pair at the first update with the default adapter initialization, regardless of the rubric gap, so the rubric gap does not by itself change the initial gradient weight. Third, the average response length decreases with the margin, which may partly reflect the larger share of the teacher’s typically shorter responses at margins three to five, although the share does not track length at every step. The margin also sets the data volume, from 148,764 reported selected pairs at margin one to 1,815 at margin five.

Comparability with the other ablations: All five margin runs score below the full-DNO reference of Table 7 (43.90% LC against GPT-4o), including the margin-2 run (37.72%), which uses the same margin. Because the sweep used a separate pipeline (Section 6), this gap is confounded with implementation differences that single runs cannot separate from run-to-run variation, so we compare the margin runs only with each other.

## 6.3. Gold Responses

We examine the role of $y ^ { \mathrm { g o l d } }$ . In addition to preference signals, DNO includes the teacher’s response as one of the candidates; in practice, gold responses may be unavailable because they cost more than preference labels alone. Removing them (row “Remove $y ^ { \mathrm { g o l d } \cdots }$ of Table 7) lowers the LC win rate against GPT-4o from 43.90% to 36.97% and against GPT-4-0613 from 69.46% to 55.36%, while the raw win rates change by −0.9 and +1.9 points. The drop is consistent with the teacher’s responses being rated well above the student’s in the first iteration (Table 8). Without $y ^ { \mathrm { g o l d } }$ DNO becomes a bootstrapped online DPO with more than two candidates per prompt. Its LC win rate against GPT-4o (36.97%) is close to that of our offline DPO baseline (37.02% in Table 1); the two runs use different data and training budgets, so the proximity is not a controlled comparison. Two controls that we did not run would separate the questions this raises: (i) offline DPO under the ablation training budget, to test whether the budget explains the proximity; and (ii) offline DPO trained from the same initialization on the Iteration-1 DNO pairs, with matched update budget, pair rule, gold inclusion and prompt coverage, to test whether iterative on-policy resampling adds anything to a single round of teacher-rated data. With preference signals alone, the variant without $y ^ { \mathrm { g o l d } }$ still scores 5.3 LC points above the base model (31.71%).

## 6.4. Pair-Selection Rule and Data Look-Back

Table 7 compares the full pair-selection rule with two restricted rules. Always marking the teacher’s response as positive and all student responses as negative (the SPIN-style rule) lowers the LC win rate by 8.73 points against GPT-4o and by 19.16 points against GPT-4-0613. A plausible reason is that this rule labels a student response as negative even when the annotator rates it above the teacher’s, which becomes more frequent in later iterations (Table 8). Accepting only $T \succ S$ pairs, i.e., trusting the teacher only when the rating confirms it (DNO-Restrictive), lands between the two against GPT-4o (−5.92 vs. −8.73 points) and about half as far below the full rule against GPT-4-0613 (−9.30 vs. −19.16); the full rule scores highest on both LC metrics. On raw win rate the SPIN-style rule is level with or above the full rule, while DNO-Restrictive is 3 to 4 points below; the benefit of the full rule therefore shows mainly in the length-controlled metric, and the full-DNO reference also has the longest responses in the table (5,324 characters vs. 2,489 to 4,802), so this benefit depends on the length adjustment, which we have not validated for responses this long (Section 8). On LC win rate our ordering, SPIN-style < DNO-Restrictive < full rule, matches the one Rosset et al. (2024, Table 1) report in their setting.

Table 6. Minimum rating margin (separate pipeline, at most 2,000 training steps per iteration). Columns as in Table 5, plus the share of pairs whose positive is the teacher’s response.
<table><tr><td>Margin</td><td>LC / raw win rate (%) vs. GPT-40</td><td>Avg. length (chars)</td><td> $T \succ S$ </td><td> $S \succ T$ </td><td> $S \succ S$ </td><td>Total</td><td>Share of pairs with teacher positive (%)</td><td>Avg. rating pos. / neg.</td></tr><tr><td>1</td><td>36.60 / 41.23</td><td>2978</td><td>45,344</td><td>24,643</td><td>78,777</td><td>148,764</td><td>30.48</td><td>5.78 / 3.70</td></tr><tr><td>2</td><td>37.72 / 42.92</td><td>2931</td><td>16,601</td><td>13,089</td><td>25,712</td><td>55,402</td><td>29.96</td><td>5.81 / 3.39</td></tr><tr><td>3</td><td>38.99 / 43.54</td><td>2619</td><td>9,377</td><td>6,784</td><td>8,765</td><td>24,926</td><td>37.62</td><td>5.79 / 2.01</td></tr><tr><td>4</td><td>37.38 / 40.98</td><td>2551</td><td>5,169</td><td>4,580</td><td>3,826</td><td>13,575</td><td>38.08</td><td>5.82 / 1.42</td></tr><tr><td>5</td><td>38.13 / 41.93</td><td>2542</td><td>638</td><td>923</td><td>254</td><td>1,815</td><td>35.15</td><td>5.97 / 0.92</td></tr></table>

Table 7. Gold responses, pair-selection rule and data look-back (ablation configuration). ∆LC: change relative to full DNO. “Mark T always positive” is the SPIN-style rule and “Accept only $T \succ S ^ { \prime \prime }$ the DNO-Restrictive rule. The look-back variant starts from the SFT checkpoint; its matched comparison is given in Section 6.4. Bold: best value per win-rate column.
<table><tr><td rowspan="2">Variant</td><td colspan="3">vs. GPT-4o (%)</td><td colspan="3"> $\mathrm { v s . } \mathrm { G P T } \mathrm { - } 4 \mathrm { - } 0 6 1 3 \mathrm { ( \% ) }$ </td><td rowspan="2">Avg. length (chars)</td></tr><tr><td>LC</td><td>∆LC</td><td>raw</td><td>LC</td><td>∆LC</td><td>raw</td></tr><tr><td>DNO (full)</td><td>43.90</td><td></td><td>41.77</td><td>69.46</td><td></td><td>64.21</td><td>5324</td></tr><tr><td>Remove  $y ^ { \mathrm { g o l d } }$ </td><td>36.97</td><td>-6.93</td><td>40.91</td><td>55.36</td><td>-14.10</td><td>66.15</td><td>2648</td></tr><tr><td>Mark T always positive</td><td>35.17</td><td>-8.73</td><td>41.99</td><td>50.30</td><td>-19.16</td><td>66.81</td><td>3207</td></tr><tr><td>Accept only  $T \succ S$ </td><td>37.98</td><td>-5.92</td><td>38.62</td><td>60.16</td><td>-9.30</td><td>60.51</td><td>4802</td></tr><tr><td>Remove data look-back</td><td>35.00</td><td>-8.90</td><td>41.61</td><td>50.10</td><td>-19.36</td><td>67.45</td><td>2489</td></tr></table>

Removing the training pairs carried over from previous iterations (row “Remove data look-back”) lowers the LC win rate by 8.90 and 19.36 points relative to the full-DNO reference, while the raw win rates are almost unchanged against GPT-4o (41.61% vs. 41.77%) and higher against GPT-4-0613 (67.45% vs. 64.21%). This variant was trained from the SFT checkpoint, so its like-for-like counterpart is the two-stage Iteration-3 run of the same run family, trained with look-back (Table 3: 44.92% / 45.14% against GPT-4o and 68.96% / 69.30% against GPT-4-0613). Against that run, removing look-back lowers the LC win rate by 9.92 and 18.86 points and the raw win rate by 3.53 and 1.85 points, so the decline holds on all four metrics. One hypothesis is that, because candidate responses are generated on-policy, a model that no longer sees earlier pairs regresses to behaviors that were already penalized; we observed only the aggregate scores, not those behaviors.

Against GPT-4o, all four variants are 5.9 to 8.9 LC points below the full-DNO reference, and the look-back variant is 9.9 points below its matched two-stage counterpart. These differences do not apportion the gain among the components.

## 6.5. Student Evolution under the Teacher–Student Framework

We examine how the rated quality of the student’s responses evolves relative to the teacher’s, repeating on two other base models the per-iteration analysis of Rosset et al. (2024, Table 2), who used a GPT-4 Turbo teacher. Table 8 compares teacher and student responses on length, rating and pair composition; each row describes the responses sampled at the start of that iteration, not the checkpoint at its end. The student’s average rating rises with every iteration, and so does the share of selected pairs whose positive is a student response. For Llama 3.1 8B Instruct, the student’s average rating in Iteration 1 is 3.28, 1.58 points below the teacher’s 4.86; the two averages are nearly equal in Iteration 2 (4.43 vs. 4.42, on a different prompt partition) and the student’s is 0.26 points higher in Iteration 3. By Iteration 3 there are also more $S \succ T$ than $T \succ S$ pairs $( 7 , 6 0 2 \mathrm { v s . } 6 , 0 5 7 )$ , showing that the selection rule admits student-positive, teachernegative pairs, as Rosset et al. (2024) also observed; this count is not a test of equilibrium attainment or teacher-level performance. Mistral 7B Instruct v0.1 shows the same trend: the student’s rating improves across iterations and its share of pairs with a student positive more than doubles, from 33.45% to 77.01%, although its rating (4.13) stays below the teacher’s (4.66) in these runs.

Table 8. Teacher (T, GPT-4o gold response) versus student $( S )$ across DNO iterations, with GPT-4o as annotator. Row “Iter. $t ^ { \ast }$ describes the responses sampled from $\pi _ { t }$ at the start of iteration t, on that iteration’s prompt partition (Section 4.1); pair counts are new pairs only, excluding look-back.
<table><tr><td></td><td>Avg. length T / S (chars)</td><td>Avg. rating  $T / S$ </td><td></td><td> $T \succ S \quad S \succ T \quad S \succ S$ </td><td></td><td>Total</td><td>Share of pairs with student positive (%)</td><td>Avg. rating pos. / neg.</td></tr><tr><td>Llama 3.1 8B, Iter. 1</td><td>1116.45 / 1521.30</td><td>4.86 / 3.28</td><td>25,093</td><td>2,373</td><td>21,543</td><td>49,009</td><td>48.80</td><td>5.74 / 1.68</td></tr><tr><td>Llama 3.1 8B, Iter. 2</td><td>1121.66 / 1744.86</td><td>4.42 / 4.43</td><td>12,852</td><td>11,342</td><td>14,058</td><td>38,252</td><td>66.40</td><td>5.62 / 2.70</td></tr><tr><td>Llama 3.1 8B, Iter. 3</td><td>1103.87 / 2003.26</td><td>4.26 / 4.52</td><td>6,057</td><td>7,602</td><td>7,747</td><td>21,406</td><td>71.70</td><td>5.62 / 2.67</td></tr><tr><td>Mistral 7B, Iter. 1</td><td>1147.87 /  806.08</td><td>5.03 / 3.48</td><td>42,576</td><td>2,273</td><td>19,129</td><td>63,978</td><td>33.45</td><td>5.69 / 2.60</td></tr><tr><td>Mistral 7B, Iter. 2</td><td>1252.91 / 2048.87</td><td>4.63 / 3.93</td><td>20,450</td><td>5,518</td><td>24,943</td><td>50,911</td><td>59.83</td><td>5.61 / 2.51</td></tr><tr><td>Mistral 7B, Iter. 3</td><td> $1 4 4 6 . 4 6 / 2 9 5 1 . 2 2$ </td><td>4.66 / 4.13</td><td>6,268</td><td>2,581</td><td>18,412</td><td>27,261</td><td>77.01</td><td>5.58 / 3.07</td></tr></table>

Three caveats qualify this picture. First, the comparison is made on the annotator’s own rubric: the same $\mathrm { G P T - } 4 0$ that produced $y ^ { \mathrm { g o l d } }$ also scores it, and training-time rubric scores and external benchmark win rates measure different things. Under AlpacaEval the Llama Iteration-3 model wins 388, loses 416 and ties one of the 805 comparisons with GPT-4o (raw win rate 48.26%, standard error 1.76), which is not distinguishable from parity. Second, the teacher’s average rating falls across iterations (4.86 to 4.26 for Llama) even though its responses are fixed, as Rosset et al. (2024) also report for their teacher; because each iteration uses a different prompt partition and the six responses are scored side by side, we cannot separate harder prompts from relative scoring, and the change in the gap should not be read as a decomposition of student improvement versus rating drift. Third, the counts are shaped by the pair-selection rule: five student samples face one teacher response, and the one-point relaxation of the margin for $S \succ T$ pairs inflates their count relative to $T \succ S$ pairs. Within these caveats, the results support letting a student response serve as the positive when the annotator rates it above the teacher ${ \bf \vec { s } , \vec { } }$ instead of always preferring the teacher as the SPIN-style rule does (Section 6.4).

## 6.6. Sources of Approximation Error

The experiments above expose several places where DNO-Prct departs from the theory of Section 2.

The large-margin approximation: Since $r _ { t } ( x , y ) \in [ 0 , 1 ]$ the target weight $\sigma ( r _ { t } ( x , y ^ { + } ) { - } r _ { t } ( x , y ^ { - } ) )$ is at most $\sigma ( 1 ) \approx$ 0.73 and cannot approach one without rescaling. Rosset et al. (2024) therefore rescale $r _ { t }  c r _ { t }$ and $\eta  c \eta = : { \tilde { \eta } } ,$ which leaves the soft-policy-iteration target unchanged, and note that the appropriate $\tilde { \eta } ,$ , the coefficient of the log-ratio inside the contrastive loss $( \mathrm { D P O ^ { \ast } s } \beta )$ , is thus expected to be larger than $\eta ;$ because a larger $\tilde { \eta }$ worsens the sample complexity of their Theorem 1, only large-margin pairs are retained so that $\tilde { \eta }$ need not be too large. Two points follow. First, although Rosset et al. (2024) call η˜ a learning rate, it is the coefficient of the log-ratio, not the optimizer’s learning rate; our learning-rate search does not test this scaling argument, and our runs use $\beta = 0 . 1$ (Appendix B). A sweep over $\beta$ at a fixed margin is the experiment that would test it. Second, a large gap on an ordinal 0–6 rating scale is not a calibrated large gap in the theoretical reward $r _ { t } ,$ so the margin filter is a modeling approximation whose validity we have not measured; in the margin sweep above, margins of one to five changed the LC win rate by at most 2.4 points at this step budget.

Rating noise: Since we use an LLM to grade responses in a batched setting, following Rosset et al. (2024), the ratings depend on the annotator’s instruction following (Zheng et al., 2023). Presenting all candidates in one query relies on the annotator to score each response separately and consistently: in GPT-4o’s rating replies we observed incomplete ratings, where fewer scores are returned than candidates, and inconsistencies in the indexing of the scores; Mistral Large was less consistent still (Section 5.4). Such replies are discarded, so the affected prompts contribute no pairs. The candidates are presented in a fixed order with $y ^ { \mathrm { g o l d } }$ last, so any position bias of the annotator (Zheng et al., 2023) falls systematically on the gold response. The ratings have no explicit length control, and the student’s responses are longer than the teacher’s from the first iteration for Llama and from the second for Mistral (Table 8).

Finite on-policy samples: The on-policy sample (Step 1 of DNO-Prct) is drawn once from $\pi _ { t }$ at the start of the iteration, so it becomes increasingly off-policy as the update proceeds, and five samples per prompt cover only a small part of the policy’s output distribution; we did not measure their diversity.

## 7. Costs of DNO

We report the cost of a three-iteration DNO run on Ultra-Feedback in terms of GPT-4o API calls and wall-clock time. By construction the pipeline makes about 2n API calls for n prompts: n calls to generate y<sup>gold</sup>, and, because the three prompt partitions are disjoint and each prompt’s six responses are rated in one batched query, a further n rating calls over the three iterations, before any retries. For the 63,967 UltraFeedback prompts this estimate is roughly 128,000 calls; the count recorded in 2024, approximately 126,000, is slightly lower. Failed calls are retried up to three times.

Each iteration uses one third of the prompts, for which we generate five on-policy responses each with vLLM (Kwon et al., 2023); generation takes about three to four hours on four A100 GPUs. Rating all responses takes five to six hours of API time, which multithreading reduces to about two hours. Training takes six to eight hours on the same GPUs, depending on the number of large-margin pairs. Summed over three iterations with multithreaded rating, the listed stages take 33 to 42 hours; the end-to-end time recorded in 2024 was 35 to 45 hours.

## 8. Limitations

The teacher is also the evaluation judge and an opponent: GPT-4o generates $y ^ { \mathrm { g o l d } }$ , rates every candidate, is one of the two AlpacaEval opponents and is the AlpacaEval judge. Between 23% and 67% of the selected pairs in each iteration have a GPT-4o response as the positive (Table 8), the GPT-4o response is always the last candidate in the rating query, and LLM judges prefer outputs that resemble their own (Panickssery et al., 2024; Wataoka et al., 2024). Li et al. (2026) measure an average preference-leakage score of about 24% when judge and data generator are the same model, as in our main runs, and about 3–9% when they share a model family. Those magnitudes come from other settings and are not a correction for ours, but they show that part of the measured gain may be GPT-4o-likeness as scored by GPT-4o rather than better responses. The open-weightannotator runs of Section 5.4 are a partial control only for the rating step, since the judge is unchanged: with Llama 3.1 70B Instruct as annotator the Llama student gains 8.4 LC points against GPT-4o, versus 13.6 with GPT-4o as teacher, and the difference cannot be separated into annotator quality and judge affinity. The comparison with the 70B model (Section 5.3) is therefore a benchmark comparison under one protocol, not a ranking. A judge outside the GPT family, an independent benchmark, matched-length analysis, and a capability-regression check on standard benchmarks are the most important additions.

Non-standard evaluation and single runs: Our evaluation protocol departs from the public leaderboard (Section 4.1), so our numbers are not comparable to those in other papers, and the agreement of LC win rates with human rankings has not been established for it. Every cell is a single run, and between-run variability is unknown. A verbatim search finds 13 of the 805 AlpacaEval instructions (10 of them at least 40 characters long) in the processed UltraFeedback data; because a match may fall inside a response, about 1.6% is an upper bound on exact prompt overlap, and nearduplicates were not searched. Zheng et al. (2025) show that a constant, adversarially crafted output reaches 86.5% LC on AlpacaEval 2.0; this does not show that our checkpoints exploit the benchmark, but high absolute LC values, including our 70.18% against GPT-4-0613, are evidence of preference under this judge rather than of quality in general.

Hyperparameter selection: The learning rate was chosen after the first epoch or iteration of each experiment. If the selection criterion was the evaluation we report, the selection is optimistic; selecting on a separate split, with the reported evaluation kept out of the selection, would remove this optimism.

Human preferences: No human-preference study is reported. Both the training signal and the evaluation are LLM judgments used as proxies for human preference, and the handling of intransitive preferences, the motivation for general-preference methods, was not tested.

Baselines: SPIN is approximated by the teacher-alwayspositive rule rather than by the original recipe; SFT on UltraFeedback is a weak baseline for an instruct model; our SPPO and online DPO runs received only the learning-rate search and ended below the base model, so they say nothing about those methods; and no SimPO baseline was run.

Response length: In the direct GPT-4o-annotated runs, response length grows with iterations, to 3,786 characters for Llama and 7,506 for Mistral (Section 5), and the rating step has no length control. We did not measure truncation for the Mistral runs or validate the length-control regression for these unusually long responses. Length growth alone does not demonstrate reward hacking, but it is the pattern reward hacking would produce; win rates at matched length would make the claim of more informative responses testable.

A 2024 snapshot: The experiments were run in 2024 with the models and APIs of that time. GPT-4o is no longer a frontier judge, Mistral 7B Instruct v0.1 is superseded, and neither concurrent Nash-style methods such as INPO (Zhang et al., 2025b) nor evaluator-trained open-weight judges (Wang et al., 2024) were compared against. We report the study because the procedural description and the design observations do not depend on the models being current; nothing has been re-run on newer models.

Algorithmic approximations: Section 6.6 lists where DNO-Prct departs from the theory it approximates; in addition, none of our experiments evaluates the uniform mixture of iterates for which the Nash guarantee holds.

## 9. Conclusion

We re-implemented Direct Nash Optimization from its published description and studied it on two open-weight instruct models with proprietary and open-weight annotators. Under our evaluation protocol, three DNO iterations raise the length-controlled AlpacaEval 2.0 win rate against GPT-4o from 31.71% to 45.32% for Llama 3.1 8B Instruct and from 6.77% to 18.44% for Mistral 7B Instruct v0.1, and the tuned 8B model scores above Llama 3.1 70B Instruct on all four reported win rates. The rating step does not require a proprietary model: same-family open-weight annotators retain 62% and 82% of the LC gain of the GPT-4o-annotated runs against GPT-4o, and against GPT-4-0613 the Llama run retains 45% and the Mistral run matches the GPT-4oannotated run. The ablations indicate which components matter: removing the gold responses or the look-back, or accepting only teacher-preferred pairs, scores 6 to 10 LC points below the full procedure, whereas the number of on-policy samples moves the LC win rate by at most 2.1 points and, in a separate sweep, the rating margin by at most 2.4 points. The defaults we would therefore keep are the teacher’s gold responses, look-back, and rating teacher and student responses together so that a student response can be the positive, with response length measured explicitly. These findings rest on single runs evaluated by a GPT-4o judge that is also the teacher; an independent judge, repeated runs and the matched controls of Sections 6.3 and 6.6 are the natural next steps.

## Acknowledgements

A large language model was used to assist with editing this manuscript; the authors take full responsibility for its content.

## References

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mane, D. Concrete problems in AI safety.´ arXiv preprint arXiv:1606.06565, 2016.

Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., and Munos, R. A general theoretical paradigm to understand learning from human preferences. In Proceedings ofthe 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Pro-

ceedings of Machine Learning Research, pp. 4447–4455. PMLR, 2024. arXiv:2310.12036.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901, 2020. arXiv:2005.14165.

Calandriello, D., Guo, Z. D., Munos, R., Rowland, M., Tang, Y., Avila Pires, B., Richemond, P. H., Le Lan, C., Valko, M., Liu, T., Joshi, R., Zheng, Z., and Piot, B. Human alignment of large language models through online preference optimisation. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 5409–5435. PMLR, 2024. arXiv:2403.08635.

Chen, W.-L., Wei, Z., Zhu, X., Feng, S., and Meng, Y. Do LLM evaluators prefer themselves for a reason? arXiv preprint arXiv:2504.03846, 2025.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Self-play fine-tuning converts weak language models to strong language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 6621–6642. PMLR, 2024. arXiv:2401.01335.

Cheng, P., Yang, Y., Li, J., Dai, Y., Hu, T., Cao, P., Du, N., and Li, X. Adversarial preference optimization: Enhancing your alignment via RM-LLM game. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 3705–3716, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.221. arXiv:2311.08045.

Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., and Stoica, I. Chatbot Arena: An open platform for evaluating LLMs by human preference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 8359–8388. PMLR, 2024. arXiv:2403.04132.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, 2017. arXiv:1706.03741.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., Liu, Z., and Sun, M. UltraFeedback: Boosting language models with scaled AI feedback.

In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 9722–9744. PMLR, 2024. arXiv:2310.01377.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. RLHF workflow: From reward modeling to online RLHF. Transactions on Machine Learning Research, 2024. arXiv:2405.07863.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Fishburn, P. C. Probabilistic social choice based on simple voting comparisons. The Review of Economic Studies, 51 (4):683–692, 1984. doi: 10.2307/2297786.

Gao, Z., Chang, J. D., Zhan, W., Oertell, O., Swamy, G., Brantley, K., Joachims, T., Bagnell, J. A., Lee, J. D., and Sun, W. REBEL: Reinforcement learning via regressing relative rewards. In Advances in Neural Information Processing Systems, volume 37, pp. 52354–52400, 2024. arXiv:2404.16767.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares, F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., Ferret, J., and Blondel, M. Direct language model alignment from online AI feedback. arXiv preprint arXiv:2402.04792, 2024.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. arXiv:1904.09751.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=nZeVKeeFYf9.

Ivison, H., Wang, Y., Liu, J., Wu, Z., Pyatkin, V., Lambert, N., Smith, N. A., Choi, Y., and Hajishirzi, H. Unpacking DPO and PPO: Disentangling best practices for learning from preference feedback. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2406.09279.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux,

M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., and El Sayed, W. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Kramer, G. H. On a class of equilibrium conditions for majority rule. Econometrica, 41(2):285–297, 1973. doi: 10.2307/1913490.

Kreweras, G. Aggregation of preference orderings. In Mathematics and Social Sciences I: Proceedings of the Seminars of Menthon-Saint-Bernard, France (1–27 July 1960) and ofGosing, Austria (3–27 July 1962)¨ , pp. 73–79. Mouton, Paris, 1965.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP), 2023. arXiv:2309.06180.

Lanchantin, J., Chen, A., Lan, J., Li, X., Saha, S., Wang, T., Xu, J., Yu, P., Yuan, W., Weston, J. E., Sukhbaatar, S., and Kulikov, I. Bridging offline and online reinforcement learning for LLMs. arXiv preprint arXiv:2506.21495, 2025.

Li, D., Sun, R., Huang, Y., Zhong, M., Jiang, B., Han, J., Zhang, X., Wang, W., and Liu, H. Preference leakage: A contamination problem in LLM-as-a-judge. In The Fourteenth International Conference on Learning Representations, 2026. arXiv:2502.01534.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. AlpacaEval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/ alpaca\_eval, 2023.

Liu, J., Zhou, Z., Liu, J., Bu, X., Yang, C., Zhong, H.-S., and Ouyang, W. Iterative length-regularized direct preference optimization: A case study on improving 7B language models to GPT-4 level. arXiv preprint arXiv:2406.11817, 2024.

Liu, K., Long, Q., Shi, Z., Su, W. J., and Xiao, J. Statistical impossibility and possibility of aligning LLMs with human preferences: From Condorcet paradox to Nash equilibrium. arXiv preprint arXiv:2503.10990, 2025.

Meng, Y., Xia, M., and Chen, D. SimPO: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2405.14734.

Meta AI. Llama Recipes (now Llama Cookbook): Scripts for fine-tuning Meta Llama models. https: //github.com/meta-llama/llama-recipes, 2024. Accessed 2026-10-14.

Mistral AI. Large enough. https://mistral.ai/ news/mistral-large-2407/, 2024. Announcement of Mistral Large 2 (Mistral-Large-Instruct-2407, 123B parameters). Accessed 2026-10-14.

Munos, R., Valko, M., Calandriello, D., Azar, M. G., Rowland, M., Guo, Z. D., Tang, Y., Geist, M., Mesnard, T., Michi, A., Selvi, M., Girgin, S., Momchev, N., Bachem, O., Mankowitz, D. J., Precup, D., and Piot, B. Nash learning from human feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 36743–36768. PMLR, 2024. arXiv:2312.00886.

OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

OpenBMB. UltraFeedback dataset. https: //huggingface.co/datasets/openbmb/ UltraFeedback, 2023. Accessed 2026-10-14.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730– 27744, 2022. arXiv:2203.02155.

Panickssery, A., Bowman, S. R., and Feng, S. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems, volume 37, pp. 68772–68802, 2024. arXiv:2404.13076.

Park, R., Rafailov, R., Ermon, S., and Finn, C. Disentangling length from quality in direct preference optimization. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 4998–5017, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.297. arXiv:2403.19159.

Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pretraining. Technical report, OpenAI, 2018. URL https: //cdn.openai.com/research-covers/ language-unsupervised/language\_ understanding\_paper.pdf.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https://cdn.openai. com/better-language-models/language\_ models\_are\_unsupervised\_multitask\_ learners.pdf.

Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., et al. Scaling language models: Methods, analysis & insights from training Gopher. arXiv preprint arXiv:2112.11446, 2021.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023. arXiv:2305.18290.

Ross, S., Gordon, G., and Bagnell, D. A reduction of imitation learning and structured prediction to noregret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 627–635. PMLR, 2011. URL https://proceedings.mlr.press/v15/ ross11a.html.

Rosset, C., Cheng, C.-A., Mitra, A., Santacroce, M., Awadallah, A., and Xie, T. Direct Nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

Sharma, A., Keh, S., Mitchell, E., Finn, C., Arora, K., and Kollar, T. A critical evaluation of AI feedback for aligning large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2402.12366.

Simpson, P. B. On defining areas of voter choice: Professor Tullock on stable voting. The Quarterly Journal of Economics, 83(3):478–490, 1969. doi: 10.2307/1880533.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pp. 3008–3021, 2020. arXiv:2009.01325.

Swamy, G., Dann, C., Kidambi, R., Wu, Z. S., and Agarwal, A. A minimaximalist approach to reinforcement learning from human feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 47345–47377. PMLR, 2024. arXiv:2401.04056.

Tang, X., Yoon, S., Son, S., Yuan, H., Gu, Q., and Bogunovic, I. RSPO: Regularized self-play alignment of large language models. In Proceedings ofthe 43rd International Conference on Machine Learning, 2026.

Tang, Y., Guo, D. Z., Zheng, Z., Calandriello, D., Cao, Y., Tarassov, E., Munos, R., Avila Pires, B., Valko, M.,<sup>´</sup> Cheng, Y., and Dabney, W. Understanding the performance gap between online and offline alignment algorithms. arXiv preprint arXiv:2405.08448, 2024.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., and Wolf, T. Zephyr: Direct distillation of LM alignment. arXiv preprint arXiv:2310.16944, 2023.

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., and Gallouedec, Q. TRL: Transformers reinforcement learn-´ ing. https://github.com/huggingface/trl, 2020.

Wang, M., Ma, C., Chen, Q., Meng, L., Han, Y., Xiao, J., Zhang, Z., Huo, J., Su, W. J., and Yang, Y. Magnetic preference optimization: Achieving last-iterate convergence for language model alignment. In The Thirteenth International Conference on Learning Representations, 2025. arXiv:2410.16714.

Wang, T., Kulikov, I., Golovneva, O., Yu, P., Yuan, W., Dwivedi-Yu, J., Pang, R. Y., Fazel-Zarandi, M., Weston, J., and Li, X. Self-taught evaluators. arXiv preprint arXiv:2408.02666, 2024.

Wataoka, K., Takahashi, T., and Ri, R. Self-preference bias in LLM-as-a-judge. arXiv preprint arXiv:2410.21819, 2024.

Wu, Y., Sun, Z., Yuan, H., Ji, K., Yang, Y., and Gu, Q. Self-play preference optimization for language model alignment. In The Thirteenth International Conference on Learning Representations, 2025. arXiv:2405.00675.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 54715–54754. PMLR, 2024. arXiv:2312.11456.

Xu, J., Lee, A., Sukhbaatar, S., and Weston, J. Some things are more CRINGE than others: Iterative preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., and Weston, J. Self-rewarding language models. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 57905–57923. PMLR, 2024. arXiv:2401.10020.

Zhang, Y., Yu, D., Ge, T., Song, L., Zeng, Z., Mi, H., Jiang, N., and Yu, D. Improving LLM general preference alignment via optimistic online mirror descent. In Advances in Neural Information Processing Systems, 2025a.

Zhang, Y., Yu, D., Peng, B., Song, L., Tian, Y., Huo, M., Jiang, N., Mi, H., and Yu, D. Iterative Nash policy optimization: Aligning LLMs with general preferences via no-regret learning. In The Thirteenth International Conference on Learning Representations, 2025b. arXiv:2407.00617.

Zhang, Y., Zhang, G., Wu, Y., Xu, K., and Gu, Q. Beyond Bradley-Terry models: A general preference model for language model alignment. In Proceedings of the 42nd International Conference on Machine Learning, 2025c.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging LLM-as-ajudge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023. Datasets and Benchmarks Track. arXiv:2306.05685.

Zheng, X., Pang, T., Du, C., Liu, Q., Jiang, J., and Lin, M. Cheating automatic LLM benchmarks: Null models achieve high win rates. In The Thirteenth International Conference on Learning Representations, 2025. arXiv:2410.07137.

## A. Rating Prompt

The annotator receives the prompt of Figure 2, reproduced verbatim from our code. The user prompt replaces {instructions}, and the candidate responses replace {candidate answers}, the K on-policy samples first (five unless stated) and $y ^ { \mathrm { g o l d } }$ last, indexed from zero; the doubled braces are escapes of Python’s string formatting and reach the annotator as single braces. The rubric is additive over six criteria, one point each, so scores range from 0 to 6, and the annotator is asked to return only a Python dictionary with one score per candidate. Our prompt follows Rosset et al. (2024) with modifications so that it transfers to annotators outside the GPT family.

![](images/f873e299133c40c0b96457602f332eead596dd7d701cdf8321ff234850c7b2be.jpg)  
Figure 2. The rating prompt used in all DNO runs, verbatim from our code.

## B. Hyperparameters

Table 9 lists the training, generation and evaluation settings.

## C. Pair-Construction Procedure

Algorithm 2 states one DNO-Prct iteration with the thresholds of Section 4.2. Ratings are integers from 0 to 6; a response is a positive if its rating is at least five. Candidates are kept in a fixed order, the K student samples first and the gold response last, and the test $y ^ { - } = y _ { x } ^ { \mathrm { g o l d } }$ is on the index, not on the text. $D _ { t } ^ { \mathrm { n e w } }$ denotes the pairs built from iteration t’s own samples, before look-back.

Table 9. Training, generation and evaluation hyperparameters.   
Hyperparameter Value   
Fine-tuning LoRA: rank $1 6 , \alpha 3 2 ,$ dropout 0.05; adapters merged after each DNO iteration   
Learning rate (SFT) $5 \times 1 0 ^ { - 5 }$   
Learning rate (DPO, DNO) $5 \times 1 0 ^ { - 6 }$ (most DNO runs)   
Learning-rate search $\{ 5 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 6 } \}$ , first epoch / iteration   
Optimizer; scheduler Adam $| \mathrm { W } ;$ constant after 100 warm-up steps; weight decay 0   
Effective batch size 2 per device $\times 4 \mathrm { G P U s } \times 4$ accumulation steps = 32   
GPUs 4 × NVIDIA A100-SXM 80 GB   
Epochs (SFT / DPO) 20 (checkpoints at 3, 10) / 3   
DNO iterations × epochs per iteration $3 \times 1$   
DPO loss; coefficient $\beta$ sigmoid; 0.1   
Reference policy π<sub>t</sub>: the iteration’s starting checkpoint with adapters disabled   
Max. sequence / prompt length 4,096 / 3,072 tokens (training)   
Seed 2024   
On-policy samples per prompt 5   
Sampling temperature / top-p 0.7 / 0.95   
Rating query all candidates in one query, fixed order, $y ^ { \mathrm { g o l d } }$ last   
Positive threshold rating $\geq 5$ on the 0–6 scale   
Rating margin $( T \succ S , S \succ S / S \succ T )$ 2 / 1 points   
Data look-back at most 30% of the pairs of t − 1, 15% of $t - 2 ,$   
Step cap (margin ablation) 2,000 training steps per iteration   
Prompt partitions 3 disjoint splits of UltraFeedback   
Evaluation decoding vLLM, chat template, temperature 0.001, at most 4,096 new tokens   
AlpacaEval v0.6.5; references by GPT-4o and GPT-4-0613 (temperature 0); judge GPT-4o,   
alpaca eval gpt4 ranking template   
Software TRL 0.8.6, transformers 4.43.3, PEFT 0.10.0, accelerate 0.32.1, datasets 2.16.1;   
vLLM

Algorithm 2 One DNO-Prct iteration (pair construction and update)   
Require: prompt partition $X _ { t } ;$ current policy $\pi _ { t } ;$ gold responses $\{ y _ { x } ^ { \mathrm { g o l d } } \}$ ; annotator A; samples per prompt $K \left( K = 5 \right.$   
unless stated); margin $m = 2 ;$ relaxed margin $m ^ { \prime } = 1 ;$ look-back fractions $( 0 . 3 0 , 0 . 1 5 , \ldots )$   
1: $D _ { t } ^ { \mathrm { n e w } } \gets \emptyset$   
2: for each prompt $x \in X _ { t }$ do   
3: sample $y _ { 1 } , \dots , y _ { K } \sim \pi _ { t } ( \cdot \mid x )$ with temperature 0.7, top-p 0.95; drop samples containing degenerate character   
sequences   
4: $C \gets ( y _ { 1 } , \dots , y _ { K } , y _ { x } ^ { \mathrm { g o l d } } ) ;$ query A once with x and C; skip x unless the reply gives one score $s ( y )$ per $y \in C$   
5: $P  \{ y \in C : s ( y ) \geq 5 \}$   
6: for each $y ^ { + } \in P$ and each $y ^ { - } \in C \setminus \{ y ^ { + } \}$ do   
7: $\mu  m ^ { \prime } \mathbf { i f } \ y ^ { + }$ is a student sample and $\dot { y } ^ { - } = y _ { x } ^ { \mathrm { g o l d } }$ else m   
8: if $s ( y ^ { + } ) - s ( y ^ { - } ) \geq \mu$ then   
9: $D _ { t } ^ { \mathrm { n e w } }  D _ { t } ^ { \mathrm { n e w } } \cup \{ ( x , y ^ { + } , y ^ { - } ) \}$   
10: end if   
11: end for   
12: end for   
13: $D _ { t } \gets D _ { t } ^ { \mathrm { n e w } }$   
14: for $k = 1 , 2 , \ldots , t - 1$ do   
15: add to $D _ { t }$ at most a fraction $0 . 3 0 \cdot 2 ^ { - ( k - 1 ) }$ of $D _ { t - k }$ , the training set of iteration $t - k$ (which includes its own   
look-back pairs), unchanged   
16: end for   
17: train LoRA adapters on $\pi _ { t }$ with the DPO loss $( \beta = 0 . 1 )$ on $D _ { t }$ for one epoch (2,000-step cap in the margin sweep),   
with reference policy $\pi _ { t } ;$ merge them to obtain $\pi _ { t + 1 }$