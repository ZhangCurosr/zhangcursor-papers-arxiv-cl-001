(b) Trained policies

# Copying Is Not Grounding: Training Faithful Long-Form RAG with Authored-Sentence Credit

Hoang Pham<sup>1</sup> Duc Anh Vu<sup>1</sup> Duy Tung Doan<sup>2</sup>

Cong-Duy Nguyen<sup>3</sup> Anh Tuan Luu<sup>1,3</sup>\*

<sup>1</sup>Nanyang Technological University <sup>2</sup>Viettel AI, Viettel Group <sup>3</sup>VinUniversity

Email: pham0100@ntu.edu.sg

## Abstract

In search engines and assistants, long-form RAG answers can mislead readers with claims their sources do not support. Faithfulness rewards, the share of supported sentences, train models toward fewer such claims. We show that these rewards are hacked by copying. A sentence copied from the passages is supported almost by construction, so trained models learn to copy rather than ground what they write. Common defenses only move the hack: a KL or copy penalty leaves copying in place or trades it for shorter, emptier answers and refusals. GRPO, the standard RL algorithm, makes it worse by giving every sentence its answer’s advantage, reinforcing copied and grounded sentences alike. We propose Authored-Sentence Credit (ASC)<sup>1</sup>, which fixes both by moving GRPO’s reward and advantage from the answer to the sentence level. The reward credits only sentences that are authored rather than copied, say more than the question, and are supported by the passages. The advantage compares each sentence with every sentence sampled for the question, so one that earns no credit ranks last and is never reinforced. Across four models and two in-domain, three held-out corpora, only ASC lowers the error rate of authored sentences and writes more supported ones in every setting, by 38% and 46% at the median under held out evaluators, giving readers more grounded information. It takes none of these shortcuts, and two LLM judges comparing whole answers prefer no baseline to the base model but rate ASC’s answers on par with it. We conclude that a faithfulness reward should credit what a model writes, not what it copies.

## 1 Introduction

Retrieval-augmented generation (RAG; Lewis et al., 2020) now answers open questions in search engines and assistants, yet its long-form answers often state claims their sources do not support, in web search (Liu et al., 2023) as in law and medicine (Magesh et al., 2025; Wu et al., 2025). Faithfulness, the share of an answer’s sentences a checker finds supported by the passages (Gao et al., 2023b; Min et al., 2023; Tang et al., 2024), is now a training signal against such claims, as a reward for reinforcement learning or to rank preference pairs (Huang et al., 2024; Chen et al., 2026; Tian et al., 2024).

![](images/0d23d523c15ac4b4f2f06f060d979393557344d3417654d4aa90fd3c11fa2fdf.jpg)

![](images/2b07d7066f168b04ad3f1324bbfae6ca26c4a2019486d683ccab8f32e3f73c2d.jpg)  
Figure 1: A faithfulness reward can be hacked by copying (ASQA). (a) Under the reward’s checker, the sentences a model writes itself are unsupported 3.2 to 3.7 times as often as those it copies. (b) Qwen3.5-9B after training: GRPO copies and nearly stops writing, GRPO + KL writes less, and ASC copies less and writes more supported sentences of its own (held-out checker).

The reward has a cheaper way up. A sentence copied from the passages is supported almost by construction: on ASQA, the sentences Qwen3.5- 9B writes itself are unsupported 3.7 times as often as those it copies, and those of Llama-3.1-8B 3.2 times (Figure 1a). The reward therefore rises with every sentence a model copies instead of writing.

Trained policies exploit this gap, a form of reward hacking (Skalse et al., 2022). GRPO (Shao et al., 2024) raises the copy share of Qwen3.5-9B from 38.8 to 72.8 (Figure 1b) and shrinks its answers from 74 to 14 words: the reward climbs because the model stops writing, not because it grounds better. Common defenses only move the hack between three shortcuts (copying, writing less and writing emptier sentences): on the same model, a KL penalty holds copying at the base level, but refusals more than double and answers grow shorter and emptier, and a copy penalty makes it decline most questions. GRPO itself makes it worse: it gives every token of an answer the same advantage, its update weight, so a copied or empty sentence in a high-reward answer is reinforced and a grounded one in a low-reward answer pushed down.

We propose Authored-Sentence Credit (ASC), to our knowledge the first faithfulness reward that pays only for what the model writes itself (Figure 4). It changes GRPO in two places and adds no model. Its reward credits a sentence only if it is not copied, says more than the question (not vacuous), and is supported by the passages. Its advantage scores each sentence against all sentences sampled for the question, so each sentence is judged by its own credit, not its answer’s. Together, they guarantee that a copied, vacuous or unsupported sentence ranks last in its group and is never reinforced.

We train four models from two families on two corpora and test them on five, three held out. Every baseline, even a sentence-level reward, lowers the error rate of authored sentences at the median, yet takes a shortcut in nearly every setting. ASC alone lowers it while writing more, with no shortcut in any setting: at the median, it cuts that error rate by 38% under an independent judge and writes 46% more supported sentences of its own, giving readers more grounded information; human annotators confirm the drop without the judge. A higher reward alone does not give better answers: every baseline raises the reward, yet two LLM judges comparing whole answers prefer none to the base model but rate ASC on par with it and above every baseline. In summary, our contributions are as follows.

• Analysis. We show that faithfulness rewards pay for copying, that KL and copy penalties only move the hack to other shortcuts, and that GRPO’s answer-level advantage barely tells copied from grounded sentences.

• Method. We propose ASC, to our knowledge the first faithfulness reward that credits only authored sentences, and prove that no uncredited sentence is ever reinforced.

• Results. Only ASC lowers the authored error rate and writes more supported sentences in all settings, with no shortcut and no quality loss.

## 2 Related Work

Faithfulness metrics favor copying. Long-form answers are scored statement by statement against their sources (Gao et al., 2023b; Min et al., 2023), and small checkers make this scoring cheap enough to serve as a reward (Tang et al., 2024; Zha et al., 2023). In summarization, such metrics favor extractive output, so faithfulness gains often come from copying (Ladhak et al., 2022; Dreyer et al., 2023), and an entailment reward can trade informativeness for consistency (Roit et al., 2023). We study this bias as a reward hack in long-form RAG, where copying has a cost for readers: in generative search, more precise citations come with more copying from the sources and lower perceived utility (Liu et al., 2023). Trivially true claims also inflate factuality scores (Jiang et al., 2025); our vacuity test removes them by checking each sentence against the question alone, as context-contrast methods do (Shi et al., 2024; Cohen-Wang et al., 2024).

Grounded generation and reward hacking. Faithfulness and factuality objectives train models through preference optimization (Tian et al., 2024; Song et al., 2025), reinforcement learning (Si et al., 2025; Huang et al., 2024), self-reflection (Asai et al., 2024) and citation tuning (Ye et al., 2024), and one method even trains RAG models to copy more so that they hallucinate less (Long et al., 2026). Optimizing such a proxy invites reward hacking (Skalse et al., 2022; Gao et al., 2023a), best studied for length (Singhal et al., 2024); for factuality, precision-only metrics and rewards favor short, less detailed answers (Wei et al., 2024; Chen et al., 2026). A KL penalty does not prevent hacking under heavy-tailed reward error (Kwa et al., 2024), and optimization amplifies a reward’s biases (Shapira et al., 2026; Moya et al., 2026). We show that faithfulness rewards are hacked by copying, and that common defenses only move the hack.

Fine-grained credit assignment. Dense signals score each sentence (Wu et al., 2023) or reasoning step (Lightman et al., 2024). GRPO normalizes rewards within a group of answers (Shao et al., 2024), and GiGPO within groups of steps (Feng et al., 2025). The closest rewards score facts or sentences for closed-book correctness (Wen et al., 2025; Gu et al., 2025) or for support in RAG (Huang et al., 2024), but none asks whether a sentence was copied. ASC withholds credit from copied and vacuous sentences and compares sentences, not answers, so these are never reinforced.

## 3 Problem Formulation and Setup

Task and reward. A prompt consists of a question $q$ and retrieved passages $P ,$ , and a policy $\pi _ { \theta }$ writes a long-form answer $y = ( y _ { 1 } , \dotsc , y _ { n } )$ of $n$ sentences. A checker g (e.g., Bespoke-MiniCheck-$7 \mathrm { B } ;$ Tang et al., 2024; Bespoke Labs, 2024) returns $g ( y _ { s } , P ) \in [ 0 , 1 ]$ , its probability that $P$ supports $y _ { s }$ , and the sentence counts as supported when $g ( y _ { s } , P ) \geq \kappa .$ . The faithfulness reward that training maximizes and evaluations report is the share of supported sentences,

$$
\begin{array} { r } { R ( y ) ~ = ~ { \frac { 1 } { n } } \sum _ { s = 1 } ^ { n } { \bf 1 } [ g ( y _ { s } , P ) \geq \kappa ] . } \end{array}\tag{1}
$$

Training. GRPO (Shao et al., 2024) samples $G$ answers to each prompt and gives every token of answer i the same advantage, its reward standardized over the group, $A ^ { ( i ) } = ( { \cal R } ( y ^ { ( i ) } ) - \mu ) / ( \sigma + \epsilon )$ with $\mu , \sigma$ the group’s mean and standard deviation; DPO (Rafailov et al., 2023) instead learns offline from pairs of answers, preferring the one with higher R.

Copied, authored and vacuous sentences. The copy share $c ( y _ { s } )$ of a sentence is the fraction of its words inside verbatim matches of at least m words with $P ,$ , and $c ( y )$ is the same fraction over an answer. A sentence is copied if $c ( y _ { s } ) \geq \tau$ and authored otherwise; $\rho ( y )$ is the share of copied sentences in an answer. A sentence is vacuous if it passes the same support test with the question in place of the passages, $g ( y _ { s } , q ) \ge \kappa \colon$ it needs no evidence to count as supported, like “The song was released.” for a question about a song. Only authored, non-vacuous sentences show whether a model can support what it writes itself; we call one grounded when $P$ supports it.

Goal and shortcuts. Training should make a model’s own sentences more reliable: it should lower the authored error rate $e _ { a } ( \pi )$ , the share of authored sentences that the passages do not support, which is 2–40% for the base model $\pi _ { 0 }$ . It should do so without writing less: the copy share $^ { c , }$ vacuous share v and refusal rate d (the share of answers that decline) should not rise above those of $\pi _ { 0 } .$ , nor the answer length ℓ fall below it. R does not encode this goal and can be hacked through three shortcuts, copying more, writing less and writing emptier sentences, with refusal as the extreme; each raises $R ,$ and writing less can even lower $e _ { a }$ by dropping risky sentences, with no authored sentence becoming more reliable. Section 4 shows that training on R takes these shortcuts, and Section 5 introduces a signal that improves on $\pi _ { 0 }$ without them.

Metrics. Two lead metrics measure the goal; the others detect shortcuts and check quality (Table 6).

• Reliability (Err). $e _ { a } ,$ judged with the passages by Gemma-4-31B (Gemma Team, 2026), which comes from a different family than the reward’s checker and is never used in training.

• Content (Supp.). Supported authored sentences per answer under the held-out MiniCheck-Flan-T5-Large (Tang et al., 2024); unlike Err, it falls when a model simply writes less.

• Shortcuts (Copy, Vac., Ref., Words). $c , v ,$ d and $\ell ;$ a policy takes a shortcut when $^ { c , }$ v or d exceeds that of $\pi _ { 0 }$ by a set margin, or ℓ falls below a set fraction of it (values in Section 6.1).

• Checks. AlignScore (Zha et al., 2023) as a second held-out checker, ASQA exact match (EM; Gao et al., 2023b), and two LLM judges of whole answers (Section 6.2).

Models and data. We train four instruction models from two families, Qwen3.5-9B, Qwen3.5-4B (Qwen Team, 2026), Llama-3.1-8B and Llama-3.2- 3B (Grattafiori et al., 2024), and evaluate each on five corpora, 20 settings of model and corpus in all. Training pools ASQA (Stelmakh et al., 2022) with ALCE’s top five passages (Gao et al., 2023b) and the question-answering part of RAGTruth (Niu et al., 2024); we evaluate in domain on their evaluation sets. Three further corpora, ClapNQ (Rosenthal et al., 2025), ELI5 (Fan et al., 2019) with ALCE’s passages and ExpertQA (Malaviya et al., 2024), are held out entirely, so a gain on them cannot come from fitting the training data. On ELI5 and ExpertQA, where base models err most and copy least, a good answer has to be written, not assembled from the passages.

## 4 Why Faithfulness Rewards Are Hacked

We ask what $R$ pays for in base-model answers, what a policy trained on it learns, and what GRPO’s advantage sees inside an answer; each finding yields one requirement that Section 5 meets.

Finding 1: copied sentences are rarely unsupported, so the reward pays for copying. We sample eight answers per question from each base model on ASQA, RAGTruth, ClapNQ and ELI5. Under the reward’s checker, authored sentences are unsupported 2.3 to 4.4 times as often as copied ones on average (Figure 2a), and more often for every model and corpus. The gap is not an artifact of the checker: two held-out checkers find it almost everywhere, and so do human labels on RAGTruth’s responses, 12.2% against 1.9% (Figure 2b). With $s _ { c }$ and $s _ { a }$ the support rates of copied and authored sentences, the reward of Eq. 1 decomposes as

![](images/34333ed6288594d072314850862f928c0e1ba63a23609b254cabe26a43570571.jpg)  
Figure 2: Unsupported rate of authored and copied sentences, with their ratio. (a) Base-model samples under the reward’s checker (mean over models). (b) The same samples under MiniCheck-Flan-T5 (T5) and AlignScore (Align), mean over models and corpora, and human labels on RAGTruth’s responses.

$$
R \ : = \ : s _ { a } \ : + \ : \rho \left( s _ { c } - s _ { a } \right) ,\tag{2}
$$

a line in ρ with slope $s _ { c } - s _ { a } > 0 ;$ at fixed authored reliability, every sentence copied instead of written raises R (proof in Appendix A.1). Sampled answers agree: among answers to the same question, those that copy more earn more reward for every model and corpus (slopes in Appendix C.1). Yet a copied sentence inherits its source’s reliability, so its credit says nothing of the model’s own grounding. This gives the first requirement, (R1): a copied sentence earns nothing, however well supported.

<table><tr><td>Training</td><td>Reward Copy</td><td>Words Vac. Ref. EM</td></tr><tr><td>Base model</td><td>87.1 38.8</td><td>73.5 21.5 6.1 45.5</td></tr><tr><td>GRPO</td><td>98.9 72.8</td><td>13.7 18.1 0.0 27.6</td></tr><tr><td>GRPO + KL</td><td>92.3 39.2</td><td>60.0 28.5 14.7 41.8</td></tr><tr><td>Copy penalty</td><td>92.0 11.8</td><td>36.9 67.4 64.5 20.2</td></tr><tr><td>DPO</td><td>94.6 53.4</td><td>55.7 16.9 1.7 41.8</td></tr><tr><td>DPO, copy-matched</td><td>90.4 44.9</td><td>59.7 21.5 6.1 43.8</td></tr></table>

Table 1: Qwen3.5-9B trained to raise R, on ASQA; metrics of Section 3, rates in %. Bold: a shortcut. Llama-3.1-8B and training curves: Appendix C.2.

Finding 2: training takes the shortcuts, and common defenses only move between them. We train Qwen3.5-9B to raise R with GRPO, alone or with one of two defenses, a KL penalty (Ziegler et al.,

![](images/8dd56db22496afc07ffa7a34248e51098d0e952017a112b86c9468d9121b5293.jpg)  
Figure 3: Qwen3.5-9B on ASQA over training (0: base model): GRPO copies and nearly stops writing, and each defense swaps one shortcut for another.

2020; Ouyang et al., 2022) or a copy penalty, and with DPO (Table 1). Plain GRPO nearly saturates the reward (98.9) through the cheapest shortcuts, and the answers get worse: copy share nearly doubles within half of training, answers shrink from 74 to 14 words (Figure 3), and exact match falls from 45.5 to 27.6. Each defense closes one shortcut and opens another: the KL penalty holds copying at the base level, but answers grow shorter and emptier and refusals rise, and the copy penalty cuts copying, but its vacuous share rises to 67.4% and it declines 64.5% of questions. DPO copies too, even on pairs whose answers copy nearly equally: preferring better-supported answers alone pulls it toward the passages. The two penalties take a shortcut on nine in ten model–corpus pairs (grey in Table 2). R pays for any sentence the checker cannot reject, so penalizing one quantity only reroutes the optimizer. This gives the second requirement, (R2): a vacuous sentence earns nothing, and dropping sentences or refusing never pays.

Finding 3: GRPO’s advantage barely tells a copied sentence from a grounded one. GRPO’s advantage belongs to the whole answer, so a sentence is rewarded for the answer it sits in: a copied or vacuous sentence in a high-reward answer is pushed up, and a grounded one in a low-reward answer is pushed down. Copying therefore barely changes a sentence’s chance of being reinforced: on Finding 1’s samples, of the sentences GRPO moves, it pushes up 72.0% of the copied ones and 75.5% of the grounded ones, and even 58.8% of the vacuous ones; grounded ones are pushed up more often everywhere, but by at most 17 points (Appendix C.3). An answer-level advantage thus cannot steer the policy away from copying: it reinforces whatever the high-reward answers contain. This gives the third requirement, (R3): sentences are compared with sentences, so that each advantage reflects the sentence’s own reliability.

![](images/86890bd84c4ff41122f060e05053fcbfad777bd3ac9ad0151ced313c6e094a07.jpg)  
Figure 4: One ASC step in GRPO. 1) The policy samples G answers to a question q with passages P. 2) A sentence earns credit only if it is authored, not vacuous and supported by P. 3) Each credit is compared with those of all sentences in the group, across answers. 4) Every token takes its sentence’s advantage; the rest of GRPO is unchanged. Example (Qwen3.5-9B, ELI5): answer $y ^ { ( i ) }$ earns the full reward although one sentence is copied and one vacuous; GRPO pushes all three up (+0.5), ASC only the grounded one (+1.0; the others −1.0).

## 5 ASC: Authored-Sentence Credit

ASC meets (R1)–(R3) with two changes to GRPO, which sentences earn credit (Section 5.1) and what each credit is compared with (Section 5.2), and leaves the rest unchanged (Section 5.3; Figure 4).

## 5.1 Which sentences earn credit

For a sampled answer y with sentences $y _ { 1 } , \ldots , y _ { n } ,$ the credit of $y _ { s }$ is the product of three tests,

$$
r _ { s } = \underbrace { \mathbf { 1 } \big [ c ( y _ { s } ) < \tau \big ] } _ { \mathrm { c o p y } } \underbrace { \mathbf { 1 } \big [ g ( y _ { s } , q ) < \kappa \big ] } _ { \mathrm { v a c u i t y } } \underbrace { \phi ( g ( y _ { s } , P ) ) } _ { \mathrm { s u p p o r t } } .\tag{3}
$$

The copy test meets (R1): a copied sentence owes its support to repeating P, not to the policy’s grounding, so the test gives it what an unsupported sentence earns. The vacuity test meets the first half of (R2): a sentence that the checker accepts from the question alone needs no evidence, so its support is no evidence of grounding either. The support test credits the remaining sentences with the graded credit

$$
\begin{array} { r } { \phi ( p ) = \operatorname* { m a x } \Big \{ 0 , 1 - \frac { \ln p } { \ln \kappa } \Big \} . } \end{array}\tag{4}
$$

Properties of the credit. $\phi$ is zero for $p \leq \kappa$ and one at $p = 1$ , so the credited sentences are exactly those above the checker’s threshold. Writing $h ( p ) = - \ln { p }$ for the checker’s surprisal at “supported”, $\phi ( p ) = 1 - h ( p ) / h ( \kappa )$ on (κ, 1]: ϕ is the only credit that is affine in surprisal, zero at the threshold and one at certainty, and it is strictly increasing and concave in p, rising fastest just above the threshold (Appendix A.2). The binary credit $\mathbf { 1 } [ p \geq \kappa ]$ pays a sentence just above the threshold as much as one the checker is certain of; the graded credit keeps the checker’s confidence and pulls the policy toward sentences the checker is sure of. Every credit lies in [0, 1], and a sentence that fails any test earns the minimum, $r _ { s } = 0$

## 5.2 A sentence baseline

Where GRPO standardizes answers (Section 3), ASC standardizes sentences. For a group of G answers with M sentences in total, let $\mu$ and σ be the mean and standard deviation of the credits of all M sentences. Sentence s of answer i receives the advantage

$$
\begin{array} { r } { a _ { s } ^ { ( i ) } = \frac { r _ { s } ^ { ( i ) } - \mu } { \sigma + \epsilon } , \qquad \epsilon > 0 . } \end{array}\tag{5}
$$

Proposition 1 shows that the credit and the baseline together meet (R1)–(R3): whatever the answers contain, a sentence failing a test ranks strictly below every sentence that earns credit.

Proposition 1 (No shortcut earns advantage). For any group and the advantage of $E q . 5 .$

(i) Credit alone sets the order: $a _ { s } ^ { ( i ) } < a _ { t } ^ { ( j ) }$ if and only $i f r _ { s } ^ { ( i ) } < r _ { t } ^ { ( j ) }$ , for sentences of the same or of different answers (R3).

(ii) Shortcuts rank last: a copied, vacuous or unsupported sentence receives the lowest advantage of its group, $- \mu / ( \sigma + \epsilon )$ , negative if any sentence earns credit (R1, R2).

(iii) Average sentences earn nothing: advantages sum to zero over the group, and a sentence ofaverage credit receives zero.

(iv) Length-free advantages: given the group’s µ and $\sigma ,$ a sentence’s advantage depends on neither its own length nor its answer’s (R2).

All four parts follow from two facts: Eq. 5 applies one strictly increasing affine map to every credit of the group, and a sentence that fails a test has the minimum credit. GRPO’s answer-level advantage breaks (i) and (ii), as Finding 3 measured, and (iv) as well (proofs in Appendix A.3).

## 5.3 Training objective

Every token t of sentence s in answer i takes that sentence’s advantage, $\hat { A } _ { i , t } ~ = ~ a _ { s } ^ { ( i ) }$ , in place of $\mathrm { G R P O ^ { \circ } s } \ A ^ { ( i ) }$ ; tokens outside every sentence, the whitespace between sentences and the end-ofsequence token, take zero. The objective is otherwise GRPO’s (Algorithm 1, Appendix B):

$$
\begin{array} { r } { \mathcal { I } ( \theta ) = \frac { 1 } { N } \sum _ { i , t } \Big ( w _ { i , t } ( \theta ) \hat { A } _ { i , t } - \beta \mathbb { D } _ { i , t } \Big ) , } \end{array}\tag{6}
$$

where $w _ { i , t } ( \theta ) = \pi _ { \theta } ( y _ { t } ^ { ( i ) } \mid q , P , y _ { < t } ^ { ( i ) } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } ^ { ( i ) } \mid$ $q , P , y _ { < t } ^ { ( i ) } )$ is the importance ratio, $\mathbb { D } _ { i , t } \mathbf { G } \mathbf { R P O } ^ { \ast } \mathbf { s }$ pertoken KL estimate (Shao et al., 2024) toward the reference $\pi _ { \mathrm { r e f } }$ , the base model, and N the number of answer tokens in the batch (Yu et al., 2025).

Gradient form. Each batch is used for one update, so the gradient is taken where $w _ { i , t } = 1$ , and the log-probabilities of a sentence’s tokens add up to its log-likelihood. Up to the KL term, the gradient of Eq. 6 is a policy gradient over sentences,

$$
\begin{array} { r } { \nabla \mathcal { I } = \frac { 1 } { N } \sum _ { i , s } a _ { s } ^ { ( i ) } \nabla \ln \pi _ { \theta } \big ( y _ { s } ^ { ( i ) } \mid y _ { < s } ^ { ( i ) } \big ) , } \end{array}\tag{7}
$$

with $\nabla = \nabla _ { \theta } , y _ { < s } ^ { ( i ) }$ the text before sentence s and the prompt $( q , P )$ left implicit (Appendix A.4). GRPO’s gradient weights the log-likelihood of the whole answer by $A ^ { ( i ) }$ , so every sentence moves with its answer; ASC moves each sentence by its own advantage, whatever the rest of its answer contains, which is the formal content of (R3).

Relation to GRPO and to sentence rewards. If every answer had a single sentence, Eq. 5 would reduce to GRPO’s own standardization with r as the reward, so ASC is GRPO with a finer unit of comparison. Without the first two tests and with the binary credit, ASC becomes a sentence-level support reward (Wu et al., 2023; Huang et al., 2024), the sentence-reward baseline of Section 6.1. The only added checker cost is one question-only call per authored sentence. ASC needs no critic, no reference answers and no penalty weight.

## 6 Experiments

## 6.1 Setup

Training. All policies are trained with GRPO at full weight on the pooled ASQA and RAGTruth training questions, with $G = 8$ answers per question, a KL coefficient $\beta \ : = \ : 0 . 0 4$ , one zero-shot prompt and the same hyper-parameters for every method (Appendix B). Every method uses the same reward checker, Bespoke-MiniCheck-7B (Tang et al., 2024; Bespoke Labs, 2024), and the same thresholds, $\kappa = \tau = { \frac { 1 } { 2 } }$ and $m = 5$ words, so the methods differ only in the training signal.

Baselines. We compare ASC with three baselines. GRPO + KL trains on the faithfulness reward R of Eq. 1, the standard recipe. Copy penalty trains on $R - \lambda c ( y )$ , the direct fix for the copy shortcut, with λ set to the support gap $s _ { c } - s _ { a }$ of the model’s own samples on the training questions, so that by Eq. 2 a copied sentence earns no more than an authored one. Sentence reward, a sentence-level support reward in the style of fine-grained RLHF (Wu et al., 2023), is ASC without its first two tests and graded credit, so it keeps the sentence baseline.

Evaluation. We decode greedily and evaluate on all five corpora with the metrics of Section 3. A policy takes a shortcut on a corpus when its copy share, vacuous share or refusal rate exceeds the base model’s by more than 3 points, or its answers fall below 85% of the base model’s words. Rows average the training seeds where a method has several (ASC on the Qwen models, binary credit) and are single runs otherwise; Appendix D gives percorpus results, every run and bootstrap significance tests, and Appendix I example answers.

## 6.2 Main results

Only ASC lowers the error rate while writing more. Every method lowers the authored error rate at the median, ASC the most (38%, against 31 to 36%), but a model can also lower it by writing less, so Err alone does not separate them (Table 2). What separates them is how much grounded content the model still writes (Supp.): ASC writes the most supported authored sentences on every model, 46% more than the base model at the median and more in every setting, whereas GRPO + KL writes fewer than the base model on three of the four models. The lower rate is not a dilution by the extra sentences: unsupported authored sentences per answer fall too, in 16 of the 20 settings. Figure 5 shows that the cut holds at every error level, including the hardest settings, where the base model errs on about 40% of its sentences. It is even larger on the held-out corpora, never seen in training, than in domain (median 38% against 32%).

<table><tr><td rowspan="2">Method</td><td colspan="6">Authored error rate (Err) ↓</td><td colspan="5">Answer metrics (mean)</td><td rowspan="2">ASQA EM</td></tr><tr><td>ASQA RAGT. ClapNQ</td><td></td><td></td><td></td><td>ELI5 Expert Mean Supp.↑ Copy</td><td></td><td></td><td></td><td>Words</td><td>Vac.</td><td>Ref.</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>5.09</td><td>2.31</td><td>3.61</td><td>6.04</td><td>6.21</td><td>4.65</td><td></td><td>2.17 28.06</td><td>97.40</td><td>27.88</td><td>4.94</td><td>45.45</td></tr><tr><td>GRPO + KL</td><td>3.97</td><td>1.20</td><td>3.19</td><td>5.38</td><td>5.85</td><td>3.92</td><td>2.03</td><td>27.17</td><td>83.00</td><td>32.23</td><td>23.32</td><td>41.78</td></tr><tr><td>Copy penalty</td><td>1.45</td><td>1.20</td><td>0.40</td><td>1.13</td><td>0.42</td><td>0.92</td><td>1.22</td><td>6.18</td><td>29.48</td><td>82.31</td><td>79.65</td><td>20.22</td></tr><tr><td>Sentence reward</td><td>3.74</td><td>0.85</td><td>3.09</td><td>3.51</td><td>4.25</td><td>3.09</td><td></td><td>2.1630.85</td><td>92.08</td><td>25.69</td><td>11.78</td><td>44.15</td></tr><tr><td>ASC</td><td>3.96</td><td>1.13</td><td>2.15</td><td>3.37</td><td>3.74</td><td>2.87</td><td>3.04 22.18</td><td></td><td>108.85</td><td>13.28</td><td>0.22</td><td>44.93</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>8.57</td><td>2.87</td><td>5.84</td><td>8.84</td><td>8.72</td><td>6.97</td><td>2.16 29.84</td><td></td><td>103.77 26.31</td><td></td><td>5.33</td><td>46.24</td></tr><tr><td>GRPO + KL</td><td>3.17</td><td>2.03</td><td>2.94</td><td>3.94</td><td>3.49</td><td>3.11</td><td>1.59</td><td>22.67</td><td>59.41</td><td>53.77</td><td>43.87</td><td>36.23</td></tr><tr><td>Copy penalty</td><td>8.17</td><td>3.06</td><td>4.92</td><td>6.73</td><td>9.07</td><td>6.39</td><td>2.30</td><td>19.73</td><td>86.35</td><td>36.27</td><td>28.13</td><td>36.34</td></tr><tr><td>Sentence reward</td><td>4.28</td><td>2.22</td><td>4.17</td><td>5.91</td><td>5.91</td><td>4.50</td><td>2.42 32.85</td><td></td><td>108.07</td><td>21.98</td><td>4.40</td><td>46.42</td></tr><tr><td>ASC</td><td>6.47</td><td>2.25</td><td>4.00</td><td>6.58</td><td>7.69</td><td>5.40</td><td>2.97 23.98</td><td></td><td>8 110.77 12.89</td><td></td><td>0.95</td><td>45.67</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>11.69</td><td>8.46</td><td>13.95</td><td>27.89</td><td>27.3417.87</td><td></td><td></td><td>1.0637.98</td><td>50.41 24.24</td><td></td><td>4.65</td><td>35.14</td></tr><tr><td>GRPO + KL</td><td>6.56</td><td>6.69</td><td>5.58</td><td>21.05</td><td>22.48</td><td>12.47</td><td></td><td>1.1041.26</td><td>49.41</td><td>25.50</td><td>4.03</td><td>34.50</td></tr><tr><td>Copy penalty</td><td>8.04</td><td>6.02</td><td>8.75</td><td>17.63</td><td>20.53</td><td>12.19</td><td></td><td>1.4115.54</td><td>39.79</td><td>29.63</td><td>6.96</td><td>28.17</td></tr><tr><td>Sentence reward</td><td>4.77</td><td>5.06</td><td>5.53</td><td>18.18</td><td>19.96</td><td>10.70</td><td></td><td>1.0442.58</td><td>49.7623.03</td><td></td><td>4.00</td><td>34.25</td></tr><tr><td>ASC</td><td>6.03</td><td>3.94</td><td></td><td>7.23 17.38</td><td></td><td>19.89 10.89</td><td></td><td>1.5724.77</td><td></td><td>55.13 13.71</td><td>0.68</td><td>35.61</td></tr><tr><td>Llama-3.2-3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>18.87</td><td>11.71</td><td></td><td>14.4738.67</td><td></td><td>40.53 24.85</td><td></td><td>0.9340.28</td><td></td><td>52.58 20.05</td><td>4.27</td><td>34.73</td></tr><tr><td>GRPO + KL</td><td>14.41</td><td>5.68</td><td>9.95</td><td>25.42</td><td>28.00</td><td>16.69</td><td>0.89</td><td>49.39</td><td>50.98</td><td>20.47</td><td>3.20</td><td>32.42</td></tr><tr><td>Copy penalty</td><td>10.73</td><td>8.99</td><td>9.94</td><td>26.76</td><td></td><td>27.0616.70</td><td>1.23</td><td>18.86</td><td>37.82</td><td>28.29</td><td>7.53</td><td>25.79</td></tr><tr><td>Sentence reward</td><td>11.82</td><td>7.61</td><td></td><td>6.3123.84</td><td></td><td>24.0114.72</td><td></td><td>0.9349.76</td><td></td><td>52.3117.36</td><td>2.00</td><td>33.03</td></tr><tr><td>ASC</td><td>15.00</td><td>7.06</td><td></td><td>8.70 24.88</td><td>24.96 16.12</td><td></td><td></td><td>1.5724.54</td><td></td><td>55.86 14.09</td><td>2.47</td><td>35.66</td></tr></table>

Table 2: Main results (0–100). Err: authored error rate on two in-domain (ASQA, RAGTruth) and three held-ou corpora, with its mean; answer metrics: means over the five corpora; EM: exact match on ASQA. Grey Err: a shortcut on that corpus (Section 6.1), not ranked; grey mean: a shortcut on at least one corpus. Bold: lowest remaining Err and most supported authored sentences per answer (Supp.); underlined: second-lowest Err.

![](images/129061f0f7ffe35576640bede4ae890cab1cf4c5111d9dc5f7535bf1663da0e3.jpg)  
Figure 5: Authored error rate of the base model against ASC, one point per model and corpus (hollow: in domain); all lie below the diagonal (dashed: median).

ASC takes no shortcut; the baselines take one almost everywhere. In every setting ASC copies less than the base model, writes fewer vacuous sentences and refuses less often, so none of its Err values is grey in Table 2, under any margin from 1 to 10 points. The baselines buy their lower error rates with a shortcut, and closing one opens another, as in Finding 2: GRPO + KL declines up to 43.9% of questions on the Qwen models, against about 5% before training, and copies more on the Llama models; the copy penalty cuts copying but declines 79.7% of questions on Qwen3.5-9B; and the sentence reward copies more than the base model in every setting, since scoring sentences one by one does not close the copy shortcut. Where a baseline takes no shortcut (5 of 60), its error rate is never significantly below ASC’s, and ASC writes significantly more supported authored sentences in all five, so no baseline matches it even there.

ASC gains faithfulness at no cost in quality. Two LLM judges, Gemma-4-31B and Mistral-Small-3.2-24B (Mistral AI, 2025) from a third model family, pick, in both orders, the answer that serves a reader better, under a rubric fixed on base-model probes before any trained answer was judged, to favor neither copied text nor padded answers (Appendix E). For every baseline the faithfulness gains cost quality (Table 3): each raises the reward over the base model, yet neither judge prefers any of them to it. ASC is the only method that pays nothing: both judges rate its answers on par with the base model’s while they carry fewer errors and more supported sentences, and prefer them to those of every baseline. Blind human annotators, with the same rubric, prefer ASC’s answer to the base model’s more often than the reverse (22 against 12 of 75 pairs) and GRPO + KL’s (28 against 14; Appendix G). Exact match on ASQA stays within one point of the base model for ASC, while the copy penalty loses up to 25 points (Table 2).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reward</td><td colspan="2">Against base</td><td colspan="2">ASC against it</td><td rowspan="2">Cov.</td></tr><tr><td></td><td>Gemma Mistral</td><td>Gemma Mistral</td><td></td></tr><tr><td>GRPO + KL</td><td>92.7</td><td>43.6</td><td>43.4</td><td>56.4</td><td></td><td>55.942.0</td></tr><tr><td>Copy penalty</td><td>90.2</td><td>34.8</td><td>30.8</td><td>64.6</td><td>68.9</td><td>31.4</td></tr><tr><td>Sentence reward</td><td>92.2</td><td>49.5</td><td>48.0</td><td>51.5</td><td>51.1</td><td>45.1</td></tr><tr><td>ASC</td><td>90.6</td><td>51.0</td><td>49.1</td><td></td><td></td><td>46.8</td></tr></table>

Table 3: Answer quality over the 20 settings: training reward, win rate (%, a tie counts half) against the base model and of ASC against each method under two judges, and coverage (%) of the reference answer; the base model’s reward is 83.5 and its coverage 45.0.

## 6.3 Each part of ASC contributes

<table><tr><td rowspan="2">Variant</td><td colspan="5">Err</td><td colspan="3">Mean</td></tr><tr><td>ASQA RAGT. ClapNQ ELI5 Expert Supp. Copy Vac.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASC</td><td>4.0</td><td>1.1</td><td>2.2</td><td>3.4</td><td>3.7</td><td></td><td>3.022.2 13.3</td><td></td></tr><tr><td>– copy test</td><td>2.4</td><td>1.5</td><td>1.6</td><td>3.5</td><td>3.5</td><td>2.6</td><td>33.3 11.2</td><td></td></tr><tr><td>— vacuity test</td><td>5.2</td><td>1.0</td><td>1.6</td><td>4.6</td><td>7.5</td><td>2.9</td><td>21.3 19.1</td><td></td></tr><tr><td>Sentence reward</td><td>3.7</td><td>0.9</td><td>3.1</td><td>3.5</td><td>4.3</td><td>2.2</td><td>30.925.7</td><td></td></tr><tr><td>Answer advantage</td><td>5.4</td><td>2.3</td><td>1.9</td><td>5.1</td><td>5.1</td><td>3.3</td><td>25.1 18.3</td><td></td></tr><tr><td>Binary credit</td><td>4.6</td><td>1.7</td><td>2.5</td><td>4.1</td><td>4.4</td><td>3.0</td><td>20.912.8</td><td></td></tr></table>

Table 4: Ablations on Qwen3.5-9B, one part of ASC changed at a time: Err on each corpus, and Supp., copy and vacuous share averaged over the five corpora (grey Err: a shortcut, as in Table 2). Binary credit: $\mathbf { 1 } [ p \geq \kappa ]$ in place of ϕ; the sentence reward also drops both tests.

Table 4 changes one part of ASC at a time on Qwen3.5-9B: removing any part brings back a shortcut or raises the error rate on most corpora. Without the copy test, copying returns (copy share 33.3 against 22.2, above the base model’s 28.1), so the variant’s lower error rates are grey, and they come with fewer supported authored sentences (2.6 against 3.0 per answer). Without the vacuity test, vacuous sentences return (19.1% against 13.3%) and the error rate on held-out ExpertQA doubles, from 3.7 to 7.5. With GRPO’s whole-answer advantage in place of the sentence baseline, the policy writes more (Supp. 3.3) but less reliably: the error rate rises on four of the five corpora and the vacuous share to 18.3%, as Finding 3 predicts. With binary credit $\mathbf { 1 } [ p \geq \kappa ]$ in place of the graded credit ϕ of Eq. 4, the error rate rises on all five corpora at similar copy and vacuous shares: the graded credit favors sentences the checker is sure of, which the judge accepts more often. The sentence reward, which drops both tests and the graded credit, copies more than the base model and writes nearly twice as many vacuous sentences as ASC.

## 6.4 Robustness

The gain holds across runs, evaluators and training, and it does not game the copy test.

• Runs. Every run of ASC, each seed of each model, has a lower authored error rate than the base model on every corpus.

• Evaluators. Two checkers never used in training also find more of ASC’s authored sentences supported than the base model’s, in all 20 settings under MiniCheck-Flan-T5 and in 19 under Align-Score. With no automatic evaluator at all, the human annotators find 24.8% of ASC’s sentences unsupported against 33.2% of the base model’s (250 ELI5 questions, Llama-3.1-8B; p = 0.01).

• Copy test. ASC does not learn to copy just below its threshold: fewer of its authored sentences sit just below it than the base model’s (9.2% against 11.3%), and with the threshold lowered to 0.4, its share of copied sentences stays below the base model’s in every setting (Appendix D.5).

• Training. ASC does not drift into a shortcut: on every model, its copy share drops within the first quarter of training and then stays flat, and its refusals never exceed the base model’s, whereas plain GRPO ends at a copy share of 72.8 (Finding 2; curves in Figure 7, Appendix D.6).

## 7 Conclusion

In this work, we showed that faithfulness rewards for long-form RAG pay for copying, and that common defenses only move this shortcut to shorter, emptier answers and refusals. We then introduced ASC, which credits only the sentences a model authors, that the question alone does not support and that the passages do, and compares each sentence with every sentence of its group. On four models, it lowers the authored error rate while writing more grounded sentences, held-out corpora included, without these shortcuts or any loss in answer quality. More broadly, we suggest asking of any reward what a policy can earn without the behavior it is meant to teach. A RAG model should be credited for what it writes, not for what it copies.

## Limitations

ASC needs no model beyond the reward’s checker: its copy test matches strings, and its vacuity and support tests reuse the checker. This keeps the method simple and lets it adopt a stronger checker as one appears, but its tests are only as reliable as that checker, which all methods in our comparison share. The copy test is lexical, so a close paraphrase counts as authored; ASC writes fewer such near-copies than the base model in 17 of the 20 settings (Appendix D.5). We train models of up to 9B parameters on two training corpora, with one training run for each baseline; the gain holds on all four models and on the three held-out corpora, and we leave larger models to future work.

ASC rewards the grounded sentences a model writes, not declining, so it refuses less than the base model, also on questions the passages do not answer, where the judge prefers the base model’s refusal (Appendix E). Pairing ASC with a groundedrefusal objective (Song et al., 2025) is a natural next step. Finally, ASC targets the reliability of what a model writes itself; where an application prefers verbatim quotation, plain faithfulness rewards already deliver it.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

Bespoke Labs. 2024. Bespoke-minicheck-7b.

Xilun Chen, Ilia Kulikov, Vincent-Pierre Berges, Barlas Oguz, Rulin Shao, Gargi Ghosh, and Wen tau Yih. 2026. Learning to reason for factuality. In Fortythird International Conference on Machine Learning.

Yung-Sung Chuang, Benjamin Cohen-Wang, Zejiang Shen, Zhaofeng Wu, Hu Xu, Xi Victoria Lin, James R. Glass, Shang-Wen Li, and Wen-Tau Yih. 2025. SelfCite: Self-supervised alignment for context attribution in large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 10839–10858. PMLR.

Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander Madry. 2024. Contextcite:

Attributing model generation to context. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Markus Dreyer, Mengwen Liu, Feng Nan, Sandeep Atluri, and Sujith Ravi. 2023. Evaluating the tradeoff between abstractiveness and factuality in abstractive summarization. In Findings of the Association for Computational Linguistics: EACL 2023, pages 2089– 2105, Dubrovnik, Croatia. Association for Computational Linguistics.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3558–3567, Florence, Italy. Association for Computational Linguistics.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Leo Gao, John Schulman, and Jacob Hilton. 2023a. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 10835–10866. PMLR.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023b. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, Singapore. Association for Computational Linguistics.

Gemma Team. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Yuzhe Gu, Wenwei Zhang, Chengqi Lyu, Dahua Lin, and Kai Chen. 2025. Mask-DPO: Generalizable finegrained factuality alignment of LLMs. In The Thirteenth International Conference on Learning Representations.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. 2022. TRUE: Re-evaluating factual consistency evaluation. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3905–3920, Seattle, United States. Association for Computational Linguistics.

Chengyu Huang, Zeqiu Wu, Yushi Hu, and Wenya Wang. 2024. Training language models to generate text with citations via fine-grained rewards. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2926–2949, Bangkok, Thailand. Association for Computational Linguistics.

Zhengping Jiang, Jingyu Zhang, Nathaniel Weir, Seth Ebner, Miriam Wanner, Kate Sanders, Daniel Khashabi, Anqi Liu, and Benjamin Van Durme. 2025. Core: Robust factual precision with informative subclaim identification. In Findings ofthe Association for Computational Linguistics: ACL 2025, pages 19833–19856, Vienna, Austria. Association for Computational Linguistics.

Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to progress in long-form question answering. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4940–4957, Online. Association for Computational Linguistics.

Thomas Kwa, Drake Thomas, and Adrià Garriga-Alonso. 2024. Catastrophic goodhart: regularizing RLHF with KL divergence does not mitigate heavytailed reward misspecification. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Faisal Ladhak, Esin Durmus, He He, Claire Cardie, and Kathleen McKeown. 2022. Faithful or extractive? on mitigating the faithfulness-abstractiveness tradeoff in abstractive summarization. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1410–1421, Dublin, Ireland. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Nelson Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in generative search engines. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 7001–7025, Singapore. Association for Computational Linguistics.

Yongchao Long, Yingying Zhang, Xianbin Wen, Xian Wu, Yuxi Zhou, and Shenda Hong. 2026. Copy-paste to mitigate large language model hallucinations. In The Fourteenth International Conference on Learning Representations.

Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. 2025. Hallucination-free? Assessing the reliability of leading AI legal research tools. Journal of Empirical Legal Studies, 22(2):216–242.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. 2024. ExpertQA: Expert-curated questions and attributed answers. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3025–3045, Mexico City, Mexico. Association for Computational Linguistics.

R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3428–3448, Florence, Italy. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Mistral AI. 2025. Mistral-Small-3.2-24B-Instruct-2506. Model card.

Christian Moya, Alex Semendinger, Guang Lin, and Elliott Thornley. 2026. Spurious correlation learning in preference optimization: Mechanisms, consequences, and mitigation via tie training. In Forty-third International Conference on Machine Learning.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10862– 10878, Bangkok, Thailand. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with

human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Blog post.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Leonard Hussenot, Orgad Keller, Nikola Momchev, Sabela Ramos Garea, Piotr Stanczyk, Nino Vieillard, Olivier Bachem, Gal Elidan, Avinatan Hassidim, Olivier Pietquin, and Idan Szpektor. 2023. Factually consistent summarization via reinforcement learning with textual entailment feedback. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6252–6272, Toronto, Canada. Association for Computational Linguistics.

Sara Rosenthal, Avirup Sil, Radu Florian, and Salim Roukos. 2025. CLAPnq: Cohesive long-form answers from passages in natural questions for RAG systems. Transactions ofthe Associationfor Computational Linguistics, 13:53–72.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Itai Shapira, Gerdus Benade, and Ariel D. Procaccia. 2026. How rlhf amplifies sycophancy. Preprint, arXiv:2602.01002.

Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024. Trusting your evidence: Hallucinate less with contextaware decoding. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 783–791, Mexico City, Mexico. Association for Computational Linguistics.

Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, and Maosong Sun. 2025. Teaching large language models to maintain contextual faithfulness via synthetic tasks and reinforcement learning. Preprint, arXiv:2505.16483.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In First Conference on Language Modeling.

Joar Max Viktor Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2022. Defining and characterizing reward gaming. In Advances in Neural Information Processing Systems.

Maojia Song, Shang Hong Sim, Rishabh Bhardwaj, Hai Leong Chieu, Navonil Majumder, and Soujanya Poria. 2025. Measuring and enhancing trustworthiness of LLMs in RAG through grounded attributions and learning to refuse. In The Thirteenth International Conference on Learning Representations.

Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. 2022. ASQA: Factoid questions meet long-form answers. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 8273–8288, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Liyan Tang, Philippe Laban, and Greg Durrett. 2024. MiniCheck: Efficient fact-checking of LLMs on grounding documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8818–8847, Miami, Florida, USA. Association for Computational Linguistics.

Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher Manning, and Chelsea Finn. 2024. Fine-tuning language models for factuality. In International Conference on Learning Representations, volume 2024, pages 44851–44866.

Liam van der Poel, Ryan Cotterell, and Clara Meister. 2022. Mutual information alleviates hallucinations in abstractive summarization. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 5956–5965, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Zixia Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V Le. 2024. Long-form factuality in large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Xueru Wen, Jie Lou, Xinyu Lu, Yuqiu Ji, Xinyan Guan, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Debing Zhang, and Le Sun. 2025. On-policy selfalignment with fine-grained knowledge feedback for hallucination mitigation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 5215–5231, Vienna, Austria. Association for Computational Linguistics.

Kevin Wu, Eric Wu, Kevin Wei, Angela Zhang, Allison Casasola, Teresa Nguyen, Sith Riantawan, Patricia Shi, Daniel Ho, and James Zou. 2025. An automated framework for assessing how well LLMs cite relevant medical references. Nature Communications, 16(1):3615.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah Smith, Mari Ostendorf, and Hannaneh Hajishirzi. 2023. Finegrained human feedback gives better rewards for language model training. In Advances in Neural Information Processing Systems, volume 36, pages 59008–59033. Curran Associates, Inc.

Xi Ye, Ruoxi Sun, Sercan Arik, and Tomas Pfister. 2024. Effective large language model adaptation for improved grounding and citation generation. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6237–6251, Mexico City, Mexico. Association for Computational Linguistics.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 113222–113244. Curran Associates, Inc.

Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. AlignScore: Evaluating factual consistency with a unified alignment function. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11328–11348, Toronto, Canada. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-tuning language models from human preferences. Preprint, arXiv:1909.08593.

## A Formal Statements and Proofs

This section proves the formal claims of Sections 4 and 5 in the order they appear: the reward as a line in the share of copied sentences (Eq. 2), the properties of the graded credit, Proposition 1, and the sentence form of the gradient (Eq. 7). Notation follows Section 3, and $\nabla$ stands for $\nabla _ { \theta }$

## A.1 The reward is a line in the share of copied sentences

Lemma 1 (Reward decomposition). Let an answer y have n sentences, $n _ { c }$ of them copied and $n _ { a } = n - n _ { c }$ authored (markup-only sentences, rare in our data, are set aside), and let $s _ { c }$ and $s _ { a }$ be the shares ofits copied and ofits authored sentences that the checker supports (either is 0 when the answer has no sentence of that kind). With $\rho = n _ { c } / n ,$ , the reward ofEq. 1 is

$$
R ( y ) = s _ { a } + \rho ( s _ { c } - s _ { a } ) .
$$

Proof. Split the sum of Eq. 1 by provenance. The copied sentences contribute $n _ { c } s _ { c }$ supported sentences and the authored ones $n _ { a } s _ { a }$ , so

$$
\begin{array} { l } { { R ( y ) \ = \ \frac { n _ { c } s _ { c } + n _ { a } s _ { a } } { n } \ = \ \rho s _ { c } + ( 1 - \rho ) s _ { a } } } \\ { { \mathrm { } } } \\ { { \mathrm { } = \ s _ { a } + \rho ( s _ { c } - s _ { a } ) . } } \end{array}
$$

At fixed support rates, R is therefore affine in $\rho$ with slope $s _ { c } - s _ { a }$ . Turning one authored sentence into a copied one raises $\rho$ by $1 / n$ and the reward by $( s _ { c } - s _ { a } ) / n$ , although no authored sentence has become more reliable. The slope is positive whenever copied sentences are supported more often than authored ones, which holds under the reward’s checker for every base model and corpus (Table 8).

## A.2 The graded credit

Lemma 2 (Graded credit). Let $\kappa \in ( 0 , 1 )$ and $\phi ( p ) = 1$ − ln p/ ln κ on $[ \kappa , 1 ]$ , the branch ofEq. 4 above the threshold. Then

(a) $\phi ( \kappa ) = 0$ and $\phi ( 1 ) = 1 ,$

(b) ϕ is strictly increasing and strictly concave on $[ \kappa , 1 ] ,$ ;

(c) with the surprisal $h ( p ) = - \ln p , \phi ( p ) = 1 -$ $h ( p ) / h ( \kappa )$ , and ϕ is the onlyfunction affine in h that satisfies (a).

Proof. (a) ln $ { \mathrm { ~  ~ 1 ~ } } { } = 0$ gives $\phi ( 1 ) \ = \ 1$ , andln $\kappa /$ ln $\kappa = 1$ gives $\phi ( \kappa ) = 0 . \left( \mathbf { b } \right)$ Since ln $\kappa < 0$ $\phi ^ { \prime } ( p ) = - \frac { 1 } { p \ln \kappa } > 0 , \qquad \phi ^ { \prime \prime } ( p ) = \frac { 1 } { p ^ { 2 } \ln \kappa } < 0 .$

(c) The identity follows from ln $p /$ ln $\begin{array} { r l } { \kappa } & { { } = } \end{array}$ $h ( p ) / h ( \kappa )$ . If $f ( h ) = \alpha + \gamma h$ satisfies (a), then $f ( h ( 1 ) ) ~ = ~ f ( 0 ) ~ = ~ \alpha ~ = ~ 1$ and $f ( h ( \kappa ) ) \ =$ $1 + \gamma h ( \kappa ) = 0 , \ s \circ \gamma = - 1 / h ( \kappa )$ and $f ( h ( p ) ) =$ $\phi ( p )$

Below the threshold Eq. 4 sets the credit to zero, so a sentence is credited exactly when the checker’s probability exceeds $\kappa ;$ above it, the credit grows with the checker’s confidence, fastest near κ.

## A.3 Proof of Proposition 1

Fix a group of G answers and index its M sentences by $k ,$ across all answers. Their credits $r _ { 1 } , \hdots , r _ { M } \in [ 0 , 1 ]$ have mean $\mu$ and standard deviation σ, and Eq. 5 gives $a _ { k } = T ( r _ { k } )$ with

$$
T ( r ) = \frac { r - \mu } { \sigma + \epsilon } , \qquad \epsilon > 0 .
$$

The proof rests on two facts. (F1) T is one map for the whole group, whichever answer a sentence belongs to, and it is strictly increasing, with slope $1 / ( \sigma + \epsilon ) > 0$ . (F2) A sentence that is copied, vacuous or unsupported has $r _ { k } = 0$ , the smallest credit in [0, 1]: Eq. 3 zeroes copied and vacuous sentences, and Eq. 4 unsupported ones.

Proof. (i) Credit alone sets the order. By (F1), $T ( r _ { k } ) < T ( r _ { l } )$ if and only if $r _ { k } < r _ { l } .$ , that is $a _ { k } <$ a<sub>l</sub> if and only if $r _ { k } < r _ { l }$ , whether sentences k and l belong to the same answer or to different ones.

(ii) Shortcuts rank last. By (F2) and (i), a copied, vacuous or unsupported sentence has the smallest advantage of the group, $a _ { k } = - \mu / ( \sigma + \epsilon )$ . If any sentence of the group earns credit, then $\mu > 0$ and this advantage is negative. A refusal earns zero whenever the checker finds it unsupported or vacuous, as for the copy penalty’s refusal in Example 1 (Appendix I).

(iii) Average sentences earn nothing. Since $\begin{array} { r } { M \mu = \sum _ { k } r _ { k } } \end{array}$

$$
\sum _ { k = 1 } ^ { M } a _ { k } \ = \ \frac { 1 } { \sigma + \epsilon } \Big ( \sum _ { k = 1 } ^ { M } r _ { k } - M \mu \Big ) \ = \ 0 ,
$$

and $a _ { k } = 0$ exactly when $r _ { k } = \mu _ { : }$ , so a sentence gains only if its credit exceeds the mean.

(iv) Length-free advantages. Given the group’s µ and $\sigma , a _ { k }$ depends on the sentence only through $r _ { k } ,$ which is computed from its text, the question and the passages $( \operatorname { E q } . 3 ) ;$ neither its number of tokens nor the length of its answer enters, except through $\mu$ and σ, which every sentence of the group shares. Part (iv) concerns the advantage; in the gradient, each token of the sentence carries it (Lemma 3).

Scope. The proposition concerns sentences that fail a test: it does not bound how many sentences an answer has, and a paraphrase whose copy share stays below $\tau$ counts as authored; Appendix D.5 measures how often such sentences occur.

Remark 1 (GRPO’s answer-level advantage). Under GRPO with the reward R, every token of answer i receives $A ^ { ( i ) } = ( R _ { i } - \mu _ { R } ) / ( \sigma _ { R } + \epsilon )$ , with $\mu _ { R } , \sigma _ { R }$ the group’s mean and standard deviation of $R ,$ and (i), (ii) and (iv) fail. Take a group of two answers: answer 1 has three supported sentences, one of them copied, so $R _ { 1 } = 1$ ; answer 2 has two grounded sentences and one unsupported sentence, so $R _ { 2 } = 2 / 3$ . Then $A ^ { ( 1 ) } > A ^ { ( 2 ) }$ , and the copied sentence of answer 1 is reinforced more than the two grounded sentences of answer 2. Deleting the unsupported sentence of answer 2 raises $R _ { 2 }$ to 1, and with it the advantage of every remaining token, so writing less pays.

## A.4 The gradient as a policy gradient over sentences

Lemma 3 (Sentence form of the gradient). With one update per batch, the gradient of the objective in Eq. 6 at $\theta = \theta _ { \mathrm { o l d } }$ is

$$
\begin{array} { l } { \displaystyle \nabla \mathcal { T } = \frac { 1 } { N } \sum _ { i , s } a _ { s } ^ { ( i ) } \nabla \ln \pi _ { \theta } \big ( y _ { s } ^ { ( i ) } \mid y _ { < s } ^ { ( i ) } \big ) } \\ { \displaystyle - \beta \nabla \frac { 1 } { N } \sum _ { i , t } \mathbb { D } _ { i , t } , } \end{array}
$$

with the prompt $( q , P )$ left implicit in the conditioning, as in $E q . \ 7 .$ Under GRPO, the first term is $\begin{array} { r } { \frac { 1 } { N } \sum _ { i } A ^ { ( i ) } \nabla \ln \pi _ { \theta } ( y ^ { ( i ) } ) } \end{array}$

Proof. Fix an answer i and drop its index. $\mathbf { A } \mathbf { t } \theta =$ $\theta _ { \mathrm { o l d } }$ the importance ratio is $w _ { t } = 1$ , so $\nabla w _ { t } \ =$ w<sub>t</sub> ∇ ln $\pi _ { \boldsymbol { \theta } } ( y _ { t } \mid y _ { < t } ) = \nabla$ ln π<sub>θ</sub> $\textstyle ( y _ { t } \mid y _ { < t } )$ . Every token t of sentence s carries $\hat { A } _ { t } = a _ { s }$ , and tokens outside every sentence carry zero and drop out, so grouping the tokens by sentence gives

$$
\begin{array} { r l } {  { \sum _ { t } \hat { A } _ { t } \nabla \ln \pi _ { \theta } ( y _ { t } \mid y _ { < t } ) } \quad } & { } \\ & { = \sum _ { s } a _ { s } \sum _ { t \in s } \nabla \ln \pi _ { \theta } ( y _ { t } \mid y _ { < t } ) } \\ & { = \sum _ { s } a _ { s } \nabla \ln \pi _ { \theta } ( y _ { s } \mid y _ { < s } ) , } \end{array}
$$

where the last step is the chain rule of probability: the log-probabilities of a sentence’s tokens add up to the log-probability of the sentence given everything before it. Under GRPO every token of answer i carries the same $A ^ { ( i ) }$ , and the same grouping over the whole answer gives $A ^ { ( i ) } \nabla \ln \pi _ { \theta } ( \mathbf { \bar { \boldsymbol { y } } } ^ { ( i ) } )$ . The KL term does not depend on the advantages and is the same for every method.

## B Setup Details

This section gives what is needed to reproduce the experiments: the data, the sentence tests and the

metrics built on them, and the training settings. The prompts are listed in Appendix H.
<table><tr><td>Corpus</td><td>Role</td><td>Train</td><td>Eval</td><td>Eval (small)</td><td>Passages</td></tr><tr><td>ASQA</td><td>train + in domain</td><td>2,850</td><td>948</td><td>300</td><td>5</td></tr><tr><td>RAGTruth</td><td>train + in domain</td><td>839</td><td>150</td><td>150</td><td>3</td></tr><tr><td>ClapNQ</td><td>held out</td><td></td><td>300</td><td>300</td><td>1</td></tr><tr><td>ELI5</td><td>held out</td><td>1</td><td>1,000</td><td>300</td><td>5</td></tr><tr><td>ExpertQA</td><td>held out</td><td>一</td><td>1,183</td><td>300</td><td>3</td></tr></table>

Table 5: The five corpora. Train: training questions; Eval: evaluation questions for the 9B and 8B models; Eval (small): those for the 4B and 3B models; Passages: passages shown with each question.

## Data.

• Training. ASQA and RAGTruth are pooled for training (3,689 questions) and evaluated on their own evaluation sets, ALCE’s 948 ASQA questions and RAGTruth’s QA test sources.

• Held out. ClapNQ (its answerable development questions), ELI5 and ExpertQA are never seen in training. ELI5 is known to be hard to ground in retrieved documents (Krishna et al., 2021), and ExpertQA was added only after every training run had finished, so nothing was tuned on it.

• No overlap. Training questions are disjoint from every evaluation set by identifier, source document and normalized question text.

• Size. The two smaller models are evaluated on at most 300 questions per corpus, to bound the cost of the judge.

Sentence tests. ASC and the metrics share three sentence tests; the reward uses only the second.

• Copy. Answer and passages are lowercased and split into words. A word of the answer is covered if some window of five consecutive answer words containing it also occurs in the passages, and the copy share is the covered fraction. A sentence is copied if the copy share of its text, with list markers, headings and emphasis stripped, is at least τ = 0.5; a sentence made only of such markup is neither copied nor authored and earns no credit.

• Support. A sentence is supported when the checker’s probability is at least κ = 0.5. Long passages are split into chunks aligned with sentences, and the maximum over chunks is taken.

• Vacuity. A sentence is vacuous when the support check passes with the question in place of the passages, a context ablation in the spirit of conditional mutual information (van der Poel et al., 2022) and SelfCite (Chuang et al., 2025).

Bespoke-MiniCheck-7B (Bespoke Labs, 2024) is the checker of the reward and of ASC; MiniCheck-Flan-T5-Large (Tang et al., 2024) and AlignScore (Zha et al., 2023) are the held-out checkers, used only in evaluation.
<table><tr><td>Metric</td><td>Definition</td><td>Instrument</td></tr><tr><td>Err</td><td>authored sentences judged un- Gemma-4- supported, over authored sen- 31B tences</td><td></td></tr><tr><td>T5, AS</td><td>authored sentences supported, MiniCheck- over authored sentences</td><td>Flan-T5, AlignScore</td></tr><tr><td>Supp.</td><td>supported authored sentences MiniCheck- per answer</td><td>Flan-T5</td></tr><tr><td>Copy</td><td>answer words inside a verba- string match tim five-word match with the passages, over words</td><td></td></tr><tr><td>Vac.</td><td>supported authored sentences Bespoke- also supported by the question MiniCheck- alone, over supported authored 7B</td><td></td></tr><tr><td>Ref.</td><td>sentences answers whose every sentence rule-based declines to answer, over an-</td><td></td></tr><tr><td>Words Reward</td><td>swers words per answer sentences supported, over sen- Bespoke- tences (Eq. 1)</td><td>MiniCheck-</td></tr><tr><td>No auth.</td><td>answers without an authored string match sentence, over answers</td><td>7B</td></tr><tr><td>EM</td><td>short answers of ASQA found ALCE string in the answer</td><td>match</td></tr></table>

Table 6: The metrics of Tables 2 and 4 and of the appendix, on a 0–100 scale except counts per answer. Only Bespoke-MiniCheck-7B is also used in training.

Metrics. The judge, Gemma-4-31B (Gemma Team, 2026) at full precision, reads the passages and one authored sentence (prompt in Appendix H); a sentence is an error when the judge rejects it outright, with a probability of “Yes” below 3.5×10<sup>−10</sup>. Its probabilities are nearly binary (46% of sentences below 0.001 and 42% above 0.999), and this rule, chosen on one half of RAGTruth’s humanlabeled responses and tested on the other, reaches a balanced accuracy of 0.81 against the human labels, above the 0.75 of the reward’s checker. Copying counts as a shortcut even though Err leaves copied sentences out: a policy that copies more submits fewer of its own sentences to the judge, so Err can fall without any sentence becoming more reliable, which Supp. makes visible. An answer refuses only when every sentence declines to answer, so a single hedge is not a refusal.

Training. All methods share the settings of Table 7: one epoch over the pooled training questions (922 steps, 32 answers per step), with rollouts sampled at the listed temperature; they differ only in the training signal, and Algorithm 1 gives one ASC update. Evaluation decodes greedily with the same token budget. For the copy penalty, λ is the support gap $s _ { c } - s _ { a }$ of the model’s own samples on the training questions under the reward’s checker: 0.071 for Qwen3.5-9B, 0.087 for Qwen3.5-4B, 0.077 for Llama-3.1-8B and 0.123 for Llama-3.2-3B. The sentence reward uses the sentence baseline of Eq. 5 with $r _ { s } = \mathbf { 1 } [ g ( y _ { s } , P ) \ge \kappa ]$ for every sentence of the answer. ASC is trained with two seeds on Qwen3.5-9B and three on Qwen3.5-4B, and the binary-credit ablation with three seeds on Qwen3.5- 9B; every other policy is one training run. Everything runs on eight H100 80GB GPUs: one training run on a 9B model takes about 3.5 hours on six of them, with the checker served on the remaining two, and evaluating one policy on all corpora, including the judge, takes about two hours on two.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Answers per question G</td><td>8</td></tr><tr><td>KL coefficient β</td><td>0.04</td></tr><tr><td>Learning rate</td><td>1e-6</td></tr><tr><td>Warm-up steps</td><td>36</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Updates per batch Loss</td><td>1 token-normalized</td></tr><tr><td>Max. answer tokens</td><td></td></tr><tr><td>Sampling temperature / top-p</td><td>1024</td></tr><tr><td>Gradient clipping</td><td>0.7 / 0.95</td></tr><tr><td>Precision</td><td>1.0</td></tr><tr><td>Standardization €</td><td>bf16, full weight 1e-4</td></tr></table>

Table 7: GRPO settings of every method and model.

Penalty strengths. Neither penalty is tuned on the evaluation corpora. The KL coefficient $\beta = 0 . 0 4$ is GRPO’s default (Shao et al., 2024), and no other value removes the shortcuts: the KLregularized objective is maximized by $\pi ^ { * } ( y ) \ \propto \quad$ $\pi _ { \mathrm { r e f } } ( y ) \exp ( R ( y ) / \beta )$ (Rafailov et al., 2023), which favors whatever raises R at every $\beta > 0 .$ , so a larger β only shrinks the reward gain together with the shortcut. The copy penalty’s λ is the smallest value at which, by Eq. 2, a copied sentence earns no more than an authored one; a larger λ would only push the policy further from the passages, which its refusals already show.

## C Analysis Details

This section gives the evidence behind the three findings of Section 4, one subsection each.

Algorithm 1 One ASC update for a prompt $( q , P )$   
Require: policy $\pi _ { \theta } ,$ reference $\pi _ { \mathrm { r e f } } ,$ checker g, thresholds   
τ, κ, group size G   
1: Sample $y ^ { ( 1 ) } , \dots , y ^ { ( G ) } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid q , P )$   
2: for each sentence $y _ { s } ^ { ( i ) }$ of each answer $\boldsymbol y ^ { ( i ) }$ do   
3: $\mathbf { i f } c ( y _ { s } ^ { ( i ) } ) \ge \tau \mathbf { o r } g ( y _ { s } ^ { ( i ) } , q ) \ge \kappa$ then   
4: $r _ { s } ^ { ( i ) }  0$ ▷ copied or vacuous   
5: else   
6: $\mathsf { \bar { \Gamma } } r _ { s } ^ { ( i ) } \gets \phi \bigl ( g ( y _ { s } ^ { ( i ) } , P ) \bigr )$ ▷ graded support   
7: end if   
8: end for   
9: µ, σ ← mean, s.d. of all credits in the group   
10: $\hat { A } _ { i , t }  ( r _ { s } ^ { ( i ) } - \mu ) / ( \sigma + \epsilon )$ for each $t \in y _ { s } ^ { ( i ) }$   
11: $\theta  \theta + \eta \nabla _ { \theta } \mathcal { I } ( \dot { \theta } )$ ▷ Eq. 6, step size η

C.1 Copied versus authored sentences (Finding 1)
<table><tr><td rowspan="2">Model</td><td rowspan="2">Corpus</td><td colspan="2">Reward checker</td><td colspan="2">MiniCheck-Flan-T5</td><td colspan="2">AlignScore</td></tr><tr><td>Auth.</td><td>Copied</td><td>Auth.</td><td>Copied</td><td>Auth.</td><td>Copied</td></tr><tr><td rowspan="4">Qwen3.5-9B</td><td>ASQA</td><td>15.3</td><td>4.2</td><td>17.4</td><td>6.1</td><td>12.2</td><td>6.3</td></tr><tr><td>RAGTruth</td><td>7.7</td><td>3.4</td><td>13.7</td><td>5.8</td><td>6.4</td><td>4.9</td></tr><tr><td>ClapNQ</td><td>3.6</td><td>1.1</td><td>10.9</td><td>5.0</td><td>3.1</td><td>0.7</td></tr><tr><td>ELI5</td><td>23.9</td><td>3.0</td><td>29.2</td><td>8.6</td><td>10.7</td><td>8.5</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>ASQA</td><td>18.1</td><td>5.5</td><td>19.0</td><td>5.3</td><td>13.9</td><td>7.5</td></tr><tr><td>RAGTruth</td><td>8.3</td><td>4.5</td><td>14.0</td><td>6.6</td><td>7.4</td><td>5.0</td></tr><tr><td>ClapNQ</td><td>5.7</td><td>1.5</td><td>13.2</td><td>5.6</td><td>3.7</td><td>1.2</td></tr><tr><td>ELÍ5</td><td>26.3</td><td>5.5</td><td>29.4</td><td>8.4</td><td>13.4</td><td>8.8</td></tr><tr><td rowspan="4">Llama-3.1-8B</td><td>ASQA</td><td>18.3</td><td>5.8</td><td>19.3</td><td>7.4</td><td>16.5</td><td>8.0</td></tr><tr><td>RAGTruth</td><td>9.1</td><td>4.6</td><td>14.9</td><td>11.2</td><td>6.2</td><td>6.7</td></tr><tr><td>ClapNQ</td><td>6.0</td><td>2.3</td><td>16.7</td><td>7.7</td><td>5.6</td><td>0.5</td></tr><tr><td>ELÍ5</td><td>24.3</td><td>6.0</td><td>29.5</td><td>9.0</td><td>22.5</td><td>9.1</td></tr><tr><td rowspan="4">Llama-3.2-3B</td><td>ASQA</td><td>26.2</td><td>7.7</td><td>21.7</td><td>7.3</td><td>19.6</td><td>8.3</td></tr><tr><td>RAGTruth</td><td>14.7</td><td>4.6</td><td>19.5</td><td>8.0</td><td>10.7</td><td>4.1</td></tr><tr><td>ClapNQ</td><td>10.5</td><td>3.0</td><td>21.6</td><td>8.4</td><td>8.1</td><td>0.9</td></tr><tr><td>ELÍ5</td><td>32.6</td><td>9.6</td><td>34.3</td><td>9.5</td><td>26.5</td><td>8.2</td></tr></table>

Table 8: Unsupported rate (%) of authored (Auth.) and copied sentences in base-model samples (eight answers per question) under the reward’s checker and the two held-out checkers of Appendix B.

Under the reward’s checker, authored sentences are unsupported 1.8 to 8.1 times as often as copied ones for every model and corpus, most on ELI5, where base models write most of the answer themselves. The held-out checkers find the same gap, MiniCheck-Flan-T5 everywhere and AlignScore in all but one case, so it is not a quirk of one checker: entailment models accept hypotheses that overlap their premise (McCoy et al., 2019), and a copied sentence overlaps it entirely. Human span labels on RAGTruth (131,087 sentences, no checker involved) agree: copied sentences carry errors at 1.9% and authored ones at 12.2%, and the error rate falls steadily with copy share, from 13.7% in the lowest decile to 0.6% in the highest.

Within a question. Real samples follow Lemma 1: among the answers to the same question, the reward rises with the copy share $c ( y )$ for every model and corpus. The pull acts even before training: picking the highest-reward answer of the eight raises the

copy share for every model and corpus.
<table><tr><td>Tercile</td><td>ROUGE-L</td><td>Sentences</td><td>Reward checker</td><td>MiniCheck-Flan-T5</td></tr><tr><td>least similar</td><td>0.27</td><td>7,362</td><td>74.8</td><td>73.3</td></tr><tr><td>middle</td><td>0.40</td><td>7,285</td><td>86.2</td><td>82.9</td></tr><tr><td>closest</td><td>0.57</td><td>7,323</td><td>91.4</td><td>89.2</td></tr></table>

Table 9: Authored, non-vacuous sentences of Qwen3.5- 9B and Llama-3.1-8B base-model samples on ASQA and RAGTruth, by tercile of ROUGE-L similarity to their closest passage sentence: mean similarity, count, and share (%) supported under each checker.

Closeness to the passages. The preference does not hinge on the copy test’s five-word cut-off: both checkers accept an authored sentence more often the closer it stays to its closest passage sentence (Table 9), so the reward’s preference for copying is a property of the checker, not of the threshold.

C.2 Training on the faithfulness reward (Finding 2)
<table><tr><td>Training</td><td>Reward</td><td>T5</td><td>Copy</td><td>Words</td><td>Vac.</td><td>Ref.</td><td>No auth.</td><td>EM</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>87.1</td><td>82.5</td><td>38.8</td><td>73.5</td><td>21.5</td><td>6.1</td><td>15.4</td><td>45.5</td></tr><tr><td>GRPO</td><td>98.9</td><td>96.1</td><td>72.8</td><td>13.7</td><td>18.1</td><td>0.0</td><td>81.0</td><td>27.6</td></tr><tr><td>GRPO + KL</td><td>92.3</td><td>87.7</td><td>39.2</td><td>60.0</td><td>28.5</td><td>14.7</td><td>19.5</td><td>41.8</td></tr><tr><td>Copy penalty</td><td>92.0</td><td>94.3</td><td>11.8</td><td>36.9</td><td>67.4</td><td>64.5</td><td>1.3</td><td>20.2</td></tr><tr><td>DPO</td><td>94.6</td><td>91.0</td><td>53.4</td><td>55.7</td><td>16.9</td><td>1.7</td><td>37.2</td><td>41.8</td></tr><tr><td>DPO, copy-matched</td><td>90.4</td><td>85.8</td><td>44.9</td><td>59.7</td><td>21.5</td><td>6.1</td><td>24.7</td><td>43.8</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>87.3</td><td>83.0</td><td>45.5</td><td>24.0</td><td>25.8</td><td>7.0</td><td>47.9</td><td>35.2</td></tr><tr><td>GRPO</td><td>99.8</td><td>99.8</td><td>4.6</td><td>7.6</td><td>58.9</td><td>0.0</td><td>5.2</td><td>0.5</td></tr></table>

Table 10: Policies trained on $R ,$ evaluated on ASQA at the final checkpoint. T5: authored support under MiniCheck-Flan-T5; No auth.: answers without an authored sentence; other columns as in Table 1. DPO, copy-matched: pairs whose two answers copy equally.

Every policy trained on R raises the reward through a shortcut rather than through better authored sentences (Table 10): authored support rises too, but on fewer authored sentences. Plain GRPO copies until 81.0% of its answers contain no authored sentence; GRPO + KL refuses and writes emptier answers; DPO copies more even when the two answers of every pair copy equally. On Llama-3.1-8B, plain GRPO settles on a vacuous template whose most frequent sentences are “A film was made.” and “A song was released.”, which earns a reward of 99.8 at an exact match of 0.5.

Over training. Shortcuts set in early and persist (Figure 6): GRPO’s answers shrink within the first quarter of training and its copy share passes 70 by mid-training, and the copy penalty’s refusals climb to 69.3% by three quarters of training, while ASC brings copy share and refusals down within the first quarter and keeps them there.

Runs. Plain GRPO is GRPO + KL with $\beta = 0 .$ The DPO policies are trained for one epoch on pairs of base-model samples ranked by R; in the copymatched pairs, the two answers’ copy shares differ by at most 0.02 and their sentence counts by at most one. Table 1, Table 10 and the GRPO point of Figure 1b come from the evaluation pass of these runs, which decoded the base model separately from the runs of Section 6, so their base values differ slightly from Tables 12 and 13 (by at most 2.7 points, Llama-3.1-8B’s vacuous share on ASQA).

## C.3 Whole-answer and sentence advantages (Finding 3)

<table><tr><td rowspan="2">Sentence</td><td rowspan="2">Count</td><td colspan="3">All sentences (%)</td><td colspan="2">Pushed up, if moved (%)</td></tr><tr><td>Up</td><td>Down</td><td>Unmoved</td><td>Pooled</td><td>Range</td></tr><tr><td>Copied</td><td>38,106</td><td>36.3</td><td>14.1</td><td>49.6</td><td>72.0</td><td>66.8–83.0</td></tr><tr><td>Vacuous</td><td>21,344</td><td>40.9</td><td>28.7</td><td>30.5</td><td>58.8</td><td>54.7-75.6</td></tr><tr><td>Grounded</td><td>72,765</td><td>50.1</td><td>16.3</td><td>33.6</td><td>75.5</td><td>70.1-93.3</td></tr><tr><td>Authored, unsupported</td><td>18,011</td><td>19.7</td><td>75.7</td><td>4.6</td><td>20.7</td><td>3.0–27.8</td></tr></table>

Table 11: How GRPO’s answer-level advantage moves each kind of sentence, on the base-model samples of Finding 1. Kinds, in the order of ASC’s tests: copied; vacuous (authored, supportable from the question alone); grounded (authored, not vacuous, supported by the reward’s checker); authored, unsupported. Count: sentences of that kind. Up, Down: the answer’s reward is above, below its group’s mean; Unmoved: equal to it, so the advantage is zero. Pushed up, if moved: the share pushed up among the sentences GRPO moves, pooled and its range over models and corpora.

Signs of GRPO’s advantage. Every sentence takes the sign of its answer’s advantage, so Table 11 counts how GRPO moves each kind of sentence. It leaves a third of all sentences, and half of the copied ones, unmoved, because their answer’s reward equals the group’s mean; among the sentences it moves, it pushes up copied and grounded ones at nearly the same rate, and most vacuous ones too. Under ASC, a copied, vacuous or unsupported sentence always takes the lowest advantage of its group (Proposition 1(ii)), so none is pushed up.

Correlation with held-out support. On the same pool as Table 9 (21,970 authored, non-vacuous sentences in 1,967 groups), we compute two advantages from the binary form of the credit in Eq. 3: GRPO’s answer-level advantage, with the credit summed over the answer, and the sentence advantage of Eq. 5. Their correlation with the support of the authored sentences under MiniCheck-Flan-T5 is 0.07 and 0.21: the same credit tracks a sentence’s reliability far more closely when sentences are compared with sentences rather than answers with answers. Answer length explains less than 2% of the variance of either advantage, so the difference does not come from length.

![](images/15ad0f3beb88b8c9c3273a0430f2a43e4e44e5d116261c0f1703ba969d476f0d.jpg)  
Figure 6: Qwen3.5-9B on ASQA over training (0: base model): every metric of Figure 3, with vacuous share and authored support under MiniCheck-Flan-T5, and ASC for comparison. Policies trained on R raise authored support by copying (GRPO) or by writing less and refusing (KL, copy penalty); ASC copies less and writes longer answers.

## D Full Results

This section expands Table 2: every metric on every corpus, counts per answer with their significance, every training run, the robustness of the shortcut rule, near-copies, and how each model trains. Significance marks test a difference with a paired bootstrap over questions (1,000 resamples, seeds pooled per question); they cover the sampling of questions, not of training runs (Appendix D.3 lists the runs), and each comparison is tested on its own.

## D.1 Every metric on every corpus

Faithfulness. The one setting in which AlignScore does not confirm the gain of Section 6.4 is a 0.2- point dip (Qwen3.5-9B on ELI5). The absolute gain is largest where the base model errs most, up to 15.6 points for Llama-3.2-3B on ExpertQA, which is the same relative cut as elsewhere (38%, Figure 5).

Shortcuts. The baselines’ shortcuts concentrate on ELI5 and ExpertQA, the held-out corpora where the base model errs most and an answer must be written rather than assembled (Section 3). There GRPO + KL declines 44 to 76% of the questions on the Qwen models, against at most 28% on the other corpora, and the copy penalty declines 94 to 95% on Qwen3.5-9B. On the Llama models, GRPO + KL and the sentence reward copy more than the base model on every corpus, and the copy penalty shortens the answers instead, to 56 to 89% of the base model’s words. ASC does the opposite: it copies less, writes fewer vacuous sentences and refuses less than the base model on every corpus, and writes longer answers on every held-out corpus.

## D.2 Counts per answer and significance

Err is a rate over authored sentences, and ASC writes more of them, so a lower rate could hide more errors per answer; Table 14 therefore counts them. Err is lower in every setting, significantly in 14 of the 20. Per answer, ASC writes 0.3 to 1.0 more judge-supported authored sentences, and its unsupported ones fall in 16 settings and rise significantly in none. Counting copied sentences too, the unsupported share of the whole answer falls in 18 settings and rises significantly in none, so answers become more faithful as a whole. Within bins of answer length, ASC still writes more supported authored sentences in all 20 settings, so the added content is not a by-product of length.

## D.3 Every training run

Table 15 lists every ASC training run separately. Each run has a lower authored error rate than the base model on every corpus it was evaluated on, so no single seed carries the gain.

## D.4 Robustness of the shortcut rule

The shortcut rule of Section 6.1 has two free choices, the margin and the word floor, and Table 16 recomputes the ranking of Table 2 under other values of both. ASC is flagged in no setting under any of them. The number of settings in which it has the lowest Err falls at wider margins, mostly because they stop flagging the sentence reward, which copies a median 3.8 points more than the base model and 14.0 more than ASC. Without any threshold the picture is the same: in 19 of the 20 settings, no baseline has both a lower Err and at least as many supported authored sentences per answer as ASC; the exception, the copy penalty on ASQA for Llama-3.2-3B, writes answers half as long. Where a baseline takes no shortcut, in 5 of its 60 results, ASC’s error rate is lower in one (Llama-3.1-8B on ExpertQA, 19.9 against 22.5 for GRPO + KL) and no different in the other four, while ASC writes significantly more judge-supported authored sentences in all five (0.4 to 0.8 per answer).

<table><tr><td rowspan="2">Method</td><td colspan="4">ASQA</td><td colspan="4">RAGTruth</td><td colspan="4">ClapNQ</td><td colspan="4">ELI5</td><td colspan="4">ExpertQA</td></tr><tr><td>Err</td><td>T5</td><td>AS</td><td>Supp.</td><td>Err</td><td>T5</td><td>AS</td><td>Supp.</td><td>Err</td><td>T5</td><td>AS</td><td>Supp.</td><td>Err</td><td>T5</td><td>AS</td><td>Supp.</td><td>Err</td><td>T5</td><td>AS</td><td>Supp.</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td></td><td>5.1 82.3 89.9</td><td></td><td>1.6</td><td>2.3</td><td>87.9</td><td>94.2</td><td>3.0</td><td></td><td>3.6 88.4</td><td>96.7</td><td>1.8</td><td>6.070.9</td><td></td><td>89.6</td><td>2.2</td><td></td><td>6.2 71.5</td><td>89.7</td><td>2.3</td></tr><tr><td>GRPO + KL</td><td>4.0</td><td>87.7</td><td>91.2</td><td>1.4</td><td>1.2</td><td>88.8</td><td>93.0</td><td>3.0</td><td>3.2</td><td>90.6</td><td>97.3</td><td>1.8</td><td>5.4</td><td>79.5</td><td>91.1</td><td>2.0</td><td>5.9</td><td>81.6</td><td>92.9</td><td>2.1</td></tr><tr><td>Copy penalty</td><td>1.5</td><td>94.3</td><td>95.1</td><td>1.4</td><td>1.2</td><td>73.5</td><td>92.8</td><td>1.2</td><td>0.4</td><td>87.6</td><td>98.0</td><td>1.5</td><td>1.1</td><td>94.0</td><td>98.8</td><td>1.1</td><td>0.4</td><td>74.5</td><td>98.6</td><td>0.9</td></tr><tr><td>Sentence reward</td><td>3.7</td><td>87.1</td><td></td><td>1.5</td><td>0.9</td><td>90.9</td><td></td><td>2.9</td><td>3.1</td><td>90.7</td><td></td><td>1.7</td><td>3.5</td><td>80.0</td><td></td><td>2.4</td><td>4.3</td><td>80.4</td><td>90.0</td><td>2.5</td></tr><tr><td>ASC</td><td></td><td></td><td>4.087.990.7</td><td>2.3</td><td>1.1</td><td></td><td>90.9 94.6</td><td>3.8</td><td>2.2</td><td>90.8 98.3</td><td></td><td>2.5</td><td>3.4</td><td>78.8 89.4</td><td></td><td>3.2</td><td></td><td>3.7 84.9</td><td>90.7</td><td>3.4</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td></td><td>8.6 80.0</td><td>87.1</td><td>1.7</td><td>2.9</td><td>87.9</td><td>92.2</td><td>2.9</td><td>5.8</td><td>89.6</td><td>96.1</td><td>1.8</td><td>8.8</td><td>72.7</td><td>88.1</td><td>2.3</td><td>8.7</td><td>64.7</td><td>85.1</td><td>2.1</td></tr><tr><td>GRPO + KL</td><td>3.2</td><td>91.8</td><td>94.3</td><td>1.4</td><td>2.0</td><td>86.3</td><td>93.9</td><td>2.3</td><td>2.9</td><td>88.6</td><td>98.2</td><td>1.5</td><td>3.9</td><td>90.5</td><td>95.0</td><td>1.5</td><td>3.5</td><td>80.8</td><td>95.9</td><td>1.4</td></tr><tr><td>Copy penalty</td><td></td><td>8.2 90.9</td><td></td><td>1.7</td><td>3.1</td><td>86.9</td><td></td><td>3.2</td><td>4.9</td><td>90.0</td><td></td><td>2.3</td><td>6.7</td><td>79.6</td><td></td><td>2.2</td><td>9.1</td><td>75.0</td><td>91.1</td><td>2.1</td></tr><tr><td>Sentence reward</td><td></td><td>4.385.3</td><td></td><td>1.7</td><td>2.2</td><td>89.7</td><td></td><td>3.0</td><td>4.2</td><td>91.5</td><td></td><td>1.8</td><td>5.9</td><td>76.8</td><td></td><td>2.8</td><td>5.9</td><td>76.3</td><td>87.1</td><td>2.8</td></tr><tr><td>ASC</td><td></td><td>6.5 87.3</td><td>90.5</td><td>2.4</td><td>2.3</td><td>89.9</td><td>94.3</td><td>3.5</td><td>4.0</td><td>91.5</td><td>97.5</td><td>2.6</td><td>6.6</td><td>80.9</td><td>89.4</td><td>3.2</td><td></td><td>7.7 81.7</td><td>88.7</td><td>3.2</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td></td><td>11.7 82.6</td><td>86.3</td><td>0.5</td><td>8.5</td><td></td><td>85.7 91.9</td><td>1.6</td><td>14.0</td><td>81.4</td><td>93.0</td><td>0.7</td><td>27.9</td><td>69.1</td><td>76.6</td><td>1.4</td><td>27.3</td><td>72.1</td><td>79.6</td><td>1.2</td></tr><tr><td>GRPO + KL</td><td>6.6</td><td>90.2</td><td>92.1</td><td>0.4</td><td>6.7</td><td>90.0</td><td>94.4</td><td>1.6</td><td>5.6</td><td>88.5</td><td>97.0</td><td>0.8</td><td>21.1</td><td>77.1</td><td>82.3</td><td>1.3</td><td>22.5</td><td>78.4</td><td>84.1</td><td>1.4</td></tr><tr><td>Copy penalty</td><td>8.0</td><td>91.6</td><td>90.9</td><td>0.8</td><td>6.0</td><td>87.1</td><td>92.8</td><td>2.0</td><td>8.8</td><td>88.3</td><td>94.4</td><td>1.1</td><td>17.6</td><td>80.2</td><td>84.0</td><td>1.5</td><td>20.5</td><td>78.0</td><td>85.1</td><td>1.5</td></tr><tr><td>Sentence reward</td><td>4.8</td><td>90.5</td><td></td><td>0.4</td><td>5.1</td><td>88.6</td><td></td><td>1.4</td><td>5.5</td><td>88.9</td><td></td><td>0.8</td><td>18.2</td><td>78.8</td><td></td><td>1.4</td><td>20.0</td><td>78.7</td><td>85.9</td><td>1.3</td></tr><tr><td>ASC</td><td></td><td>6.086.6</td><td>91.4</td><td>0.8</td><td>3.9</td><td></td><td>88.2 95.8</td><td>2.2</td><td></td><td>7.2 90.4 97.9</td><td></td><td>1.3</td><td>17.4</td><td>80.8 84.8</td><td></td><td>1.8</td><td>19.9</td><td>81.5</td><td>86.8</td><td>1.8</td></tr><tr><td>Llama-3.2-3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6</td><td>38.7</td><td>63.3</td><td></td><td></td><td>40.5</td><td>63.6</td><td></td><td></td></tr><tr><td>Base model GRPO + KL</td><td>18.9 14.4</td><td>84.9 89.0</td><td>81.8 91.5</td><td>0.5 0.4</td><td>11.7 5.7</td><td>83.8 89.8</td><td>89.2 90.3</td><td>1.2 1.1</td><td>10.0</td><td>14.5 75.4 88.1</td><td>91.7 97.5</td></table>

Table 12: Faithfulness on every corpus: authored error rate under the judge (Err; grey: a shortcut, not ranked; bold, underlined: lowest and second-lowest remaining Err, as in Table 2), authored support under MiniCheck-Flan-T5 (T5) and AlignScore (AS), and supported authored sentences per answer (Supp.); –: not scored.
<table><tr><td rowspan="2">Method</td><td colspan="3">ASQA</td><td colspan="4">RAGTruth</td><td colspan="4">ClapNQ</td><td colspan="4">ELI5</td><td colspan="4">ExpertQA</td></tr><tr><td>Copy</td><td>Words</td><td>Vac.</td><td>Ref.</td><td>Copy</td><td>Words</td><td>Vac. Ref.</td><td>Copy</td><td>Words</td><td>Vac.</td><td>Ref.</td><td>Copy</td><td>Words</td><td>Vac.</td><td>Ref.</td><td>Copy</td><td>Words</td><td>Vac.</td><td>Ref.</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>38.9</td><td>73.3 22.5</td><td>6.1</td><td>30.1</td><td>119.3</td><td>16.6</td><td>2.7</td><td>38.6</td><td>82.8</td><td>21.2</td><td>4.3</td><td>13.5</td><td>102.1</td><td>38.6</td><td>4.3</td><td>19.2</td><td>109.6</td><td>40.5</td><td>7.3</td></tr><tr><td>GRPO + KL</td><td>39.3</td><td>60.0</td><td>28.4 14.7</td><td></td><td>29.7 113.6</td><td>18.6</td><td>7.3</td><td>39.1</td><td>81.3</td><td>21.7</td><td>5.0</td><td>11.7</td><td>77.0</td><td>43.4</td><td>45.9</td><td>16.2</td><td>83.1</td><td>49.0</td><td>43.7</td></tr><tr><td>Copy penalty</td><td>11.8</td><td>36.9</td><td>67.4 64.5</td><td></td><td>5.0</td><td>34.3 81.6</td><td>78.7</td><td>11.4</td><td>41.3</td><td>72.7</td><td>66.0</td><td>0.9</td><td>16.3</td><td>95.2</td><td>95.4</td><td>1.8</td><td>18.7</td><td>94.8</td><td>93.7</td></tr><tr><td>Sentence reward</td><td>42.3</td><td>66.4</td><td>22.8</td><td>7.3</td><td>34.0</td><td>116.7 15.9</td><td>2.7</td><td>42.2</td><td>80.6</td><td>19.0</td><td>2.0</td><td>14.5</td><td>92.2</td><td>35.9</td><td>28.2</td><td>21.3</td><td>104.5</td><td>34.8</td><td>18.8</td></tr><tr><td>ASC</td><td>28.6</td><td>80.9</td><td>10.9</td><td>0.2</td><td>22.7</td><td>122.4</td><td>12.6 0.3</td><td>28.4</td><td>89.7</td><td>13.7</td><td>0.2</td><td>12.9</td><td>123.5</td><td>14.1</td><td>0.3</td><td>18.3</td><td>127.8</td><td>15.2</td><td>0.1</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>38.2</td><td>84.0</td><td>23.9</td><td>5.7</td><td>32.7</td><td>118.9 19.2</td><td>3.3</td><td>40.3</td><td>86.0</td><td>20.2</td><td>3.0</td><td>15.9</td><td>109.0</td><td>37.2</td><td>7.0</td><td>22.2</td><td>121.0</td><td>31.1</td><td>7.7</td></tr><tr><td>GRPO + KL</td><td>32.6</td><td>49.7</td><td>43.7 27.7</td><td></td><td>28.4</td><td>93.8 34.5</td><td>22.7</td><td>35.8</td><td>68.0</td><td>39.9</td><td>22.7</td><td>6.9</td><td>38.8</td><td>75.9</td><td>75.7</td><td>9.8</td><td>46.8</td><td>74.9</td><td>70.7</td></tr><tr><td>Copy penalty</td><td>24.4</td><td>58.0 37.9</td><td>28.0</td><td></td><td>23.2 115.5</td><td>24.0</td><td>10.7</td><td>27.7</td><td>82.5</td><td>26.4</td><td>12.3</td><td>9.8</td><td>83.8</td><td>46.4</td><td>48.7</td><td>13.7</td><td>92.0</td><td>46.8</td><td>41.0</td></tr><tr><td>Sentence reward</td><td>43.2</td><td>83.9 17.8</td><td>2.0</td><td>34.0</td><td>119.2</td><td>18.2</td><td>1.3</td><td>43.7</td><td>87.6</td><td>19.9</td><td>1.3</td><td>17.9</td><td>118.1</td><td>28.7</td><td>13.7</td><td>25.5</td><td>131.5</td><td>25.4</td><td>3.7</td></tr><tr><td>ASC</td><td>28.4</td><td>86.3</td><td>12.1</td><td>0.4</td><td>25.0</td><td>119.9 12.5</td><td>0.4</td><td>29.4</td><td>91.9</td><td>12.0</td><td>0.4</td><td>15.3</td><td>124.7</td><td>13.5</td><td>2.4</td><td>21.9</td><td>131.0</td><td>14.4</td><td>1.0</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>45.4</td><td>24.1 28.5</td><td>6.9</td><td>43.1</td><td></td><td>69.5 17.8</td><td>0.7</td><td>43.4</td><td></td><td>36.5 21.0</td><td>0.3</td><td>25.3</td><td>60.2</td><td>26.5</td><td>7.8</td><td>32.7</td><td>61.8</td><td></td><td>7.6</td></tr><tr><td>GRPO + KL</td><td>50.1</td><td>21.3</td><td>29.5</td><td>3.4</td><td>44.0</td><td>69.0 23.0</td><td>2.7</td><td>45.2</td><td>38.6</td><td>20.5</td><td>0.3</td><td>31.3</td><td>55.1</td><td>26.2</td><td>7.7</td><td>35.7</td><td>63.0</td><td>27.5 28.3</td><td>6.1</td></tr><tr><td>Copy penalty</td><td>14.2</td><td>13.7</td><td>34.2 10.4</td><td></td><td>22.8</td><td>61.5 21.9</td><td>4.0</td><td>17.6</td><td>29.9</td><td>22.3</td><td>1.0</td><td>9.3</td><td>44.4</td><td>36.5</td><td>12.1</td><td>13.8</td><td>49.5</td><td>33.2</td><td>7.3</td></tr><tr><td>Sentence reward</td><td>51.7</td><td>22.4 27.9</td><td>3.2</td><td></td><td>47.0</td><td>68.7 18.7</td><td>2.0</td><td>47.1</td><td>40.0</td><td>19.0</td><td>0.0</td><td>29.9</td><td>56.5</td><td>24.3</td><td>9.2</td><td>37.3</td><td>61.1</td><td>25.3</td><td>5.7</td></tr><tr><td>ASC</td><td>30.2</td><td>27.7</td><td>9.9</td><td>1.3</td><td>26.2</td><td>74.3 16.1</td><td>0.0</td><td>26.2</td><td>44.8</td><td>14.1</td><td>0.0</td><td>17.9</td><td>61.5</td><td>13.2</td><td>1.1</td><td>23.4</td><td></td><td>67.3 15.3</td><td>1.0</td></tr><tr><td>Llama-3.2-3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model</td><td>49.3</td><td>22.4</td><td>21.3</td><td>7.0</td><td>45.7</td><td>68.1 15.3</td><td>2.0</td><td>45.8</td><td>40.4</td><td>26.0 22.5</td><td>3.7 2.0</td><td>27.0 34.8</td><td>71.8</td><td>15.1 16.5</td><td>2.7 2.7</td><td>33.6 44.3</td><td>60.2</td><td>22.5</td><td>6.0 5.3</td></tr><tr><td>GRPO + KL Copy penalty</td><td>57.9 14.7</td><td>22.5 12.6</td><td>25.8 34.5</td><td>4.7 10.0</td><td>55.4 29.4</td><td>63.9 17.4 60.2 17.8</td><td>1.3 2.0</td><td>54.6 22.0</td><td>41.8 26.8</td></table>

Table 13: Shortcut metrics on every corpus: copy share, words per answer, vacuous share (Vac.) and refusal rate (Ref.). A policy takes a shortcut on a corpus when its copy share, vacuous share or refusal rate exceeds the base model’s by more than 3 points, or its words fall below 85% of the base model’s.

<table><tr><td rowspan="2">Corpus</td><td colspan="2">Err</td><td colspan="2">Unsupp. (J)</td><td colspan="2">Supp. (J)</td><td colspan="2">Whole</td></tr><tr><td>Base</td><td>ASC</td><td>Base</td><td>ASC</td><td>Base ASC</td><td></td><td>Base</td><td>ASC</td></tr><tr><td colspan="9">Qwen3.5-9B</td></tr><tr><td>ASQA</td><td>5.1</td><td>3.9*</td><td>0.10 0.10</td><td></td><td>1.8</td><td>2.5</td><td>13.4 10.5*</td><td></td></tr><tr><td>RAGT.</td><td>2.3</td><td>1.1</td><td>0.08</td><td>0.05</td><td>3.4</td><td>4.1</td><td>10.2</td><td>8.5</td></tr><tr><td>ClapNQ</td><td>3.6</td><td>2.2</td><td>0.07</td><td>0.06</td><td>2.0</td><td>2.7</td><td>9.5</td><td>8.8</td></tr><tr><td>ELÍ5</td><td>6.0</td><td>3.4*</td><td>0.19</td><td>0.14*</td><td>2.9</td><td>3.9</td><td></td><td>27.4 20.6*</td></tr><tr><td>Expert</td><td>6.2</td><td>3.7*</td><td>0.20</td><td>0.15*</td><td>2.9</td><td>3.9</td><td></td><td>24.6 13.9*</td></tr><tr><td colspan="9">Qwen3.5-4B</td></tr><tr><td>ASQA</td><td>8.6</td><td>6.5</td><td>0.18</td><td>0.17</td><td>1.9</td><td>2.5</td><td></td><td>14.610.9*</td></tr><tr><td>RAGT.</td><td>2.9</td><td>2.3</td><td>0.09</td><td>0.09</td><td>3.2</td><td>3.8</td><td>10.4</td><td>9.4</td></tr><tr><td>ClapNQ</td><td>5.8</td><td>4.0*</td><td>0.12</td><td>0.11</td><td>1.9</td><td>2.7</td><td>9.1</td><td>7.9</td></tr><tr><td>ELÍ5</td><td>8.8</td><td>6.6*</td><td>0.28</td><td>0.26</td><td>2.9</td><td>3.7</td><td></td><td>24.6 18.1*</td></tr><tr><td>Expert</td><td>8.7</td><td>7.7</td><td>0.28</td><td>0.30</td><td>3.0</td><td>3.6</td><td></td><td>30.0 16.2*</td></tr><tr><td colspan="9">Llama-3.1-8B</td></tr><tr><td>ASQA</td><td>11.7</td><td>6.0*</td><td>0.07</td><td>0.05</td><td>0.5</td><td>0.8</td><td>11.911.5</td><td></td></tr><tr><td>RAGT.</td><td>8.5</td><td>3.9*</td><td>0.15</td><td>0.10</td><td>1.7</td><td>2.4</td><td></td><td>11.1 11.6</td></tr><tr><td>ClapNQ</td><td>14.0</td><td>7.2*</td><td>0.12</td><td>0.10</td><td>0.7</td><td>1.3</td><td>13.3</td><td>9.1*</td></tr><tr><td>ELÍ5</td><td>27.9</td><td>917.4*</td><td>0.55</td><td>0.38*</td><td>1.4</td><td>1.8</td><td></td><td>26.1 17.9*</td></tr><tr><td>Expert</td><td></td><td>27.3 19.9*</td><td>0.46</td><td>0.43</td><td>1.2</td><td>1.7</td><td></td><td>21.5 16.2*</td></tr><tr><td colspan="9">Llama-3.2-3B</td></tr><tr><td>ASQA</td><td>18.915.0</td><td></td><td>0.100.13</td><td></td><td>0.4</td><td>0.7</td><td></td><td>10.2 11.8</td></tr><tr><td>RAGT.</td><td>11.7</td><td>7.1*</td><td>0.17</td><td>0.16</td><td>1.3</td><td>2.1</td><td></td><td>12.211.4</td></tr><tr><td>ClapNQ</td><td>14.5</td><td>8.7*</td><td>0.11</td><td>0.13</td><td>0.7</td><td>1.3</td><td></td><td>15.913.8</td></tr><tr><td>ELÍ5</td><td></td><td>38.7 24.9*</td><td>0.84</td><td>0.69*</td><td>1.3</td><td>2.1</td><td></td><td>31.623.6*</td></tr><tr><td>Expert</td><td></td><td>40.5 25.0*</td><td></td><td>0.660.57</td><td>1.0</td><td>1.7</td><td></td><td>28.020.4*</td></tr></table>

Table 14: Base model against ASC on every corpus (ASC’s seeds pooled per question): authored error rate (Err), unsupported and supported authored sentences per answer under the judge (Unsupp. (J), Supp. (J)), and the share (%) of all sentences, copied ones included, that MiniCheck-Flan-T5 does not support (Whole). <sup>∗</sup> (on Err, Unsupp. and Whole): the 95% bootstrap interval over questions of the difference from the base model excludes zero. Pooling seeds per question makes Err differ slightly from Table 2.
<table><tr><td>Model</td><td>Run</td><td>ASQA</td><td>RAGT.</td><td>ClapNQ</td><td>ELI5</td><td>Expert</td></tr><tr><td>Qwen3.5-9B</td><td>Base model</td><td>5.1</td><td>2.3</td><td>3.6</td><td>6.0</td><td>6.2</td></tr><tr><td></td><td>Seed 1</td><td>4.5</td><td>1.5</td><td>2.0</td><td>3.3</td><td>4.1</td></tr><tr><td></td><td>Seed 2</td><td>3.4</td><td>0.8</td><td>2.3</td><td>3.5</td><td>3.4</td></tr><tr><td>Qwen3.5-4B</td><td>Base model</td><td>8.6</td><td>2.9</td><td>5.8</td><td>8.8</td><td>8.7</td></tr><tr><td></td><td>Seed 1</td><td>6.6</td><td>2.0</td><td>3.5</td><td>6.3</td><td>6.5</td></tr><tr><td></td><td>Seed 2</td><td>5.7</td><td>2.6</td><td>4.0</td><td>6.3</td><td>8.6</td></tr><tr><td></td><td>Seed 3</td><td>7.1</td><td>2.1</td><td>4.5</td><td>7.1</td><td>7.9</td></tr><tr><td>Llama-3.1-8B</td><td>Base model</td><td>11.7</td><td>8.5</td><td>14.0</td><td>27.9</td><td>27.3</td></tr><tr><td></td><td>Seed 1</td><td>6.0</td><td>3.9</td><td>7.2</td><td>17.4</td><td>19.9</td></tr><tr><td>Llama-3.2-3B</td><td>Base model</td><td>18.9</td><td>11.7</td><td>14.5</td><td>38.7</td><td>40.5</td></tr><tr><td></td><td>Seed 1</td><td>15.0</td><td>7.1</td><td>8.7</td><td>24.9</td><td>25.0</td></tr></table>

Table 15: Authored error rate of every ASC training run on every corpus, with the base model above.

## D.5 Near-copies

A policy trained against a copy test could learn to copy just below its threshold. Section 6.4 rules this out for verbatim copying (just below: a copy share of 0.40 to 0.50), and with the threshold lowered to 0.4, ASC also keeps its lower error rate. Beyond verbatim matches, an authored sentence of ASC is as close to its closest passage sentence as one of the base model (median ROUGE-L precision 0.42 for both; Lin, 2004), and ASC writes fewer near-copies, with a ROUGE-L of at least 0.8, in 17 of the 20 settings. Within terciles of closeness, its authored error rate is below the base model’s in 58 of the 60 tercile and setting pairs, so its gain is not a product of closeness.

<table><tr><td rowspan="2">Margin</td><td rowspan="2">ASC flagged</td><td colspan="3">ASC lowest Err</td><td rowspan="2">Baseline cells flagged</td></tr><tr><td>75%</td><td>85%</td><td>95%</td></tr><tr><td>1</td><td>0</td><td>20</td><td>20</td><td>20</td><td>59</td></tr><tr><td>2</td><td>0</td><td>18</td><td>18</td><td>18</td><td>57</td></tr><tr><td>3</td><td>0</td><td>18</td><td>18</td><td>18</td><td>55</td></tr><tr><td>4</td><td>0</td><td>15</td><td>15</td><td>16</td><td>48</td></tr><tr><td>5</td><td>0</td><td>15</td><td>15</td><td>16</td><td>43</td></tr><tr><td>7.5</td><td>0</td><td>12</td><td>12</td><td>14</td><td>37</td></tr><tr><td>10</td><td>0</td><td>8</td><td>8</td><td>12</td><td>29</td></tr></table>

Table 16: Table 2 under other shortcut rules. Margin: points above the base model at which copy share, vacuous share or refusals flag a shortcut. ASC flagged: settings (of 20) in which ASC takes a shortcut. ASC lowest Err: settings in which it has the lowest Err among policies without a shortcut, for word floors of 75%, 85% and 95% of the base model’s words. Baseline cells flagged: baseline Err cells (of 60) with a shortcut, floor 85%. Table 2 uses a margin of 3 and a floor of 85%.

## D.6 Training dynamics

Figure 7 follows GRPO + KL and ASC through training on every model, and the copy penalty on Qwen3.5-9B and Llama-3.1-8B. The baselines shortcuts build up over training: GRPO + KL’s refusals rise from the first checkpoint on the Qwen models, up to 29% of answers for Qwen3.5-4B, and its copy share rises on the Llama models, while the copy penalty’s answers shrink as its refusals and vacuous share grow. Their authored support ends at or above ASC’s, but over the fewer sentences they still write. ASC settles early on every model: its copy share falls within the first quarter of training and then stays flat, its vacuous share falls by mid-training, and it ends with longer answers and fewer refusals than the base model.

## E Answer Quality

Section 6.2 compares whole answers with LLM judges. This section gives the protocol, how its rubric was chosen without favoring any method, the result in every setting, a second judge, and checks that need no judge preference: reference coverage, refusals and answerable questions.

![](images/d17acd160410e95bc6a70d0d4a6bbebf5fc709ba394a76bc709226e754f0a9e0.jpg)  
Figure 7: All four models on ASQA over training (0: base model): GRPO + KL, ASC, and the copy penalty where its intermediate checkpoints were evaluated.

Protocol. The judge is Gemma-4-31B, the judge of the authored error rate. It sees the question, the passages and two answers and returns the answer that serves a reader better, or a tie, with a onesentence reason (prompt in Appendix H). Every pair is judged in both orders; a pair is a win only if both orders agree, any other outcome is a tie, and a tie counts half in the win rate. We compare ASC (first seed) with the base model and each baseline, and each baseline with the base model, on up to 300 questions per setting, about 5,400 pairs each.

Choosing the rubric. LLM judges favor longer answers and a fixed position (Zheng et al., 2023; Wang et al., 2024), and a judge that favored quoted or long text would rank methods by style. We wrote three rubrics and measured each on probes built from base-model answers only (Table 17): the same content as a verbatim passage sentence or in the model’s own words (404 pairs), an answer padded with a restated sentence (200 pairs), and an answer with one supported sentence replaced by an unsupported one (153 pairs), which a useful judge must notice. The rule, fixed in advance, keeps the rubrics that prefer the intact answer at least 80% of the time, takes the one closest to 50% on the copy probe, and breaks near-ties by the smaller preference for padding. The chosen rubric states that quoting is neither a merit nor a fault and that length is no merit, and counts a refusal as right only when the passages lack the answer; it was fixed, and its hash recorded, before any answer of a trained model was judged.

<table><tr><td rowspan="2">Probe (ideal value)</td><td colspan="3">Rubric</td></tr><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>Copied form preferred (50)</td><td>44.8</td><td>47.2</td><td>45.9</td></tr><tr><td>Padded answer preferred (≤50)</td><td>3.8</td><td>15.5</td><td>11.5</td></tr><tr><td>Intact answer preferred (high)</td><td>99.3</td><td>99.3</td><td>99.7</td></tr></table>

Table 17: Probes (%) for the three candidate rubrics, built from base-model answers only: two biases (copied form, padding) and the sensitivity to a planted error. Rubric 2 (bold) is the one used.

![](images/3a1daababb7813c441e0b7d27e5baa31865980d4e8b042d05159a0654bd6ac2b.jpg)  
Figure 8: Gain in the training reward over the base model against (a) the win rate against the base model and (b) the change in judge-supported authored sentences per answer. Markers: mean over the 20 settings with 95% intervals; light points: ASC in each setting. Every method raises the reward, but only ASC keeps the base model’s quality and adds content.

Preference. Table 18 gives every setting. Significant verdicts favor ASC more often than the other method in every comparison: over the base model in 5 settings (the base model in 2), and over the copy penalty, GRPO + KL and the sentence reward in 14, 9 and 3 (each of them in at most 2). Figure 8 adds the spread over the 20 settings to Table 3, and the change in supported authored sentences: only ASC adds supported content. Controlled for length, by a logistic fit of the verdict on the difference in words, its win rates are 48.9% against the base model, 50.1% against GRPO + KL, 54.5% against the copy penalty and 49.2% against the sentence reward; this control also removes the credit for the extra supported sentences, which come with more words, so it likely understates ASC’s win rates. A second seed on Qwen3.5-9B gives the same ordering (52.1% and 56.4% against the base model and GRPO + KL).

Second judge. Mistral-Small-3.2-24B-Instruct, from a third model family, judges the same pairs with the same rubric and ranks the methods as Gemma does (Table 3), although its copy probe shows a bias against ASC (it prefers the copied form in 62.9% of the pairs). On the pairs that both judges decide, 45% of all pairs, they agree on 81.7% (κ 0.61).

Reference coverage. ASQA, ClapNQ and ELI5 come with reference answers; the judge splits each reference into key points and checks which ones an answer conveys. ASC covers the most key points on average (Table 3), significantly more than the base model in 4 of the 12 settings and less in none. The NLI model TRUE (Honovich et al., 2022), which is not an LLM judge, agrees with Gemma on 90.7% of the key points (κ 0.80) and gives the same direction (+1.3 points).

Refusals. Refusing is right when the passages lack the answer, and the base model mostly refuses for that reason: the judge finds that the passages do not answer 85.4% of its 575 refusals. GRPO + KL, the sentence reward and ASC also refuse mostly on such questions (79.0%, 93.7% and 88.8% of their refusals), but the copy penalty does not: 47.2% of its 3,821 refusals fall on questions the passages answer fully or in part. ASC answers most of the questions the base model rightly refuses, with an authored error rate of 5.5%, and there the judge prefers the refusal (ASC wins 32.3% of 181 pairs); where both models answer, ASC wins 51.5%.

Answerable questions. On the questions whose passages contain the answer (labeled by the judge; Mistral agrees on 84% of 300 sampled questions), ASC has a lower error rate than the base model in 19 of the 20 settings and writes significantly more supported authored sentences in all 20 (Table 19).

## F Ablation Details

Each ablation of Section 6.3 changes one part of ASC on Qwen3.5-9B and keeps every other setting of Appendix B; the answer-advantage variant sums the graded credit over the answer. Table 20 gives every corpus. The vacuity test matters where vacuous sentences are common: without it, Err rises on ASQA, ELI5 and ExpertQA, the three corpora where the base model writes the most vacuous sentences, and falls slightly on RAGTruth and ClapNQ. Removing the copy test or the graded credit, or using the answer-level advantage, raises Err or brings copying back on at least four of the five corpora.

## G Human Evaluation

Two graduate-level annotators experienced in the field label sentences and compare answers, blind to the method that wrote them.

Sentence labels. The annotators label authored sentences as supported or unsupported by the passages. On a judge-validity set drawn from basemodel and trained answers on ASQA and ELI5, the judge agrees with people at least as often on the sentences of ASC (90.8%) as on the base model’s (85.5%), in both model families. In a direct test with no judge, one random authored sentence from the base model and one from ASC for each of 250 ELI5 questions answered by Llama-3.1-8B, 24.8% of the sentences of ASC are unsupported against 33.2% of the base model’s (exact McNemar p = 0.01). On 150 shared sentences, the annotators agree with a Cohen’s kappa of 0.65.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Corpus</td><td colspan="4">Win rate of ASC against</td><td colspan="4">Controlled for length</td><td colspan="2">Coverage</td></tr><tr><td>Base</td><td>GRPO + KL</td><td>Copy pen.</td><td>Sent. rew.</td><td>Base</td><td>GRPO + KL</td><td>Copy pen.</td><td>Sent. rew.</td><td>Base</td><td>ASC</td></tr><tr><td>Qwen3.5-9B</td><td>ASQA</td><td>45.5</td><td>52.0</td><td>72.2</td><td>50.7</td><td>43.9</td><td>43.9</td><td>57.4</td><td>45.2</td><td>50.9</td><td>52.0</td></tr><tr><td></td><td>RAGTruth</td><td>52.3</td><td>55.0</td><td>88.3</td><td>52.7</td><td>52.5</td><td>52.6</td><td>61.1</td><td>50.9</td><td></td><td></td></tr><tr><td></td><td>ClapNQ</td><td>54.2</td><td>53.0</td><td>78.3</td><td>52.7</td><td>52.7</td><td>51.4</td><td>59.1</td><td>50.7</td><td>89.5</td><td>93.5</td></tr><tr><td></td><td>ELI5</td><td>51.2</td><td>51.7</td><td>76.5</td><td>53.2</td><td>47.4</td><td>43.6</td><td>55.9</td><td>47.4</td><td>14.4</td><td>14.9</td></tr><tr><td></td><td>ExpertQA</td><td>50.5</td><td>59.5</td><td>83.8</td><td>54.8</td><td>45.0</td><td>48.8</td><td>51.7</td><td>48.8</td><td></td><td></td></tr><tr><td>Qwen3.5-4B</td><td>ASQA</td><td>49.5</td><td>61.8</td><td>59.7</td><td>48.8</td><td>49.4</td><td>50.9</td><td>54.2</td><td>48.5</td><td>50.9</td><td>51.2</td></tr><tr><td></td><td>RAGTruth</td><td>54.3</td><td>65.0</td><td>52.0</td><td>53.7</td><td>52.3</td><td>55.0</td><td>50.4</td><td>52.8</td><td></td><td></td></tr><tr><td></td><td>ClapNQ</td><td>51.0</td><td>62.7</td><td>52.8</td><td>52.0</td><td>49.2</td><td>51.3</td><td>50.2</td><td>51.1</td><td>91.0</td><td>92.6</td></tr><tr><td></td><td>ELI5</td><td>41.3</td><td>62.7</td><td>44.0</td><td>42.7</td><td>39.4</td><td>53.6</td><td>42.5</td><td>42.1</td><td>14.1</td><td>14.9</td></tr><tr><td></td><td>ExpertQA</td><td>46.7</td><td>72.2</td><td>57.2</td><td>49.3</td><td>45.7</td><td>53.0</td><td>51.1</td><td>50.3</td><td>一</td><td>一</td></tr><tr><td>Llama-3.1-8B</td><td>ASQA</td><td>53.0</td><td>58.0</td><td>69.4</td><td>57.6</td><td>47.7</td><td>50.8</td><td>51.8</td><td>53.2</td><td>37.9</td><td>38.3</td></tr><tr><td></td><td>RAGTruth</td><td>48.7</td><td>49.0</td><td>61.4</td><td>48.3</td><td>43.7</td><td>43.2</td><td>50.5</td><td>41.8</td><td></td><td></td></tr><tr><td></td><td>ClapNQ</td><td>58.0</td><td>57.3</td><td>72.8</td><td>52.3</td><td>47.5</td><td>50.6</td><td>54.8</td><td>46.7</td><td>63.6</td><td>71.2</td></tr><tr><td></td><td>ELÍ5 ExpertQA</td><td>49.2</td><td>49.8</td><td>51.2</td><td>44.7</td><td>49.2</td><td>48.4</td><td>48.3</td><td>43.2</td><td>13.1</td><td>13.1</td></tr><tr><td></td><td></td><td>48.2</td><td>49.5</td><td>60.7</td><td>51.5</td><td>45.4</td><td>47.7</td><td>56.8</td><td>48.1</td><td>一</td><td></td></tr><tr><td>Llama-3.2-3B</td><td>ASQA</td><td>52.0</td><td>53.3</td><td>67.9</td><td>57.5</td><td>50.8</td><td>51.0</td><td>53.6</td><td>55.1</td><td>36.6</td><td>37.6</td></tr><tr><td></td><td>RAGTruth</td><td>50.7</td><td>51.0</td><td>55.7</td><td>54.7</td><td>52.4</td><td>48.3</td><td>48.4</td><td>54.6</td><td></td><td></td></tr><tr><td></td><td>ClapNQ</td><td>55.2</td><td>56.2</td><td>73.8</td><td>49.5</td><td>51.4</td><td>51.9</td><td>56.3</td><td>48.5</td><td>65.2</td><td>70.6</td></tr><tr><td></td><td>ELÍ5</td><td>54.3</td><td>53.2</td><td>54.3</td><td>53.2</td><td>54.3</td><td>52.1</td><td>50.1</td><td>52.0</td><td>12.7</td><td>12.0</td></tr><tr><td></td><td>ExpertQA</td><td>54.8</td><td>53.2</td><td>59.5</td><td>52.3</td><td>52.6</td><td>52.2</td><td>56.6</td><td>51.0</td><td>一</td><td></td></tr></table>

Table 18: Win rate (%) of ASC against the base model and each baseline in every setting under Gemma-4-31B, as judged and controlled for the difference in words, and reference coverage (%) of the base model and ASC (–: no reference answers). Bold: the 95% bootstrap interval over questions lies above 50 (ASC preferred), or for coverage, ASC’s difference from the base model lies above zero; italic: below. ASC: first seed.

<table><tr><td rowspan="2" colspan="2">Corpus Questions</td><td colspan="2">Err</td><td colspan="2">Supp.</td></tr><tr><td>Base</td><td>ASC</td><td>Base</td><td>ASC</td></tr><tr><td colspan="6">Qwen3.5-9B</td></tr><tr><td>ASQA</td><td>663</td><td>3.9</td><td>3.3</td><td></td><td>1.54 2.22*</td></tr><tr><td>RAGT.</td><td>91</td><td>1.2</td><td>1.3</td><td></td><td>3.16 3.96*</td></tr><tr><td>ClapNQ</td><td>235</td><td>3.1</td><td>2.4</td><td></td><td>1.83 2.57*</td></tr><tr><td>ELÍ5</td><td>248</td><td>3.7</td><td>2.5*</td><td></td><td>3.08 3.53*</td></tr><tr><td>Expert</td><td>363</td><td>5.2</td><td>3.5*</td><td></td><td>3.12 3.82*</td></tr><tr><td colspan="6">Qwen3.5-4B</td></tr><tr><td>ASQA</td><td>214</td><td>8.5</td><td>5.2*</td><td></td><td>1.672.37*</td></tr><tr><td>RAGT.</td><td>91</td><td>2.6</td><td>1.6</td><td></td><td>3.07 3.75*</td></tr><tr><td>ClapNQ</td><td>235</td><td>5.2</td><td>3.6</td><td></td><td>1.87 2.65*</td></tr><tr><td>ELÍ5</td><td>70</td><td>7.3</td><td>4.3*</td><td></td><td>2.74 3.38*</td></tr><tr><td>Expert</td><td>86</td><td>6.3</td><td>4.6</td><td></td><td>3.05 3.66*</td></tr><tr><td colspan="6">Llama-3.1-8B</td></tr><tr><td>ASQA</td><td>663</td><td>7.1</td><td>3.1*</td><td></td><td>0.46 0.78*</td></tr><tr><td>RAGT.</td><td>91</td><td>3.9</td><td>1.2</td><td></td><td>1.71 2.42*</td></tr><tr><td>ClapNQ</td><td>235</td><td>9.7</td><td>5.2*</td><td>0.75</td><td>1.35*</td></tr><tr><td>ELÍ5</td><td>248</td><td>10.6</td><td>5.8*</td><td></td><td>1.561.94*</td></tr><tr><td>Expert</td><td>363</td><td>10.3</td><td>6.9*</td><td>1.25</td><td>51.78*</td></tr><tr><td colspan="6">Llama-3.2-3B</td></tr><tr><td>ASQA</td><td>214</td><td>16.0</td><td>10.4</td><td></td><td>0.43 0.75*</td></tr><tr><td>RAGT.</td><td>91</td><td>8.4</td><td>4.2</td><td></td><td>1.13 2.11*</td></tr><tr><td>ClapNQ</td><td>235</td><td>10.2</td><td>6.9</td><td></td><td>0.561.26*</td></tr><tr><td>ELÍ5</td><td>70</td><td>17.2</td><td>9.8</td><td></td><td>1.41 1.93*</td></tr><tr><td>Expert</td><td>86</td><td>15.5</td><td>11.8</td><td></td><td>1.07 1.73*</td></tr></table>

Table 19: Base model against ASC on the questions whose passages contain the answer: questions, authored error rate (Err) and supported authored sentences per answer (Supp.). <sup>∗</sup>: a significant difference from the base model (bootstrap over questions, Appendix D).

Answer preference. The annotators compare 150 pairs of answers, 75 of ASC against the base model and 75 against GRPO + KL, drawn with a fixed seed across the four models and five corpora; the order of the two answers is random, and they follow the rubric of Appendix E. Counting a tie as half a win, they prefer ASC to the base model in 56.7% of the pairs and to GRPO + KL in 59.3%. They agree with a Cohen’s kappa of 0.66, and on the pairs where both decide, their verdict matches Gemma’s in 89.8% and Mistral’s in 88.9%.

## H Prompts

Figure 9 gives every prompt verbatim. All models answer with the same prompt in one user turn, and thinking is disabled on the Qwen models. The sentence judge’s verdict is read as its probability of answering “Yes”; the pairwise rubric was fixed before any trained answer was judged (Appendix E).

<table><tr><td rowspan="2">Method</td><td colspan="4">ASQA</td><td colspan="4">RAGTruth</td><td colspan="4">ClapNQ</td><td colspan="4">ELI5</td><td colspan="4">ExpertQA</td></tr><tr><td>Err Copy</td><td></td><td>Vac.</td><td>Supp.</td><td>Err</td><td>Copy</td><td>Vac.</td><td>Supp.</td><td>Err</td><td>Copy</td><td>Vac.</td><td>Supp.</td><td>Err</td><td>Copy</td><td>Vac.</td><td>Supp.</td><td>Err</td><td>Copy</td><td>Vac.</td><td>Supp.</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASC</td><td>4.0</td><td>28.6</td><td>10.9</td><td>2.3</td><td>1.1</td><td>22.7</td><td>12.6</td><td>3.8</td><td>2.2</td><td>28.4 13.7</td><td></td><td>2.5</td><td>3.4</td><td>12.9</td><td>14.1</td><td></td><td>3.2 3.7</td><td>18.3</td><td>15.2</td><td>3.4</td></tr><tr><td>copy test</td><td>2.4</td><td>46.3</td><td>11.1</td><td>1.4</td><td>1.5</td><td>31.8</td><td>10.0</td><td>3.4</td><td>1.6</td><td>42.2</td><td>11.5</td><td>1.9</td><td>3.5</td><td>19.1</td><td>10.4</td><td>3.1</td><td>3.5</td><td>26.9</td><td>12.7</td><td>3.2</td></tr><tr><td>vacuity test</td><td>5.2</td><td>26.9</td><td>17.2</td><td>2.0</td><td>1.0</td><td>24.0</td><td>15.9</td><td>3.7</td><td>1.6</td><td>26.7</td><td>17.8</td><td>2.4</td><td>4.6</td><td>11.3</td><td>23.4</td><td>3.0 7.5</td><td></td><td>17.8</td><td>21.4</td><td>3.3</td></tr><tr><td>Sentence reward</td><td>3.7</td><td>42.3</td><td>22.8</td><td>1.5</td><td>0.9</td><td>34.0</td><td>15.9</td><td>2.9</td><td>3.1</td><td>42.2</td><td>19.0</td><td>1.7</td><td>3.5</td><td>14.5</td><td>35.9</td><td>2.4</td><td>4.3</td><td>21.3</td><td>34.8</td><td>2.5</td></tr><tr><td>Answer advantage</td><td>5.4</td><td>33.1</td><td>15.2</td><td>2.8</td><td>2.3</td><td>26.3</td><td>12.5</td><td>4.4</td><td>1.9</td><td>33.4</td><td>15.2</td><td>2.9</td><td>5.1</td><td>13.0</td><td>24.0</td><td>3.2</td><td>5.1</td><td>19.9</td><td>24.6</td><td>3.3</td></tr><tr><td>Binary credit</td><td>4.6</td><td>26.6</td><td>10.1</td><td>2.3</td><td>1.7</td><td>21.7</td><td>12.0</td><td>3.8</td><td>2.5</td><td>26.6</td><td>12.9</td><td>2.5</td><td>4.1</td><td>12.3</td><td>13.7</td><td>2.9</td><td>4.4</td><td>17.4</td><td>15.2</td><td>3.3</td></tr></table>

Table 20: Ablations on Qwen3.5-9B on every corpus: authored error rate (Err; grey, bold and underlined as in Table 4), copy and vacuous share, and supported authored sentences per answer (Supp.).

## I Example Answers

Examples 1–4 show one question for each model, two from training corpora and two from held-out ones, answered by all five policies of Table 2. Each example is chosen to be representative of its setting. Among the questions whose passages and five answers fit in one box, and which the base model answers without declining any part, we take the one whose answers best show each policy’s typical behavior in that setting (Appendix D): ASC copies no more than the base model and writes more grounded sentences, as it does at the median, and as many baselines as one question allows take their most frequent shortcut there. Each box gives the question, every passage as the model saw it, and the five answers. In the passages, highlighted text is repeated verbatim, five words or more, by at least one answer, so the copied material is visible at a glance. In the answers, every sentence is marked as copied , authored and supported , authored and unsupported<sub>U</sub> or vacuous<sub>V</sub> (supported or not under the judge; vacuous under the reward’s checker), and ASC’s answer is shaded.

In Example 1, the base model copies its first two sentences and adds an unsupported claim about the Qt toolkit; GRPO + KL keeps the two copied sentences and the sentence reward copies a third, and the copy penalty declines the question, as it does on 95.4% of ELI5 questions for Qwen3.5-9B. ASC conveys the same content in its own words, in four grounded sentences, with no error. In Example 2, GRPO + KL and the sentence reward copy two and three of their three sentences, the copy penalty ends on a vacuous one, and ASC copies one of its four sentences and grounds the other three. In Example 3, the base model overstates the passage (California taxes no lottery winnings at all), GRPO + KL and the copy penalty decline a question the passage answers, the sentence reward opens with a copied sentence, and ASC writes four grounded sentences, including the federal withholding rules. In Example 4, GRPO + KL and the sentence reward shrink the answer to one copied sentence and the copy penalty to a vacuous one; ASC writes three grounded sentences.

<table><tr><td>Answer prompt (all models, one user turn)</td></tr><tr><td>You are answering a question using only the passages provided below.</td></tr><tr><td>Write a complete, self-contained answer in your own words, as connected prose. Use only information that is stated in the passages; do not add facts, examples, or explanations from anywhere else. If the passages do not contain the answer, say that they do not.</td></tr><tr><td>Do not refer to the passages by number, do not write citation markers such as [1], and do not restate the question or add a</td></tr><tr><td>preamble. Give the answer and nothing else.</td></tr><tr><td>{passages} {question}</td></tr></table>

<table><tr><td>Sentence judge behind Err (Gemma-4-31B)</td></tr><tr><td>Below are a source document and a single statement taken from a response written about it. Judge only whether the statement is supported by the source document: every fact it asserts must be stated in the document or follow directly from it.</td></tr><tr><td>Answer No if the statement contradicts the document, or if it adds any detail the document does not contain—even a plausible one. Do not use outside knowledge, and do not judge whether the statement is well written or useful.</td></tr><tr><td></td></tr><tr><td>### Source document {passages}</td></tr><tr><td>### Statement</td></tr><tr><td>{sentence} Is the statement fully supported by the source document? Answer Yes or No.</td></tr></table>

<table><tr><td>Pairwise judge (Gemma-4-31B and Mistral-Small-3.2-24B)</td></tr><tr><td>You are comparing two answers to the same question. Both were written from the passages below, which are the only information the writer had. Decide which answer serves a reader better. First check the content. Go through the claims of each answer and compare each one with the passages: is it stated in the passages or does it follow directly from them? A claim that adds a detail the passages do not contain, or that contradicts</td></tr><tr><td>them, is an error, and errors weigh heavily. Then ask which answer gives the reader more of what the question asks for, correctly. Then, only to separate answers that are otherwise equal, prefer the clearer one. Judge what the claims say, not how they are worded. Many accurate answers restate the passages in new words; others quote them. Neither form is better: do not reward an answer for sounding close to the passages, and do not reward it for sounding</td></tr><tr><td>original. Do not reward length by itself; a repeated or padded point adds nothing. If the passages do contain the answer, declining to answer is a failure; if they do not, saying so is correct. Ignore the order in which the answers appear. Choose</td></tr><tr><td>“tie&quot; if neither answer is better. ### Question</td></tr><tr><td>{question} ### Passages</td></tr><tr><td>{passages}</td></tr><tr><td>### Answer A {answer_a}</td></tr><tr><td>### Answer B {answer_b} Reply with one JSON object and nothing else: {&quot;reason&quot;: &quot;&lt;one sentence&gt;&quot;, &quot;verdict&quot;: &quot;A&quot; I &quot;B&quot; | &quot;tie&quot;}</td></tr></table>

Figure 9: The three prompts, verbatim: the answer prompt shared by all models (top), the sentence judge behind the authored error rate (middle), and the pairwise judge of Appendix E (bottom).

## Passages

## Answers

## Passages

## Answers

## Passages

[1] California State Lottery. All prizes for Fantasy 5, Daily Derby, Daily 3, Daily 4, and non-jackpot SuperLotto Plus, Mega Millions, and Powerball prizes, are paid out in one payment, less 25 % or 33 % (depending upon the winner’s tax documentation) Federal withholding if the prize is over \$5,000. Merchandise prizes over \$5,000 are subject to 33 % Federal withholding. Scratchers tickets are generally one - payment prizes; however, some games have annuity options for payments each year, or per week. California does not tax California Lottery winnings, however it taxes lottery winnings from other jurisdictions. For SuperLotto Plus and Mega Millions jackpots, the player may choose a single cash payout for a floating percentage of the jackpot, or an annuity. The SuperLotto Plus, Mega Millions, and Powerball payment schedule are on a graduated basis over 30 annual payments. Until 2005, when California joined Mega Millions, the payment choice on SuperLotto Plus had to be made when the ticket was bought. Since then, there is a 60 - day window after winning, in which the choice of cash or annuity is to be made.

## Answers

Base model: California does not tax lottery winnings. However, the state does tax lottery winnings from other jurisdictions. Additionally, federal withholding applies to prizes over \$5,000, with rates of 25% or 33% depending on the winner’s tax documentation, and merchandise prizes over \$5,000 are subject to a 33% federal withholding rate.

GRPO + KL: The provided passages do not contain the answer to this question.<sub>V</sub>

Copy penalty: The provided passages do not contain the answer to what the specific tax rate is for lottery winnings in California.

Sentence reward: The provided passages indicate that California does not tax California Lottery winnings. However, the text notes that California does tax lottery winnings from other jurisdictions.<sub>A</sub>

ASC: California does not impose a tax on lottery winnings won within the state, although it does tax winnings obtained from other jurisdictions. Regarding federal withholding, prizes exceeding \$5,000 for games such as Fantasy 5, Daily Derby, Daily 3, Daily 4, and non-jackpot SuperLotto Plus, Mega Millions, and Powerball are subject to a federal withholding rate of either 25% or 33%, depending on the winner’s tax documentation. Merchandise prizes over \$5,000 are specifically subject to a 33% federal withholding rate. For the jackpots of SuperLotto Plus and Mega Millions, players may elect a single cash payout or an annuity option, with the payment schedule for these games, as well as Powerball, following a graduated basis over 30 annual payments.<sub>A</sub>

## Example 4: Llama-3.2-3B on ASQA (in domain) Question: What kind of series is game of thrones?

## Passages

[1] A Game ofThrones. A Game of Thrones A Game of Thrones is the first novel in “A Song of Ice and Fire”, a series of fantasy novels by American author George R. R. Martin. It was first published on August 1, 1996. The novel won the 1997 Locus Award and was nominated for both the 1997 Nebula Award and the 1997 World Fantasy Award. The novella “Blood of the Dragon”, comprising the Daenerys Targaryen chapters from the novel, won the 1997 Hugo Award for Best Novella. In January 2011 the novel became a “New York Times” Bestseller and reached #1 on the list

[2] Game of Thrones. in a fantasy setting of magic and dragons. In a 2012 study of deaths per episode, it ranked second out of 40 recent U.S. TV drama series (with an average of 14). The series is generally praised for what is perceived as a sort of medieval realism. George R.R. Martin set out to make the story feel more like historical fiction than contemporary fantasy, with less emphasis on magic and sorcery and more on battles, political intrigue, and the characters, believing that magic should be used moderately in the epic fantasy genre. Martin has stated that “the true horrors of

[3] Game ofThrones (season 1). The majority of reviews for the first season were very positive, with critics noting the high production values, the well-realized world, compelling characters, and giving particular note to the strength of the child actors. Tim Goodman’s review for “The Hollywood Reporter” stated, “a few minutes into HBO’s epic Game of Thrones series, it’s clear that the hype was right and the wait was worth it”. Mary McNamara from the “Los Angeles Times” called it “...a great and thundering series of political and psychological intrigue bristling with vivid characters, cross-hatched with tantalizing plotlines and seasoned with a splash of fantasy”. “New

[4] Game of Thrones. in the pilot, playing a character described as “a charismatic socialite hiding a dark secret.” Game of Thrones Game of Thrones is an American fantasy drama television series created by David Benioff and D. B. Weiss. It is an adaptation of “A Song of Ice and Fire”, George R. R. Martin’s series of fantasy novels, the first of which is “A Game of Thrones”. It is filmed in Belfast and elsewhere in Northern Ireland, Canada, Croatia, Iceland, Malta, Morocco, Scotland, Spain, and the United States. The series premiered on HBO in the United States on April 17, 2011, and its

[5] A Game of Thrones (board game). A Game of Thrones (board game) A Game of Thrones is a strategy board game created by Christian T. Petersen and released by Fantasy Flight Games in 2003. The game is based on the “A Song of Ice and Fire” fantasy series by George R. R. Martin. It was followed in 2004 by the expansion “A Clash of Kings”, and in 2006 by the expansion “A Storm of Swords”. “A Game of Thrones” allows the players to take on the roles of several of the Great Houses vying for control of the Seven Kingdoms, including House Stark, House Lannister, House

## Answers

Base model: A fantasy series set in a medieval-like world with magic and dragons, with a focus on battles, political intrigue, and character development, and an adaptation of a series of fantasy novels by George R.R. Martin.<sub>A</sub>

GRPO + KL: Game of Thrones is a fantasy series.<sub>C</sub>

Copy penalty: The series is a fantasy series.<sub>V</sub>

Sentence reward: Game of Thrones is a fantasy series.<sub>C</sub>

ASC: The series is a fantasy series with elements of historical fiction, and it is set in a medieval-like world with magic and dragons. The author, George R.R. Martin, aimed to create a more realistic story with a focus on battles, politics, and character development, rather than relying heavily on magic. The series has been adapted into a television show, which was praised for its production values, characters, and plotlines.<sub>A</sub>