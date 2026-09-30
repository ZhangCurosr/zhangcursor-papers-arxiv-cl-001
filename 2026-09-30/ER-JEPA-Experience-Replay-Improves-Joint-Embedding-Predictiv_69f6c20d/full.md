# ER-JEPA: Experience Replay Improves Joint-Embedding Predictive Learning in Language Models

Jingnan Pu¹, Zi-En Fan¹, Feng Lian¹,\*

1School of Automation Science and Technology, Xi’an Jiaotong University No. 28, West Xianning Road, Xi'an, Shaanxi 710049, China \*Corresponding author

Large language models (LLMs) excel at token-level generation but may learn undesirable abstract semantics and lack comprehensive perception. LLM-JEPA mitigates this by aligning different views of the same underlying knowledge via a joint-embedding predictive architecture (JEPA). However, strong alignment does not necessarily lead to accurate, stable predictions. To address this, we propose ER-JEPA, which adds an episodic replay path to LLM-JEPA. ER-JEPA stores training pairs in a memory. At each step, it stores and retrieves relevant data to provide additional supervision. This enables learning from both the current batch and stored training pairs, providing additional supervision for token prediction and representation alignment. Experiments across multiple datasets (NL-RX GSM8K, Spider, and NQ-Open) demonstrate that ER-JEPA consistently outperforms LLM-JEPA.

![](images/a729310c82ee81002a61aef6aabde132c6fe94bb9273e31a88766ac8dd75f104.jpg)

![](images/3c8e931fdceb0917bc5fcbf30966d6ccde0c388bb8b7a45a27d0898c8c50da61.jpg)  
Figure 1ER-JEPA outperforms LLM-JEPA across tasks.

## 1 Introduction

Large language models (LLMs) have achieved remarkable advances in fluent generation (Pillutla et al., 2021; Yu et al., 2023; Li et al., 2025; Gong et al., 2024; Li et al., 2024; Liu et al., 2024), few-shot learning (Brown et al., 2020; Wei et al., 2021; Wang et al., 2024), and broad downstream transfer (Rafailov et al., 2023; Zeng et al., 2022; Huan et al., 2025). These abilities require models to extract shared structure from large-scale data and convert it into dependable predictions on new problems. Autoregressive pre-training addresses both needs through next-token prediction (NTP) (see Figure 2(a)), which provides a strong learning signal and can lead to semantic representations under the right conditions (Jin & Rinard, 2024). However, a token-level objective may also favor local coherence over the representations needed for comprehensive perception and reasoning.

Joint-Embedding Predictive Architectures (JEPAs) shift prediction from the token level to the semantic level. Recent work on JEPAs has shown promising advances in representation learning and provable benefits for perception tasks (Assran et al., 2023; Bardes et al., 2024; Assran et al., 2025; Chen et al., 2026). Inspired by JEPAs, Huang et al. introduced LLM-JEPA (Huang et al., 2026), which adds an embedding space predictive loss to the standard language modeling loss (see Figure 2(b)). This loss aligns different views of the same underlying knowledge while preserving generative capability. It predicts both the next token and the target representation, improving the perception capability of LLMs.

Despite these advances, better representation alignment does not by itself guarantee reliable predictions. We observe that LLM-JEPA continues to make errors even after its alignment loss has converged (Figure 3). Some previously correct predictions also become incorrect later in training. Therefore, progress in representation learning is not automatically converted into predictions that are learned and retained. This leaves an open question: how can LLM-JEPA correct persistent errors and retain correct predictions during training?

We explore experience replay as a way to address this question by revisiting previously seen training examples and providing additional supervision as the model changes. Like standard fine-tuning, LLM-JEPA supervises an example only when it appears in the current mini-batch. Between two visits, many updates change the model, and the example is not checked again until the data schedule reaches it in the next epoch. Replay can reduce this gap by revisiting past examples during training. The replay mechanism is widely used in continual learning to preserve learned knowledge (Shi et al., 2024; Huang et al., 2024; Deng et al., 2025), and Complementary Learning Systems (CLS) theory gives it a similar role in the brain (Kumaran et al., 2016). In CLS, a fast system stores individual experiences and replays them to a slow system that extracts shared structure. From this perspective, LLM-JEPA is similar to the former but has no counterpart to the fast system. This motivates our research question: can experience replay improve the predictive performance of LLM-JEPA?

To answer it, we propose ER-JEPA, which adds an episodic replay (ER) pathway that stores past training examples and replays them during later updates (see Figure 2(c)). It enables the model to learn from useful past experiences beyond the current mini-batch, providing additional supervision signals and strengthening the perception capabilities of LLMs. The replay path is removed after training, so ER-JEPA has the same architecture and inference cost as LLM-JEPA.

In summary, the contributions of this paper are as follows:

• We propose ER-JEPA, which extends LLM-JEPA with an episodic replay path that revisits past training examples to provide additional supervision for token prediction and representation alignment. We show that learning from past samples continues to provide alignment updates and reinforces the knowledge important for perception and reasoning.

• We implement content-based, uniform, and hard selection within a shared replay path. We demonstrate that the replay branch itself, rather than any specific retrieval rule, is the central source of improvement.

• We evaluate ER-JEPA across multiple datasets. The results show that ER-JEPA consistently outperforms LLM-JEPA, demonstrating the benefit of episodic replay for abstraction and generalization.

## 2 Methodology

ER-JEPA extends LLM-JEPA with a replay path over stored training examples. We first introduce the joint training objective in Section 2.1. We then explain how replay provides additional supervision in Section 2.2.

## 2.1 ER-JEPA Overview and Objective

Our ER-JEPA builds on LLM-JEPA, retaining its token prediction and representation alignment objectives while adding an episodic replay path to learn from past training examples. The original LLM-JEPA objective

![](images/ae23c1ccd76374e3c8c8fc048282c2036080fc81a182cde4f0148a84c9ce2572.jpg)  
Figure 2 Comparison of baseline training, LLM-JEPA, and ER-JEPA. (a) The baseline learns through next token prediction. (b) LLM-JEPA adds prediction in the embedding space to learn abstract relations between inputs and targets. (c) ER-JEPA adds a replay path to LLM-JEPA. It stores past training examples and replays their tokens, providing additional supervision for both token prediction and representation alignment. Past experience thus continues to shape learning as the model changes, helping it correct errors and reinforce learned knowledge for more reliable predictions.

is

$$
\mathcal { L } _ { \mathrm { L L M - J E P A } } = \underbrace { \sum _ { \ell = 2 } ^ { L } \mathcal { L } _ { \mathrm { L L M } } ( \mathrm { T e x t } _ { 1 : \ell - 1 } , \mathrm { T e x t } _ { \ell } ) } _ { \mathrm { g e n e r a t i v e ~ c a p a b i l i t i e s ~ ( L L M ) } } + \lambda \underbrace { d ( \mathrm { P r e d } _ { \phi } ( \mathrm { E n c } _ { \theta } ( \mathrm { T e x t } ) ) , \ \mathrm { E n c } _ { \theta } ( \mathrm { C o d e } ) ) } _ { \mathrm { a b s t r a c t i o n ~ c a p a b i l i t i e s ~ ( J E P A ) } } .\tag{1}
$$

Here Text and Code denote the source and target views, and L is the source sequence length. Encθ and $\mathrm { P r e d } _ { \phi }$ denote the encoder and predictor. The function d measures cosine distance, and λ controls the JEPA loss weight.

For a fresh mini-batch $B _ { t }$ of size $B ,$ let $x _ { i } ^ { u }$ and $\boldsymbol { x } _ { i } ^ { a }$ denote the source and target tokens of example $i ,$ and $x _ { i } ^ { \mathrm { f u l l } } = x _ { i } ^ { u } \| x _ { i } ^ { a }$ . Define

$$
z _ { i } ^ { u } = \mathrm { E n c } _ { \theta } ( x _ { i } ^ { u } ) , \qquad p _ { i } = \mathrm { P r e d } _ { \phi } ( z _ { i } ^ { u } ) , \qquad z _ { i } ^ { a } = \mathrm { E n c } _ { \theta } ( x _ { i } ^ { a } ) .\tag{2}
$$

Let $\mathcal { T } _ { i } ^ { \mathrm { n e w } }$ contain the target-token positions in $x _ { i } ^ { \mathrm { f u l l } }$ . The loss on new data is

$$
\mathcal { L } _ { \mathrm { L L M - J E P A - n e w } } ( \mathcal { B } _ { t } ) = \frac { \gamma } { Z _ { t } ^ { \mathrm { n e w } } } \sum _ { i = 1 } ^ { B } \sum _ { \ell \in \mathcal { T } _ { i } ^ { \mathrm { n e w } } } \mathcal { L } _ { \mathrm { L L M } } \big ( x _ { i , < \ell } ^ { \mathrm { f u l l } } , x _ { i , \ell } ^ { \mathrm { f u l l } } \big ) + \frac { \lambda } { B } \sum _ { i = 1 } ^ { B } d ( p _ { i } , z _ { i } ^ { a } ) ,\tag{3}
$$

where $\begin{array} { r } { Z _ { t } ^ { \mathrm { n e w } } = \sum _ { i = 1 } ^ { B } | T _ { i } ^ { \mathrm { n e w } } | } \end{array}$ and $\gamma = 1$ is the language-modeling loss weight.

The full ER-JEPA objective adds a replay term,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E R - J E P A } } ( \mathcal { B } _ { t } ) = \mathcal { L } _ { \mathrm { L L M - J E P A - n e w } } ( \mathcal { B } _ { t } ) + \beta \mathcal { L } _ { \mathrm { L L M - J E P A - r e p l a y } } ( \mathcal { B } _ { t } ) , } \end{array}\tag{4}
$$

where $\beta \geq 0$ controls the strength of replay. $\mathcal { L } _ { \mathrm { { L L M - J E P A - r e p l a y } } }$ is defined in Section 2.2.

## 2.2 Replay-Based Training

Content-based, uniform, and hard replay are three implementations of the same replay path. They share the replay objective below. The selection rules for all three strategies are given in Appendix D.3. Unless otherwise specified, ER-JEPA uses content-based replay.

At each step, a replay strategy selects up to $R$ entries from the episodic memory, where R is the replay budget. Let $R _ { t }$ be the number of selected entries and $( \widetilde { \tau } _ { i } ^ { u } , \widetilde { \tau } _ { i } ^ { a } )$ be the stored source and target tokens for entry i. Memory storage and retrieval are described in Appendix D.1.

For each replay entry i, we concatenate the stored source and target tokens as $\widetilde { \tau } _ { i } ^ { \mathrm { f u l l } } = \widetilde { \tau } _ { i } ^ { u } \| \widetilde { \tau } _ { i } ^ { a }$ . The current model recomputes the source and target representations,

$$
z _ { i } ^ { u , \mathrm { r e p } } = \mathrm { E n c } _ { \theta } \big ( \widetilde { \tau } _ { i } ^ { u } \big ) , \qquad z _ { i } ^ { a , \mathrm { r e p } } = \mathrm { E n c } _ { \theta } \big ( \widetilde { \tau } _ { i } ^ { a } \big ) , \qquad p _ { i } ^ { \mathrm { r e p } } = \mathrm { P r e d } _ { \phi } \big ( z _ { i } ^ { u , \mathrm { r e p } } \big ) .\tag{5}
$$

Let ${ \mathcal { T } } _ { i } ^ { \mathrm { r e p } }$ contain the target-token positions in $\widetilde { \tau } _ { i } ^ { \mathrm { f u l l } }$ . For $R _ { t } > 0$ , the replay loss is

$$
\mathcal { L } _ { \mathrm { L L M - J E P A - r e p l a y } } ( \mathcal { B } _ { t } ) = \frac { \gamma } { Z _ { t } ^ { \mathrm { r e p } } } \sum _ { i = 1 } ^ { R _ { t } } \sum _ { \ell \in \mathcal { T } _ { i } ^ { \mathrm { r e p } } } \mathcal { L } _ { \mathrm { L L M } } \big ( \widetilde { \tau } _ { i , < \ell } ^ { \mathrm { f u l l } } , \widetilde { \tau } _ { i , \ell } ^ { \mathrm { f u l l } } \big ) + \frac { \lambda } { R _ { t } } \sum _ { i = 1 } ^ { R _ { t } } d \big ( p _ { i } ^ { \mathrm { r e p } } , z _ { i } ^ { a , \mathrm { r e p } } \big ) ,\tag{6}
$$

where $\begin{array} { r } { Z _ { t } ^ { \mathrm { r e p } } = \sum _ { i = 1 } ^ { R _ { t } } | T _ { i } ^ { \mathrm { r e p } } | } \end{array}$ . The loss is zero when $R _ { t } = 0$ . The normalization matches the new-data loss in Eq. (3).

The current and replay losses jointly update the shared model through Eq. (4). Replay-time JEPA errors refresh the selected memory scores, and replay counts are incremented as specified in Appendix D.2. At inference time, the episodic pathway is removed.

(a) Training Alignment Loss  
![](images/e3c95b1696de0fbca274caf3b62cbf9ecd9ca0ee25869ed0fa73be8e83bb46ab.jpg)

(b) Alignment vs. Correctness  
![](images/ad84ca298aebc2e32b89bba99d26169f91442edb12b22d8f40f6ba6ffcb4b53c.jpg)

(c) Prediction History  
![](images/09d59b8c5c6dc141893711e5c580ef04be2134c9c5727cab113ab1d7192afcb0.jpg)  
(e) Alignment vs. Correctness  
(f) Prediction History

(d) Training Alignment Loss  
![](images/789eb11f9ce65f7646e707fb6180c07fe8b71ebf37d7ba468bac681d33467297.jpg)

![](images/3f27b0fd596125ba921e05f18e9c3c4420b2a1d0eb86a523b37a6d855f55ac88.jpg)

![](images/224e78353592b1753bd7817a06f7c0b4e1150d5f962733dfd1353492017a4510.jpg)  
Figure 3LLM-JEPA can achieve strong alignment yet still make errors and lose previously correct predictions. For LLM-JEPA, the JEPA loss converges in (a). Yet in (b) and (c), errors remain even when cosine similarities between inputs and targets are close to one. Some correct predictions also become incorrect as training continues. For ER-JEPA, the JEPA loss also converges in (d). In (e) and (f), more predictions are correct despite slightly lower cosine similarities. At each checkpoint, ER-JEPA consistently has more correct answers than LLM-JEPA. These results suggest that replay supervision helps resolve errors that alignment alone leaves unresolved. This figure uses Llama-3.2-1B on 2000 SYNTH test examples with seed 82. Heatmap rows are sorted independently within each method by final-checkpoint correctness and prediction history.

## 3 Experiments: Episodic Replay Improves LLM-JEPA

This section evaluates whether episodic replay improves JEPA-based LLM training. After describing the experimental setup in Section 3.1, we report results across tasks and compare accuracy at matched training

compute in Section 3.2. We then examine the role of historical replay in Section 3.3, analyze prediction quality across regex structures and lengths and report resource costs in Section 3.4. Further ablations are provided in Appendix E.

## 3.1 Experimental Setup

We evaluate ER-JEPA under the same task families used in the LLM-JEPA protocol. Experiments are conducted on Llama-3.2-1B-Instruct (Grattafiori et al., 2024). We use five datasets with naturally paired input-target structures: NL-RX-SYNTH and NL-RX-TURK (Locascio et al., 2016) for generating regular expressions from descriptions in natural language, GSM8K (Cobbe et al., 2021) for mathematical reasoning, Spider (Yu et al., 2018) for text-to-SQL generation, and NQ-Open (Lee et al., 2019) for open-domain question answering. Following the evaluation protocol of LLM-JEPA, we report exact match accuracy of the generated regular expression for NL-RX-SYNTH and NL-RX-TURK, exact match accuracy of the final answer for GSM8K and NQ-Open, and execution accuracy for Spider. Detailed experimental settings are provided in Appendix A.

## 3.2 ER-JEPA Outperforms LLM-JEPA

Episodic replay improves LLM-JEPA across tasks. We first test whether replay provides benefits beyond representation alignment alone. We fine-tune Llama-3.2-1B-Instruct with different training objectives on all five datasets described above. As shown in Figure 1 (left), ER-JEPA achieves higher performance than LLM-JEPA on all five datasets.

The consistent gains show that the benefit of ER-JEPA is not limited to one type of task. The method improves both structured generation tasks, such as regular expression and SQL generation, and more general generation tasks, such as mathematical reasoning and open-domain question answering. These results suggest that episodic replay improves the use of training data in LLM-JEPA. By storing separated traces of past examples and retrieving relevant traces for the current context, ER-JEPA allows the model to reuse useful past experiences during optimization. This mechanism provides additional training signals beyond the current mini-batch and leads to more effective fine-tuning.

ER-JEPA achieves higher accuracy at the same training compute. To test whether the gains persist at the same training compute, we evaluate all methods at the same budgets from 40.05 to 240.31 PFLOPs on SYNTH with Llama-3.2-1B. We also evaluate three replay policies, including content, uniform, and hard replay, to test whether the benefit depends on the replay selection rule. The three policies share the same replay path and differ only in how stored examples are selected.

Figure 4(a) shows that all three replay policies outperform LLM-JEPA at every evaluated compute budget. At 240.31 PFLOPs, content replay reaches $8 3 . 6 5 \pm 1 . 4 8 \%$ , while LLM-JEPA reaches $6 8 . 7 5 \pm 4 . 9 2 \%$ , a gain of 14.90 percentage points. The normalized AUC also increases from $5 6 . 6 1 \pm 3 . 4 1 \%$ to $7 3 . 7 3 \pm 1 . 1 5 \%$ . The gains across all three rules show that the benefit is not limited to content-based selection. At matched training compute, ER-JEPA achieves higher mean accuracy than LLM-JEPA on SYNTH with all three replay policies.

Replay reduces accuracy drops during training. The mean curves in Figure 4(a) rise at every budget for all methods, but individual runs behave differently. Baseline and LLM-JEPA often lose accuracy that they reached earlier in training. These drops happen at different budgets in different seeds, so averaging hides them. Figure 4(b) illustrates this pattern for seed 84. Baseline loses accuracy after an initial increase, and LLM-JEPA shows a decline at a later checkpoint. ER-JEPA shows strong resistance to overfitting under all three replay policies. These results suggest that the replay path mitigates overfitting and helps retain the accuracy gained earlier in training. Appendix Figure 13 shows the other four seeds.

## 3.3 Understanding the Role of Historical Replay

To analyze these gains, we first compare ER-JEPA with token-matched LLM-JEPA to assess whether additional token exposure alone can reproduce the improvement. We then compare historical replay with a control that applies the same replay loss to current-batch examples, assessing the additional benefit of revisiting past samples. We also examine which errors replay corrects and whether it keeps correct predictions. Appendix B analyzes a single replay update.

![](images/812854aa839ca96ea931f774da76949f75602dd675c3d685f352d2a0a0426ec0.jpg)  
(a) Five seeds.

![](images/00d0364b88bc8a6c90a9832bdf0c12a761a8ad4f717e23c2780583923c0c634f.jpg)  
(b) Seed = 84.  
Figure 4 ER-JEPA improves accuracy at matched training compute and reduces overfitting. (a) All three replay policies outperform LLM-JEPA at every evaluated budget. These gains support the benefit of introducing the replay path. (b) Baseline loses accuracy after an early increase. LLM-JEPA delays this decline but still loses accuracy later. Content and uniform replay maintain their gains, while hard replay shows only a small final decline. These trends suggest that replay mitigates overfitting. They complement the mean curves in Figure 1 (right). Results use Llama-3.2-1B on SYNTH at six compute budgets from 40.05 to 240.31 PFLOPs.

Content replay achieves higher mean accuracy than token-matched LLM-JEPA. We first test whether extra token exposure can explain the gain. The token-matched control uses the token budget of content replay and retains the LLM-JEPA objective. Figure 5 shows that content replay reaches 83.65 ± 1.48%, compared with 62.25 ± 20.80% for token-matched LLM-JEPA. These results indicate that extra token exposure alone does not reproduce the gain.

Historical replay achieves higher mean accuracy than the current-batch control. We next examine whether historical samples provide value beyond an additional loss on the current batch. The current-batch control applies the replay loss to current-batch examples instead of stored examples. Figure 6 shows that all three historical replay policies have higher mean accuracy than this control. Uniform replay reaches 85.04 ± 0.54%, which is 3.80 percentage points (pp) above the control, and it is higher in all five seeds. Content replay reaches 83.65% (+2.41 pp), and hard replay reaches 83.41% (+2.17 pp). These results are consistent with a benefit from reusing historical samples.

ER-JEPA achieves higher correction rates on examples mispredicted by the baseline. As discussed in the Introduction, a model can align well with its target representation and still predict the wrong answer, sometimes even losing ground it had already covered. This raises a natural question: when Baseline errs, can LLM-JEPA and ER-JEPA fixits mistakes? To find out, we test both models on the SYNTH inputs where Baseline fails at the final matched compute point, checking whether each model recovers the exact target expression. We sort these failures by length relative to the target, including over-generation (extra regex tokens), under-generation (missing tokens), and same-length mismatch (correct length, wrong expression). Table 1 shows that ER-JEPA achieves a higher mean correction rate in all three categories. The largest gain occurs for over-generation, where the rate rises from 53.07% with LLM-JEPA to 84.51% with ER-JEPA. Complete correction counts and rates for each seed are reported in Appendix Table 3. Appendix Figure 10 further compares the two methods on inputs that LLM-JEPA predicts incorrectly.

To further examine whether replay helps correct errors and retain correct predictions during training, we track predictions on the same test inputs across successive checkpoints. Figure 3 shows more correct predictions for ER-JEPA in (f) than for LLM-JEPA in (c) at every evaluated checkpoint. Figure 7(b) shows a higher mean correction rate for ER-JEPA than for LLM-JEPA. In Figure 7(c), the mean rate of correct predictions becoming incorrect decreases from 8.85% with LLM-JEPA to 6.65% with ER-JEPA. These results suggest that replay supports both error correction and the retention of correct predictions. Example inputs, targets, and predictions from the three methods are shown in Appendix Figure 11.

![](images/3c14ec9a9926b207417122fee4ea23ab6fe186429e82499a4188cb8e0cbd5f0a.jpg)  
Figure 5 ER-JEPA achieves higher mean accuracy with replay under matched token exposure. Results use Llama-3.2-1B on SYNTH over five seeds.

![](images/6d3c41349f303c2f7b02e2a0a771bc4e943d03be88fbf08b3463f0148860d198.jpg)  
Figure 6 Historical samples provide additional value. Results use Llama-3.2-1B on SYNTH over five seeds.

Table 1 ER-JEPA achieves higher mean correction rates across all three Baseline error categories. Results use Llama-3.2-1B on SYNTH at 240.309 PFLOPs. Values are means ± one sample standard deviation over five seeds. Correction rates are computed separately for each seed before averaging.
<table><tr><td>Baseline error category Baseline errors Method</td><td></td><td></td><td></td><td>Corrected Correction rate (%)</td></tr><tr><td>Over-generation</td><td> $7 1 8 . 4 \pm 1 1 2 . 5$ </td><td>LLM-JEPA ER-JEPA</td><td> $3 7 8 . 8 \pm 6 4 . 6$ </td><td> $5 3 . 0 7 \pm 7 . 9 2$ </td></tr><tr><td>Under-generation</td><td></td><td>LLM-JEPA</td><td> $6 1 0 . 8 \pm 1 2 6 . 1$   $0 . 6 \pm 0 . 5$ </td><td> $8 4 . 5 1 \pm 4 . 7 0$   $2 4 . 0 0 \pm 2 5 . 1 0$ </td></tr><tr><td></td><td> $2 . 6 \pm 1 . 5$ </td><td>ER-JEPA</td><td> $1 . 2 \pm 0 . 8$ </td><td> $5 4 . 6 7 \pm 4 4 . 0 7$ </td></tr><tr><td>Same-length</td><td> $1 6 8 . 4 \pm 1 0 . 9$ </td><td>LLM-JEPA</td><td> $3 4 . 8 \pm 5 . 8$ </td><td> $2 0 . 5 8 \pm 2 . 3 5$ </td></tr><tr><td>mismatch</td><td></td><td>ER-JEPA</td><td> $4 2 . 4 \pm 7 . 0$ </td><td> $2 5 . 0 7 \pm 2 . 6 2$ </td></tr></table>

## 3.4 Fine-grained Analysis of Prediction Quality

We next examine where the accuracy gains occur within the SYNTH test set, focusing on the structure and length of the target regular expressions.

Accuracy gains span multiple forms of symbolic composition. To examine how these gains relate to symbolic structure, we analyze the SYNTH test set using six structural subsets defined by the target regular expressions, including complement, intersection, alternation, quantifiers, boundaries and anchors, and character classes. Detailed definitions and classification rules are provided in Appendix Table 2. Figure 8 shows that ER-JEPA achieves higher mean exact match accuracy than LLM-JEPA in all six subsets. The largest gains occur for expressions containing quantifiers, intersections, and character classes, with improvements of 16.90, 16.07, and 15.42 percentage points, respectively. These structures encode repetition, conjunction, and restrictions on allowed characters, all of which must be preserved when translating a natural language description into a regular expression. The gains therefore span several forms of constraint composition. This pattern is consistent with replay reinforcing the mapping from linguistic constraints to their symbolic implementation.

Replay reduces sensitivity to expression length. We next examine whether this advantage persists for longer expressions. Target regex length serves as a proxy for complexity. We divide the SYNTH test set into four groups: ≤ 10, 11–15, 16–20, and > 20 regex tokens. Figure 9 shows that Regular is highly sensitive to expression length. Its mean accuracy drops from 83.64% in the shortest group to 33.53% in the longest group, a decrease of 50.11 percentage points. LLM-JEPA reduces this gap, but its accuracy still falls from 85.15% to 56.76%. ER-JEPA achieves the highest accuracy in all four groups. Its accuracy decreases from 89.80% to 78.53%, a drop of 11.27 percentage points, compared with 28.39 for LLM-JEPA. These results suggest that episodic replay helps the model generate longer symbolic expressions more reliably.

![](images/83954bdcaf2756d93b52809b75e472213acf5e826d338d4fe00708e714573aa7.jpg)

![](images/74d6e2b4768b0032394d14033d7846a48a8a1249fb197ee103cd01d9c4ecf1a8.jpg)

(c) Correct Answers Lost  
![](images/904d5d7a2094479e09a0033630d4d027655097cd26079d0df89291a78176ed8f.jpg)  
Figure 7 ER-JEPA achieves higher accuracy and corrects more errors during training. (a) Test accuracy at comparable training compute for SFT, LLM-JEPA, and ER-JEPA. (b) The fraction of wrong predictions that become correct at the next checkpoint. (c) The fraction of correct predictions that become wrong at the next checkpoint. Compared with LLM-JEPA, ER-JEPA improves error correction across all five seeds, while the reduction in lost correct predictions is smaller and varies across seeds. Results use Llama-3.2-1B on the same 2,000 SYNTH test examples

![](images/6a3fd8c81195bbd71f8fa3179756548acd1abf16b6c28e26660f07cec011c620.jpg)

Figure 8 ER-JEPA achieves higher prediction accuracy across all six evaluated regex structures with episodic replay. Compared with LLM-JEPA, ER-JEPA improves accuracy by 16.90, 16.07, and 15.42 percentage points for expressions containing quantifiers, intersections, and character classes, respectively.  
![](images/4d02d22e8fa9b3a5ab0653bce82145a7375b381d309071d1da2000b9c8fff505.jpg)  
Figure 9 ER-JEPA maintains higher accuracy on longer regular expressions with episodic replay. ER-JEPA leads in all four length groups and shows a smaller decline from the shortest to the longest group than Regular and LLM-JEPA. Lines show mean exact match accuracy over five seeds, and shaded bands indicate one standard deviation.

Training gains come without additional inference cost. At inference, ER-JEPA uses the same generation procedure as LLM-JEPA. We remove the episodic memory and replay path after training. Following LLM-JEPA, we use only the fine-tuned language model for inference. Replay therefore adds no overhead to inference time and requires no additional computation or memory during inference.

Ablation studies on memory addressing, memory capacity and JEPA objective hyperparameters are provided in Appendix E.

## 4 Related Work

Joint-Embedding Predictive Architectures (JEPAs) learn representations by predicting target embeddings from context. LeCun (2022) proposed JEPA as a framework for learning world models that predict abstract states without reconstructing every input detail. A related approach, data2vec (Baevski et al., 2022), predicts contextualized representations from masked inputs using the same learning method across speech, vision, and language.

In vision, I-JEPA (Assran et al., 2023) predicts the representations of target image regions from a context region, learning semantic features without relying on hand-crafted augmentations. V-JEPA (Bardes et al., 2024) extends latent prediction to video and learns representations that support downstream tasks with a frozen encoder. Image World Models (Garrido et al., 2024) broaden the prediction task beyond masked regions to include the effects of photometric transformations. To better understand latent prediction, Littwin et al. (2024) analyze its learning dynamics in deep linear models. Their analysis identifies an implicit bias toward features with high regression coefficients.

Recent work also applies JEPA to language and vision-language learning. VL-JEPA (Chen et al., 2026) predicts continuous text embeddings for vision-language tasks and uses a decoder when text output is needed For language models, LLM-JEPA (Huang et al., 2026) combines next-token prediction with a JEPA loss over paired textual views. This joint objective encourages semantic structure in the learned representations while preserving generative capabilities. Our work builds on LLM-JEPA by adding episodic replay, which reuses past training examples to provide additional supervision for token prediction and representation alignment.

## 5 Conclusion

We propose ER-JEPA, which adds an episodic replay path to LLM-JEPA. The replay path revisits stored training pairs with the current model and provides additional learning signals. At matched training compute, ER-JEPA outperforms LLM-JEPA with all three selection rules. Replay also corrects more errors during training and avoids most large accuracy drops. These results show that revisiting past examples is a simple and effective way to improve LLM-JEPA.

We view the current implementation of ER-JEPA as a first step toward introducing episodic replay into the JEPA framework, and future work could further refine and extend this direction. In particular, since the overall training objective is formulated as a linear combination of the new data JEPA loss and the replay loss, the relative weighting between different loss terms, such as the replay weight β, currently needs to be selected through grid search. This tuning process introduces additional computational cost and may motivate future work on adaptive weighting strategies. Moreover, ER-JEPA inherits the need of LLM-JEPA for datasets with paired views, such as text and code. We hope future work will explore more efficient and general data augmentation strategies for constructing informative paired views, thereby broadening the application regime of ER-JEPA.

## References

[1] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15619–15629, 2023.

[2] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[3] Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., and Auli, M. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International conference on machine learning, pp. 1298–1312. PMLR, 2022.

[4] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

[5] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

[6] Chen, D., Shukor, M., Moutakanni, T., Chung, W., Yu, L., Kasarla, T., Bolourchi, A., LeCun, Y., and Fung, P. VL-JEPA: Joint embedding predictive architecture for vision-language. In The Fourteenth International Conference on Learning Representations, 2026.

[7] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Schulman, J., Hilton, J., Knight, M., Weller, A., Amodei, D., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[8] Deng, J., Wu, Q., Ju, P., Lin, S., Liang, Y., and Shroff, N. Unlocking the power of rehearsal in continual learning: A theoretical perspective, 2025. URL https://arxiv.org/abs/2506.00205.

[9] Garrido, Q., Assran, M., Ballas, N., Bardes, A., Najman, L., and LeCun, Y. Learning and leveraging world models in visual representation learning. arXiv preprint arXiv:2403.00504, 2024.

[10] Gong, L., Wang, S., Elhoushi, M., and Cheung, A. Evaluation of llms on syntax-aware code fill-in-the-middle tasks. arXiv preprint arXiv:2403.04814, 2024.

[11] Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[12] Huan, M. Z., Li, Y., Zheng, T., Xu, X., Kim, S., Du, M., Poovendran, R., Neubig, G., and Yue, X. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. ArXiv, abs/2507.00432, 2025.

[13] Huang, H., LeCun, Y., and Balestriero, R. LLM-JEPA: Large language models meet joint embedding predictive architectures. In The Fourteenth International Conference on Learning Representations, 2026.

[14] Huang, J., Cui, L., Wang, A., Yang, C., Liao, X., Song, L., Yao, J., and Su, J. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal, 2024. URL https://arxiv.org/abs/2403.01244.

[15] Jin, C. and Rinard, M. Emergent representations of program semantics in language models trained on programs. In Forty-irst International Conference on Machine Learning, 2024.

[16] Kumaran, D., Hassabis, D., and McClelland, J. L. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20(7):512–534, 2016.

[17] LeCun, Y. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

[18] Lee, K., Chang, M.-W., and Toutanova, K. Latent retrieval for weakly supervised open domain question answering. In Korhonen, A., Traum, D., and Màrquez, L. (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 6086–6096, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1612.

[19] Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. ArXiv, abs/2406.11939, 2024.

[20] Li, X., Tu, H., Hui, M., Wang, Z., Zhao, B., Xiao, J., Ren, S., Mei, J., Liu, Q., Zheng, H., Zhou, Y., and Xie, C What if we recaption billions of web images with LLaMA-3? In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 35957–35976. PMLR, 13–19 Jul 2025.

[21] Littwin, E., Saremi, O., Advani, M., Thilak, V., Nakkiran, P., Huang, C., and Susskind, J. How jepa avoids noisy features: The implicit bias of deep linear self distillation networks. Advances in Neural Information Processing Systems, 37:91300–91336, 2024.

[22] Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[23] Locascio, N., Narasimhan, K., DeLeon, E., Kushman, N., and Barzilay, R. Neural generation of regular expressions from natural language with minimal domain knowledge. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1918–1923, 2016.

[24] Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., and Harchaoui, Z. Mauve: Measuring the gap between neural text and human text using divergence frontiers. Advances in Neural Information Processing Systems, 34:4816–4828, 2021.

[25] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[26] Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., Wang, Z., Ebrahimi, S., and Wang, H. Continual learning of large language models: A comprehensive survey, 2024. URL https://arxiv.org/abs/2404.16789.

[27] Wang, S., Chen, Z., Shi, C., Shen, C., and Li, J. Mixture of demonstrations for in-context learning. Advances in Neural Information Processing Systems, 37:88091–88116, 2024.

[28] Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

[29] Yu, L., Simig, D., Flaherty, C., Aghajanyan, A., Zettlemoyer, L., and Lewis, M. Megabyte: Predicting million-byte sequences with multiscale transformers. Advances in Neural Information Processing Systems, 36:78808–78823, 2023.

[30] Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018.

[31] Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022.

## A Detailed Experimental Setup

The experimental design follows the LLM-JEPA protocol. This design allows us to isolate the effect of the proposed episodic replay pathway, rather than introducing confounding changes in data, model scale, or training configuration. All experiments are conducted under the same LoRA fine-tuning setting with rank 512. Under this setting, we compare three fine-tuning objectives: the standard supervised fine-tuning objective, the LLM-JEPA objective, and the proposed ER-JEPA objective.

For each combination of model and dataset, we use a learning rate of 1e-5 and train for 6 epochs. We adopt this LoRA configuration because the LLM-JEPA study found that it outperforms fine-tuning all model parameters under the same setting. The standard supervised objective, LLM-JEPA, and ER-JEPA all share this same configuration, learning rate, and training schedule. This helps isolate the effects of the training objective and the episodic-replay extension on performance.

All experiments are repeated with five fixed random seeds, {82, 23, 37, 84, 4}. We report the mean performance across seeds. Statistical significance is assessed using paired one-tailed t-tests over seed-matched runs. All training runs are conducted on a server equipped with four NVIDIA A800 GPUs.

## B How Historical Replay Can Provide Additional Correction Directions

LLM-JEPA combines token generation with representation alignment, but both losses directly supervise only the current batch. We give a conditional explanation of how replay changes one gradient-descent step. Replay can supply a descent direction for the selected historical loss that current-batch gradients do not provide.

The local limitation of the LLM-JEPA objective. Let $\omega$ collect the trainable LoRA coordinates, counting shared parameters once. For a nonempty current batch B, let $\tau _ { B }$ contain its supervised-token positions and $Z _ { B } = | { \mathcal { T } } _ { B } | > 0$ . Write $\pi _ { q } = \operatorname { s o f t m a x } ( f _ { q } ( \omega ) )$ for the token probabilities, $y _ { q }$ for the target token, and $e _ { i } = p _ { i } / \lVert p _ { i } \rVert _ { 2 } - z _ { i } ^ { a } / \lVert z _ { i } ^ { a } \rVert _ { 2 }$ for the alignment residual. Equation (3) gives the LLM-JEPA objective

$$
L _ { \mathcal { B } } ( \omega ) = \underbrace { \frac { \gamma } { Z _ { \mathcal { B } } } \sum _ { q \in \mathcal { T } _ { \mathcal { B } } } - \log \pi _ { q , y _ { q } } } _ { \mathrm { t o k e n ~ g e n e r a t i o n } } + \underbrace { \frac { \lambda } { 2 | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \Vert e _ { i } \Vert _ { 2 } ^ { 2 } } _ { \mathrm { r e p r e s e n t a t i o n ~ a l i g n m e n t } } ,\tag{7}
$$

where $\gamma , \lambda > 0$ are fixed. Fix a parameter state $\omega _ { 0 }$ . Hold the examples, sequences, masks, and forward-pass randomness fixed. Repeated occurrences use identical views and forward realizations. For all current and replay examples, assume finite logits, twice continuously differentiable maps, and representation norms above the normalization floor near $\omega _ { \mathrm { 0 } }$

At ωo, let $J _ { q } = \partial f _ { q } / \partial \omega$ and $A _ { i } = \partial e _ { i } / \partial \omega$ , differentiating both representation branches. The two losses supply the gradient

$$
g _ { \mathcal { B } } = \nabla L _ { \mathcal { B } } ( \omega _ { 0 } ) = \frac { \gamma } { Z _ { \mathcal { B } } } \sum _ { q \in \mathcal { T } _ { \mathcal { B } } } J _ { q } ^ { \top } ( \pi _ { q } - \mathbf { e } _ { y _ { q } } ) + \frac { \lambda } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } A _ { i } ^ { \top } e _ { i } ,\tag{8}
$$

where ${ \mathbf { e } } _ { y _ { q } }$ is the target-token basis vector. Neither term provides a component along directions in

$$
{ \mathcal { K } } = \{ v : \ J _ { q } v \in \operatorname { s p a n } \{ { \bf 1 } \} \ \forall q \in { \mathcal { T } } _ { \mathcal { B } } , \quad A _ { i } v = 0 \ \forall i \in \mathcal { B } \} .\tag{9}
$$

These directions preserve current token probabilities and alignment residuals to first order. They may still change losses on historical examples. Let P be the orthogonal projector onto $\kappa$ The subspace K may contain only the zero vector.

ER-JEPA adds $\beta L _ { \mathcal { R } }$ , with fixed $\beta > 0$ , where $L _ { \mathcal { R } }$ applies the same joint objective to a nonempty replay multiset with supervised tokens, as in Eq. (6). The analysis also allows replay to overlap B. Occurrences are counted separately and representations are recomputed. Hold the selected multiset R fixed throughout the comparison. Write $g _ { \mathcal { R } } = \nabla L _ { \mathcal { R } } ( \omega _ { 0 } )$

Proposition B.1 (Conditional local effect of historical replay). At the fxed state above, $P g _ { B } = 0$ . The same holds for a gradient obtained by repeating current examples or reweighting them with fixed weights. Define $d = - P g _ { \mathcal { R } } . \ I f P g _ { \mathcal { R } } \neq 0$ , then

$$
\begin{array} { r } { g _ { \mathcal { B } } ^ { \top } d = 0 , \qquad g _ { \mathcal { R } } ^ { \top } d = - \| P g _ { \mathcal { R } } \| _ { 2 } ^ { 2 } < 0 . } \end{array}\tag{10}
$$

For the gradient-descent updates ωc $= \omega _ { 0 } - \eta g _ { B }$ and $\omega _ { \mathrm { E } } = \omega _ { 0 } - \eta ( g _ { B } + \beta g _ { \mathcal R } )$ , we have

$$
P ( \omega _ { \mathrm { E } } - \omega _ { 0 } ) = \eta \beta d , \qquad P ( \omega _ { \mathrm { C } } - \omega _ { 0 } ) = 0 .\tag{11}
$$

Their replay losses satisfy

$$
L _ { \mathcal { R } } ( \omega _ { \mathrm { E } } ) - L _ { \mathcal { R } } ( \omega _ { \mathrm { C } } ) = - \eta \beta \left( \| P g _ { \mathcal { R } } \| _ { 2 } ^ { 2 } + \| ( I - P ) g _ { \mathcal { R } } \| _ { 2 } ^ { 2 } \right) + O ( \eta ^ { 2 } )\tag{12}
$$

as $\eta  0 . \ I f g _ { \mathcal R } \neq 0$ , then $L _ { \mathcal { R } } ( \omega _ { \mathrm { E } } ) < L _ { \mathcal { R } } ( \omega _ { \mathrm { C } } )$ for all sufficiently small $\eta > 0$ . This loss comparison does not require $P g _ { \mathcal R } \neq 0$

Proof. Let $v \in \kappa$ . For each supervised token, write $J _ { q } v = c _ { q } { \bf 1 }$ . Then

$$
( J _ { q } \boldsymbol { v } ) ^ { \top } ( \pi _ { q } - \mathbf { e } _ { y _ { q } } ) = c _ { q } ( 1 - 1 ) = 0 , \qquad ( A _ { i } \boldsymbol { v } ) ^ { \top } e _ { i } = 0 .
$$

Thus every token and alignment contribution is orthogonal to $\kappa .$ Equation (8) gives $P g _ { B } = 0$ . Repeating these contributions or changing their fixed weights preserves this property.

Since $P = P ^ { \top } = P ^ { 2 }$ , we have $g _ { B } ^ { \top } d = 0$ and $\begin{array} { r } { g _ { \mathcal { R } } ^ { \top } d = - g _ { \mathcal { R } } ^ { \top } P g _ { \mathcal { R } } = - \| P g _ { \mathcal { R } } \| _ { 2 } ^ { 2 } } \end{array}$ . This proves $\operatorname { E q . } \left( 1 0 \right)$ when $P g _ { \mathcal R } \neq 0$ Applying P to the two updates gives Eq. (11).

Taylor expansion at $\omega _ { 0 }$ gives

$$
\begin{array} { r l } & { L _ { \mathcal { R } } ( \omega _ { \mathrm { C } } ) = L _ { \mathcal { R } } ( \omega _ { 0 } ) - \eta g _ { \mathcal { R } } ^ { \top } g _ { \mathcal { B } } + O ( \eta ^ { 2 } ) , } \\ & { L _ { \mathcal { R } } ( \omega _ { \mathrm { E } } ) = L _ { \mathcal { R } } ( \omega _ { 0 } ) - \eta g _ { \mathcal { R } } ^ { \top } g _ { \mathcal { B } } - \eta \beta \| g _ { \mathcal { R } } \| _ { 2 } ^ { 2 } + O ( \eta ^ { 2 } ) . } \end{array}
$$

Subtracting cancels the shared term $- \eta g _ { \mathcal { R } } ^ { \top } g _ { \mathcal { B } }$ . The orthogonal decomposition $g _ { \mathcal { R } } = P g _ { \mathcal { R } } + ( I - P ) g _ { \mathcal { R } }$ then gives Eq. (12). The smoothness assumptions bound the absolute remainder by $C \eta ^ { 2 }$ for some fixed $C > 0$ and all sufficiently small η. If $g _ { \mathcal { R } } \neq 0$ , choosing η small enough that $C \eta < \beta \| g _ { \mathcal { R } } \| _ { 2 } ^ { 2 }$ makes the difference strictly negative.

The term $\| \boldsymbol { P } \boldsymbol { g } _ { \mathcal { R } } \| _ { 2 } ^ { 2 }$ is the contribution of the additional directions to the first-order loss difference. It is positive only when $P g _ { \mathcal R } \neq 0$ . The projector is an analytical device; ER-JEPA uses the full replay gradient. The loss comparison also applies to ordinary replay or any added smooth loss with a nonzero gradient. This analysis characterizes a conditional local effect on the replay loss; its implications for predictive performance are evaluated empirically.

ER-JEPA predictions on examples LLM-JEPA gets wrong  
![](images/d4111052fd5d3d4012837452fbe40cda424b669f19876dc315de5aa13b5141a4.jpg)  
Figure 10 ER-JEPA predictions on examples that LLM-JEPA predicts incorrectly. Rows indicate LLM-JEPA error categories, and columns indicate ER-JEPA prediction categories for the same input and seed. The row total n counts LLM-JEPA errors pooled across five seeds. Each cell shows its count divided by n, expressed as a percentage to one decimal place. Results use Llama-3.2-1B on 2,000 SYNTH test examples per seed at a matched training compute budget of 240.309 PFLOPs.

## C Additional Empirical Analyses

## C.1 Regex Structural Subsets

Classification follows the regex grammar. Escaped literal symbols and symbols inside character classes are not counted as separate operators. Each example is assigned to every applicable subset, so the subsets may overlap.

## C.2 Error Correction Across Seeds

Table 3 reports the individual results summarized in Table 1. For each seed, we identify Baseline errors among the same 2,000 SYNTH test inputs and evaluate both methods on those inputs. Syntax errors are excluded from the three length-based categories. A prediction counts as corrected only when it exactly matches the target. Each correction rate uses the Baseline error count in its row as the denominator. ER-JEPA uses content replay.

## C.3 Examples Selected by a Fixed Rule

Figure 11 illustrates Baseline errors corrected by ER-JEPA, grouped by the Baseline error taxonomy. The selection contains exactly two LLM-JEPA-correct cases and seven LLM-JEPA errors; all three methods are compared on the same input and seed for each example. The full evaluation, using 2,000 test inputs for each of five seeds, gives mean accuracies of 53.63%, 68.75%, and 83.65% for Baseline, LLM-JEPA, and ER-JEPA, respectively. Figure 10 reports ER-JEPA predictions on inputs that LLM-JEPA predicts incorrectly.

## C.4 Individual Seed Accuracy at Matched PFLOPs

Figure 13 complements the seed 84 results in Figure 4(b). Across all five seeds, Baseline has a local accuracy decline in four seeds, and LLM-JEPA has a decline in all five. Such declines occur less often with replay. These results suggest that replay mitigates overfitting across individual runs.

Model performance on regex generation -SYNTH
<table><tr><td colspan="6">(a) Over-generation</td></tr><tr><td></td><td>Example 1</td><td>Seed 37 | Test #864 Example 2</td><td></td><td>Seed 82 | Test #1507 Example 3</td><td>Seed 4 | Test #1376</td></tr><tr><td>Input</td><td>lines not containing a number, 4 or more times</td><td>lines ending with the string &#x27;dog&#x27; at least once or a lower-case letter</td><td></td><td>lines with the string &#x27;dog&#x27; or a capital letter before a number at least once</td></tr><tr><td>Ground truth</td><td>~(.*([0-9]){4,}.*)</td><td>(.*)(((dog)+)([a-z]))</td><td></td><td>((dog)|(([A-Z]).*([0-9]).*))+</td></tr><tr><td>Baseline</td><td>~(.*([0-9]){4,}.*)*</td><td>(.*)((dog)+)|([a-z]))(.*)</td><td>×</td><td>((dog)|([A-Z])).*(([0-9])+).*.* ×</td></tr><tr><td>LLM-JEPA</td><td>~(.*([0-9]){4,}.*)</td><td>(.*)((dog)+)([a-z]))(.*)</td><td>× ((dog)([A-Z])).*(([0-9])+).*</td><td>×</td></tr><tr><td>ER-JEPA</td><td>~(.*([0-9]){4,}.*)</td><td>(.*)(((dog)+)([a-z]))</td><td>√ ((dog)|([A-Z]).*([0-9]).*))+</td><td>√</td></tr><tr><td></td><td>Extra * admits forbidden digits.</td><td>Extra suffix removes the end constraint.</td><td></td><td>Digit requirement reaches both branches.</td></tr><tr><td colspan="5"></td></tr><tr><td colspan="5">(b) Under-generation Seed 82 | Test #1250 Example 2</td></tr><tr><td></td><td>Example 1</td><td>Seed 4  Test #1816 Example 3</td><td></td><td>Seed 82 | Test #1205</td></tr><tr><td>Input</td><td>lines ending with the string &#x27;dog&#x27; before a number or the string &#x27;truck&#x27;</td><td>lines starting with the string &#x27;dog&#x27; or containing only a character</td><td>lines with the string &#x27;dog&#x27; before ending with the string &#x27;truck&#x27;, 6 or more times</td><td></td></tr><tr><td>Ground truth</td><td>((.*)(dog)).*(([0-9])|(truck)).*</td><td>((dog)|(.))(.*)*)</td><td></td><td>(dog).*((.*)(truck){6,})).*</td></tr><tr><td>Baseline</td><td>(.*)((dog).*([0-9]).*)|(truck)</td><td>× ((dog)|()*)</td><td>×</td><td>(dog).*(.*)(truck)){6,}.*</td></tr><tr><td>LLM-JEPA</td><td>((.*)(dog)).*(([0-9])|(truck)).*</td><td>((dog)(.*))|(.)</td><td>×</td><td>(dog).*((.*)(truck)){6,}</td></tr><tr><td>ER-JEPA</td><td>((.*)(dog)).*(([0-9])(truck)).*</td><td>((dog)(.))(*))</td><td></td><td>(dog).*((.*)((truck){6,})).*</td></tr><tr><td>Alternation escapes the dog prefix.</td><td></td><td>Empty-string and suffix errors.</td><td></td><td>Repeat scope allows separated trucks.</td></tr><tr><td colspan="5">(c) Same-length mismatch</td></tr><tr><td colspan="5">Example 1 Seed 23 | Test #341 Example 2</td></tr><tr><td>Input</td><td></td><td>Seed 82 | Test #242 Example 3 lines not ending with the string &#x27;dog&#x27;</td><td>lines not having a capital letter or a</td><td>Seed 37 | Test #239</td></tr><tr><td></td><td>lines ending with a number, 2 or more times</td><td>or the string &#x27;truck&#x27;</td><td>vowel at least once</td></tr><tr><td>Ground truth</td><td>(.*)(([0-9]){2,})</td><td>~((.*)((dog)|(truck)))</td><td>~((([A-Z])|([AEIOUaeiou]))+)</td></tr><tr><td>Baseline</td><td>((.*)([0-9]){2,} ×</td><td>~((.*)(dog))|(truck)) X</td><td>((~([A-Z]))|([AEIOUaeiou]))+ ×</td></tr><tr><td>LLM-JEPA</td><td>((.*)([0-9]){2,}</td><td>~((.*)(dog))|(truck)) ×</td><td>(~([A-Z])|([AEIOUaeiou])+) ×</td></tr><tr><td>ER-JEPA</td><td>(.*)(([0-9]) {2,})</td><td>~((.*)((dog)|(truck))) √</td><td>~(([A-Z])|([AEIOUaeiou]))+) √</td></tr><tr><td></td><td>√ Digits need not be consecutive.</td><td>Negation misses the truck suffix.</td><td>Negation covers only one branch.</td></tr></table>

Evidence uses whole-string matching; "" is empty; + joins strings; "truck" \* 5 repeats truck five times.

Figure 11 Example predictions from Baseline, LLM-JEPA, and ER-JEPA on the SYNTH dataset.

NQ-OPEN prediction examples
<table><tr><td>Question</td><td>Reference</td><td>Baseline</td><td>LLM-JEPA</td><td>ER-JEPA</td></tr><tr><td>who did the jets play in super bowl 3</td><td>Baltimore Colts</td><td>Pittsburgh Steelers</td><td>Pittsburgh Steelers</td><td>Baltimore Colts</td></tr><tr><td>medical term for the cause of a disease</td><td>etiology</td><td>etiology</td><td>disease process</td><td>etiology</td></tr><tr><td>how many sisters does joey have in friends</td><td>seven</td><td>two</td><td>two</td><td>two</td></tr><tr><td>who played bat masterson in the tv show</td><td>Gene Barry</td><td>Gene Kelly</td><td>Gene Barry</td><td>Gene Barry</td></tr><tr><td>when did the war of the roses start</td><td>22 May 1455; 1455</td><td>1485</td><td>1485</td><td>1485</td></tr></table>

Figure 12 Example predictions from Baseline, LLM-JEPA, and ER-JEPA on the NQ-OPEN dataset. Correct and incorrect answers are shown in dark green and dark red, respectively. For each prediction, only the first semicolon-separated answer is displayed.

![](images/082b09071c9f7550ed3326f8678fafdc3f475584a7a02637121e74bc877e8bd6.jpg)

![](images/b93f05ae064ed171728dc204ba26543c81cd63cd0d912c4fea6c398a711c0591.jpg)

![](images/404e56575df6cef87e546fdbac37303736fe400713fdfe635d138575f89b54ba.jpg)

![](images/76c99667f2e0e5fa0e0d04294c1849dbe814247dad2c0a8cd4d70252ddce565f.jpg)  
Figure 13 Raw accuracy at six shared target PFLOPs for seeds 4, 23, 37, and 82. Each panel shows one seed. Seed 84 is shown in Figure 4(b). The panels do not show means or error bars.

Table 2Definitions of the six regex structural subsets. Each subset contains examples whose target regular expressions include the specified feature.
<table><tr><td>Subset</td><td>Classification rule</td><td>Meaning</td></tr><tr><td>Complement</td><td>Contains the operator ~.</td><td>Excludes strings matched by a subexpression.</td></tr><tr><td>Intersection</td><td>Contains the operator &amp;.</td><td>Requires multiple patterns to hold simultaneously.</td></tr><tr><td>Alternation</td><td>Contains the operator I.</td><td>Allows a choice between alternative patterns.</td></tr><tr><td>Quantifier</td><td>Contains *, +, ?, or a brace quantifier such as {m,n}.</td><td>Specifies repetition counts or optionality</td></tr><tr><td>Boundary / anchor</td><td>Contains \b, \B, ^, or $.</td><td>Constrains matching positions relative to word boundaries or string endpoints.</td></tr><tr><td>Character class</td><td>such as [a-z].</td><td>Contains a bracketed character class, Specifies a set of allowed characters.</td></tr></table>

## D ER-JEPA Implementation Details

## D.1 Episodic Memory and Retrieval

ER-JEPA maintains a fixed-capacity episodic memory $\mathcal { M } _ { t }$ Each active slot stores a source-target token pair, a sparse address, and memory-management metadata. We denote these entries by

$$
m _ { n } = \bigl ( s _ { n } , \tau _ { n } ^ { u } , \tau _ { n } ^ { a } , \sigma _ { n } , r _ { n } \bigr ) , \qquad n \in \mathcal { A } _ { t } ,\tag{13}
$$

where $\boldsymbol { A } _ { t }$ is the set of active slots, $s _ { n }$ is the address stored at insertion, $\sigma _ { n }$ is the running JEPA error, and $r _ { n }$ is the replay count. Examples in the current mini-batch are excluded from replay. Writing, replacement, and score updates follow Appendix D.2.

Content-based retrieval uses the current source representation $z _ { j } ^ { u } = \operatorname { E n c } _ { \theta } ( x _ { j } ^ { u } )$ as a cue. Let Φ be the frozen random projection with sparse Top-K selection defined in Appendix D.3.1. We normalize the cue and stored addresses as

$$
\hat { s } _ { j } ^ { \mathrm { c u e } } = \frac { \Phi ( z _ { j } ^ { u } ) } { \Vert \Phi ( z _ { j } ^ { u } ) \Vert _ { 2 } } , \qquad \hat { s } _ { n } = \frac { s _ { n } } { \Vert s _ { n } \Vert _ { 2 } } .\tag{14}
$$

For each cue, we retrieve up to κ slots with the highest cosine similarities,

$$
\begin{array} { r } { K _ { t , j } = \mathrm { T o p } _ { \mathrm { m i n } ( \kappa , N _ { t } ) } \left( \left\{ \left( n , \langle \hat { s } _ { j } ^ { \mathrm { c u e } } , \hat { s } _ { n } \rangle \right) \mid n \in \mathcal { A } _ { t } \right\} \right) , \qquad N _ { t } = | \mathcal { A } _ { t } | . } \end{array}\tag{15}
$$

Here Top returns slot indices in descending similarity order. We concatenate the lists in batch order, keep the first occurrence of each slot, and retain at most R entries, where R is the replay budget. If fewer than R distinct slots are retrieved, we replay only those slots. Let $\boldsymbol { S } _ { t } = \left( \boldsymbol { n } _ { t , 1 } , \ldots , \boldsymbol { n } _ { t , R _ { t } } \right)$ denote the selected slot list, where $R _ { t } = | S _ { t } | \leq R$ . The token pair for replay entry i is

$$
( \widetilde { \tau } _ { i } ^ { u } , \widetilde { \tau } _ { i } ^ { a } ) = ( \tau _ { n _ { t , i } } ^ { u } , \tau _ { n _ { t , i } } ^ { a } ) , \quad \quad i = 1 , \ldots , R _ { t } .\tag{16}
$$

The selection rules for all three replay strategies are given in Appendix D.3.

## D.2 Shared Episodic Memory

Each new slot is initialized with

$$
\sigma _ { n } = d ( p _ { n } , z _ { n } ^ { a } ) , \qquad r _ { n } = 0 ,\tag{17}
$$

Table 3 Correction counts and rates for each seed on the same Baseline errors. Both methods use the same inputs and seed within each row. Corrected subsets may overlap between methods. The counts and rates in Table 1 are summarized across these five seeds.
<table><tr><td>Seed</td><td>Baseline errors</td><td>LLM-JEPA corrected</td><td>Correction rate (%)</td><td>ER-JEPA corrected</td><td>Correction rate (%)</td></tr><tr><td colspan="6">(a) Over-generation</td></tr><tr><td>4</td><td>901</td><td>448</td><td>49.72</td><td>807</td><td>89.57</td></tr><tr><td>23</td><td>697</td><td>435</td><td>62.41</td><td>583</td><td>83.64</td></tr><tr><td>37</td><td>705</td><td>375</td><td>53.19</td><td>608</td><td>86.24</td></tr><tr><td>82</td><td>591</td><td>344</td><td>58.21</td><td>455</td><td>76.99</td></tr><tr><td>84</td><td>698</td><td>292</td><td>41.83</td><td>601</td><td>86.10</td></tr><tr><td colspan="6">(b) Under-generation</td></tr><tr><td>4</td><td>1</td><td>0</td><td>0.00</td><td>1</td><td>100.00</td></tr><tr><td>23</td><td>5</td><td>1</td><td>20.00</td><td>2</td><td>40.00</td></tr><tr><td>37</td><td>3</td><td>0</td><td>0.00</td><td>1</td><td>33.33</td></tr><tr><td>82</td><td>2</td><td>1</td><td>50.00</td><td>2</td><td>100.00</td></tr><tr><td>84</td><td>2</td><td>1</td><td>50.00</td><td>0</td><td>0.00</td></tr><tr><td colspan="6">(c) Same-length mismatch</td></tr><tr><td>4</td><td>161</td><td>29</td><td>18.01</td><td>36</td><td>22.36</td></tr><tr><td>23</td><td>156</td><td>31</td><td>19.87</td><td>38</td><td>24.36</td></tr><tr><td>37</td><td>173</td><td>41</td><td>23.70</td><td>46</td><td>26.59</td></tr><tr><td>82</td><td>184</td><td>41</td><td>22.28</td><td>53</td><td>28.80</td></tr><tr><td>84</td><td>168</td><td>32</td><td>19.05</td><td>39</td><td>23.21</td></tr></table>

where $p _ { n }$ is the predicted target representation and $z _ { n } ^ { a }$ is the target readout. When the store is full, the slot with the lowest replay-adjusted significance is evicted

$$
n ^ { \star } = \arg \operatorname* { m i n } _ { n \in \mathcal { A } _ { t } } \frac { \sigma _ { n } } { 1 + r _ { n } } .\tag{18}
$$

After a replay forward pass, the significance of each selected slot is updated as

$$
\sigma _ { n } \gets ( 1 - \eta ) \sigma _ { n } + \eta d ( p _ { n } ^ { \mathrm { r e p } } , z _ { n } ^ { a , \mathrm { r e p } } ) , \qquad r _ { n } \gets r _ { n } + 1 , \qquad \eta \in ( 0 , 1 ) ,\tag{19}
$$

where η is the update rate and $p _ { n } ^ { \mathrm { r e p } } , z _ { n } ^ { a , \mathrm { r e p } }$ are recomputed during replay. Unselected slots retain their scores.   
These rules are shared by all three selection policies.

## D.3 Replay Selection Policies

The content-based, uniform, and hard replay strategies differ only in how replay slots are selected. All use the memory rules in Appendix D.2 and the replay loss in Eq. (6). Let At be the active slots, $N _ { t } = | A _ { t } |$ , and R the replay budget. A policy $\pi \in$ {content, uniform, hard} returns an ordered slot list $S _ { t } ^ { \pi }$ , and $S _ { t } = S _ { t } ^ { \pi }$ . Repeated slots contribute once per occurrence. All policies return an empty selection when $N _ { t } = 0$

## D.3.1 Content Replay

Content replay retrieves stored examples using current source representations $z ^ { u }$ . Representations are mapped to sparse addresses with a frozen random projection and Top-K selection,

$$
\begin{array} { r l } & { \qquad W _ { \mathrm { P S } } \in \mathbb { R } ^ { S \times H } , \qquad [ W _ { \mathrm { P S } } ] _ { \ell k } \sim \mathcal { N } ( 0 , 1 / H ) , } \\ & { [ \Phi ( z ) ] _ { \ell } = \mathrm { s t o p } { - } \mathrm { g r a d } ( \mathbb { I } [ \ell \in \mathcal { T } _ { K } ( W _ { \mathrm { P S } } z ) ] ( W _ { \mathrm { P S } } z ) _ { \ell } ) . } \end{array}\tag{20}
$$

H and $S > H$ are the representation and address dimensions, and $\mathcal { T } _ { K } ( h )$ contains the indices of the K largest-magnitude entries of $h ,$ retaining their sign and magnitude. The projection is fixed throughout training.

For current example $j ,$ the cue is $\hat { s } _ { j } ^ { \mathrm { c u e } } = \Phi ( z _ { j } ^ { u } ) / \Vert \Phi ( z _ { j } ^ { u } ) \Vert _ { 2 }$ . Let $s _ { n }$ be the address stored at insertion and $\hat { s } _ { n } = s _ { n } / \lVert s _ { n } \rVert _ { 2 }$ . The min $( \kappa , N _ { t } )$ slots with the highest cosine similarity are retrieved,

$$
\begin{array} { r } { \mathcal { K } _ { t , j } = \mathrm { T o p } _ { \operatorname* { m i n } ( \kappa , N _ { t } ) } \left( \left\{ \left( n , \langle \hat { s } _ { j } ^ { \mathrm { c u e } } , \hat { s } _ { n } \rangle \right) \vert n \in \mathcal { A } _ { t } \right\} \right) , } \end{array}\tag{21}
$$

returned in descending score order. Lists from all examples in the batch are concatenated in batch order, deduplicated, and truncated to R entries,

$$
\mathcal { C } _ { t } = \mathrm { P r e f i x } _ { R } \big ( \mathrm { U n i q u e } \big ( \boldsymbol { K } _ { t , 1 } \big | \big | \cdots \big | \big | \boldsymbol { K } _ { t , | \mathcal { B } _ { t } | } \big ) \big ) , \qquad \mathcal { S } _ { t } ^ { \mathrm { c o n t e n t } } = \mathcal { C } _ { t } ,\tag{22}
$$

where Unique keeps the first occurrence of each slot and PrefixR keeps at most R entries. The default policy uses the resulting list without repetition. For controls with a fixed replay count, we instead use $S _ { t } ^ { \mathrm { c o n t e n t } } = \mathrm { F i l l } _ { R } ( \mathcal { C } _ { t } )$ , where $\operatorname { F i l l } _ { R }$ repeats a shorter nonempty list until it reaches R entries; an empty list remains empty. Cosine similarity is used only for retrieval and contributes no training loss.

## D.3.2 Uniform Replay

Uniform replay samples stored examples with equal probability, without using cue similarity or significance. When $R \leq N _ { t }$ R slots are sampled without replacement: for any subset $\mathcal { U } \subseteq \mathcal { A } _ { t }$ of size $R ,$

$$
\operatorname* { P r } \left( \operatorname { s e t } ( S _ { t } ^ { \mathrm { u n i f o r m } } ) = \mathcal { U } \mid \mathcal { M } _ { t } \right) = { \binom { N _ { t } } { R } } ^ { - 1 } , \qquad \operatorname* { P r } ( n \in S _ { t } ^ { \mathrm { u n i f o r m } } \mid \mathcal { M } _ { t } ) = \frac { R } { N _ { t } } .\tag{23}
$$

When $0 < N _ { t } < R$ R slots are drawn independently with replacement, each with probability $1 / N _ { t }$

Uniform replay estimates the mean JEPA error over active memory. Let $d _ { n } ( t )$ be the error of stored example n under the current model. For a nonempty memory and fixed parameters,

$$
\mathbb { E } \left[ \frac { 1 } { R } \sum _ { n \in S _ { t } ^ { \mathrm { u n i f o r m } } } d _ { n } ( t ) \Biggm | \mathcal { M } _ { t } , \theta \right] = \frac { 1 } { N _ { t } } \sum _ { n \in \mathcal { A } _ { t } } d _ { n } ( t ) .\tag{24}
$$

## D.3.3 Hard Replay

Hard replay selects the min $( R , N _ { t } )$ slots with the largest significance $\sigma _ { n }$ (Appendix D.2),

$$
\mathcal { H } _ { t } = \mathrm { T o p } _ { \operatorname* { m i n } ( R , N _ { t } ) } \left( \{ ( n , \sigma _ { n } ) \ | \ n \in \mathcal { A } _ { t } \} \right) , \qquad \mathcal { S } _ { t } ^ { \mathrm { h a r d } } = \mathrm { F i l l } _ { R } ( \mathcal { H } _ { t } ) ,\tag{25}
$$

returned in descending score order, with $\operatorname { F i l l } _ { R }$ as in Appendix D.3.1. A large JEPA error indicates poor representation alignment, not necessarily an incorrect generated answer.

## E Ablation Studies

## E.1 Sparse and Dense Memory Addressing

We compare content replay using sparse addresses and dense source representations. Sparse addresses are formed by a frozen random projection and Top-K selection. Both variants use cosine similarity, with all other settings fixed. Table 5 reports mean accuracies of 86.37% for sparse retrieval and 86.21% for dense retrieval. The two strategies achieve similar mean accuracy, with a difference of 0.16 percentage points.

## E.2 Hyperparameter Ablations

Memory capacity. Table 6 reports accuracy after four epochs, and Figure 14 shows the learning curves.   
Increasing memory capacity from 10 to $1 0 ^ { 4 }$ raises accuracy from 84.22% to 84.98%.

Objective hyperparameters. We also vary the JEPA loss weight λ and predictor depth $k ,$ following the LLM-JEPA design. Figure 15 reports the results.

Table4 Some regular expressions generated by Llama-3.2-1B-Instruct after fine-tuning with $\mathcal { L } _ { \mathrm { L L M } } , \mathcal { L } _ { \mathrm { L L M - J E P A } }$ and $\mathcal { L } _ { \mathrm { E R - J E P A } }$ losses. Color code: wrong, extra, missing
<table><tr><td>Model / target Regular expression</td></tr><tr><td>Input: lines not having the string &quot;dog&quot; followed by a number, 3 or more times Ground truth  $\sim ( ( \deg . ^ { * } [ 0 . 9 ] . ^ { * } ) \{ 3 , \} )$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { L L M } }$   $\sim ( ( \deg . ^ { * } [ 0 . 9 ] . ^ { * } ) \{ 3 , \} )$   $\mathcal { L } _ { \mathrm { L L M - J E P A } }$   $\sim ( ( \deg . ^ { * } [ 0 . 9 ] . ^ { * } ) \{ 3 , \} )$   $\mathcal { L } _ { \mathrm { E R - J E P A } }$   $\sim ( ( \deg . ^ { * } [ 0 . 9 ] . ^ { * } ) \{ 3 , \} )$ </td></tr><tr><td>Input: lines containing ending with a vowel, zero or more times Ground truth  $\cdot ^ { * } ( . ^ { * } ) ( ( [ \mathrm { A E I O U a e i o u } ] ) ^ { * } ) . ^ { * }$   $\mathcal { L } _ { \mathrm { L L M } }$   $\cdot ^ { * } ( . * ) ( ( [ \mathrm { A E I O U a e i o u } ] ) ^ { * } ) . ^ { * } \cdot ^ { * } . ^ { * }$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { L L M - J E P A } }$   $\cdot ^ { * } ( . \mathsf { * } ) ( ( [ \mathrm { A E I O U a e i o u } ] ) ^ { * } ) . ^ { * } \ . ^ { * }$   $\mathcal { L } _ { \mathrm { E R - J E P A } }$   $\cdot ^ { * } ( . ^ { * } ) ( ( [ \mathrm { A E I O U a e i o u } ] ) ^ { * } ) . ^ { * }$  Input: lines with a number or a character before a vowel</td></tr><tr><td>Ground truth  $( ( [ 0 - 9 ] ) | ( . ) ) . ^ { * } ( [ \mathrm { A E I O U a e i o u } ] ) . ^ { * }$   $\mathcal { L } _ { \mathrm { L L M } }$   $( ( [ 0 \mathrm { - } 9 ] ) | ( . ) ) . ^ { * } ( [ \mathrm { A E I O U a e i o u } ] ) . ^ { * } \ . ^ { * }$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { L L M - J E P A } }$   $( ( [ 0 - 9 ] ) | ( . ) ) . ^ { * } ( [ \mathrm { A E I O U a e i o u } ] ) . ^ { * }$   $\mathcal { L } _ { \mathrm { E R - J E P A } }$   $( ( [ 0 - 9 ] ) | ( . ) ) . ^ { * } ( [ \mathrm { A E I O U a e i o u } ] ) . ^ { * }$ </td></tr><tr><td>Input: lines ending with containing the string</td></tr><tr><td>Ground truth  $( ( . ^ { * } ) ( . ^ { * } \mathrm { d o g . } ^ { * } ) ) \{ 7 , \}$ </td></tr><tr><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { L L M } }$   $( ( . ^ { * } ) ( . ^ { * } ( \arg . ^ { * } ) ) \{ 7 , \} \cdot ^ { * } ) ^ { * }$ </td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { L L M - J E P A } }$   $( ( . ^ { * } ) \ ( \ ( . ^ { * } \mathrm { d o g . ^ { * } } ) ) \{ 7 , \} \ )$ </td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>_</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td> $( . ^ { * } ) \ ( \ ( . ^ { * } \mathrm { d o g . ^ { * } } ) 7 , ) \ )$ </td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { E R - J E P A } }$ </td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr></table>

Table 5 Sparse and dense memory addressing for content replay. All other training settings are fixed. Results report mean accuracy, standard deviation, minimum, and maximum over five seeds.
<table><tr><td>Method</td><td>Accuracy (%) ↑</td><td>Min</td><td>Max</td></tr><tr><td>ER-JEPA (Sparse Retrieval)</td><td> ${ \bf 8 6 . 3 7 \pm 0 . 3 5 }$ </td><td>85.75</td><td>86.60</td></tr><tr><td>Dense Retrieval</td><td> $8 6 . 2 1 \pm 0 . 3 7$ </td><td>85.85</td><td>86.70</td></tr></table>

Table 6 Ablation on episodic memory capacity M for Meta-Llama-3.2-1B-Instruct on NL-RX-SYNTH. Values at epoch 4 are mean ± standard deviation over five seeds. The observed training costs are not compute matched.
<table><tr><td>Memory capacity M</td><td>Accuracy (%)↑</td><td>PFLOPs</td><td>Time (min)</td><td>Tokens (M)</td></tr><tr><td> $1 0$ </td><td> $8 4 . 2 2 \pm 0 . 7 5$ </td><td> $1 8 1 . 5 3 6 \pm 0 . 2 1 1$ </td><td> $1 0 . 4 8 \pm 0 . 0 7$ </td><td> $8 . 5 8 1 \pm 0 . 1 6 6$ </td></tr><tr><td> $1 0 ^ { 2 }$ </td><td> $8 4 . 3 2 \pm 0 . 6 5$ </td><td> $2 1 9 . 6 0 5 \pm 0 . 5 3 3$ </td><td> $1 1 . 6 2 \pm 0 . 2 2$ </td><td> $9 . 4 2 7 \pm 0 . 1 7 5$ </td></tr><tr><td> $1 0 ^ { 3 }$ </td><td> $8 4 . 7 6 \pm 0 . 5 2$ </td><td> $2 1 9 . 7 6 6 \pm 0 . 4 3 9$ </td><td> $1 6 . 7 4 \pm 1 . 3 3$ </td><td> $1 0 . 9 6 9 \pm 0 . 0 9 7$ </td></tr><tr><td> $1 0 ^ { 4 }$ </td><td> ${ \bf 8 4 . 9 8 \pm 0 . 3 5 }$ </td><td> $2 1 9 . 6 6 7 \pm 0 . 3 9 3$ </td><td> $2 2 . 2 3 \pm 0 . 1 7$ </td><td> $1 1 . 2 5 2 \pm 0 . 0 1 0$ </td></tr></table>

![](images/81b1ac7328db7746e9d018b07350a0a1bf826baf211ab174b293eb4405cd2f37.jpg)

Figure 14 Accuracy across four training epochs for four episodic memory capacities. Points show means over five seeds. Error bars show one sample standard deviation. These runs are not compute matched.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>k=0</td><td rowspan=1 colspan=1>k=1</td><td rowspan=1 colspan=1>k=2</td><td rowspan=1 colspan=1>k=3</td></tr><tr><td rowspan=1 colspan=1>λ = 0.5</td><td rowspan=1 colspan=1>85.05%</td><td rowspan=1 colspan=1>85.55%</td><td rowspan=1 colspan=1>86.05%</td><td rowspan=1 colspan=1>85.70%</td></tr><tr><td rowspan=1 colspan=1>λ = 1.0</td><td rowspan=1 colspan=1>85.80%</td><td rowspan=1 colspan=1>85.95%</td><td rowspan=1 colspan=1>86.60%</td><td rowspan=1 colspan=1>86.30%</td></tr><tr><td rowspan=1 colspan=1>λ = 2.0</td><td rowspan=1 colspan=1>86.30%</td><td rowspan=1 colspan=1>86.40%</td><td rowspan=1 colspan=1>85.85%</td><td rowspan=1 colspan=1>86.10%</td></tr></table>

Figure 15 Ablation on the JEPA hyperparameters λ and k. Each entry reports accuracy (%), with the best result in bold.