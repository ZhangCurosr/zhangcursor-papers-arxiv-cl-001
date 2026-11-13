# From Large Language Models to Efficient System One Models via Adaptive Computation

Minh-Anh Nguyen<sup>1</sup> <sup>1</sup>Nanyang Technological University, Singapore

## Abstract

Transferring LLMs into System One decision models typically replaces the language modeling head with a decision head while retaining the full backbone, making every decision pay for all layers. Yet it remains unclear how much depth such decisions require. We find that decision information is concentrated in intermediate layers, with early layers build ing essential features and the final six layers removable without loss, while independently trained smaller models largely mimic the full model rather than developing specialized decision capabilities. These findings suggest that efficiency requires not simply smaller backbones, but shared computation across different depths. Building on this, we propose NESTONE, which jointly trains a full model with prefix submodels that share its final layers and decision head through self-distillation. A lightweight gate reads the hidden state at each exit to determine whether additional computation is needed, allowing one model to operate at multiple layers. Across seven decision benchmarks, our approach matches or outperforms independently trained models at every depth, while achieving better accuracy-efficiency trade-offs through adaptive computation.

## 1 Introduction

Dual-process theories distinguish fast, intuitive System One thinking from slow, deliberate System Two thinking (Kahneman, 2011). LLM-based agents increasingly require both. Alongside long chain-ofthought reasoning, they make frequent, lightweight decisions, such as routing requests, selecting tools, flagging unsafe prompts, and recognizing intents, often under tight latency constraints (TypeSafe AI, 2026). System One decision models address this need by fine-tuning an LLM to predict a distribution over candidate options at a single position, without generating tokens (InternLM Team, 2026; Li and Wang, 2026). For example, Intern-Decision-4B, built on Qwen3.5-4B (Qwen Team, 2026), achieves 90.0% macro accuracy across seven decision benchmarks. However, replacing the language modeling head with a decision head leaves the full backbone intact: every decision still traverses all 32 layers, incurring a latency of approximately 1.45 ms per layer at batch size one in our setting.

![](images/f35fd117b83328d0c441bddfbacf07b4bc53ad166f778dff18cd2f375b11a9e3.jpg)

![](images/1ae020e15e016f23697719140e3ff971d7fafe69879a70bceac41d8d4a61c83b.jpg)  
Figure 1: Left: Accuracy versus computation depth for independently trained models, and NESTONE w/o the stop gate. Right: Average depth of the gate versus the share of easy decisions; on the four largest benchmarks, accuracy stays within 0.1 points of the fixed 24-layer exit. All models use Qwen3.5-4B and the same training data; models are averaged over 3 seeds.

This raises a fundamental question: how much computational depth does a System One decision actually require? Fine-tuning is necessary to equip an LLM with decision capabilities that its base model lacks (Section 3), but whether all inherited layers are necessary remains unclear. A natural way to reduce inference cost is to truncate the backbone, trading computational depth for potentially lower accuracy. As shown in the left panel of Figure 1, this trade-off is highly non-uniform: reducing the model from 32 to 24 layers largely preserves performance, whereas further truncation causes substantial degradation, with accuracy dropping from 84.3% to 72.8% and 61.5% at 16 and 12 layers, respectively. This highlights the limitation of fixed-depth inference, which imposes the same computational budget on all inputs. Yet, not every decision requires the same amount of computation. This motivates adaptive depth allocation (Figure 1, right), which allows computation to stop once sufficient depth is reached, reducing inference cost while preserving accuracy.

We find that decision computation is concentrated in intermediate layers, while independently trained smaller models largely replicate the full model rather than develop specialized decision capabilities (Section 3). First, decisions emerge within a narrow band of middle layers: integratedgradient attribution places 50% of the decision by layer 20 and 80% by layer 23. Consistently, ablating layers 24–29 leaves accuracy unchanged, while ablating layers 15–23 eliminates the decision margin (Section 3.1). Second, independently trained smaller models behave more like truncated full models than independent specialists: 89–92% of their correct decisions are already made by the full model, and their distinct wins exceed full-model seed disagreement by only 2–3 points, mostly on low-confidence decisions (Section 3.2). These findings show that training smaller models or truncating the backbone alone is insufficient. Instead, decision models should learn to make reliable decisions at multiple depths, adapting computation to input difficulty.

Based on these findings, we propose NESTONE, a framework that transforms an LLM into a multidepth System One decision model. NESTONE jointly trains the full model and nested prefix sub-models with shared final layers and decision head, combining decision supervision and selfdistillation from the full model. This enables a single training run to produce multiple exits within one checkpoint. At each exit, a lightweight gate reads the hidden state to determine whether further computation is needed before producing the decision. Our contributions are threefold: (i) We show that decision information is concentrated in intermediate layers and that independently trained smaller models largely reproduce, rather than specialize in, the full model’s decisions; (ii) We propose NESTONE, a nested multi-depth training framework that combines decision supervision, self-distillation, and a hidden-state stopping gate for transfering from LLMs to System One Models efficiency; (iii) We demonstrate that NESTONE matches or outperforms independently trained models at every depth while reducing inference cost through adaptive early exiting, using a single training run to obtain multiple models.

## 2 Background and Related Work

System One Decision Models. System One models take a state, a typed question, and a set of options, and return a distribution over the options in a single forward pass without generating text (TypeSafe AI, 2026). Following Jev, a growing number of such models are built by adapting a pretrained LLM. Open recipes read the option-symbol logits of an instruction-tuned LLM at the last position, with no training, or add LoRA adapters and a decision head initialized from the language-model head (Geuter et al., 2026). Intern-Decision (InternLM Team, 2026) fully fine-tunes Qwen3.5 (Qwen Team, 2026) at 0.8B, 2B, and 4B parameters, Microsoft-Decision-1 is post-trained from Qwen3.5-9B (Microsoft, 2026), CLM trains a decision model contrastively from Qwen3-8B (Kwok et al., 2026), and CS1 trains a 4B model on 12 million clinical decisions (Zhang et al., 2026). Other work changes how options are read out. Jeva encodes the request and each option separately so that the decision is permutation-equivariant, can abstain, and can reuse option states across requests (Li and Wang, 2026); Prism-1 scores each option in its own attention block so that neither option order nor added rivals change the result (Teng et al., 2026); and Considerthen-Choose removes order dependence at inference without training (Huang et al., 2026).

Evaluating System One Models. A second line of work asks where System One models can replace LLMs. System1-Arena compares ten System One models with open and closed LLMs through a shared decision interface and finds that the fast models suffice for many single-step decisions but fall behind on long-horizon tasks (Chen et al., 2026). Other studies evaluate System One models as decision points in agent harnesses (Li, 2026), and test their transfer to low-resource languages (Fadli et al., 2026). These studies treat each model as a black box and measure what it outputs.

Layer Pruning. The deeper layers of LLMs are known to be partly redundant. Removing a block of late layers costs little on question answering (Gromov et al., 2024; Men et al., 2024), and merging or pruning layers followed by distillation recovers most of the original accuracy at a smaller size (Yang et al., 2024; Muralidharan et al., 2024; Xia et al., 2024). These methods produce one smaller model with one fixed depth, and they select layers by similarity or importance scores computed for generation rather than by where a decision is computed.

Existing work on System One models focuses on how to train an LLM into a decision model, whether through readout design, training data, or architecture, and on how the resulting models behave. Each of these models keeps the full depth of its backbone, and cost is reduced only by choosing a smaller backbone. To our knowledge, no prior work asks where inside the LLM the decision is computed, or what a smaller decision model learns relative to the full one. We address both questions first (Section 3). The answers then guide an efficient way to turn one LLM into a System One model with several depths: nested prefix training with self-distillation, and a gate that reads the hidden state before the readout is computed (Section 4).

## 3 Understanding Depth in System One Models

We investigate what enables an LLM to become a System One decision model by comparing the model before and after decision fine-tuning, along with smaller variants obtained by progressively removing layers from the fine-tuned model. Specifically, we ask two questions: (i) Where in the network are decisions computed (Section 3.1)? (ii) What do smaller models learn compared to the full model (Section 3.2)?

Setting. We study Qwen3.5-4B (Qwen Team, 2026), whose 32 layers we index 0–31, and Intern-Decision-4B (InternLM Team, 2026), a released decision model built on the same backbone. To establish a controlled reference, we re-train Qwen3.5-4B using the same training recipe as Intern-Decision-4B. We refer to this re-trained model as full32, which serves as the starting point and reference for all our models. Specifically, full32 is fully finetuned with cross-entropy on the option symbols (AdamW, learning rate $2 \times 1 0 ^ { - 6 }$ with cosine decay, 2 epochs, about 131K tokens per step) on 44,941 decisions drawn from public training splits (Zhang et al., 2015; Liu et al., 2024; Lin et al., 2023; Ji et al., 2023; Casanueva et al., 2020; FitzGerald et al., 2023; Clark et al., 2019) and deduplicated against all test states. It reaches $8 4 . 3 \pm 0 . 2 \%$ over three seeds, compared with 90.0% for Intern-Decision-4B, whose larger training set covers the J-Hard and Typed benchmarks better. We consider two layer-cutting strategies: a middle cut, Mk, which retains layers $0 , \ldots , k { - 3 }$ and the final two layers (30–31), and a tail cut, Tk, which retains only the first k layers $( 0 , \ldots , k { - } 1 )$

## 3.1 The Decision Is Written at Mid Depth

When Does the Decision Become Readable? We examine when the decision-related information emerges among layers and when it becomes accessible to the model’s own readout. At each layer, we extract the residual state at the readout position and evaluate it using two complementary methods. The logit lens (Nostalgebraist, 2020) applies the model’s final normalization and output head to the intermediate state, measuring whether the model can already produce the correct answer at that depth. In contrast, a linear probe (Alain and Bengio, 2016), trained to predict the gold option from the same state, measures how much decisionrelevant information is linearly recoverable, regardless of whether the model’s own readout can access it. Thus, the gap between the probe and logit-lens accuracies indicates information that has been encoded in the residual stream but is not yet usable by the model’s output head.

Figure 2(a,b) shows a consistent pattern across the base model and Intern-Decision-4B. At layer 15, the probe substantially outperforms the logit lens (76.1% vs. 26.2% for the base model; 77.9% vs. 38.3% for Intern-Decision-4B), indicating that decision information is already encoded but not yet accessible to the model’s readout. This gap closes over layers 19–23: for Intern-Decision-4B, the probe reaches 90.1% at layer 20, matching its final accuracy (90.0%), while the logit lens rises from 64.4% at layer 19 to 89.4% at layer 23. The two readouts then differ by only 0.1 points (vs. 2.8 points for the base model). Neither improves beyond layer 23, suggesting that layers 24–31 add no linearly recoverable decision information. Finetuning raises the probe plateau by about seven points (89.5% vs. 83.2% at layer 23) without shifting the depth at which decisions become readable.

Which Layers Shape the Decision? Readability shows where the answer is available, not which layers build it there. We therefore decompose the readout state, i.e. the residual stream at the readout position after the last body layer, as $\textbf { h } = \mathbf { h } _ { 0 } +$ $\sum \ell = 0 \sum \ell$ , where $\mathbf { h } _ { 0 }$ is the token embedding and $\Delta _ { \ell }$ is the contribution of layer ℓ, i.e. the sum of the attention and MLP outputs that it adds to the residual stream. Let $s _ { o } ( \mathbf { h } )$ be the logit of option o obtained by passing h through decision head. With gold option y among K options, the decision margin is

![](images/f75a280cd71a720b3cbedab501d07a14323143fcecd6864a7410729aecc93e04.jpg)

![](images/a5182f579b23bc5feaecc203f85a7d5ead7f973762bd4fc8e00bdbb3e6f9ca8f.jpg)

![](images/a1a9138516c96935d749b6905ca7351cefe5bb90ae92f195b9185a55884fa5b2.jpg)  
Figure 2: Where the decision becomes readable and where it is written. (a) Linear-probe accuracy for the gold option at the readout position of each layer, and (b) logit-lens accuracy using each model’s own head, macro-averaged over the seven benchmarks; dashed lines mark each model’s own accuracy. (c) Cumulative share of the decision margin contributed by layers up to ℓ in full32, measured with integrated gradients. The shaded band marks layers 15–23.

$$
m ( { \bf h } ) = s _ { y } ( { \bf h } ) - \frac { 1 } { K - 1 } \sum _ { o \neq y } s _ { o } ( { \bf h } ) ,
$$

so a larger margin puts more probability on the gold option. Integrated gradients (IG) (Sundararajan et al., 2017) attribute the margin to these contributions. Along the straight path $\mathbf { h } ( a ) = \mathbf { h } _ { 0 } +$ $a ( \mathbf { h } - \mathbf { h } _ { 0 } )$ from the embedding to the readout state, layer ℓ receives

$$
\mathrm { I G } _ { \ell } = \Delta _ { \ell } ^ { \top } \int _ { 0 } ^ { 1 } \nabla _ { \mathbf { h } } m \big ( \mathbf { h } ( a ) \big ) d a ,
$$

its contribution multiplied by the average sensitivity of the margin along the path. A positive $\operatorname { I G } _ { \ell }$ means that layer ℓ moves the decision distribution toward the gold option, and a negative one moves it away. The attributions add up to the total margin change, $\begin{array} { r } { \sum _ { \ell } \mathrm { I G } _ { \ell } = m ( \mathbf { h } ) - m ( \mathbf { h } _ { 0 } ) } \end{array}$

Figure 2(c) reveals three distinct regimes in full32. Early layers (0–14) account for only 22% of the final margin, with no individual layer contributing more than 7%. Middle layers (15–23) contribute another 73%, raising the cumulative margin from 22% to 95%. In contrast, late layers exhibit substantial but largely canceling updates: layers 27 and 28 contribute 30% and 35%, respectively, raising the running margin to $1 . 6 \times$ its final value, before layer 29 reverses this increase (−63%). Consequently, layers 24–29 contribute only 5% net, primarily rescaling the margin without changing the preferred option. Overall, the middle layers play the dominant role in shaping the final decision distribution of a fine-tuned System One model.

<table><tr><td>Cut Removed</td><td>(a) Removing layers (Intern-Decision-4B, 90.05) Acc</td><td>∆ [95% CI]</td></tr><tr><td>M26</td><td>24-29</td><td>90.1</td></tr><tr><td>T24 24-31</td><td></td><td>+0.0 [-0.9, +0.9] 89.4 -0.6 [−1.2, −0.1]</td></tr><tr><td>M24</td><td>22-29 88.8</td><td>-1.2 [−2.1, −0.4]</td></tr><tr><td>T22</td><td>22-31</td><td>78.5 -11.6[-14.1, -9.3]</td></tr></table>

<table><tr><td colspan="2">(b) Mean ablation (full32)</td></tr><tr><td>Ablated Margin</td><td>∆Acc</td></tr><tr><td rowspan="2">0 95% 6 31%</td><td>-46.6</td></tr><tr><td>-7.8</td></tr><tr><td>24-29 -6% 34%</td><td>+1.7</td></tr><tr><td>20-23</td><td>-2.3 53%</td></tr><tr><td> $1 9 + 2 3$  102%</td><td>-5.5</td></tr><tr><td> $1 5 { - } 2 3$ </td><td>-57.8</td></tr></table>

Table 1: (a) Accuracy after removing layers without retraining. (b) Share of the mean decision margin removed, and accuracy change, when the contribution of a single early layer (0, 6) or jointly of a group of layers is replaced by its mean (348 decisions, 50 per benchmark).

Which Layers Can Be Removed? Table 1 evaluates layer importance through two complementary interventions. First, we remove layers without retraining (Table 1a), using a middle cut Mk or a tail cut Tk. Second, we perform joint mean ablation (Table 1b): for each layer ℓ, we estimate its mean contribution $\bar { \Delta } _ { \ell }$ across token positions and approximately 1,700 decisions (up to 400 per benchmark), then replace each layer’s contribution with this mean at every position. This preserves the residual stream’s typical scale while removing decisionspecific information. Joint ablation measures the group’s collective contribution to decision-making, accounting for interactions among its layers while preventing them from compensating for one another. We report the resulting change in mean margin and accuracy over 348 decisions.

The two interventions reveal distinct layer roles. In Intern-Decision-4B, removing layers 24–29 without retraining leaves accuracy unchanged, while removing layers 24–31 costs only 0.6 points. Similarly, full32 retains 84.4% accuracy under T24, compared with 84.2% uncut; cutting into the middle band is substantially more damaging, with T22 losing 11.6 points. Mean ablation confirms this pattern: jointly ablating layers 24–29 slightly improves margin and accuracy, whereas ablating layers 15–23 eliminates the entire margin and reduces accuracy by 57.8 points. Early layers, however, remain essential despite their small direct margin contributions: ablating layer 0 costs 46.6 points and layer 6 costs 7.8, reflecting their role in constructing features used by later layers. Consistent with this, a 16-layer model trained only on layers 8–21 and 30–31, the layers most affected by finetuning, achieves just 35.7%, compared with 73.8% for the M16 prefix model.

Takeaway. A decision-tuned LLM writes its decision in a band of middle layers (15–23) on top of feature-building early layers, and its last body layers carry no decision.

## 3.2 Diagnosing Decision Behavior

We next examine what independently trained smaller models learn relative to full32. We finetune M24, M16, and M12 using the same data and recipe as full32, obtaining 83.5%, 73.8%, and 62.9% accuracy, respectively. We compare their predictions with full32 decision by decision, categorizing each as both correct, only full32 correct, only the smaller model correct, or both incorrect. To establish a noise floor, we also compare full32 against two independently trained seeds (43 and 44), which uniquely solve 1.4–2.0% of decisions missed by the reference seed (Table 2, top).

Smaller Models Mostly Reproduce Full32’s Decisions. Table 2 shows that M24 rarely makes a correct prediction that full32 misses: 1.9% of decisions, within the noise floor of 1.4–2.0%, and its subset score (.967) matches that of another full32 seed (.968–.977). M16 and M12 remain largely subsets of full32: 91% and 89% of their correct predictions, respectively, are also correct under full32. Moreover, they lose substantially more decisions than they uniquely solve, by roughly a factor of three to four (1,012 vs. 352 for M16; 1,513 vs. 425 for M12).

<table><tr><td colspan="4">Only correct Acc | pfull</td></tr><tr><td>Model Acc full</td><td>small Sub. Rec.</td><td>≥.8</td><td>&lt;.2</td></tr><tr><td colspan="6">Noise floor: other seeds of full32 seed 43 84.1 1.5 1.4 .977 .977 100.0 0.7 2.0 .968 .972 100.0 0.9</td></tr><tr><td colspan="4">seed 44 84.5 1.8 T24 cut 84.5 1.5</td></tr><tr><td colspan="4">1.7 M24 83.5 2.7 M16</td></tr><tr><td colspan="4">.975 .975 1.9 .967 .959 99.7 M12</td></tr><tr><td colspan="4">73.8 14.4 3.8 .914 .795 84.1 16.7 19.5</td></tr><tr><td colspan="4">62.925.6 4.3.891 .666 70.4</td></tr></table>

Table 2: Decision-level comparison with full32 (seed 42). Noise floor: full32 trained with seeds 43 and 44, compared in the same way. Only correct: share of decisions answered correctly by only full32 or the smaller model. Sub. and Rec.: P(full correct | small correct) and P(small correct | full correct), respectively. Acc | p<sub>full</sub>: smaller-model accuracy when full32 assigns the gold option probability ≥ 0.8 or < 0.2. M24, M16, and M12 are independently trained; T24 is truncated from full32 without retraining. All rates are macro-averaged across benchmarks.

Unique Correct Predictions Are Rare and Uncertain. M16 and M12 uniquely solve only 3.8– 4.3% of decisions, just 2–3 points above the noise floor. These wins are low-confidence (median confidence 0.59–0.61; full32 gold probability 0.23– 0.33) and concentrated in Typed and AGNews (78%), with none in J-Easy or J-Orig. Noise-floor disagreements show the same pattern: median confidence of 0.52–0.55, with 71–73% from Typed and AGNews.

Smaller Models Also Lose Reliable Decisions. When full32 assigns the gold option probability ≥ 0.8, M24, M16, and M12 achieve 99.7%, 84.1%, and 70.4% accuracy, respectively, while other full32 seeds reach 100%. When full32 is confidently wrong (p < 0.2), smaller models recover only 17–20% of these decisions, compared with 0.7–0.9% for other seeds. Their apparent gains therefore reflect greater disagreement with full32, rather than systematic correction of its errors. Overall, smaller models reproduce many of full32’s correct decisions but lose substantial reliable capability without consistently correcting its mistakes.

(a) Nested training with self-distillation  
![](images/1b3155b54ff7fae99f454334d7d5081e10d609222b2315ffa145687b80011b77.jpg)

(b) Adaptive inference with stop gates  
![](images/898fbf4afa266d331e26be4682f0e92834ba5e2a580c2bac525f59bd90bffe74.jpg)  
Figure 3: Overview of NESTONE. (a) Nested training with self-distillation. Prefix exits share the full model’s weights and final block. The full model learns from the gold option y, while each sub-model learns from both y and the full model’s distribution. (b) Adaptive inference with stop gates. The model runs layer by layer, with a lightweight gate evaluating the hidden state at each intermediate exit. If $g _ { e } \geq \tau ,$ , it runs the final block and returns a decision; otherwise, it continues from the saved states without recomputation.

Takeaway. Smaller decision models trained on their own do not learn a different solution; they learn a degraded version of the full model’s. This suggests training them as parts of the full model, sharing its weights and using its own output distribution as a target, so that each smaller model inherits as much of the full model’s skill as its depth allows.

## 4 Methodology

Section 3 shows that intermediate layers already encode sufficient information to identify the correct option, while smaller decision models are best viewed as prefixes of the full model that should learn to imitate it. Building on these findings, NE-STONE trains nested prefix sub-models within the full model using self-distillation to facilitate information sharing across depths (Sections 4.1 and 4.2). A lightweight gate then selects the exit depth for each decision based on question difficulty (Section 4.3).

## 4.1 Prefix Sub-Models with a Shared Coda

Let $f _ { 0 } , \ldots , f _ { 3 1 }$ be the layers of the decision model, and let $\mathbf { h } ^ { ( e ) } = f _ { e } \circ \cdots \circ f _ { 0 } ( x )$ be the residual stream after layer e. We call the last two layers together

with the decision head the coda C = head ◦ $f _ { 3 1 } \circ$ $f _ { 3 0 }$ . An exit after layer e reads out the decision distribution

$$
p _ { e } ( \cdot \mid x ) = \mathrm { s o f t m a x } \big ( C ( \mathbf { h } ^ { ( e ) } ) \mathcal { O } / T _ { e } \big ) ,
$$

where O is the set of option symbols and $T _ { e }$ is a temperature fitted per exit. The sub-model $S _ { k }$ with k layers uses the exit after layer $e _ { k } = k { - } 3 , \mathbf { e g . } ~ S _ { 1 2 } ,$ $S _ { 1 6 }$ , and $S _ { 2 4 }$ exit after layers 9, 13, and 21, and $S _ { 3 2 }$ is the full model. This design follows Section 3.1. Prefixes keep the early feature-building layers that no smaller model can do without, and $S _ { 2 4 }$ reaches into the decision band while dropping the late layers that carry no decision. The coda is shared so that every exit reads out through the same final layers and head. Because each sub-model is a prefix of the next, the states computed for a shallow exit are reused when a decision continues to a deeper exit, so stopping early never requires recomputation.

## 4.2 Nested Training Objective

Following Section 3.2, we train all sub-models jointly with the full model, so that they share its weights and learn from its output. Let y be the gold option and $p _ { k }$ the distribution of $S _ { k }$ . At each step

we compute

$$
\begin{array} { c } { { \mathcal { L } = \displaystyle \mathrm { C E } ( p _ { 3 2 } , y ) + \sum _ { k \in { \mathcal K } _ { t } } \left[ \left( 1 - \alpha _ { t } \right) \mathrm { C E } ( p _ { k } , y ) \right. } } \\ { { \left. + \alpha _ { t } \mathrm { K L } \big ( \mathrm { s g } ( p _ { 3 2 } ) \left\| p _ { k } \right) \right] , } } \end{array}
$$

where sg stops the gradient and the KL term is taken over the options of the decision. The full model is trained only on the gold option, and each sub-model learns both from the gold option and from the full model’s distribution (Hinton et al., 2015). This self-distillation term makes each submodel imitate the full model, as Section 3.2 suggests. We linearly increase $\alpha _ { t }$ from 0 to 0.5 over the first half of training, so that the full model’s distribution becomes a target only once it is informative. The full model serves as its own teacher, so NESTONE needs no external decision signal. All parameters are fine-tuned with the recipe of full32.

## 4.3 Choosing Depth per Decision

A single checkpoint now offers exits at intermediate depths, as well as the full model. To choose an exit for each decision, we attach a lightweight gate to each intermediate exit. The gate reads the hidden state $\mathbf { z } _ { e }$ at the readout position immediately after the corresponding middle layer:

$$
g _ { e } ( x ) = \sigma \big ( \mathbf { w } _ { e } ^ { \top } \mathbf { z } _ { e } + b _ { e } \big ) ,
$$

where $\sigma$ is the sigmoid function. Each gate is a logistic regression model trained to predict whether its corresponding exit answers correctly, using labels $c _ { e } = { \bf 1 }$ [arg max $p _ { e } = y ]$ and class-balanced weights. At inference, the model runs layer by layer. At each intermediate exit, it stops if $g _ { e } ( x ) \geq$ $\tau ,$ runs the coda, and returns arg max $p _ { e } ;$ otherwise, it continues from the saved states. The final exit always answers.

Answering at exit e costs $e + 3$ layers: the prefix plus the coda. Because the gate reads the hidden state before the coda, evaluating it requires only a dot product and a sigmoid operation. A single threshold $\tau$ is shared across exits and selected on held-out data as the most aggressive value for which macro accuracy remains within $\delta$ points of always exiting at the final intermediate depth.

## 5 Experimental Results

We evaluate NESTONE on three questions: (i) Does one nested checkpoint match models trained separately at each depth (Section 5.1)? (ii) Can the gate choose depth per decision without losing accuracy (Section 5.2)? (iii) Why does nested training help (Section 5.3)?

<table><tr><td></td><td></td><td>Layers Separate NESTONE</td><td>∆ [95% CI]</td></tr><tr><td>32</td><td> ${ \bf 8 4 . 3 _ { \pm 0 . 2 } }$ </td><td> $8 4 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td> $- 0 . 3 \left[ - 1 . 3 , + 0 . 8 \right]$ </td></tr><tr><td>24</td><td> $8 3 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> ${ \bf 8 3 . 7 \pm 0 . 3 }$ </td><td> $+ 0 . 2 \ : [ - 1 . 1 , + 1 . 5 ]$ </td></tr><tr><td>16</td><td> $7 2 . 8 { \scriptstyle \pm 0 . 9 }$ </td><td> ${ \bf 7 4 . 7 _ { \pm 0 . 9 } }$ </td><td> $+ 1 . 8 \left[ - 0 . 1 , + 3 . 8 \right]$ </td></tr><tr><td>12</td><td> $6 1 . 5 { \scriptstyle \pm 1 . 5 }$ </td><td> ${ \bf 6 4 . 4 _ { \pm 0 . 4 } }$ </td><td> $+ 2 . 9 \ [ + 0 . 7 , + 5 . 3 ]$ </td></tr></table>

Table 3: Benchmark average (%) of separately trained models and of the exits of one NESTONE checkpoint $( \mathrm { m e a n } _ { \pm \mathrm { s d } }$ over 3 seeds). $\Delta \colon$ NESTONE minus separate, with a CI that resamples decisions and seeds.

Setup. All models start from Qwen3.5-4B and use the data, recipe, and benchmark of Section 3. We compare NESTONE with full32, M24, M16, and M12, each trained separately on its own. Every model, nested or separate, is trained with three seeds, which change the data order and, for NE-STONE, which sub-models are sampled. One NE-STONE run takes 80 minutes on a single H200, against 94 minutes for training the four separate models. At batch size one, the 12-, 16-, 24-, and 32-layer exits take 18.4, 23.8, 34.7, and 46.5 ms per decision. For early exit, we count cost as the mean number of layers run per decision, including the coda. We fit each gate and its threshold on one benchmark-stratified half of the benchmark together with the calibration split, and report results on the other half. Each half is used once for testing.

## 5.1 One Checkpoint versus Separate Models

One Checkpoint Matches or Outperforms Separate Models at Every Depth. Table 3 compares the four exits of NESTONE with independently trained models at matching depths. No exit performs significantly worse, while the shallower exits achieve higher accuracy. The largest gain occurs at 12 layers (64.4% vs. 61.5%, +2.9 points), with a confidence interval excluding zero even when resampling seeds. At 16 layers, the gain is 1.8 points but does not reach significance. The 24-layer exit achieves 83.7%, comparable to M24 (+0.2 points) and full32 (−0.6 points), while using 25% fewer layers and reducing latency from 46.5 to 34.7 ms. At 32 layers, accuracy is only 0.3 points below full32, indicating that joint training imposes little cost on the full model. Thus, a single 80-minute training run replaces four separate runs while yielding the best accuracy at 12, 16, and 24 layers.

<table><tr><td rowspan="2">Rule</td><td rowspan="2">δ ∆Avg ∆L4</td><td colspan="3">Mean layers</td></tr><tr><td>All</td><td>Easy</td><td>Hard</td></tr><tr><td>Fixed exit 13 Fixed exit 21</td><td>-9.89 一 0.00</td><td>-4.49 16.0 0.00 24.0</td><td>16.0 24.0</td><td>16.0 24.0</td></tr><tr><td>Max-prob Confidence gate State + confidence State gate (ours)</td><td>0.25 -0.30 0.25 -0.39 0.25 -0.26 0.25 -1.56</td><td>-0.04 -0.06 +0.11 +0.08</td><td>23.8 23.4 23.2 22.3 23.4 19.7 21.6 20.2</td><td>27.8 27.5 27.9 24.0</td></tr><tr><td>Max-prob Confidence gate State + confidence</td><td>2 -1.35 2 -1.36</td><td>-0.12 -0.13</td><td>21.3 19.4 21.1 19.0</td><td>26.9 26.6</td></tr><tr><td>State gate (ours)</td><td>2-1.37 2-1.68</td><td>-0.09 20.2 +0.03 20.2</td><td>15.4 17.5</td><td>27.3 23.9</td></tr></table>

Table 4: Early exit over exits 9, 13, and 21 of NESTONE (seed 42). ∆Avg: change in benchmark average relative to always exiting after layer 21 (83.8%). ∆L4: accuracy change on the four largest benchmarks (AGNews, Wild-Jailbreak, Typed, J-Hard; 96% of decisions), weighted by decisions. Mean layers include the coda and every readout computed. Easy: AGNews and WildJailbreak; hard: Typed and J-Hard. Max-prob stops when the calibrated top probability clears the threshold; the confidence gate is a logistic regression on top probability, margin, and entropy.

## 5.2 Choosing Depth per Decision

The Hidden State Predicts Whether an Exit Is Correct. We first measure how well each signal separates correct from wrong answers at an exit, using AUROC on the test halves. The state gate reaches 0.86, 0.89, and 0.88 at exits 9, 13, and 21, against 0.85, 0.86, and 0.84 for the calibrated top probability, and 0.86, 0.86, and 0.84 for the confidence gate that combines top probability, margin, and entropy. The hidden state thus carries more information about correctness than the exit’s own confidence, especially at exit 13 and 21, and it is available before the coda is computed.

Stopping Rules. We compare the state gate with three rules based on the exit distribution $p _ { e } \colon$ Maxprob, which uses the calibrated top probability; confidence gate, a logistic regression on the top probability, top-two margin, and entropy; and State + confidence, which combines these features with the hidden state $\mathbf { z } _ { e }$ . All rules use the same training labels, class-balanced weights, and fitting data. They evaluate exits 9, 13, and 21 in order, stopping at the first score above τ or answering at exit 21 otherwise; τ is selected as in Section 4.3. The key difference is inference cost: distribution-based rules must run the coda to obtain $p _ { e }$ , adding two layers at each checked exit, whereas the state gate reads $\mathbf { z } _ { e }$ before the coda and runs it only once, at the selected exit. Fixed exits 13 and 21 serve as references.

The State Gate Gives the Best Trade-off. Table 4 compares the gating rules at two accuracy thresholds. The state gate achieves the lowest average depth at both $\delta \ : = \ : 0 . 2 5$ and $\delta = 2 ,$ using 21.6 and 20.2 layers, respectively, compared with 24 layers for the fixed exit. On hard benchmarks, it uses 23.9–24.0 layers, whereas confidence-based rules incur two additional layers per readout, resulting in an average depth of 26.6–27.9 layers. On the four largest benchmarks, which account for 96% of decisions, the state gate preserves the fixed-exit accuracy, with changes of only +0.08 and +0.03 points at the two thresholds. The other rules also remain within 0.13 points of the fixed exit, but achieve savings differently: confidence-based rules save computation on easy decisions while spending more on hard ones, whereas the state gate reduces depth on easy decisions (to 20.2 and 17.5 layers) without incurring additional computation on hard decisions. Randomly mixing exits 13 and 21 to match the same average depth instead reduces accuracy by 3.0 and 4.7 points.

The Accuracy Drop Is Concentrated on Small Benchmarks. The state gate’s macro-accuracy drop (−1.56 and −1.68 points) comes almost entirely from the three smallest benchmarks, which contain only 430 of 12,351 decisions but account for three of seven macro-averaged benchmarks. At $\delta = 0 . 2 5$ , it misses seven additional decisions on these benchmarks but loses none on larger ones. Compute savings depend on traffic mix: average depth ranges from 24 layers on hard-only traffic to 20.2 on easy-only traffic (22.1 for an even mix), and to 17.5 layers at δ = 2 (20.7 for an even mix; Figure 1, right). Since the selected threshold varies across training splits (0.80 vs. 0.92), it should be calibrated on held-out deployment-like traffic. Adding the full model as a final exit further improves the trade-off: at $\delta = 2 ,$ , the gate matches its accuracy on the four largest benchmarks (−0.01 points) while using 23.9 layers on average, 19.8 on easy decisions, and 31.3 on hard ones.

## 5.3 Why Nested Training Helps

Shallow Exits Keep More of the Full Model’s Skill. Section 3.2 found that separately trained small models are degraded copies of the full model. Table 5 shows that nesting reduces this degradation rather than adding new skills. At 12 layers, the exit wins no more decisions of its own than M12 does (3.6% vs. 3.8%). Instead, it loses fewer of full32’s correct decisions (23.4% vs. 26.5%), so its recall of full32’s skill rises from 0.66 to 0.69. On decisions full32 is confident about, it is right 72.8% of the time, against 69.5% for M12. The 16- layer exit shows the same pattern: recall rises from 0.79 to 0.81, and accuracy on confident decisions from 83.3% to 86.0%, with no more wins of its own (3.5% vs. 3.6%). This is the effect the selfdistillation term targets: the shallow exit imitates the full model and keeps more of what the full model knows.

<table><tr><td></td><td colspan="3">Only correct</td><td colspan="3">Acc | pfull</td></tr><tr><td>Model</td><td>Acc</td><td>full</td><td>small</td><td>Rec.</td><td>≥.8</td><td>&lt;.2</td></tr><tr><td>full32, seeds 43/44</td><td>84.3</td><td>1.6</td><td>1.7</td><td>.975</td><td>100.0</td><td>0.8</td></tr><tr><td>M12 NESTONE 12</td><td>61.5 64.4</td><td>26.5 23.4</td><td>3.8 3.6</td><td>.657 .688</td><td>69.5 72.8</td><td>15.8 14.8</td></tr><tr><td>M16</td><td>72.8</td><td>15.1</td><td>3.6</td><td>.785</td><td>83.3</td><td>17.3</td></tr><tr><td>NESTONE 16</td><td>74.7</td><td>13.2</td><td>3.5</td><td>.810</td><td>86.0</td><td>17.9</td></tr><tr><td>M24</td><td>83.5</td><td>2.9</td><td>2.2</td><td>.956</td><td>99.4</td><td>3.8</td></tr><tr><td>NESTONE 24</td><td>83.7</td><td>3.4</td><td>2.8</td><td>.950</td><td>98.8</td><td>3.4</td></tr><tr><td></td><td>84.0</td><td>2.7</td><td>2.4</td><td></td><td></td><td></td></tr><tr><td>NESTONE 32</td><td></td><td></td><td></td><td>.961</td><td>99.3</td><td>2.8</td></tr></table>

Table 5: Decision-by-decision comparison with full32 (seed 42), as in Table 2, averaged over 3 seeds of each model. NESTONE k is the k-layer exit.

Deep Exits Stay Close to the Full Model. At 24 and 32 layers, the nested exits keep as much of full32’s skill as M24 (recall 0.95–0.96 vs. 0.96), and their own wins (2.4–2.8%) are close to those of M24 (2.2%). Nesting neither adds nor removes skill at these depths. This matches Table 3, where the deep exits tie their separate counterparts. The benefit of nesting at depth is therefore cost, not accuracy: one checkpoint provides the 24-layer exit and the full model together, and the gate can choose between depths for each decision.

Where Does NESTONE Write the Decision? Figure 4 shows that nested training shifts decision formation toward earlier layers, reversing the delay caused by plain fine-tuning. In the 32-layer exit, layers 0–16 account for half of the final margin (bootstrap range 15–16), and layers 0–20 account for 80% (19–20), compared with layers 20 and 23 in full32. The first 20 layers contribute 77% of the margin, versus 47% in full32, so the full model largely commits to its decision within the prefix shared by the 24-layer exit. Later layers mainly rescale this decision: their cumulative contribution peaks at 2.5× the final margin before layer 29 removes the excess. Jointly ablating layers 24–29 increases the margin by 4% and reduces accuracy by only 0.6 points. The 24-layer exit follows a similar pattern: layers 15–21 form its decision, with layer 20 alone contributing 48%. Thus, nested training concentrates decision formation in layers shared across deeper exits.

![](images/93ab028062ef6f329c51f8115267e3aef9efd45345de6bddce7de0f1ddcf5c79.jpg)  
Figure 4: Cumulative share of the decision margin contributed by layers up to ℓ, measured with integrated gradients on 348 decisions. The 24-layer exit of NE-STONE reads out after layer 21. The shaded band marks layers 15–23.

## 6 Conclusion

We studied adaptive computation for System One decision-making in LLMs. Our analysis reveals that decisions primarily emerge in middle layers, with later layers contributing little additional information, and that smaller independently trained models largely replicate the full model’s decisions. Motivated by these findings, NESTONE jointly trains nested prefix sub-models via self-distillation and employs a hidden-state gate for adaptive early exiting. Across three seeds, a single checkpoint matches or outperforms independently trained models at all exit depths, improving 12-layer accuracy by 2.9 points and approaching the 32-layer model within 0.6 points at 24 layers. Adaptive exiting further reduces computation by 16–27% on easy decisions without sacrificing accuracy, while retaining fixed-depth exits for harder cases. These results highlight the benefits of nested training for aligning shallow exits with full-model decisions. Future work will investigate end-to-end latency, gate calibration under real-world traffic, and scaling to larger models and finer-grained exits.

## References

Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020. Efficient´ intent detection with dual sentence encoders. In Proceedings ofthe 2nd Workshop on Natural Language Processing for Conversational AI.

Tianyu Chen, Xinyang Liu, Ruyi An, Zhendong Wang, and Mingyuan Zhou. 2026. System1-Arena: Which decisions can be left to system 1? arXiv preprint arXiv:2610.21244.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics.

Romi Fadli, Rusdah, Wahyu Ramadani Putri Utami, and Zulfa Malika Amana. 2026. Does system one transfer? calibrated typed-decision models, label budgets, and translation for Indonesian regional languages. arXiv preprint arXiv:2610.14112.

Jack FitzGerald et al. 2023. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.

Jonathan Geuter, Andrew Cheng, and David Alvarez-Melis. 2026. Releasing the inner Jev: A simple recipe for open decision models. arXiv preprint arXiv:2610.17205.

Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts. 2024. The un reasonable ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Jianguo Huang, Yao Long Teng, Yanchen Deng, and Bo An. 2026. Consider, then choose: Order-invariant decisions in system one models. arXiv preprint arXiv:2610.18233.

InternLM Team. 2026. Intern-Decision: Multimodal structured decision models. https: //github.com/internlm/Intern-Decision. Model weights: https://huggingface.co/ collections/internlm/intern-decision.

Jiaming Ji et al. 2023. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track.

Daniel Kahneman. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux, New York.

Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, and Azalia Mirhoseini. 2026. Contrastive language models: A system one model for fast and generalizable decisionmaking. Blog post, https://contrastive-lm. notion.site.

Jiawei Li. 2026. Fast models, slow evidence: A paired and self-audited evaluation of system-1 decision models for LLM agent harnesses. arXiv preprint arXiv:2610.02267.

Qi Li and Xinchao Wang. 2026. Jeva: System one models from first principles. arXiv preprint arXiv:2610.18773.

Zi Lin et al. 2023. ToxicChat: Unveiling hidden challenges of toxicity detection in real-world user-AI conversation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023.

Weiwen Liu et al. 2024. ToolACE: Winning the points of LLM function calling. arXiv preprint arXiv:2409.00920.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. ShortGPT: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853.

Microsoft. 2026. Introducing Microsoft-Decision-1, our model for fast decision-making. https://commandline.microsoft.com/ microsoft-decision-1-model-foundry/.

Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. 2024. Compact language models via pruning and knowledge distillation. In Advances in Neural Information Processing Systems.

Nostalgebraist. 2020. Interpreting GPT: The logit lens. LessWrong blog post.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning.

Yao Long Teng, Jianguo Huang, Yanchen Deng, Jiayi Cai, and Bo An. 2026. Prism-1: Position- and rivalindependent scoring for system one models. arXiv preprint arXiv:2610.22066.

TypeSafe AI. 2026. Introducing system one models and Jev. https://typesafe.ai/blog/ introducing-system-one-models-and-jev.

Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024. Sheared LLaMA: Accelerating language model pre-training via structured pruning. In International Conference on Learning Representations.

Yifei Yang, Zouying Cao, and Hai Zhao. 2024. LaCo: Large language model pruning via layer collapse. In Findings of the Association for Computational Linguistics: EMNLP 2024.

Andrew Zhang, Trinh T. L. Vuong, Sophia J. Wagner, Dongmin Bang, Tong Ding, Yuchang Su, Mary Cipperman, Christina S. Chen, Judith Lefkes, Ramin Kahidi, Ming Y. Lu, Caiwei Tian, Tianhao Luo, Jingnan Ma, Rowland Pettit, Long Phi Le, and Faisal Mahmood. 2026. A general-purpose system one decision model for clinical records. arXiv preprint arXiv:2610.27099.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems.